#!/usr/bin/env python3
"""Staged-OCR worker. Runs ONLY inside .venv-ocr -- must never import torch,
and must never be imported by the main env (paddleocr's dependency chain pulls
in torch via modelscope, which collides with the main env's own torch install
at the DLL level on Windows; process isolation via a separate venv is the
fix, not import order).

Unlike scripts/ocr_worker.py, this routes the read through the SHIPPING staged
engine (ai.ocr.PaddleOCR via build_ocr_engine) instead of calling PaddleOCR
directly. The difference is what gets measured:

  - the raw worker asks PaddleOCR to *detect* text regions and join them, which
    re-finds what the caller already located;
  - the staged engine is rec_only (det=False), enforces the width floor, and
    runs the six preprocessing variants one at a time before choosing the
    highest-confidence read. This is the actual pipeline this benchmark swaps
    the stub for, so the number it produces is the number the pipeline ships.

Protocol: reads a JSON list of [{"path": "...", "box": [x, y, w, h] | null}]
on stdin, writes JSON {path: {"text", "confidence", "image_quality",
"variant", "agreement"}} on stdout, with "error" instead on a frame that
raised. One process, one engine load, run once over however many paths are
given -- not one subprocess per image, because loading PP-OCRv4 costs seconds
and the corpus has thousands of frames.

The frame is loaded and the crop is cut here, because PlateCandidate carries
the plate box in FULL FRAME coordinates and the engine's cut_crop/read_crop
pair does the padding and the width floor. Passing the crop ahead of time
would bypass both, which is exactly the stage-boundary those two exist to
enforce.
"""
import os

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

import json
import sys
from pathlib import Path

# This worker is launched by path from .venv-ocr's own interpreter, which puts
# scripts/ on sys.path -- not the repository root. Without this line the ai.*
# imports below raise ModuleNotFoundError, and the parent process only sees a
# non-zero exit and a traceback on stderr, so the failure reads as "paddle is
# broken" rather than "the package root is not importable". There is no
# packaging file in this repository to make ai/ importable any other way
# (docs/REPOSITORY.md section 5.1), which is the same reason tests/conftest.py
# does this.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.contracts.stages import PlateCandidate
from ai.ocr import build_ocr_engine
from ai.quality import plate_quality

# The shipping engine. Same config the live pipeline uses: rec_only PaddleOCR,
# English, CPU if that is all the venv has. use_gpu is optimistically True; the
# engine's own device-flag fallback degrades to CPU cleanly.
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = build_ocr_engine({"name": "paddle", "use_gpu": True})
        _engine.load()
    return _engine


def _candidate(box: list[float] | None) -> PlateCandidate | None:
    """Row plate_bbox is [x, y, w, h]; the pipeline and the width floor use
    xyxy. Convert here, once, so an x/y/w/h slip cannot leak into a width
    measurement that the quality score then silently trusts."""
    if box is None:
        return None
    x, y, w, h = box
    x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    # PlateCandidate needs a detector confidence. A located box in a benchmark
    # row is ground-truth annotation or a rendered box, not a detector output,
    # so there is no measured value to report. 1.0 is the honest answer for
    # "this box is exactly where the plate is" and it is what the quality
    # score's detector_confidence term wants when the box is trusted.
    return PlateCandidate((x1, y1, x2, y2), detector_confidence=1.0)


def read_reading(path: str, box: list | None) -> dict:
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        return {"text": None, "confidence": None, "error": f"cannot read {path}"}
    candidate = _candidate(box)
    if candidate is None:
        return {"text": None, "confidence": None, "error": "invalid box"}
    engine = get_engine()
    # The crop is cut HERE, once, and handed to both the engine and the quality
    # scorer. The live pipeline does exactly this in _offer_crop: cut_crop returns
    # the padded view, read_crop and plate_quality consume the same pixels, so the
    # weight describes the evidence that was actually read. Cutting twice with
    # different assumptions is how padding drift hides a score difference.
    crop = engine.cut_crop(frame, candidate)
    if crop is None or crop.size == 0:
        return {"text": None, "confidence": None, "image_quality": 0.0}
    read = engine.read_crop(np.ascontiguousarray(crop), candidate)
    # image_quality via the same staged scorer, against the same crop, with the
    # plate's SCENE width (not the padded crop's width) -- see pipeline._offer_crop.
    quality = plate_quality(
        crop,
        plate_width_px=candidate.plate_width_px,
        detector_confidence=candidate.detector_confidence,
    )
    if read is None:
        return {"text": None, "confidence": None, "image_quality": quality}
    return {
        "text": read.text,
        "confidence": read.confidence,
        "image_quality": quality,
        # Which preprocessing variant won, and how many of the six agreed with
        # it. Reported because max-of-six confidence is biased upward by
        # construction -- pick the best of six noisy reads and one will look good
        # by luck. agreement is the honest companion to confidence, and it is the
        # only way a report can distinguish "five variants said GJ01AB1234" from
        # "one did, at the same confidence".
        "variant": read.variant,
        "agreement": read.agreement,
    }


def main() -> int:
    requests = json.loads(sys.stdin.read())
    out = {}
    for item in requests:
        path = item["path"]
        try:
            out[path] = read_reading(path, item.get("box"))
        except Exception as exc:  # noqa: BLE001 -- one bad frame must not kill the batch
            out[path] = {"text": None, "confidence": None, "error": str(exc)}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
