#!/usr/bin/env python3
"""OCR worker. Runs ONLY inside .venv-ocr -- must never import torch, and
must never be imported by the main env (paddleocr's dependency chain pulls in
torch via modelscope, which collides with the main env's own torch install
at the DLL level on Windows; process isolation via a separate venv is the
fix, not import order).

Protocol: reads a JSON list of [{"path": "...", "box": [x, y, w, h] | null}]
on stdin, writes JSON {path: {"text": str|None, "confidence": float|None}}
on stdout. One process, one PaddleOCR() load, run once over however many
paths are given -- not one subprocess per image.

PP-OCRv6 crashes on this machine's CPU oneDNN backend
(NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...); PP-OCRv4
with mkldnn disabled does not. Both settings are forced below, not left to
environment defaults, so this worker behaves the same regardless of who
invokes it.

A single-line plate is usually detected as several separate text regions
(e.g. "AO", "3172") rather than one box -- they are joined left-to-right by
each region's leftmost x-coordinate into one string, with the mean of the
per-region confidence scores reported as the reading's confidence.
"""
import os

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

import json
import sys

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

_ocr = None


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version="PP-OCRv4",
        )
    return _ocr


def read_reading(path: str, box: list | None) -> dict:
    img = Image.open(path).convert("RGB")
    if box is not None:
        x, y, w, h = box
        img = img.crop((x, y, x + w, y + h))
    arr = np.array(img)

    texts, scores, lefts = [], [], []
    for res in get_ocr().predict(arr):
        rec_texts = res.get("rec_texts") or []
        rec_scores = res.get("rec_scores") or []
        rec_boxes = res.get("rec_boxes")
        for i, t in enumerate(rec_texts):
            if not t:
                continue
            left = float(rec_boxes[i][0]) if rec_boxes is not None else float(i)
            texts.append(t)
            scores.append(float(rec_scores[i]) if i < len(rec_scores) else 0.0)
            lefts.append(left)

    if not texts:
        return {"text": None, "confidence": None}
    order = sorted(range(len(texts)), key=lambda i: lefts[i])
    joined = "".join(texts[i] for i in order)
    return {"text": joined, "confidence": round(sum(scores) / len(scores), 4)}


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
