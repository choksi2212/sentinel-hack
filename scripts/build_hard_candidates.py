#!/usr/bin/env python3
"""Sample frames from license-verified sources into TRINETRA-HARD's six
slices, compute plate_width_px/width_bucket, write candidate rows.

Per TASKS.md Phase 4: NOTHING here is ground truth. Every row is
label_source: "ocr_candidate" and awaits human verification (Phase 5).

Sources used (must have a verified row in datasets/LICENSES.md):
  - justjuu_plates: real photos, real plate bboxes -> width computed from bbox,
    slice assigned by a brightness/blur heuristic on the plate crop (a proxy,
    not a human judgment -- flagged as such in slice_reason).
  - synthetic_plates: whole-image plate crops (fixed 512x128) -> always
    width_bucket ">100", slice "easy". Plate text is read off the filename
    (the generator's ground truth), used as the OCR-candidate text -- still
    label_source "ocr_candidate" per instruction, never "human".

No OCR engine is available in this environment (no tesseract binary, only
the pytesseract wheel would install -- the actual OCR engine does not).
justjuu_plates rows therefore carry plate_text: null. This is reported, not
guessed -- see docs/manuals/akshat/RUNLOG.md.

perspective slice cannot be computed: none of the verified sources carry
plate rotation/pose data, so it is not sampled here at all (0/45) rather than
guessed from a 2D axis-aligned bbox.
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"

SLICE_TARGETS = {
    "easy": 60, "motion_blur": 60, "night": 60,
    "glare": 45, "perspective": 45, "tiny": 30,
}

WIDTH_BUCKETS = [
    (100, ">100"), (80, "80-100"), (60, "60-80"),
    (40, "40-60"), (30, "30-40"), (0, "<30"),
]


def width_bucket(width_px: float) -> str:
    for floor, label in WIDTH_BUCKETS:
        if width_px >= floor:
            return label
    return "<30"


def classify_crop(crop_bgr: np.ndarray) -> tuple[str, str]:
    """Heuristic slice + reason from a plate crop. Not a human judgment."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < 60:
        return "night", f"heuristic: mean brightness {brightness:.0f} < 60"
    if brightness > 200:
        return "glare", f"heuristic: mean brightness {brightness:.0f} > 200"
    if blur_var < 80:
        return "motion_blur", f"heuristic: Laplacian variance {blur_var:.0f} < 80"
    return "easy", f"heuristic: no condition triggered (brightness {brightness:.0f}, blur_var {blur_var:.0f})"


def rows_from_justjuu(limit_per_split: int | None = None):
    parquet_dir = RAW / "justjuu_plates" / "data"
    for path in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        n = len(df) if limit_per_split is None else min(limit_per_split, len(df))
        for i in range(n):
            row = df.iloc[i]
            objs = row["objects"]
            bboxes = objs["bbox"] if objs is not None else []
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for j, bbox in enumerate(bboxes):
                x, y, w, h = [float(v) for v in bbox]
                if w < 5 or h < 5:
                    continue
                x0, y0 = max(int(x), 0), max(int(y), 0)
                x1, y1 = min(int(x + w), img_np.shape[1]), min(int(y + h), img_np.shape[0])
                crop = img_np[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                if w < 40:
                    slice_name, reason = "tiny", f"plate_width_px {w:.0f} < 40 (spec: tiny regardless of other conditions)"
                else:
                    slice_name, reason = classify_crop(crop)
                yield {
                    "source_dataset": "justjuu_plates",
                    "clip_id": f"justjuu_{path.stem}_{i}",
                    "frame_path": f"datasets/raw/justjuu_plates/data/{path.name}#row{i}",
                    "camera_id": "th_cam_single_frame",
                    "stream_session_id": f"justjuu_{path.stem}_{i}",
                    "track_id": j,
                    "plate_bbox": [x, y, w, h],
                    "plate_width_px": w,
                    "eligible": True,
                    "plate_text": None,
                    "slice": slice_name,
                    "slice_reason": reason,
                }


def rows_from_synthetic(limit: int | None = None):
    base = RAW / "synthetic_plates" / "generated"
    paths = sorted(base.rglob("*.png"))
    if limit is not None:
        paths = paths[:limit]
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            w, _h = im.size
        plate_text = path.stem  # filename IS the plate string (generator's own output)
        yield {
            "source_dataset": "synthetic_plates",
            "clip_id": f"synthetic_{i}",
            "frame_path": f"datasets/raw/synthetic_plates/generated/{path.relative_to(base).as_posix()}",
            "camera_id": "th_cam_single_frame",
            "stream_session_id": f"synthetic_{i}",
            "track_id": 0,
            "plate_bbox": [0, 0, w, _h],
            "plate_width_px": float(w),
            "eligible": True,
            "plate_text": plate_text,
            "slice": "easy",
            "slice_reason": "synthetic whole-image plate crop, fixed 512x128 -> always easy/>100px",
        }


def build(sample_cap_per_source: int = 3000) -> tuple[list[dict], dict[str, int]]:
    slice_counts = {k: 0 for k in SLICE_TARGETS}
    rows = []
    obs_n = 0
    for gen in (rows_from_justjuu(sample_cap_per_source), rows_from_synthetic(sample_cap_per_source)):
        for r in gen:
            target = SLICE_TARGETS[r["slice"]]
            if slice_counts[r["slice"]] >= target:
                continue
            obs_n += 1
            r["obs_id"] = f"th_{obs_n:04d}"
            r["source_pts_ms"] = 0
            r["width_bucket"] = width_bucket(r["plate_width_px"])
            r["label_source"] = "ocr_candidate"
            r["label_confidence"] = "probable"
            rows.append(r)
            slice_counts[r["slice"]] += 1
    return rows, slice_counts


def main() -> int:
    rows, counts = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")
    for slice_name, target in SLICE_TARGETS.items():
        got = counts[slice_name]
        flag = "" if got >= target else "  <-- UNDER TARGET, reported honestly, not padded"
        print(f"  {slice_name:12s} {got:3d}/{target:3d}{flag}")
    return 0


def demo():
    assert width_bucket(150) == ">100"
    assert width_bucket(90) == "80-100"
    assert width_bucket(70) == "60-80"
    assert width_bucket(50) == "40-60"
    assert width_bucket(35) == "30-40"
    assert width_bucket(10) == "<30"
    dark = np.full((20, 60, 3), 10, dtype=np.uint8)
    bright = np.full((20, 60, 3), 250, dtype=np.uint8)
    assert classify_crop(dark)[0] == "night"
    assert classify_crop(bright)[0] == "glare"
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
