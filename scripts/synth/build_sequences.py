#!/usr/bin/env python3
"""Synthetic multi-frame track corpus -- the TRINETRA-HARD headline (Phase 4R).

Ground truth comes from generation, never OCR: the true plate string is the
synthetic_plates filename (the generator's own output), known exactly, never
guessed. Every row is label_source: "synthetic_truth".

For each of ~300 sampled plates, render one track of 8-15 frames simulating a
vehicle crossing the camera's field of view: plate width sweeps continuously
from <30px to >100px (or the reverse), so every width bucket gets real data
in every track, and one degradation (motion_blur / night / glare /
perspective / none="easy") is baked into every frame of that track. Frames
narrower than 40px are always slice "tiny" regardless of the track's
condition (SPEC_TRINETRA_HARD: tiny overrides). ~10% of frames per track are
deliberately rendered unreadable (extreme blur or <15px) with eligible:
false -- kept, never dropped, to measure fabrication.

Everything is seeded for reproducibility.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
SYNTH_DIR = ROOT / "datasets" / "raw" / "synthetic_plates" / "generated"
OUT = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"

SEED = 20260905
N_TRACKS = 300
FPS = 25
BASE_W, BASE_H = 512, 128  # synthetic_plates native size

# Track-level condition weights, chosen to roughly land near the original
# per-slice targets (60/60/45/45 for the four non-tiny, non-easy slices;
# "tiny" happens automatically per-frame from the width sweep, not assigned
# here). Counts are reported honestly after generation, not forced to match.
CONDITIONS = ["easy", "motion_blur", "night", "glare", "perspective"]
CONDITION_WEIGHTS = [90, 60, 60, 45, 45]  # sums to 300 tracks

UNREADABLE_FRAME_PROB = 0.10


def width_bucket(w: float) -> str:
    if w >= 100:
        return ">100"
    if w >= 80:
        return "80-100"
    if w >= 60:
        return "60-80"
    if w >= 40:
        return "40-60"
    if w >= 30:
        return "30-40"
    return "<30"


def rng_normal(rng: random.Random, shape, sigma: float) -> np.ndarray:
    """Gaussian noise array using the stdlib Random for full seed control."""
    flat = [rng.gauss(0, sigma) for _ in range(int(np.prod(shape)))]
    return np.array(flat, dtype=np.float32).reshape(shape)


def apply_condition(img: np.ndarray, condition: str, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Bake one dominant degradation into a BGR uint8 image. Returns (image, params)."""
    params = {"condition": condition}
    out = img.copy()
    if condition == "motion_blur":
        k = rng.choice([7, 9, 11, 13])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = np.ones(k)
        kernel /= k
        out = cv2.filter2D(out, -1, kernel)
        params["kernel_size"] = k
    elif condition == "night":
        mult = rng.uniform(0.20, 0.45)
        noise_sigma = rng.uniform(5, 12)
        out = np.clip(out.astype(np.float32) * mult, 0, 255)
        out = out + rng_normal(rng, out.shape, noise_sigma)
        out = np.clip(out, 0, 255).astype(np.uint8)
        params["brightness_mult"] = round(mult, 3)
        params["noise_sigma"] = round(noise_sigma, 2)
    elif condition == "glare":
        add = rng.uniform(120, 200)
        out = np.clip(out.astype(np.float32) + add, 0, 255).astype(np.uint8)
        params["brightness_add"] = round(add, 1)
    elif condition == "perspective":
        h, w = out.shape[:2]
        shift = rng.uniform(0.08, 0.20) * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0, 0], [w, 0], [w - shift, h], [shift, h]])
        m = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        params["perspective_shift_px"] = round(shift, 1)
    return out, params


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def width_trajectory(n_frames: int, rng: random.Random) -> list[int]:
    """Monotonic width sweep across every bucket (<30 .. >100 or reverse)."""
    w_start, w_end = rng.randint(15, 25), rng.randint(110, 160)
    if rng.random() < 0.5:
        w_start, w_end = w_end, w_start
    return [round(w_start + (w_end - w_start) * i / (n_frames - 1)) for i in range(n_frames)]


def build_track(track_idx: int, plate_path: Path, rng: random.Random) -> list[dict]:
    plate_text = plate_path.stem
    with Image.open(plate_path) as im:
        base = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

    n_frames = rng.randint(8, 15)
    widths = width_trajectory(n_frames, rng)
    condition = rng.choices(CONDITIONS, weights=CONDITION_WEIGHTS, k=1)[0]
    camera_id = f"th_synth_cam_{track_idx:04d}"
    session_id = f"th_synth_sess_{track_idx:04d}"
    track_id = 1

    rows = []
    for frame_idx, w in enumerate(widths):
        h = max(1, round(w * BASE_H / BASE_W))
        resized = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        degraded, params = apply_condition(resized, condition, rng)

        forced_unreadable = rng.random() < UNREADABLE_FRAME_PROB
        if forced_unreadable:
            extreme_k = rng.choice([15, 19, 23])
            kernel = np.zeros((extreme_k, extreme_k))
            kernel[extreme_k // 2, :] = np.ones(extreme_k)
            kernel /= extreme_k
            degraded = cv2.filter2D(degraded, -1, kernel)
            params["forced_unreadable_blur_kernel"] = extreme_k

        quality = rng.randint(55, 90)
        degraded = jpeg_roundtrip(degraded, quality)
        params["jpeg_quality"] = quality

        eligible = not forced_unreadable
        slice_name = "tiny" if w < 40 else condition
        slice_reason = (
            f"forced unreadable frame (extreme blur), width {w}px"
            if forced_unreadable
            else (f"width {w}px < 40 -> tiny (spec: overrides other conditions)"
                  if w < 40 else f"track condition: {condition}, width {w}px")
        )

        rows.append({
            "source_dataset": "synthetic_plates",
            "clip_id": f"synth_track_{track_idx:04d}",
            "frame_path": f"datasets/raw/synthetic_plates/generated/{plate_path.relative_to(SYNTH_DIR).as_posix()}#frame{frame_idx}",
            "source_pts_ms": round(frame_idx * (1000 / FPS)),
            "camera_id": camera_id,
            "stream_session_id": session_id,
            "track_id": track_id,
            "plate_bbox": [0, 0, w, h],
            "plate_width_px": float(w),
            "width_bucket": width_bucket(w),
            "eligible": eligible,
            "plate_text": plate_text,
            "label_source": "synthetic_truth",
            "label_confidence": "certain",
            "slice": slice_name,
            "slice_reason": slice_reason,
            "degradation_params": params,
            "plate_bbox_source": "rendered",
        })
    return rows


def build(n_tracks: int = N_TRACKS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    all_plates = sorted(SYNTH_DIR.rglob("*.png"))
    sampled = rng.sample(all_plates, n_tracks)
    rows = []
    obs_n = 0
    for track_idx, plate_path in enumerate(sampled):
        track_rng = random.Random(f"{seed}:{track_idx}")
        for row in build_track(track_idx, plate_path, track_rng):
            obs_n += 1
            row["obs_id"] = f"th_{obs_n:05d}"
            rows.append(row)
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    slice_counts = collections.Counter(r["slice"] for r in rows)
    n_ineligible = sum(1 for r in rows if not r["eligible"])
    print(f"wrote {len(rows)} rows across {N_TRACKS} tracks -> {OUT}")
    for s, n in sorted(slice_counts.items()):
        print(f"  {s:12s} {n}")
    print(f"  ineligible (forced unreadable) {n_ineligible} ({n_ineligible / len(rows):.1%})")
    return 0


def demo():
    assert width_bucket(150) == ">100"
    assert width_bucket(35) == "30-40"
    assert width_bucket(10) == "<30"
    rng = random.Random(1)
    traj = width_trajectory(10, rng)
    assert len(traj) == 10
    assert traj[0] != traj[-1]
    img = np.full((20, 60, 3), 128, dtype=np.uint8)
    blurred, params = apply_condition(img, "motion_blur", random.Random(2))
    assert blurred.shape == img.shape and "kernel_size" in params
    dark, params2 = apply_condition(img, "night", random.Random(3))
    assert dark.mean() < img.mean()
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
