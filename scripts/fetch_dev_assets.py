#!/usr/bin/env python3
"""Extract dev assets for Manas from the 31 RESERVED indian_road clips (see
datasets/trinetra-hard/CLIP_RESERVATION.md -- RESERVED only, never
TRAIN_SAFE). Writes MP4s + a JPEG sequence to datasets/dev_assets/.

Why a script instead of committing the videos via Git LFS: this repo
already treats run output as reproducible-from-a-script rather than
committed-as-a-blob (see .gitignore's "runs/"/"benchmarks/out/" section --
"the script that produces it is committed instead"). The same reasoning
applies here: these clips are a deterministic function of 5 tar shards
Manas already needs anyway (same datasets/raw/indian_road junction this
lane uses), so shipping ~5-10MB of video through git history forever is
worse than shipping this file plus the exact SHA-256 of the 5 source tars,
which is what actually proves reproducibility.

IMPORTANT CAVEAT, read before using these for anything latency-sensitive:
indian_road's downloaded frames are 1 fps keyframes (per the dataset's own
README: "Keyframe extraction -- 1 frame/second via FFmpeg"), not the
original continuous footage. The MP4s built here are real video files
(playable, seekable) but are honestly a 1fps slideshow, not smooth 24-30fps
motion -- there is no higher-framerate source available locally to build
anything closer to real motion video. Treat these as decode/ingestion
smoke-test assets (does FrameSequenceSource read the right frame count,
timestamps, resolution), not as a source for testing temporal smoothness.

Usage:
  py -3.11 scripts/fetch_dev_assets.py            # build all 6 assets
  py -3.11 scripts/fetch_dev_assets.py --verify-only   # just check tar hashes
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDIAN_ROAD = ROOT / "datasets" / "raw" / "indian_road"
OUT_DIR = ROOT / "datasets" / "dev_assets"
FRAME_RE = re.compile(r"^(.+)_(\d{4})\.jpg$")

# Recorded 2026-09-08 from the same 5 shards this lane's benchmark corpus is
# built from. If these don't match, the local indian_road checkout differs
# from the one these clip picks were made against -- stop, don't guess.
SOURCE_TAR_SHA256 = {
    "train-00000-of-00646.tar": "f446b888e263138f497d4adb9423d066b35f5d157fbd56234803617c3efb319d",
    "train-00001-of-00646.tar": "8d36814acd43c4842a25be2b204d1f9e908061d65362cc2f9bace51047ba6a0f",
    "train-00002-of-00646.tar": "4e3fdad30576bf4e0e93c9548a0d5376889c4aa3a6c335cbf5c9c4f7727b1585",
    "train-00003-of-00646.tar": "1f462f33d51b1a88f3194b23a32dca64f84be45eecb9f333abf3ad24ec94dbef",
    "train-00004-of-00646.tar": "af9e4a0b0d0b586e65b551af3e98c83f85d6fbccd18568dad1fa11ace5901092",
}

# name -> (clip_id, frame_start, frame_end_inclusive, kind, note)
ASSETS = {
    "clip_daytime_highway": (
        "000ebc57-d7e6-4e62-97a6-2cbc399724a1", 0, 59, "mp4",
        "General, visible plates, daytime highway, 60s (60 frames @ 1fps).",
    ),
    "clip_daytime_city": (
        "0031e84e-43d4-4457-b85b-d90163c7267e", 0, 59, "mp4",
        "General, visible plates, daytime city street, 60s.",
    ),
    "clip_daytime_village": (
        "003bf0b5-6eb4-4359-bffa-3debea8e6e5e", 0, 59, "mp4",
        "General, visible plates, daytime village road, 60s.",
    ),
    "clip_night": (
        "004c8b8d-11af-44a9-9e61-a4e34e643006", 0, 89, "mp4",
        "Night, residential road, 90s window (scene_attributes.json: "
        "timeofday=night, weather=clear).",
    ),
    "clip_hard_scene_cut": (
        "0078bb37-7ca2-4bdd-8563-626117130fb7", 0, 59, "mp4",
        "Hard scene cut at frame 8 (64-bin greyscale histogram L1 distance "
        "0.748 between frames 7 and 8, the largest jump found scanning all "
        "five 180-frame RESERVED clips). Night, city street.",
    ),
    "frame_sequence_100": (
        "0049d23b-5aaf-4bc4-9f9c-a9c2c03dbc9e", 0, 99, "jpeg_sequence",
        "100 consecutive frames, plain JPEGs, for FrameSequenceSource.",
    ),
}


def verify_source_tars() -> bool:
    ok = True
    for name, expected in SOURCE_TAR_SHA256.items():
        path = INDIAN_ROAD / "data" / name
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            ok = False
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            print(f"HASH MISMATCH: {name}\n  expected {expected}\n  actual   {actual}", file=sys.stderr)
            ok = False
    return ok


def load_clip_frames(clip_id: str, frame_start: int, frame_end: int) -> list[tuple[int, bytes]]:
    """[(frame_idx, jpg_bytes), ...] for frame_start..frame_end inclusive, sorted."""
    found: dict[int, bytes] = {}
    for name in sorted(SOURCE_TAR_SHA256):
        shard_path = INDIAN_ROAD / "data" / name
        t = tarfile.open(shard_path)
        for entry in t.getnames():
            m = FRAME_RE.match(entry)
            if not m or m.group(1) != clip_id:
                continue
            idx = int(m.group(2))
            if frame_start <= idx <= frame_end and idx not in found:
                found[idx] = t.extractfile(entry).read()
        if len(found) == frame_end - frame_start + 1:
            break
    return sorted(found.items())


def build_mp4(frames: list[tuple[int, bytes]], out_path: Path) -> None:
    tmp_dir = out_path.with_suffix(".frames_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, (_idx, data) in enumerate(frames):
            (tmp_dir / f"f{i:04d}.jpg").write_bytes(data)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", "1",
                "-i", str(tmp_dir / "f%04d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1",
                str(out_path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_jpeg_sequence(frames: list[tuple[int, bytes]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (_idx, data) in enumerate(frames):
        (out_dir / f"frame_{i:04d}.jpg").write_bytes(data)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)

    if not verify_source_tars():
        print("BLOCKED: source tar verification failed, see above.", file=sys.stderr)
        return 1
    print("OK: all 5 source tars match the recorded SHA-256.")
    if args.verify_only:
        return 0

    if shutil.which("ffmpeg") is None:
        print("BLOCKED: ffmpeg not found on PATH -- required to build the MP4 assets.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (clip_id, start, end, kind, note) in ASSETS.items():
        frames = load_clip_frames(clip_id, start, end)
        if len(frames) != end - start + 1:
            print(f"BLOCKED: {name} ({clip_id}) expected {end - start + 1} frames, found {len(frames)}", file=sys.stderr)
            return 1
        if kind == "mp4":
            out_path = OUT_DIR / f"{name}.mp4"
            build_mp4(frames, out_path)
        else:
            out_path = OUT_DIR / name
            build_jpeg_sequence(frames, out_path)
        print(f"{name}: {len(frames)} frames -> {out_path}  ({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
