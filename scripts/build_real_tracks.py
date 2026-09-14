#!/usr/bin/env python3
"""Real-footage tracks from indian_road, for stability diagnostics only
(Phase 4S). NOT part of the accuracy headline -- indian_road carries no
plate-region annotation, only vehicle-level BDD100K boxes, so the true plate
string here is genuinely unknown. Every row: plate_text: null,
eligible: false, label_source: "unverified_real" (excluded from scorer.py's
accuracy path by construction -- only "human"/"synthetic_truth" are scored).

Source discipline: samples ONLY from the 31 clip_ids in
datasets/trinetra-hard/CLIP_RESERVATION.md's RESERVED table (never
TRAIN_SAFE) -- this is eval-only data by the same reservation Manas is
building on.

Real clip_id, real per-frame track_id (BDD100K/ByteTrack annotation), real
frame index -> source_pts_ms (indian_road keyframes are extracted at 1fps per
its own README, so source_pts_ms = frame_index * 1000). camera_id and
stream_session_id are NOT synthesised as arbitrary IDs: stream_session_id is
the real clip_id itself, camera_id is a constant label for the one physical
device class this whole dataset comes from (a CP Plus dashcam fleet) --
indian_road has no per-camera metadata to differentiate further.

indian_road has NO plate-region bboxes. plate_bbox here is a heuristic
sub-region of the real vehicle box2d (bottom-center ~15% of vehicle width) --
flagged plate_bbox_source: "estimated_from_vehicle_bbox", not a measurement.
"""
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDIAN_ROAD = ROOT / "datasets" / "raw" / "indian_road"
RESERVATION = ROOT / "datasets" / "trinetra-hard" / "CLIP_RESERVATION.md"
INDEX_PATH = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"
SHARDS = sorted(INDIAN_ROAD.glob("data/train-*-of-00646.tar"))
FRAME_RE = re.compile(r"^(.+)_(\d{4})\.json$")
FPS = 1  # indian_road README: "Keyframe extraction -- 1 frame/second via FFmpeg"
MIN_TRACK_FRAMES = 2  # temporal continuity is the whole point of this phase


def reserved_clip_ids(text: str) -> set[str]:
    ids = set()
    in_reserved = False
    for line in text.splitlines():
        if line.startswith("## RESERVED"):
            in_reserved = True
            continue
        if line.startswith("## ") and in_reserved:
            break
        if in_reserved and line.startswith("|"):
            cell = line.split("|")[1].strip()
            if cell and cell != "clip_id" and not set(cell) <= {"-"}:
                ids.add(cell)
    return ids


def load_clip_frames(clip_ids: set[str]) -> dict[str, list[tuple[int, dict, str]]]:
    """clip_id -> [(frame_idx, annotation_json, shard_name), ...] sorted by frame_idx."""
    by_clip: dict[str, list[tuple[int, dict, str]]] = {cid: [] for cid in clip_ids}
    for shard in SHARDS:
        t = tarfile.open(shard)
        for name in t.getnames():
            if not name.endswith(".json"):
                continue
            m = FRAME_RE.match(name)
            if not m or m.group(1) not in clip_ids:
                continue
            data = json.loads(t.extractfile(name).read())
            by_clip[m.group(1)].append((int(m.group(2)), data, shard.name))
    for cid in by_clip:
        by_clip[cid].sort(key=lambda x: x[0])
    return by_clip


def tracks_in_clip(frames: list[tuple[int, dict, str]]) -> dict[int, list[tuple[int, dict, str]]]:
    """track_id -> [(frame_idx, label_dict, shard_name), ...] with >= MIN_TRACK_FRAMES appearances."""
    by_track: dict[int, list[tuple[int, dict, str]]] = {}
    for frame_idx, data, shard_name in frames:
        for label in data.get("labels", []):
            by_track.setdefault(label["track_id"], []).append((frame_idx, label, shard_name))
    return {tid: appearances for tid, appearances in by_track.items() if len(appearances) >= MIN_TRACK_FRAMES}


def estimate_plate_bbox(vehicle_box2d: dict) -> list[float]:
    """Heuristic sub-region: bottom-center ~15% of vehicle width, near the base."""
    x1, y1, x2, y2 = vehicle_box2d["x1"], vehicle_box2d["y1"], vehicle_box2d["x2"], vehicle_box2d["y2"]
    veh_w, veh_h = x2 - x1, y2 - y1
    plate_w = max(veh_w * 0.15, 1.0)
    plate_h = plate_w * 0.32  # typical plate aspect ratio
    cx = x1 + veh_w / 2
    px = cx - plate_w / 2
    py = y2 - veh_h * 0.12 - plate_h  # just above the bottom edge of the vehicle
    return [round(px, 1), round(py, 1), round(plate_w, 1), round(plate_h, 1)]


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


def build() -> list[dict]:
    clip_ids = reserved_clip_ids(RESERVATION.read_text(encoding="utf-8"))
    by_clip = load_clip_frames(clip_ids)
    rows = []
    for clip_id, frames in sorted(by_clip.items()):
        if not frames:
            continue
        for track_id, appearances in sorted(tracks_in_clip(frames).items()):
            for frame_idx, label, shard_name in appearances:
                box = label["box2d"]
                plate_bbox = estimate_plate_bbox(box)
                plate_w = plate_bbox[2]
                rows.append({
                    "source_dataset": "indian_road",
                    "clip_id": clip_id,
                    "frame_path": f"datasets/raw/indian_road/data/{shard_name}#{clip_id}_{frame_idx:04d}.jpg",
                    "source_pts_ms": frame_idx * round(1000 / FPS),
                    "camera_id": "th_cam_cpplus_dashcam_fleet",
                    "stream_session_id": clip_id,
                    "track_id": track_id,
                    "plate_bbox": plate_bbox,
                    "plate_width_px": plate_w,
                    "width_bucket": width_bucket(plate_w),
                    "eligible": False,
                    "plate_text": None,
                    "label_source": "unverified_real",
                    "label_confidence": "certain",
                    "slice": "tiny" if plate_w < 40 else "easy",
                    "slice_reason": (
                        f"indian_road has no plate-region annotation; plate_bbox is an estimate "
                        f"from the real vehicle box2d (category={label['category']}, "
                        f"track_id={track_id}), not a measurement. Diagnostic-only, never scored "
                        f"for accuracy."
                    ),
                    "degradation_params": None,
                    "plate_bbox_source": "estimated_from_vehicle_bbox",
                    "track_type": "real_footage",
                })
    return rows


def main() -> int:
    if not RESERVATION.exists():
        print(f"BLOCKED: {RESERVATION} not found", file=sys.stderr)
        return 1
    new_rows = build()
    if not new_rows:
        print("BLOCKED: 0 tracks with >=2 frame appearances found in the RESERVED clips", file=sys.stderr)
        return 1

    existing = []
    if INDEX_PATH.exists():
        with INDEX_PATH.open(encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
    existing = [r for r in existing if r.get("label_source") != "unverified_real"]  # replace, don't duplicate

    obs_n = len(existing)
    for r in new_rows:
        obs_n += 1
        r["obs_id"] = f"th_{obs_n:05d}"

    with INDEX_PATH.open("w", encoding="utf-8") as f:
        for r in existing + new_rows:
            f.write(json.dumps(r) + "\n")

    n_tracks = len({(r["clip_id"], r["track_id"]) for r in new_rows})
    n_clips = len({r["clip_id"] for r in new_rows})
    print(f"wrote {len(new_rows)} unverified_real rows across {n_tracks} tracks in {n_clips} RESERVED clips")
    print(f"index.jsonl total rows: {len(existing) + len(new_rows)}")
    return 0


def demo():
    reserved_text = "## RESERVED\n| clip_id | frames |\n|---|---|\n| aaa | 60 |\n## TRAIN_SAFE\n| bbb | 60 |\n"
    assert reserved_clip_ids(reserved_text) == {"aaa"}

    box = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
    bbox = estimate_plate_bbox(box)
    assert len(bbox) == 4 and bbox[2] > 0

    frames = [
        (0, {"labels": [{"track_id": 1, "box2d": box, "category": "car"}]}, "s0.tar"),
        (1, {"labels": [{"track_id": 1, "box2d": box, "category": "car"}]}, "s0.tar"),
        (2, {"labels": [{"track_id": 2, "box2d": box, "category": "car"}]}, "s0.tar"),  # single appearance
    ]
    tracks = tracks_in_clip(frames)
    assert list(tracks.keys()) == [1]  # track 2 dropped, only 1 appearance
    assert len(tracks[1]) == 2
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
