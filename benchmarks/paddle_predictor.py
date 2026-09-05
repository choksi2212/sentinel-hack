#!/usr/bin/env python3
"""Real predictor: PaddleOCR, run in an isolated venv (.venv-ocr) via
scripts/ocr_worker.py, because paddle and the main env's torch collide at
the Windows DLL level in the same process (confirmed: importing paddle then
torch raises OSError loading torch/lib/shm.dll -- process isolation, not
import order, is the fix).

Same contract as stub_predictor: predict(row, fusion_enabled) -> str | None.

Ground truth for synthetic_truth rows is the generator's own filename, never
OCR -- so scoring PaddleOCR's reading against it is a real, non-circular
measurement (this is why this predictor only makes sense for synthetic_truth
rows; indian_road's unverified_real rows have no known truth to score
against and are skipped here, same as they are in scorer.py).

Caching: OCR is run ONCE over every distinct frame in the corpus (not once
per (row, fusion_enabled) call -- that would mean thousands of subprocess
round-trips). Results land in benchmarks/cache/ocr_readings.json, keyed by
the row's own frame_path. fusion OFF reads that row's own cached reading;
fusion ON reads every reading across the row's TrackKey and returns the
single highest-confidence non-null reading (NOT per-character majority
vote): OCR readings on different frames of the same track often have
different lengths (a region gets merged, split, or dropped from frame to
frame), so per-character position voting isn't even well-defined most of
the time. Picking the single most confident whole reading is simple, always
well-defined, and mirrors a real fusion pipeline that would surface its best
single detection for a track rather than trying to synthesize a character-
level Frankenstein string across misaligned OCR outputs.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"
WORKER = ROOT / "scripts" / "ocr_worker.py"
CACHE_DIR = ROOT / "benchmarks" / "cache"
CACHE_FRAMES_DIR = CACHE_DIR / "frames"
CACHE_FILE = CACHE_DIR / "ocr_readings.json"

OCR_VERSION = "PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)"
PADDLEX_MODEL_DIRS = ["PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"]

_cache: dict[str, dict] | None = None
_track_index: dict[tuple, list[str]] | None = None


def _sanitize(frame_path: str) -> str:
    return frame_path.replace("/", "_").replace("#", "__").replace(" ", "_").replace(":", "_")


def materialize_synthetic_frames() -> None:
    """Render every synthetic_truth frame to a real PNG under CACHE_FRAMES_DIR,
    reusing scripts/synth/build_sequences.py's own renderer so the pixels
    OCR sees are exactly the pixels ground truth was generated from."""
    sys.path.insert(0, str(ROOT))
    from scripts.synth import build_sequences
    import cv2

    CACHE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def sink(frame_path: str, image_bgr):
        out_path = CACHE_FRAMES_DIR / f"{_sanitize(frame_path)}.png"
        if not out_path.exists():
            cv2.imwrite(str(out_path), image_bgr)

    build_sequences.build(frame_sink=sink)


def run_ocr_batch(frame_paths: list[str]) -> dict[str, dict]:
    """One subprocess, one PaddleOCR() load, every frame in this batch."""
    requests = [
        {"path": str(CACHE_FRAMES_DIR / f"{_sanitize(fp)}.png"), "box": None}
        for fp in frame_paths
    ]
    result = subprocess.run(
        [str(VENV_PYTHON), str(WORKER)],
        input=json.dumps(requests),
        capture_output=True,
        text=True,
        check=True,
    )
    by_materialized_path = json.loads(result.stdout)
    # remap materialized-file-path keys back to the row's own frame_path
    return {
        fp: by_materialized_path.get(str(CACHE_FRAMES_DIR / f"{_sanitize(fp)}.png"), {"text": None, "confidence": None})
        for fp in frame_paths
    }


def build_cache(rows: list[dict]) -> dict[str, dict]:
    synthetic_frame_paths = sorted({
        r["frame_path"] for r in rows if r.get("source_dataset") == "synthetic_plates"
    })
    materialize_synthetic_frames()
    cache = run_ocr_batch(synthetic_frame_paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if not CACHE_FILE.exists():
            raise FileNotFoundError(
                f"{CACHE_FILE} not found -- run benchmarks/paddle_predictor.py directly once "
                "to build it, or call build_cache(rows) from your own script."
            )
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return _cache


def _track_key(row: dict) -> tuple:
    return (row["camera_id"], row["stream_session_id"], row["track_id"])


def set_track_index(rows: list[dict]) -> None:
    """Must be called once with the full row set before fusion-on predictions
    can group frames by TrackKey."""
    global _track_index
    idx: dict[tuple, list[str]] = {}
    for r in rows:
        idx.setdefault(_track_key(r), []).append(r["frame_path"])
    _track_index = idx


def predict(row: dict, fusion_enabled: bool) -> str | None:
    cache = load_cache()
    if not fusion_enabled:
        reading = cache.get(row["frame_path"])
        return reading["text"] if reading else None

    if _track_index is None:
        raise RuntimeError("call set_track_index(rows) before predicting with fusion_enabled=True")
    frame_paths = _track_index.get(_track_key(row), [row["frame_path"]])
    best_text, best_conf = None, -1.0
    for fp in frame_paths:
        reading = cache.get(fp)
        if not reading or reading.get("text") is None:
            continue
        conf = reading.get("confidence") or 0.0
        if conf > best_conf:
            best_text, best_conf = reading["text"], conf
    return best_text


def weights_info() -> tuple[str | None, str]:
    """(sha256_or_None, note) for run.py's weights_sha256/notes fields.
    Hashes PaddleOCR's downloaded det+rec model files if they're on disk;
    otherwise records the version string and why the hash is unavailable."""
    models_dir = Path.home() / ".paddlex" / "official_models"
    hashers = []
    missing = []
    for name in PADDLEX_MODEL_DIRS:
        model_dir = models_dir / name
        if not model_dir.is_dir():
            missing.append(str(model_dir))
            continue
        for f in sorted(model_dir.rglob("*")):
            if f.is_file():
                hashers.append((f.relative_to(models_dir).as_posix(), f.read_bytes()))
    if missing:
        return None, (
            f"weights_sha256 unavailable: model dir(s) not found on disk ({', '.join(missing)}). "
            f"OCR version: {OCR_VERSION}."
        )
    combined = hashlib.sha256()
    for rel_path, data in sorted(hashers):
        combined.update(rel_path.encode("utf-8"))
        combined.update(hashlib.sha256(data).digest())
    return combined.hexdigest(), f"weights_sha256 is a combined hash of {len(hashers)} model files ({OCR_VERSION})."


def main() -> int:
    index_path = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"
    with index_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"building OCR cache for {sum(1 for r in rows if r['source_dataset']=='synthetic_plates')} synthetic frames...")
    cache = build_cache(rows)
    n_read = sum(1 for v in cache.values() if v.get("text"))
    print(f"wrote {CACHE_FILE}: {len(cache)} frames, {n_read} non-null readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
