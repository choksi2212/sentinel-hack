"""real_predictor -- the benchmark's window onto the shipping AI pipeline.

MONDAY.md contract: `predict(row, fusion_enabled) -> str | None`, called once
per row by `benchmarks/run.py`'s dict comprehension. It is the swap for
`stub_predictor`, and the two rules that govern its correctness are exactly the
two the stub breaks on purpose:

  1. Never read row["plate_text"]. The prediction must come from pixels, not
     from the label column. That is what turns the report into a measurement.
     Pinned by tests/test_real_predictor.py, which greps this file.
  2. Fuse by TrackKey, never by obs_id order, and align on source_pts_ms, not
     on observed_at (there is no observed_at field; the sequence timestamp is
     the only time a row carries).

Three staged modules do the real work, so the benchmark exercises the same code
the pipeline ships rather than a benchmark-local reimplementation:

  - ai/ocr      cuts the padded crop, enforces MIN_OCR_PLATE_WIDTH_PX, runs the
                six preprocessing variants, returns the best read.
  - ai/quality  scores that same crop; its output is the temporal-fusion weight.
  - ai/fusion   turns one track's observations into one string by weighted
                evidence share, with the per-frame floor MIN_FUSION_WEIGHT.

That routing is the whole point. benchmarks/paddle_predictor.py asks PaddleOCR to
*detect* text in the whole frame and keeps the most confident whole reading per
track; it is a fair baseline for "what does off-the-shelf OCR do", and it is not
this pipeline. The delta between the two reports is what the staged design bought.

Why set_rows exists before predict:
    predict is invoked once per row and the harness has no notion of "the next
    row of this track". A track spans many rows, and its fused answer is only
    known once every frame of it has been read. So the caller hands over the full
    row set once (set_rows), frame materialisation and the OCR batch run once,
    and predict is then a pure lookup. fusion_enabled=True without set_rows is a
    caller error and raises, rather than quietly answering from a single frame
    and reporting it as consensus.

Why OCR runs in a subprocess:
    paddleocr's dependency chain pulls in torch, which collides with the main
    env's own torch at the DLL level on Windows (importing one then the other
    raises OSError loading torch/lib/shm.dll). Process isolation via .venv-ocr is
    the fix, not import order. This module and everything it returns are
    numpy-only; scripts/real_ocr_worker.py is the only thing that imports paddle.

Frame materialisation:
    TRINETRA-HARD rows carry no pixels. A synthetic row's frame_path names a
    source PNG plus a per-frame degradation step ("...KL61AVY6032.png#frame7"),
    and the generator is deterministic from its seeds, so the exact pixels come
    from re-running it once with a frame_sink -- the approach
    paddle_predictor.materialize_synthetic_frames established. Rendering per row
    instead would re-render the whole track once per row, O(track) to O(track^2),
    and two calls would not be guaranteed to agree.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ai.contracts.stages import PlateObservation
from ai.fusion.consensus import MIN_FUSION_WEIGHT, consensus_gain, fuse_observations

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"
WORKER = ROOT / "scripts" / "real_ocr_worker.py"
CACHE_DIR = ROOT / "benchmarks" / "cache"
CACHE_FRAMES_DIR = CACHE_DIR / "frames"
# Deliberately not paddle_predictor's ocr_readings.json. Same frames, different
# engine and a different set of fields; one file for both would mean whichever
# predictor ran last silently decided what the other one measured.
CACHE_FILE = CACHE_DIR / "staged_readings.json"

OCR_VERSION = "PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)"
# Only the recogniser. The staged engine is rec_only -- the plate box comes from
# ai/plate (here, from the row's rendered bbox), so PP-OCRv4's detector is never
# loaded and hashing it would claim provenance over weights that did not run.
REC_MODEL_DIR = "PP-OCRv4_mobile_rec"

# scripts/synth/build_sequences.py renders at 25 fps, which the corpus confirms:
# consecutive frames of a track are 40 ms apart (source_pts_ms 0, 40, 80, ...).
SEQUENCE_FPS = 25

# Whether an `eligible: false` row is answered with None without looking at the
# pixels. Default False, and that is a considered choice rather than an oversight.
#
# MONDAY.md suggests returning None for ineligible rows "unless you're
# deliberately testing the fabrication path". This lane is: scorer.py excludes
# ineligible rows from n_eligible and n_correct entirely, so this flag cannot
# move the headline accuracy number by even one row. The only field it touches is
# fabrication_count -- and gating on row["eligible"] would drive that to 0 by
# construction, for every predictor, forever. A metric that reads 0 because the
# predictor was told the answer measures nothing.
#
# Left False, fabrication_count measures the thing MIN_FUSION_WEIGHT was built to
# reduce: how often the pipeline emits a plate for a frame where the plate is
# genuinely unreadable. In this corpus that surface is real, not theoretical --
# 690 ineligible synthetic rows, 649 of them wide enough that the width floor
# will not refuse them, and 403 of 600 tracks mix eligible with ineligible
# frames, so fusion-on carries its consensus onto ineligible rows and should
# fabricate strictly more than fusion-off. That trade is a finding worth
# reporting. Set True to reproduce the suppressed-by-fiat number for comparison;
# diagnostics() records which mode ran so a report cannot be misread.
RETURN_NONE_ON_INELIGIBLE = False

# Anchor for the observation's observed_at. The live pipeline anchors on a
# replay base plus the frame's pts; an offline row has no wallclock, and reaching
# for datetime.now() would make two runs over the same frozen corpus produce
# different observations. A fixed origin keeps the field timezone-aware (event
# validation requires that) and the run reproducible.
_OBSERVED_AT_ORIGIN = "1970-01-01T00:00:00+00:00"

# The rows this corpus can produce pixels for. indian_road rows point inside a
# .tar shard and have no loader; they are also unverified_real, which scorer.py
# drops before eligibility is even considered, so they contribute to no reported
# number -- not accuracy, not fabrication. Reading them would cost the run an
# hour and change nothing.
MATERIALIZABLE_SOURCE_DATASET = "synthetic_plates"

# frame_path -> {"text", "confidence", "image_quality", "variant", "agreement"}
_readings: dict[str, dict] | None = None
# TrackKey tuple -> rows of that track, ascending source_pts_ms
_track_index: dict[tuple, list[dict]] | None = None
# TrackKey tuple -> the track's fused plate string, or None
_fused: dict[tuple, str | None] = {}
# TrackKey tuple -> consensus_gain() for that track, for the diagnostics block
_gain: dict[tuple, dict] = {}


_WINDOWS_ILLEGAL_CHARS = str.maketrans({
    "/": "_", "#": "__", " ": "_", ":": "_",
    # Windows-reserved characters. The width_bucket names ">100" and "<30" appear
    # in fixed-distance frame_paths, and cv2.imwrite fails on them by returning
    # False with no exception -- which is how those two buckets once lost every
    # frame silently. Same mapping as paddle_predictor._sanitize, deliberately,
    # so both predictors read the one materialised frame directory.
    "<": "lt", ">": "gt", '"': "_", "|": "_", "?": "_", "*": "_",
})


def _sanitize(frame_path: str) -> str:
    return frame_path.translate(_WINDOWS_ILLEGAL_CHARS)


def _frame_file(frame_path: str) -> Path:
    return CACHE_FRAMES_DIR / f"{_sanitize(frame_path)}.png"


def _track_key(row: dict) -> tuple:
    """The TrackKey, all three fields.

    Not (camera_id, track_id) and emphatically not track_id alone: track_id is
    reused across sessions, and in this corpus every synthetic row carries
    track_id 1. Keying on it alone would merge all 600 tracks into one and report
    a single confident plate for the whole dataset.
    """
    return (row["camera_id"], row["stream_session_id"], row["track_id"])


def _is_materializable(row: dict) -> bool:
    return row.get("source_dataset") == MATERIALIZABLE_SOURCE_DATASET


def _frame_index(source_pts_ms: int) -> int:
    """Recover the generator's frame index from the sequence timestamp.

    Re-derived rather than read off the row, because schema.json carries no
    frame_index field -- so the observation's frame identity comes from the one
    timing field the row does have, and stays independent of how it was built.
    """
    return round(source_pts_ms * SEQUENCE_FPS / 1000)


def _observed_at(source_pts_ms: int) -> str:
    origin = datetime.fromisoformat(_OBSERVED_AT_ORIGIN)
    return (origin + timedelta(milliseconds=max(0, int(source_pts_ms)))).isoformat()


def _xyxy(bbox) -> tuple[int, int, int, int]:
    """Row plate_bbox is [x, y, w, h]; every stage contract is xyxy.

    Converted once, here, at the boundary. A predictor that passed xywh into
    PlateCandidate would get a plate_width_px equal to x2 (the corpus's boxes all
    start at x=0, so it would even look plausible) and quietly mis-score every
    resolution term in ai/quality.
    """
    x, y, w, h = bbox
    return (round(x), round(y), round(x + w), round(y + h))


def _cache_meta() -> dict:
    """What the cached readings depend on, so a stale cache is detected.

    Frames alone are not enough: the readings also depend on the engine and on
    the worker that drives it. Without this, editing the variant list or the
    width floor and re-running would silently re-report yesterday's numbers.
    """
    worker_sha = (
        hashlib.sha256(WORKER.read_bytes()).hexdigest() if WORKER.exists() else None
    )
    return {
        "ocr_version": OCR_VERSION,
        "worker_sha256": worker_sha,
        "min_fusion_weight": MIN_FUSION_WEIGHT,
        "fields": ["text", "confidence", "image_quality", "variant", "agreement"],
    }


def _load_cached_readings() -> dict[str, dict] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        blob = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict) or blob.get("meta") != _cache_meta():
        return None
    readings = blob.get("readings")
    return readings if isinstance(readings, dict) else None


def _save_cached_readings(readings: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"meta": _cache_meta(), "readings": readings}, indent=2),
        encoding="utf-8",
    )


def _materialize_frames(frame_paths: list[str]) -> None:
    """Render the deterministic synthetic frames to PNGs under CACHE_FRAMES_DIR.

    build_all renders the whole corpus in one pass -- there is no per-frame entry
    point, and there should not be, since the degradation of frame N depends on
    the generator's seed state. So this is skipped entirely when every frame we
    need is already on disk, which is what makes a re-run cheap.
    """
    if all(_frame_file(fp).exists() for fp in frame_paths):
        return

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2

    from scripts.synth import build_sequences

    CACHE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def sink(frame_path: str, image_bgr) -> None:
        out = _frame_file(frame_path)
        if not out.exists():
            cv2.imwrite(str(out), image_bgr)

    build_sequences.build_all(frame_sink=sink)


def _run_ocr_batch(requests: list[dict]) -> dict[str, dict]:
    """One subprocess, one engine load, every frame in this batch.

    Not one subprocess per frame: loading PP-OCRv4 costs seconds, and 6,822
    frames of that is hours of model loading to do minutes of reading.
    """
    if not requests:
        return {}
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"{VENV_PYTHON} not found. The staged OCR engine runs in an isolated venv "
            "because paddle and this env's torch collide at the DLL level; create it "
            "with 'py -3.11 -m venv .venv-ocr' and install paddleocr into it. "
            "MANUAL STEP -- see ai/README.md."
        )
    result = subprocess.run(
        [str(VENV_PYTHON), str(WORKER)],
        input=json.dumps(requests),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _batch_ocr(frame_paths: list[str], bbox_by_frame: dict[str, list]) -> dict[str, dict]:
    """Read every distinct frame once, keyed back to the row's own frame_path.

    box is the row's plate_bbox, converted to xyxy inside the worker. Passing it
    rather than None (which is what paddle_predictor does) is what puts the
    plate's real scene width in front of the width floor and the quality scorer:
    a `<30` row must be refused for being 27 px wide, not read anyway because the
    materialised file happened to be handed over whole.
    """
    requests = [
        {"path": str(_frame_file(fp)), "box": bbox_by_frame[fp]} for fp in frame_paths
    ]
    by_file = _run_ocr_batch(requests)
    out: dict[str, dict] = {}
    for fp in frame_paths:
        raw = by_file.get(str(_frame_file(fp)), {})
        out[fp] = {
            "text": raw.get("text"),
            "confidence": raw.get("confidence"),
            "image_quality": raw.get("image_quality", 0.0),
            "variant": raw.get("variant"),
            "agreement": raw.get("agreement"),
            "error": raw.get("error"),
        }
    return out


def _observation_for(row: dict) -> PlateObservation | None:
    """One row plus its cached reading -> one PlateObservation, or None.

    None covers three different things, all of them the pipeline's valid "no
    plate here" answer rather than a failure: the engine refused the crop on the
    width floor, no preprocessing variant produced characters, or the row has no
    materialised pixels at all.
    """
    if _readings is None:
        return None
    reading = _readings.get(row["frame_path"])
    if reading is None:
        return None
    text = reading.get("text")
    confidence = reading.get("confidence")
    if not text or confidence is None:
        return None
    bbox = _xyxy(row["plate_bbox"])
    pts_ms = int(row["source_pts_ms"])
    return PlateObservation(
        camera_id=row["camera_id"],
        stream_session_id=row["stream_session_id"],
        track_id=row["track_id"],
        plate_bbox_xyxy=bbox,
        plate_width_px=bbox[2] - bbox[0],
        plate_raw=text,
        ocr_confidence=float(confidence),
        image_quality=float(reading.get("image_quality") or 0.0),
        frame_index=_frame_index(pts_ms),
        pts_ms=pts_ms,
        observed_at=_observed_at(pts_ms),
    )


def set_rows(rows: list[dict], *, use_cache: bool = True) -> None:
    """Preload the full row set: materialise frames, read them, index by TrackKey.

    Call once before any fusion_enabled=True prediction. A caller that passes a
    filtered set (run.py's --track-type) indexes exactly that set, so fusion can
    never reach into a track it was not given -- which is what would let a
    per-bucket report borrow evidence from frames outside its own bucket.
    """
    global _readings, _track_index, _fused, _gain

    frame_paths, bbox_by_frame = [], {}
    for row in rows:
        if not _is_materializable(row):
            continue
        frame_path = row["frame_path"]
        if frame_path in bbox_by_frame:
            # Distinct rows can name the same rendered frame. Read it once: the
            # pixels are identical, and re-reading them is the one genuinely
            # expensive thing this module does.
            continue
        bbox_by_frame[frame_path] = row["plate_bbox"]
        frame_paths.append(frame_path)
    frame_paths.sort()

    cached = _load_cached_readings() if use_cache else None
    if cached is not None and all(fp in cached for fp in frame_paths):
        _readings = cached
    elif frame_paths:
        _materialize_frames(frame_paths)
        fresh = _batch_ocr(frame_paths, bbox_by_frame)
        # Keep any still-valid readings for frames outside this call's row set, so
        # running --track-type approach then fixed_distance does not re-read the
        # first half of the corpus.
        _readings = {**(cached or {}), **fresh}
        _save_cached_readings(_readings)
    else:
        _readings = {}

    _track_index = {}
    for row in rows:
        _track_index.setdefault(_track_key(row), []).append(row)

    _fused, _gain = {}, {}
    for key, track_rows in _track_index.items():
        # Ordered by source_pts_ms, never by obs_id: obs_id order happens to
        # agree today and is not the contract, and fusion that silently depends
        # on row order is fusion that breaks the first time the index is
        # regenerated or filtered.
        ordered = sorted(track_rows, key=lambda r: int(r["source_pts_ms"]))
        observations = [o for o in (_observation_for(r) for r in ordered) if o is not None]
        if not observations:
            _fused[key] = None
            continue
        fused = fuse_observations(observations)
        _fused[key] = fused.normalized if fused is not None else None
        _gain[key] = consensus_gain(observations)


def reset() -> None:
    """Drop in-memory state. For tests and for a second run in one process."""
    global _readings, _track_index, _fused, _gain
    _readings, _track_index, _fused, _gain = None, None, {}, {}


def predict(row: dict, fusion_enabled: bool) -> str | None:
    """The MONDAY.md contract: one row in, one plate string or None out.

    fusion off -- this frame's own staged read, which is what a pipeline with no
    temporal memory would emit.

    fusion on -- the track's fused answer, which every row of the track shares.
    That is deliberate and it is what the real pipeline does: the plate belongs to
    the vehicle, not to the frame, and one event is emitted per track. It does
    mean a track's answer is scored once per frame here, so fabrication_count is
    counted in frames rather than vehicles; diagnostics() reports the track count
    alongside it so the two are not confused.
    """
    if RETURN_NONE_ON_INELIGIBLE and not row.get("eligible", False):
        return None

    if not fusion_enabled:
        if _readings is None:
            return None
        reading = _readings.get(row["frame_path"])
        return (reading.get("text") or None) if reading else None

    if _track_index is None:
        raise RuntimeError(
            "real_predictor: fusion_enabled=True requires set_rows(rows) first. A track's "
            "consensus is only defined once every frame of it has been read; answering from "
            "one frame and labelling it fusion would report a number for a stage that did "
            "not run."
        )
    return _fused.get(_track_key(row))


def diagnostics() -> dict:
    """Summary numbers for run.py's `diagnostics` block, per MONDAY.md.

    Confidence and per-frame detail stay inside this module -- predict() returns
    a bare string by contract -- so this is where the run records what the staged
    stages actually did. `fusion_changed_tracks` is the before/after consensus
    number: how many tracks disagreed with their own best single frame.
    """
    readings = _readings or {}
    read = [r for r in readings.values() if r.get("text")]
    qualities = [float(r.get("image_quality") or 0.0) for r in read]
    confidences = [float(r.get("confidence") or 0.0) for r in read]
    variant_wins: dict[str, int] = {}
    for r in read:
        variant = r.get("variant")
        if variant:
            variant_wins[variant] = variant_wins.get(variant, 0) + 1
    gains = list(_gain.values())
    return {
        "frames_read": len(readings),
        "frames_with_text": len(read),
        "frames_unread": len(readings) - len(read),
        "frame_errors": sum(1 for r in readings.values() if r.get("error")),
        "mean_ocr_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "mean_image_quality": (sum(qualities) / len(qualities)) if qualities else None,
        "variant_wins": dict(sorted(variant_wins.items())),
        "tracks": len(_track_index or {}),
        "tracks_with_plate": sum(1 for v in _fused.values() if v),
        "fusion_changed_tracks": sum(1 for g in gains if g.get("changed")),
        "observations_below_fusion_floor": sum(
            int(g.get("observations", 0)) - int(g.get("eligible_observations", 0))
            for g in gains
        ),
        "min_fusion_weight": MIN_FUSION_WEIGHT,
        "return_none_on_ineligible": RETURN_NONE_ON_INELIGIBLE,
    }


def weights_info() -> tuple[str | None, str]:
    """(sha256_or_None, note) for run.py's weights_sha256/notes fields.

    Hashes the recogniser weights only, because rec is all the staged engine
    loads. A hash covering PP-OCRv4's detector would assert provenance over a
    model that never ran, which is worse than no hash: it would compare equal
    across two runs that differed in the only model either of them used.
    """
    models_dir = Path.home() / ".paddlex" / "official_models"
    model_dir = models_dir / REC_MODEL_DIR
    suffix = (
        f" Recogniser-only hash: the staged engine is rec_only (the plate box comes from "
        f"the row's rendered bbox), so PP-OCRv4's detector is never loaded. "
        f"OCR version: {OCR_VERSION}. Fusion floor: {MIN_FUSION_WEIGHT}. "
        f"eligible:false rows answered from pixels rather than suppressed "
        f"({RETURN_NONE_ON_INELIGIBLE=}), so fabrication_count is a measurement."
    )
    if not model_dir.is_dir():
        return None, f"weights_sha256 unavailable: {model_dir} not on disk.{suffix}"
    files = [f for f in sorted(model_dir.rglob("*")) if f.is_file()]
    if not files:
        return None, f"weights_sha256 unavailable: {model_dir} is empty.{suffix}"
    combined = hashlib.sha256()
    for f in files:
        combined.update(f.relative_to(models_dir).as_posix().encode("utf-8"))
        combined.update(hashlib.sha256(f.read_bytes()).digest())
    return combined.hexdigest(), (
        f"weights_sha256 is a combined hash of {len(files)} recogniser file(s).{suffix}"
    )


def reading_for(frame_path: str) -> dict | None:
    """One cached reading. For the MONDAY.md sanity checklist and for the taxonomy,
    which needs the per-frame confidence and quality that predict() cannot return."""
    return (_readings or {}).get(frame_path)
