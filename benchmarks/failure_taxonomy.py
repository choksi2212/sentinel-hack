#!/usr/bin/env python3
"""Classify every miss in the fixed-distance fusion-ON run into exactly one
of the 10 buckets ai/quality/taxonomy.py accepts (read-only import -- this
lane never writes ai/). Writes benchmarks/reports/FAILURE_TAXONOMY.json.

Auto-classification priority (first match wins, one bucket per miss):
  1. fusion_wrong     -- correct with fusion OFF, wrong with fusion ON.
                         Checked FIRST: it's a comparative signal (fusion
                         made this specific row worse) that the other rules
                         would otherwise silently absorb into "too small" or
                         "ocr wrong", hiding the one thing this bucket exists
                         to surface.
  2. plate_too_small  -- this row's width_bucket has mean plate height < 20px
                         (per the fixed-distance fusion-OFF report). NOTE:
                         ai/quality/taxonomy.py's own docstring says
                         "Plate < 30 px" for this bucket, not 20px. Followed
                         20px here per explicit instruction; the mismatch
                         against the ai/ lane's documented threshold is real
                         and is not silently resolved either way.
  3. plate_miss       -- fusion-ON prediction is None (plate >= 20px, so not
                         already caught by rule 2).
  4. ocr_partial      -- prediction returned, edit distance 1-2 from truth.
  5. ocr_wrong        -- prediction returned, edit distance >= 3 (catch-all).

vehicle_miss, track_broken, track_merged, duplicate, dropped_frame: this is
a synthetic single-frame-crop pipeline (no vehicle-detection or tracking
stage exists upstream of the plate crop), so these are structurally zero --
never populated, not "measured and found clean."
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for ai.quality.taxonomy
from ai.quality.taxonomy import FailureTaxonomy  # noqa: E402  (read-only import, this lane never writes ai/)

from benchmarks import paddle_predictor  # noqa: E402
from benchmarks.scorer import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"
OFF_REPORT = ROOT / "benchmarks" / "reports" / "e2e_fusion_off_paddle_fixed_distance_001.json"
ON_REPORT = ROOT / "benchmarks" / "reports" / "e2e_fusion_on_paddle_fixed_distance_001.json"
OUT = ROOT / "benchmarks" / "reports" / "FAILURE_TAXONOMY.json"

HEIGHT_FLOOR_PX = 20  # per this task's explicit instruction -- see module docstring
STRUCTURALLY_ZERO = ["vehicle_miss", "track_broken", "track_merged", "duplicate", "dropped_frame"]


def load_fixed_distance_rows() -> list[dict]:
    with INDEX_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("track_type") == "fixed_distance" and r.get("eligible")]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, stdlib only."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def classify(row: dict, pred_off: str | None, pred_on: str | None, bucket_heights: dict[str, float]) -> str:
    truth = normalize(row["plate_text"])
    off_norm = normalize(pred_off)
    on_norm = normalize(pred_on)
    correct_off = off_norm is not None and off_norm == truth
    correct_on = on_norm is not None and on_norm == truth
    assert not correct_on, "classify() must only be called on ON-run misses"

    if correct_off:
        return "fusion_wrong"
    if bucket_heights[row["width_bucket"]] < HEIGHT_FLOOR_PX:
        return "plate_too_small"
    if on_norm is None:
        return "plate_miss"
    dist = edit_distance(on_norm, truth)
    return "ocr_partial" if 1 <= dist <= 2 else "ocr_wrong"


def build() -> dict:
    off_report = json.loads(OFF_REPORT.read_text(encoding="utf-8"))
    bucket_heights = {b: v["mean_height_px"] for b, v in off_report["by_plate_width"].items()}

    rows = load_fixed_distance_rows()
    paddle_predictor.set_track_index(rows)

    def predict(row, fusion_enabled):
        return paddle_predictor.predict(row, fusion_enabled)

    counts = {k: 0 for k in STRUCTURALLY_ZERO}
    counts.update({"plate_miss": 0, "plate_too_small": 0, "ocr_wrong": 0, "ocr_partial": 0, "fusion_wrong": 0})
    examples: dict[str, list[str]] = {k: [] for k in counts}

    n_total = 0
    for row in rows:
        pred_off = predict(row, False)
        pred_on = predict(row, True)
        truth = normalize(row["plate_text"])
        on_norm = normalize(pred_on)
        if on_norm is not None and on_norm == truth:
            continue  # not a miss
        n_total += 1
        bucket = classify(row, pred_off, pred_on, bucket_heights)
        counts[bucket] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append(row["obs_id"])

    taxonomy = FailureTaxonomy()
    taxonomy.record_many(counts)
    verdict = taxonomy.verdict()

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    leader, leader_n = ranked[0]
    runner_n = ranked[1][1] if len(ranked) > 1 else 0
    margin_ok = runner_n == 0 or leader_n >= 1.25 * runner_n

    return {
        "dataset_manifest_sha256": off_report.get("dataset_manifest_sha256"),
        "source_reports": [
            "benchmarks/reports/e2e_fusion_off_paddle_fixed_distance_001.json",
            "benchmarks/reports/e2e_fusion_on_paddle_fixed_distance_001.json",
        ],
        "counts": counts,
        "total_classified": n_total,
        "meets_min_sample_30": n_total >= 30,
        "leader": leader,
        "leader_count": leader_n,
        "runner_up_count": runner_n,
        "leader_at_least_1_25x_runner_up": margin_ok,
        "verdict": verdict.to_dict(),
        "example_obs_ids": {k: v for k, v in examples.items() if v},
        "notes": [
            "vehicle_miss, track_broken, track_merged, duplicate, dropped_frame are "
            "structurally zero on this corpus: there is no vehicle-detection or "
            "tracking stage in the synthetic single-frame-crop pipeline for these "
            "buckets to ever apply to. Set to 0 because they cannot occur here, not "
            "because they were measured and found clean.",
            "plate_too_small threshold used here is <20px mean plate height per "
            "explicit instruction for this task. ai/quality/taxonomy.py's own "
            "docstring documents this bucket as 'Plate < 30 px' -- that is a real "
            "discrepancy against this lane's own taxonomy module, reported here "
            "rather than silently resolved either way.",
            "Classification priority (first match wins): fusion_wrong, then "
            "plate_too_small, then plate_miss, then ocr_partial/ocr_wrong. "
            "fusion_wrong is checked first because it is a comparative signal "
            "(this specific row got worse under fusion) that the other rules "
            "would otherwise silently absorb.",
        ],
    }


def main() -> int:
    result = build()
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"total classified: {result['total_classified']} (>= 30: {result['meets_min_sample_30']})")
    print(f"leader: {result['leader']} ({result['leader_count']}) vs runner-up {result['runner_up_count']} "
          f"-- >=1.25x: {result['leader_at_least_1_25x_runner_up']}")
    print(f"verdict: {result['verdict']['status']}")
    return 0


def demo():
    assert edit_distance("ABC", "ABC") == 0
    assert edit_distance("ABC", "ABD") == 1
    assert edit_distance("ABC", "XYZ") == 3
    assert edit_distance("AB01CD1234", "AB01CD1235") == 1

    heights = {">100": 31.0, "60-80": 15.0}
    base = {"plate_text": "AB01CD1234", "width_bucket": "60-80"}
    assert classify(base, "AB01CD1234", "WRONG", heights) == "fusion_wrong"
    assert classify({**base, "plate_text": "ZZ99ZZ9999"}, None, "GARBAGE12", heights) == "plate_too_small"
    assert classify({**base, "width_bucket": ">100", "plate_text": "ZZ99ZZ9999"}, None, None, heights) == "plate_miss"
    assert classify({**base, "width_bucket": ">100", "plate_text": "AB01CD1234"}, None, "AB01CD1235", heights) == "ocr_partial"
    assert classify({**base, "width_bucket": ">100", "plate_text": "AB01CD1234"}, None, "ZZZZZZZZZZ", heights) == "ocr_wrong"
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
