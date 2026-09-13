#!/usr/bin/env python3
"""Export the per-frame OCR-confidence ("fusion weight") distribution from the
fixed-distance, fusion-ON paddle run -- the run that produced 113 fabrications
(see benchmarks/reports/e2e_fusion_on_paddle_fixed_distance_001.json).

For Manas -- input to setting the eligibility gate operating point (his floor
is 0.10, untuned). Writes benchmarks/reports/FUSION_WEIGHT_DISTRIBUTION.json:
a histogram of each row's own frame's OCR confidence, split by whether the
row was eligible and whether the track's fused answer was correct for it, so
he can see where fabricating frames sit relative to recovering ones.

"Fusion weight" here is this predictor's own consensus signal -- OCR
confidence -- not ai/contracts/stages.py's PlateObservation.fusion_weight
(ocr_confidence * image_quality); this predictor never computes an
image_quality term, so the two are not the same field. Said plainly in notes
below rather than silently presented as if they matched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import paddle_predictor
from benchmarks.scorer import normalize

INDEX_PATH = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"
OUT_PATH = ROOT / "benchmarks" / "reports" / "FUSION_WEIGHT_DISTRIBUTION.json"
SOURCE_REPORT = ROOT / "benchmarks" / "reports" / "e2e_fusion_on_paddle_fixed_distance_001.json"

BIN_EDGES = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
BIN_LABELS = [f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}" for i in range(10)]
NULL_BIN = "no_reading"
CATEGORIES = ["eligible_correct", "eligible_incorrect", "ineligible_fabricated", "ineligible_not_fabricated"]


def _bin_for(weight: float | None) -> str:
    if weight is None:
        return NULL_BIN
    idx = min(int(weight * 10), 9)  # 1.0 lands in the last bin, not a 12th bin
    return BIN_LABELS[idx]


def _empty_histogram() -> dict[str, int]:
    return {label: 0 for label in [NULL_BIN, *BIN_LABELS]}


def build(rows: list[dict]) -> dict:
    paddle_predictor.set_track_index(rows)
    cache = paddle_predictor.load_cache()

    histogram = {cat: _empty_histogram() for cat in CATEGORIES}
    n_by_category = {cat: 0 for cat in CATEGORIES}

    for row in rows:
        own_reading = cache.get(row["frame_path"])
        own_weight = own_reading.get("confidence") if own_reading else None
        fused_pred = paddle_predictor.predict(row, fusion_enabled=True)

        if row["eligible"]:
            correct = normalize(fused_pred) == normalize(row["plate_text"])
            category = "eligible_correct" if correct else "eligible_incorrect"
        else:
            fabricated = fused_pred is not None
            category = "ineligible_fabricated" if fabricated else "ineligible_not_fabricated"

        histogram[category][_bin_for(own_weight)] += 1
        n_by_category[category] += 1

    return {
        "source_report": str(SOURCE_REPORT.relative_to(ROOT).as_posix()),
        "weight_field": (
            "this row's own frame's OCR confidence (benchmarks/cache/ocr_readings.json"
            "[frame_path]['confidence']); 'no_reading' means PaddleOCR returned no text "
            "for this exact frame at all, regardless of what the track's fused answer was"
        ),
        "n_rows": len(rows),
        "n_by_category": n_by_category,
        "histogram": histogram,
        "notes": [
            "This is NOT ai/contracts/stages.py's PlateObservation.fusion_weight "
            "(ocr_confidence * image_quality) -- this predictor never computes an "
            "image_quality term, so its consensus rule (and this export) uses raw "
            "OCR confidence only. Do not treat the two fields as interchangeable.",
            "'eligible_incorrect' includes both a wrong reading and a row where this "
            "predictor returned nothing at all for the whole track -- both count as "
            "not correct, only the bin location (often 'no_reading') distinguishes them.",
            "Category counts come from the same fixed-distance fusion-ON run as "
            f"{SOURCE_REPORT.name}: n_eligible/n_correct/fabrication_count there should "
            "match n_by_category's eligible_* / ineligible_fabricated sums here.",
        ],
    }


def main() -> int:
    with INDEX_PATH.open(encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in all_rows if r.get("track_type") == "fixed_distance"]

    report = build(rows)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}: {report['n_by_category']}")
    return 0


def demo():
    """Self-check: bin edges, category split and null handling, no real cache needed."""
    assert _bin_for(None) == NULL_BIN
    assert _bin_for(0.0) == "0.0-0.1"
    assert _bin_for(0.09) == "0.0-0.1"
    assert _bin_for(0.10) == "0.1-0.2"
    assert _bin_for(0.999) == "0.9-1.0"
    assert _bin_for(1.0) == "0.9-1.0"  # exactly 1.0 must not overflow into a 12th bin
    hist = _empty_histogram()
    assert set(hist) == {NULL_BIN, *BIN_LABELS} and all(v == 0 for v in hist.values())
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
