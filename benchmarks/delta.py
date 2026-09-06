#!/usr/bin/env python3
"""Fusion before/after delta table, by width bucket, with raw counts.
Per SPEC_BENCHMARK §1 -- the headline deliverable.

  py -3.11 -m benchmarks.delta --before benchmarks/reports/e2e_fusion_off_001.json \\
      --after benchmarks/reports/e2e_fusion_on_001.json --out benchmarks/reports/FUSION_DELTA.md
"""
import argparse
import json
import sys
from pathlib import Path

WIDTH_ORDER = [">100", "80-100", "60-80", "40-60", "30-40", "<30"]

PREDICTOR_DISCLAIMERS = {
    "stub": (
        "`benchmarks/stub_predictor.py`, canned/illustrative — not a real model. "
        "Replace before citing this table as a measurement."
    ),
    "paddle": (
        "`benchmarks/paddle_predictor.py`, PaddleOCR (see `weights_sha256`/notes below "
        "for the exact model files) — a real OCR baseline, not the final production "
        "pipeline. Ground truth is synthetic-generated (see below), so this is a real, "
        "non-circular measurement of this specific OCR engine."
    ),
}


def predictor_disclaimer(predictor: str) -> str:
    return PREDICTOR_DISCLAIMERS.get(
        predictor,
        f"predictor `{predictor}` has no known disclaimer text registered in `delta.py` "
        "-- add one to PREDICTOR_DISCLAIMERS before trusting this table.",
    )


def rate_cell(n: int, correct: int) -> str:
    if n == 0:
        return "n/a (0/0)"
    return f"{correct / n:.2f} ({correct}/{n})"


OCR_FLOOR_HEIGHT_PX = 20  # below this, treat a 0.00 rate as an operational floor, not a bug
# empirically measured on this corpus (fixed-distance, PP-OCRv4-mobile): 60-80px
# (15px height) and 40-60px (12px height) are equally at zero as 30-40/<30, so
# the line is drawn above all four dead buckets, not just the two shortest.


def height_cell(mean_height_px: float | None) -> str:
    if mean_height_px is None:
        return "n/a"
    flag = " (below floor)" if mean_height_px < OCR_FLOOR_HEIGHT_PX else ""
    return f"{mean_height_px:.0f}px{flag}"


def build_table(before: dict, after: dict) -> str:
    lines = [
        "| Bucket | n | plate height (px) | fusion OFF | fusion ON | delta |",
        "|---|---|---|---|---|---|",
    ]
    total_n = total_before_correct = total_after_correct = 0
    for bucket in WIDTH_ORDER:
        b = before["by_plate_width"][bucket]
        a = after["by_plate_width"][bucket]
        n = b["n"]  # before/after share the same dataset -> same n per bucket
        rate_b = b["correct"] / n if n else None
        rate_a = a["correct"] / n if n else None
        delta = f"{(rate_a - rate_b):+.2f}" if (rate_a is not None and rate_b is not None) else "n/a"
        lines.append(
            f"| {bucket} | {n} | {height_cell(b.get('mean_height_px'))} | "
            f"{rate_cell(n, b['correct'])} | {rate_cell(n, a['correct'])} | {delta} |"
        )
        total_n += n
        total_before_correct += b["correct"]
        total_after_correct += a["correct"]
    delta_all = (
        f"{(total_after_correct / total_n - total_before_correct / total_n):+.2f}"
        if total_n else "n/a"
    )
    lines.append(
        f"| ALL | {total_n} | -- | {rate_cell(total_n, total_before_correct)} | "
        f"{rate_cell(total_n, total_after_correct)} | {delta_all} |"
    )
    return "\n".join(lines)


def build_report(before_path: Path, after_path: Path) -> str:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    predictor = before.get("predictor", "unknown")
    parts = [
        f"**Predictor: `{predictor}`** ({predictor_disclaimer(predictor)})",
        "",
        "Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- "
        "or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, "
        "reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be "
        "presented as one.",
        "",
        "# FUSION_DELTA",
        "",
        f"Before: `{before_path.name}` (manifest `{before.get('dataset_manifest_sha256')}`, "
        f"commit `{before.get('git_commit')}`)",
        f"After: `{after_path.name}` (manifest `{after.get('dataset_manifest_sha256')}`, "
        f"commit `{after.get('git_commit')}`)",
        "",
        build_table(before, after),
        "",
        f"Fabrication count -- OFF: {before.get('fabrication_count')}, "
        f"ON: {after.get('fabrication_count')} (never folded into the rate above).",
    ]
    def _rate(report: dict, bucket: str) -> float | None:
        c = report["by_plate_width"][bucket]
        return (c["correct"] / c["n"]) if c["n"] else None

    floor_buckets = [
        bucket for bucket in WIDTH_ORDER
        if (h := before["by_plate_width"][bucket].get("mean_height_px")) is not None
        and h < OCR_FLOOR_HEIGHT_PX
        # height alone isn't enough to claim "detects no text" -- verify the
        # rate is actually near zero in BOTH fusion states before saying so
        # (an approach-track corpus can show ~0.20 in a short bucket purely
        # from the track-consensus artifact, which would make this note false)
        and (_rate(before, bucket) or 0) < 0.05
        and (_rate(after, bucket) or 0) < 0.05
    ]
    if floor_buckets:
        parts += [
            "",
            f"**{', '.join(floor_buckets)}** average under ~{OCR_FLOOR_HEIGHT_PX}px of "
            "plate height — below ~20px of plate height this engine detects no text "
            "regardless of fusion. An empirical floor measured on this corpus, not a "
            "scorer defect.",
        ]
    if before.get("notes") or after.get("notes"):
        parts += ["", "**Notes:**"]
        for n in before.get("notes", []):
            parts.append(f"- (before) {n}")
        for n in after.get("notes", []):
            parts.append(f"- (after) {n}")
    return "\n".join(parts) + "\n"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    report = build_report(Path(args.before), Path(args.after))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
