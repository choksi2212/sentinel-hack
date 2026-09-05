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


def rate_cell(n: int, correct: int) -> str:
    if n == 0:
        return "n/a (0/0)"
    return f"{correct / n:.2f} ({correct}/{n})"


def build_table(before: dict, after: dict) -> str:
    lines = ["| Bucket | n | fusion OFF | fusion ON | delta |", "|---|---|---|---|---|"]
    total_n = total_before_correct = total_after_correct = 0
    for bucket in WIDTH_ORDER:
        b = before["by_plate_width"][bucket]
        a = after["by_plate_width"][bucket]
        n = b["n"]  # before/after share the same dataset -> same n per bucket
        rate_b = b["correct"] / n if n else None
        rate_a = a["correct"] / n if n else None
        delta = f"{(rate_a - rate_b):+.2f}" if (rate_a is not None and rate_b is not None) else "n/a"
        lines.append(
            f"| {bucket} | {n} | {rate_cell(n, b['correct'])} | {rate_cell(n, a['correct'])} | {delta} |"
        )
        total_n += n
        total_before_correct += b["correct"]
        total_after_correct += a["correct"]
    delta_all = (
        f"{(total_after_correct / total_n - total_before_correct / total_n):+.2f}"
        if total_n else "n/a"
    )
    lines.append(
        f"| ALL | {total_n} | {rate_cell(total_n, total_before_correct)} | "
        f"{rate_cell(total_n, total_after_correct)} | {delta_all} |"
    )
    return "\n".join(lines)


def build_report(before_path: Path, after_path: Path) -> str:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    parts = [
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
