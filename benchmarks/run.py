#!/usr/bin/env python3
"""One command, unattended, writes a locked report. Per SPEC_BENCHMARK §4.

  py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion off \\
      --out benchmarks/reports/

Never crashes on a missing prediction (a missing prediction is a miss).
Runs to completion on a partially-labeled set, scoring only label_source:
human rows -- if there are none yet, it still writes a report, with that
degradation recorded in notes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.scorer import score
from benchmarks import stub_predictor

ROOT = Path(__file__).resolve().parent.parent
MACHINE = "RTX 4060 8GB"
RUNTIME = "torch 2.x + CUDA 12.x"


def load_rows(dataset: str) -> list[dict]:
    path = ROOT / "datasets" / dataset / "index.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def next_run_id(out_dir: Path, task: str) -> str:
    existing = sorted(out_dir.glob(f"{task}_*.json"))
    n = len(existing) + 1
    return f"{task}_{n:03d}"


def build_report(rows: list[dict], predictions: dict, args) -> dict:
    scored = score(rows, predictions)
    notes = []
    human_rows = [r for r in rows if r.get("label_source") == "human"]
    if not human_rows:
        notes.append(
            "0 human-verified rows in this dataset -- all rows are label_source: "
            "ocr_candidate and are excluded from scoring per SPEC_BENCHMARK §2. "
            "This report reflects an empty ground-truth set, not model performance."
        )
    notes.append("weights_sha256: n/a -- stub predictor has no weights file to hash.")

    return {
        "run_id": args.run_id,
        "task": args.suite,
        "dataset_manifest_sha256": sha256_file(ROOT / "datasets" / args.dataset / "index.jsonl"),
        "git_commit": git_commit(),
        "weights_sha256": None,
        "machine": MACHINE,
        "runtime": RUNTIME,
        "source_mode": "file",
        "fusion_enabled": args.fusion == "on",
        "n_eligible": scored["n_eligible"],
        "n_correct": scored["n_correct"],
        "e2e_correct_plate_event_rate": scored["e2e_correct_plate_event_rate"],
        "fabrication_count": scored["fabrication_count"],
        "by_plate_width": scored["by_plate_width"],
        "by_slice": scored["by_slice"],
        "diagnostics": {
            "precision": None, "recall": None, "map50": None,
            "ocr_exact_accuracy": None, "cer": None,
            "fuzzy_match_rate": scored["fuzzy_match_rate"],
            "fps": None, "latency_p50_ms": None, "latency_p95_ms": None,
            "vram_peak_mb": None, "real_time_factor": None,
        },
        "notes": notes,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="e2e")
    p.add_argument("--dataset", default="trinetra-hard")
    p.add_argument("--fusion", choices=["on", "off"], required=True)
    p.add_argument("--out", default="benchmarks/reports/")
    p.add_argument("--predictor", default="stub", choices=["stub"])
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{args.suite}_fusion_{args.fusion}"
    if args.run_id is None:
        args.run_id = next_run_id(out_dir, task_name)

    rows = load_rows(args.dataset)
    fusion_enabled = args.fusion == "on"
    predictions = {r["obs_id"]: stub_predictor.predict(r, fusion_enabled) for r in rows}

    report = build_report(rows, predictions, args)
    out_path = out_dir / f"{args.run_id}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
