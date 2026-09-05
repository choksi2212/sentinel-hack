#!/usr/bin/env bash
# One command, no prompts, writes every report. Per TASKS.md Phase 6R.
# build sequences -> build real tracks -> freeze manifests -> leakage check ->
# license check -> fusion OFF -> fusion ON -> delta -> stability
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/9 build synthetic sequence corpus (Phase 4R) =="
py -3.11 scripts/synth/build_sequences.py

echo "== 2/9 build real-footage tracks (Phase 4S) =="
py -3.11 scripts/build_real_tracks.py

echo "== 3/9 freeze manifests =="
py -3.11 scripts/freeze_manifest.py indian_road justjuu_plates cctv_accident synthetic_plates gujarat_plates indian_plates_yolo

echo "== 4/9 split-leakage check =="
py -3.11 scripts/check_split_leakage.py

echo "== 5/9 license check =="
py -3.11 scripts/check_licenses.py

echo "== 6/9 scorer self-check (six fixtures) =="
py -3.11 benchmarks/scorer.py

echo "== 7/9 fusion OFF =="
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion off --out benchmarks/reports/

echo "== 8/9 fusion ON =="
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion on --out benchmarks/reports/

OFF_REPORT=$(ls -t benchmarks/reports/e2e_fusion_off_*.json | head -1)
ON_REPORT=$(ls -t benchmarks/reports/e2e_fusion_on_*.json | head -1)

echo "== delta: $OFF_REPORT vs $ON_REPORT =="
py -3.11 -m benchmarks.delta --before "$OFF_REPORT" --after "$ON_REPORT" --out benchmarks/reports/FUSION_DELTA.md

echo "== 9/9 stability diagnostic =="
py -3.11 -m benchmarks.stability

echo "== done. reports in benchmarks/reports/ =="
