# SPEC — Benchmark harness and reports

Owner: Akshat. Everything here exists to produce one credible table.

---

## 1. The headline

**Fusion before/after delta, by width bucket, with raw counts.**

```
Bucket    n     fusion OFF        fusion ON         delta
>100     38     0.92 (35/38)      0.95 (36/38)      +0.03
80-100   44     0.86 (38/44)      0.91 (40/44)      +0.05
60-80    61     0.74 (45/61)      0.85 (52/61)      +0.11
40-60    72     0.51 (37/72)      0.68 (49/72)      +0.17
30-40    49     0.31 (15/49)      0.47 (23/49)      +0.16
<30      36     0.11 (4/36)       0.19 (7/36)       +0.08
ALL     300     0.58 (174/300)    0.69 (207/300)    +0.11
```

Illustrative shape only — those numbers are invented. The point is the format:
every cell carries `rate (correct/total)`, and the row that matters to a judge
is `40-60`, not `ALL`.

Only rows with `label_source: human` and `eligible: true` enter this table.

## 2. Primary metric definition

**E2E correct-plate event rate** = (events where the emitted plate string
exactly equals ground truth, after normalisation) / (eligible observations).

- Matching is on `TrackKey = (camera_id, stream_session_id, track_id)`,
  aligned on `source_pts_ms`. Never `observed_at` — wall-clock drifts between
  the pipeline and the source, and a drifted alignment silently scores the
  wrong frame.
- Normalisation before comparison: uppercase, strip whitespace and hyphens.
  Nothing else. Do **not** apply OCR-confusion fuzzing (`0`↔`O`, `1`↔`I`,
  `8`↔`B`) in the scorer — that is a pipeline feature being measured, not a
  scoring convenience. Fuzzy-match rate may be reported as a separate
  diagnostic column.
- **Fabrication count** is reported separately: rows where `eligible: false`
  but the system emitted a plate string. Never folded into the error rate.

## 3. Report shape — locked

Append-only to `benchmarks/reports/<task>_<run_id>.json`:

```json
{
  "run_id": "e2e_fusion_off_001",
  "task": "vehicle_detection | plate_detection | ocr | temporal_fusion | e2e",
  "dataset_manifest_sha256": "",
  "git_commit": "",
  "weights_sha256": "",
  "machine": "RTX 4060 8GB",
  "runtime": "torch 2.x + CUDA 12.x",
  "source_mode": "file",
  "fusion_enabled": false,
  "n_eligible": 0,
  "n_correct": 0,
  "e2e_correct_plate_event_rate": 0.0,
  "fabrication_count": 0,
  "by_plate_width": {
    ">100":   {"n": 0, "correct": 0, "rate": null},
    "80-100": {"n": 0, "correct": 0, "rate": null},
    "60-80":  {"n": 0, "correct": 0, "rate": null},
    "40-60":  {"n": 0, "correct": 0, "rate": null},
    "30-40":  {"n": 0, "correct": 0, "rate": null},
    "<30":    {"n": 0, "correct": 0, "rate": null}
  },
  "by_slice": {
    "easy": null, "motion_blur": null, "night": null,
    "glare": null, "perspective": null, "tiny": null
  },
  "diagnostics": {
    "precision": null, "recall": null, "map50": null,
    "ocr_exact_accuracy": null, "cer": null, "fuzzy_match_rate": null,
    "fps": null, "latency_p50_ms": null, "latency_p95_ms": null,
    "vram_peak_mb": null, "real_time_factor": null
  },
  "notes": []
}
```

`dataset_manifest_sha256`, `git_commit`, and `weights_sha256` are mandatory.
An unhashed weights file produces an uncitable result — if the weights cannot
be hashed, record why in `notes` rather than leaving the field empty.

Leaderboard rows accumulate in `benchmarks/TRINETRA_MODEL_LEADERBOARD.csv`.

## 4. Harness

One command, unattended, writes a report:

```bash
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion off \
  --out benchmarks/reports/
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion on \
  --out benchmarks/reports/
py -3.11 -m benchmarks.delta \
  --before benchmarks/reports/e2e_fusion_off_001.json \
  --after  benchmarks/reports/e2e_fusion_on_001.json \
  --out    benchmarks/reports/FUSION_DELTA.md
```

The harness must:

- run to completion on a partially-labeled set, scoring only `human` rows
- never crash on a missing prediction — a missing prediction is a miss
- write the report even when the run is degraded, with the degradation in
  `notes`
- be runnable with a stub predictor (§5) so it can be validated before the
  pipeline is ready

## 5. Stub predictor — build this first

`benchmarks/stub_predictor.py` returns canned predictions for a handful of
fixture rows. It exists so the scorer, the report writer, and the delta table
can all be verified before touching the real pipeline.

Validate the scorer against a fixture set with known answers:

| Fixture | Expects |
|---|---|
| exact match | counted correct |
| case/space differs | counted correct after normalisation |
| `0` vs `O` | counted **incorrect**, `fuzzy_match_rate` increments |
| no prediction | counted incorrect, no crash |
| `eligible: false` + prediction emitted | `fabrication_count` increments, excluded from rate |
| `label_source: ocr_candidate` | excluded from every reported number |

If the scorer passes all six, the numbers it produces on real data can be
trusted. If it has not been tested against these, they cannot.

## 6. Diagnostics are not the headline

Report mAP, CER, FPS, latency, and VRAM as diagnostics — they explain *why*
the primary number moved. A model that gains 3 mAP and 0 E2E correct-plate rate
is discarded regardless of how good the mAP looks.

Accuracy-vs-latency: this is a multi-camera system, so throughput per GPU is
the product constraint. mAP 96 @ 12 FPS loses to mAP 94 @ 40 FPS.
