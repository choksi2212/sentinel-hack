# Failure taxonomy — baseline benchmark

**Status:** complete for the OCR/plate path. Classified from the data lane's
PaddleOCR runs on TRINETRA-HARD (fixed-distance tracks, fusion off and on),
2,728 misses, manifest `b5df9dfd…`. Scope caveat in *What this does not cover*
below — it is load-bearing and should be read before the verdict is quoted.

This is the deliverable that decides whether any money is spent. Owner's manual
section 6: after the baseline benchmark, **every** miss is classified into
**exactly one** of ten buckets, and the shape of that histogram — not a hunch —
is what unlocks (or does not unlock) the rented A100.

**Verdict: `plate_too_small` dominant. No A100 rental.** Not "not yet" — the
dominant bucket has no software fix.

---

## The rule

1. Classify every miss into exactly one bucket. A miss that could be two things
   is a miss the classification is not finished on.
2. **One dominant bucket** is the precondition for training anything.
3. **Two co-dominant buckets** means the analysis is not finished. Training on a
   split histogram is a guess with a credit card attached — do not.
4. `plate_too_small` is a legitimate finding with **no software fix**. If it
   dominates, the honest deliverable is a width-bucket report and a
   camera-placement recommendation, not a training run.

"Dominant" is defined mechanically so it is not an argument: the leading bucket
must carry at least a quarter more misses than the runner-up (margin 1.25), over
a floor of at least 30 classified misses. Anything short of that reports as
*co-dominant* or *insufficient* rather than being rounded up to a decision. The
logic lives in [`ai/quality/taxonomy.py`](../../ai/quality/taxonomy.py) and is
pinned by [`tests/test_taxonomy.py`](../../tests/test_taxonomy.py).

---

## The ten buckets

| Bucket | Symptom | What would actually help | Whose lane |
|---|---|---|---|
| `vehicle_miss` | No vehicle detected | Vehicle detector training / Indian road data | other |
| `plate_miss` | Vehicle found, plate not | Plate detector training on small plates | other |
| `plate_too_small` | Plate < 20 px | Nothing in software — camera placement, or accept it | **no fix** |
| `ocr_wrong` | Plate found, text wrong | OCR training / synthetic corpus | **this lane's A100** |
| `ocr_partial` | Some characters correct | Temporal consensus / more frames | other |
| `fusion_wrong` | Best single frame right, consensus wrong | Fusion weighting | other |
| `track_broken` | One vehicle split across tracks | Tracker tuning | other |
| `track_merged` | Two vehicles in one track | Discontinuity / session handling | other |
| `duplicate` | One vehicle, several sightings | Dedup window | other |
| `dropped_frame` | Vehicle never sampled | Sampling interval / throughput | other |

Only **`ocr_wrong`** is a miss the OCR-recogniser fine-tune in
[`config/training.yaml`](../../config/training.yaml) could fix. A dominant bucket
is necessary but not sufficient to rent the A100: it also has to be *that*
bucket. Every other row is a real remedy that belongs to a different lane or to
camera placement, and none of them is bought with the fine-tune.

---

## Results

Source: `benchmarks/reports/FAILURE_TAXONOMY.json`, classified by the data lane
from `e2e_fusion_{off,on}_paddle_fixed_distance_001.json`. Re-derived here by
`ai.quality.taxonomy.FailureTaxonomy` rather than copied, so the numbers in this
table and the verdict below come from the same code that gates the spend.

| Bucket | Count | Share |
|---|---|---|
| `plate_too_small` | 2019 | 74.0% |
| `plate_miss` | 300 | 11.0% |
| `ocr_partial` | 225 | 8.2% |
| `ocr_wrong` | 180 | 6.6% |
| `fusion_wrong` | 4 | 0.1% |
| `vehicle_miss` | 0 | — |
| `track_broken` | 0 | — |
| `track_merged` | 0 | — |
| `duplicate` | 0 | — |
| `dropped_frame` | 0 | — |
| **total** | **2728** | 100% |

**Dominant bucket:** `plate_too_small` (74.0%, 6.7× the runner-up)
**Verdict:** `dominant` — clears the floor of 30 and the margin of 1.25
**Points at:** `no_software_fix`
**Unlocks the OCR fine-tune:** **false**

### What this does not cover

The five zeros are **structural, not measured clean**. The corpus behind this
histogram is single-frame plate crops with no vehicle-detection and no tracking
stage, so `vehicle_miss`, `track_broken`, `track_merged`, `duplicate` and
`dropped_frame` had nothing to occur in. They are zero because they could not
happen here, and quoting this table as full-pipeline coverage would be a
misreading. The verdict is sound **for the OCR and plate-detection path**, which
is the only path the A100 was ever proposed for.

Two further limits worth stating before anyone quotes the 74%:

- The threshold is a **mean plate height below 20 px**, measured by the data
  lane as the point detection collapses. This module originally assumed 30 px;
  the measurement outranked the assumption and `PLATE_TOO_SMALL_PX` is now 20.
  The direction matters: 20 px is the setting *more* favourable to training,
  since it leaves the 20–30 px band in the fixable buckets rather than the
  no-fix one. The no-train verdict therefore holds under the assumption most
  generous to spending, which is the only way it is worth much.
- Accuracy is measured on **synthetic plates**; the real-footage rows in
  TRINETRA-HARD carry no verified ground-truth text and can only detect
  fabrication. This bounds how far the histogram generalises to real cameras.

---

## From verdict to spend

The verdict maps to exactly one next action, and this run landed on the second:

- **`ocr_wrong` dominant** → the fine-tune is the justified spend. Not this run:
  `ocr_wrong` is 6.6%, fourth place.

- **`plate_too_small` dominant** → **stop.** Deliver the width-bucket report and
  a camera-placement recommendation. No training run. ← **this run**

- **Any other bucket dominant** → a real fix, but not this lane's A100. Route it
  to the owning lane (detector data, tracker tuning, dedup, throughput).

- **Co-dominant or insufficient** → the analysis is not finished. Classify more
  misses; do not open the gate.

### What this saves, and what it costs

At $1.64/hour against a 4-hour ceiling, the refused rental is **$6.56 not spent**
on the configured run — and up to ~$41 across the wider 25-hour envelope the
manual sketches. That is the smaller half of the point. The larger half is that
a fine-tune aimed at `ocr_wrong` would have improved 6.6% of misses while 74%
sat untouched, and the demo would have looked no better after a day of training
and a bill. The gate in `config/training.yaml` stays shut, and it stays shut for
a reason that is written down and reproducible rather than argued.

The honest deliverable in its place is the width-bucket table plus a
camera-placement recommendation: at these mounting positions and zoom levels, a
large share of plates arrive below the ~20 px floor, and no model recovers
information the sensor did not capture. That is a finding, not a failure — and
it is the kind a judge can check.

The dry run of the fine-tune loop — batching, the frozen-backbone phase,
checkpoint selection on `val_exact_match`, early stop, and the
ship-only-if-better decision — can still be exercised at any time without a
dataset and without spending anything:

```bash
python scripts/train.py --smoke
```
