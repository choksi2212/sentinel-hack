# Failure taxonomy — baseline benchmark

**Status:** scaffold. The bucket counts below are placeholders (`—`) until the
baseline benchmark has run on the local 12 GB card. The labelled data and the
benchmark clips are not this lane's to produce (see the data-engineering
manual), so the numbers are filled in after that step, not before.

This is the deliverable that decides whether any money is spent. Owner's manual
section 6: after the baseline benchmark, **every** miss is classified into
**exactly one** of ten buckets, and the shape of that histogram — not a hunch —
is what unlocks (or does not unlock) the rented A100.

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
| `plate_too_small` | Plate < 30 px | Nothing in software — camera placement, or accept it | **no fix** |
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

## Results — MANUAL STEP (fill from the baseline benchmark)

```
-------------------------------------------------------------------
  MANUAL STEP REQUIRED — populate from the local baseline benchmark
  across the 11 reporting clips. Classify each miss into one bucket;
  counts feed ai.quality.taxonomy.FailureTaxonomy, which produces the
  verdict below.
-------------------------------------------------------------------
```

| Bucket | Count | Share |
|---|---|---|
| `vehicle_miss` | — | — |
| `plate_miss` | — | — |
| `plate_too_small` | — | — |
| `ocr_wrong` | — | — |
| `ocr_partial` | — | — |
| `fusion_wrong` | — | — |
| `track_broken` | — | — |
| `track_merged` | — | — |
| `duplicate` | — | — |
| `dropped_frame` | — | — |
| **total** | **—** | 100% |

**Dominant bucket:** —
**Verdict:** — (`dominant` / `co_dominant` / `insufficient`)
**Unlocks the OCR fine-tune:** — (true only if the dominant bucket is `ocr_wrong`)

---

## From verdict to spend

The verdict maps to exactly one next action:

- **`ocr_wrong` dominant** → the fine-tune is the justified spend. Proceed to the
  `config/training.yaml` gate, which *still* requires a labelled dataset, a
  held-out split, and a measured baseline before it opens. Confirm with:

  ```bash
  python scripts/train.py
  ```

- **`plate_too_small` dominant** → stop. Deliver the width-bucket report and a
  camera-placement recommendation. No training run.

- **Any other bucket dominant** → a real fix, but not this lane's A100. Route it
  to the owning lane (detector data, tracker tuning, dedup, throughput).

- **Co-dominant or insufficient** → the analysis is not finished. Classify more
  misses; do not open the gate.

The dry run of the fine-tune loop — batching, the frozen-backbone phase,
checkpoint selection on `val_exact_match`, early stop, and the
ship-only-if-better decision — can be exercised at any time without a dataset
and without spending anything:

```bash
python scripts/train.py --smoke
```
