# real_predictor — handing the benchmark to the staged pipeline

The reciprocal of [`docs/manuals/akshat/MONDAY.md`](../manuals/akshat/MONDAY.md).
That document says what `benchmarks/run.py` needs; this one says what the AI lane
built, where it is, and the exact patch that connects the two.

- **Module:** [`ai/real_predictor.py`](../../ai/real_predictor.py) — `predict(row, fusion_enabled) -> str | None`
- **OCR worker:** [`scripts/real_ocr_worker.py`](../../scripts/real_ocr_worker.py) — runs only inside `.venv-ocr`
- **Tests:** [`tests/test_real_predictor.py`](../../tests/test_real_predictor.py) — 41 tests, no models or dataset required

It routes through `ai/ocr` → `ai/quality` → `ai/fusion`: the crop, the width
floor, the six preprocessing variants, the quality score that becomes the fusion
weight, and weighted-evidence consensus per TrackKey. That is the code the
pipeline ships, so the report measures the pipeline rather than a benchmark-local
approximation of it.

`benchmarks/paddle_predictor.py` stays. It asks PaddleOCR to detect text in the
whole frame and keeps each track's most confident single reading — a fair
off-the-shelf baseline, and a different thing. **The delta between the two
reports is the number worth having**, because it prices the staged design rather
than asserting it.

---

## The patch to `benchmarks/run.py`

Five hunks, each anchoring on a unique line in today's `run.py`. Nothing in
`scorer.py` or `delta.py` changes.

```diff
@@ imports @@
 from benchmarks.scorer import score
 from benchmarks import stub_predictor
 from benchmarks import paddle_predictor
+from ai import real_predictor

@@ build_report @@
     if args.predictor == "stub":
         weights_sha256 = None
         notes.append("weights_sha256: n/a -- stub predictor has no weights file to hash.")
+    elif args.predictor == "real":
+        weights_sha256, weights_note = real_predictor.weights_info()
+        notes.append(weights_note)
     else:
         weights_sha256, weights_note = paddle_predictor.weights_info()
         notes.append(weights_note)

@@ build_report, the diagnostics block @@
         "diagnostics": {
             "precision": None, "recall": None, "map50": None,
             "ocr_exact_accuracy": None, "cer": None,
             "fuzzy_match_rate": scored["fuzzy_match_rate"],
             "fps": None, "latency_p50_ms": None, "latency_p95_ms": None,
             "vram_peak_mb": None, "real_time_factor": None,
+            **(real_predictor.diagnostics() if args.predictor == "real" else {}),
         },

@@ main @@
-    p.add_argument("--predictor", default="stub", choices=["stub", "paddle"])
+    p.add_argument("--predictor", default="stub", choices=["stub", "paddle", "real"])
@@ main, after rows are filtered @@
-    predictor_module = stub_predictor if args.predictor == "stub" else paddle_predictor
-    if args.predictor == "paddle":
-        paddle_predictor.set_track_index(rows)
+    predictor_module = {
+        "stub": stub_predictor,
+        "paddle": paddle_predictor,
+        "real": real_predictor,
+    }[args.predictor]
+    if args.predictor == "paddle":
+        paddle_predictor.set_track_index(rows)
+    elif args.predictor == "real":
+        real_predictor.set_rows(rows)
     predictions = {r["obs_id"]: predictor_module.predict(r, fusion_enabled) for r in rows}
```

`set_rows(rows)` is the counterpart of `paddle_predictor.set_track_index(rows)`
and is **required before any `--fusion on` run** — it materialises the frames,
runs the OCR batch, and precomputes each track's consensus. `predict` with
`fusion_enabled=True` and no preload raises rather than answering from a single
frame and calling it fusion. It takes the row set *after* `--track-type`
filtering, so a per-bucket report cannot borrow evidence from outside its bucket.

Report filenames do not collide: `run.py` already appends the predictor name, so
these land as `e2e_fusion_{off,on}_real_001.json`.

---

## Before the first run

`.venv-ocr` and the PP-OCRv4 weights are a **MANUAL STEP** on a fresh machine —
paddle and this repo's torch collide at the Windows DLL level in one process, so
the OCR worker lives in its own interpreter:

```bash
py -3.11 -m venv .venv-ocr && .venv-ocr/Scripts/pip install paddleocr paddlepaddle opencv-python
```

`_run_ocr_batch` raises a `FileNotFoundError` naming this step if the venv is
missing, rather than surfacing a Windows error code from `subprocess`.

First `set_rows` renders every synthetic frame to `benchmarks/cache/frames/`
(reusing `paddle_predictor`'s sanitiser, so both predictors share the one
directory) and writes readings to `benchmarks/cache/staged_readings.json`.
Subsequent runs reuse them. The cache records the OCR version, a hash of the
worker, and `MIN_FUSION_WEIGHT`; change any of those and it re-reads rather than
re-reporting yesterday's numbers.

---

## Then run it

```bash
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion off --predictor real --out benchmarks/reports/
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion on  --predictor real --out benchmarks/reports/
py -3.11 -m benchmarks.delta --before benchmarks/reports/e2e_fusion_off_real_001.json --after benchmarks/reports/e2e_fusion_on_real_001.json --out benchmarks/reports/FUSION_DELTA_real.md
```

MONDAY.md's checklist applies unchanged. Two of its items are worth re-reading
before trusting the first report:

- **`non-null predictions == 0`** would mean the worker never read anything —
  check `diagnostics.frame_errors` and `diagnostics.frames_with_text`, which
  distinguish "the engine declined" from "every frame raised".
- **A rate of exactly 0.0 or 1.0** is a red flag, not a result. `predict` never
  reads `row["plate_text"]`; `tests/test_real_predictor.py` pins that on the AST,
  so a suspiciously good number is a bug somewhere else.

---

## Two things that read differently from the paddle report

**1. `fabrication_count` is now a measurement, not a zero.**

MONDAY.md suggests returning `None` for `eligible: false` rows. This predictor
does not, by default (`RETURN_NONE_ON_INELIGIBLE = False`), and the reasoning is
in the module. Short version: `scorer.py` excludes ineligible rows from
`n_eligible` and `n_correct` entirely, so the choice **cannot move the headline
accuracy rate by one row** — the only field it touches is `fabrication_count`.
Gating on `row["eligible"]` would drive that to 0 by construction, for every
predictor, forever, and a metric that reads 0 because the predictor was told the
answer measures nothing.

Left off, it measures exactly what `MIN_FUSION_WEIGHT` was built to reduce. The
surface is real: 690 ineligible synthetic rows, 649 of them wide enough that the
width floor will not refuse them, and **403 of 600 tracks mix eligible with
ineligible frames** — so `--fusion on` carries its consensus onto ineligible rows
and should fabricate strictly more than `--fusion off`. That trade is a finding
for `FUSION_DELTA`, not a defect to suppress. Set the flag `True` to produce the
suppressed number for comparison; `diagnostics.return_none_on_ineligible` records
which mode ran, so a report cannot be misread either way.

One caveat to state wherever the number is quoted: with fusion on, a track's
answer is scored once per frame, so this counts fabricated **frames**, not
fabricated vehicles. `diagnostics.tracks` is alongside it for the conversion.

**2. `weights_sha256` covers the recogniser only.**

The staged engine is rec_only — the plate box comes from the row's rendered bbox,
so PP-OCRv4's detector is never loaded. Hashing it would claim provenance over a
model that did not run, and would compare equal across two runs that differed in
the only model either of them used. The note in `notes` says so explicitly.

---

## What it does not cover

- **indian_road rows are never read.** They point inside a `.tar` shard with no
  loader, and `scorer.py` drops `unverified_real` rows before eligibility is even
  considered — they contribute to no reported number, not accuracy and not
  fabrication. `predict` returns `None` and the worker is never asked for them.
  If real-footage inference is ever wanted, that loader is new work, not a
  configuration change.
- **Accuracy is measured on synthetic plates.** Same limit as the
  [failure taxonomy](failure_taxonomy.md): the real-footage rows carry no
  verified ground-truth text, so they can detect fabrication and nothing else.
- **`ai/detect`, `ai/track` and `ai/plate` do not run here.** The corpus hands
  over the plate box directly, so this measures the OCR → quality → fusion path,
  which is the path the A100 was ever proposed for. The five structural zeros in
  the failure taxonomy have the same cause and the same scope caveat.
