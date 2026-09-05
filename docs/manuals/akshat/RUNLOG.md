## 2026-09-05 — Stopped OCR job, committed item-2 code, wrote OCR findings doc
**STATUS: OK — job killed cleanly, no data corruption**

- Killed the running full-corpus OCR cache rebuild (PIDs for
  `benchmarks.paddle_predictor` + 2 `ocr_worker.py` subprocesses) — its
  output would have been unusable: `plate_bbox`/`plate_width_px` use the
  full render canvas, not a tight plate crop, so every width bucket is
  mislabelled (see next entry for the fix).
- Committed item 2's code (fixed-distance builder, `track_type` schema
  field, `run.py --track-type` filter, `run_all.sh` stages) **separately**
  from the tainted `index.jsonl` — corrected per explicit instruction not to
  hold code hostage to a long-running job again.
- `docs/manuals/akshat/OCR_BASELINE_FINDINGS.md` written: condition
  breakdown at the (mislabelled, but condition-grouping is independent of
  the width bug) `>100` bucket — easy 64%, glare 49%, perspective 52%,
  night 26%, **motion_blur 0.5%**. Motion blur defeats PP-OCRv4-mobile
  almost completely regardless of plate size — a real, actionable finding
  for Manas's model selection, flagged as provisional pending the crop fix.

## 2026-09-05 — DIAGNOSIS: why >100px detection is 39.5%, not near-total
**STATUS: root cause found — real renderer, wrong crop box, not "bare text on a plain background"**

Dumped 10 clean/`easy`/eligible `>100` frames to `benchmarks/cache/inspect/`
(gitignored, inspect locally) plus their un-degraded 512x128 base images.
Visual + quantitative findings:

1. **The renderer is genuinely plate-like.** Correct Indian plate conventions
   (yellow=private, white=commercial, green=electric), visible plate body/
   border, correct bold sans-serif characters, correct character spacing,
   adequate contrast. This is NOT "bare text on a plain background" — that
   specific failure mode is ruled out.
2. **But `plate_bbox` does not match the plate region.** Every base image is
   a 512x128 render of a *tilted plate mounted against a sky background*,
   not a tight plate crop. Measured on 4 samples: the plate occupies ~94-97%
   of canvas **width** but only **~51-60% of canvas height** (one sample,
   the green EV plate, measured ~99% height with a naive saturation-based
   mask — false positive from the mask, not evidence that one is tightly
   cropped; visual inspection shows the same sky padding on all 4).
   `build_sequences.py` always records `plate_bbox: [0, 0, w, h]` — the
   **whole canvas**, sky included, never a tight crop.
3. **Consequence:** `plate_width_px` (and therefore `width_bucket`) is
   computed from the full canvas width, which is a reasonable proxy for
   plate width (that dimension is ~94-97% accurate) — but the *vertical*
   resolution actually available to the OCR engine is far less than the
   nominal frame height suggests, because ~40-49% of every frame's height is
   non-plate background. A "137px wide" frame is not "137px of plate text
   detail" once ~45% of its height is sky.
4. **This alone does not explain 39.5%.** Breaking the same `>100` bucket
   down by condition shows the real driver: `easy` 64% (192/298), `glare`
   49%, `perspective` 52%, `night` 26%, **`motion_blur` 0.5% (1/215)** —
   motion blur and low light genuinely defeat a lightweight PP-OCRv4 mobile
   model regardless of width. The 39.5% headline figure is a width-bucket
   average across all five conditions, not a "clean plate" number — the
   aggregate obscures this the same way an unweighted "ALL" rate obscures
   per-bucket collapse (CLAUDE.md §5's own point, just one level up).
5. Even within `easy` alone, 64% (not "near-total") is still lower than
   expected for a clean plate. The un-cropped sky padding (point 2) is the
   most likely remaining contributor — recommend a tight plate crop before
   building any new track type (item 2), since a fixed-distance "35px"
   track built from the same un-cropped renderer will carry the same defect.

**Action for item 2**: fixed-distance tracks should crop tight to the plate
region (or at minimum this should be fixed before trusting their numbers),
not reuse `build_sequences.py`'s `[0, 0, w, h]` full-canvas box as-is.

## 2026-09-05 — Fixed: stale "canned/illustrative" disclaimer on real PaddleOCR output
**STATUS: OK**

- `delta.py` hardcoded the stub's disclaimer text regardless of which
  predictor actually ran — `FUSION_DELTA_paddle.md` was telling readers real
  PaddleOCR output was "canned/illustrative, not a real model," the opposite
  of true. Fixed: `PREDICTOR_DISCLAIMERS` dict keyed by `predictor`, `stub`
  and `paddle` each get accurate text; an unregistered predictor name now
  gets an explicit "no disclaimer registered" warning instead of silently
  reusing the wrong one.
- Both `FUSION_DELTA.md` (stub) and `FUSION_DELTA_paddle.md` regenerated;
  confirmed correct labels on both.

## 2026-09-05 — PaddleOCR real predictor: first genuine number
**STATUS: OK — process-level isolation fixed the DLL conflict**

- Fix: `.venv-ocr` (isolated venv, no torch) resolves the earlier DLL collision.
  `paddleocr` import still hit a second, unrelated bug inside that venv —
  PP-OCRv6 crashes on this CPU's oneDNN backend
  (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute`); fixed by
  forcing `ocr_version="PP-OCRv4"` + `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False`
  in `scripts/ocr_worker.py`. Reported here since it's a real deviation from
  "just works," not silently patched over.
- `scripts/ocr_worker.py` (runs only in `.venv-ocr`): one PaddleOCR() load,
  reads `[{"path","box"}]` JSON on stdin, writes `{path: {"text","confidence"}}`
  on stdout. Multiple detected text regions per plate are joined
  left-to-right by x-coordinate into one string (a single-line plate is
  usually detected as several fragments, not one box).
- `benchmarks/paddle_predictor.py` (main env): materializes every
  `synthetic_truth` frame to a real PNG (reusing `build_sequences.py`'s own
  renderer via a new `frame_sink` hook — same pixels ground truth was
  generated from), runs OCR **once** over all 3,437 frames in one subprocess
  (~35 min), caches to `benchmarks/cache/ocr_readings.json` keyed by
  `frame_path` (gitignored). Fusion ON consensus: **highest-confidence whole
  reading across the TrackKey**, not per-character majority vote — OCR
  readings vary in length/fragment count frame to frame, so position voting
  isn't well-defined; picking the single most confident whole string is and
  mirrors a real fusion pipeline surfacing its best detection.
- `--predictor paddle` added to `run.py`; report's `predictor` field records
  which was used; `weights_sha256` is a real combined hash of the 30
  PP-OCRv4 det+rec model files on disk (path: `~/.paddlex/official_models/`).
- **Detection rate correlates with difficulty exactly as expected** (genuine
  signal, not a bug): >100px 39.5% (428/1083) down to 30-40px and <30px 0%
  (0/304, 0/390); easy slice 46%, motion_blur ~0%, tiny 0%.
- **Fusion-on side effect worth flagging**: because consensus propagates one
  track's best reading to every frame in that track, a track's fusion-on
  outcome is close to binary (right or wrong for the whole track), which is
  why every width bucket lands near the same ~0.21-0.23 rate under fusion ON
  — an emergent property of this consensus rule, not a bug (verified: no
  cross-track key collisions). It also **increases fabrication** (0 -> 164):
  a genuinely-unreadable frame can inherit a confident reading from elsewhere
  in its track. A real fusion design would need a per-frame eligibility gate
  before propagating a track consensus, not just the confidence score.
- All 6 scorer fixtures + regression-checked stub report still pass after
  wiring `--predictor paddle` in.
- Reports use distinct run ids (`*_paddle_001.json`, `FUSION_DELTA_paddle.md`)
  — the stub reports (`FUSION_DELTA.md`, run `_005`) are untouched.

## 2026-09-05 — BLOCKED: PaddleOCR install fails on import (DLL conflict with torch)
**STATUS: BLOCKED — stopped per instruction, did not substitute another engine**

- `pip install paddleocr paddlepaddle` succeeded (paddleocr 3.7.0, paddlepaddle
  3.3.1, both report "Successfully installed").
- `import paddleocr` (and even bare `import paddle` followed by `import torch`)
  fails: `OSError: [WinError 127] The specified procedure could not be found.
  Error loading ".../torch/lib/shm.dll" or one of its dependencies.`
  Reproduced twice. Plain `import torch` alone (no paddle) succeeds — importing
  `paddle` first corrupts the DLL search state so `torch`'s own `shm.dll`
  cannot resolve afterward. `paddleocr` unconditionally imports `modelscope`,
  which unconditionally imports `torch`, so any use of PaddleOCR in this
  process hits this conflict.
- No `PaddleOCR()` instantiation attempted — the import itself never
  completes. `benchmarks/paddle_predictor.py`, `--predictor paddle`, and the
  real-predictor run (2b-2e) are **not implemented** — there is nothing
  working to wire in yet.
- Not worked around: no alternate OCR engine substituted, no DLL/env hack
  attempted, per instruction ("report the error and stop").
- Likely fix (human, next session): a clean venv for paddleocr separate from
  the torch install already in this environment, or resolving the torch/MKL
  DLL conflict directly (this doesn't look paddleocr-specific — it's a
  Windows DLL search-order collision between paddle's and torch's native
  libraries in the same process).

## 2026-09-05 — Probabilistic stub + empty-bucket fixture
**STATUS: OK**

- `stub_predictor.py` rewritten: per-bucket probabilistic hit rate (seeded on
  `obs_id:fusion_enabled`, fully deterministic) instead of the old 1.00/0.00
  threshold. Fusion-off targets ~0.95/0.88/0.72/0.51/0.30/0.12 by bucket;
  fusion-on adds an uplift largest in the middle buckets. Misses return a
  1-character-corrupted plate string 85% of the time, `None` otherwise — the
  "no prediction" path still gets exercised, just not exclusively.
- Re-ran the full pipeline (run 005): `FUSION_DELTA.md` now shows fractional
  rates in every bucket (e.g. 40-60: 0.48 OFF -> 0.72 ON), landing close to
  the targets (seeded noise). Verified by hand: `delta` is computed from
  exact unrounded rates, not from the rounded display cells — no contradiction
  between displayed delta and raw counts.
- `scorer.py`: added fixture 7 — a bucket with `n=0` emits
  `{"n": 0, "correct": 0, "rate": None}`, no `ZeroDivisionError`. All 6
  original SPEC_BENCHMARK §5 fixtures + fixture 7 pass.

## 2026-09-05 — Cleanup + MONDAY.md
**STATUS: OK**

- Deleted `scripts/build_hard_candidates.py` — dead code since Phase 4R,
  unused by `run_all.sh`.
- `run.py`/`delta.py`/`stability.py`: report + JSON now carry an explicit
  `predictor` field/first-line label so stub output can never be mistaken
  for a real measurement. Reports regenerated (run 004).
- `docs/manuals/akshat/MONDAY.md` written: the two-line swap in `run.py` to
  replace `stub_predictor` with the real pipeline, the exact
  `predict(row, fusion_enabled) -> str | None` contract, and a 6-point
  checklist for prediction/ground-truth misalignment.

## 2026-09-05 — Phase 1B: Clip reservation
**FOR MANAS — clip reservation published, see datasets/trinetra-hard/CLIP_RESERVATION.md**

- 62 clip_ids / 5,000 frames recovered from indian_road (only 5/646 shards local).
- 31 clips / 2,460 frames RESERVED for eval, 31 clips / 2,540 frames TRAIN_SAFE.
- Split is by clip_id (never frame) per CLAUDE.md — safe to start training on
  TRAIN_SAFE list now.

## 2026-09-05 — Phase 1+2: Recon + Licenses
**STATUS: OK, with 1 finding for human sign-off (docs/manuals/akshat/RECON.md)**

- indian_road clip identity: recoverable, `{clip_id}_{frame}.ext` filename UUID
  prefix, 0 regex failures. Only 5/646 shards local (62 clips, 5000 frames).
- Plate-bbox datasets: gujarat_plates, indian_plates_yolo, kedarsai_plates,
  justjuu_plates. Vehicle-only: indian_road, traffic_vehicle.
- LICENSES.md: 4 verified rows (indian_road, justjuu_plates, cctv_accident,
  synthetic_plates), 5 flagged incl. gujarat_plates + indian_plates_yolo — no
  embedded license evidence found for those 2, contra TASKS.md's "6 rows"
  expectation. Reported, not guessed.
- `check_licenses.py` passes (no manifests reference an unverified asset yet).

## 2026-09-05 — Phase 3: Manifests + leakage checker
**STATUS: OK**

- `freeze_manifest.py`, `check_split_leakage.py` written, both pass `--demo`
  (leakage checker proven on a deliberately-broken fixture).
- Manifests generated for the 4 license-verified datasets only: indian_road
  (17 files), justjuu_plates (11), cctv_accident (13), synthetic_plates (18,000).
  Flagged datasets (gujarat_plates, indian_plates_yolo, kedarsai_plates,
  traffic_vehicle, fanvid) intentionally have no manifest — no LICENSES.md row.
- `check_split_leakage.py` on `CLIP_RESERVATION.md`: OK, 0 leaks, 62 clip_ids.
- `check_licenses.py`: OK, no manifest references an unverified asset.

## 2026-09-05 — Phase 4: Candidate labels
**STATUS: OK, badly under target (196/300) — reported honestly, not padded**

- `schema.json` written per SPEC_TRINETRA_HARD §4. `build_hard_candidates.py`
  samples only from license-verified sources (justjuu_plates, synthetic_plates)
  — the flagged datasets (gujarat_plates, indian_plates_yolo, kedarsai_plates)
  hold most of the real plate-bbox data but are excluded per LICENSES.md.
- Counts: easy 60/60, night 60/60, glare 45/45, tiny 30/30, **motion_blur 1/60**
  (real blur is rare in static photos), **perspective 0/45** (no source has
  plate pose/rotation data — not guessed).
- No OCR engine in this environment (no tesseract binary). justjuu_plates rows
  have `plate_text: null`; synthetic_plates rows use the filename (the
  generator's own ground truth). Every row is `label_source: ocr_candidate`.
- **Superseded** by Phase 4R (synthetic sequence corpus) — see below. This
  phase's real-photo candidate approach is replaced for the headline number.

## 2026-09-05 — Phase 5 CANCELLED (plan change)
**Human verification removed from the pipeline.**

`scripts/verify_ui.py` deleted. Phases 5 and 7 no longer exist — ground truth
now comes from synthetic generation (Phase 4R), never from a human or from
OCR. The pipeline runs end to end unattended. See Phase 4R/4S/6R entries below.

## 2026-09-05 — Phase 6: Scorer and harness
**STATUS: OK — gate passed, all six fixtures green**

- `benchmarks/scorer.py`: all 6 SPEC_BENCHMARK §5 fixtures pass (exact match,
  case/space, 0/O fuzzy-not-correct, missing prediction, fabrication on
  ineligible, ocr_candidate fully excluded).
- `stub_predictor.py`, `run.py` (`-m benchmarks.run`), `delta.py`
  (`-m benchmarks.delta`) written and run end-to-end on the real 196-row
  `index.jsonl` — no crash, correctly degrades to `n_eligible: 0` /
  `rate: null` everywhere because 0 rows were `label_source: human` at the
  time (that scoring path is now superseded — see Phase 4R/6R below). Reports
  + `FUSION_DELTA.md` committed as proof the harness ran clean; re-run against
  the synthetic corpus.
- Every report cites `dataset_manifest_sha256` + `git_commit`;
  `weights_sha256: null` with reason in `notes` (stub has no weights).
- Gate cleared — downstream numbers, once real labels exist, can be trusted.

## 2026-09-05 — Correction 1/3: license rows resolved
**STATUS: OK**

- Operator confirmed `gujarat_plates` (Kaggle `paneraghanshyam/gujarat-vehicle-
  number-plates-yolo-ready`, Apache 2.0) and `indian_plates_yolo` (Kaggle
  `deepakat002/indian-vehicle-number-plate-yolo-annotation`, CC0) from the
  Kaggle dataset pages — Kaggle licenses live on the page, not embedded in the
  archive, so the earlier flag was a false negative. LICENSES.md now has 6
  verified rows, matching TASKS.md.
- Manifests frozen: gujarat_plates (711 files), indian_plates_yolo (321).
- `check_licenses.py`: still OK.
- Both are now usable for TRINETRA-HARD candidate sourcing.

## 2026-09-05 — Phase 4R: synthetic sequence corpus (new headline)
**STATUS: OK**

- `scripts/synth/build_sequences.py` — 300 tracks sampled from `synthetic_plates`
  (seed 20260905), 8-15 frames each, **3,437 total rows**. Every row
  `label_source: synthetic_truth` — text is the generator's own filename,
  never OCR, never human. Width sweeps every bucket per track (real fusion
  test data at every width). Per-frame degradation baked in and recorded in
  `degradation_params`: motion_blur/night/glare/perspective/easy, plus ~10%
  (9.7% actual) deliberately-unreadable frames, `eligible: false`, kept.
- Row counts by slice: easy 742, motion_blur 552, night 582, glare 407,
  perspective 460, tiny 694 — **note this is a structural change from
  SPEC_TRINETRA_HARD's original ~300-single-observation design**: it's now
  ~300 multi-frame *tracks* (3,437 rows), per the plan-change instruction.
- `schema.json` extended: `label_source` enum +`synthetic_truth`, new optional
  `degradation_params`, `plate_bbox_source`.
- `scorer.py` updated to treat `synthetic_truth` as scoreable (same as
  `human`); `ocr_candidate` still fully excluded. All 6 SPEC_BENCHMARK §5
  fixtures re-run and pass, plus 1 new assertion for `synthetic_truth`.
- Smoke-tested `benchmarks.run` against the new corpus end-to-end: real
  non-null rates per width bucket, no crash.

## 2026-09-05 — Phase 4S: real-footage tracks + stability (diagnostic only)
**STATUS: OK — 4,816 indian_road rows added, never scored for accuracy**

- `scripts/build_real_tracks.py`: **4,816 rows / 580 tracks / all 31 RESERVED
  indian_road clips** (only, never TRAIN_SAFE). Real clip_id, real track_id
  (ByteTrack annotation), real frame index -> `source_pts_ms` (1fps per the
  dataset's own README). `eligible: false`, `plate_text: null`, always —
  indian_road has **no plate-region annotation**, only vehicle-level BDD100K
  boxes, so true plate text is genuinely unknown here. `plate_bbox` is a
  documented heuristic estimate from the real vehicle box
  (`plate_bbox_source: estimated_from_vehicle_bbox`), not a measurement.
- New `label_source: unverified_real` — excluded from `scorer.py`'s accuracy
  path by construction (only `human`/`synthetic_truth` are scored); confirmed
  `n_eligible` unchanged (3,102) after adding these rows.
- `benchmarks/stability.py`: agreement-across-frames-of-a-TrackKey diagnostic,
  written to `STABILITY.md`/`.json`, clearly headed as non-accuracy. 580
  tracks, 4,812 frames, 550/580 tracks show higher agreement with the stub's
  simulated fusion. This is a plumbing validation (illustrative stub
  predictor), not a real fusion measurement.
- `check_split_leakage.py` still OK (62 clip_ids, 0 leaks) — the reservation
  itself is untouched.

## 2026-09-05 — Phase 6R: single-command unattended flow
**STATUS: OK — full pipeline runs clean, no prompts**

- `scripts/run_all.sh`: sequences -> real tracks -> freeze manifests (6
  datasets) -> leakage check -> license check -> scorer self-check -> fusion
  OFF -> fusion ON -> delta -> stability. One command, exits non-zero on any
  step failure (`set -euo pipefail`), no interaction required.
- Fixed a real bug found while running it: `run.py`'s notes wrongly said "0
  human-verified rows" even when 3,102 `synthetic_truth` rows were scored —
  it only checked for `label_source == "human"`. Fixed to check both
  scoreable sources; notes now correctly report the 4,816 `unverified_real`
  rows excluded instead.
- `FUSION_DELTA.md` first line now states ground truth is synthetic-generated
  and indian_road is stability-only, per instruction. Fusion delta on the
  synthetic corpus: ALL 0.47 (1465/3102) OFF -> 1.00 (3102/3102) ON — this is
  the stub predictor's designed behavior (illustrative, not a real result).
  Fabrication count OFF 169 / ON 335 (stub always emits ground truth on
  fusion, including for ineligible frames — an intentional illustrative
  worst-case, not a real model property).
- All 9 steps verified green end to end.

## 2026-09-05 — Phase 0: Environment
**STATUS: OK**

- Repo toplevel: `A:/trinetra hackathon final/sentinel-hack`, branch `data/trinetra-hard`, tracking `origin/data/trinetra-hard`.
- `datasets/raw` junction resolves, contains 9 dirs: cctv_accident, fanvid, gujarat_plates, indian_plates_yolo, indian_road, justjuu_plates, kedarsai_plates, synthetic_plates, traffic_vehicle.
- `.gitignore` line 30 covers `datasets/raw/`.
- `py -3.11 --version` → Python 3.11.0. OK.
- No blockers. Proceeding to Phase 1.
