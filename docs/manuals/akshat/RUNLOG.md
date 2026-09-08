## 2026-09-06 — FINDINGS.md: width/height header fix, 2/519 acknowledged both ways
**STATUS: OK**

- `delta.py` table header: `Bucket` -> `width bucket`, `plate height (px)` ->
  `mean plate height (px)` — was ambiguous (60-80 read as a height, not a
  width bucket). Both `FUSION_DELTA_paddle_approach.md` and
  `_fixed_distance.md` regenerated with the corrected header.
- `FINDINGS.md` Finding B: table header matches; added explicit sentence
  that `60-80px` is not perfectly zero (2/519 correct without fusion) rather
  than letting "fusion changes nothing" round it away.
- `FINDINGS.md` Finding C: added the same 2/519 fact from the other
  direction — fusion turned those same 2 correct readings into 0 at
  `60-80px`. Same track-level consensus mechanism as the fabrication cost,
  stated as the other side of the tradeoff: it can overwrite a correct
  single-frame answer, not only rescue or fabricate one.
- Regression: `scorer.py` still passes (6+2 fixtures).

## 2026-09-06 — Floor threshold fixed to 20px, FINDINGS.md written
**STATUS: OK — one more correctness catch along the way**

- `delta.py`: `OCR_FLOOR_HEIGHT_PX` 10 -> 20px, reworded to "below ~20px of
  plate height this engine detects no text regardless of fusion — an
  empirical floor measured on this corpus, not a scorer defect." Both
  tables regenerated; fixed-distance now correctly flags all 4 dead buckets
  (60-80/40-60/30-40/<30), not just the two under the old 10px line.
- **Caught while regenerating**: the floor note is height-only, so it also
  fired on the **approach** table — where those same buckets actually read
  ~0.19-0.20 (the track-consensus artifact), not zero. The note would have
  said "detects no text" next to a table showing 20%. Fixed: the note now
  requires the rate be under 0.05 in **both** fusion states before firing,
  not just the height. Approach table: no floor note (correctly — its
  numbers aren't a floor, they're an artifact). Fixed-distance: unchanged,
  fires correctly.
- `docs/manuals/akshat/FINDINGS.md` written: 3 findings with raw counts
  (fusion >100px 0.19->0.43 n=494; ~20px hard floor; fabrication 0->113),
  approach-vs-fixed-distance as the methodology note. Under 2 pages.
- Regression: `scorer.py` (6+2 fixtures), `paddle_predictor.py --demo`
  both pass.

## 2026-09-06 — THE HONEST PER-BUCKET RESULT: approach-track flatness confirmed as artifact
**STATUS: OK — item 2's hypothesis confirmed by real data, both tables committed**

Full corrected corpus (tight crop + Windows-illegal-char fix), 6,822 frames
OCR'd, 1,160 non-null readings. `--track-type approach` vs `fixed_distance`,
run separately, `FUSION_DELTA_paddle_approach.md` / `_fixed_distance.md`:

- **approach** (width sweeps within track): fusion ON lands at ~0.19-0.21 in
  **every** bucket including `<30`/`30-40` — this is the corpus artifact
  flagged before: one easy frame's reading gets credited to the whole
  track. Kept as a labeled artifact demonstration, not an accuracy claim.
- **fixed_distance (the honest table)**: >100 0.19->0.43 (+0.24), 80-100
  0.02->0.16 (+0.13) — fusion helps, real signal, at real size. **60-80,
  40-60, 30-40, <30 are ~0.00 for BOTH fusion OFF and ON** — not just the
  two buckets flagged `(below OCR floor)` by the 10px height cutoff
  (30-40 @ 8px, <30 @ 5px). **60-80 (15px) and 40-60 (12px) are equally
  dead in this table despite being above that 10px line** — the 10px flag
  catches the two most extreme buckets, not the true practical floor for
  this engine, which is closer to ~20px (80-100's own rate is only 0.02
  OFF / 0.16 ON). Flagging this explicitly rather than letting the
  automatic note imply "only 2 buckets are floored."
- Fabrication count (fixed_distance): 0 OFF, 113 ON — same mechanism as
  before (consensus can propagate a confident reading onto a frame that
  was individually unreadable), now on a corpus where that mechanism can't
  hide behind a sweep.
- All 4 reports + both delta tables committed.

## 2026-09-08 — Gave the 2 orphaned rules real homes (my earlier check was incomplete)
**STATUS: OK — both facts already existed, I'd just missed the file**

- Correction to my own earlier finding: I checked `SPEC_BENCHMARK.md` and
  `SPEC_TRINETRA_HARD.md` and reported neither rule existed anywhere
  tracked. I hadn't checked `docs/manuals/akshat/DATASETS.md` — it already
  had the clip-split rule in more depth than CLAUDE.md did (§"leakage
  rule") and already stated the license-row mandate in its own intro prose,
  just not as a citable numbered section.
- `DATASETS.md`: promoted the intro's license-row sentence into a real
  `## 1. The license-row mandate` section; renumbered the existing 6
  sections to 2-7 (verified first: nothing else in the repo cited
  `DATASETS.md` by section number, so this was safe).
- `SPEC_TRINETRA_HARD.md` §3 already had the clip-split rule verbatim — no
  edit needed there, just repointed citations to it.
- Repointed all 6 standalone statements: `datasets/LICENSES.md` (2) and
  `scripts/check_licenses.py` (1) now cite `DATASETS.md` §1;
  `datasets/trinetra-hard/CLIP_RESERVATION.md`, `RUNLOG.md`, and
  `scripts/check_split_leakage.py` now cite `SPEC_TRINETRA_HARD.md` §3.
- Regression: `check_licenses.py`, `check_split_leakage.py`, `scorer.py` all pass.

## 2026-09-08 — Item 3: dev assets for Manas — script, not LFS
**STATUS: OK — 6 assets, all built and verified, chose script over LFS**

- `scripts/fetch_dev_assets.py`: extracts from the 31 RESERVED indian_road
  clips only (never TRAIN_SAFE) — 3 daytime clips (highway/city/village,
  60s each, visible plates), 1 night clip (90s, residential road,
  `scene_attributes.json` confirms `timeofday: night`), 1 hard-scene-cut
  clip (60s; cut located by scanning all five 180-frame RESERVED clips'
  64-bin greyscale-histogram frame-to-frame distance — largest jump 0.748
  at frame 8), and a 100-frame plain-JPEG sequence for `FrameSequenceSource`.
- **Chose the script+SHA-256 fallback deliberately, not because LFS was
  unavailable** — both `git lfs` and the GitHub endpoint work on this
  machine. Reasoning: these clips are a deterministic function of 5 tar
  shards Manas needs anyway; a script + recorded SHA-256 of those 5 shards
  proves reproducibility directly, committing ~80MB of video through git
  history forever does not, and it matches this repo's own existing
  convention (`.gitignore`: "the script that produces it is committed
  instead"). Script verifies all 5 tar hashes before extracting anything.
- Ran it end to end: all 6 assets built, `ffprobe`-verified (`clip_night.mp4`:
  1920x994, 1fps, 90 frames, 90.000000s duration — matches exactly).
- **Honest caveat, in the README and the script's own docstring**:
  indian_road's local frames are 1fps keyframes, not continuous footage —
  these MP4s are real, playable, correct-duration video, but a 1fps
  slideshow, not smooth motion. Flagged for decode/ingestion smoke tests
  only, not motion/latency testing.
- Generated binaries gitignored (`datasets/dev_assets/*.mp4`,
  `frame_sequence_100/`); only the script + README are committed.

## 2026-09-08 — Item 4: by_slice.motion_blur == 0.0 verified real
**STATUS: OK — confirmed real, one important correction to earlier framing**

- **`MONDAY.md` does not currently mention `motion_blur` anywhere** — grepped
  it directly, zero matches. The "flagged in MONDAY.md" premise doesn't hold
  for the file as it exists; noting this rather than pretending it's there.
- Checked every paddle report's `by_slice.motion_blur`: it is `0.0` (a real
  float) not `null` in 5 of 6 — `scorer.py`'s own by-slice logic already
  emits `null` for `n=0` (that's the empty-bucket fixture from a few items
  back), so a `0.0` value is proof by construction that `n>0`. Recomputed
  exact n/correct directly: **approach OFF n=498 correct=0; approach ON
  n=498 correct=0; fixed_distance OFF n=406 correct=0** — all three are a
  genuine, substantial-n zero, not an empty slice.
- **Correction to earlier framing**: `fixed_distance ON` is **not** zero —
  **n=406, correct=50, rate=0.123**. Fusion recovers ~12% of motion-blur
  frames when tracks vary condition per-frame (fixed_distance design) rather
  than per-track (approach design, which is the same track-consensus
  mechanism already flagged for the width-bucket artifact: a fixed_distance
  track can mix a blurred frame with a non-blurred one, letting the track's
  best reading get credited to the blurred frame too). This means
  `OCR_BASELINE_FINDINGS.md`'s claim "motion blur defeats PP-OCRv4-mobile
  ... independent of plate size" was accurate for the approach-track data it
  was measured on (1/215, pre-track-type-split) but is not the full picture
  now that fixed_distance exists — worth a follow-up note there, not done
  in this pass since it wasn't asked for.

## 2026-09-08 — Item 2: FAILURE_TAXONOMY.json, dominant bucket found
**STATUS: OK — verdict is DOMINANT, plate_too_small, no software fix**

- `benchmarks/failure_taxonomy.py`: classifies every miss in the
  fixed-distance fusion-ON run (eligible rows only) into exactly one of the
  10 keys `ai/quality/taxonomy.py` accepts (read-only import — never wrote
  `ai/`). Priority: `fusion_wrong` first (correct OFF, wrong ON — checked
  first so it isn't silently absorbed by the other rules), then
  `plate_too_small` (bucket mean height <20px), then `plate_miss` (None
  prediction), then `ocr_partial`/`ocr_wrong` (edit distance 1-2 vs 3+).
- **Result: 2,728 classified (>=30: yes). Leader `plate_too_small` 2,019 vs
  runner-up `ocr_partial` 300 — 6.7x, far past 1.25x. Verdict: DOMINANT,
  points_at `no_software_fix`.** Full breakdown: plate_miss 300,
  plate_too_small 2019, ocr_wrong 180, ocr_partial 225, fusion_wrong 4.
- `vehicle_miss`/`track_broken`/`track_merged`/`duplicate`/`dropped_frame`
  set to 0 with an explicit note: structurally zero (no vehicle-detection
  or tracking stage in this synthetic pipeline), not measured-and-clean.
- **Flagged, not silently resolved**: `ai/quality/taxonomy.py`'s own
  docstring documents `plate_too_small` as "Plate < 30 px"; this task's
  explicit instruction was 20px. Used 20px as instructed; noted the
  mismatch against the ai/ lane's own module in both the JSON's `notes`
  and here.

## 2026-09-08 — Item 1: untracked CLAUDE.md, reworded its 8 §5 citations
**STATUS: OK, with a real citation-target problem flagged**

- `git rm --cached CLAUDE.md`, added to `.gitignore` — stays on disk, keeps
  working, no longer tracked.
- Found the 8 citations across the 6 files named: `datasets/LICENSES.md`
  (2), `datasets/trinetra-hard/CLIP_RESERVATION.md` (1),
  `docs/manuals/akshat/FINDINGS.md` (1), `docs/manuals/akshat/RUNLOG.md` (2),
  `scripts/check_licenses.py` (1), `scripts/check_split_leakage.py` (1).
- **Only 2 of the 8 are actually facts SPEC_BENCHMARK.md documents**
  (fabrication-counted-separately -> §2, never-a-single-average -> §1 — both
  reworded to cite those, correctly, not a blind §5->§5 swap: SPEC_BENCHMARK's
  own §5 is "Stub predictor," unrelated). **The other 6 are about the
  LICENSES.md row-mandate (3x) and the indian_road clip-split-by-ID rule
  (3x) — neither fact exists in SPEC_BENCHMARK.md, and I checked
  SPEC_TRINETRA_HARD.md too; it isn't there either.** Rewording those to cite
  SPEC_BENCHMARK.md would have been a false citation pointing at a section
  that doesn't contain the claim. Reworded those 6 to state the rule
  standalone ("this lane's standing rule") instead of inventing a home for
  them. Flagging this rather than silently complying — these 2 facts may
  need a real documented home if they matter to Manas/Mihir/Parth too.
- Regression: `check_licenses.py`, `check_split_leakage.py`, `scorer.py`
  all still pass.

## 2026-09-06 — Character-height column + a real silent-failure bug found while adding it
**STATUS: fixed, OCR rebuild restarted with corrected data**

- `scorer.py`: `by_plate_width[bucket]` now also carries `mean_height_px`.
  `delta.py`'s table gains a `plate height (px)` column, flags any bucket
  averaging under 10px as `(below OCR floor)`, and adds an explicit note:
  "Operational floor, not a bug" when triggered — a ~0.00 rate in `<30`/
  `30-40` should read as a floor, not a defect, per instruction.
- **Found while wiring this up**: `benchmarks/paddle_predictor.py`'s
  `_sanitize()` didn't strip `<`/`>` — and two width buckets are literally
  named `<30` and `>100`. Every `fixed_distance` frame in those two buckets
  (1,129 of 3,385 — the >100 and <30 buckets **entirely**, 559+570) silently
  failed `cv2.imwrite` (Windows-illegal filename chars, no exception raised)
  during the overnight rebuild. The OCR subprocess was running and consuming
  real CPU the whole time — it just never had those frames to read, and
  would have produced misleadingly-clean-looking null/error entries for
  100% of exactly the two buckets this delta-table feature was built to
  explain honestly. Caught by checking actual file counts against expected
  rows, not by trusting the process was "running" (it was, and unhelpfully).
- Fixed: `_sanitize()` now maps `<`/`>` to `lt`/`gt` (plus `"`, `|`, `?`, `*`
  for completeness), leaving every previously-working filename byte-
  identical (verified). Added `paddle_predictor.py --demo` — this repo's
  first runnable check for this specific class of bug.
- Rebuild restarted; all 6,822 synthetic frames now materialize (verified:
  `ls benchmarks/cache/frames | wc -l` == 6822, was silently stuck at 5,693).
- Committing this now, before the ~75 min OCR run, per this session's own
  precedent: code doesn't wait on a long job.

## 2026-09-05 — Fixed the untight plate crop -- width buckets did NOT move
**STATUS: fixed, with an important correction to the original premise**

- `compute_tight_plate_bbox()` added to `build_sequences.py`: crops each
  source plate image tight to the plate body (excludes the sky background
  the renderer places around the tilted plate) using a top-strip median
  colour as the sky reference (robust to a foreign object touching one
  bottom corner in one sample, which broke an earlier all-4-corners
  attempt). `load_tight_plate()` applies this once per plate before any
  resize/degrade step, so every subsequent frame (both track types) is
  built from the tight crop, not the full 512x128 canvas. Aspect ratio is
  now the tight crop's own measured ratio, not the fixed 512:128 assumption.
- **Correction: the width_bucket distribution is IDENTICAL before and after**
  (verified: `>100` 1083/1083, `80-100` 551/551, `60-80` 554/554, `40-60`
  555/555, `30-40` 304/304, `<30` 390/390 for approach; matching for
  fixed_distance). This contradicts the original premise that "every row is
  in the wrong width bucket" -- **horizontally**, the tight-crop measurement
  showed the plate already spans ~94-100% of the canvas width in every
  sample checked (the tilted render fits the plate to the frame width by
  construction). The real defect was **vertical**: canvas height was fixed
  at a 4:1 aspect ratio regardless of the plate's true ~5:1-7:1 ratio,
  wasting 17-49% of every frame's height on sky. `plate_width_px` (and
  therefore `width_bucket`) was correct; **effective detail per labeled
  width was not** -- every frame now has a shorter, fully-plate height at
  the same labeled width, which should measurably help detection without
  moving any row between buckets.
- Cleared `benchmarks/cache/frames/` and `ocr_readings.json` (gitignored,
  stale -- keyed by `frame_path`, which is unchanged by this fix, so old
  un-cropped PNGs would silently survive a re-materialize otherwise).
- All checks re-run clean: `scorer.py` (6+2 fixtures), `check_split_leakage.py`
  (0 leaks), `build_sequences.py --demo` (new assertions for
  `compute_tight_plate_bbox`, including the no-plate-found fallback).
- Committing this now, before the OCR rebuild (~70+ min) — per correction,
  code and verified-correct data go in on their own, not held for a long job.

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
   per-bucket collapse (SPEC_BENCHMARK.md §1's own point, just one level up).
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
- Split is by clip_id (never frame) per `SPEC_TRINETRA_HARD.md` §3 — safe to start training on
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
