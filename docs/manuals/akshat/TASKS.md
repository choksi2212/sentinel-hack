# TASKS — Akshat's lane

Ordered. Each phase ends with a commit, a push, and a `RUNLOG.md` entry.
Phases marked **[HUMAN]** cannot be done by an agent.

---

## Phase 0 — Environment (15 min)

- [ ] Clone repo, branch `data/trinetra-hard` off `origin/manas`
- [ ] `git push -u origin data/trinetra-hard` — repoints upstream off `manas`
- [ ] Confirm `git rev-parse --show-toplevel` is the repo, not a parent
- [ ] Junction `datasets/raw` → `A:\projects\Trinetra hackathon final\datasets\raw`
- [ ] Confirm `ls datasets/raw/` shows 9 directories
- [ ] Confirm `.gitignore` covers `datasets/raw/`
- [ ] `py -3.11 --version` works

**Gate:** if the junction fails, everything downstream is blocked. Stop and report.

## Phase 1 — Reconnaissance (30 min, read-only)

- [ ] Per dataset: tree to depth 2, file count, extension histogram, media type
- [ ] `indian_road`: state exactly which path component or filename pattern
      encodes **clip identity**. If not recoverable → **blocking finding**
- [ ] Count distinct clips and frames per clip in `indian_road`
- [ ] Existing annotation formats per dataset (YOLO txt, COCO json, CSV, none)
- [ ] Which datasets have plate-region bboxes vs vehicle-only

**Gate:** the clip-identity answer determines whether the fusion measurement is
possible at all. It is the single most important output of the night.

## Phase 2 — Licenses (20 min)

- [ ] `datasets/LICENSES.md` with a 9-field row per verified asset (6 rows)
- [ ] `traffic_vehicle`, `kedarsai_plates`, `fanvid` → flagged, **not** guessed
- [ ] `fanvid` marked eval-only in `used_for`
- [ ] `scripts/check_licenses.py` — fails if a manifest references an asset
      with no row

## Phase 3 — Manifests and leakage (45 min)

- [ ] `scripts/freeze_manifest.py` — walks a dataset, emits
      `datasets/manifests/<name>.sha256` (relative path + hash + size)
- [ ] `scripts/check_split_leakage.py` — asserts no `clip_id` in two splits;
      exits non-zero on violation
- [ ] Manifests generated for all verified datasets
- [ ] Leakage checker passes on a deliberately-broken fixture (proves it works)

## Phase 4 — Ground-truth schema and candidate labels (2–3 h)

- [ ] `datasets/trinetra-hard/schema.json` — row schema per SPEC_TRINETRA_HARD §4
- [ ] `scripts/build_hard_candidates.py` — samples frames into the six slices,
      computes `plate_width_px` and `width_bucket`, writes rows with
      `label_source: "ocr_candidate"`
- [ ] OCR pass produces candidate `plate_text` for each row
- [ ] `index.jsonl` written with **every** row `label_source: ocr_candidate`
- [ ] Per-slice counts reported honestly — under-filled slices flagged, not padded

**Nothing here counts as ground truth.** These are candidates awaiting a human.

## Phase 5 — Verification UI (1 h)

- [ ] `scripts/verify_ui.py` — single-file local page: shows the cropped plate
      region plus the candidate string, keyboard-driven
      (`Enter` accept · type-to-correct · `x` mark ineligible · `?` mark probable)
- [ ] Writes back `label_source: "human"` and `label_confidence`
- [ ] Progress saved continuously — survives a browser close
- [ ] Target throughput: 300 rows in under 2 hours

## Phase 6 — Scorer and harness (2 h)

- [ ] `benchmarks/scorer.py` — primary metric per SPEC_BENCHMARK §2
- [ ] `benchmarks/stub_predictor.py` — canned predictions
- [ ] Six scorer fixtures pass (SPEC_BENCHMARK §5 table)
- [ ] `benchmarks/run.py` — one command, writes the locked report JSON
- [ ] `benchmarks/delta.py` — before/after table by width bucket with raw counts
- [ ] Full loop verified end to end **against the stub**, on candidate rows

**Gate:** if the scorer has not passed all six fixtures, no number it produces
is trustworthy. Do not proceed to real data.

## Phase 7 — **[HUMAN]** Verification pass (Monday AM, ~2 h)

- [ ] Verify all 300 candidate labels through the UI
- [ ] Mark ineligible rows honestly — they measure fabrication
- [ ] Freeze: `MANIFEST.sha256` + `FREEZE.md` with per-slice counts and gaps

## Phase 8 — **[HUMAN]** Measurement (Monday, ~3 h)

- [ ] Fusion OFF run → report
- [ ] Fusion ON run → report
- [ ] `FUSION_DELTA.md` — the headline table
- [ ] Fabrication counts recorded for both runs
- [ ] Every report cites manifest hash, git commit, weights hash

## Phase 9 — **[HUMAN]** Submission (Monday PM)

- [ ] Accuracy claims in the submission match the reports exactly
- [ ] Sub-40 px number stated plainly, not buried
- [ ] Known gaps in `FREEZE.md` acknowledged rather than hidden
- [ ] Tag `v0.1-submission`

---

## Blocking findings to report immediately

| Finding | Consequence |
|---|---|
| Clip identity not recoverable in `indian_road` | fusion measurement invalid |
| A slice cannot reach half its target count | report actual n, do not pad |
| A dataset has no verifiable license | exclude it |
| Scorer fails any of the six fixtures | all downstream numbers void |
| `datasets/raw` junction broken | everything blocked |
