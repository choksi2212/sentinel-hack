# RUNLOG — Akshat's lane

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
- Next: either get the 3 flagged datasets license-cleared (unlocks
  perspective/tiny variety), or accept 196/300 and rebalance targets in
  FREEZE.md (Phase 7, human).

## 2026-09-05 — Phase 5: Verification UI
**STATUS: OK**

- `scripts/verify_ui.py` — one file, stdlib `http.server` + inline HTML/JS,
  no framework. Enter accepts/corrects, `x` marks ineligible, `?` marks
  probable, arrows navigate. Writes `label_source: human` back to
  `index.jsonl` after every single action (atomic tmp-file replace).
- Smoke-tested end to end: server up, `/api/rows` (196 rows), `/api/image/`
  (real PNG crop, 18,659 bytes), `/api/label/` POST round-tripped correctly.
  Test write reverted afterward — index.jsonl is back to Phase 4's output.
- Not yet run for real (that's Phase 7, human, Monday AM).

## 2026-09-05 — Phase 6: Scorer and harness
**STATUS: OK — gate passed, all six fixtures green**

- `benchmarks/scorer.py`: all 6 SPEC_BENCHMARK §5 fixtures pass (exact match,
  case/space, 0/O fuzzy-not-correct, missing prediction, fabrication on
  ineligible, ocr_candidate fully excluded).
- `stub_predictor.py`, `run.py` (`-m benchmarks.run`), `delta.py`
  (`-m benchmarks.delta`) written and run end-to-end on the real 196-row
  `index.jsonl` — no crash, correctly degrades to `n_eligible: 0` /
  `rate: null` everywhere because 0 rows are `label_source: human` yet
  (expected — that's Phase 7, human). Reports + `FUSION_DELTA.md` committed
  as proof the harness runs clean; re-run for real after Phase 7.
- Every report cites `dataset_manifest_sha256` + `git_commit`;
  `weights_sha256: null` with reason in `notes` (stub has no weights).
- Gate cleared — downstream numbers, once real labels exist, can be trusted.

## 2026-09-05 — Phase 0: Environment
**STATUS: OK**

- Repo toplevel: `A:/trinetra hackathon final/sentinel-hack`, branch `data/trinetra-hard`, tracking `origin/data/trinetra-hard`.
- `datasets/raw` junction resolves, contains 9 dirs: cctv_accident, fanvid, gujarat_plates, indian_plates_yolo, indian_road, justjuu_plates, kedarsai_plates, synthetic_plates, traffic_vehicle.
- `.gitignore` line 30 covers `datasets/raw/`.
- `py -3.11 --version` → Python 3.11.0. OK.
- No blockers. Proceeding to Phase 1.
