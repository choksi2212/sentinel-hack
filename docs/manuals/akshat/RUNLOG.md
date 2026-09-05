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

## 2026-09-05 — Phase 0: Environment
**STATUS: OK**

- Repo toplevel: `A:/trinetra hackathon final/sentinel-hack`, branch `data/trinetra-hard`, tracking `origin/data/trinetra-hard`.
- `datasets/raw` junction resolves, contains 9 dirs: cctv_accident, fanvid, gujarat_plates, indian_plates_yolo, indian_road, justjuu_plates, kedarsai_plates, synthetic_plates, traffic_vehicle.
- `.gitignore` line 30 covers `datasets/raw/`.
- `py -3.11 --version` → Python 3.11.0. OK.
- No blockers. Proceeding to Phase 1.
