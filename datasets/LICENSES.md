# LICENSES

One row per dataset actually used. No row → excluded (CLAUDE.md §5). 9 fields:
`name, source, license, license_url_or_evidence, commercial_use, attribution_required, used_for, verified_by, verified_on`.

## Verified (license evidence found embedded in the asset itself)

| name | source | license | evidence | commercial_use | attribution_required | used_for | verified_by | verified_on |
|---|---|---|---|---|---|---|---|---|
| indian_road | HF `thirdeyelabs/indian-road-dataset` | CC BY 4.0 | `datasets/raw/indian_road/README.md` front-matter `license: cc-by-4.0` + body license section, attributed to ThirdEye Labs | yes | yes (ThirdEye Labs) | train+eval (vehicle detection, fusion source video) | agent recon | 2026-09-05 |
| justjuu_plates | HF dataset card | CC BY 4.0 | `datasets/raw/justjuu_plates/README.md` front-matter `license: cc-by-4.0` | yes | yes | train+eval (plate bbox) | agent recon | 2026-09-05 |
| cctv_accident | HF dataset card | CC0 1.0 | `datasets/raw/cctv_accident/README.md` `license: cc0-1.0` + body "Creative Commons Zero v1.0 Universal" | yes | no | **unused** — accident/non_accident classification, out of scope for plate/vehicle detection task; kept only for provenance record | agent recon | 2026-09-05 |
| synthetic_plates | self-generated (this lane, `scripts/synth/`) | owned — no external rights holder | folder structure (18,000 PNGs under `generated/<Indian state>/`) is consistent with synthetic plate generation, not a scraped/photographed source; no camera EXIF, no external attribution present | yes | no | train+eval (synthetic plate crops) | agent recon | 2026-09-05 |
| gujarat_plates | Kaggle `paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready` | Apache 2.0 | operator-verified from the Kaggle dataset page (Kaggle licenses are not embedded in the download, they live on the page) | yes | no | train+eval (plate bbox) | operator (2026-09-05 correction) | 2026-09-05 |
| indian_plates_yolo | Kaggle `deepakat002/indian-vehicle-number-plate-yolo-annotation` | CC0 | operator-verified from the Kaggle dataset page | yes | no | train+eval (plate bbox, video-clip source: `vid-1/vid-2/vid-3`) | operator (2026-09-05 correction) | 2026-09-05 |

## Flagged — NOT used pending human verification (CLAUDE.md §5: no row → excluded)

| name | why flagged |
|---|---|
| traffic_vehicle | Roboflow re-export (`README.dataset.txt`) claims "License: CC BY 4.0" for a UA-DETRAC-derived sample, but the original UA-DETRAC license terms are more restrictive (research-only) and the Roboflow claim is not independently verified here. Per TASKS.md, flagged not guessed. |
| kedarsai_plates | No LICENSE, README, or dataset card found anywhere in the tree — zero embedded provenance. Per TASKS.md, flagged not guessed. |
| fanvid | Dataset card states `license: mit`, but the CSVs contain real, named celebrities' identities (`IdentityOrText` column, e.g. "Chiwetel Ejiofor") and face/plate crops from their public videos — a likeness/publicity-rights concern independent of the stated code/data license. Per TASKS.md, marked **eval-only** if ever used, and excluded from LICENSES.md rows pending human sign-off. |

**Resolved 2026-09-05:** `gujarat_plates` and `indian_plates_yolo` are now verified above — the operator confirmed both from their Kaggle dataset pages (Kaggle license terms live on the page, not in the downloaded archive, so the earlier "zero embedded provenance" flag was a false negative from an agent that only checks the archive itself). 6 rows verified, matching TASKS.md's original expectation.
