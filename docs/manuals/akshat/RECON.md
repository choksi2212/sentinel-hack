# RECON — Phase 1 findings (2026-09-05)

## Clip identity in `indian_road` — RECOVERABLE, not blocking

Files inside each WebDataset tar shard (`data/train-NNNNN-of-00646.tar`) are named
`{clip_id}_{frame:04d}.{jpg,png,json}`, where `clip_id` is a UUID
(e.g. `000db725-8180-4770-8b6a-4eb74aeda9f9`). Regex `^(.+)_(\d{4})\.ext$` matched
100% of entries (0 failures) across shards 0, 1, 4 (3000+ files checked).

Only 5 of 646 shards are present locally (~5.6 GB budget): shards 00000-00004,
5,000 frames, **62 distinct clips** (not the full 8,441). `split_leakage` checks
and TRINETRA-HARD sampling from `indian_road` must draw from this 62-clip pool —
report actual n, do not imply the full corpus is available.

`annotations/detection.json` "name" field uses `clip_id/frame` (slash) per the
dataset README, but actual shard filenames use `clip_id_frame` (underscore).
Use the filename convention — it's what's actually on disk.

## Per-dataset summary

| Dataset | Format | Clip identity | Plate bbox? | Files |
|---|---|---|---|---|
| indian_road | WebDataset tar, BDD100K JSON + PNG seg mask | yes, filename UUID prefix (5/646 shards local) | no — vehicle-level only (car/truck/bus/motorcycle/autorickshaw/...) | 5000 imgs |
| gujarat_plates | YOLO txt, 1 class `plate` | no (standalone images) | yes | 355 img/label pairs |
| indian_plates_yolo | YOLO txt, 1 class `number_plate` | yes — `vid-1/vid-2/vid-3` dirs | yes | 160 img/label pairs |
| kedarsai_plates | YOLO txt, multi-object per file (no classes.txt found — class ids only) | no | yes (plate class present in multi-object labels) | 2021 labels / ~2083 images |
| justjuu_plates | HF parquet, bbox+category `license_plate` | no (train/val/test image splits, no clip field) | yes | 6176/1765/882 |
| synthetic_plates | PNG crops only, no bbox (image == plate) | no | n/a (pre-cropped) | 18,000 |
| traffic_vehicle | Roboflow YOLO txt, 4 classes (car/bus/truck/van) | no | no — vehicle-only | 2,815 img (train+valid+test) |
| cctv_accident | HF parquet, bbox+category (accident/non_accident) | no | no — not plate/vehicle-detection relevant | 2,763 |
| fanvid | CSV, `Clip ID`,`VideoID`,`FrameNo`,bbox, identity/text | yes — `Clip ID`+`VideoID` | no — face/celebrity identity, not plates | rows across 8 CSVs |

## License flags (for Phase 2)

- `traffic_vehicle`: Roboflow re-export of UA-DETRAC sample — README claims CC BY 4.0
  but original UA-DETRAC license terms need independent verification before trusting
  the Roboflow claim.
- `kedarsai_plates`: no LICENSE/README found in the tree — **flagged, not guessed**.
- `fanvid`: real celebrity likenesses/identities — flagged, and per TASKS.md must be
  marked `used_for: eval-only` regardless of license terms.
