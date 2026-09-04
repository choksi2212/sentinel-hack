# DATASETS — inventory, licenses, usage rules

Owner: Akshat. `datasets/LICENSES.md` is the authoritative register; this
document is the reasoning behind it.

**"Free to download" ≠ "safe to reuse."** This is a government-facing
submission. Every asset needs a license row before use. No row → excluded.

---

## 1. Physical layout

```
A:\trinetra hackathon final\datasets\raw\   ← 5.6 GB, read-only
```

Reachable from the repo as `datasets/raw/` via a directory junction:

```bash
cd sentinel-hack/datasets
MSYS_NO_PATHCONV=1 cmd /c mklink /J raw "A:\trinetra hackathon final\datasets\raw"
```

Junction, not copy — the repo stays small and `datasets/raw/` stays gitignored.
All manifest paths are written relative to the repo root
(`datasets/raw/indian_road/...`) so they resolve for anyone with the junction.

## 2. The nine directories

| Folder | Upstream | License | Status | Use |
|---|---|---|---|---|
| `indian_road` | thirdeyelabs/indian-road-dataset | CC BY 4.0 | **verified** | temporal fusion, `FrameSequenceSource`, TRINETRA-HARD |
| `justjuu_plates` | justjuu/license-plate-detection | CC BY 4.0 | **verified** | plate detection train |
| `cctv_accident` | justjuu/traffic-accident-cctv-object-detection | CC0 | **verified** | vehicle detection train |
| `gujarat_plates` | paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready | Apache 2.0 | **verified** | GJ-plate OCR train — regionally on-point |
| `indian_plates_yolo` | deepakat002/indian-vehicle-number-plate-yolo-annotation | CC0 | **verified** | plate detection train, YOLO labels ready |
| `synthetic_plates` | abtexp/synthetic-indian-license-plates | CC0 | **verified** | synthetic corpus (18,000 plates) |
| `traffic_vehicle` | — | **unknown** | ⚠ unverified | blocked until a row exists |
| `kedarsai_plates` | — | **unknown** | ⚠ unverified | blocked until a row exists |
| `fanvid` | FANVID | **eval-only** | ⚠ needs row | evaluation only — never training |

**Three rows are missing.** `traffic_vehicle`, `kedarsai_plates`, and `fanvid`
must each get a complete row or be excluded. Excluding them costs little —
the six verified sets cover every task.

## 3. `LICENSES.md` row format — nine fields

| Field | Example |
|---|---|
| `asset` | `indian_road` |
| `upstream` | `kaggle.com/datasets/thirdeyelabs/indian-road-dataset` |
| `license` | `CC BY 4.0` |
| `license_url` | `https://creativecommons.org/licenses/by/4.0/` |
| `commercial_use` | `yes` |
| `attribution_required` | `yes` |
| `redistribution` | `yes, with attribution` |
| `used_for` | `eval (TRINETRA-HARD), fusion measurement` |
| `verified_on` | `2026-09-05` |

`used_for` is the field that matters at audit time. It is what proves an
eval-only asset never entered training.

## 4. The leakage rule

**`indian_road` splits by clip ID, never by frame.**

Consecutive frames from one clip are near-duplicates. A frame-level split puts
frame 870 in training and frame 871 in test, and the model scores well by
having memorised the exact vehicle. Perceptual hashing does not catch this —
the frames genuinely differ, just not in any way that matters.

`scripts/check_split_leakage.py` must assert: **no `clip_id` appears in more
than one split.** It runs in CI and it runs before every benchmark.

If clip identity is not recoverable from directory structure or filenames,
that is a blocking finding — report it, do not invent a scheme. Falling back
to a filename-prefix heuristic without verifying it is how silent leakage gets
in.

## 5. Non-`indian_road` sources

Single-frame. No temporal continuity, therefore:

- usable for TRINETRA-HARD slices that do not need sequence
- **not** usable for the fusion before/after measurement — fusion needs
  multiple observations of one `TrackKey`

For these, synthesise a stable `camera_id` / `stream_session_id` / `track_id`
per image so the row shape stays uniform. Record in `notes` that the key is
synthetic, so nobody later mistakes it for real multi-frame data.

## 6. What goes in git

| Tracked | Ignored |
|---|---|
| `datasets/LICENSES.md` | `datasets/raw/**` |
| `datasets/manifests/*.sha256` | `datasets/**/images/` |
| `datasets/trinetra-hard/index.jsonl` | `datasets/**/labels/` |
| `datasets/trinetra-hard/FREEZE.md` | `*.zip` `*.tar*` `*.mp4` `*.mkv` |
| `datasets/synthetic/index.jsonl` | model weights, checkpoints |

Indices and hashes are tracked. Pixels never are.
