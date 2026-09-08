# dev_assets — for Manas

Generate with:

```bash
py -3.11 scripts/fetch_dev_assets.py
```

Requires `ffmpeg` on PATH and the same `datasets/raw/indian_road` checkout
this lane uses (the script verifies the 5 source tar shards against a
recorded SHA-256 before extracting anything — a hash mismatch means your
checkout differs from the one these clip picks were made against, and the
script stops rather than silently extracting the wrong frames).

**Chose script-over-LFS deliberately, not because LFS failed.** Both `git
lfs` and a GitHub-reachable endpoint are available on this machine. The
script is still the better artifact: these clips are a deterministic
function of 5 tar shards you already need for anything else in this lane,
so a script + SHA-256 proves reproducibility directly, while committing
~80MB of video through git history forever does not — it matches this
repo's own existing convention for run output (`.gitignore`: "the script
that produces it is committed instead").

## What you get (6 assets, all from RESERVED clips — see
`datasets/trinetra-hard/CLIP_RESERVATION.md` — never TRAIN_SAFE)

| File | Source clip_id | Frames | Notes |
|---|---|---|---|
| `clip_daytime_highway.mp4` | `000ebc57-...` | 0-59 | Daytime highway, visible plates, 60s |
| `clip_daytime_city.mp4` | `0031e84e-...` | 0-59 | Daytime city street, visible plates, 60s |
| `clip_daytime_village.mp4` | `003bf0b5-...` | 0-59 | Daytime village road, visible plates, 60s |
| `clip_night.mp4` | `004c8b8d-...` | 0-89 | Night, residential road, 90s |
| `clip_hard_scene_cut.mp4` | `0078bb37-...` | 0-59 | Hard cut at frame 8 (histogram jump 0.748) |
| `frame_sequence_100/` | `0049d23b-...` | 0-99 | 100 plain JPEGs, `frame_0000.jpg`.. `frame_0099.jpg` |

## Read this before testing anything latency- or motion-sensitive

indian_road's locally-downloaded frames are **1 fps keyframes**, per the
dataset's own README ("Keyframe extraction — 1 frame/second via FFmpeg"),
not the original continuous footage. The MP4s here are real, playable,
seekable video files (confirmed via `ffprobe`: correct resolution, frame
count, and duration) — but they are honestly a 1fps slideshow, not smooth
24-30fps motion. There is no higher-framerate source available locally to
build anything closer to real motion video. Use these for decode/ingestion
smoke tests (does `FrameSequenceSource` read the right frame count,
timestamps, resolution) — not for testing temporal smoothness or motion
compensation, which these clips cannot represent honestly.

## License

`indian_road` is CC BY 4.0 (attributed to ThirdEye Labs — see
`datasets/LICENSES.md`). These clips are a direct excerpt, same license,
same attribution requirement.
