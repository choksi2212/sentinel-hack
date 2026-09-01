# TRINETRA — EXECUTION MANUAL
## Manas / Niklaus — AI Lead & System Architect

**Version 2.0 · 2026-09-01 · 6 build days to the qualification gate**

> **Precedence.** [`docs/TRINETRA_Canonical_Contracts.md`](docs/TRINETRA_Canonical_Contracts.md) is normative for every schema. Contract blocks reproduced below are marked `COPIED FROM CANONICAL — DO NOT EDIT HERE`. If you need a contract changed, change it there and tell the other three.

---

## 1. Your job in one paragraph

You own the boundary between messy reality and clean data. Everything upstream of you is unreliable — streams drop, PTS lies, plates are 40 pixels wide, OCR guesses. Everything downstream of you assumes structure. Your deliverable is a single `EventEnvelope` that Mihir can persist without thinking and Parth can render without caveats, plus the honest measurement that says how often it is right. You also own the architecture decisions that keep the other three unblocked.

**You own:** `ai/contracts/` · `ai/media/` · `ai/detect/` · `ai/track/` · `ai/plate/` · `ai/ocr/` · `ai/fusion/` · `ai/quality/` · `ai/worker.py` · the A100 decision · the architecture walkthrough.

**You do not own:** the database, the API, the UI, the datasets. When you need something from those, you file a contract change, not a patch.

**Your machine:** Ryzen 9 9955HX · RTX 5070 Ti 12 GB · 32 GB RAM. The 12 GB is the real constraint — design for it, not for the A100.

---

## 2. Day plan — anchored to the real calendar

Today is **1 September 2026**. Submission is **7 September**. Hackathon is **10–11 September**. You have **six build days**. Any 10-day or 14-day plan you have seen in an earlier document is superseded by this table.

| Date | Day | You must finish | Proof |
|---|---|---|---|
| **Sep 1** | D1 | `FrameEnvelope`, `EventEnvelope`, `MediaSource` protocol, `VideoFileSource`, `FrameSequenceSource`, repo skeleton, contracts committed and announced | Contracts merged; Parth and Mihir unblocked; a clip produces `FrameEnvelope`s with monotonic PTS |
| **Sep 2** | D2 | RF-DETR vehicle detection + ByteTrack + session/discontinuity handling + `SentinelRTSPSource` | Forced reconnect yields **two** sessions and two track sets, not one |
| **Sep 3** | D3 | Plate detection baseline + PaddleOCR + quality scoring + top-K crop selection + **failure taxonomy** | Plate strings from a real clip, accuracy broken out by width bucket |
| **Sep 4** | D4 | Temporal fusion + normalization + validation + `EventEnvelope` emission + POST to Mihir | First real event persisted end to end (**G2 + G3**) |
| **Sep 5** | D5 | Integration hardening, fault injection, `SyntheticReplaySource`, help Parth and Mihir close gaps | Full pipeline survives §8 fault list |
| **Sep 6** | D6 | Benchmark freeze, A100 go/no-go, architecture walkthrough rehearsed | Reports written; walkthrough under 3 min |
| Sep 7 | — | **SUBMIT** | — |
| Sep 8 | D7 | Live swap attempt; A100 training if gated GO | Config-only swap proven, or documented reason live is unavailable |
| Sep 9 | D8 | Final freeze, rehearsal | §K7 of the technical plan |

**D1 is the highest-leverage day of the project.** Two people are blocked until your contracts land. Write them first, before any model code.

---

## 3. Contracts — write these first, on D1

### 3.1 FrameEnvelope

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §2)

```python
# ai/contracts/frame.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

SourceMode = Literal["live_rtsp", "live_hls", "file", "frames", "synthetic"]

@dataclass(frozen=True)
class FrameEnvelope:
    camera_id: str                # "cam04" — Sentinel ID, verbatim lowercase
    stream_session_id: str        # UUID str, minted at connect
    frame_index: int              # monotonic within session, starts at 0
    pts_ms: int                   # SOURCE timeline position, milliseconds
    wallclock_utc: Optional[str]  # ISO-8601 Z; None for pure file replay
    frame_bgr: np.ndarray         # HxWx3 uint8, BGR
    width: int
    height: int
    source_mode: SourceMode
```

### 3.2 EventEnvelope v1.1

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §3)

```json
{
  "schema_version": "1.1",
  "event_id": "9f2c1d84-7b3e-4a51-9c02-6d8e1f4a7b90",
  "camera_id": "cam04",
  "stream_session_id": "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05",
  "track_id": 42,
  "observed_at": "2026-09-01T10:03:21.234Z",
  "source_pts_ms": 343100,
  "source_mode": "file",
  "vehicle": {"type": "car", "confidence": 0.93, "bbox_xyxy": [812, 344, 1102, 587]},
  "plate": {
    "raw": "GJ 01 AB 1234",
    "normalized": "GJ01AB1234",
    "confidence": 0.94,
    "match_state": "probable",
    "plate_width_px": 62,
    "evidence_count": 3,
    "bbox_xyxy": [901, 502, 963, 523]
  },
  "image_quality": 0.88,
  "evidence": {
    "snapshot_uri": "local://artifacts/snapshots/cam04_3a7f1e02_42_343100_9f2c1d84.jpg",
    "plate_crop_uri": "local://artifacts/snapshots/cam04_3a7f1e02_42_343100_9f2c1d84_plate.jpg"
  },
  "model": {
    "detector": "rfdetr-small",
    "detector_weights_sha256": "a1b2c3…",
    "plate_detector": "rtdetrv2-justjuu",
    "ocr": "paddleocr-3.0",
    "tracker": "bytetrack",
    "pipeline_version": "0.1.0"
  }
}
```

**Four things that changed from the version you may have in your head:**

| Was | Is now | Why |
|---|---|---|
| `"timestamp"` | `"observed_at"` | There are two clocks; one name for both was the bug |
| `"camera_id": "CAM_001"` | `"camera_id": "cam04"` | Sentinel publishes `cam01`–`cam30`; an invented format forces a rewrite of every fixture on live day |
| `"schema_version": "1.0"` | `"1.1"` | Breaking change |
| no session, no PTS | `stream_session_id`, `source_pts_ms` | Without these, tracks merge across reconnects |

Also added and **required**: `plate.plate_width_px` (Akshat cannot produce width-bucket reports without it), `plate.evidence_count` (Parth renders it next to confidence), `source_mode` (drives the LIVE/REPLAY badge), and the whole `model` block (provenance — without it no benchmark number is citeable).

### 3.3 Internal stage contracts

These are yours alone; nobody else consumes them. But **every one carries the session**, because a stage contract without a session is where the merge bug gets reintroduced.

```python
# ai/contracts/stages.py
@dataclass(frozen=True)
class DetectorResult:
    bbox_xyxy: tuple[int, int, int, int]
    class_name: str            # car|motorcycle|bus|truck|auto_rickshaw|other
    confidence: float

@dataclass(frozen=True)
class TrackResult:
    camera_id: str
    stream_session_id: str     # ← never omit
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    class_name: str
    confidence: float
    frame_index: int
    pts_ms: int

@dataclass(frozen=True)
class PlateObservation:
    camera_id: str
    stream_session_id: str     # ← never omit
    track_id: int
    plate_bbox_xyxy: tuple[int, int, int, int]
    plate_width_px: int
    plate_raw: str
    ocr_confidence: float
    image_quality: float
    frame_index: int
    pts_ms: int
    observed_at: str
```

### 3.4 The TrackKey rule — internalize this

```
TrackKey = (camera_id, stream_session_id, track_id)
```

ByteTrack restarts numbering at 1 on every new session. A dict keyed `(camera_id, track_id)` will merge the car that left cam04 before a reconnect with a different car that entered after it. The result is a journey showing a vehicle crossing Ahmedabad in four seconds. Nobody notices for two days, and then every number you have produced is suspect.

Mint a new session on: initial connect · reconnect after failure · hard scene discontinuity · replay restart. On new session, **flush** tracker state, evidence buffers, and every in-flight fusion accumulator.

---

## 4. Media layer — D1 and D2

### 4.1 The protocol

```python
# ai/media/base.py
from typing import Protocol, Optional

class MediaSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[FrameEnvelope]: ...   # None = end of stream
    def close(self) -> None: ...
    @property
    def session_id(self) -> str: ...
```

Five implementations, all in `ai/media/`, and **nothing else in the codebase may open a video**:

| Class | `source_mode` | Build on |
|---|---|---|
| `VideoFileSource` | `file` | **D1 — first** |
| `FrameSequenceSource` | `frames` | D1 |
| `SentinelRTSPSource` | `live_rtsp` | D2 |
| `SentinelHLSSource` | `live_hls` | D2 (thin; browser path is Mihir's proxy) |
| `SyntheticReplaySource` | `synthetic` | D5 |

Build `VideoFileSource` **before** the RTSP adapter. Not as a fallback — as the primary development path. Sentinel is live-only with no seeking, which means you cannot iterate on the same footage twice. You will spend most of the six days running the same clip over and over, and that is the correct way to work.

### 4.2 Timing — the rule you will be tempted to break

```python
pts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))     # CORRECT
```

Forbidden: `CAP_PROP_FPS` (lies on RTSP — frequently reports 90000 or 0), `frame_index / fps` (drifts), `time.time()` at arrival (measures your network, not the video).

Keep **both** clocks, always:
- `pts_ms` / `source_pts_ms` → where in the video
- `wallclock_utc` / `observed_at` → when the system saw it

They diverge during replay, during `--mode fast`, and during network stalls. Reporting one as the other is the easiest possible way to make a false claim, and it is the kind a judge will catch.

### 4.3 Sampling and buffering

```python
TARGET_INTERVAL_MS = 100                     # ~10 inferences/sec/camera
if pts_ms - last_emitted_pts_ms >= TARGET_INTERVAL_MS:
    emit(envelope)
    last_emitted_pts_ms = pts_ms
```

Buffer depth **1**. Latest frame wins; overwrite the old one; count the drop.

Freshness beats completeness. An operator needs the vehicle passing now, not a perfect analysis of the one that passed 40 seconds ago. An unbounded queue turns a 200 ms deficit into 2 s of lag per second of runtime — five minutes in, your "live" view is ten minutes stale, and shortly after that the process is OOM-killed. On demo day.

### 4.4 Worker loop

```
connect
  → mint stream_session_id (uuid4)
  → POST/INSERT stream_sessions row
  → loop:
        read + decode
        build FrameEnvelope
        sample by PTS
        latest-frame buffer (depth 1, overwrite, count drops)
        hand to pipeline
  → on failure:
        close capture
        mark session ended (end_reason)
        flush tracker + evidence + fusion state
        sleep(backoff + jitter)
        mint a NEW session, reconnect
```

**Never** write `while True: cap = cv2.VideoCapture(url)`. No backoff means you hammer the Sentinel server; no session reset means tracks merge across the gap; no accounting means you cannot distinguish a flaky camera from a flaky network.

### 4.5 Reconnect with jitter

```python
BACKOFF_BASE_MS, BACKOFF_MAX_MS = 500, 30_000
delay = min(BACKOFF_BASE_MS * (2 ** attempt), BACKOFF_MAX_MS)
delay *= (0.5 + random.random())          # 50%–150% jitter
```

Jitter is not decoration. Thirty workers losing a shared upstream will otherwise retry at identical instants and convert a brief outage into a self-inflicted DoS against the grid. Reset `attempt` after 30 s of sustained healthy reads, not on the first good frame.

### 4.6 RTSP specifics

```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cap = cv2.VideoCapture(f"{RTSP_BASE}/{camera_id}", cv2.CAP_FFMPEG)
```

TCP is mandatory for the AI path. UDP drops packets under load and produces corrupt macroblocks that read as detector failures — you will spend an afternoon debugging a model that is fine.

Manual check before writing any code:

```bash
ffplay -rtsp_transport tcp rtsp://103.250.160.189:8554/stream/cam04
```

### 4.7 PTS validation

| Condition | Action |
|---|---|
| `pts_ms` decreases | Discontinuity → new session |
| Jump forward > 5,000 ms | Discontinuity → new session |
| 0 / unavailable for N frames | Synthetic monotonic clock, **log warning, mark session `pts_unreliable`** |
| Identical across frames | Stalled decoder → force reconnect |

A `pts_unreliable` session must never substantiate a latency claim.

### 4.8 Scene discontinuity — the Sentinel-specific trap

Sentinel streams **loop with hard cuts**. A loop boundary is visually indistinguishable from someone physically re-aiming the camera: every tracked vehicle vanishes, unrelated ones appear instantly.

Two cheap independent signals:

1. PTS discontinuity (§4.7)
2. **Global histogram delta** — downscaled grayscale histogram correlation between consecutive *emitted* frames. A correlation collapse across a single 100 ms step is not physically possible for a fixed camera.

On detection: end session (`end_reason='discontinuity'`), flush everything, mint a new session. Skip this and ByteTrack will cheerfully associate a car leaving at the loop end with a different car entering at the loop start.

### 4.9 Adapter acceptance tests — write these on D1

| Test | Pass condition |
|---|---|
| `VideoFileSource`, 60 s clip | ~600 emitted frames, `pts_ms` monotonic 0→60000 |
| `FrameSequenceSource`, 100 JPEGs | 100 frames, `frame_index` 0→99 |
| `SyntheticReplaySource`, seed 42, twice | **Byte-identical** event stream |
| RTSP killed mid-stream | New session, backoff observed, tracker flushed |
| **All five adapters** | Downstream path byte-identical; only `source_mode` differs |

The last row is the invariant test. Put it in CI on D1 and it will catch the day someone writes `if source == "live"` in the wrong place.

---

## 5. CV pipeline — D2 and D3

### 5.1 Stage order — do not reorder

```
FRAME → VEHICLE DETECT → TRACK → VEHICLE QUALITY GATE
      → PLATE DETECT → PLATE QUALITY RANK → OCR
      → TEMPORAL FUSION → NORMALIZE → VALIDATE → DEDUP → EVENT
```

Two choices worth understanding rather than just following:

**Plate detection is its own model.** A plate is ~40×15 px in a 1920×1080 frame. A detector trained on cars will not reliably find it. Adding "plate" as a class to the vehicle detector is the most common shortcut in ALPR projects and it caps your accuracy before you start.

**Fusion sits between OCR and normalization.** Single-frame OCR on a 60 px plate is close to a coin flip; three frames voting is a decision. This is the highest accuracy-per-effort stage in the entire pipeline and it requires zero training. If you build only one thing well, build this.

### 5.2 Vehicle detection — RF-DETR

**RF-DETR is Roboflow's, Apache-2.0**, for core code and **Nano through Large** weights. Plus / XL / 2XL carry different licensing and are **excluded**.

It is **not** an Ultralytics model. If you see it attributed to Ultralytics anywhere, that is wrong and it matters: Ultralytics YOLO is AGPL-3.0, and AGPL is precisely the exposure this stack exists to avoid. Ultralytics may appear in a benchmark comparison table. It may not appear in the shipped pipeline.

```python
with torch.inference_mode():
    detections = detector(frame_tensor)
```

Always `inference_mode()`, never bare `eval()` with gradients live — on a 12 GB card the difference decides whether you OOM.

Classes: `car`, `motorcycle`, `bus`, `truck`, `auto_rickshaw`, `other`.

**Check auto-rickshaw coverage before accepting any pretrained baseline.** COCO-derived class maps usually lack it, and in Gujarat that is not a rounding error.

Fallbacks if RF-DETR underperforms on our hardware: **RTMDet, YOLOX** — both Apache-2.0. These are legitimate options; do not treat RF-DETR as the only permitted detector.

### 5.3 Tracking — ByteTrack

MIT. Produces `track_id`, local to `(camera_id, stream_session_id)`.

Flush completely on session change. No cross-camera association — that is Re-ID and it is deferred to the final stage. Resist it; it will eat two days and produce nothing demonstrable.

### 5.4 Vehicle quality gate — where the GPU budget is won

Reject cheaply before spending plate-detection compute:

| Reject when | Because |
|---|---|
| Vehicle bbox height < 60 px | Plate cannot exceed ~20 px; OCR is hopeless |
| Bbox touches frame edge on ≥2 sides | Plate likely cut off |
| Bbox shrinking and already small | Vehicle departing; better frames already captured |
| Detector confidence < 0.35 | Probably not a vehicle |

### 5.5 Plate detection

Start with `justjuu/rtdetr-v2-license-plate-detection` (Apache-2.0). Its published 0.97 mAP / 0.88 small-object numbers are **on its own test set**. They are a starting point, not our result. Ours comes from TRINETRA-HARD, and it will be lower.

Run on the **vehicle crop**, not the full frame. This raises the plate's effective resolution and eliminates false positives on road signage and shop hoardings.

### 5.6 Plate quality score

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §4.1)

```python
score = (0.30 * sharpness_norm        # variance of Laplacian, normalized
       + 0.25 * resolution_norm       # plate width vs 100 px reference
       + 0.25 * detector_confidence
       + 0.20 * exposure_norm)        # penalize clipped histograms
```

Keep top **K = 3…5** crops per track. This same score is the fusion weight in §5.8 — one function, two uses, so it must be stable.

### 5.7 OCR — PaddleOCR

Apache-2.0. Preprocessing variants, applied **one at a time** and measured individually:

`raw` · `grayscale` · `contrast stretch` · `adaptive threshold` · `2× upscale` · `mild sharpening`

Stacking these blindly makes the pipeline slower *and* less accurate simultaneously. Measure each, keep what pays.

**Super-resolution is not a default.** It is a plausible-looking trap: it hallucinates characters, turning `GJ01AB1234` into `GJ01A81234` — and does so with *higher* apparent confidence than the blurry truth. If you adopt it at all, justify it on TRINETRA-HARD with a width-bucket breakdown proving it does not increase confident-wrong readings. Confident-wrong is worse than unreadable in a police system.

### 5.8 Temporal fusion — your most important 20 lines

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §4.3)

```python
from collections import defaultdict

def fuse(observations):
    """observations for ONE TrackKey"""
    score, count = defaultdict(float), defaultdict(int)
    for o in observations:
        key = normalize_plate(o["text"])
        if not key:
            continue
        score[key] += o["ocr_confidence"] * o["image_quality"]
        count[key] += 1
    if not score:
        return None
    best = max(score, key=score.get)
    return {
        "normalized": best,
        "confidence": score[best] / sum(score.values()),
        "evidence_count": count[best],
    }
```

The worked example to memorize, because it is also your walkthrough slide:

| Frame | OCR | conf | quality | weight |
|---|---|---|---|---|
| 1 | `GJ01AB1234` | 0.91 | 0.90 | 0.819 |
| 2 | `GJ01AB1234` | 0.94 | 0.92 | 0.865 |
| 3 | `GJ01AB1234` | 0.88 | 0.87 | 0.766 |
| 4 | `GJ01A81234` | 0.63 | 0.55 | 0.347 |

`GJ01AB1234` = 2.450 across 3 frames vs `GJ01A81234` = 0.347 across 1 → **GJ01AB1234, evidence_count 3, confidence 0.876**. A single-frame system that happened to sample frame 4 would have been confidently wrong.

Ask Akshat for the **before-vs-after-consensus** benchmark. That delta is the strongest technical claim you can make, and it costs no GPU.

### 5.9 Normalization and validation

```python
def normalize_plate(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())
```

Store **both** `raw` and `normalized`. `raw` is the audit trail; `normalized` is the search key. Never normalize in place.

Indian grammar check — **soft**:

```
^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$
```

A miss **downgrades confidence**. It never rewrites the string and never drops the observation. BH-series and older formats exist; a hard filter silently deletes real plates.

### 5.10 match_state

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §3.3)

```python
def match_state(evidence_count, fused_confidence, exact_watchlist_hit):
    if exact_watchlist_hit:                              return "exact"
    if evidence_count >= 2 and fused_confidence >= 0.80: return "probable"
    return "low_confidence"
```

Four values: `exact` · `probable` · `low_confidence` · `unreadable`. Mihir's `vehicle_sightings.match_state` column has a `CHECK` on exactly these — emit anything else and ingest returns 422.

### 5.11 plate: null is a correct answer

An unreadable plate is **information**: a vehicle passed and could not be identified. Emit `plate: null`, set `match_state: "unreadable"`, and move on. Fabricating a guess to avoid a null is the worst failure mode available to this system — and unlike a null, it is undetectable downstream.

---

## 6. Failure taxonomy — D3 deliverable, gates the A100

After the baseline benchmark, classify **every** miss into exactly one bucket:

| Bucket | Symptom | What would actually help |
|---|---|---|
| `vehicle_miss` | No vehicle detected | Vehicle detector training / Indian road data |
| `plate_miss` | Vehicle found, plate not | Plate detector training on small plates |
| `plate_too_small` | Plate < 30 px | Nothing in software — camera placement, or accept it |
| `ocr_wrong` | Plate found, text wrong | OCR training / synthetic corpus |
| `ocr_partial` | Some characters correct | Temporal consensus / more frames |
| `fusion_wrong` | Best single frame right, consensus wrong | Fusion weighting |
| `track_broken` | One vehicle split across tracks | Tracker tuning |
| `track_merged` | Two vehicles in one track | Discontinuity / session handling |
| `duplicate` | One vehicle, several sightings | Dedup window |
| `dropped_frame` | Vehicle never sampled | Sampling interval / throughput |

**One dominant bucket** is the precondition for training anything. Two co-dominant buckets means the analysis is not finished, and training would be a guess with a credit card attached.

Note that `plate_too_small` is a legitimate finding with **no** software fix. If that is your dominant bucket, the honest deliverable is a width-bucket report and a camera-placement recommendation — not a training run.

---

## 7. Benchmark and the A100 decision — D6

### 7.1 Primary metric

```
E2E correct-plate event rate = correct final plate events / eligible vehicle events
```

*Eligible* = plate human-readable in ≥1 sampled frame of ground truth. *Correct* = fused `normalized` exactly equals ground truth.

mAP, CER, FPS, p95, VRAM are **diagnostics**. They explain the primary number; they never replace it. A plate detector at 0.97 mAP is entirely compatible with a system that reads almost no plates, because detection is one of nine stages and stages multiply.

### 7.2 Always report by width bucket

`>100` · `80–100` · `60–80` · `40–60` · `30–40` · `<30` px.

A single average is not an acceptable deliverable. "92%" typically decomposes into 98% above 80 px and 51% below 40 px, and the second number is the one that decides whether this works on real infrastructure. Publish the breakdown yourself — it is strictly better than having a judge extract it from you.

### 7.3 Accuracy vs latency

| Situation | Decision |
|---|---|
| Higher accuracy, similar latency | Take the accuracy |
| Small accuracy gain, large latency cost | Take the latency |
| Large accuracy gain, modest latency cost | Take the accuracy |
| No measurable gain | Do not adopt |
| Improves only the synthetic test | Not a win |

Worked case: mAP 96% @ 12 FPS vs mAP 94% @ 40 FPS → **take 94% @ 40 FPS.** Multi-camera is throughput-bound. This is a product decision, not a modelling one.

### 7.4 GPU memory guard — 12 GB is the design target

The A100 is 80 GB and you will have it for hours, not days. Code that only runs there is useless for six of eight days.

- Log `torch.cuda.memory_allocated()` and `memory_reserved()` at each stage
- Model size selectable by config: Nano / Small / Medium
- **One active checkpoint in VRAM at a time**
- `del` + `torch.cuda.empty_cache()` at stage boundaries
- Always `torch.inference_mode()`

### 7.5 Warm-up

Discard the **first 2** predictions from every timing measurement. CUDA autotuning and lazy allocation make them 5–20× slower; including them makes every latency number fiction.

### 7.6 The 11-row rental gate

| # | Check | Owner |
|---|---|---|
| 1 | Offline pipeline runs end to end | You |
| 2 | Baseline benchmark recorded | Akshat |
| 3 | TRINETRA-HARD frozen | Akshat |
| 4 | Failure taxonomy written | **You** |
| 5 | Bottleneck is **one** component | **You** |
| 6 | Local smoke train completes | **You** |
| 7 | Dataset manifest frozen (SHA-256) | Akshat |
| 8 | Split leakage check green | Akshat |
| 9 | Licences recorded for every training asset | Akshat |
| 10 | `config/training.yaml` committed | **You** |
| 11 | Eval runs unattended, one command | Akshat |

Any red row → **no rental**. Renting before knowing which component fails is the most expensive mistake available to you.

### 7.7 Cost

$1.64/hr running ($1.59 GPU + $0.007 container + $0.042 volume) · $0.083/hr stopped · ~10 h ≈ **$16.40** · ~25 h ≈ **$41**.

Prepare locally → upload frozen data → run → download weights → **stop the pod**. One forgotten overnight pod costs more than the entire planned budget.

### 7.8 The ratchet

```
BASELINE → TARGET FAILURE ANALYSIS → LOCAL SMOKE TRAIN → FROZEN DATA/CONFIG
        → A100 TRAIN → FROZEN HARD TEST → KEEP ONLY IF END-TO-END VALUE IMPROVES
```

That last clause is the whole policy. +3 mAP and +0 end-to-end correct-plate rate means the model is **discarded**, whatever it cost. Say this out loud before you rent anything.

### 7.9 It is fine not to train

If the pretrained stack plus temporal fusion clears the bar, ship it with an honest benchmark. "We measured, found the bottleneck was plate size rather than model capacity, and did not spend money we could not justify" is a *stronger* answer to a judge than an unmotivated fine-tune. Do not train to look serious.

---

## 8. Fault injection — rehearse on D5

| Fault | Required behaviour |
|---|---|
| Kill RTSP mid-stream | Backoff, new session, tracker flush, camera → `degraded` |
| Kill Redis | Your POSTs still succeed |
| Kill Postgres | `DEPENDENCY_UNAVAILABLE`, you retry, **no data loss** |
| Replay a duplicate event | `{"status":"duplicate"}` |
| Naive timestamp | `422 VALIDATION_FAILED`, field `observed_at` |
| `camera_id: "cam99"` | `422 UNKNOWN_CAMERA` |
| Black / corrupt frame | No crash, **no fabricated plate** |
| Scene cut | New session, no cross-boundary merge |
| Unplug network | Offline mode continues |

Retry policy for POSTs: exponential backoff, bounded queue on disk, `event_id` stable across retries so Mihir's idempotency works. Never drop an event because the backend blinked.

---

## 9. Integration — what you owe the other three

| To | What | When |
|---|---|---|
| **Mihir** | `EventEnvelope` v1.1 spec + 12 fixtures | **D1** |
| **Mihir** | First real POST from the pipeline | D4 |
| **Parth** | Confirmation that `camera_id` is `cam04`-style; `source_mode` semantics | **D1** |
| **Akshat** | Which failure bucket dominates; what data would move it | D3 |
| **Akshat** | Which metrics to report and in which buckets | D3 |
| **All** | Architecture walkthrough | D6 |

Fixtures on D1 are non-negotiable. Parth builds against mocks and Mihir builds against fixtures — if they arrive on D3, two people lose two days and the schedule has no slack to absorb it.

---

## 10. Architecture walkthrough — you present this

Under three minutes, no slides.

1. Problem: 80,000 cameras, 26 departments, no unified vehicle intelligence
2. Insight: **you cannot centralize the video — centralize the intelligence**
3. Sources: RTSP, HLS, MP4, frames, synthetic
4. One `FrameEnvelope` — nothing downstream knows which
5. PTS-driven sampling, latest-frame buffer
6. Detection → tracking → `TrackKey`
7. Quality gate → dedicated plate detection
8. Top-K ranking → OCR
9. **Temporal consensus** — the four-frame table from §5.8
10. Normalize → validate → dedup
11. One `EventEnvelope`, eight provenance fields
12. Persist before publish → search, journey, alert
13. Honest confidence: evidence count, not probability
14. Resilience: unplug the network, the demo keeps running

Steps 9 and 14 are what separate this from every other ALPR submission. Land them.

---

## 11. Honesty rules — you are the technical authority, so you enforce these

| Never say | Say |
|---|---|
| "the exact path the vehicle took" | **"observed movement sequence"** |
| "predicts where the vehicle will go" | **"Camera Search Prioritization"** |
| "87% probability" | "confidence 0.87 from 3 agreeing observations" |
| "40 FPS live" *(measured on 5× replay)* | "40 FPS on replay at 5× acceleration" |
| "confirmed match" *(from fuzzy distance)* | "candidate match, requires review" |
| replay shown as "ONLINE" | LIVE / REPLAY badge |

Do not multiply uncalibrated confidences and present the product — the arithmetic is meaningless on uncalibrated scores and someone will ask.

Fuzzy matching generates **candidates only**. It never rewrites `normalized`, never produces `match_state: "exact"`, never raises a confirmed alert on its own. Common Indian confusion pairs to weight: `0↔O`, `1↔I↔L`, `8↔B`, `5↔S`, `2↔Z`, `6↔G`.

---

## 12. Anti-patterns — each of these has cost a real project

| Do not | Consequence |
|---|---|
| `while True: cap = cv2.VideoCapture(url)` | Hammers the grid, no session reset |
| Key on `(camera_id, track_id)` | Merged vehicles, fabricated journeys |
| `CAP_PROP_FPS` / arrival time as video time | Every temporal claim becomes wrong |
| Unbounded queue | Latency drift → OOM mid-demo |
| OCR on every frame | Burns the budget multi-camera needs |
| Super-resolution by default | Confident-wrong plates |
| Multiply uncalibrated confidences | An indefensible number |
| Hardcode `cam01…cam30` | Breaks when the catalogue changes |
| Live feed as the only dev path | One outage stops the project |
| Cross-camera Re-ID this week | Two days gone, nothing demonstrable |
| Rent the A100 before §7.6 is green | Money for no measured gain |
| Ship Ultralytics YOLO | AGPL-3.0 exposure |
| Attribute RF-DETR to Ultralytics | Reintroduces the licence problem you avoided |

---

## 13. Definition of done — yours specifically

- [ ] Five media adapters, all satisfying `MediaSource`, all producing identical downstream behaviour
- [ ] Contracts committed on D1 and announced to all three
- [ ] Twelve fixtures delivered on D1
- [ ] PTS-driven sampling with depth-1 buffer and drop accounting
- [ ] Reconnect with exponential backoff + jitter; new session; state flushed
- [ ] Scene discontinuity detection working on a looping stream
- [ ] Detection → tracking with correct `TrackKey` semantics, proven by `camera_reconnect.json` yielding **two** tracks
- [ ] Vehicle quality gate cutting wasted plate-detection work
- [ ] Plate detection on vehicle crops
- [ ] Top-K crop selection with the locked quality score
- [ ] OCR with per-variant measurements
- [ ] Temporal fusion with a before/after benchmark delta
- [ ] Normalization + soft grammar check + feasibility flagging
- [ ] `EventEnvelope` v1.1 emitted, POSTed, retried idempotently
- [ ] `plate: null` handled correctly end to end
- [ ] Failure taxonomy written with one dominant bucket named
- [ ] Benchmark reports with width buckets and full provenance
- [ ] A100 go/no-go decided against all 11 rows
- [ ] Fault list in §8 rehearsed
- [ ] Architecture walkthrough under 3 minutes
- [ ] Live swap proven to be configuration-only, or the reason live is unavailable documented

---

## 14. Your daily loop

```bash
docker compose up -d
alembic upgrade head
curl http://localhost:8000/health/ready

python -m ai.worker --config config/offline.yaml --camera cam04
curl "http://localhost:8000/api/v1/search/vehicles?plate=GJ01AB1234"
pytest -q
```

If that chain is not green, fix it before writing anything new. Six days has no slack for a broken baseline.

**Final principle:** *We don't centralize every video. We centralize intelligence.* Every decision you make should be defensible by that sentence.
