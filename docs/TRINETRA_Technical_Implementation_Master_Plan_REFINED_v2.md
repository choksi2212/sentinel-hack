# TRINETRA — TECHNICAL IMPLEMENTATION MASTER PLAN

**Initial Qualification Stage · Live Sentinel Grid + Full Offline Continuity Architecture**

**Version 2.0 · Consolidated 2026-09-01 · Status: AUTHORITATIVE**

> **Precedence.** For any schema, enum, API shape, DDL or algorithm, [`TRINETRA_Canonical_Contracts.md`](TRINETRA_Canonical_Contracts.md) is normative. This document owns *procedure*: how each component is built, run, measured, hardened and demonstrated. Where an earlier version of this plan conflicts with either document, both of these win — the earlier text is superseded, not merely supplemented.

---

## PART A — FOUNDATIONS

### A1. What this document is

An implementation procedure detailed enough that a competent engineer — or an AI agent — can build TRINETRA from it without asking a clarifying question. Every section states what to build, what "done" means, and what failure mode it prevents.

Companion documents:

| Document | Owns |
|---|---|
| `docs/TRINETRA_Main_Plan_REFINED_v2.md` | Strategy, scope, schedule, policy |
| `docs/TRINETRA_Canonical_Contracts.md` | **All** schemas, DDL, API shapes, algorithms |
| `TRINETRA_Manas_Niklaus_AI_Lead_Execution_Manual.md` | AI + architecture procedure |
| `TRINETRA_Mihir_Backend_Data_Platform_Execution_Manual.md` | Backend procedure |
| `TRINETRA_Akshat_..._Execution_Manual.md` | Datasets, synthetic, benchmark procedure |
| `TRINETRA_Parth_..._Execution_Manual.md` | Frontend + demo procedure |

### A2. Scope boundary — initial stage vs final stage

Every row's right column is something we will be asked about. The correct answer is always "designed for, deliberately deferred" — never "not considered".

| Initial stage — build now | Final stage — design for, defer |
|---|---|
| Media adapter for a handful of feeds | 80,000-camera production deployment |
| Vehicle detection on one machine | Edge / regional inference federation |
| Single-camera tracking | Cross-camera Re-ID and embedding search |
| Plate detection + OCR | Vector database at grid scale |
| Temporal OCR consensus | Learned camera-transition prediction |
| Sighting persistence | HA, multi-region replication |
| Vehicle search | Full case management |
| Observed-sequence journey | Multi-agency RBAC |
| Watchlist alerts | Full audit + compliance regime |
| Basic camera health | Fleet-wide observability platform |

### A3. System requirements

| # | Requirement | Priority | Implementation |
|---|---|---|---|
| 1 | Operate with **no live feed at all** | **MUST** | `VideoFileSource` + `FrameSequenceSource` + seeded JSON fixtures |
| 2 | Consume Sentinel RTSP when available | MUST-when-available | RTSP-over-TCP adapter + PTS timing |
| 3 | HLS browser preview | SHOULD | Backend credential proxy + `hls.js` |
| 4 | No hardcoded camera list | **MUST** | `scripts/sync_cameras.py`, upsert on `external_camera_id` |
| 5 | Recover from stream failure | **MUST** | Exponential backoff + jitter, new session, tracker flush |
| 6 | Detect scene discontinuity | **MUST** | PTS-jump + global-histogram delta → new session |
| 7 | Store event provenance | **MUST** | The eight fields in §A4 |
| 8 | Reproduce any run | **MUST** | Seeds + manifest SHA-256 + weights SHA-256 + git commit |
| 9 | Avoid all paid software and data | **MUST** | §J5 zero-cost policy |

### A4. Event provenance — the eight mandatory fields

`camera_id` · `stream_session_id` · `track_id` · `source_pts_ms` · `observed_at` · `model_version` · `weights_hash` · `pipeline_version`

Without all eight, a benchmark number is an anecdote and a police-facing claim is unverifiable. These are not optional debug metadata; they are the difference between evidence and assertion.

### A5. The architectural invariant

> Every media source becomes the same `FrameEnvelope`. Every AI run emits the same `EventEnvelope`. **No business layer may know whether the source is live, recorded or synthetic.**

Acceptance test: switching `SOURCE_MODE=file` ↔ `SOURCE_MODE=live_rtsp` requires **configuration change only**. A single `if source == "live"` outside `ai/media/` breaks the invariant.

### A6. Component boundaries — exactly one owner each

| Component | Owner | Path |
|---|---|---|
| Media adapters | Manas | `ai/media/` |
| Contracts | Manas | `ai/contracts/` |
| Detection + tracking | Manas | `ai/detect/`, `ai/track/` |
| Plate detection + OCR | Manas | `ai/plate/`, `ai/ocr/` |
| Fusion + normalization | Manas | `ai/fusion/` |
| Ingest, persistence, dedup | Mihir | `backend/app/services/` |
| Search, journey, watchlist API | Mihir | `backend/app/api/v1/` |
| Realtime (Redis + WS) | Mihir | `backend/app/realtime/` |
| Datasets, manifests, licences | Akshat | `datasets/` |
| Synthetic generation, TRINETRA-HARD | Akshat | `datasets/synthetic/` |
| Benchmark execution | Akshat + Manas | `benchmark/` |
| UI, GIS, demo mode | Parth | `frontend/` |

Two people editing one component is how a 6-day schedule dies. If a change needs two owners, it needs a contract change instead.

---

## PART B — MEDIA LAYER

### B1. Worker loop — canonical structure

```
connect
  → create stream_session_id (UUID)
  → INSERT stream_sessions row
  → loop:
        read + decode frame
        build FrameEnvelope (camera_id, session, frame_index, pts_ms, wallclock, bgr, w, h, mode)
        sample by PTS   (skip unless pts_ms - last_emitted >= TARGET_INTERVAL_MS)
        push to latest-frame buffer (depth 1, overwrite)
        hand to AI pipeline
  → on failure:
        close capture
        UPDATE stream_sessions SET ended_at, end_reason
        flush tracker + evidence buffers
        sleep(backoff + jitter)
        mint a NEW session and reconnect
```

**Never** `while True: cap = cv2.VideoCapture(url)`. That construct has no backoff (it hammers the camera server), no session reset (tracks merge across the gap), and no failure accounting (you cannot tell a flaky camera from a flaky network).

### B2. Backpressure

Depth-1 latest-frame buffer. Producer overwrites; consumer takes whatever is current. Dropped frames are **counted and logged**, never silently discarded — the drop rate is the primary signal that a machine is over-subscribed.

Unbounded queues are forbidden. A 200 ms per-frame deficit at 10 fps accumulates 2 seconds of lag per second of runtime; after five minutes the "live" view is ten minutes stale, and shortly after that the process is OOM-killed mid-demo.

### B3. Reconnect policy

```python
BACKOFF_BASE_MS   = 500
BACKOFF_MAX_MS    = 30_000
delay = min(BACKOFF_BASE_MS * (2 ** attempt), BACKOFF_MAX_MS)
delay = delay * (0.5 + random.random())     # jitter: 50%–150%
```

Jitter is not decoration. Without it, 30 workers that lose a shared upstream all retry at exactly the same instants and turn a brief outage into a self-inflicted denial of service against the Sentinel server.

Reset `attempt` to 0 after a sustained-healthy interval (30 s of successful reads), not on the first successful frame.

### B4. Stream health

Per camera, tracked in memory and surfaced via `GET /api/v1/cameras/{id}`:

| Signal | Meaning |
|---|---|
| `frames_read` / `frames_emitted` / `frames_dropped` | Throughput and pressure |
| `last_frame_at` | Staleness → `degraded` after 10 s, `offline` after 30 s |
| `reconnect_count`, `last_end_reason` | Flakiness |
| `session_count` | How fragmented the observation history is |
| `decode_error_count` | Transport quality (high on UDP → confirms TCP requirement) |

### B5. PTS validation

Trust PTS, but verify it:

| Condition | Action |
|---|---|
| `pts_ms` decreases | Discontinuity → new session |
| `pts_ms` jumps forward > 5,000 ms | Discontinuity → new session |
| `pts_ms` is 0 or unavailable for N consecutive frames | Fall back to a synthetic monotonic clock, **log a warning, and mark the session `pts_unreliable`** |
| `pts_ms` identical across frames | Stalled decoder → force reconnect |

A session flagged `pts_unreliable` must not be used to substantiate a latency or timing claim.

### B6. Scene discontinuity detection

Sentinel streams loop with hard cuts. A loop boundary is visually identical to a camera being physically re-aimed: every tracked vehicle vanishes and unrelated ones appear instantly.

Detect with two cheap independent signals:

1. **PTS discontinuity** (§B5)
2. **Global histogram delta** — compare a downscaled grayscale histogram between consecutive emitted frames; a correlation drop below threshold across a single 100 ms step is not physically plausible for a fixed camera

On detection: end the session (`end_reason='discontinuity'`), flush tracker and evidence buffers, mint a new session. Without this, ByteTrack happily associates a car leaving frame at the loop end with a different car entering at the loop start, and produces a vehicle that teleports across the city.

### B7. Evidence buffer

Per `TrackKey`, hold a bounded ring buffer of the best crops:

```
max_crops_per_track = 5
eviction = lowest plate_quality score
flush on: track end, session end, or timeout (track idle > 3 s)
```

Bounded by construction. A busy junction with 40 simultaneous tracks must not be able to grow this without limit.

### B8. Adapter acceptance tests

| Test | Pass condition |
|---|---|
| `VideoFileSource` on a 60 s clip | ~600 emitted frames at 100 ms interval, `pts_ms` monotonic 0→60000 |
| `FrameSequenceSource` on 100 JPEGs | 100 frames, `frame_index` 0→99 |
| `SyntheticReplaySource` seed 42, run twice | Byte-identical event stream |
| `SentinelRTSPSource` forced kill mid-stream | New session minted, backoff observed, tracker flushed |
| All five adapters | Downstream code path byte-identical; only `source_mode` differs |

The last row is the invariant test. Run it in CI.

---

## PART C — CV PIPELINE

### C1. Stage order — do not reorder

```
FRAME → VEHICLE DETECT → TRACK → VEHICLE QUALITY GATE
      → PLATE DETECT → PLATE QUALITY RANK → OCR
      → TEMPORAL FUSION → NORMALIZE → VALIDATE → DEDUP → EVENT
```

Two choices worth defending because they separate a demo from a system:

- **Plate detection is its own model**, not a class of the vehicle detector. A plate is ~40×15 px in a 1920×1080 frame. A model trained to find cars will not reliably find it.
- **Fusion sits between OCR and normalization.** Single-frame OCR on a 60 px plate is close to a coin flip; three frames voting is a decision. This is the highest-accuracy-per-unit-effort stage in the pipeline and it requires no training.

### C2. Vehicle detection

RF-DETR (**Roboflow**, Apache-2.0), Nano / Small / Medium. Plus / XL / 2XL are differently licensed and excluded.

```python
with torch.inference_mode():
    detections = detector(frame_tensor)
```

Classes: `car`, `motorcycle`, `bus`, `truck`, `auto_rickshaw`, `other`. Auto-rickshaws matter in Gujarat and are frequently absent from COCO-derived class maps — verify coverage before accepting a pretrained baseline.

### C3. Tracking

ByteTrack (MIT). Produces `track_id`, local to `(camera_id, stream_session_id)` — see Contracts §1.2.

Flush completely on session change. Do not attempt cross-camera association in the initial stage; that is Re-ID and it is deferred.

### C4. Vehicle quality gate

Before spending plate-detection GPU on a crop, reject it cheaply:

| Reject when | Rationale |
|---|---|
| Vehicle bbox height < 60 px | The plate cannot be above ~20 px; OCR is hopeless |
| Bbox touches frame edge on ≥2 sides | Vehicle partially outside frame; plate likely cut |
| Bbox area is shrinking and already small | Vehicle departing; better frames already seen |
| Detector confidence < 0.35 | Probably not a vehicle |

This gate is where the multi-camera GPU budget is actually won.

### C5. Plate detection

Start with `justjuu/rtdetr-v2-license-plate-detection` (Apache-2.0). Its published 0.97 mAP / 0.88 small-object figures are **on its own test set** — treat them as a starting point, not as our number. Ours comes from TRINETRA-HARD.

Run on the vehicle crop, not the full frame: it raises the effective resolution of the plate region and cuts false positives on road signage and shop hoardings.

### C6. Plate quality ranking

```python
score = (0.30 * sharpness_norm      # variance of Laplacian
       + 0.25 * resolution_norm     # plate width vs 100 px reference
       + 0.25 * detector_confidence
       + 0.20 * exposure_norm)      # penalize clipped histograms
```

Keep the top **K = 3…5** crops per track. Running OCR on every plate in every frame is the single largest avoidable cost in the system.

### C7. OCR

PaddleOCR (Apache-2.0). Preprocessing variants — apply **one at a time**, measure each, keep only what pays:

`raw` · `grayscale` · `contrast stretch` · `adaptive threshold` · `2× upscale` · `mild sharpening`

Stacking preprocessing blindly is how a pipeline becomes both slower and less accurate at once.

**Super-resolution is not a default.** It is a plausible-looking accuracy trap: it hallucinates characters, turning `GJ01AB1234` into `GJ01A81234` with *higher* apparent confidence. If adopted at all, it must be justified on TRINETRA-HARD with a width-bucket breakdown showing it does not increase confident-wrong readings.

### C8. Temporal fusion

Contracts §4.3, verbatim. The benchmark that justifies this stage's existence must report accuracy **before vs after** consensus — that delta is one of the strongest technical talking points available, and it costs no GPU.

### C9. Validation

1. **Normalize** (Contracts §4.2)
2. **Indian plate grammar** — `^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$` as a soft check. A grammar miss **downgrades confidence**; it never rewrites the string and never discards the observation. BH-series and older formats exist and a hard filter would silently delete them.
3. **Spatio-temporal feasibility** (Contracts §4.5) — flag, never drop.

### C10. Failure taxonomy — the deliverable that gates the A100

After the baseline benchmark, classify every miss into exactly one bucket:

| Bucket | Symptom | Fix that would help |
|---|---|---|
| `vehicle_miss` | No vehicle detected | Vehicle detector training / Indian road data |
| `plate_miss` | Vehicle found, plate not | Plate detector training on small plates |
| `plate_too_small` | Plate < 30 px | Nothing in software — camera placement or accept |
| `ocr_wrong` | Plate found, text wrong | OCR training / synthetic corpus |
| `ocr_partial` | Some characters right | Temporal consensus / more frames |
| `fusion_wrong` | Best single frame was right, consensus wrong | Fusion weighting |
| `track_broken` | Same vehicle split across tracks | Tracker tuning |
| `track_merged` | Two vehicles in one track | Discontinuity / session handling |
| `duplicate` | One vehicle, multiple sightings | Dedup window |
| `dropped_frame` | Vehicle never sampled | Sampling interval / throughput |

**One dominant bucket** is the precondition for training anything. Two co-dominant buckets means the analysis is not finished.

---

## PART D — BACKEND AND DATA

### D1. Ingest

Contracts §6.1. Validation order matters: cheap structural checks before database lookups, so a malformed flood cannot generate 10,000 camera queries.

### D2. Transaction boundary

Contracts §5.9, verbatim. **Persist before publish.** The database is truth; the WebSocket is a notification.

### D3. Idempotency

`event_id` is the primary key of `ingestion_events`. A duplicate POST returns `200 {"status":"duplicate"}` and mutates nothing. The AI worker retries on network failure, so non-idempotent ingest silently double-counts sightings.

### D4. Error contract

Contracts §6.6. Every response carries `request_id`; the same value appears in the server log line. During a live demo that pairing is the whole debugging story.

### D5. Dedup

Contracts §4.7. Verify with `camera_reconnect.json`: two events, same `camera_id` and `track_id`, **different** `stream_session_id` → must yield **two** `vehicle_tracks` rows and **two** sightings. One row means `uq_trackkey` is wrong.

### D6. Search performance

```sql
EXPLAIN ANALYZE
SELECT * FROM plate_observations
WHERE plate_normalized = 'GJ01AB1234'
ORDER BY observed_at DESC LIMIT 50;
```

Must use `idx_obs_normalized_time`. A sequential scan here is the failure that turns a smooth demo into a ten-second pause.

Targets: search < 100 ms · ingest < 200 ms · journey < 300 ms.

### D7. Load smoke

```bash
python scripts/load_smoke.py \
  --endpoint http://localhost:8000/api/v1/events/vehicle-sighting \
  --count 500 --concurrency 10
```

Pass: zero 5xx, p95 under 200 ms, no connection-pool exhaustion. Run it *before* the demo, not during.

### D8. Realtime

Redis pub/sub → WebSocket fan-out. Contracts §6.7.

Reconnect test:
1. Open dashboard, confirm alerts arrive
2. Kill the backend
3. Observe UI shows disconnected — **not** a frozen stale view
4. Restart backend
5. UI reconnects automatically with backoff
6. UI **refetches from REST** and shows alerts raised during the outage

Step 6 is the one usually missed. A socket-only client silently loses every event from the gap.

### D9. Degraded modes

| Dependency down | Required behaviour |
|---|---|
| Redis | Ingest still succeeds; alerts persist; UI finds them on refresh |
| Postgres | Ingest returns `DEPENDENCY_UNAVAILABLE`; AI retries; **no data loss** |
| AI workers | API and UI stay up; cameras show `degraded`/`offline` |
| Frontend | Backend unaffected; `curl` still demonstrates the pipeline |

Every one of these must be *rehearsed*, because at least one will happen on demo day.

---

## PART E — FRONTEND CONTRACT

### E1. Required screens

| Screen | Must show |
|---|---|
| **Dashboard** | Live counters, recent sightings, alert feed, LIVE/REPLAY badge, dependency health |
| **Vehicle Search** | Plate query, time range, camera filter, exact vs fuzzy **separated**, confidence + evidence count |
| **Journey** | Ordered sightings on a map, **dashed** connectors, mandatory disclaimer, feasibility flags |
| **Cameras** | Catalogue, status, last-seen, HLS preview via backend proxy |
| **Alerts** | Watchlist hits, `match_state`, acknowledge action |
| **Offline Demo** | Deterministic fixture-driven run, works with the network unplugged |

### E2. Contract stability

`frontend/src/types/api.ts` mirrors Contracts §6.5 exactly. `/api/v1` changes additively only. A rename requires `/api/v2`.

### E3. Non-negotiable UI honesty requirements

| Requirement | Why |
|---|---|
| LIVE / REPLAY badge always visible, driven by `SystemStatus.is_live` | Presenting replay as live is the most damaging possible misrepresentation |
| Journey connectors **dashed**, never solid arrows | A solid arrow asserts a route we did not observe |
| Journey disclaimer rendered from the API's `disclaimer` field | The client cannot forget it if it comes from the server |
| Confidence shown with `evidence_count` | "0.94 from 3 observations" is honest; "94%" is not |
| Fuzzy results visually separated and labelled "candidate" | Fuzzy distance never confirms |
| `plate: null` rendered as "Unreadable", never blank or guessed | A null is information |
| Missing `snapshot_uri` → graceful placeholder | Broken image icons read as a broken system |
| `lat`/`lon` null → listed but not plotted | Do not invent coordinates |

---

## PART F — DATASETS AND BENCHMARKING

### F1. Licence register

Contracts §11. Nine mandatory fields per asset. `datasets/LICENSES.md` is a hard gate on training: an asset without a completed row cannot be used.

### F2. Do-not-mix rules

Contracts §11 status column plus the Main Plan §6 table. Enforced mechanically:

```bash
python scripts/check_split_leakage.py --manifest benchmark/manifests/final.json
```

Fails on overlap by file hash **or by source group**. Source-group disjointness matters more: two frames of the same vehicle at the same junction are not independent samples, even though their hashes differ.

### F3. Manifest freezing

```bash
python scripts/hash_manifest.py \
  --input  benchmark/manifests/final.json \
  --output benchmark/manifests/final.sha256
```

The SHA-256 goes into every benchmark report. A report without it cannot be reproduced and therefore cannot be cited.

### F4. Benchmark suite — seven tasks

| Task | Measures | Key diagnostic |
|---|---|---|
| Vehicle detection | precision, recall, mAP@50 | recall on small/occluded vehicles |
| Plate detection | precision, recall, mAP@50 | **small-plate recall** |
| OCR | exact-match accuracy, CER | accuracy by width bucket |
| **Temporal OCR** | accuracy **before vs after** consensus | the consensus delta |
| Runtime | FPS, p50/p95 latency, VRAM peak | real-time factor |
| Stream resilience | reconnects, sessions, dropped frames | recovery time |
| **End-to-end** | **correct-plate event rate** | by width bucket and condition |

Report shape: Contracts §7.3. Reports are append-only under `benchmark/reports/`.

### F5. TRINETRA-HARD

Contracts §7.4. ~1,000 observations, disjoint source groups, rebalanced exactly once before final freeze, never trained or tuned on.

### F6. Regression set

```
regression/{tiny_plate, night, blur, glare, occlusion, ocr_confusions}/
```

Small, fast, run on every meaningful change. TRINETRA-HARD answers "how good is it"; the regression set answers "did I just break something", and it must be cheap enough to run without thinking.

### F7. Synthetic generation

Main Plan §8 chain. Seeded and reproducible. Every generated image gets a manifest row recording its exact degradation parameters — that manifest is how you discover you never generated any 30–40 px night plates.

### F8. Model registry

```python
@dataclass
class ModelArtifact:
    model_id: str; version: str; task: str
    weights_path: str; config_path: str; sha256: str

class ModelRunner:
    def load(self, artifact: ModelArtifact) -> None: ...
    def predict(self, input): ...
```

Every benchmark report cites `weights_sha256`. An unhashed weights file is an unciteable result.

---

## PART G — TRAINING AND THE A100

### G1. The ratchet

```
BASELINE → TARGET FAILURE ANALYSIS → LOCAL SMOKE TRAIN → FROZEN DATA/CONFIG
        → A100 TRAIN → FROZEN HARD TEST → KEEP ONLY IF END-TO-END VALUE IMPROVES
```

The last clause is the policy. A model that gains 3 mAP points and 0 end-to-end correct-plate rate is discarded, regardless of cost.

### G2. Verification matrix — required before renting

| # | Check | Evidence | Owner |
|---|---|---|---|
| 1 | Offline pipeline runs end to end | Event in DB from an MP4 | Manas |
| 2 | Baseline benchmark recorded | `benchmark/reports/*.json` | Akshat |
| 3 | TRINETRA-HARD exists and is frozen | Manifest + SHA-256 | Akshat |
| 4 | Failure taxonomy written | Named component + bucket | Manas |
| 5 | Bottleneck is **one** component | Failure analysis doc | Manas |
| 6 | Local smoke train completes | Loss curve, any duration | Manas |
| 7 | Dataset manifest frozen | SHA-256 committed | Akshat |
| 8 | Split leakage check green | CI output | Akshat |
| 9 | Licences recorded for every training asset | `LICENSES.md` | Akshat |
| 10 | Training config committed | `config/training.yaml` | Manas |
| 11 | Eval runs unattended in one command | Writes a report | Akshat |

Any red row → **no rental**. Renting before knowing which component fails is the most expensive mistake available to this project.

### G3. GPU memory guard — 12 GB / 8 GB reality

The team's cards are 12 GB (Manas), 8 GB (Akshat), 8 GB (Parth). The A100 is 80 GB. Code that only runs on the A100 is useless for six of the eight days.

- Log `torch.cuda.memory_allocated()` and `memory_reserved()` at each stage
- Model size selectable by config (Nano / Small / Medium)
- **One active checkpoint in VRAM at a time**
- Release intermediate tensors explicitly; `del` + `torch.cuda.empty_cache()` at stage boundaries
- Always `torch.inference_mode()` for inference — never plain `eval()` with gradients live

### G4. Warm-up and batching

Discard the first **2** predictions from every timing measurement. CUDA kernel autotuning and lazy allocation make the first calls 5–20× slower, and including them makes every latency number a fiction.

| Situation | Batch size |
|---|---|
| Single live camera | 1 (latency dominates) |
| Multi-camera live | Small batch across cameras, bounded by VRAM |
| Offline benchmark | Largest that fits — throughput dominates |
| Training | Per `config/training.yaml`, tuned on the A100 |

### G5. Cost discipline

| Item | Rate |
|---|---|
| A100 SXM 80 GB | $1.59 / hr |
| Container | $0.007 / hr |
| Volume | $0.042 / hr |
| **Running total** | **$1.64 / hr** |
| Stopped (volume only) | $0.083 / hr |
| ~10 h | ≈ $16.40 |
| ~25 h | ≈ $41 |

Procedure: prepare locally → upload frozen data → run → download weights → **stop the pod**. A pod left running overnight costs more than the entire planned budget.

### G6. Export and packaging

Export weights with their `sha256`, config, dataset manifest hash, and the training command line. A weights file without those four things cannot be reproduced and cannot be defended.

---

## PART H — RUNTIME OPERATIONS

### H1. Startup order

```
1 PostgreSQL/PostGIS   2 Redis   3 FastAPI   4 AI workers   5 Frontend
Gate: GET /health/ready reports postgres:true, redis:true before step 4.
```

### H2. AI worker load order

```
validate config → load camera catalogue → resolve model artifacts (verify sha256)
→ load models (one at a time, log VRAM) → warm up (2 discarded predicts)
→ open media source → mint session → enter loop
```

Fail fast and loudly on a `sha256` mismatch. Silently running yesterday's weights invalidates every number produced that day.

### H3. Snapshot policy

| Rule | Value |
|---|---|
| Naming | `<camera>_<session>_<track>_<pts>_<event>.jpg` |
| Example | `cam04_3a7f1e02_42_343100_9f2c1d84.jpg` |
| Location | `$SNAPSHOT_DIR` — never inside the repo |
| Retention | Prototype-lifetime only; documented |
| Reference | `snapshot_uri` is **untrusted metadata**, never a filesystem path the backend dereferences |

The naming scheme is deliberately self-describing: given only a filename you can reconstruct which camera, which session, which track and which moment produced it.

### H4. Observability fields

Every log line: `timestamp` · `level` · `component` · `camera_id` · `stream_session_id` · `request_id` or `event_id` · `message`. Structured (JSON) so a demo-day failure can be grepped in seconds rather than read.

### H5. Fault injection — rehearse these

| Injected fault | Expected behaviour |
|---|---|
| Kill RTSP mid-stream | Backoff, new session, tracker flush, camera `degraded` |
| Kill Redis | Ingest still succeeds; alerts persist |
| Kill Postgres | `DEPENDENCY_UNAVAILABLE`, AI retries, no loss |
| Replay a duplicate event | `{"status":"duplicate"}` |
| POST a naive timestamp | `422 VALIDATION_FAILED`, field `observed_at` |
| POST `camera_id: "cam99"` | `422 UNKNOWN_CAMERA` |
| Feed a black/corrupt frame | No crash; no fabricated plate |
| Unplug the network at demo | Offline demo mode still runs |

### H6. Determinism

- Fixed seeds for all synthetic generation and replay
- `SyntheticReplaySource` with seed 42, run twice → **byte-identical** event stream
- Frozen manifests with hashes
- No wall-clock dependence in any code path that affects output
- Demo mode is reproducible on demand — the same run, every time

### H7. Replay controls

```bash
python scripts/replay.py --manifest data/replay/sentinel_offline_v1.json --mode realtime
# modes: realtime | fast | step
```

`fast` is for iteration. **`fast` output never substantiates a live performance claim** — see §K3.

---

## PART I — TESTING AND VERIFICATION

### I1. Smoke chain — the daily green path

```
docker compose up -d
alembic upgrade head
python scripts/sync_cameras.py            # or seed_demo.py when offline
curl http://localhost:8000/health/ready
curl -X POST http://localhost:8000/api/v1/events/vehicle-sighting \
     -H "Content-Type: application/json" \
     -d @tests/fixtures/ai_event_high_confidence.json
curl "http://localhost:8000/api/v1/search/vehicles?plate=GJ01AB1234"
curl http://localhost:8000/api/v1/journey/GJ01AB1234
curl http://localhost:8000/api/v1/alerts
pytest -q
```

If this chain is not green, nothing else matters that day.

### I2. Fixtures

Contracts §9. Twelve named files. Fixtures validate **contracts**; they never justify an accuracy claim.

### I3. End-to-end contract test

One test that runs an MP4 through the real pipeline into the real database and asserts: a track exists, ≥2 plate observations exist, the fused plate matches ground truth, exactly one sighting row exists, search returns it, journey orders it, and a watchlist match produces exactly one alert.

That single test is the definition of done in §L1, expressed as code.

### I4. Database reset

```bash
docker compose down
docker volume rm trinetra_pgdata
docker compose up -d
alembic upgrade head
python scripts/seed_demo.py
```

Rehearse this. A clean-room rebuild from a fresh clone must reach the smoke chain in under 15 minutes, and knowing that is what makes a demo-day database problem survivable.

### I5. Backup before the release candidate

```bash
pg_dump "$DATABASE_URL" > backups/trinetra_pre_rc.sql
```

---

## PART J — REPOSITORY, CONFIG, POLICY

### J1. Repository layout

Contracts §10.

### J2. Branch integration order

Mihir API skeleton + health → Parth UI shell + mocks → Akshat manifests + fixtures → Manas media adapter + baseline AI → AI event ingestion → search/journey → realtime alerts → demo resilience.

Rationale: the two people who can block everyone else (Mihir on contracts, Parth on fixtures) go first, and Parth's mock layer means the UI never waits on the AI.

### J3. Environment

Contracts §8.1. `.env.example` is committed; `.env` never is. `VITE_` variables are public by construction.

### J4. Config layering

Contracts §8.4. Five files. `python scripts/validate_config.py` at startup.

### J5. Zero-cost policy

| Allowed | Not allowed |
|---|---|
| Open-source software with compatible licences | Paid APIs |
| Compatible public datasets (recorded) | Paid datasets |
| Team-generated synthetic data | Paid SaaS |
| Local storage | Paid map or geocoding services |
| Local GPUs | Managed cloud databases |
| **RunPod A100 for training only** | Anything else requiring payment |

Maps: OpenStreetMap tiles via Leaflet/MapLibre. Coordinates come from the camera catalogue, never a paid geocoder.

### J6. Security and privacy baseline

| Control | Rule |
|---|---|
| Secrets | Environment variables only; never Git; never `VITE_` |
| Input validation | Typed schemas at every boundary; specific failing field |
| Transport | HTTPS for the HLS proxy; RTSP over TCP |
| Demo access | Minimal guard on any exposed instance |
| Traceability | `request_id` / `event_id` in every response and log line |
| Data minimization | Keep only what the prototype requires |
| Honesty | Confidence and evidence count shown wherever a plate is shown |
| Untrusted metadata | AI-supplied URIs are metadata, never dereferenced paths |
| Biometrics | No facial recognition — out of scope by design |

Facial recognition is excluded on proportionality grounds, not effort: this is a vehicle intelligence system, and adding biometric identification would transform its privacy profile while contributing nothing to the stated problem.

### J7. Dev profiles

| Profile | Sources | Use |
|---|---|---|
| `fixture` | JSON only, no AI | Frontend and API work |
| `offline` | MP4 + frames | Default full-pipeline development |
| `synthetic` | Seeded generator | Determinism and stress |
| `benchmark` | Frozen manifests | Measurement |
| `live` | Sentinel RTSP/HLS | Integration and final demo |

---

## PART K — RELEASE, DEMO, FREEZE

### K1. Release candidate procedure

1. Green smoke chain (§I1) from a clean clone
2. `pytest -q` green
3. Regression set green
4. TRINETRA-HARD run recorded with width buckets
5. Fault injection rehearsed (§H5)
6. Offline demo verified with the network physically unplugged
7. `pg_dump` backup taken
8. Version tagged
9. Demo script rehearsed end to end, timed
10. Projector test passed (§K4)

Tags: `v0.1.0-initial` → `v0.1.1-fix-ocr-consensus` → `v0.2.0-qualified`.

### K2. Architecture walkthrough — under 3 minutes, no slides

1. The problem: 80,000 cameras, 26 departments, no unified vehicle intelligence
2. The insight: **you cannot centralize the video — centralize the intelligence**
3. Sources: RTSP, HLS, MP4, frames, synthetic
4. One `FrameEnvelope` — nothing downstream knows the difference
5. PTS-driven sampling, latest-frame buffer
6. Vehicle detection → tracking → `TrackKey`
7. Quality gate → dedicated plate detection
8. Top-K crop ranking → OCR
9. **Temporal consensus** — show the four-frame worked example
10. Normalization → validation → dedup
11. One `EventEnvelope`, eight provenance fields
12. Persist before publish → search, journey, alert
13. Honest confidence: evidence count, not a probability
14. Resilience: unplug the network, the demo continues

Step 9 and step 14 are the two moments that distinguish this from every other ALPR submission.

### K3. Honesty rules

| Never say | Say instead |
|---|---|
| "the exact path the vehicle took" | **"observed movement sequence"** |
| "predicts where the vehicle will go" | **"Camera Search Prioritization"** |
| "87% probability" | "confidence 0.87 from 3 agreeing observations" |
| "40 FPS live" *(measured on accelerated replay)* | "40 FPS on replay at 5× acceleration" |
| "confirmed match" *(from fuzzy distance)* | "candidate match, requires review" |
| showing replay as "ONLINE" | LIVE / REPLAY badge, always visible |

Do not multiply uncalibrated confidences. Label every screen and every report REPLAY or LIVE.

### K4. Judge projector test

| # | Check |
|---|---|
| 1 | Readable at 1280×720 |
| 2 | Readable at 1920×1080 |
| 3 | No text clipped at either resolution |
| 4 | Map tiles load, or a documented offline tile fallback exists |
| 5 | Alert animation visible from 3 metres |
| 6 | Colour contrast survives a washed-out projector |
| 7 | Full demo runs with **Wi-Fi off** |

Row 7 has decided hackathons. Test it on the actual projector if at all possible.

### K5. Demo recovery plan

| Failure | Recovery |
|---|---|
| Live feed unavailable | Switch `config/live.yaml` → `config/offline.yaml`; announce it plainly |
| Backend crash | `docker compose restart backend`; UI reconnects automatically |
| Database corrupt | Restore `backups/trinetra_pre_rc.sql` |
| GPU OOM | Drop to the Nano model via config |
| Network dead | Offline demo mode, fixtures only |
| Laptop dies | Second machine with the same clone + dump, rehearsed |

### K6. Status signals during the demo

`GREEN` running normally · `YELLOW` degraded but functional · `RED` component down, recovery in progress · `BLUE` replay/offline mode · `BLACK` demo halted, fall back to walkthrough.

### K7. Final 24-hour freeze

- No new features
- No dependency upgrades
- No model swaps
- Bug fixes only, each with a test
- Demo rehearsed at least twice, timed
- Backups taken and **verified by restoring them**

### K8. Stop conditions

Stop work on a component and ship what exists when: the end-to-end path is green and the component is a refinement; or the remaining time is under the time needed to test the change; or the change would require a contract edit inside the freeze window.

---

## PART L — DEFINITION OF DONE

### L1. Core

From a cold `docker compose up` with **no live feed**, an offline clip produces:

1. A vehicle track with a valid `TrackKey`
2. Multiple plate observations for that track
3. A temporally fused, normalized plate
4. A persisted, deduplicated sighting with all eight provenance fields
5. A searchable result from `GET /api/v1/search/vehicles`
6. An observed-sequence journey across ≥3 cameras with feasibility annotation
7. A watchlist alert delivered over WebSocket and rendered without a refresh
8. A benchmark report with a width-bucket breakdown
9. A licence line for every dataset used
10. The same downstream path consuming Sentinel media with **only configuration changed**

### L2. Resilience

| Scenario | Required behaviour |
|---|---|
| Stream drops | Reconnect with backoff; new session; tracker flushed |
| Scene cut / loop boundary | New session; no cross-boundary track merge |
| Redis down | Events persist; alerts appear on refresh |
| Postgres briefly down | `DEPENDENCY_UNAVAILABLE`; AI retries; no loss |
| Duplicate event | `{"status":"duplicate"}`; nothing changes |
| Unreadable plate | `plate: null` persisted; no fabricated string |
| Implausible journey segment | Flagged and downgraded, never dropped |
| Projector / network failure | Deterministic offline demo mode |

### L3. Judge readiness

- Architecture walkthrough in under 3 minutes without slides
- Every accuracy number has a width-bucket breakdown behind it
- Every dataset has a licence line
- "Is any of this AGPL?" → no, and here is the register
- "How does this reach 80,000 cameras?" → the metadata-first argument
- "Are you doing facial recognition?" → no, and here is why not
- The demo survives the network being unplugged mid-presentation

### L4. Final go / no-go — three independent gates

| Gate | Decision | Precondition |
|---|---|---|
| **Qualification work** | GO / NO-GO | §L1 items 1–9 green offline |
| **Live integration** | GO / NO-GO | Live swap changes configuration only (§A5) |
| **A100 training** | GO / NO-GO | All 11 rows of §G2 green |

These are separate. Failing the live gate does not block qualification. Failing the A100 gate does not block either — it simply means we ship the pretrained baseline with an honest benchmark, which is a perfectly defensible outcome.

### L5. First-day procedure

**Step 1 — verify the toolchain**

```bash
nvidia-smi
ffmpeg -version
ffprobe -version
python --version
node --version
docker --version
git --version
```

**Step 2 — bring up infrastructure**

```bash
docker compose up -d
alembic upgrade head
curl http://localhost:8000/health/ready
```

**Step 3 — probe the grid (if reachable)**

```bash
curl -s https://cctv.corp8.cloud/cameras.json
ffplay -rtsp_transport tcp rtsp://103.250.160.189:8554/stream/cam04
```

If either fails, that is expected and not a blocker — proceed to Step 4 and continue entirely offline.

**Step 4 — seed offline**

```bash
python scripts/seed_demo.py
python scripts/generate_demo_events.py \
  --scenario sentinel_vehicle_journey_v1 --seed 42 --count 100 --output tests/fixtures
```

**Step 5 — prove the ingest contract**

```bash
curl -X POST http://localhost:8000/api/v1/events/vehicle-sighting \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/ai_event_high_confidence.json
```

**Steps 6–10** — search → journey → alerts → `pytest -q` → UI at `http://localhost:5173`, badge reading **REPLAY**.

### L6. Final technical principle

> **We don't centralize every video. We centralize intelligence.**

Metadata-first is why it scales. Source independence is why it survives. Honest confidence is why it can be trusted. Zero-cost licensing is why it can be handed over.

---

## APPENDIX A — Command reference

```bash
# Grid
curl -s https://cctv.corp8.cloud/cameras.json
ffplay -rtsp_transport tcp rtsp://103.250.160.189:8554/stream/cam04
python scripts/sync_cameras.py --url "$CAMERA_CATALOGUE_URL"

# Infrastructure
docker compose up -d
alembic upgrade head
uvicorn backend.app.main:app --reload --port 8000
curl http://localhost:8000/health/ready

# Replay and demo data
python scripts/replay.py --manifest data/replay/sentinel_offline_v1.json --mode realtime
python scripts/generate_demo_events.py --scenario sentinel_vehicle_journey_v1 \
       --seed 42 --count 100 --output tests/fixtures
python scripts/seed_demo.py

# Datasets and benchmarks
python scripts/check_split_leakage.py --manifest benchmark/manifests/final.json
python scripts/hash_manifest.py --input benchmark/manifests/final.json \
       --output benchmark/manifests/final.sha256

# Testing
pytest -q
python scripts/load_smoke.py --endpoint http://localhost:8000/api/v1/events/vehicle-sighting \
       --count 500 --concurrency 10

# Backup / reset
pg_dump "$DATABASE_URL" > backups/trinetra_pre_rc.sql
docker compose down && docker volume rm trinetra_pgdata && docker compose up -d
```

## APPENDIX B — First offline assets (Day 1)

| Asset | Quantity |
|---|---|
| Replay clips, 30–120 s | 3 — one day, one night, one blurred/rainy |
| Labelled plate crops | 200–500 |
| Event fixtures | 100, spanning all twelve cases in Contracts §9 |
| Replay layout | `data/replay/cam04/clip_day_01.mp4`, `cam04/clip_night_01.mp4`, `cam10/clip_junction_01.mp4`, `manifest.json` |

## APPENDIX C — Sentinel grid reference

| Property | Value |
|---|---|
| Catalogue | `https://cctv.corp8.cloud/cameras.json` |
| HLS | `https://cctv.corp8.cloud/<id>/index.m3u8` (password-protected) |
| RTSP | `rtsp://103.250.160.189:8554/stream/<id>` |
| WHEP | `http://103.250.160.189:8889/stream/<id>/whep` |
| Ports | 8554/TCP · 8889/TCP · 8189/UDP |
| Camera IDs | `cam01` … `cam30` |
| Behaviour | Live-only, no seeking, monotonic PTS, loops with hard scene cuts |

## APPENDIX D — Failure triage matrix

| Symptom | First diagnosis |
|---|---|
| No frames arriving | Transport — check RTSP TCP, then network, then credentials |
| Frames arrive, no detections | Model loaded? Correct input normalization? Confidence threshold? |
| Vehicles detected, no plates | Plate detector running on the crop or the full frame? Crop too small? |
| Plates found, OCR garbage | Preprocessing variant; check crop is upright and not over-sharpened |
| OCR right sometimes, wrong often | Insufficient frames reaching fusion — check quality gate and top-K |
| Same vehicle, many sightings | Dedup key or window (Contracts §4.7) |
| Vehicles merged across a scene cut | Discontinuity detection (§B6) |
| Journey shows a teleport | `uq_trackkey` missing `stream_session_id` |
| Latency growing over time | Unbounded queue — buffer depth must be 1 |
| Alerts missing after a reconnect | Client not refetching from REST (§D8 step 6) |

## APPENDIX E — Superseded material

Sections numbered beyond this document's Part L in the version 1.0 technical plan are superseded. Their substantive content has been folded into Parts B through L above. Three specific corrections carried over from that version:

1. **RF-DETR is Roboflow's, Apache-2.0** for Nano–Large. It is not an Ultralytics model. Plus/XL/2XL are excluded.
2. **RTMDet and YOLOX remain available as fallback detectors** (Apache-2.0). An earlier edit removed them without replacement.
3. **The day-numbered plan in this document is superseded** by `docs/TRINETRA_Main_Plan_REFINED_v2.md` §15, which is anchored to the real calendar: six build days (1–6 September), submission 7 September, hackathon 10–11 September.

## APPENDIX F — References

External sources consulted for the dataset, model and licensing decisions:

- Gujarat Police Sentinel problem statements — `sentinel.gujarat.gov.in/problems`
- Roboflow RF-DETR — model card and licence terms (Apache-2.0, Nano–Large)
- PaddleOCR — Apache-2.0 licence and model zoo
- ByteTrack — MIT licence, original paper
- justjuu `rtdetr-v2-license-plate-detection` — model card, Apache-2.0
- justjuu / thundarstrom licence-plate datasets — dataset cards, CC BY 4.0
- FANVID — dataset card and paper, CC BY 4.0
- CCPD — repository and paper, MIT
- Open Images V7 — annotation and image licence terms
- BDD100K — repository licence and data terms
- DataCluster Labs sample — CC BY-NC-ND terms
- UFPR-ALPR — academic use agreement
- Ultralytics — AGPL-3.0 licence terms
- PostGIS — geography type and GIST index documentation
- RunPod — A100 SXM 80 GB pricing page

Every licence claim in this plan and in Contracts §11 must be re-verified against the primary source before any weights are trained or shipped. Licence terms change; a citation is a pointer, not a guarantee.
