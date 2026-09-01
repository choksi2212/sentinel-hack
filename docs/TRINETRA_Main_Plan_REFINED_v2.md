# TRINETRA — MASTER PLAN

**Vendor-neutral, metadata-first CCTV vehicle intelligence for the Gujarat Police Sentinel grid.**

**Version 2.0 · Consolidated 2026-09-01 · Status: AUTHORITATIVE**

> **Precedence:** For any data structure, schema, API shape, enum or algorithm, [`TRINETRA_Canonical_Contracts.md`](TRINETRA_Canonical_Contracts.md) is normative and overrides this document. This document owns *strategy, scope, schedule and policy*.

---

## 0. Context, deadlines and what this document is for

| Item | Value |
|---|---|
| Competition | Gujarat Police **"Sentinel"** hackathon — `sentinel.gujarat.gov.in/problems` |
| Grid scale (stated) | 80,000+ cameras · 26 departments · 34 districts |
| Test grid available to us | ~30–50 feeds, `cam01`–`cam30` |
| **Application / shortlist deadline** | **7 September 2026** |
| **Hackathon** | **10–11 September 2026** |
| Today | **1 September 2026** |
| **Build days before the qualification gate** | **6** (Sep 1, 2, 3, 4, 5, 6 — submit on the 7th) |

**Team and ownership**

| Person | Domain | Machine |
|---|---|---|
| Manas / Niklaus | AI + architecture, contracts, media adapters, fusion | Ryzen 9 9955HX · **RTX 5070 Ti 12 GB** · 32 GB |
| Mihir | Backend, database, API, realtime | (as provisioned) |
| Akshat | Computer vision datasets, synthetic generation, benchmarking | **RTX 4060 8 GB** |
| Parth | Frontend, dashboard, GIS, demo | i7-14650HX · **RTX 5050 8 GB** · 24 GB |

Already installed across the team: PostgreSQL, PostGIS, Docker, CUDA, Node.js, Python.
**The only paid resource permitted:** RunPod A100 SXM 80 GB, for training/fine-tuning only, and only after §11's gate passes.

**Read order for a new contributor (or an AI asked to implement this):**
1. This document — §1, §2, §3, §4 (what we are building and not building)
2. `TRINETRA_Canonical_Contracts.md` — every schema, in full
3. Your own execution manual (Manas / Mihir / Akshat / Parth)
4. `TRINETRA_Technical_Implementation_Master_Plan_REFINED_v2.md` — the operational detail

---

## 1. Executive strategy

Build **one reliable vertical slice** around the vehicle-sighting workflow, end to end, before adding any second feature. Benchmark before training. Train exactly one component: the one with a *measured* target-domain bottleneck.

```
SENTINEL RTSP ──┐
SENTINEL HLS  ──┤
OFFLINE MP4   ──┼→ MediaSource → FrameEnvelope → CV pipeline → EventEnvelope
FRAME SEQUENCE──┤                                                    ↓
SYNTHETIC     ──┘                                          PostgreSQL / PostGIS
                                                                    ↓
                                                          REST + Redis + WebSocket
                                                                    ↓
                                                       React dashboard · GIS · alerts
```

**The USP, in one sentence:** *a vendor-neutral, metadata-first CCTV intelligence platform.*

**The closing line:** *We don't centralize every video. We centralize intelligence.*

Why that framing is the whole strategy: nobody can move 80,000 video streams to a datacenter. TRINETRA moves *events* — a few hundred bytes of structured metadata per vehicle sighting — and leaves the video where it already is. That is what makes the architecture credible at grid scale, and it is why "centralized video storage" appears in the banned list in §16.

### 1.1 The full pipeline — every stage, in order

```
STREAM
  → FRAME SAMPLING            (PTS-driven, 100 ms interval, latest-frame buffer)
  → VEHICLE DETECTION         (RF-DETR)
  → TRACKING                  (ByteTrack → TrackKey)
  → VEHICLE QUALITY GATE      (skip hopeless crops before spending GPU on them)
  → PLATE DETECTION           (dedicated model — not a class of the vehicle detector)
  → PLATE QUALITY RANKING     (top K=3–5 crops per track)
  → OCR                       (PaddleOCR)
  → TEMPORAL FUSION           (weighted consensus across frames)
  → NORMALIZATION             (GJ 01 AB 1234 → GJ01AB1234)
  → VALIDATION                (Indian plate grammar + spatio-temporal feasibility)
  → DEDUP                     (one sighting per vehicle per camera pass)
  → SIGHTING EVENT
  → SEARCH / GIS MAP / WATCHLIST ALERT
```

Two design choices in that chain are worth defending explicitly because they are the difference between a demo and a system:

- **Plate detection is its own model**, not a class in the vehicle detector. A plate is 40×15 px in a 1920×1080 frame. A detector trained to find cars will not reliably find it; a detector trained on plate crops will.
- **Fusion sits between OCR and normalization.** A single frame's OCR on a 60 px plate is a coin flip. Three frames voting is a decision. This is the highest-leverage accuracy stage in the whole pipeline and it costs no training.

---

## 2. Architectural invariant — source independence

> Every media source is converted into the same `FrameEnvelope`. Every AI run emits the same `EventEnvelope`. **No business layer may know whether the source is live, recorded or synthetic.**

This is not an aesthetic preference. It is the risk control for the single largest project threat: **the Sentinel grid may be unavailable, throttled, or password-blocked for the entire competition window.** If any part of the backend, database or UI can tell the difference, then losing the feed stops the project. If nothing downstream can tell, then losing the feed changes one line of configuration.

**The acceptance test for this invariant:** switching between `SOURCE_MODE=file` and `SOURCE_MODE=live_rtsp` requires **a configuration change only** — zero code edits, zero schema changes, zero UI changes. If a single `if source == "live"` appears outside `ai/media/`, the invariant is broken.

### 2.1 Media sources and their downstream impact

| Source | `source_mode` | Used for | Downstream impact |
|---|---|---|---|
| Sentinel RTSP (TCP) | `live_rtsp` | Live AI inference | **None** — same `FrameEnvelope` |
| Sentinel HLS (proxied) | `live_hls` | Browser preview | **None** — same business layer |
| MP4 replay | `file` | Offline full-pipeline development | **None** |
| Frame directory | `frames` | CV debugging | **None** |
| Synthetic replay | `synthetic` | Deterministic stress / demo | **None** |
| JSON fixture | *(n/a — bypasses AI)* | API + frontend contract testing | **None** |

### 2.2 Sentinel grid — the actual endpoints

| Property | Value |
|---|---|
| Catalogue | `curl -s https://cctv.corp8.cloud/cameras.json` |
| HLS | `https://cctv.corp8.cloud/<id>/index.m3u8` — CDN, **password-protected** |
| RTSP | `rtsp://103.250.160.189:8554/stream/<id>` |
| WHEP (WebRTC) | `http://103.250.160.189:8889/stream/<id>/whep` |
| Ports | 8554/TCP (RTSP) · 8889/TCP (WHEP/HLS) · 8189/UDP (WebRTC media) |
| Camera IDs | `cam01` … `cam30` |

Behavioural facts that dictate the adapter design:

- **Live-only.** No seeking, no history, no re-request of a past moment. If you miss it, it is gone. → the offline replay path is not a convenience, it is the only way to iterate on the same footage twice.
- **Monotonic PTS.** Use it as the source clock (Contracts §2.1).
- **Streams loop with hard scene cuts.** A loop boundary looks exactly like a camera being physically re-aimed: every tracked vehicle vanishes and new ones appear instantly. → discontinuity detection must mint a new `stream_session_id`, or the tracker will invent vehicles that teleport.
- **HLS is password-protected** → the browser must never hold that credential. Proxy it through the backend (Contracts §8.3).
- **RTSP-over-TCP for AI, HLS for browser.** UDP drops packets under load and the resulting corrupt macroblocks read as detector failures — you will spend hours debugging a model that is fine.

---

## 3. Qualification scope — what MUST exist

| # | Requirement | Priority | How it is satisfied |
|---|---|---|---|
| 1 | Operate with **no live feed at all** | **MUST** | `VideoFileSource` + `FrameSequenceSource` + seeded JSON fixtures |
| 2 | Consume Sentinel RTSP when available | MUST-when-available | RTSP-over-TCP adapter + PTS timing |
| 3 | HLS browser preview | SHOULD | Backend proxy + `hls.js` |
| 4 | Dynamic camera catalogue — **no hardcoded camera list** | **MUST** | `scripts/sync_cameras.py` upsert on `external_camera_id` |
| 5 | Recover from stream failure | **MUST** | Exponential backoff + jitter, new session, tracker flush |
| 6 | Detect scene discontinuity | **MUST** | PTS-jump + global-histogram change → new session |
| 7 | Store event provenance | **MUST** | 8 provenance fields (§3.2) |
| 8 | Reproduce any run | **MUST** | Seeds + manifest SHA-256 + weights SHA-256 + git commit |
| 9 | Avoid all paid software and data | **MUST** | §17 zero-cost policy |
| 10 | Vehicle detection + single-camera tracking | MUST | RF-DETR + ByteTrack |
| 11 | Dedicated plate detection | MUST | justjuu RT-DETRv2 baseline, then custom if needed |
| 12 | PaddleOCR baseline | MUST | Apache-2.0 |
| 13 | Multi-frame temporal OCR consensus | MUST | Contracts §4.3 |
| 14 | Indian plate normalization + conservative fuzzy candidates | MUST | Contracts §4.2, §4.6 |
| 15 | Sighting persistence / search / journey | MUST | Contracts §5, §6 |
| 16 | Watchlist alerts over WebSocket | MUST | Contracts §6.7 |
| 17 | Camera health + failure surface in UI | MUST | `GET /api/v1/cameras` + status badges |
| 18 | Deterministic demo mode | **MUST** | Seeded replay + fixtures, reproducible on demand |

### 3.1 Scope boundary — initial stage vs final stage

Stating this explicitly protects the schedule. Every row's right column is a thing we will be *asked about* and should answer as "designed for, deferred", never "not thought about".

| Initial stage (build now) | Final stage (design for, defer) |
|---|---|
| Media adapter for a handful of feeds | 80,000-camera deployment |
| Vehicle detection on one machine | Edge/regional inference federation |
| Single-camera tracking | Cross-camera Re-ID and embedding search |
| Plate detection + OCR | Vector database at grid scale |
| Temporal OCR consensus | Learned camera-transition prediction |
| Sighting persistence | HA, multi-region replication |
| Vehicle search | Full case management |
| Observed-sequence journey | Multi-agency RBAC |
| Watchlist alerts | Full audit + compliance regime |
| Basic camera health | Fleet-wide observability platform |

### 3.2 Event provenance — the eight mandatory fields

Every persisted sighting must be able to answer *"where did this claim come from?"*:

`camera_id` · `stream_session_id` · `track_id` · `source_pts_ms` · `observed_at` · `model_version` · `weights_hash` · `pipeline_version`

Without these, a benchmark number is an anecdote and a police-facing claim is unverifiable.

---

## 4. Explicitly deferred to the final stage

- 80,000-camera production deployment
- Advanced vehicle Re-ID / vector retrieval
- Large-scale FAISS or vector infrastructure
- Learned camera-transition prediction
- Regional / edge orchestration
- Kafka or Kubernetes — **unless a measurement demands them**
- Full RBAC / audit / case-management stack
- Full multi-stream video wall

Each of these is a legitimate final-stage feature. Each is also a perfect way to arrive at 7 September with an impressive architecture diagram and no working plate pipeline.

---

## 5. Dataset strategy

Indian-first and target-domain-first. A dataset is admitted only when **license, provenance, use compatibility and failure-mode relevance** are all documented in `datasets/LICENSES.md`.

**Free to download ≠ safe to reuse.** A HuggingFace or Kaggle page showing `CC BY 4.0` proves the uploader *claimed* that license; it does not prove they held the rights to the underlying images. For a police-facing system, record the provenance chain, not the badge.

### 5.1 Dataset register

| Resource | License | Status | Purpose |
|---|---|---|---|
| thundarstrom Indian plates — 3,742 images | CC BY 4.0 | CORE after record | Indian plate localization |
| justjuu plate detection — 8,823 (6,176 / 1,765 / 882) | CC BY 4.0 | CORE after record | Plate detector training |
| justjuu `rtdetr-v2-license-plate-detection` — 0.97 mAP, 0.88 small-object | Apache-2.0 | **CORE BASELINE** | Zero-training plate baseline |
| FANVID — 1,463 clips @180×320, 20–60 FPS, 49 plates, 31,096 boxes | CC BY 4.0 | CORE TEMPORAL BENCHMARK | Low-res temporal recognition — **eval only** |
| CCPD — 300k+ | MIT | AUXILIARY | Localization/geometry robustness — **not** Indian OCR grammar |
| Own synthetic Indian CCTV corpus | Ours | CORE | OCR + controlled CCTV degradation |
| TRINETRA-HARD | Ours | **CORE FROZEN TEST** | Unseen difficult benchmark |
| thirdeyelabs Indian road — 210 GB | verify | OPTIONAL | Vehicle adaptation only after measured failure; sample 10–30k |
| Open Images V7 | annots CC BY 4.0 / imgs CC BY 2.0 | OPTIONAL | Generic auxiliary — verify per-image terms |
| BDD100K | repo BSD-3 / data terms differ | RESEARCH ONLY | Terms review before shipping any weights |
| Gamester03 — 1,709 rows | **unknown** | **PENDING — DO NOT TRAIN** | Blocked pending provenance |
| DataCluster sample | **CC BY-NC-ND** | **EXCLUDED** | NC-ND forbids training and derivatives |
| UFPR-ALPR | academic / non-commercial | **EXCLUDED** | Incompatible with core policy |
| VeRi-776 / CityFlow / VehicleID | agreement-bound | DEFERRED | Final-stage Re-ID after licensing review |

### 5.2 Mixture hypotheses — starting points, to be revised by measurement

**Plate detector training mix**

| Portion | Share |
|---|---|
| Indian real plates | 35% |
| General real plates | 25% |
| Hard synthetic (small/blurred/angled) | 20% |
| Augmentation of the above | 20% |

**OCR training mix**

| Portion | Share |
|---|---|
| Synthetic Indian plates | 50% |
| Real Indian plates | 30% |
| Low-resolution / temporal samples | 20% |

These are **hypotheses**, not settled configuration. They get adjusted based on the failure taxonomy from the baseline benchmark — not on intuition, and not before a baseline exists.

---

## 6. Dataset roles — do not mix them

This table is the most frequently violated rule in ALPR projects, and every violation produces a number that looks good and means nothing.

| Data | Primary role | **Do NOT use it for** | Why the misuse is fatal |
|---|---|---|---|
| Indian plate datasets | Plate localization, domain adaptation | Assuming correct OCR grammar | Localization accuracy says nothing about reading `GJ01AB1234` correctly |
| CCPD | Localization / geometry robustness | Indian OCR target | Chinese plate grammar. An OCR head tuned on it learns the wrong character distribution and the wrong plate layout |
| FANVID | Temporal hard **evaluation** | Routine training | It is one of only two unbiased low-res temporal signals available. Training on it converts evidence into self-congratulation |
| Synthetic Indian plates | OCR pretraining / augmentation | Claiming real-world accuracy | Synthetic degradation is a model of CCTV, not CCTV. Real glare, real sensor noise and real motion blur differ |
| TRINETRA-HARD | **Final evaluation only** | Training **or tuning** | The moment a hyperparameter is chosen by looking at it, it stops measuring generalization |
| Sentinel clips | Runtime / target-domain validation | Uncontrolled training corpus | Unlabeled, unbalanced, loops with duplicates; training on it silently overfits to 30 specific camera angles |

**Operational enforcement:** `python scripts/check_split_leakage.py --manifest benchmark/manifests/final.json` runs in CI and fails the build on any overlap by file hash **or by source group**. Source-group disjointness matters more than file-hash disjointness — two different frames of the same vehicle at the same junction are not independent samples.

---

## 7. TRINETRA-HARD — the frozen benchmark

~1,000 labeled observations, drawn from source groups **disjoint** from all training and validation data. Rebalanced **exactly once**, before the final freeze. Never trained on. Never tuned on.

| Bucket | Target |
|---|---|
| Large / easy | 200 |
| Blur / motion | 200 |
| Night / low light | 200 |
| Glare / exposure | 150 |
| Angle / perspective | 150 |
| Tiny plate | 100 |

**Reporting is by plate pixel width, always:**

| Bucket | Width | | Difficulty tier | Representative width |
|---|---|---|---|---|
| B1 | > 100 px | | Easy | > 100 px |
| B2 | 80 – 100 px | | Medium | 60 – 100 px |
| B3 | 60 – 80 px | | Hard | 30 – 60 px |
| B4 | 40 – 60 px | | Extreme | < 30 px |
| B5 | 30 – 40 px | | | |
| B6 | < 30 px | | | |

A single averaged accuracy figure is **not an acceptable deliverable**. "92%" typically decomposes into 98% above 80 px and 51% below 40 px, and the second number is the one that determines whether this works on real infrastructure. Publishing the breakdown ourselves is also strictly better than having a judge extract it.

Owner: Akshat. Consumers: Manas (model selection), everyone (the submission's accuracy claim).

---

## 8. Synthetic CCTV plate data

Generate Indian registration text inside plate geometry, embed it in vehicle/road context, then **degrade it toward real CCTV conditions**. Clean high-resolution rectangles are worse than useless — they teach the model that plates are always legible.

```
Indian registration text
  → permitted font rendering            (license-checked fonts only)
  → plate geometry                      (correct Indian layout, spacing, aspect)
  → vehicle / road context              (paste into a real scene, not a blank canvas)
  → perspective warp                    (cameras look down and sideways, never straight on)
  → downscale                           (to the target width bucket — this is the point)
  → blur / noise / compression          (motion blur, sensor noise, H.264 artifacts)
  → glare / brightness / occlusion      (headlights, sodium lamps, partial obstruction)
  → manifest with BOTH raw and normalized label
```

The generator must be **seeded and reproducible**, and must emit a manifest row per image recording the exact degradation parameters used. When the model fails on 30–40 px night plates, that manifest is how you find out whether you generated any.

Deliberately generate across all four difficulty tiers, weighted toward Hard and Extreme — the easy cases are already solved by the pretrained baseline.

---

## 9. Model stack — refined and licence-cleared

| Task | Starting candidates | License | Selection rule |
|---|---|---|---|
| Vehicle detection | **RF-DETR** (Roboflow) Nano / Small / Medium | **Apache-2.0** | Pretrained first; train only if the target benchmark reveals material weakness |
| Tracking | **ByteTrack** | MIT | Keep unless track continuity measurably fails |
| Plate detection | justjuu **RT-DETRv2** baseline, then RF-DETR custom | Apache-2.0 | Compare target recall, **small-plate recall**, latency, and downstream OCR accuracy |
| OCR | **PaddleOCR** | Apache-2.0 | Temporal consensus before any custom OCR |
| Custom OCR | 36-character recognizer | Ours | Only after measured failure survives consensus |
| Matching | Normalization + position-aware fuzzy candidates | MIT-style | **Never confirm on fuzzy distance alone** |
| Re-ID | Deferred | — | Final stage only |
| Fallback detectors | RTMDet, YOLOX | Apache-2.0 | Available if RF-DETR underperforms on our hardware |
| Benchmark comparison only | Ultralytics YOLO | **AGPL-3.0** | **Never shipped.** Comparison numbers only |

### 9.1 Two licensing facts that must not be garbled

1. **RF-DETR is Roboflow's, and it is Apache-2.0** for the core code and the **Nano through Large** weights. **Plus / XL / 2XL carry different licensing and are excluded** from the zero-cost core path. RF-DETR is *not* an Ultralytics model — attributing it to Ultralytics reintroduces precisely the AGPL-3.0 exposure this stack exists to avoid.
2. **Ultralytics YOLO is AGPL-3.0.** Its network-use copyleft obligations are incompatible with delivering this to a police department as a prototype. It may appear in a benchmark table as a comparison baseline. It may not appear in the shipped pipeline.

### 9.2 Full stack, locked

**AI** — RF-DETR · ByteTrack · PaddleOCR · RapidFuzz-style position-aware matching · PyTorch · OpenCV
**Backend** — FastAPI · SQLAlchemy · Alembic · PostgreSQL 16 + PostGIS 3.4 (`GEOGRAPHY(Point,4326)` + GIST) · Redis pub/sub · WebSocket
**Frontend** — React · TypeScript · Vite · Leaflet or MapLibre · TanStack Query · Tailwind · hls.js
**Infra** — Docker Compose (single host)

**Banned, deliberately:** Kubernetes · Kafka · microservice decomposition · blockchain · chatbot layer · **facial recognition** · centralized video storage.

Facial recognition is excluded on scope and proportionality grounds, not just effort: this is a *vehicle* intelligence system, and adding biometric identification would change its privacy profile entirely while contributing nothing to the stated problem.

---

## 10. End-to-end performance metric

```
E2E correct-plate event rate  =  correct final plate events / eligible vehicle events
```

Break it down by **camera, plate width bucket, lighting, blur, vehicle type and source mode**. Detector mAP, OCR CER, FPS and p95 latency are **diagnostics** — they explain the primary number and are never reported in its place.

Rationale: mAP 0.97 on a plate-detection test set is compatible with a system that reads almost no plates correctly, because detection is one of nine stages and every stage multiplies. The only number that describes the product is the fraction of vehicles that come out the far end with the right plate string attached.

Full report shape: Contracts §7.3. Accuracy-vs-latency decision table: Contracts §7.5 — with the worked case that mAP 94% @ 40 FPS beats mAP 96% @ 12 FPS, because a multi-camera system is throughput-bound.

---

## 11. Training policy — the ratchet

```
BASELINE                        measure the pretrained stack on TRINETRA-HARD
  ↓
TARGET FAILURE ANALYSIS         name the ONE failing component and the bucket it fails in
  ↓
LOCAL SMOKE TRAIN               tiny run on the 12 GB / 8 GB card; prove the loop works
  ↓
FROZEN DATA/CONFIG              manifest SHA-256 + config committed; no edits after this
  ↓
A100 TRAIN                      RunPod, hours not days
  ↓
FROZEN HARD TEST                TRINETRA-HARD, untouched
  ↓
KEEP ONLY IF END-TO-END VALUE IMPROVES
```

The last line is the entire policy. A model that improves plate-detection mAP by 3 points and improves the end-to-end correct-plate rate by 0 is **discarded**, regardless of how much it cost to train.

### 11.1 Money

| Item | Rate |
|---|---|
| A100 SXM 80 GB (GPU) | $1.59 / hr |
| Container | $0.007 / hr |
| Volume | $0.042 / hr |
| **Total running** | **$1.64 / hr** |
| Stopped (volume only) | $0.083 / hr |
| ~10 hours | ≈ **$16.40** |
| ~25 hours | ≈ **$41** |

Cost discipline: prepare everything locally, upload frozen data, run, download weights, **stop the pod**. A pod left running overnight costs more than the entire planned training budget.

### 11.2 Verification matrix — required before renting the A100

Renting the GPU before knowing which component fails is the most expensive mistake available to this project. All of the following must be true first:

| # | Check | Evidence |
|---|---|---|
| 1 | Offline pipeline runs end to end | Event in DB from an MP4 |
| 2 | Baseline benchmark recorded | `benchmark/reports/*.json` |
| 3 | TRINETRA-HARD exists and is frozen | Manifest + SHA-256 |
| 4 | Failure taxonomy written | Named component + named bucket |
| 5 | The bottleneck is **one** component | Failure analysis document |
| 6 | Local smoke train completes | Loss curve, any duration |
| 7 | Dataset manifest frozen | SHA-256 committed |
| 8 | Split leakage check passes | `check_split_leakage.py` green |
| 9 | Licenses recorded for every training asset | `datasets/LICENSES.md` |
| 10 | Training config committed | `config/training.yaml` |
| 11 | Evaluation script runs unattended | One command, writes a report |

If any row is red, the answer is **no rental**.

---

## 12. Sentinel runtime rules

- AI consumes **RTSP over TCP**.
- Camera list is **discovered dynamically** from the catalogue; never hardcoded.
- **PTS**, not `CAP_PROP_FPS`, is source timing truth.
- **Bounded, latest-frame** buffering. Depth 1.
- Reconnect with **exponential backoff + jitter** (jitter matters: without it, 30 workers reconnect in lockstep and DDoS the camera server).
- **Reset tracker and session state** after any reconnect or hard scene discontinuity.
- Track identity is `camera_id + stream_session_id + track_id` — always all three.
- HLS is for **browser preview**, proxied through the backend.
- Never `while True: cap = cv2.VideoCapture(url)`.

---

## 13. Offline continuity

> If live feeds are unavailable for the **entire** competition window, the project continues with **exactly the same** AI, backend and frontend interfaces — using MP4 replay, frame sequences, synthetic data and JSON fixtures.

**This must be tested before the team depends on Sentinel, not after it fails.** Concretely: the offline path is built and green on **Day 1**, and the live swap is attempted on Day 7. That ordering is deliberate. The reverse ordering — build against live, add offline later — is how teams lose a competition to someone else's network outage.

### 13.1 First offline assets needed (Day 1, Akshat + Manas)

- 3 clips, 30–120 s each: **1 day, 1 night, 1 blurred/rainy**
- 200–500 plate crops with labels
- 100 event fixtures spanning high-confidence / low-confidence / unreadable / duplicate / reconnect / implausible-journey

### 13.2 The live swap

```yaml
# config/offline.yaml            →     # config/live.yaml
source_mode: file                      source_mode: live_rtsp
replay_manifest: data/replay/...       rtsp_transport: tcp
                                       rtsp_base: rtsp://103.250.160.189:8554/stream
```

Nothing else changes. If anything else needs to change, §2's invariant was violated somewhere and that is the bug to fix.

---

## 14. Team integration gates

Each gate has named owners and a binary pass condition. A gate is not "mostly working".

| # | Gate | Owners | Pass condition |
|---|---|---|---|
| G1 | **Media** | Manas + Mihir | Camera config → valid `FrameEnvelope` stream |
| G2 | **CV** | Manas + Akshat | Replay clip → valid `EventEnvelope` (schema-validated) |
| G3 | **Persistence** | Manas + Mihir | Event POSTed → row in `vehicle_sightings` → returned by search |
| G4 | **Realtime** | Mihir + Parth | Watchlist match → alert visible in UI without refresh |
| G5 | **Journey** | Mihir + Parth | Ordered multi-camera observations render on the map with disclaimer |
| G6 | **Offline continuity** | All | Full workflow, Sentinel unplugged, from a cold `docker compose up` |
| G7 | **Live swap** | All | Only source configuration changes; zero code edits |

### 14.1 Checkpoint meetings

| CP | When | Participants | Subject |
|---|---|---|---|
| CP1 | End D1 | All | Contracts frozen, repo skeleton merged, health endpoint green |
| CP2 | End D2 | Manas + Akshat | Detector + tracker baseline numbers on TRINETRA-HARD |
| CP3 | End D3 | Manas + Akshat | Plate detection + OCR baseline; failure taxonomy drafted |
| CP4 | End D4 | Manas + Mihir | First real `EventEnvelope` persisted end to end (**G2 + G3**) |
| CP5 | End D5 | Mihir + Parth | Search, journey, alerts live in the UI (**G4 + G5**) |
| CP6 | End D6 | All | Benchmark frozen, demo rehearsed, submission package assembled (**G6**) |
| CP7 | D7–D8 | All | Live swap attempt (**G7**), A100 go/no-go |

---

## 15. Schedule — 6 days to the qualification gate

The earlier "10-day plan" predates the calendar being pinned down. **There are 6 build days.** This is the reconciled schedule and it supersedes every other day-numbered plan in every TRINETRA document.

| Date | Day | MUST exist by end of day | Hard exit criterion |
|---|---|---|---|
| **Sep 1** | D1 | Repo skeleton · contracts frozen · offline MP4 adapter → `FrameEnvelope` · DB migrations · API skeleton + `/health/ready` · UI shell with mocks | **G1** |
| **Sep 2** | D2 | Vehicle detection (RF-DETR) + ByteTrack + session/discontinuity handling · baseline numbers recorded | Track IDs survive a forced reconnect as **two** tracks |
| **Sep 3** | D3 | Plate detection baseline + PaddleOCR + quality ranking + top-K crop selection · failure taxonomy | Plate strings out of a real clip, accuracy broken down by width bucket |
| **Sep 4** | D4 | Temporal consensus · normalization · validation · dedup · `EventEnvelope` emitted · backend ingest + search | **G2 + G3** |
| **Sep 5** | D5 | Journey + watchlist + Redis + WebSocket · UI wired to real API · GIS map · LIVE/REPLAY badge | **G4 + G5** |
| **Sep 6** | D6 | TRINETRA-HARD benchmark frozen · demo rehearsed twice on the projector · submission package complete | **G6** |
| **Sep 7** | — | **SUBMIT** | Application filed |
| Sep 8 | D7 | Sentinel live swap attempt · A100 go/no-go per §11.2 | **G7** or documented reason live is unavailable |
| Sep 9 | D8 | Hardening · fault injection · demo recovery drill · final freeze | Recovery plan rehearsed |
| **Sep 10–11** | — | **HACKATHON** | Present |

### 15.1 Parallel lanes — nobody blocks anybody

| Person | D1–D2 | D3–D4 | D5–D6 |
|---|---|---|---|
| **Manas** | Contracts, media adapters, `FrameEnvelope`, detector + tracker | Plate detection, OCR, quality ranking, fusion, event emission | Benchmark freeze, integration, A100 decision, architecture walkthrough |
| **Mihir** | Docker Compose, migrations, API skeleton, health, ingest endpoint | Ingest validation, dedup, search, watchlist matching | Journey, alerts, Redis + WS, load smoke, `EXPLAIN ANALYZE` |
| **Akshat** | Offline clips, first crops, licence register, manifest tooling | Synthetic generator, TRINETRA-HARD assembly, annotation QA | Benchmark runs, width-bucket reports, regression set |
| **Parth** | UI shell, routing, mock API layer, type definitions | Search + camera screens against real API | Journey map, alert feed, demo mode, projector rehearsal |

**Critical-path dependency:** Parth is unblocked on D1 **only if** the contracts and fixtures land on D1. That is why CP1 is a hard gate and why `tests/fixtures/` is a Day-1 deliverable rather than a testing afterthought.

---

## 16. Anti-overengineering rule

Do not spend the qualification schedule on Re-ID, Kafka, Kubernetes, large-scale vector search, or production infrastructure while the plate pipeline or the end-to-end workflow remains unreliable.

### 16.1 Anti-patterns — do not do these

| Anti-pattern | What it costs |
|---|---|
| Running the full pipeline on every frame | GPU budget that multi-camera scale needs |
| Unbounded queues | Latency drift → OOM crash mid-demo |
| Global track IDs | Merged vehicles, fabricated journeys |
| Arrival time as video time | Every temporal claim becomes wrong |
| Uncalibrated confidence presented as probability | A claim that cannot survive questioning |
| Hardcoded camera IDs | Breaks the day the catalogue changes |
| Live feed as the only dev path | One outage stops the project |
| `while True: cap = cv2.VideoCapture(url)` | Hammers the camera server; no session reset |
| Super-resolution as a default | Hallucinated characters: `GJ01AB1234` → `GJ01A81234` |
| Renting the A100 before knowing which component fails | Money spent for no measured gain |
| Centralized video storage | Contradicts the entire USP |

---

## 17. Zero-cost policy

| Allowed | Not allowed |
|---|---|
| Open-source software with compatible licenses | Paid APIs of any kind |
| Compatible public datasets (recorded in `LICENSES.md`) | Paid datasets |
| Team-generated synthetic data | Paid SaaS |
| Local storage | Paid map or geocoding services |
| Local GPUs | Managed cloud databases |
| **RunPod A100 for training only** | Anything requiring a credit card beyond RunPod |

Maps use OpenStreetMap tiles via Leaflet/MapLibre. Geocoding, if ever needed, is done from the camera catalogue's own coordinates — never a paid geocoder.

---

## 18. Security and privacy baseline

| Control | Rule |
|---|---|
| Secrets | Environment variables only. **Never** in Git. Never in `VITE_` variables (those compile into the public browser bundle) |
| Input validation | Typed schemas at every boundary; reject with the specific failing field |
| Transport | HTTPS for HLS proxy; RTSP over TCP |
| Demo access | Minimal guard on the demo instance |
| Traceability | `request_id` in every response **and** the matching server log line; `event_id` in every event |
| Data minimization | Keep only what the prototype requires |
| Honesty in the UI | Always display confidence and evidence count alongside a plate |
| Untrusted metadata | AI-supplied `snapshot_uri` / `plate_crop_uri` are **metadata, never filesystem paths**. Never dereference them to build a path |
| Biometrics | No facial recognition. Out of scope by design |

---

## 19. Honesty rules — non-negotiable

These exist because a police-facing system that overclaims is worse than one that underclaims, and because a judge will test exactly these points.

| Never say | Say instead | Why |
|---|---|---|
| "the exact path the vehicle took" | **"observed movement sequence"** | We observed 4 points. We did not observe the road between them |
| "predicts where the vehicle will go" | **"Camera Search Prioritization"** | We rank cameras worth checking. We do not forecast behaviour |
| "confidence 87% probability" | "confidence score 0.87 from 3 agreeing observations" | The score is uncalibrated evidence, not a probability |
| "runs at 40 FPS live" *(measured on accelerated replay)* | "40 FPS on replay at 5× acceleration" | Replay acceleration never substantiates a live claim |
| "confirmed match" *(from fuzzy distance)* | "candidate match, requires review" | Fuzzy distance never confirms |
| showing replay as "ONLINE" | **LIVE / REPLAY badge, always visible** | The most damaging possible misrepresentation |

Do not multiply uncalibrated confidences together and present the product. Label every screen, every report and every number as REPLAY or LIVE.

---

## 20. Definition of done

### 20.1 Core

TRINETRA qualifies technically when, from a cold `docker compose up` with **no live feed**, an offline clip produces:

1. a vehicle track with a valid `TrackKey`
2. multiple plate observations for that track
3. a temporally fused, normalized plate
4. a persisted, deduplicated sighting with full provenance
5. a searchable result via `GET /api/v1/search/vehicles`
6. an observed-sequence journey across ≥3 cameras with feasibility annotation
7. a watchlist alert delivered over WebSocket and rendered without a page refresh

…and the **same downstream path** consumes Sentinel media when accessible, with only configuration changed.

### 20.2 Resilience

| Scenario | Required behaviour |
|---|---|
| Stream drops | Reconnect with backoff; new session; tracker flushed |
| Scene cut / loop boundary | New session; no cross-boundary track merge |
| Redis down | Events still persist; alerts appear on refresh |
| Database briefly unavailable | Ingest returns `DEPENDENCY_UNAVAILABLE`; AI retries; no data loss |
| Duplicate event replayed | `{"status":"duplicate"}`; nothing changes |
| Unreadable plate | `plate: null` event persisted; no fabricated string |
| Implausible journey segment | Flagged and downgraded; never silently dropped |
| Projector / network failure at demo | Deterministic offline demo mode from fixtures |

### 20.3 Judge readiness

- The architecture walkthrough runs in under 3 minutes without slides
- Every accuracy number has a width-bucket breakdown behind it
- Every dataset has a license line
- The licensing story survives the question "is any of this AGPL?"
- The scale story answers "how does this reach 80,000 cameras?" with the metadata-first argument
- The privacy story answers "are you doing facial recognition?" with "no, and here is why not"
- The demo survives the network being unplugged mid-presentation

---

## 21. Final principle

> **We don't centralize every video. We centralize intelligence.**

Every architectural decision in TRINETRA follows from that sentence. Metadata-first is why it scales. Source independence is why it survives. Honest confidence is why it can be trusted. Zero-cost licensing is why it can be handed over.
