# TRINETRA — CANONICAL CONTRACTS

**Version 1.1 · Status: NORMATIVE · Consolidated 2026-09-01**

---

## 0. How to use this document

This file is the **single source of truth** for every data structure that crosses a person boundary in TRINETRA. It exists because four separate manuals previously carried four incompatible versions of the same event schema.

**Rules of precedence:**

1. If this document and any other TRINETRA document disagree, **this document wins**.
2. Nobody edits a contract in their own manual. Contracts are changed **here**, in one commit, announced to all four owners.
3. Every schema below is reproduced verbatim in the manual of the person who consumes it. Those copies are marked `COPIED FROM CANONICAL CONTRACTS — DO NOT EDIT HERE`.
4. Breaking changes require a `schema_version` bump plus a migration note in §12.

**Owners:** Manas/Niklaus produces `FrameEnvelope` and `EventEnvelope`. Mihir consumes `EventEnvelope` and produces the REST/WS contracts. Parth consumes the REST/WS contracts. Akshat consumes the dataset/manifest contracts.

---

## 1. Identity model — read this before anything else

Five different IDs exist. Confusing them is the most likely silent-corruption bug in the project.

| ID | Scope | Type | Assigned by | Stable? |
|---|---|---|---|---|
| `external_camera_id` | Global, provider-owned | `TEXT` e.g. `cam04` | Sentinel catalogue | **Permanent** |
| `cameras.id` | Internal DB | `UUID` | Postgres | Permanent |
| `stream_session_id` | One runtime connection | `UUID` | AI worker at connect | Dies on reconnect |
| `track_id` | One camera + one session | `INTEGER` | ByteTrack | **Reused across sessions** |
| `event_id` | Global within TRINETRA | `UUID` | AI worker per event | Permanent |

### 1.1 The `camera_id` format decision — LOCKED

```
On the wire (EventEnvelope, REST, WS, fixtures, UI):
    camera_id  =  the Sentinel catalogue ID, verbatim, lowercase
                  cam01 … cam30

FORBIDDEN:  CAM_001   CAM-001   Cam04   CAMERA_4   cam4  (no zero-pad drift)
```

Rationale: the Sentinel grid publishes `cam01`–`cam30` at `https://cctv.corp8.cloud/cameras.json`. Any invented format guarantees that every mock fixture, DB seed and UI screenshot has to be redone on the day live access arrives. Human-readable labels live in `cameras.name` (`"Ashram Road Junction"`), never in the ID.

Offline replay clips reuse the same namespace so that swapping to live changes nothing: a clip recorded for cam04 is `data/replay/cam04/clip_day_01.mp4` and emits `camera_id: "cam04"`.

### 1.2 TrackKey — the invariant

```
TrackKey = (camera_id, stream_session_id, track_id)
```

A tracker ID **alone is never globally meaningful**. ByteTrack restarts numbering at 1 on every new session. Two vehicles observed on cam04 before and after a reconnect will both be `track_id: 42`. Any table, dict, cache or DB row keyed on `(camera_id, track_id)` will merge them into one vehicle and produce a journey that never happened.

A new `stream_session_id` is minted on:
- initial connect
- any reconnect after transport/decoder failure
- detected hard scene discontinuity
- replay restart / manifest change

On new session: **flush** tracker state, evidence buffers and all in-flight temporal OCR accumulators.

---

## 2. FrameEnvelope — the source boundary

Every media source converges here. Downstream code must never call `cv2.VideoCapture` or FFmpeg directly.

```python
# ai/contracts/frame.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

SourceMode = Literal["live_rtsp", "live_hls", "file", "frames", "synthetic"]

@dataclass(frozen=True)
class FrameEnvelope:
    camera_id: str                # "cam04" — §1.1
    stream_session_id: str        # UUID str, minted at connect
    frame_index: int              # monotonic within session, starts at 0
    pts_ms: int                   # SOURCE timeline position, milliseconds
    wallclock_utc: Optional[str]  # ISO-8601 Z; None for pure file replay
    frame_bgr: np.ndarray         # HxWx3 uint8, BGR (OpenCV order)
    width: int
    height: int
    source_mode: SourceMode
```

### 2.1 Timing rules — LOCKED

```python
pts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))     # CORRECT
```

**Never** derive source time from `CAP_PROP_FPS` (lies on RTSP), from `frame_index / fps` (drifts), or from `time.time()` at arrival (measures your network, not the video).

Both timelines are always kept:
- `pts_ms` / `source_pts_ms` → **where in the video** this happened
- `wallclock_utc` / `observed_at` → **when the system saw it**

They diverge by design during replay, `--speed 5.0`, and network stalls. Reporting one as the other is the easiest way to make a false claim.

### 2.2 Sampling and buffering — LOCKED

```
TARGET_INTERVAL_MS = 100        # ~10 inferences/sec/camera
Emit a frame to AI only when  pts_ms - last_emitted_pts_ms >= TARGET_INTERVAL_MS
Buffer depth = 1 (latest frame wins; drop the old one)
```

Principle: **freshness beats completeness.** A live operator needs the vehicle that is passing now, not a perfect analysis of the vehicle that passed 40 seconds ago. Unbounded queues convert a 200 ms processing deficit into minutes of accumulated latency and eventual OOM.

### 2.3 Required adapters

| Class | `source_mode` | Input | Notes |
|---|---|---|---|
| `SentinelRTSPSource` | `live_rtsp` | `rtsp://103.250.160.189:8554/stream/<id>` | **Force TCP.** AI path. |
| `SentinelHLSSource` | `live_hls` | `https://cctv.corp8.cloud/<id>/index.m3u8` | Password-protected. Preview only. |
| `VideoFileSource` | `file` | `.mp4` path | First-class dev path, not a fallback. |
| `FrameSequenceSource` | `frames` | directory of `%06d.jpg` | CV debugging. |
| `SyntheticReplaySource` | `synthetic` | seeded generator | Deterministic stress/demo. |

All five satisfy:

```python
class MediaSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[FrameEnvelope]: ...   # None = end of stream
    def close(self) -> None: ...
    @property
    def session_id(self) -> str: ...
```

RTSP-over-TCP is mandatory for the AI path: UDP loses packets under load and produces corrupt macroblocks that read as detector failures.

---

## 3. EventEnvelope v1.1 — the CV→backend boundary

**This replaces every earlier event schema in every manual.** Changes from the previously-circulating variants: `timestamp` → `observed_at`; added `stream_session_id`, `source_pts_ms`, `source_mode`, `match_state`, `plate_width_px`, `evidence_count`, `model.*`.

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
  "vehicle": {
    "type": "car",
    "confidence": 0.93,
    "bbox_xyxy": [812, 344, 1102, 587]
  },
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

### 3.1 Field rules

| Field | Required | Rule |
|---|---|---|
| `schema_version` | ✅ | Exactly `"1.1"`. Backend rejects unknown majors. |
| `event_id` | ✅ | UUIDv4, generated once by AI. Idempotency key. |
| `camera_id` | ✅ | §1.1 format. Backend rejects unknown cameras. |
| `stream_session_id` | ✅ | UUID. **Never omit** — see §1.2. |
| `track_id` | ✅ | Integer ≥ 0. Meaningless without the other two. |
| `observed_at` | ✅ | ISO-8601 **timezone-aware**, UTC `Z`. Naive datetimes rejected. |
| `source_pts_ms` | ✅ | Integer ≥ 0. Source timeline. |
| `source_mode` | ✅ | One of the five §2 values. |
| `vehicle.type` | ✅ | `car` \| `motorcycle` \| `bus` \| `truck` \| `auto_rickshaw` \| `other` |
| `vehicle.confidence` | ✅ | Float `0.0 … 1.0` |
| `plate` | ⬜ | **Whole object may be `null`** when no plate was read. |
| `plate.raw` | ✅ if plate | Exactly what OCR returned, unmodified. Preserved for audit. |
| `plate.normalized` | ⬜ | May be `null` if normalization yields `""`. |
| `plate.match_state` | ✅ if plate | §3.3 |
| `plate.plate_width_px` | ✅ if plate | Integer. **Required** — without it §7 width-bucket reporting is impossible. |
| `plate.evidence_count` | ✅ if plate | Number of frames whose OCR agreed on `normalized`. |
| `image_quality` | ✅ | Float `0.0 … 1.0`, from §4.1. |
| `evidence.*` | ⬜ | URIs may be absent. **Treated by backend as untrusted metadata** — never dereferenced, never used to build a filesystem path. |
| `model.*` | ✅ | Provenance. Required for every reproducibility claim. |

### 3.2 Never invent a plate

`plate: null` is a **valid, expected, correct** event. An unreadable plate is real information: it says a vehicle passed and could not be identified. Fabricating a guess to avoid a null is the worst possible failure mode in a police system.

### 3.3 `match_state` — LOCKED enum

```
"exact"           normalized plate matched a watchlist entry character-for-character
"probable"        ≥2 agreeing observations, strong confidence, no exact watchlist hit
"low_confidence"  single observation or weak confidence
"unreadable"      plate located but no usable text  (plate.normalized is null)
```

Derivation:

```python
def match_state(evidence_count: int, fused_confidence: float, exact_watchlist_hit: bool) -> str:
    if exact_watchlist_hit:                              return "exact"
    if evidence_count >= 2 and fused_confidence >= 0.80: return "probable"
    return "low_confidence"
```

**`match_state` is descriptive, not an investigative conclusion.** It never appears in the UI as "confirmed".

### 3.4 Event state machine

```
NEW → OBSERVED → PLATE_CANDIDATE → { CONFIRMED | PROBABLE | UNREADABLE }
    → PERSISTED → WATCHLIST_CHECKED → ALERTED → CLOSED
```

A weak observation must not reach `ALERTED` by accident. Only `CONFIRMED` (exact normalized watchlist match) creates an alert with `match_state="exact"`. `PROBABLE` may create an alert **explicitly labelled as a candidate**. `UNREADABLE` never creates an alert.

---

## 4. Algorithms that must be identical everywhere

### 4.1 Image quality score — LOCKED weights

```python
# ai/quality.py
def plate_quality(crop_bgr) -> float:
    """Returns 0.0..1.0. Used for crop ranking AND as the temporal-fusion weight."""
    return (0.30 * sharpness_norm(crop_bgr)        # variance of Laplacian, normalized
          + 0.25 * resolution_norm(crop_bgr)       # plate pixel width vs 100px reference
          + 0.25 * detector_confidence             # from the plate detector
          + 0.20 * exposure_norm(crop_bgr))        # penalize clipped hi/lo histograms
```

Keep the **top K = 3…5** crops per track. Running OCR on every frame of every plate of every vehicle on every camera is the single largest avoidable cost in the pipeline.

### 4.2 Plate normalization — LOCKED

```python
import re
def normalize_plate(raw: str) -> str:
    """GJ 01 AB 1234 → GJ01AB1234.  'gj-01-ab-1234' → GJ01AB1234."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())
```

Both `raw` and `normalized` are stored. `raw` is the audit trail; `normalized` is the search key. Normalization is **never** applied to `raw` in place.

### 4.3 Temporal OCR consensus — LOCKED

```python
from collections import defaultdict

def fuse(observations):
    """observations: [{text, ocr_confidence, image_quality}, ...] for ONE TrackKey."""
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
        "confidence": score[best] / sum(score.values()),   # share of total evidence
        "evidence_count": count[best],
    }
```

Worked example — why this matters:

| Frame | OCR text | conf | quality | weight |
|---|---|---|---|---|
| 1 | `GJ01AB1234` | 0.91 | 0.90 | 0.819 |
| 2 | `GJ01AB1234` | 0.94 | 0.92 | 0.865 |
| 3 | `GJ01AB1234` | 0.88 | 0.87 | 0.766 |
| 4 | `GJ01A81234` | 0.63 | 0.55 | 0.347 |

`GJ01AB1234` = 2.450 (3 frames) vs `GJ01A81234` = 0.347 (1 frame) → **fused = GJ01AB1234, evidence_count = 3, confidence = 0.876**. A single-frame system that happened to pick frame 4 would have been wrong.

### 4.4 Confidence calibration — the honesty rule

```
HIGH    ≥3 agreeing observations AND fused confidence ≥ 0.85
MEDIUM  ≥2 agreeing observations
LOW     otherwise
```

Detector confidence, OCR confidence and fusion share are **relative evidence, not probabilities**. Do not multiply them together and present the product as a likelihood — that arithmetic is meaningless on uncalibrated scores and will be challenged.

### 4.5 Travel feasibility — LOCKED

```python
required_speed_kmh = haversine_km(a.lat, a.lon, b.lat, b.lon) / max(hours_between(a, b), 1e-6)
```

If `required_speed_kmh` exceeds a configured plausibility ceiling (start: 150 km/h urban), **downgrade confidence and flag the segment**. Never silently delete the sighting — a flagged implausible pair is evidence of an OCR error or a genuinely unusual event, and both are worth surfacing. Road-network routing (OSRM/OSM) is deferred; straight-line distance is a deliberate, documented under-estimate, which makes the check conservative in the right direction.

### 4.6 Fuzzy matching — the safety rule

Position-aware fuzzy matching (RapidFuzz-style) **generates candidates only**. It may never:
- silently rewrite `plate.normalized`
- create a `match_state: "exact"`
- independently raise a confirmed watchlist alert

Exact normalized equality is exact search. Everything else is a ranked candidate list shown to a human.

Common Indian-plate confusion pairs to weight explicitly: `0↔O`, `1↔I↔L`, `8↔B`, `5↔S`, `2↔Z`, `6↔G`.

### 4.7 Sighting dedup — LOCKED

```python
dedupe_key = sha256(f"{camera_id}|{stream_session_id}|{track_id}|{normalized_plate or ''}")
DEDUP_WINDOW_SECONDS = 10
```

On a repeat within the window: **update** `last_seen_at`, increment `observation_count`, and keep the **best** evidence (highest `ocr_confidence × image_quality`). Do not insert a second sighting row. Do not discard the underlying `plate_observations` rows — those are the audit trail that proves the consensus in §4.3.

Without this, one row per frame per vehicle lands in the database and search results become unusable within minutes.

---

## 5. Database schema — complete and authoritative

PostgreSQL 16 + PostGIS 3.4. Image pinned: `postgis/postgis:16-3.4`. Never `:latest`.

```sql
-- alembic/versions/000_extensions.sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 5.1 cameras

```sql
CREATE TABLE cameras (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_camera_id  TEXT NOT NULL UNIQUE,          -- 'cam04' — §1.1, immutable
    name                TEXT NOT NULL,                 -- 'Ashram Road Junction'
    location            GEOGRAPHY(Point, 4326),        -- NULL allowed until surveyed
    rtsp_url            TEXT,
    hls_url             TEXT,
    status              TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (status IN ('online','offline','degraded','unknown')),
    last_seen_at        TIMESTAMPTZ,
    first_registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_cameras_external ON cameras (external_camera_id);
CREATE INDEX idx_cameras_location ON cameras USING GIST (location);
```

`external_camera_id` is **immutable once seeded**. A camera that vanishes from the catalogue is marked `offline` — **never deleted**, because historical sightings reference it.

### 5.2 stream_sessions

```sql
CREATE TABLE stream_sessions (
    id            UUID PRIMARY KEY,                    -- the stream_session_id from AI
    camera_id     UUID NOT NULL REFERENCES cameras(id),
    source_mode   TEXT NOT NULL
                  CHECK (source_mode IN ('live_rtsp','live_hls','file','frames','synthetic')),
    source_ref    TEXT,                                -- URL or file path (no credentials)
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    end_reason    TEXT   -- 'eof'|'reconnect'|'discontinuity'|'shutdown'|'error'
);
CREATE INDEX idx_sessions_camera_started ON stream_sessions (camera_id, started_at DESC);
```

This makes `stream_session_id` a real foreign key rather than a loose string, and gives camera health a home separate from transient connection state.

### 5.3 vehicle_tracks

```sql
CREATE TABLE vehicle_tracks (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id         UUID NOT NULL REFERENCES cameras(id),
    stream_session_id UUID NOT NULL REFERENCES stream_sessions(id),
    track_id          INTEGER NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ,
    first_pts_ms      BIGINT,
    last_pts_ms       BIGINT,
    vehicle_type      TEXT,
    vehicle_color     TEXT,
    CONSTRAINT uq_trackkey UNIQUE (camera_id, stream_session_id, track_id)
);
CREATE INDEX idx_tracks_camera_time ON vehicle_tracks (camera_id, started_at DESC);
```

**`uq_trackkey` is the fix for the most dangerous defect in the earlier schema.** Without `stream_session_id` in this constraint, cam04's `track_id 42` from before a reconnect and `track_id 42` from after it collapse into one vehicle, and every journey built from that data is silently wrong.

### 5.4 plate_observations

```sql
CREATE TABLE plate_observations (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vehicle_track_id UUID NOT NULL REFERENCES vehicle_tracks(id) ON DELETE CASCADE,
    plate_raw        TEXT,
    plate_normalized TEXT,
    ocr_confidence   REAL CHECK (ocr_confidence BETWEEN 0 AND 1),
    image_quality    REAL CHECK (image_quality BETWEEN 0 AND 1),
    plate_width_px   INTEGER,                          -- enables §7 width buckets
    observed_at      TIMESTAMPTZ NOT NULL,             -- system time
    source_pts_ms    BIGINT,                           -- source timeline
    frame_index      INTEGER,
    snapshot_uri     TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_obs_normalized_time ON plate_observations (plate_normalized, observed_at DESC)
    WHERE plate_normalized IS NOT NULL;
CREATE INDEX idx_obs_track ON plate_observations (vehicle_track_id);
```

Every individual OCR read lands here. This is the audit trail behind §4.3.

### 5.5 vehicle_sightings — the deduplicated, searchable unit

```sql
CREATE TABLE vehicle_sightings (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dedupe_key        TEXT NOT NULL UNIQUE,            -- §4.7
    camera_id         UUID NOT NULL REFERENCES cameras(id),
    stream_session_id UUID NOT NULL REFERENCES stream_sessions(id),
    vehicle_track_id  UUID REFERENCES vehicle_tracks(id),
    plate_normalized  TEXT,
    plate_raw         TEXT,
    confidence        REAL CHECK (confidence BETWEEN 0 AND 1),
    match_state       TEXT NOT NULL
                      CHECK (match_state IN ('exact','probable','low_confidence','unreadable')),
    evidence_count    INTEGER NOT NULL DEFAULT 1,
    plate_width_px    INTEGER,
    vehicle_type      TEXT,
    source_mode       TEXT NOT NULL,
    location          GEOGRAPHY(Point, 4326),          -- denormalized for fast GIS
    first_seen_at     TIMESTAMPTZ NOT NULL,
    last_seen_at      TIMESTAMPTZ NOT NULL,
    first_pts_ms      BIGINT,
    best_snapshot_uri TEXT,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_sight_plate_time  ON vehicle_sightings (plate_normalized, first_seen_at DESC)
    WHERE plate_normalized IS NOT NULL;
CREATE INDEX idx_sight_camera_time ON vehicle_sightings (camera_id, first_seen_at DESC);
CREATE INDEX idx_sight_location    ON vehicle_sightings USING GIST (location);
CREATE INDEX idx_sight_time        ON vehicle_sightings (first_seen_at DESC);
```

`match_state` lives here — which is where Manas's emitted field previously had nowhere to go.

### 5.6 watchlist

```sql
CREATE TABLE watchlist (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_normalized TEXT NOT NULL,
    reason           TEXT,
    priority         TEXT NOT NULL DEFAULT 'medium'
                     CHECK (priority IN ('low','medium','high','critical')),
    active           BOOLEAN NOT NULL DEFAULT true,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT
);
CREATE UNIQUE INDEX uq_watchlist_active ON watchlist (plate_normalized) WHERE active;
```

### 5.7 alerts

```sql
CREATE TABLE alerts (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    watchlist_id UUID NOT NULL REFERENCES watchlist(id),
    sighting_id  UUID NOT NULL REFERENCES vehicle_sightings(id),
    match_state  TEXT NOT NULL
                 CHECK (match_state IN ('exact','probable')),   -- never low_confidence
    confidence   REAL,
    priority     TEXT NOT NULL,
    acknowledged BOOLEAN NOT NULL DEFAULT false,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_alert_once UNIQUE (watchlist_id, sighting_id)
);
CREATE INDEX idx_alerts_created ON alerts (created_at DESC);
```

`uq_alert_once` makes alert creation idempotent. The `CHECK` forbids alerting on `low_confidence` — see §3.4.

### 5.8 ingestion_events — idempotency ledger

```sql
CREATE TABLE ingestion_events (
    event_id       UUID PRIMARY KEY,                   -- AI's event_id — the idempotency key
    schema_version TEXT NOT NULL,
    camera_id      TEXT NOT NULL,                      -- external id as received
    received_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    status         TEXT NOT NULL
                   CHECK (status IN ('accepted','rejected','duplicate')),
    error_code     TEXT,
    raw_payload    JSONB NOT NULL
);
CREATE INDEX idx_ingest_received ON ingestion_events (received_at DESC);
```

Re-POSTing the same `event_id` returns `200` with `{"status":"duplicate"}` and changes nothing. Retry-safe ingestion is mandatory: the AI worker will retry on network failure.

### 5.9 Transaction boundary — LOCKED

```
BEGIN
  INSERT ingestion_events          (event_id PK → duplicate detection)
  INSERT stream_sessions           (ON CONFLICT DO NOTHING)
  INSERT/UPDATE vehicle_tracks     (ON CONFLICT uq_trackkey DO UPDATE)
  INSERT plate_observations
  INSERT/UPDATE vehicle_sightings  (ON CONFLICT dedupe_key DO UPDATE — §4.7)
  SELECT watchlist WHERE active AND plate_normalized = :normalized
  INSERT alerts                    (ON CONFLICT uq_alert_once DO NOTHING)
COMMIT
-- only now:
PUBLISH redis/websocket alert
```

**Persist before publish.** The database is the source of truth; the WebSocket is a notification. If Redis is down, the alert still exists and the dashboard finds it on next refresh. If you publish first and the transaction rolls back, you have shown an operator an alert that does not exist.

---

## 6. REST API — `/api/v1`

### 6.1 Ingest

```
POST /api/v1/events/vehicle-sighting
Content-Type: application/json
Body: EventEnvelope v1.1  (§3)

200 {"status":"accepted","event_id":"…","sighting_id":"…","alert_created":true}
200 {"status":"duplicate","event_id":"…"}
422 error envelope (§6.6)
```

Validation, in order: `schema_version` supported → required fields present → `observed_at` timezone-aware → confidences in `[0,1]` → `camera_id` exists in `cameras` → enums valid. Reject with the **specific** failing field, never a generic 400.

### 6.2 Search

```
GET /api/v1/search/vehicles?plate=GJ01AB1234&from=…&to=…&camera_id=cam04&fuzzy=false&limit=50

200 {
  "query": {"plate":"GJ01AB1234","normalized":"GJ01AB1234","fuzzy":false},
  "count": 3,
  "results": [ VehicleSighting, … ]      // §6.5 shape
}
```

`fuzzy=true` adds a `candidates` array with `match_state` never above `probable`, plus a `distance` field. Exact and fuzzy results are **never merged into one undifferentiated list**.

### 6.3 Journey

```
GET /api/v1/journey/{plate_normalized}?from=…&to=…

200 {
  "plate": "GJ01AB1234",
  "disclaimer": "Observed movement sequence. Not a confirmed route.",
  "sighting_count": 4,
  "sightings": [ VehicleSighting sorted by first_seen_at ASC ],
  "segments": [
    {
      "from_camera_id": "cam04", "to_camera_id": "cam14",
      "from_time": "2026-09-01T10:03:21Z", "to_time": "2026-09-01T10:12:04Z",
      "straight_line_km": 3.2,
      "elapsed_seconds": 523,
      "required_speed_kmh": 22.0,
      "feasible": true,
      "note": null
    }
  ]
}
```

`segments[].feasible=false` carries `note: "required speed 15000 km/h exceeds plausibility ceiling"`. The `disclaimer` field is **mandatory in the response body**, not merely a UI convention — that way no client can render a journey without it.

### 6.4 Cameras, watchlist, alerts, health

```
GET  /api/v1/cameras                    → [{camera_id, name, lat, lon, status, last_seen_at}]
GET  /api/v1/cameras/{camera_id}        → detail + active session + recent sighting count
POST /api/v1/cameras/sync               → refresh from catalogue (§8.2) → {added, updated, missing}

GET  /api/v1/watchlist                  → active entries
POST /api/v1/watchlist                  → {plate, reason, priority}
DELETE /api/v1/watchlist/{id}           → soft delete (active=false)

GET  /api/v1/alerts?limit=50&acknowledged=false
POST /api/v1/alerts/{id}/acknowledge

GET  /health/live                       → {"status":"ok"}
GET  /health/ready                      → {"status":"ok","postgres":true,"redis":true,
                                            "cameras_registered":30,"source_mode":"file"}
GET  /api/v1/system/status              → the SystemStatus object the UI header renders
GET  /api/v1/metrics/benchmark          → last frozen benchmark summary (§7)
```

### 6.5 `VehicleSighting` — the wire shape consumed by the frontend

```typescript
// frontend/src/types/api.ts
export type SourceMode  = "live_rtsp" | "live_hls" | "file" | "frames" | "synthetic";
export type MatchState  = "exact" | "probable" | "low_confidence" | "unreadable";

export interface VehicleSighting {
  sighting_id: string;
  camera_id: string;              // "cam04" — §1.1
  camera_name: string;
  lat: number | null;             // null until the camera is surveyed
  lon: number | null;
  first_seen_at: string;          // ISO-8601 Z  (NOT "timestamp")
  last_seen_at: string;
  source_pts_ms: number | null;
  source_mode: SourceMode;        // drives the LIVE/REPLAY badge
  plate: string | null;           // normalized; null = unreadable
  plate_raw: string | null;
  confidence: number | null;
  match_state: MatchState;
  evidence_count: number;
  plate_width_px: number | null;
  vehicle_type: string | null;
  snapshot_uri: string | null;    // may be absent — render a graceful placeholder
}

export interface JourneySegment {
  from_camera_id: string; to_camera_id: string;
  from_time: string; to_time: string;
  straight_line_km: number; elapsed_seconds: number;
  required_speed_kmh: number; feasible: boolean; note: string | null;
}

export interface JourneyResponse {
  plate: string;
  disclaimer: string;             // render this. always.
  sighting_count: number;
  sightings: VehicleSighting[];
  segments: JourneySegment[];
}

export interface Alert {
  alert_id: string; plate: string; camera_id: string; camera_name: string;
  match_state: Exclude<MatchState, "low_confidence" | "unreadable">;
  confidence: number | null; priority: "low"|"medium"|"high"|"critical";
  created_at: string; acknowledged: boolean; snapshot_uri: string | null;
}

export interface SystemStatus {
  api: "ok" | "degraded";
  postgres: boolean;
  redis: boolean;
  source_mode: SourceMode;
  cameras_total: number;
  cameras_live: number;
  cameras_replay: number;
  is_live: boolean;               // false ⇒ UI must show REPLAY, never "ONLINE"
}
```

### 6.6 Error envelope — LOCKED

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "observed_at must be timezone-aware ISO-8601",
    "field": "observed_at",
    "request_id": "req-7f2a1c",
    "details": {}
  }
}
```

Codes: `VALIDATION_FAILED` · `UNKNOWN_CAMERA` · `UNSUPPORTED_SCHEMA_VERSION` · `DUPLICATE_EVENT` · `NOT_FOUND` · `DEPENDENCY_UNAVAILABLE` · `INTERNAL_ERROR`. Every response carries `request_id`, and the same value appears in the server log line — that is the entire debugging story during a live demo.

### 6.7 WebSocket

```
WS /ws/alerts

server → client:
{"type":"alert","data": Alert }
{"type":"sighting","data": VehicleSighting }
{"type":"system","data": SystemStatus }
{"type":"heartbeat","ts":"2026-09-01T10:03:21Z"}
```

Client rules: reconnect with exponential backoff + jitter; on reconnect **refetch from REST** rather than trusting accumulated socket state; deduplicate by `alert_id`; preserve server timestamps (never stamp arrival time as event time). The socket is a notification channel — the database is the truth.

### 6.8 API versioning

`/api/v1` changes **additively only**. Adding an optional field is fine. Removing a field, renaming a field, or changing a type requires `/api/v2`. Parth's types are generated from this contract, and a silent rename breaks the build at the worst possible moment.

---

## 7. Metrics contract

### 7.1 Primary metric — LOCKED

```
E2E correct-plate event rate  =  correct final plate events / eligible vehicle events
```

An *eligible vehicle event* is a vehicle whose plate is human-readable in at least one sampled frame of the ground-truth clip. A *correct final plate event* is one whose fused `normalized` string exactly equals ground truth.

Everything else — detector mAP, OCR CER, FPS, p95 latency, VRAM — is a **diagnostic**, used to explain the primary number, never reported as the headline.

### 7.2 Mandatory width-bucket breakdown

No accuracy number may be reported as a single average.

| Bucket | Plate pixel width |
|---|---|
| B1 | > 100 px |
| B2 | 80 – 100 px |
| B3 | 60 – 80 px |
| B4 | 40 – 60 px |
| B5 | 30 – 40 px |
| B6 | < 30 px |

Reason: a headline "92% accurate" routinely decomposes into 98% above 80 px and 51% below 40 px. The second number decides whether the system works on real CCTV, and it is the number a judge will probe.

Difficulty tiers used by the synthetic generator: Easy > 100 px · Medium 60–100 · Hard 30–60 · Extreme < 30.

### 7.3 Benchmark report — LOCKED shape

```json
{
  "run_id": "detector_v1_001",
  "task": "vehicle_detection | plate_detection | ocr | temporal_fusion | e2e",
  "dataset_manifest_sha256": "…",
  "git_commit": "…",
  "weights_sha256": "…",
  "machine": "RTX 5070 Ti 12GB",
  "runtime": "torch 2.x + CUDA 12.x",
  "source_mode": "file",
  "e2e_correct_plate_event_rate": 0.0,
  "by_plate_width": {
    ">100": null, "80-100": null, "60-80": null,
    "40-60": null, "30-40": null, "<30": null
  },
  "by_condition": {"day": null, "night": null, "blur": null, "glare": null, "angle": null},
  "diagnostics": {
    "precision": null, "recall": null, "map50": null, "small_plate_recall": null,
    "ocr_exact_accuracy": null, "cer": null,
    "fps": null, "latency_p50_ms": null, "latency_p95_ms": null,
    "vram_peak_mb": null, "real_time_factor": null
  },
  "notes": []
}
```

Written append-only to `benchmark/reports/<task>_<run_id>.json`. Leaderboard rows accumulate in `benchmark/TRINETRA_MODEL_LEADERBOARD.csv`.

### 7.4 TRINETRA-HARD — frozen benchmark

~1,000 labeled observations. **Never trained on. Never tuned on. Rebalanced exactly once, before final freeze.** Source groups disjoint from all training and validation data.

| Bucket | Target count |
|---|---|
| Large / easy | 200 |
| Blur / motion | 200 |
| Night / low light | 200 |
| Glare / exposure | 150 |
| Angle / perspective | 150 |
| Tiny plate | 100 |

Owner: Akshat. Consumed by: Manas (model selection), everyone (the accuracy claim on the submission).

### 7.5 Accuracy vs latency decision table

| Condition | Decision |
|---|---|
| Higher accuracy, similar latency | Prefer higher accuracy |
| Small accuracy gain, large latency cost | Prefer lower latency |
| Large accuracy gain, modest latency cost | Prefer higher accuracy |
| No measurable gain | Do not adopt |
| Improves only the synthetic test | Do not treat as a win |

Worked example: **A** = mAP 96% @ 12 FPS vs **B** = mAP 94% @ 40 FPS → **B wins.** This is a multi-camera system; throughput per GPU is the product constraint, not peak single-stream accuracy.

---

## 8. Configuration and environment

### 8.1 `.env.example` — commit this; never commit `.env`

```bash
APP_ENV=dev
API_PORT=8000
DATABASE_URL=postgresql+psycopg://trinetra:localdev@localhost:5432/trinetra
REDIS_URL=redis://localhost:6379/0

SOURCE_MODE=file                 # file | frames | synthetic | live_rtsp | live_hls
CAMERA_CATALOGUE_URL=https://cctv.corp8.cloud/cameras.json
RTSP_BASE=rtsp://103.250.160.189:8554/stream
RTSP_TRANSPORT=tcp
TARGET_INFERENCE_INTERVAL_MS=100

MODEL_DIR=./artifacts/models
SNAPSHOT_DIR=./artifacts/snapshots
REPLAY_MANIFEST=./data/replay/manifest.json

# Frontend (Vite) — PUBLIC. Anything here ships to the browser.
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_APP_MODE=auto               # mock | api | demo | auto

# NEVER put here, never commit: Sentinel HLS password, any credential.
# The HLS host is password-protected → proxy it through the backend (§8.3).
```

`VITE_`-prefixed variables are compiled into the browser bundle. A secret placed there is public. This is not a style preference.

### 8.2 Camera catalogue sync

```bash
python scripts/sync_cameras.py --url "$CAMERA_CATALOGUE_URL"
```

- Upsert on `external_camera_id`. **Never** hardcode `cam01…cam30`.
- Refresh periodically (every N minutes), not per frame.
- A camera absent from the catalogue → `status='offline'`. **Never delete** — sightings reference it.
- Credentials come from the environment and are never written into `cameras.rtsp_url`/`hls_url` in a form the API would return.

### 8.3 HLS preview proxy

The Sentinel HLS host requires a password. The browser must never hold it.

```
GET /api/v1/cameras/{camera_id}/preview.m3u8   → backend fetches upstream with credentials,
                                                  rewrites segment URLs, streams to client
```

Frontend plays that backend URL with `hls.js`. Browsers cannot play RTSP at all — this proxy is the only browser preview path.

### 8.4 Config layering

```
config/base.yaml          shared defaults
config/offline.yaml       SOURCE_MODE=file, replay manifest, deterministic seeds
config/live.yaml          SOURCE_MODE=live_rtsp, TCP transport, reconnect policy
config/benchmark.yaml     frozen manifests, batch sweeps allowed, no live sources
config/training.yaml      dataset manifests, augmentation, A100 params
```

Validated at startup by `python scripts/validate_config.py`. **Switching between offline and live must change configuration only** — never code. That property is the single acceptance test for source independence.

---

## 9. Test fixtures — version-controlled, exact filenames

```
tests/fixtures/
├── ai_event_high_confidence.json      exact watchlist match, 4 agreeing frames
├── ai_event_low_confidence.json       1 observation, confidence 0.51, low_confidence
├── ai_event_unreadable.json           plate: null
├── ai_event_duplicate.json            same event_id as high_confidence → "duplicate"
├── ai_event_bad_timestamp.json        naive datetime → 422 VALIDATION_FAILED
├── ai_event_unknown_camera.json       camera_id "cam99" → 422 UNKNOWN_CAMERA
├── camera_reconnect.json              two events, same camera+track_id, DIFFERENT session
├── scene_discontinuity.json           PTS jump backwards → new session expected
├── journey_four_cameras.json          ordered multi-camera sequence
├── journey_implausible.json           15,000 km/h segment → feasible:false
├── watchlist_match.json               watchlist seed + matching event
└── search_response.json               canonical GET /search/vehicles body for Parth's mocks
```

`camera_reconnect.json` is the fixture that proves §1.2. It must produce **two** rows in `vehicle_tracks`, not one. If it produces one, `uq_trackkey` is wrong.

```bash
python scripts/generate_demo_events.py \
  --scenario sentinel_vehicle_journey_v1 --seed 42 --count 100 --output tests/fixtures
```

Fixtures validate **contracts**. They never justify an accuracy claim.

---

## 10. Repository contract

```
trinetra/
├── ai/
│   ├── contracts/          # Manas — FrameEnvelope, EventEnvelope. Versioned.
│   ├── media/              # Manas — the five adapters, nothing else
│   ├── detect/  track/  plate/  ocr/  fusion/  quality/
│   └── worker.py
├── backend/app/
│   ├── schemas/            # Mihir — MUST mirror §3 exactly
│   ├── models/  api/v1/  services/  db/
│   └── main.py
├── frontend/src/
│   ├── types/api.ts        # Parth — MUST mirror §6.5 exactly
│   └── api/  components/  pages/  hooks/
├── datasets/
│   ├── LICENSES.md         # Akshat — mandatory register
│   ├── manifests/          # Akshat — frozen for final benchmark
│   └── synthetic/
├── benchmark/
│   ├── manifests/  reports/  TRINETRA_MODEL_LEADERBOARD.csv
├── regression/             # tiny_plate/ night/ blur/ glare/ occlusion/ ocr_confusions/
├── data/replay/            # cam04/ cam10/ cam14/ manifest.json
├── config/                 # base|offline|live|benchmark|training .yaml
├── scripts/
├── tests/fixtures/
├── alembic/
├── docker-compose.yml
└── .env.example
```

| Path | Owner | Change rule |
|---|---|---|
| `ai/contracts/` | Manas | Versioned; breaking change needs all-owner sign-off |
| `ai/media/` | Manas | Adapters only — no business logic |
| `ai/fusion/` | Manas | Pure aggregation — no I/O |
| `backend/app/schemas/` | Mihir | Must mirror §3 |
| `frontend/src/types/` | Parth | Must mirror §6.5 |
| `datasets/manifests/` | Akshat | Frozen before final benchmark |
| `benchmark/reports/` | Manas + Akshat | Append-only evidence |

### 10.1 Branch integration order

Merge in this order to avoid deadlock: Mihir API skeleton + health → Parth UI shell + mocks → Akshat manifests + fixtures → Manas media adapter + baseline AI → AI event ingestion → search/journey → realtime alerts → demo resilience.

### 10.2 Startup order

```
1 PostgreSQL/PostGIS   2 Redis   3 FastAPI   4 AI workers   5 Frontend
Gate: GET /health/ready must report postgres+redis true before step 4.
```

### 10.3 Component boundaries — one owner each

| Component | Owner |
|---|---|
| Media adapters | Manas |
| Detection + tracking | Manas |
| Plate detection + OCR | Manas |
| Temporal fusion + normalization | Manas |
| Event contract | Manas |
| Ingest + persistence + dedup | Mihir |
| Search + journey + watchlist API | Mihir |
| Realtime (Redis + WS) | Mihir |
| Datasets + manifests + licences | Akshat |
| Synthetic generation + TRINETRA-HARD | Akshat |
| Benchmark execution + reports | Akshat (+ Manas) |
| UI, GIS, demo mode | Parth |

---

## 11. Licensing register — mandatory fields

`datasets/LICENSES.md` must record, per asset: name · official source URL · access date · exact declared license · provenance (and original dataset if derived) · modifications made · reuse permitted for our purpose (Y/N) · redistribution permitted (Y/N) · attribution text · use in project · status `APPROVED` / `RESTRICTED` / `EXCLUDED` / `PENDING`.

**Free to download ≠ safe to reuse.** A repository declaring `CC BY 4.0` does not prove every source image was licensed correctly by the uploader; record the provenance chain, not just the badge.

| Asset | License | Status | Role — and what it must NOT be used for |
|---|---|---|---|
| thundarstrom Indian plates (3,742 img) | CC BY 4.0 | CORE after record | Indian plate localization. Not proof of OCR grammar. |
| justjuu plate detection (8,823: 6,176/1,765/882) | CC BY 4.0 | CORE after record | Plate detector training set. |
| justjuu `rtdetr-v2-license-plate-detection` | Apache-2.0 | CORE BASELINE | Zero-training plate baseline. Its 0.97 mAP is on *its* test set. |
| FANVID (1,463 clips @180×320, 49 plates, 31,096 boxes) | CC BY 4.0 | CORE BENCHMARK | **Evaluation only.** Training on it destroys its value. |
| CCPD (300k+) | MIT | AUXILIARY | Localization/geometry pretraining **only**. Never Indian OCR grammar — its plate grammar is Chinese. |
| Own synthetic Indian CCTV corpus | Ours | CORE | OCR pretraining + controlled degradation. Not evidence of field accuracy. |
| TRINETRA-HARD | Ours | CORE FROZEN TEST | Final evaluation. **Never training or tuning.** |
| thirdeyelabs Indian road (210 GB) | verify | OPTIONAL | Sample 10–30k only, and only after a measured vehicle-detection failure. |
| Open Images V7 | annots CC BY 4.0 / images CC BY 2.0 | OPTIONAL | Verify terms per image. |
| Gamester03 (1,709 rows) | **unknown** | **PENDING — DO NOT TRAIN** | Blocked until provenance and exact terms are verified. |
| DataCluster sample | CC BY-**NC-ND** | **EXCLUDED** | NC-ND forbids ML training and derivatives. |
| UFPR-ALPR | academic / non-commercial | **EXCLUDED** | Restriction incompatible with the core policy. |
| VeRi-776 / CityFlow / VehicleID | agreement-bound | DEFERRED | Final-stage Re-ID only, after licensing review. |
| BDD100K | repo BSD-3, data terms differ | RESEARCH ONLY | Do not ship weights trained on it without a terms review. |
| Ultralytics YOLO | **AGPL-3.0** | BENCHMARK COMPARISON ONLY | Never shipped. AGPL network-use obligations. |
| RF-DETR (**Roboflow**) Nano–Large | **Apache-2.0** | CORE | Plus/XL/2XL are differently licensed → **excluded**. |
| RTMDet / YOLOX | Apache-2.0 | FALLBACK | Available if RF-DETR underperforms on our hardware. |
| ByteTrack | MIT | CORE | |
| PaddleOCR | Apache-2.0 | CORE | |
| PostGIS / OSM tiles | open | CORE | No paid map or geocoding service. |

**RF-DETR is Roboflow's, Apache-2.0. It is not an Ultralytics model.** Attributing it to Ultralytics reintroduces exactly the AGPL exposure the stack was designed to avoid.

---

## 12. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-09-01 | Original per-manual schemas (four incompatible variants). Superseded. |
| **1.1** | **2026-09-01** | **Consolidation.** `timestamp` → `observed_at`. Added `stream_session_id`, `source_pts_ms`, `source_mode`, `match_state`, `plate_width_px`, `evidence_count`, `model.*`. `camera_id` format locked to Sentinel IDs (`cam04`). Added `stream_sessions` and `vehicle_sightings` tables; added `uq_trackkey`; added `plate_width_px`/`source_pts_ms`/`match_state` columns. Locked dedup key, error envelope, `SystemStatus`, journey `disclaimer`. Corrected RF-DETR attribution to Roboflow. |

### 12.1 Migration note for anyone holding v1.0 code

```
"timestamp"                 → "observed_at"
"CAM_001" / "CAM-001"       → "cam04"        (Sentinel ID, lowercase)
"schema_version": "1.0"     → "1.1"
add: stream_session_id, source_pts_ms, source_mode,
     plate.match_state, plate.plate_width_px, plate.evidence_count, model.*
DB:  + stream_sessions, + vehicle_sightings,
     vehicle_tracks     + stream_session_id + UNIQUE(camera_id, stream_session_id, track_id)
     plate_observations + source_pts_ms + plate_width_px
```

---

## 13. Forbidden constructs — with the reason

| Do not | Because |
|---|---|
| `while True: cap = cv2.VideoCapture(url)` | Hot reconnect loop; hammers the camera, no backoff, no session reset. |
| Key anything on `(camera_id, track_id)` | Merges vehicles across reconnects. §1.2. |
| `CAP_PROP_FPS` or arrival time as video time | Both lie on RTSP. §2.1. |
| Unbounded frame queue | Converts a small deficit into minutes of latency, then OOM. §2.2. |
| Run OCR on every frame | Wastes the GPU budget multi-camera scale needs. §4.1. |
| Super-resolution by default | Hallucinates characters: `GJ01AB1234` → `GJ01A81234`. Benchmark before adopting. |
| Multiply uncalibrated confidences | The product is meaningless. §4.4. |
| Hardcode `cam01…cam30` | Catalogue is dynamic. §8.2. |
| Publish before commit | Shows operators alerts that don't exist. §5.9. |
| Secrets in `VITE_` vars | They compile into the public bundle. §8.1. |
| Present replay as live | Falsifies performance. §7. |
| Call a journey "the route" | It is an observed sighting sequence. §6.3. |
| Train on TRINETRA-HARD or FANVID | Destroys the only unbiased evidence you have. §7.4. |
| Ship Ultralytics YOLO | AGPL-3.0. Benchmark comparison only. §11. |
| `postgis/postgis:latest` | Moves under you mid-build. Pin `16-3.4`. §5. |
