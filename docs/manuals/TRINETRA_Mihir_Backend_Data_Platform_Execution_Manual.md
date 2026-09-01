# TRINETRA — EXECUTION MANUAL
## Mihir — Backend & Data Platform

**Version 2.0 · 2026-09-01 · 6 build days to the qualification gate**

> **Precedence.** [`docs/TRINETRA_Canonical_Contracts.md`](docs/TRINETRA_Canonical_Contracts.md) is normative for every schema. Blocks marked `COPIED FROM CANONICAL — DO NOT EDIT HERE` are reproduced verbatim; to change one, change it there and tell the other three.

---

## 1. Your job in one paragraph

You are the contract between Manas and Parth. Manas produces events at an unpredictable rate from an unreliable source; Parth needs sub-200 ms search over whatever has arrived so far. You own the schema that makes both possible, the API that neither of them should have to think about, and the property that the system loses nothing when a dependency dies. You are also the only person who can make the demo fail in a way nobody can work around — if the schema is wrong, everything above it is wrong.

**You own:** `backend/app/models/` · `backend/app/api/` · `backend/app/services/` · `backend/alembic/` · `docker-compose.yml` · the DDL · `/health`.

**You do not own:** the AI pipeline, the UI, the datasets. You define the boundary and enforce it with HTTP status codes.

---

## 2. Day plan — anchored to the real calendar

Today is **1 September 2026**. Submission is **7 September**. Hackathon is **10–11 September**. Six build days. Any 10-day or 14-day plan in an older document is superseded.

| Date | Day | You must finish | Proof |
|---|---|---|---|
| **Sep 1** | D1 | `docker-compose.yml`, PostGIS up, Alembic initialized, **all 8 tables migrated**, `/health` | `alembic upgrade head` clean from empty DB; PostGIS 3.4 confirmed |
| **Sep 2** | D2 | `POST /api/v1/ingest/events` with validation, idempotency, `stream_sessions` | All 12 fixtures accepted or correctly rejected (**G1**) |
| **Sep 3** | D3 | Search endpoints + indexes + `EXPLAIN ANALYZE` | Index scans, no sequential scans; <200 ms on 10k rows |
| **Sep 4** | D4 | Journey, watchlist, alerts, Redis pub/sub, WebSocket | Real event from Manas → alert → WS push (**G3 + G4**) |
| **Sep 5** | D5 | HLS proxy, degraded mode, fault injection, load smoke | Every dependency killable without data loss |
| **Sep 6** | D6 | Seed data, `pg_dump`, reset script, demo dataset frozen | One command restores a full demo DB |
| Sep 7 | — | **SUBMIT** | — |
| Sep 8–9 | D7–D8 | Live swap support, freeze | Config-only swap |

**D1 is the highest-leverage day.** Manas cannot POST and Parth cannot integrate until your schema exists. Get all eight tables migrated on day one even if the endpoints are stubs.

---

## 3. Stack — pinned, no substitutions

| Component | Version | Note |
|---|---|---|
| PostgreSQL + PostGIS | **`postgis/postgis:16-3.4`** | Pinned. See §3.1 |
| Redis | `redis:7-alpine` | Pub/sub only — not a datastore |
| FastAPI | latest | |
| SQLAlchemy 2.x | latest | Typed ORM style |
| Alembic | latest | Every schema change is a migration |
| GeoAlchemy2 | latest | PostGIS types |
| psycopg | `psycopg[binary]` | v3 |

```bash
pip install fastapi "uvicorn[standard]" pydantic-settings sqlalchemy \
            "psycopg[binary]" alembic geoalchemy2 redis httpx pytest pytest-asyncio
```

### 3.1 Why the image tag is pinned

`postgis/postgis:latest` is forbidden. Two developers who pull a week apart get different PostGIS versions, and a PostGIS minor upgrade can change GIST index behaviour. The failure mode is the worst kind: your machine works, someone else's doesn't, and the difference is invisible in the repo. On demo day the machine that matters may not be yours.

`GEOGRAPHY(Point,4326)` — not `GEOMETRY`. `GEOGRAPHY` gives correct metre distances on a sphere. `GEOMETRY` with lat/lon computes distance in **degrees**, which is meaningless and, worse, *looks* like it works: sorting is roughly right at Gujarat's latitude, so the bug survives casual testing and surfaces when a judge asks "how far apart are those cameras?"

---

## 4. The schema — write this on D1

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §5)

Eight tables. If you have a four-table version in your head from an earlier draft, that version cannot store a session, a plate width, a source PTS, or a match state — which means it cannot store what Manas emits or serve what Parth renders.

### 4.1 cameras

```sql
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE cameras (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_camera_id   TEXT NOT NULL UNIQUE,     -- 'cam04' — Sentinel ID, verbatim
    name                 TEXT NOT NULL,
    location             GEOGRAPHY(Point, 4326) NOT NULL,
    district             TEXT,
    stream_url_rtsp      TEXT,
    stream_url_hls       TEXT,
    is_active            BOOLEAN NOT NULL DEFAULT TRUE,
    health_state         TEXT NOT NULL DEFAULT 'unknown',   -- unknown|healthy|degraded|offline
    last_seen_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_cameras_location ON cameras USING GIST (location);
CREATE INDEX idx_cameras_external ON cameras (external_camera_id);
```

**Two IDs, deliberately.** `id` is your internal UUID FK target. `external_camera_id` is Sentinel's string, stored **verbatim lowercase** (`cam01`…`cam30`). Never rewrite it to `CAM_001`, `CAM-001`, `Cam04`, or `cam4` — the catalogue is the authority, and a "prettier" internal format means every join needs a translation layer that will be wrong exactly once, at the worst time.

### 4.2 stream_sessions — new, and load-bearing

```sql
CREATE TABLE stream_sessions (
    id               UUID PRIMARY KEY,             -- minted by the AI worker
    camera_id        UUID NOT NULL REFERENCES cameras(id),
    source_mode      TEXT NOT NULL,                -- live_rtsp|live_hls|file|frames|synthetic
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ,
    end_reason       TEXT,                         -- eof|error|discontinuity|manual|timeout
    pts_reliable     BOOLEAN NOT NULL DEFAULT TRUE,
    frames_read      BIGINT NOT NULL DEFAULT 0,
    frames_dropped   BIGINT NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sessions_camera_started ON stream_sessions (camera_id, started_at DESC);
```

The `id` is generated upstream, not by you — Manas mints it at connect time and every event in that connection carries it. `frames_dropped` and `pts_reliable` are how you prove or disqualify a performance claim after the fact.

### 4.3 vehicle_tracks — note the unique constraint

```sql
CREATE TABLE vehicle_tracks (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id          UUID NOT NULL REFERENCES cameras(id),
    stream_session_id  UUID NOT NULL REFERENCES stream_sessions(id),
    track_id           INTEGER NOT NULL,
    started_at         TIMESTAMPTZ NOT NULL,
    ended_at           TIMESTAMPTZ,
    vehicle_type       TEXT,
    vehicle_color      TEXT,
    CONSTRAINT uq_trackkey UNIQUE (camera_id, stream_session_id, track_id)
);
CREATE INDEX idx_tracks_session ON vehicle_tracks (stream_session_id);
```

**This constraint is the single most important line in the schema.**

`track_id` is an integer that ByteTrack restarts from 1 on every reconnect. Without `stream_session_id` in the key, the vehicle that was track 7 before a network blip and the unrelated vehicle that becomes track 7 after it collapse into one row. The visible symptom is a journey showing a car crossing Ahmedabad in four seconds — but you will only see that if you happen to look. Meanwhile every count, every journey, and every accuracy number is quietly wrong.

A `UNIQUE` constraint here means the database refuses the merge even if application code has a bug. That is the point: correctness enforced at the storage layer, not by remembering.

### 4.4 plate_observations

```sql
CREATE TABLE plate_observations (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_track_id   UUID NOT NULL REFERENCES vehicle_tracks(id) ON DELETE CASCADE,
    plate_raw          TEXT,
    plate_normalized   TEXT,
    ocr_confidence     REAL,
    image_quality      REAL,
    plate_width_px     INTEGER,
    source_pts_ms      BIGINT,
    frame_index        INTEGER,
    observed_at        TIMESTAMPTZ NOT NULL,
    snapshot_uri       TEXT
);
CREATE INDEX idx_obs_track ON plate_observations (vehicle_track_id);
CREATE INDEX idx_obs_normalized ON plate_observations (plate_normalized)
    WHERE plate_normalized IS NOT NULL;
```

Per-frame evidence. Both `plate_raw` and `plate_normalized`: raw is the audit trail, normalized is the search key.

`plate_width_px` is required for Akshat's width-bucket reports — without it there is no way to answer "how does accuracy degrade with plate size?", which is the question that actually determines whether this works on real infrastructure. `source_pts_ms` + `frame_index` make any observation reproducible from the source clip.

The partial index skips the nulls, which will be a large fraction of rows.

### 4.5 vehicle_sightings — what Parth actually reads

```sql
CREATE TABLE vehicle_sightings (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id          UUID NOT NULL REFERENCES cameras(id),
    vehicle_track_id   UUID REFERENCES vehicle_tracks(id) ON DELETE SET NULL,
    plate_normalized   TEXT,
    plate_raw          TEXT,
    vehicle_type       TEXT,
    confidence         REAL,
    match_state        TEXT NOT NULL,       -- exact|probable|low_confidence|unreadable
    evidence_count     INTEGER NOT NULL DEFAULT 1,
    plate_width_px     INTEGER,
    image_quality      REAL,
    source_mode        TEXT NOT NULL,
    first_seen_at      TIMESTAMPTZ NOT NULL,
    last_seen_at       TIMESTAMPTZ NOT NULL,
    location           GEOGRAPHY(Point, 4326) NOT NULL,
    snapshot_uri       TEXT,
    dedupe_key         TEXT NOT NULL UNIQUE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sight_plate_time ON vehicle_sightings (plate_normalized, first_seen_at DESC);
CREATE INDEX idx_sight_time       ON vehicle_sightings (first_seen_at DESC);
CREATE INDEX idx_sight_location   ON vehicle_sightings USING GIST (location);
CREATE INDEX idx_sight_camera     ON vehicle_sightings (camera_id, first_seen_at DESC);
```

One fused, deduplicated row per vehicle-per-camera-pass. This is the read model: search, journey, and map all hit this table and nothing else. Never make Parth join four tables to draw a marker.

`location` is denormalized from `cameras` on purpose. It costs 32 bytes and removes a join from the hottest query in the system.

`first_seen_at` / `last_seen_at` — **not** `timestamp`. A sighting is an interval, not an instant; a vehicle is visible for two or three seconds. Parth's `VehicleSighting` interface uses `first_seen_at`, so these names must match exactly.

`dedupe_key UNIQUE` is your idempotency guarantee. Manas retries POSTs on network failure with a stable `event_id`; the unique index turns "did this arrive twice?" from application logic into a database fact.

### 4.6 watchlist

```sql
CREATE TABLE watchlist (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_normalized  TEXT NOT NULL,
    reason            TEXT,
    severity          TEXT NOT NULL DEFAULT 'medium',   -- low|medium|high|critical
    added_by          TEXT,
    is_active         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX uq_watchlist_active ON watchlist (plate_normalized)
    WHERE is_active;
```

Store the normalized form only. A watchlist entry typed as `GJ 01 AB 1234` that never matches `GJ01AB1234` is a silent failure of the system's core promise — normalize on write, in the API layer, before insert.

### 4.7 alerts

```sql
CREATE TABLE alerts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    watchlist_id        UUID NOT NULL REFERENCES watchlist(id),
    vehicle_sighting_id UUID NOT NULL REFERENCES vehicle_sightings(id) ON DELETE CASCADE,
    match_state         TEXT NOT NULL,
    confidence          REAL,
    acknowledged_at     TIMESTAMPTZ,
    acknowledged_by     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_alert_state CHECK (match_state IN ('exact', 'probable')),
    CONSTRAINT uq_alert_once UNIQUE (watchlist_id, vehicle_sighting_id)
);
CREATE INDEX idx_alerts_created ON alerts (created_at DESC);
```

Two constraints doing real work:

`chk_alert_state` makes it **structurally impossible** to raise an alert from a `low_confidence` reading. Fuzzy matching produces candidates for human review; it does not produce alerts. Enforcing that in the database rather than in a code path means it cannot be bypassed by a hurried change on D6.

`uq_alert_once` prevents alert storms. Without it, a re-ingested event fires a fresh alert, and the operator's screen fills with duplicates of the same match — which is how real operators learn to ignore alerts.

### 4.8 ingestion_events

```sql
CREATE TABLE ingestion_events (
    id            BIGSERIAL PRIMARY KEY,
    event_id      UUID,
    camera_id     TEXT,
    outcome       TEXT NOT NULL,     -- accepted|duplicate|rejected
    error_code    TEXT,
    error_detail  TEXT,
    received_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_ingest_received ON ingestion_events (received_at DESC);
```

Your audit log. When Manas says "I sent 400 events" and the map shows 340, this table tells you which 60 were rejected and why — in seconds instead of an afternoon. Log every outcome including the successes.

### 4.9 Transaction boundary — "persist before publish"

```python
async with session.begin():
    track   = await upsert_track(...)          # ON CONFLICT (uq_trackkey) DO UPDATE
    obs     = await insert_observations(...)
    sighting = await upsert_sighting(...)      # ON CONFLICT (dedupe_key) DO NOTHING
    alert   = await check_watchlist(sighting)
# transaction committed HERE
await redis.publish("alerts", payload)         # only now
```

Publish **after** commit, never inside the transaction.

If you publish first and the commit then fails, Parth's UI shows an alert for a sighting that does not exist. The operator clicks it, gets a 404, and stops trusting the system — and you cannot recover, because the notification is already gone. A dropped notification is a UI refresh away from being fixed. A phantom alert is not fixable at all.

Redis is pub/sub only. It stores nothing that matters. Killing Redis must degrade live push, never lose data.

---

## 5. API — `/api/v1`

### 5.1 Ingest

```
POST /api/v1/ingest/events
```

Body: `EventEnvelope` v1.1 (Contracts §3). Responses:

| Status | Body | When |
|---|---|---|
| 201 | `{"status":"accepted","sighting_id":"…"}` | New |
| 200 | `{"status":"duplicate","sighting_id":"…"}` | `dedupe_key` exists |
| 422 | error envelope | Validation failed |
| 503 | error envelope | Postgres unavailable |

Validation order — cheapest first:

1. `schema_version == "1.1"` → else `422 SCHEMA_VERSION_UNSUPPORTED`
2. `camera_id` resolves in `cameras` → else `422 UNKNOWN_CAMERA`
3. `observed_at` is timezone-aware → else `422 VALIDATION_FAILED`
4. `match_state` in the four allowed values → else `422 VALIDATION_FAILED`
5. `plate.normalized` matches `^[A-Z0-9]+$` when present
6. `bbox_xyxy` has 4 ints, `x2 > x1`, `y2 > y1`

**Reject naive timestamps.** A datetime without an offset is ambiguous by up to a day, and the ambiguity surfaces as a journey with events out of order. `2026-09-01T10:03:21` → 422. `2026-09-01T10:03:21.234Z` → accepted.

Do not silently coerce bad input. A 422 with a field name costs Manas thirty seconds; a silent coercion costs a day of debugging the wrong component.

### 5.2 Search

```
GET /api/v1/search/vehicles?plate=GJ01AB1234&from=…&to=…&camera_id=cam04
                           &vehicle_type=car&min_confidence=0.8&limit=50&offset=0
GET /api/v1/search/vehicles/nearby?lat=23.02&lon=72.57&radius_m=2000&from=…&to=…
```

Normalize the inbound `plate` with the **same** function Manas uses before querying — otherwise `GJ 01 AB 1234` typed by an operator finds nothing while the data sits right there. Support prefix search (`GJ01AB%`) for partial recall, which is the realistic case: a witness remembers four characters.

Nearby uses `ST_DWithin(location, ST_MakePoint(lon, lat)::geography, radius_m)` — argument order `(lon, lat)`, which is the reverse of how humans say it and a reliable source of "why is my result in the ocean".

### 5.3 Journey — the endpoint with a required disclaimer

```
GET /api/v1/journey?plate=GJ01AB1234&from=…&to=…
```

```json
{
  "plate_normalized": "GJ01AB1234",
  "segments": [
    {
      "from_camera": {"external_camera_id": "cam04", "name": "Ring Road", "lat": 23.02, "lon": 72.57},
      "to_camera":   {"external_camera_id": "cam07", "name": "Ashram Chowk", "lat": 23.05, "lon": 72.60},
      "departed_at": "2026-09-01T10:03:21.234Z",
      "arrived_at":  "2026-09-01T10:11:02.881Z",
      "distance_km": 4.31,
      "required_speed_kmh": 33.6,
      "is_feasible": true,
      "match_state": "probable"
    }
  ],
  "disclaimer": "Observed movement sequence between camera detections. Not a confirmed route."
}
```

The `disclaimer` field is **mandatory and non-null on every response**. It is a field, not a comment, precisely so it cannot be dropped by a frontend refactor — Parth renders it, and the API is what guarantees it is there to render.

The reason: consecutive sightings at cam04 and cam07 tell you the vehicle was at both. They say nothing about the roads between. Cameras cover a fraction of a percent of road-km. Presenting interpolation as a route in a police tool invites a decision based on a line you drew, and that is the one failure mode with consequences outside the demo room.

Feasibility:

```python
required_speed_kmh = haversine_km(a, b) / max(hours_between(a, b), 1e-6)
is_feasible = required_speed_kmh <= 150.0
```

**Flag, never drop.** An infeasible segment usually means an OCR misread — which is exactly what an analyst needs to see. Hiding it hides the error. 150 km/h is a deliberately generous ceiling; the goal is catching impossibilities, not policing traffic.

### 5.4 Remaining endpoints

```
GET    /api/v1/cameras
GET    /api/v1/cameras/{camera_id}
GET    /api/v1/cameras/{camera_id}/preview.m3u8      # HLS proxy — §7
GET    /api/v1/watchlist
POST   /api/v1/watchlist
DELETE /api/v1/watchlist/{id}
GET    /api/v1/alerts?acknowledged=false&limit=50
POST   /api/v1/alerts/{id}/acknowledge
GET    /api/v1/stats/system
GET    /health/live
GET    /health/ready
WS     /ws/alerts
```

`/health/live` = process is up. `/health/ready` = Postgres reachable **and** Alembic at head. Two endpoints because "running" and "usable" are different states, and conflating them means your startup ordering silently breaks.

### 5.5 Error envelope

```json
{"error": {"code": "VALIDATION_FAILED", "message": "observed_at must be timezone-aware", "field": "observed_at"}}
```

| Code | Status |
|---|---|
| `VALIDATION_FAILED` | 422 |
| `UNKNOWN_CAMERA` | 422 |
| `SCHEMA_VERSION_UNSUPPORTED` | 422 |
| `DUPLICATE_EVENT` | 200 |
| `NOT_FOUND` | 404 |
| `DEPENDENCY_UNAVAILABLE` | 503 |
| `INTERNAL_ERROR` | 500 |

Same shape for every failure. Parth writes one error handler instead of seven.

### 5.6 Versioning

`/api/v1` is **additive-only** after D3. New optional field, fine. Renamed field, removed field, changed type, or narrowed enum → new version. Parth's build breaking at 11 pm on Sep 6 because a field got renamed is a self-inflicted wound with no recovery time.

---

## 6. WebSocket

```
WS /ws/alerts
```

Messages:

```ts
{ type: "alert",        payload: Alert }
{ type: "sighting",     payload: VehicleSighting }
{ type: "camera_state", payload: { external_camera_id: string, health_state: string } }
{ type: "heartbeat",    payload: { server_time: string } }
```

Heartbeat every 15 s. Without it, a dead connection behind a NAT is indistinguishable from a quiet night — the UI shows "connected" and nothing arrives, and you find out during the demo.

The **six-step reconnect test**, run it on D5:

1. Connect, confirm alerts arrive
2. Kill the backend
3. UI shows disconnected within 20 s
4. Restart the backend
5. UI reconnects automatically with backoff
6. Alerts flow again **without a page refresh**

A WebSocket that needs a manual refresh has not been implemented; it has been demoed once.

---

## 7. HLS proxy

```
GET /api/v1/cameras/{camera_id}/preview.m3u8
```

The Sentinel HLS endpoints are password-protected. Browsers cannot present those credentials cleanly, and you must not hand them to the frontend — anything in the browser bundle is public, and `VITE_`-prefixed variables are compiled into the JavaScript.

So: the backend holds the credential, fetches upstream, and re-serves the manifest with rewritten segment URLs. Credentials live in `.env`, which is gitignored. `.env.example` carries placeholders only.

Never commit real stream passwords. Never pass them to Parth.

---

## 8. Performance

| Query | Target |
|---|---|
| Exact plate lookup | **< 100 ms** |
| Filtered search | **< 200 ms** |
| Journey reconstruction | **< 300 ms** |

Prove it, don't assume it:

```sql
EXPLAIN ANALYZE
SELECT * FROM vehicle_sightings
WHERE plate_normalized = 'GJ01AB1234'
ORDER BY first_seen_at DESC LIMIT 50;
```

`Index Scan using idx_sight_plate_time` → good. `Seq Scan` → the index is unused; find out why (usually a type mismatch or a function applied to the column) before adding another index.

Load smoke on D5:

```bash
python scripts/seed_synthetic.py --sightings 10000
ab -n 200 -c 10 "http://localhost:8000/api/v1/search/vehicles?plate=GJ01AB1234"
```

10k rows is representative of a 30-camera demo day. Do not tune for a million rows you will never have — that time belongs to correctness.

### 8.1 Degraded mode

| Dependency down | Behaviour | Never |
|---|---|---|
| Redis | Live push stops; ingest and search work | Reject writes |
| Postgres | 503 `DEPENDENCY_UNAVAILABLE`; Manas retries | Lose events silently |
| AI worker | API serves historical data; cameras → `degraded` | Blank the UI |
| Sentinel grid | Offline mode with recorded clips | Stop the demo |

Every one of these must be true on D5, tested by actually killing the container — not by reading the code and believing it.

---

## 9. Camera sync

```bash
curl -s https://cctv.corp8.cloud/cameras.json > data/cameras.json
python scripts/sync_cameras.py --input data/cameras.json
```

`sync_cameras.py` upserts on `external_camera_id`, storing the ID **verbatim**. Coordinates that are missing or `(0,0)` get a documented Ahmedabad placeholder and a `coords_placeholder` flag — a camera at (0,0) sits in the Gulf of Guinea and one such marker discredits the whole map.

Never hardcode `cam01`–`cam30`. The catalogue is the source of truth; a hardcoded list is wrong the moment the organizers add a camera, and you will not be told.

---

## 10. Startup order

```
1. docker compose up -d postgres redis
2. wait for pg_isready
3. alembic upgrade head
4. python scripts/sync_cameras.py
5. uvicorn app.main:app
6. python -m ai.worker            (Manas)
7. npm run dev                    (Parth)
```

The worker starting before migrations produces a cascade of foreign-key errors that read like a bug in Manas's code. `/health/ready` returning 200 is the gate for steps 6 and 7 — publish that fact so nobody guesses.

---

## 11. Fixtures — accept or reject each correctly

Manas delivers these on D1. All twelve must behave as specified by end of D2 (**G1**).

| Fixture | Expected |
|---|---|
| `valid_full_event.json` | 201 |
| `valid_null_plate.json` | 201 — `plate: null` is legal |
| `duplicate_event.json` | 200 `duplicate` |
| `unknown_camera.json` | 422 `UNKNOWN_CAMERA` |
| `naive_timestamp.json` | 422 `VALIDATION_FAILED` |
| `bad_schema_version.json` | 422 `SCHEMA_VERSION_UNSUPPORTED` |
| `invalid_match_state.json` | 422 |
| `negative_bbox.json` | 422 |
| `camera_reconnect.json` | **2 sessions, 2 tracks, not 1** |
| `low_confidence_watchlist_hit.json` | Sighting stored, **no alert** |
| `exact_watchlist_hit.json` | Sighting + alert + WS push |
| `discontinuity.json` | New session |

`camera_reconnect.json` is the test that catches the merge bug. `low_confidence_watchlist_hit.json` is the test that proves the CHECK constraint is doing its job. Neither is optional.

---

## 12. Anti-patterns

| Do not | Consequence |
|---|---|
| `postgis/postgis:latest` | Works here, breaks there, invisible in the repo |
| `GEOMETRY` for lat/lon | Distances in degrees that look plausible |
| Omit `stream_session_id` from the track key | Merged vehicles, fabricated journeys |
| Name it `timestamp` | Parth's interface breaks; interval collapses to instant |
| Publish inside the transaction | Phantom alerts, unrecoverable |
| Treat Redis as a datastore | Data loss on restart |
| Raw SQL string interpolation | Injection; use bound parameters |
| Coerce invalid input silently | Manas debugs the wrong component for a day |
| Alert on `low_confidence` | Alert fatigue; operators learn to ignore alerts |
| Hardcode the camera list | Wrong the moment the catalogue changes |
| Rename a field after D3 | Parth's build breaks with no recovery time |
| Commit `.env` | Credential leak in a public repo |
| Skip Alembic, edit tables by hand | Nobody else can reproduce your database |

---

## 13. Definition of done

- [ ] `docker-compose.yml` with pinned `postgis/postgis:16-3.4` and `redis:7-alpine`
- [ ] All **8** tables migrated via Alembic from an empty database
- [ ] `uq_trackkey UNIQUE (camera_id, stream_session_id, track_id)` present
- [ ] `chk_alert_state CHECK (match_state IN ('exact','probable'))` present
- [ ] `dedupe_key UNIQUE` present and proven idempotent
- [ ] GIST indexes on both `GEOGRAPHY` columns
- [ ] Ingest accepts/rejects all 12 fixtures correctly
- [ ] `plate: null` accepted end to end
- [ ] Persist-before-publish verified by killing Redis mid-ingest
- [ ] Search < 200 ms on 10k rows, proven with `EXPLAIN ANALYZE`
- [ ] Journey returns a non-null `disclaimer` on every response
- [ ] Infeasible segments flagged, never dropped
- [ ] Watchlist normalizes on write
- [ ] Alerts fire once per (watchlist, sighting)
- [ ] WebSocket survives the 6-step reconnect test
- [ ] HLS proxy works with no credentials in the frontend
- [ ] `/health/live` and `/health/ready` distinguish up from usable
- [ ] All four degraded modes verified by killing containers
- [ ] `sync_cameras.py` idempotent, IDs verbatim
- [ ] `pg_dump` demo snapshot + one-command restore
- [ ] `.env.example` complete; `.env` gitignored; no secrets committed

---

## 14. Your daily loop

```bash
docker compose up -d
alembic upgrade head
uvicorn app.main:app --reload
curl http://localhost:8000/health/ready
for f in tests/fixtures/*.json; do
  curl -s -o /dev/null -w "%{http_code} $f\n" -X POST \
    -H 'Content-Type: application/json' --data @"$f" \
    http://localhost:8000/api/v1/ingest/events
done
pytest -q
```

**Final principle:** the AI can be improved after the hackathon. A wrong schema cannot be fixed during it. Get the eight tables and the two constraints right on D1, and everything above you has a chance.
