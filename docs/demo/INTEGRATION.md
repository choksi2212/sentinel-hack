# INTEGRATION — pointing this frontend at a real backend

Ordered. Work top to bottom. Every step says what to type and what proves it
worked. Nothing here has been run against a real backend — it is verified
against the contract and against this app's measured behaviour when the
backend is absent.

---

## 1. Point at the backend, and CONFIRM it took

Edit `.env.local` (create it from RECOVERY.md if missing):

```
VITE_API_BASE_URL=http://localhost:8000     # Mihir's host:port
VITE_WS_BASE_URL=ws://localhost:8000
VITE_APP_MODE=api
```

`npm run dev` auto-restarts on this file — **measured: `[vite] .env.local
changed, restarting server...` then `server restarted.` in under 1s.** Then
reload the browser tab.

**CONFIRM — do not assume.** The app logs one line on boot:

```
[trinetra] mode=api, mock layer OFF, api base http://localhost:8000
```

Three things must all be true, or you are not testing what you think:

| Check | Wrong looks like |
|---|---|
| `mode=api` | `mode=mock` → the edit did not take, reload again |
| `mock layer OFF` | `mock layer ON` → MSW is still answering, everything will "work" and prove nothing |
| **NO `[MSW] Mocking enabled` line follows it** | if present, you are still on mocks |
| `api base` = Mihir's URL | a `:5173`/`:4173` origin → `.env.local` missing (RECOVERY.md §d) |

Also check System status → Build → "API base URL". It shows the compiled-in
value.

## 2. First request

```powershell
curl.exe -i http://localhost:8000/health/live
```

Healthy: `HTTP/1.1 200` and `{"status":"ok"}` (Canonical §6.4). Root-level —
**not** under `/api/v1`.

If this fails, nothing else is worth trying: wrong port, server not up, or
firewall. Fix here before opening the UI.

## 3. Endpoint order — most diagnostic first

Each step isolates one failure class. Do not skip ahead; a later failure is
ambiguous if an earlier one was not proven.

| # | Endpoint | Proves | If it fails |
|---|---|---|---|
| 1 | `GET /health/live` | process up, port right | not the app's problem yet |
| 2 | `GET /health/ready` | postgres + redis up (§6.4 returns `postgres`, `redis`, `cameras_registered`, `source_mode`) | dependencies, not the API |
| 3 | `GET /api/v1/system/status` | the `/api/v1` prefix and routing; drives the LIVE/REPLAY badge | 404 here + 200 on health = prefix/route mismatch |
| 4 | `GET /api/v1/cameras` | first typed collection; exercises `readCamera` | reveals `camera_id` vs `external_camera_id`, `status` vs `health_state` immediately |
| 5 | `GET /api/v1/search/vehicles?plate=…` | the `VehicleSighting` shape — the biggest type in the contract | reveals `plate` vs `plate_normalized` |
| 6 | `GET /api/v1/journey/{plate}` | path parameter + mandatory `disclaimer` + segments | a journey without `disclaimer` renders "Journey data incomplete" by design |
| 7 | `GET /api/v1/alerts` | `priority` + `acknowledged` | reveals `severity` / `acknowledged_at` |
| 8 | `GET /api/v1/watchlist`, then `POST`, then `DELETE /{id}` | the write paths | writes do not retry — one failure is one failure |
| 9 | `POST /api/v1/alerts/{id}/acknowledge` | the other write | |
| 10 | `WS /ws/alerts` | realtime | envelope + heartbeat, see §5 |
| 11 | `GET /api/v1/metrics/benchmark` | known divergent, see §4 | cosmetic — leave for last |

Not implemented by this frontend, do not test: `searchNearby`,
`cameras/{id}`, `cameras/sync`, `cameras/{id}/preview.m3u8`.

## 4. Live contract conflicts — symptom and fix

Task-0 audit against the real contract eliminated most of the conflicts we
had been carrying. **These are the ones that survive.**

### 4a. Benchmark report shape — REAL divergence, ours to fix

Canonical §7.3 and our reader disagree, and this is not a rename:

| Contract §7.3 | We read |
|---|---|
| `dataset_manifest_sha256` | `manifest_sha256` |
| `e2e_correct_plate_event_rate` (a RATE, 0–1) | `eligible_events` + `correct_events` (COUNTS — absent from the contract) |
| `by_plate_width: {">100": 0.98, …}` scalar per bucket | `by_plate_width: {">100": {eligible_events, correct_events}}` |
| `by_condition`, `diagnostics`, `notes` | not read |
| — | `failure_buckets` (absent from the contract) |

**Symptom:** System status → Benchmark reads *"The benchmark report could not
be read. Required fields were missing, so nothing is shown rather than a
partial result."*

**Not a one-line fix.** `readBenchmark` discards when `eligible_events` /
`correct_events` are absent, and the contract never sends them. Either the
panel is rebuilt against §7.3, or Akshat also emits counts. Decide before
spending time on it — the panel is not on the demo critical path.

### 4b. DB column names leaking onto the wire — absorbed, watch the counters

`external_camera_id` (§5.1 `cameras` column) and `plate_normalized` (§5.5
`vehicle_sightings` column) are **real contract names — for the database**.
The wire shape (§6.5, §6.4) is `camera_id` and `plate`. A naive ORM
serializer emits the column names.

**Symptom:** everything renders correctly, and System status → Contract drift
→ Fallbacks lists `external_camera_id` or `plate_normalized` with a count.

**Fix:** none needed on our side — the adapters absorb both. Tell Mihir; it
is a serializer fix, not a contract change.

### 4c. Conflicts that turned out NOT to exist

Do not spend time on these — the contract settles them our way:

- **`/stats/system`** — absent from the contract. §6.4 is `/api/v1/system/status`. We match.
- **journey as a query parameter** — §6.3 is `GET /api/v1/journey/{plate_normalized}`, a path parameter. We match.
- **`payload` WS envelope** — absent from the contract. §6.7 is `"data"`. We tolerate `payload` anyway and log it.
- **`is_feasible`, `health_state`, `severity`, `acknowledged_at`, `by_width_bucket`** — none appear anywhere in the contract, DB or wire. Our fallbacks for these are dead code.

## 5. WebSocket (§6.7)

Server → client, exactly:

```
{"type":"alert","data": Alert }
{"type":"sighting","data": VehicleSighting }
{"type":"system","data": SystemStatus }
{"type":"heartbeat","ts":"2026-09-01T10:03:21Z"}
```

`data`, not `payload`. Heartbeat carries **`ts` at the top level, unwrapped** —
a parser reading `msg.data.ts` throws on every beat. There is no
`camera_state` message. If a frame arrives under `payload`, the console logs
`[ws] envelope key "payload" received for type "alert" (canonical is "data")`
and it still works.

Watch System status → Realtime: WebSocket state, seconds since last heartbeat,
reconnects, malformed frames, unknown message types.

## 6. Reading the drift counters

System status → **Contract drift**, two separate lists. They count since page
load and count adapter READS, not distinct records — reload before a clean
measurement.

**Fallbacks** — a name moved, the record survived:
- one or two, on a big list → a few odd records
- **count ≈ number of records fetched → systematic. The backend uses the other
  spelling everywhere.** That is a contract divergence to raise with Mihir.
- the key names the legacy spelling that arrived, e.g. `external_camera_id`

**Discards** — a record was unusable and dropped:
- the key names the **first missing required field**
- **1–2 → one malformed record.** Not a contract problem.
- **discards ≈ total records → systematic.** A required field is missing from
  every row; the list on screen is shorter than what the server sent.
- Search shows this inline too: *"Server reported N matches; M shown, K
  unusable and dropped on read."*

Zero of both reads *"No drift detected since page load."* — that is the number
you want.

## 7. CORS — recognise it in ten seconds, not twenty minutes

The backend is on `:8000`, the UI on `:5173`. Different origin. If Mihir has
not enabled CORS, **every request fails and it looks exactly like the backend
being down.**

The console line is unmistakable:

```
Access to fetch at 'http://localhost:8000/api/v1/cameras' from origin
'http://localhost:5173' has been blocked by CORS policy: No
'Access-Control-Allow-Origin' header is present on the requested resource.
```

In the UI this surfaces as **"Cannot reach the server / Check that the API is
running"** — the same panel as a dead backend, because `fetch` rejects with a
`TypeError` either way and the browser withholds the response from JS.

**Discriminator:** `curl.exe -i http://localhost:8000/health/live` from the
same machine. If curl gets 200 and the browser cannot, it is CORS, not the
backend. Fix is on Mihir's side (`Access-Control-Allow-Origin`), not ours.

A `403`/`401` that DOES reach us renders an error panel with a `request_id` —
that is not CORS.

## 8. Falling back to mocks mid-session

**Under `npm run dev`: yes, no rebuild, measured under 15 seconds.**
Edit `VITE_APP_MODE=mock` in `.env.local` → Vite prints
`.env.local changed, restarting server...` and restarts in under a second →
reload the tab. Verified: badge returns to REPLAY, mock data serves.

**Under `npm run preview` (static `dist/`): no.** The values are compiled in;
you need `npm run build` again — measured **7.5s wall** (guard + typecheck +
vite build) — then restart preview.

`VITE_MOCK_SHAPE=hostile` serves the execution manuals' field spellings for
every conflict at once. Useful as a rehearsal; not needed against a real
backend.
