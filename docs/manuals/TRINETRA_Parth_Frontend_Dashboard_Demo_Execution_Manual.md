# TRINETRA — EXECUTION MANUAL
## Parth — Frontend, Dashboard & Demo

**Version 2.0 · 2026-09-01 · 6 build days to the qualification gate**

> **Precedence.** [`docs/TRINETRA_Canonical_Contracts.md`](docs/TRINETRA_Canonical_Contracts.md) is normative for every type. Blocks marked `COPIED FROM CANONICAL — DO NOT EDIT HERE` are verbatim.

---

## 1. Your job in one paragraph

You own the only part of this system anyone will actually see. Judges will not read the fusion code or the DDL; they will watch a screen for eight minutes and form a conclusion. That makes you responsible for two things at once: an interface an operator could plausibly use, and an interface that does not overstate what the system knows. Those pull against each other — a solid line on a map is prettier and less true than a dashed one — and resolving that tension honestly is the actual job.

**You own:** `frontend/src/` · all screens · the demo script · the projector check · the LIVE/REPLAY badge.

**You do not own:** the API shape, the schema, the models. You consume contracts and report when they don't work.

**Your machine:** i7-14650HX · RTX 5050 8 GB · 24 GB RAM. Not GPU-bound work; your constraint is the browser.

---

## 2. Day plan — anchored to the real calendar

Today is **1 September 2026**. Submission is **7 September**. Hackathon is **10–11 September**. Six build days.

| Date | Day | You must finish | Proof |
|---|---|---|---|
| **Sep 1** | D1 | Vite + TS + Tailwind + Leaflet scaffold; **all types from Contracts §6.5**; MSW mocks | App renders map with mocked sightings; zero `any` |
| **Sep 2** | D2 | Live Map + Camera Grid against mocks | Markers, clustering, camera health colours |
| **Sep 3** | D3 | Search screen wired to Mihir's real API | Real search results on screen (**G5**) |
| **Sep 4** | D4 | Journey view + Alerts + WebSocket | Real alert appears without refresh (**G4**) |
| **Sep 5** | D5 | System Status, hls.js preview, reconnect UX, empty/error states | Kill the backend — UI degrades, doesn't blank |
| **Sep 6** | D6 | Demo script, projector check, recovery drills | 8-minute run-through twice, no dead air |
| Sep 7 | — | **SUBMIT** | — |
| Sep 8–9 | D7–D8 | Live badge verification, final rehearsal | LIVE vs REPLAY correct in both modes |

**Build against mocks from D1.** Do not wait for Mihir. MSW with fixtures matching Contracts §6.5 means you are never blocked, and it means the day the real API arrives you find out in minutes whether it matches the contract.

---

## 3. Stack

| Thing | Choice |
|---|---|
Build | Vite
Language | TypeScript, `strict: true`
UI | React 18 + Tailwind
Map | Leaflet (or MapLibre GL) + marker clustering
Data | TanStack Query
Realtime | native `WebSocket`
Video | **hls.js**
Mocks | MSW

`strict: true` and no `any` in API types. The types are the contract; `any` deletes the one mechanism that tells you Mihir renamed a field.

**hls.js is required**, not optional. Native `<video>` HLS playback works in Safari and nowhere else that matters. Without hls.js there is no camera preview in Chrome on demo day.

---

## 4. Types — copy these on D1

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §6.5)

```ts
// src/types/api.ts

export type MatchState = "exact" | "probable" | "low_confidence" | "unreadable";
export type SourceMode = "live_rtsp" | "live_hls" | "file" | "frames" | "synthetic";
export type HealthState = "unknown" | "healthy" | "degraded" | "offline";

export interface VehicleSighting {
  sighting_id: string;
  camera_id: string;          // "cam04" — Sentinel ID, lowercase
  camera_name: string;
  lat: number;
  lon: number;
  first_seen_at: string;      // ISO-8601 with Z
  last_seen_at: string;
  plate_normalized: string | null;
  plate_raw: string | null;
  vehicle_type: string | null;
  confidence: number | null;
  match_state: MatchState;
  evidence_count: number;
  plate_width_px: number | null;
  image_quality: number | null;
  source_mode: SourceMode;
  snapshot_uri: string | null;
}

export interface JourneySegment {
  from_camera: { external_camera_id: string; name: string; lat: number; lon: number };
  to_camera:   { external_camera_id: string; name: string; lat: number; lon: number };
  departed_at: string;
  arrived_at: string;
  distance_km: number;
  required_speed_kmh: number;
  is_feasible: boolean;
  match_state: MatchState;
}

export interface JourneyResponse {
  plate_normalized: string;
  segments: JourneySegment[];
  disclaimer: string;         // ALWAYS render this
}

export interface Alert {
  alert_id: string;
  watchlist_id: string;
  sighting: VehicleSighting;
  match_state: "exact" | "probable";
  confidence: number | null;
  severity: "low" | "medium" | "high" | "critical";
  reason: string | null;
  acknowledged_at: string | null;
  created_at: string;
}

export interface Camera {
  external_camera_id: string;
  name: string;
  lat: number;
  lon: number;
  district: string | null;
  is_active: boolean;
  health_state: HealthState;
  last_seen_at: string | null;
  coords_placeholder?: boolean;
}

export interface SystemStatus {
  is_live: boolean;           // drives the LIVE/REPLAY badge
  source_mode: SourceMode;
  cameras_total: number;
  cameras_healthy: number;
  cameras_degraded: number;
  cameras_offline: number;
  events_last_minute: number;
  db_ok: boolean;
  redis_ok: boolean;
  pipeline_version: string;
}

export interface ApiError {
  error: { code: string; message: string; field?: string };
}
```

### 4.1 What changed from an earlier draft you may be holding

| Was | Is now | Why it matters to you |
|---|---|---|
`timestamp: string` | `first_seen_at` + `last_seen_at` | A sighting is an interval — a vehicle is visible 2–3 s. Rendering an instant loses information and the field name won't exist. |
`plate: string \| null` | `plate_normalized` + `plate_raw` | Show `raw` to the operator, search on `normalized` |
`camera_id: "CAM_001"` | `camera_id: "cam04"` | Sentinel publishes `cam01`–`cam30`. Hardcoding `CAM-001` anywhere means every fixture and label breaks on live day. |
— | `match_state` | Drives colour and wording. Confidence alone is not enough. |
— | `evidence_count` | "3 observations" is more honest and more legible than "0.87" |
— | `plate_width_px` | Explains *why* something was uncertain |
— | `source_mode` | The LIVE/REPLAY badge depends on it |

**Search your code for `CAM-001` and `CAM_001` before D3.** Any occurrence is a future bug.

---

## 5. Six required screens

| # | Screen | Must show |
|---|---|---|
| 1 | **Live Map** | Clustered sighting markers, camera health colours, recent-events feed, LIVE/REPLAY badge |
| 2 | **Search** | Plate (partial ok) + time + camera + type + min confidence; results with snapshot, `match_state`, `evidence_count` |
| 3 | **Journey** | Ordered sightings, **dashed** connectors, infeasible flagged, disclaimer visible |
| 4 | **Alerts** | Live feed, severity, acknowledge action, `match_state` |
| 5 | **Camera Grid** | All cameras, health, last-seen, hls.js preview |
| 6 | **System Status** | Live vs replay, camera counts, events/min, DB + Redis health, pipeline version |

Screen 6 is not filler. Judges ask "is this actually running?" — being able to click one tab that answers it, with dependency health and a pipeline version, is worth more than a seventh feature.

### 5.1 Live Map

- Cluster markers — 30 cameras × many sightings will otherwise stack into an unreadable pile
- Camera colours: `healthy` green · `degraded` amber · `offline` grey · `unknown` slate
- Marker colour by `match_state`, **not** by raw confidence
- A camera with `coords_placeholder: true` gets a distinct hollow marker and a tooltip saying the location is approximate. One marker in the Gulf of Guinea discredits the entire map.
- Recent-events sidebar, newest first, capped at ~50 items in the DOM

### 5.2 Search

Debounce input 300 ms. Support partial plates — the realistic case is a witness who remembers four characters, so `GJ01` must return something useful.

Every result row shows the snapshot thumbnail, `plate_raw` as the display string, `match_state` as a labelled chip, and `evidence_count` as "N observations". Never show a bare `0.87`.

Empty state is a real state: "No sightings match these filters" with the active filters listed. A blank panel reads as a broken app, and on a projector nobody can tell the difference.

### 5.3 Journey — the screen with the most honesty risk

```
┌──────────────────────────────────────────────────────────────┐
│  Journey — GJ01AB1234            [ 10:00 → 11:00 ]  [Search] │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ● cam04  Ring Road            10:03:21   probable   ×3    │
│   ┊                                                          │
│   ┊  ┈┈┈┈┈┈┈┈┈  4.31 km · 33.6 km/h · plausible             │
│   ┊                                                          │
│   ● cam07  Ashram Chowk         10:11:02   probable   ×2    │
│   ┊                                                          │
│   ┊  ┈┈┈┈┈┈┈┈┈  61.8 km · 212 km/h · ⚠ NOT PLAUSIBLE        │
│   ┊                                                          │
│   ● cam19  Sarkhej Circle       10:28:44   low_conf   ×1    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ ⓘ Observed movement sequence between camera detections.      │
│   Not a confirmed route.                                     │
└──────────────────────────────────────────────────────────────┘
```

Three non-negotiables:

**Connectors are dashed.** A solid line between cam04 and cam07 asserts a route. What the system knows is that the vehicle was at both cameras. Cameras cover a fraction of a percent of road-km; everything between is inference. In a police tool, a solid line invites an operational decision based on a line you drew — and that is the one failure mode with consequences outside the demo room.

**The disclaimer is always visible.** Not a tooltip, not behind an info icon. The API returns `disclaimer` as a mandatory non-null field specifically so it cannot be lost in a refactor; render it as a persistent footer.

**Infeasible segments are shown, not hidden.** `is_feasible: false` gets a warning treatment with the required speed. An infeasible segment usually means an OCR misread, and that is exactly what an analyst needs to see. Filtering it out hides the system's own error from the person who could catch it.

### 5.4 Alerts

Newest first. Severity as colour. Acknowledge posts and optimistically updates.

Only `exact` and `probable` reach you — the backend has a `CHECK` constraint enforcing it. Don't build UI for `low_confidence` alerts; that path doesn't exist by design.

Cap the rendered list. An unbounded live feed over an eight-minute demo becomes thousands of nodes and a stuttering projector.

### 5.5 Camera Grid

Card per camera: name, `external_camera_id`, health chip, last-seen relative time, preview button.

Preview uses hls.js against Mihir's proxy:

```
GET /api/v1/cameras/{camera_id}/preview.m3u8
```

**Never** the upstream Sentinel HLS URL. Those endpoints are password-protected, and any credential in the frontend is public — `VITE_`-prefixed variables are compiled into the bundle. The proxy exists precisely so the browser never sees a credential.

Mount hls.js lazily, on click, and destroy the instance on unmount. Eight simultaneous HLS players will saturate the network and the CPU, and the symptom is a frozen dashboard during the demo.

### 5.6 System Status

Read `GET /api/v1/stats/system`. Show `is_live`, `source_mode`, the four camera counts, events/min, `db_ok`, `redis_ok`, `pipeline_version`.

When `redis_ok` is false, say "Live updates unavailable — data is current on refresh." That is both accurate and reassuring. Silently showing stale data is neither.

---

## 6. UI honesty requirements

| Requirement | Implementation |
|---|---|
Never present replay as live | LIVE / REPLAY badge from `is_live`, always visible, top bar |
Never assert a route | Dashed connectors + mandatory disclaimer |
Never show a bare confidence number | `match_state` chip + "N observations" |
Never hide an infeasible segment | Warning treatment with required speed |
Never invent a plate | `plate_normalized: null` renders "Unreadable", not a guess |
Never fake a marker location | `coords_placeholder` gets a distinct marker |
Never show "connected" when it isn't | WS state chip: connected / reconnecting / offline |
Never say "predicts where it will go" | Label it **"Camera Search Prioritization"** |

### 6.1 The badge

```tsx
{status.is_live
  ? <Badge tone="green">LIVE</Badge>
  : <Badge tone="amber">REPLAY · {status.source_mode}</Badge>}
```

Top bar, every screen, no exceptions. If a judge sees a dashboard labelled "ONLINE" and then learns it was a recorded file, everything else you say is discounted — and reasonably so. The badge costs ten lines and buys the credibility of the whole demo.

There is a related trap: if replay runs accelerated, the events-per-minute figure is inflated. Label it `events/min (replay ×5)` when accelerated. Replay acceleration never substantiates a live throughput claim.

---

## 7. Realtime

```ts
const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
```

| Message | Action |
|---|---|
`alert` | Prepend to alerts, toast, flash the map marker |
`sighting` | Prepend to the live feed, add/update the marker |
`camera_state` | Update the camera's health colour |
`heartbeat` | Reset the liveness timer |

Reconnect with exponential backoff + jitter, same policy as the AI worker. If no heartbeat arrives for 30 s, treat the connection as dead even if the socket claims open — a dead connection behind a NAT is otherwise indistinguishable from a quiet night, and the UI will confidently show "connected" while nothing flows.

The six-step test, on D5:

1. Connect, confirm alerts arrive
2. Kill the backend
3. UI shows disconnected within 20 s
4. Restart the backend
5. UI reconnects on its own
6. Alerts flow again **without a page refresh**

Step 6 is the whole test. A WebSocket that needs F5 has been demoed, not implemented — and demo day is exactly when the backend hiccups.

---

## 8. Configuration

```bash
# frontend/.env.example
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_APP_MODE=offline        # offline | live
```

**Nothing secret goes in a `VITE_` variable.** Vite inlines them into the bundle at build time; they are readable by anyone with the page open. No stream passwords, no API keys, no upstream URLs with credentials. If the frontend appears to need a credential, the answer is a backend proxy, not an environment variable.

Offline → live must be a **config change only**. If flipping `VITE_APP_MODE` requires a code edit, the source-independence invariant is broken on your side, and the live swap on D7 will fail under time pressure.

---

## 9. Demo script — 8 minutes, rehearsed twice on D6

| Time | Beat | Line |
|---|---|---|
0:00 | Problem | "80,000 cameras, 26 departments, no unified vehicle intelligence." |
0:45 | Insight | "**We don't centralize every video. We centralize intelligence.**" |
1:15 | Live Map | Point at the badge first. Say whether it's live or replay. |
2:00 | Camera Grid | Open one preview. Point out one degraded camera and say the system handles it. |
2:45 | Search | Partial plate → results. Point at `evidence_count`. |
3:45 | **Journey** | "Observed movement sequence." Point at the dashed lines and the disclaimer. |
4:45 | Infeasible segment | "212 km/h — the system flags this rather than hiding it. Probably an OCR misread." |
5:30 | Watchlist → Alert | Add a plate, trigger a sighting, alert arrives with no refresh. |
6:30 | Resilience | **Unplug the network.** The dashboard keeps working. |
7:15 | Honest limits | Width-bucket accuracy. "Below 30 pixels we largely fail, and we can tell you exactly how often." |
7:45 | Close | "We don't centralize every video. We centralize intelligence." |

Two beats do disproportionate work. **6:30** — unplugging the network on purpose is the single most memorable thing in an eight-minute demo, and no competitor will do it. **7:15** — volunteering your failure rate before being asked converts a weakness into evidence of rigour. Both require nerve; rehearse them until they're boring.

### 9.1 Projector check — 7 items, do it on D6 and again before the demo

1. 1024×768 and 1920×1080 both usable
2. Contrast readable in a bright room — light theme available, thin grey text avoided
3. Font sizes legible from 5 m
4. No horizontal scroll at any supported width
5. No reliance on hover — a projector audience never sees a hover state
6. Map tiles cached or a local tile source configured
7. Zoomed to 125% browser zoom, still functional

Item 6 is the one that fails. Leaflet's default tiles need internet. Your best demo beat is unplugging the network. Cache the tiles for the demo area, or the map goes grey exactly when you want applause.

### 9.2 Recovery drills

| If | Do |
|---|---|
Backend dies | Switch to the Search screen — cached results persist; talk while it restarts |
WS disconnects | Point at the reconnecting chip: "the UI tells you the truth about its own state" |
No live feed | Switch to REPLAY, say so out loud, continue |
Map tiles fail | Continue on the list views; mention offline tiles are configurable |
A plate reads wrong | Use it: "this is the confident-wrong case; note the low match_state" |
Laptop stutters | Close preview players first — they're the biggest cost |

Every failure has a scripted line. Rehearsed recovery reads as competence; dead air reads as a broken project. Practise these once each on D6.

---

## 10. Performance

- Virtualize any list over 100 rows
- Cap the DOM: alerts ≤ 100, recent events ≤ 50
- Cluster markers; never render 1,000 individually
- Lazy-mount hls.js on click; destroy on unmount; one player at a time if possible
- TanStack Query `staleTime` ≥ 5 s for search, no polling where a WS message will do
- Memoize map marker layers — re-creating them on every WS message will stutter

The target machine is a demo laptop also running Postgres, Redis, the API, and a GPU inference worker. Frontend inefficiency shows up as a stuttering projector, and the audience attributes it to the whole system.

---

## 11. Anti-patterns

| Do not | Consequence |
|---|---|
Solid journey lines | Asserts a route the system cannot know |
Hide the disclaimer in a tooltip | The honesty guarantee stops being visible |
Filter out infeasible segments | Hides the system's own errors from the analyst |
Show replay without the badge | One discovery discredits the entire demo |
Bare confidence numbers | Implies calibration that doesn't exist |
Hardcode `CAM-001` | Breaks on live day, everywhere at once |
Put a credential in `VITE_` | Public in the bundle |
Point hls.js at upstream Sentinel | Password-protected; leaks credentials |
Native `<video>` for HLS | Works only in Safari |
Unbounded live lists | Stuttering projector by minute six |
`any` in API types | You won't notice a renamed field until the demo |
Blank screen on empty results | Reads as broken on a projector |
Default Leaflet tiles with no cache | Grey map during the unplug beat |
WebSocket that needs F5 | Fails the moment the backend hiccups |
Wait for the real API before building | You lose two of six days |
Add a 7th screen instead of polishing 6 | Half-finished features read worse than fewer complete ones |

---

## 12. Definition of done

- [ ] Vite + TS `strict` + Tailwind + Leaflet, zero `any` in API types
- [ ] All types from Contracts §6.5 copied exactly, `first_seen_at` not `timestamp`
- [ ] Zero occurrences of `CAM-001` / `CAM_001` anywhere in the codebase
- [ ] MSW mocks from D1; app fully navigable without a backend
- [ ] All **6** screens implemented
- [ ] LIVE / REPLAY badge on every screen, driven by `is_live`
- [ ] Accelerated replay labels events/min accordingly
- [ ] Journey: dashed connectors, infeasible flagged, disclaimer permanently visible
- [ ] `plate_normalized: null` renders "Unreadable", never a guess
- [ ] `match_state` chips + "N observations" instead of bare confidence
- [ ] `coords_placeholder` cameras visually distinct
- [ ] hls.js preview via the backend proxy; no credentials in the frontend
- [ ] WebSocket passes the 6-step reconnect test without a refresh
- [ ] Heartbeat timeout treats a silent socket as dead
- [ ] Empty, loading, and error states for every screen
- [ ] One error handler for the `ApiError` envelope
- [ ] Lists virtualized; DOM capped; markers clustered
- [ ] 7-item projector check passed at 1024×768 and 1920×1080
- [ ] Map tiles cached for the demo area
- [ ] 8-minute demo rehearsed twice, both nerve beats included
- [ ] All 6 recovery drills practised
- [ ] `VITE_APP_MODE` flips offline↔live with no code change

---

## 13. Your daily loop

```bash
npm run dev
npm run typecheck
npm run build
```

`npm run build` daily, not on D6. Vite's dev server tolerates type errors that a production build rejects, and discovering that at 11 pm on Sep 6 is a fully avoidable disaster.

**Final principle:** you are the system's face and its conscience. Every element that overstates what we know is a liability; every element that shows a limit honestly is an asset. The dashed line, the badge, and the width-bucket slide are not modesty — they are the strongest things on the screen.
