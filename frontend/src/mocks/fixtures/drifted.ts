// Deliberately NON-canonical payloads, used to exercise the adapter fallbacks
// and the discard path.
//
// WHAT THIS FILE TESTS, AND WHY IT IS SMALLER THAN IT WAS
//
// It used to carry six drifted spellings: plate_normalized, is_feasible,
// health_state, acknowledged_at, external_camera_id and by_width_bucket. The
// canonical contract then landed, and four of those appear NOWHERE in it --
// not on the wire, not as a database column. They came from execution manuals
// the contract supersedes. Fixtures for them were testing a backend the
// contract says cannot exist, and a harness that exercises impossible inputs
// reports confidence it has not earned.
//
// TWO drifted spellings survive, and only these two, because the contract
// itself makes them plausible:
//
//   plate_normalized    Canonical 5.5 -- the vehicle_sightings COLUMN. The
//                       wire field (6.5) is `plate`.
//   external_camera_id  Canonical 5.1 -- the cameras COLUMN, immutable once
//                       seeded. The wire field (6.4, 6.5) is `camera_id`.
//
// Both are real names for real things, one layer down. An ORM that serializes
// a row straight to JSON emits the column name, so seeing one arrive is a
// specific, predictable failure rather than a hypothetical.
//
// This file is the second entry in NONCANON_EXEMPT in scripts/guard.mjs. It
// has to contain those two names in order to simulate an API that sends them;
// every other guard rule still applies here.
//
// Typed as `unknown` on purpose. These are wire payloads, not canonical
// values, and typing them as VehicleSighting or Camera would assert exactly
// the conformance the adapters exist to check.

// 1. Sends `plate_normalized` where Canonical 6.5 says `plate`. Everything
//    else is present, so the record survives and the fallback fires.
export const driftedSighting: unknown = {
  sighting_id: "sig-drift-001",
  camera_id: "cam07",
  camera_name: "Maninagar station approach",
  lat: 23.0009,
  lon: 72.6001,
  first_seen_at: "2026-09-02T09:31:00.000Z",
  last_seen_at: "2026-09-02T09:31:02.400Z",
  source_pts_ms: 512000,
  source_mode: "file",
  plate_normalized: "GJ07XY4321",
  plate_raw: "GJ07XY4321",
  confidence: 0.88,
  match_state: "exact",
  evidence_count: 4,
  plate_width_px: 72,
  vehicle_type: "car",
  snapshot_uri: null,
};

// 2. A journey whose sightings carry the column spelling. Proves the fallback
//    reaches nested records, not just top-level ones, and that a journey
//    survives the substitution intact.
export const driftedJourney: unknown = {
  plate: "GJ07XY4321",
  disclaimer:
    "Observed movement sequence. Not a confirmed route.",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "sig-drift-001",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 23.0009,
      lon: 72.6001,
      first_seen_at: "2026-09-02T09:31:00.000Z",
      last_seen_at: "2026-09-02T09:31:02.400Z",
      source_pts_ms: 512000,
      source_mode: "file",
      plate_normalized: "GJ07XY4321",
      plate_raw: "GJ07XY4321",
      confidence: 0.88,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 72,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "sig-drift-002",
      camera_id: "cam19",
      camera_name: "Sardar Bridge north",
      lat: 23.0421,
      lon: 72.5588,
      first_seen_at: "2026-09-02T09:44:10.000Z",
      last_seen_at: "2026-09-02T09:44:12.900Z",
      source_pts_ms: 1304000,
      source_mode: "file",
      plate_normalized: "GJ07XY4321",
      plate_raw: "GJ07XY4321",
      confidence: 0.74,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 44,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    {
      from_camera_id: "cam07",
      to_camera_id: "cam19",
      from_time: "2026-09-02T09:31:02.400Z",
      to_time: "2026-09-02T09:44:10.000Z",
      straight_line_km: 7.4,
      elapsed_seconds: 768,
      required_speed_kmh: 34.7,
      feasible: true,
      note: null,
    },
  ],
};

// 3. A sighting with NO source_mode under any spelling. Canonical 6.5 makes it
//    required and there is no honest default -- "synthetic" would discredit a
//    real sighting and "live_rtsp" would fake provenance -- so readSighting
//    must DISCARD this record and count it against source_mode.
//
//    This one has nothing to do with drifted names. It is the discard path,
//    which is a different guarantee from the fallback path and needs its own
//    fixture.
export const sightingMissingSourceMode: unknown = {
  sighting_id: "sig-drift-003",
  camera_id: "cam12",
  camera_name: "Odhav ring road",
  lat: 23.0201,
  lon: 72.6602,
  first_seen_at: "2026-09-02T09:50:00.000Z",
  last_seen_at: "2026-09-02T09:50:01.800Z",
  source_pts_ms: 90000,
  plate: "GJ12AA0001",
  plate_raw: "GJ12AA0001",
  confidence: 0.62,
  match_state: "probable",
  evidence_count: 1,
  plate_width_px: 38,
  vehicle_type: "car",
  snapshot_uri: null,
};

// ---------------------------------------------------------------------------
// THE HOSTILE SET, as it now stands.
//
// Selected by VITE_MOCK_SHAPE=hostile. It serves a backend that is wrong in
// ways the contract PERMITS a backend to be wrong -- which after the audit is
// a much shorter list than it was:
//
//   1. Column names on the wire (plate_normalized, external_camera_id).
//      Absorbed by the adapters; the drift counters name which arrived.
//   2. lat / lon ABSENT rather than null. Canonical 6.5 types them
//      `number | null`; a serializer that omits nulls sends neither. This is
//      the one that used to crash Leaflet.
//   3. A camera `status` value outside 5.1's CHECK constraint.
//
// (3) needs saying explicitly rather than leaving implicit, because it is the
// one input here the contract forbids: 5.1 constrains status to
// online/offline/degraded/unknown at the database level, so a conforming
// backend cannot emit "healthy". It is kept ANYWAY, deliberately, for one
// reason: CHECK constraints protect the database, not the JSON encoder. A
// status enum widened in code and not in the migration, or a value mapped on
// the way out, reaches us without violating anything the database enforced.
// We handle it -- unrecognised status degrades to "unknown" and is counted --
// and testing that we handle it costs one fixture.
//
// What is NOT here any more, and must not come back without a contract
// change: is_feasible, health_state, severity, acknowledged_at,
// coords_placeholder, by_width_bucket.

export const hostileCameras: unknown = [
  // lat/lon ABSENT rather than null, and the column spelling for the id.
  {
    external_camera_id: "cam01",
    name: "Ashram Road junction",
    status: "online",
    last_seen_at: "2026-09-02T10:00:00.000Z",
  },
  {
    external_camera_id: "cam04",
    name: "Paldi circle",
    status: "degraded",
    last_seen_at: "2026-09-02T10:02:00.000Z",
  },
  // Status outside 5.1's CHECK. See the note above: kept on purpose, to prove
  // an unrecognised value degrades to "unknown" instead of crashing.
  {
    external_camera_id: "cam07",
    name: "Maninagar station approach",
    status: "healthy",
    last_seen_at: "2026-09-02T09:40:00.000Z",
  },
  {
    external_camera_id: "cam19",
    name: "Sardar Bridge north",
    status: "offline",
    last_seen_at: null,
  },
];

export const hostileSearch: unknown = {
  query: { plate: "GJ01AB1234", normalized: "GJ01AB1234", fuzzy: false },
  count: 2,
  results: [
    {
      sighting_id: "sig-h-001",
      external_camera_id: "cam01",
      camera_name: "Ashram Road junction",
      first_seen_at: "2026-09-02T10:00:00.000Z",
      last_seen_at: "2026-09-02T10:00:02.000Z",
      source_mode: "synthetic",
      plate_normalized: "GJ01AB1234",
      confidence: 0.94,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 88,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "sig-h-002",
      external_camera_id: "cam07",
      camera_name: "Maninagar station approach",
      first_seen_at: "2026-09-02T10:06:00.000Z",
      last_seen_at: "2026-09-02T10:06:03.000Z",
      source_mode: "synthetic",
      plate_normalized: "GJ01AB1234",
      confidence: 0.71,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 41,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
};

// Canonical field names throughout -- `plate`, `camera_id`, `priority`,
// `acknowledged` -- because the contract offers no alternative spelling for
// any of them. What makes this hostile is lat/lon absence upstream and the
// column spellings elsewhere; alerts themselves have no plausible drift, so
// this fixture asserts none.
export const hostileAlerts: unknown = [
  {
    alert_id: "alr-h-001",
    plate: "GJ01AB1234",
    camera_id: "cam01",
    camera_name: "Ashram Road junction",
    match_state: "exact",
    confidence: 0.94,
    priority: "critical",
    created_at: "2026-09-02T10:00:00.000Z",
    acknowledged: false,
    snapshot_uri: null,
  },
  {
    alert_id: "alr-h-002",
    plate: "GJ05CD5678",
    camera_id: "cam07",
    camera_name: "Maninagar station approach",
    match_state: "probable",
    confidence: 0.68,
    priority: "high",
    created_at: "2026-09-02T09:55:00.000Z",
    acknowledged: true,
    snapshot_uri: null,
  },
];

export const hostileJourney: unknown = {
  plate_normalized: "GJ01AB1234",
  disclaimer:
    "Observed movement sequence. Not a confirmed route.",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "sig-h-001",
      external_camera_id: "cam01",
      camera_name: "Ashram Road junction",
      first_seen_at: "2026-09-02T10:00:00.000Z",
      last_seen_at: "2026-09-02T10:00:02.000Z",
      source_mode: "synthetic",
      plate_normalized: "GJ01AB1234",
      confidence: 0.94,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 88,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "sig-h-002",
      external_camera_id: "cam07",
      camera_name: "Maninagar station approach",
      first_seen_at: "2026-09-02T10:06:00.000Z",
      last_seen_at: "2026-09-02T10:06:03.000Z",
      source_mode: "synthetic",
      plate_normalized: "GJ01AB1234",
      confidence: 0.71,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 41,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    {
      from_camera_id: "cam01",
      to_camera_id: "cam07",
      from_time: "2026-09-02T10:00:02.000Z",
      to_time: "2026-09-02T10:06:00.000Z",
      straight_line_km: 6.2,
      elapsed_seconds: 358,
      required_speed_kmh: 62.3,
      feasible: true,
      note: null,
    },
  ],
};
