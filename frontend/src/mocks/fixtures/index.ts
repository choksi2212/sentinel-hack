// Canonical-shaped development fixtures. Deliberately not a happy path:
// null plates, null coordinates, sub-30px plate widths, low_confidence
// matches and every camera status are all represented, because a fixture set
// that only contains clean rows produces a UI that breaks the first time it
// meets real data.
//
// Coordinates are approximate Ahmedabad locations, for development only.
import type { Alert, SystemStatus, VehicleSighting } from "../../types/api";
import type { Camera, SearchResponse } from "../../types/ui";

// A sighting is an interval, never an instant. This helper derives
// last_seen_at from a gap so the invariant cannot be typo'd away: every
// fixture below spans between 1.5 and 3.5 seconds.
function interval(
  firstSeenAt: string,
  gapSeconds: number,
): { first_seen_at: string; last_seen_at: string } {
  const start = new Date(firstSeenAt);
  return {
    first_seen_at: start.toISOString(),
    last_seen_at: new Date(start.getTime() + gapSeconds * 1000).toISOString(),
  };
}

// 12 cameras: 1 offline, 2 degraded, 1 unknown, 8 online.
// cam09..cam12 have null coordinates and must be listed, never plotted.
export const cameras: Camera[] = [
  {
    camera_id: "cam01",
    name: "Ashram Road junction",
    lat: 23.0339,
    lon: 72.5661,
    status: "online",
    last_seen_at: "2026-09-02T09:16:11.000Z",
  },
  {
    camera_id: "cam02",
    name: "Nehru Bridge east",
    lat: 23.0258,
    lon: 72.5734,
    status: "online",
    last_seen_at: "2026-09-02T09:16:09.000Z",
  },
  {
    camera_id: "cam03",
    name: "CG Road crossing",
    lat: 23.0301,
    lon: 72.5602,
    status: "degraded",
    last_seen_at: "2026-09-02T09:12:44.000Z",
  },
  {
    camera_id: "cam04",
    name: "Paldi circle",
    lat: 23.0122,
    lon: 72.5573,
    status: "online",
    last_seen_at: "2026-09-02T09:16:12.000Z",
  },
  {
    camera_id: "cam05",
    name: "Vastrapur lake north",
    lat: 23.0367,
    lon: 72.5289,
    status: "offline",
    last_seen_at: "2026-09-02T07:41:03.000Z",
  },
  {
    camera_id: "cam06",
    name: "SG Highway toll",
    lat: 23.0472,
    lon: 72.5115,
    status: "online",
    last_seen_at: "2026-09-02T09:16:10.000Z",
  },
  {
    camera_id: "cam07",
    name: "Maninagar station approach",
    lat: 22.9963,
    lon: 72.6018,
    status: "degraded",
    last_seen_at: "2026-09-02T09:10:58.000Z",
  },
  {
    camera_id: "cam08",
    name: "Sabarmati riverfront west",
    lat: 23.0521,
    lon: 72.5804,
    status: "online",
    last_seen_at: "2026-09-02T09:16:08.000Z",
  },
  {
    camera_id: "cam09",
    name: "Naroda industrial gate",
    lat: null,
    lon: null,
    status: "unknown",
    last_seen_at: null,
  },
  {
    camera_id: "cam10",
    name: "Bopal approach road",
    lat: null,
    lon: null,
    status: "online",
    last_seen_at: "2026-09-02T09:15:52.000Z",
  },
  {
    camera_id: "cam11",
    name: "Chandkheda flyover",
    lat: null,
    lon: null,
    status: "online",
    last_seen_at: "2026-09-02T09:16:01.000Z",
  },
  {
    camera_id: "cam12",
    name: "Odhav ring road",
    lat: null,
    lon: null,
    status: "online",
    last_seen_at: "2026-09-02T09:15:47.000Z",
  },
];

// 15 sightings. Includes 3 unreadable plates, 2 low_confidence matches with
// a single observation, a 24px plate width (below the 30px threshold where
// reads stop being trustworthy), a null snapshot and 2 sightings with no
// coordinates.
export const sightings: VehicleSighting[] = [
  {
    sighting_id: "sig-001",
    camera_id: "cam01",
    camera_name: "Ashram Road junction",
    lat: 23.0339,
    lon: 72.5661,
    ...interval("2026-09-02T09:14:03.000Z", 2.2),
    source_pts_ms: 128400,
    source_mode: "synthetic",
    plate: "GJ01AB1234",
    plate_raw: "GJ01AB1234",
    confidence: 0.94,
    match_state: "exact",
    evidence_count: 6,
    plate_width_px: 96,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-001.jpg",
  },
  {
    sighting_id: "sig-002",
    camera_id: "cam02",
    camera_name: "Nehru Bridge east",
    lat: 23.0258,
    lon: 72.5734,
    ...interval("2026-09-02T09:14:47.000Z", 3.1),
    source_pts_ms: 172100,
    source_mode: "synthetic",
    plate: "GJ01AB1234",
    plate_raw: "GJ01AB1Z34",
    confidence: 0.88,
    match_state: "exact",
    evidence_count: 5,
    plate_width_px: 88,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-002.jpg",
  },
  {
    sighting_id: "sig-003",
    camera_id: "cam04",
    camera_name: "Paldi circle",
    lat: 23.0122,
    lon: 72.5573,
    ...interval("2026-09-02T09:15:31.000Z", 1.8),
    source_pts_ms: 216900,
    source_mode: "synthetic",
    plate: "GJ01AB1234",
    plate_raw: "GJ01AB1234",
    confidence: 0.91,
    match_state: "exact",
    evidence_count: 4,
    plate_width_px: 74,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-003.jpg",
  },
  {
    sighting_id: "sig-004",
    camera_id: "cam03",
    camera_name: "CG Road crossing",
    lat: 23.0301,
    lon: 72.5602,
    ...interval("2026-09-02T09:11:12.000Z", 2.6),
    source_pts_ms: 91200,
    source_mode: "synthetic",
    plate: null,
    plate_raw: "GJ0?A?12??",
    confidence: null,
    match_state: "unreadable",
    evidence_count: 2,
    plate_width_px: 41,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-004.jpg",
  },
  {
    sighting_id: "sig-005",
    camera_id: "cam06",
    camera_name: "SG Highway toll",
    lat: 23.0472,
    lon: 72.5115,
    ...interval("2026-09-02T09:09:58.000Z", 3.4),
    source_pts_ms: 78300,
    source_mode: "synthetic",
    plate: "GJ05CD5678",
    plate_raw: "GJ05CD5678",
    confidence: 0.82,
    match_state: "probable",
    evidence_count: 3,
    plate_width_px: 61,
    vehicle_type: "truck",
    snapshot_uri: "/snapshots/sig-005.jpg",
  },
  {
    sighting_id: "sig-006",
    camera_id: "cam07",
    camera_name: "Maninagar station approach",
    lat: 22.9963,
    lon: 72.6018,
    ...interval("2026-09-02T09:08:22.000Z", 1.5),
    source_pts_ms: 62800,
    source_mode: "synthetic",
    plate: "GJ18KL9012",
    plate_raw: "GJ18KL9O12",
    confidence: 0.44,
    match_state: "low_confidence",
    evidence_count: 1,
    plate_width_px: 24,
    vehicle_type: "motorcycle",
    snapshot_uri: "/snapshots/sig-006.jpg",
  },
  {
    sighting_id: "sig-007",
    camera_id: "cam08",
    camera_name: "Sabarmati riverfront west",
    lat: 23.0521,
    lon: 72.5804,
    ...interval("2026-09-02T09:13:05.000Z", 2.9),
    source_pts_ms: 143700,
    source_mode: "synthetic",
    plate: "GJ05CD5678",
    plate_raw: "GJ05CD5678",
    confidence: 0.79,
    match_state: "probable",
    evidence_count: 3,
    plate_width_px: 58,
    vehicle_type: "truck",
    snapshot_uri: "/snapshots/sig-007.jpg",
  },
  {
    sighting_id: "sig-008",
    camera_id: "cam01",
    camera_name: "Ashram Road junction",
    lat: 23.0339,
    lon: 72.5661,
    ...interval("2026-09-02T09:06:41.000Z", 2.0),
    source_pts_ms: 41500,
    source_mode: "synthetic",
    plate: "GJ27MN3456",
    plate_raw: "GJ27MN3456",
    confidence: 0.9,
    match_state: "exact",
    evidence_count: 7,
    plate_width_px: 103,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-008.jpg",
  },
  {
    sighting_id: "sig-009",
    camera_id: "cam05",
    camera_name: "Vastrapur lake north",
    lat: 23.0367,
    lon: 72.5289,
    ...interval("2026-09-02T07:38:19.000Z", 3.3),
    source_pts_ms: 12400,
    source_mode: "synthetic",
    plate: null,
    plate_raw: null,
    confidence: null,
    match_state: "unreadable",
    evidence_count: 1,
    plate_width_px: 33,
    vehicle_type: null,
    snapshot_uri: null,
  },
  {
    sighting_id: "sig-010",
    camera_id: "cam02",
    camera_name: "Nehru Bridge east",
    lat: 23.0258,
    lon: 72.5734,
    ...interval("2026-09-02T09:04:12.000Z", 2.4),
    source_pts_ms: 27100,
    source_mode: "synthetic",
    plate: "GJ27MN3456",
    plate_raw: "GJ27MN3456",
    confidence: 0.86,
    match_state: "exact",
    evidence_count: 5,
    plate_width_px: 81,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-010.jpg",
  },
  {
    sighting_id: "sig-011",
    camera_id: "cam03",
    camera_name: "CG Road crossing",
    lat: 23.0301,
    lon: 72.5602,
    ...interval("2026-09-02T09:02:55.000Z", 1.7),
    source_pts_ms: 19800,
    source_mode: "synthetic",
    plate: "GJ38PQ7890",
    plate_raw: "GJ38P07890",
    confidence: 0.39,
    match_state: "low_confidence",
    evidence_count: 1,
    plate_width_px: 27,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-011.jpg",
  },
  {
    sighting_id: "sig-012",
    camera_id: "cam09",
    camera_name: "Naroda industrial gate",
    lat: null,
    lon: null,
    ...interval("2026-09-02T08:57:30.000Z", 2.8),
    source_pts_ms: null,
    source_mode: "file",
    plate: "GJ05CD5678",
    plate_raw: "GJ05CD5678",
    confidence: 0.77,
    match_state: "probable",
    evidence_count: 2,
    plate_width_px: 52,
    vehicle_type: "truck",
    snapshot_uri: "/snapshots/sig-012.jpg",
  },
  {
    sighting_id: "sig-013",
    camera_id: "cam10",
    camera_name: "Bopal approach road",
    lat: null,
    lon: null,
    ...interval("2026-09-02T08:51:07.000Z", 3.5),
    source_pts_ms: null,
    source_mode: "file",
    plate: "GJ01AB1234",
    plate_raw: "GJ01AB1234",
    confidence: 0.83,
    match_state: "probable",
    evidence_count: 3,
    plate_width_px: 66,
    vehicle_type: "car",
    snapshot_uri: "/snapshots/sig-013.jpg",
  },
  {
    sighting_id: "sig-014",
    camera_id: "cam11",
    camera_name: "Chandkheda flyover",
    lat: null,
    lon: null,
    ...interval("2026-09-02T08:44:36.000Z", 2.1),
    source_pts_ms: null,
    source_mode: "frames",
    plate: null,
    plate_raw: "??38PQ78??",
    confidence: null,
    match_state: "unreadable",
    evidence_count: 1,
    plate_width_px: 29,
    vehicle_type: null,
    snapshot_uri: "/snapshots/sig-014.jpg",
  },
  {
    sighting_id: "sig-015",
    camera_id: "cam12",
    camera_name: "Odhav ring road",
    lat: 23.0198,
    lon: 72.6647,
    ...interval("2026-09-02T08:39:14.000Z", 2.7),
    source_pts_ms: 8200,
    source_mode: "synthetic",
    plate: "GJ18KL9012",
    plate_raw: "GJ18KL9012",
    confidence: 0.87,
    match_state: "exact",
    evidence_count: 4,
    plate_width_px: 78,
    vehicle_type: "motorcycle",
    snapshot_uri: "/snapshots/sig-015.jpg",
  },
];

function byPlate(plate: string): VehicleSighting[] {
  return sightings.filter((sighting) => sighting.plate === plate);
}

const exactResults = byPlate("GJ01AB1234");

export const searchExact: SearchResponse = {
  query: { plate: "GJ01AB1234", normalized: "GJ01AB1234", fuzzy: false },
  count: exactResults.length,
  results: exactResults,
};

// Fuzzy search keeps candidates in their own region, never merged into
// results. No candidate rises above "probable" -- a near-miss on the plate
// string is not evidence of an exact match.
//
// Candidates carry their own sighting_ids. They used to be spread from rows
// in `sightings`, which meant a candidate and a result could share an id;
// the Search screen renders both regions on one page, so React would have
// seen duplicate keys.
const fuzzyResults = byPlate("GJ05CD5678");

export const searchFuzzy: SearchResponse = {
  query: { plate: "GJ05CD5678", normalized: "GJ05CD5678", fuzzy: true },
  count: fuzzyResults.length,
  results: fuzzyResults,
  candidates: [
    {
      sighting_id: "cand-001",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T09:07:18.000Z", 2.3),
      source_pts_ms: 58200,
      source_mode: "synthetic",
      plate: "GJ05CD5670",
      plate_raw: "GJ05CD567O",
      confidence: 0.68,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 47,
      vehicle_type: "truck",
      snapshot_uri: null,
      distance: 1,
    },
    {
      sighting_id: "cand-002",
      camera_id: "cam03",
      camera_name: "CG Road crossing",
      lat: 23.0301,
      lon: 72.5602,
      ...interval("2026-09-02T08:49:02.000Z", 1.9),
      source_pts_ms: 33100,
      source_mode: "synthetic",
      plate: "GJ05C05678",
      plate_raw: "GJ05C05678",
      confidence: 0.51,
      match_state: "low_confidence",
      evidence_count: 1,
      plate_width_px: 26,
      vehicle_type: null,
      snapshot_uri: null,
      distance: 1,
    },
    {
      sighting_id: "cand-003",
      camera_id: "cam11",
      camera_name: "Chandkheda flyover",
      lat: null,
      lon: null,
      ...interval("2026-09-02T08:31:44.000Z", 3.2),
      source_pts_ms: null,
      source_mode: "frames",
      plate: "GJ06CD5678",
      plate_raw: "GJ06CD5678",
      confidence: 0.63,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 39,
      vehicle_type: "truck",
      snapshot_uri: null,
      distance: 2,
    },
  ],
};

// is_live false and source_mode "synthetic" so the badge reads REPLAY by
// default. A demo that claims to be live when it is not is the one lie this
// screen cannot afford.
export const systemStatus: SystemStatus = {
  api: "ok",
  postgres: true,
  redis: true,
  source_mode: "synthetic",
  cameras_total: 12,
  cameras_live: 0,
  cameras_replay: 12,
  is_live: false,
};

// ---------------------------------------------------------------------
// Journey fixtures.
//
// Typed `unknown`, not JourneyResponse, and deliberately so: several of these
// omit required keys in order to exercise the adapter's discard and
// three-state paths. Typing them as the canonical shape would assert exactly
// the conformance readJourney exists to check, and would not compile.
// ---------------------------------------------------------------------

// A: five sightings, four adjacent pairs, one orphan segment and one segment
// the adapter must discard.
//
// The sightings array is deliberately OUT OF CHRONOLOGICAL ORDER so the
// client-side sort is actually exercised. Chronological order is
// cam04 -> cam07 -> cam19 -> cam23 -> cam09; array order is not.
//
// cam19 and cam23 are intentionally absent from the `cameras` fixture above,
// which exercises the unknown-camera_id name fallback on the screen.
export const journeyFourCamera: unknown = {
  plate: "GJ01AB1234",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 5,
  sightings: [
    {
      sighting_id: "jsig-003",
      camera_id: "cam19",
      camera_name: "Sardar Bridge north",
      lat: 23.0455,
      lon: 72.5912,
      ...interval("2026-09-02T10:09:10.000Z", 3.1),
      source_pts_ms: 331000,
      source_mode: "synthetic",
      // Could not be read. Renders "Unreadable", never a guess.
      plate: null,
      plate_raw: "GJ01A?12?4",
      confidence: null,
      match_state: "unreadable",
      evidence_count: 1,
      plate_width_px: 37,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "jsig-001",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T10:00:00.000Z", 2.2),
      source_pts_ms: 12000,
      source_mode: "synthetic",
      plate: "GJ01AB1234",
      plate_raw: "GJ01AB1234",
      confidence: 0.93,
      match_state: "exact",
      evidence_count: 6,
      plate_width_px: 91,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "jsig-005",
      camera_id: "cam09",
      camera_name: "Naroda industrial gate",
      // No surveyed coordinates. Listed, never plotted.
      lat: null,
      lon: null,
      ...interval("2026-09-02T10:21:05.000Z", 1.7),
      source_pts_ms: null,
      source_mode: "frames",
      plate: "GJ01AB1234",
      plate_raw: "GJ01AB1234",
      confidence: 0.81,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 54,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "jsig-004",
      camera_id: "cam23",
      camera_name: "Vatva flyover",
      lat: 22.9871,
      lon: 72.6294,
      ...interval("2026-09-02T10:14:45.000Z", 2.6),
      source_pts_ms: 402000,
      source_mode: "synthetic",
      plate: "GJ01AB1234",
      plate_raw: "GJ01AB1234",
      confidence: 0.44,
      match_state: "low_confidence",
      evidence_count: 1,
      // Below the 30px threshold where a read stops being trustworthy.
      plate_width_px: 24,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "jsig-002",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T10:04:30.000Z", 1.9),
      source_pts_ms: 174000,
      source_mode: "synthetic",
      plate: "GJ01AB1234",
      plate_raw: "GJ01AB1234",
      confidence: 0.88,
      match_state: "exact",
      evidence_count: 5,
      plate_width_px: 83,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    // S1 -- ordinary, assessed and passed.
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T10:00:00.000Z",
      to_time: "2026-09-02T10:04:30.000Z",
      straight_line_km: 4.1,
      elapsed_seconds: 270,
      required_speed_kmh: 54.7,
      feasible: true,
      note: null,
    },
    // S2 -- assessed and failed. Flagged, never filtered.
    {
      from_camera_id: "cam07",
      to_camera_id: "cam19",
      from_time: "2026-09-02T10:04:30.000Z",
      to_time: "2026-09-02T10:09:10.000Z",
      straight_line_km: 16.4,
      elapsed_seconds: 280,
      required_speed_kmh: 212,
      feasible: false,
      note: "Requires 212 km/h between these cameras. Check for an OCR error before treating this as a real movement.",
    },
    // S3 -- feasible and required_speed_kmh keys BOTH omitted. Never
    // assessed; must not render as assessed-and-passed.
    {
      from_camera_id: "cam19",
      to_camera_id: "cam23",
      from_time: "2026-09-02T10:09:10.000Z",
      to_time: "2026-09-02T10:14:45.000Z",
      straight_line_km: 5.2,
      elapsed_seconds: 335,
      note: null,
    },
    // S4 -- to_time omitted. readSegment must DISCARD this one.
    {
      from_camera_id: "cam23",
      to_camera_id: "cam09",
      from_time: "2026-09-02T10:14:45.000Z",
      straight_line_km: 7.7,
      elapsed_seconds: 380,
      required_speed_kmh: 73,
      feasible: true,
      note: null,
    },
    // S5 -- well formed, but matches no adjacent pair. An orphan is counted
    // and shown, never dropped.
    {
      from_camera_id: "cam11",
      to_camera_id: "cam12",
      from_time: "2026-09-02T10:30:00.000Z",
      to_time: "2026-09-02T10:36:00.000Z",
      straight_line_km: 3.9,
      elapsed_seconds: 360,
      required_speed_kmh: 39,
      feasible: true,
      note: null,
    },
  ],
};

// B: one sighting, zero segments. A one-node journey is a VALID result, not
// an empty state. The disclaimer still renders.
export const journeySingleSighting: unknown = {
  plate: "GJ05CD6789",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 1,
  sightings: [
    {
      sighting_id: "jsig-solo",
      camera_id: "cam06",
      camera_name: "SG Highway toll",
      lat: 23.0472,
      lon: 72.5115,
      ...interval("2026-09-02T11:02:14.000Z", 2.4),
      source_pts_ms: 88000,
      source_mode: "synthetic",
      plate: "GJ05CD6789",
      plate_raw: "GJ05CD6789",
      confidence: 0.9,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 77,
      vehicle_type: "truck",
      snapshot_uri: null,
    },
  ],
  segments: [],
};

// C: well formed except that disclaimer is absent. readJourney must return
// null and the screen must become an error state.
export const journeyMissingDisclaimer: unknown = {
  plate: "GJ18EF4321",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "jsig-nd-1",
      camera_id: "cam02",
      camera_name: "Nehru Bridge east",
      lat: 23.0258,
      lon: 72.5734,
      ...interval("2026-09-02T12:00:00.000Z", 2.0),
      source_pts_ms: 5000,
      source_mode: "synthetic",
      plate: "GJ18EF4321",
      plate_raw: "GJ18EF4321",
      confidence: 0.86,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 68,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "jsig-nd-2",
      camera_id: "cam08",
      camera_name: "Sabarmati riverfront west",
      lat: 23.0521,
      lon: 72.5804,
      ...interval("2026-09-02T12:06:40.000Z", 3.4),
      source_pts_ms: 9000,
      source_mode: "synthetic",
      plate: "GJ18EF4321",
      plate_raw: "GJ18EF4321",
      confidence: 0.79,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 59,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [],
};

// E: the same camera pair twice. cam04 -> cam07 -> cam04 -> cam07.
// Stored out of order; sorts to that sequence. Three segments, one per pair,
// each with a from_time inside its own pair's interval. E1 and E3 share a
// camera pair and are separated only by time, which is exactly what the
// tiebreak has to resolve.
export const journeyRepeatedPair: unknown = {
  plate: "GJ22KL0007",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 4,
  sightings: [
    // pair index 2 start / pair index 1 end -- third chronologically
    {
      sighting_id: "rep-3",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T14:20:00.000Z", 2.1),
      source_pts_ms: 120000,
      source_mode: "synthetic",
      plate: "GJ22KL0007",
      plate_raw: "GJ22KL0007",
      confidence: 0.9,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 84,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    // fourth chronologically
    {
      sighting_id: "rep-4",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T14:31:00.000Z", 2.9),
      source_pts_ms: 180000,
      source_mode: "synthetic",
      plate: "GJ22KL0007",
      plate_raw: "GJ22KL0007",
      confidence: 0.87,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 79,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    // first chronologically
    {
      sighting_id: "rep-1",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T14:00:00.000Z", 1.8),
      source_pts_ms: 20000,
      source_mode: "synthetic",
      plate: "GJ22KL0007",
      plate_raw: "GJ22KL0007",
      confidence: 0.93,
      match_state: "exact",
      evidence_count: 5,
      plate_width_px: 90,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    // second chronologically
    {
      sighting_id: "rep-2",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T14:09:00.000Z", 3.2),
      source_pts_ms: 70000,
      source_mode: "synthetic",
      plate: "GJ22KL0007",
      plate_raw: "GJ22KL0007",
      confidence: 0.89,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 82,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    // E1 -- pair 0, cam04 -> cam07. from_time inside [14:00:00, 14:09:03.2].
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T14:00:01.000Z",
      to_time: "2026-09-02T14:09:00.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 539,
      required_speed_kmh: 55.4,
      feasible: true,
      note: null,
    },
    // E2 -- pair 1, cam07 -> cam04. Only candidate for its pair.
    {
      from_camera_id: "cam07",
      to_camera_id: "cam04",
      from_time: "2026-09-02T14:09:02.000Z",
      to_time: "2026-09-02T14:20:00.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 658,
      required_speed_kmh: 45.4,
      feasible: true,
      note: null,
    },
    // E3 -- pair 2, cam04 -> cam07 again. from_time is twenty minutes after
    // E1's, which is what tells the two apart.
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T14:20:01.000Z",
      to_time: "2026-09-02T14:31:00.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 659,
      required_speed_kmh: 45.3,
      feasible: true,
      note: null,
    },
  ],
};

// F: unresolvable. One pair, two candidate segments, BOTH cam04 -> cam07 and
// BOTH with a from_time inside the pair's interval. The tiebreak cannot
// separate them, so neither is attached and neither is consumed.
export const journeyAmbiguousPair: unknown = {
  plate: "GJ22KL0008",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "amb-1",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T15:00:00.000Z", 2.0),
      source_pts_ms: 30000,
      source_mode: "synthetic",
      plate: "GJ22KL0008",
      plate_raw: "GJ22KL0008",
      confidence: 0.91,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 86,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "amb-2",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T15:10:00.000Z", 2.4),
      source_pts_ms: 90000,
      source_mode: "synthetic",
      plate: "GJ22KL0008",
      plate_raw: "GJ22KL0008",
      confidence: 0.88,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 81,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T15:00:30.000Z",
      to_time: "2026-09-02T15:10:00.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 570,
      required_speed_kmh: 52.4,
      feasible: true,
      note: null,
    },
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T15:01:15.000Z",
      to_time: "2026-09-02T15:10:00.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 525,
      required_speed_kmh: 56.9,
      feasible: true,
      note: null,
    },
  ],
};

// H: candidates by camera identity, none by time. Two segments both
// cam04 -> cam07, and NEITHER from_time falls inside the pair's interval
// [15:00:00, 15:10:02.4] -- one is an hour early, one an hour late. That is
// not ambiguity: there was nothing to be ambiguous between.
export const journeyNoTimeMatch: unknown = {
  plate: "GJ22KL0010",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "ntm-1",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T17:00:00.000Z", 2.0),
      source_pts_ms: 31000,
      source_mode: "synthetic",
      plate: "GJ22KL0010",
      plate_raw: "GJ22KL0010",
      confidence: 0.9,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 86,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "ntm-2",
      camera_id: "cam07",
      camera_name: "Maninagar station approach",
      lat: 22.9963,
      lon: 72.6018,
      ...interval("2026-09-02T17:10:00.000Z", 2.4),
      source_pts_ms: 91000,
      source_mode: "synthetic",
      plate: "GJ22KL0010",
      plate_raw: "GJ22KL0010",
      confidence: 0.87,
      match_state: "exact",
      evidence_count: 3,
      plate_width_px: 80,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    // An hour before the pair's interval.
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T16:00:00.000Z",
      to_time: "2026-09-02T16:09:30.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 570,
      required_speed_kmh: 52.4,
      feasible: true,
      note: null,
    },
    // An hour after it.
    {
      from_camera_id: "cam04",
      to_camera_id: "cam07",
      from_time: "2026-09-02T18:00:00.000Z",
      to_time: "2026-09-02T18:09:30.000Z",
      straight_line_km: 8.3,
      elapsed_seconds: 570,
      required_speed_kmh: 52.4,
      feasible: true,
      note: null,
    },
  ],
};

// I: a well-formed segment joining a placed sighting to an UNPLACED one.
// This is the only fixture that reaches the undrawable-connector path.
// Fixture A does not: its unplaced node sits in the one pair whose segment
// readSegment discards, so that pair is a gap and never becomes a candidate
// connector at all.
export const journeyUndrawableConnector: unknown = {
  plate: "GJ22KL0011",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sighting_count: 2,
  sightings: [
    {
      sighting_id: "und-1",
      camera_id: "cam04",
      camera_name: "Paldi circle",
      lat: 23.0122,
      lon: 72.5573,
      ...interval("2026-09-02T18:00:00.000Z", 2.0),
      source_pts_ms: 33000,
      source_mode: "synthetic",
      plate: "GJ22KL0011",
      plate_raw: "GJ22KL0011",
      confidence: 0.9,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 87,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "und-2",
      camera_id: "cam09",
      camera_name: "Naroda industrial gate",
      // Not surveyed. The connector into this node cannot be drawn, and the
      // camera's position must NOT be substituted for the sighting's.
      lat: null,
      lon: null,
      ...interval("2026-09-02T18:11:30.000Z", 2.2),
      source_pts_ms: null,
      source_mode: "frames",
      plate: "GJ22KL0011",
      plate_raw: "GJ22KL0011",
      confidence: 0.83,
      match_state: "probable",
      evidence_count: 2,
      plate_width_px: 61,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    {
      from_camera_id: "cam04",
      to_camera_id: "cam09",
      from_time: "2026-09-02T18:00:01.000Z",
      to_time: "2026-09-02T18:11:30.000Z",
      straight_line_km: 11.2,
      elapsed_seconds: 689,
      required_speed_kmh: 58.5,
      feasible: true,
      note: null,
    },
  ],
};

// G: the sighting_count key is omitted entirely. Everything else is well
// formed. readJourney must NOT return null and must NOT invent a count.
export const journeyNoCount: unknown = {
  plate: "GJ22KL0009",
  disclaimer:
    "Sightings are individual camera observations. Connecting lines are inferred, not observed routes.",
  sightings: [
    {
      sighting_id: "nc-1",
      camera_id: "cam02",
      camera_name: "Nehru Bridge east",
      lat: 23.0258,
      lon: 72.5734,
      ...interval("2026-09-02T16:00:00.000Z", 2.3),
      source_pts_ms: 40000,
      source_mode: "synthetic",
      plate: "GJ22KL0009",
      plate_raw: "GJ22KL0009",
      confidence: 0.9,
      match_state: "exact",
      evidence_count: 4,
      plate_width_px: 85,
      vehicle_type: "car",
      snapshot_uri: null,
    },
    {
      sighting_id: "nc-2",
      camera_id: "cam06",
      camera_name: "SG Highway toll",
      lat: 23.0472,
      lon: 72.5115,
      ...interval("2026-09-02T16:08:20.000Z", 1.9),
      source_pts_ms: 95000,
      source_mode: "synthetic",
      plate: "GJ22KL0009",
      plate_raw: "GJ22KL0009",
      confidence: 0.85,
      match_state: "probable",
      evidence_count: 3,
      plate_width_px: 74,
      vehicle_type: "car",
      snapshot_uri: null,
    },
  ],
  segments: [
    {
      from_camera_id: "cam02",
      to_camera_id: "cam06",
      from_time: "2026-09-02T16:00:01.000Z",
      to_time: "2026-09-02T16:08:20.000Z",
      straight_line_km: 6.9,
      elapsed_seconds: 499,
      required_speed_kmh: 49.8,
      feasible: true,
      note: null,
    },
  ],
};

// All four priorities represented. An alert is only ever exact or probable;
// the database forbids a low_confidence alert path.
export const alerts: Alert[] = [
  {
    alert_id: "alr-001",
    plate: "GJ01AB1234",
    camera_id: "cam01",
    camera_name: "Ashram Road junction",
    match_state: "exact",
    confidence: 0.94,
    priority: "critical",
    created_at: "2026-09-02T09:14:05.000Z",
    acknowledged: false,
    snapshot_uri: "/snapshots/sig-001.jpg",
  },
  {
    alert_id: "alr-002",
    plate: "GJ05CD5678",
    camera_id: "cam06",
    camera_name: "SG Highway toll",
    match_state: "probable",
    confidence: 0.82,
    priority: "high",
    created_at: "2026-09-02T09:10:01.000Z",
    acknowledged: false,
    snapshot_uri: "/snapshots/sig-005.jpg",
  },
  {
    alert_id: "alr-003",
    plate: "GJ27MN3456",
    camera_id: "cam02",
    camera_name: "Nehru Bridge east",
    match_state: "exact",
    confidence: 0.86,
    priority: "medium",
    created_at: "2026-09-02T09:04:14.000Z",
    acknowledged: true,
    snapshot_uri: "/snapshots/sig-010.jpg",
  },
  {
    alert_id: "alr-004",
    plate: "GJ18KL9012",
    camera_id: "cam12",
    camera_name: "Odhav ring road",
    match_state: "exact",
    confidence: 0.87,
    priority: "low",
    created_at: "2026-09-02T08:39:17.000Z",
    acknowledged: true,
    snapshot_uri: "/snapshots/sig-015.jpg",
  },
  {
    alert_id: "alr-005",
    plate: "GJ01AB1234",
    camera_id: "cam04",
    camera_name: "Paldi circle",
    match_state: "exact",
    confidence: 0.91,
    priority: "critical",
    created_at: "2026-09-02T09:15:33.000Z",
    acknowledged: false,
    snapshot_uri: "/snapshots/sig-003.jpg",
  },
  {
    alert_id: "alr-006",
    plate: "GJ05CD5678",
    camera_id: "cam08",
    camera_name: "Sabarmati riverfront west",
    match_state: "probable",
    confidence: 0.79,
    priority: "high",
    created_at: "2026-09-02T09:13:08.000Z",
    acknowledged: false,
    snapshot_uri: "/snapshots/sig-007.jpg",
  },
  {
    alert_id: "alr-007",
    plate: "GJ27MN3456",
    camera_id: "cam01",
    camera_name: "Ashram Road junction",
    match_state: "exact",
    confidence: 0.9,
    priority: "medium",
    created_at: "2026-09-02T09:06:43.000Z",
    acknowledged: false,
    snapshot_uri: "/snapshots/sig-008.jpg",
  },
  {
    alert_id: "alr-008",
    plate: "GJ05CD5678",
    camera_id: "cam09",
    camera_name: "Naroda industrial gate",
    match_state: "probable",
    confidence: 0.77,
    priority: "low",
    created_at: "2026-09-02T08:57:33.000Z",
    acknowledged: true,
    snapshot_uri: "/snapshots/sig-012.jpg",
  },
];

// Benchmark report, Canonical 7.3 shape EXACTLY.
//
// Typed `unknown` like the journey fixtures: these are wire payloads and go
// through readBenchmark, so typing them as BenchmarkReport would assert the
// conformance the adapter exists to check.
//
// 7.3 carries a RATE per bucket and no event counts anywhere. The previous
// fixture invented {eligible_events, correct_events} per bucket plus a
// failure_buckets map, none of which 7.3 defines -- which is exactly why a
// conforming response was being discarded.
//
// The numbers encode the finding the 7.15 demo beat exists to show: near
// perfect above 80px, collapsing to nothing below 30. Canonical 7.2 names
// this the number a judge will probe.
export const benchmarkCanonical: unknown = {
  run_id: "detector_v1_001",
  task: "e2e",
  dataset_manifest_sha256:
    "9f2c4a7be1d8035c6ea4b90f17d2c8e5a3b6014fd97e2c8a5b3f0d61e4a7c92b",
  git_commit: "a1b2c3d",
  weights_sha256: "7e1f9c204b6d38a5c0e2f7b91d43a68c5f0b2e7d1a94c63805fe2b7d41c9a608",
  machine: "RTX 5070 Ti 12GB",
  runtime: "torch 2.x + CUDA 12.x",
  source_mode: "file",
  e2e_correct_plate_event_rate: 0.71,
  by_plate_width: {
    ">100": 0.98,
    "80-100": 0.93,
    "60-80": 0.83,
    "40-60": 0.64,
    "30-40": 0.38,
    "<30": 0.0,
  },
  by_condition: { day: 0.82, night: 0.55, blur: 0.41, glare: 0.49, angle: 0.6 },
  diagnostics: {
    precision: 0.9, recall: 0.85, map50: 0.88, small_plate_recall: 0.35,
    ocr_exact_accuracy: 0.79, cer: 0.08,
    fps: 24, latency_p50_ms: 41, latency_p95_ms: 88,
    vram_peak_mb: 7100, real_time_factor: 1.2,
  },
  notes: [],
};

// Same report with dataset_manifest_sha256 ABSENT, and one bucket sent as
// null. A run without the manifest hash cannot be reproduced, so it is not
// evidence -- the panel still renders and says so. A null bucket is 7.3's
// "not run", which the panel distinguishes from a measured 0.0: below 30px
// 0.0 IS the finding, and collapsing the two would delete it.
export const benchmarkNoManifest: unknown = {
  run_id: "detector_v1_002",
  task: "e2e",
  source_mode: "file",
  e2e_correct_plate_event_rate: 0.66,
  by_plate_width: {
    ">100": 0.97,
    "80-100": 0.91,
    "60-80": 0.8,
    "40-60": 0.59,
    "30-40": null,
    "<30": 0.0,
  },
  notes: [],
};
