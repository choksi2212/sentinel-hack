// The ONE file allowed to name non-canonical fields. scripts/guard.mjs
// exempts it by path (`rel.includes("api/adapters.ts")`), which skips every
// rule for this file -- including the `any` ban. CONVENTIONS.md rule 13 still
// applies, so unknown input is narrowed with local guards, never cast to any.
//
// It exists because Canonical Contracts 6.5 and one teammate's manual
// disagree on several field names. We would rather learn which the API
// actually sends at runtime, counted and warned about, than on integration
// day.
import type {
  Alert,
  JourneySegment,
  MatchState,
  SourceMode,
  VehicleSighting,
} from "../types/api";
import type { Camera, CameraStatus } from "../types/ui";
import type { BenchmarkBucketKey, BenchmarkReport } from "../types/ui";
import { BENCHMARK_BUCKET_ORDER } from "../types/ui";

// api.ts declares required_speed_kmh as a required number and is off limits.
// A segment whose speed the API did not send is still a segment worth
// drawing, so the adapter widens exactly that one field and nothing else.
// When the API settles, this type disappears and readSegment returns
// JourneySegment again.
export type AdaptedJourneySegment = Omit<
  JourneySegment,
  "required_speed_kmh" | "feasible"
> & {
  required_speed_kmh?: number;
  // Absent means the backend never assessed this segment. That is a third
  // state, distinct from assessed-and-passed, and it is rendered as one.
  feasible?: boolean;
};

// Two counters, not one. A rename ("the API spells it is_feasible") is
// cosmetic; a discard ("3 sightings were unusable") means we are losing
// evidence. Collapsed into a single map the second hides inside the first.
//
// Rendered on the System Status screen later. Exported as live Maps so a
// panel can read counts without the adapter knowing anything about the UI.
export const drift: {
  fallbacks: Map<string, number>;
  discards: Map<string, number>;
} = { fallbacks: new Map(), discards: new Map() };

export interface DriftEntry {
  name: string;
  count: number;
}

export interface DriftSummary {
  fallbacks: DriftEntry[];
  discards: DriftEntry[];
  totalFallbacks: number;
  totalDiscards: number;
}

const warnedFallbacks = new Set<string>();
const warnedDiscards = new Set<string>();

// A non-fatal substitution: the record survived, a name or a default moved.
// Increments always, warns once per distinct name. A silent adapter would
// hide contract drift, which is the opposite of why this file exists.
function recordFallback(name: string, detail: string): void {
  drift.fallbacks.set(name, (drift.fallbacks.get(name) ?? 0) + 1);
  if (!warnedFallbacks.has(name)) {
    warnedFallbacks.add(name);
    console.warn(`[trinetra] drift, fallback: ${name} -- ${detail}`);
  }
}

// A dropped record, keyed on the first missing required field in canonical
// order. Exactly one increment per discarded record: counting every missing
// field would push the total past the number of records actually lost.
function recordDiscard(field: string, detail: string): void {
  drift.discards.set(field, (drift.discards.get(field) ?? 0) + 1);
  if (!warnedDiscards.has(field)) {
    warnedDiscards.add(field);
    console.warn(`[trinetra] drift, discarded record: ${field} -- ${detail}`);
  }
}

function toEntries(counts: Map<string, number>): DriftEntry[] {
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

// Snapshot for the status screen. Totals are included so a panel can render
// "3 sightings discarded" without summing a map itself.
export function driftSummary(): DriftSummary {
  const fallbacks = toEntries(drift.fallbacks);
  const discards = toEntries(drift.discards);
  return {
    fallbacks,
    discards,
    totalFallbacks: fallbacks.reduce((sum, entry) => sum + entry.count, 0),
    totalDiscards: discards.reduce((sum, entry) => sum + entry.count, 0),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

const SOURCE_MODES: Record<SourceMode, true> = {
  live_rtsp: true,
  live_hls: true,
  file: true,
  frames: true,
  synthetic: true,
};

const MATCH_STATES: Record<MatchState, true> = {
  exact: true,
  probable: true,
  low_confidence: true,
  unreadable: true,
};

function readSourceMode(value: unknown): SourceMode | null {
  return typeof value === "string" && Object.hasOwn(SOURCE_MODES, value)
    ? (value as SourceMode)
    : null;
}

function readMatchState(value: unknown): MatchState | null {
  return typeof value === "string" && Object.hasOwn(MATCH_STATES, value)
    ? (value as MatchState)
    : null;
}

export function readSighting(raw: unknown): VehicleSighting | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "a sighting was not a JSON object");
    return null;
  }

  const sighting_id = readString(raw["sighting_id"]);
  const camera_id = readString(raw["camera_id"]);
  const first_seen_at = readString(raw["first_seen_at"]);
  const last_seen_at = readString(raw["last_seen_at"]);
  const source_mode = readSourceMode(raw["source_mode"]);
  const match_state = readMatchState(raw["match_state"]);

  // Required, and none of them has an honest default. "synthetic" would
  // discredit a real sighting, "live_rtsp" would fake provenance, and an
  // invented id or interval is a fabricated observation. Drop and count.
  if (
    sighting_id === null ||
    camera_id === null ||
    first_seen_at === null ||
    last_seen_at === null ||
    source_mode === null ||
    match_state === null
  ) {
    const field =
      sighting_id === null
        ? "sighting_id"
        : camera_id === null
          ? "camera_id"
          : first_seen_at === null
            ? "first_seen_at"
            : last_seen_at === null
              ? "last_seen_at"
              : source_mode === null
                ? "source_mode"
                : "match_state";
    recordDiscard(
      field,
      `a sighting was discarded: ${field} was missing or unrecognised`,
    );
    return null;
  }

  // Soft default: camera_id is a real observed value, not an invention.
  let camera_name = readString(raw["camera_name"]);
  if (camera_name === null) {
    recordFallback("camera_name", "camera_name missing; falling back to camera_id");
    camera_name = camera_id;
  }

  let evidence_count = readNumber(raw["evidence_count"]);
  if (evidence_count === null) {
    recordFallback("evidence_count", "evidence_count missing; falling back to 0");
    evidence_count = 0;
  }

  // Prefer canonical `plate`. Fall back to the manual's `plate_normalized`
  // only when `plate` is absent -- a present-but-null plate is a real
  // "we could not read it", not a gap to fill from another key.
  let plate: string | null = null;
  if ("plate" in raw) {
    plate = readString(raw["plate"]);
  } else if ("plate_normalized" in raw) {
    const fallback = readString(raw["plate_normalized"]);
    if (fallback !== null) {
      recordFallback(
        "plate_normalized",
        "API sent plate_normalized; Canonical 6.5 calls this field plate",
      );
      plate = fallback;
    }
  }

  return {
    sighting_id,
    camera_id,
    camera_name,
    lat: readNumber(raw["lat"]),
    lon: readNumber(raw["lon"]),
    first_seen_at,
    last_seen_at,
    source_pts_ms: readNumber(raw["source_pts_ms"]),
    source_mode,
    plate,
    plate_raw: readString(raw["plate_raw"]),
    confidence: readNumber(raw["confidence"]),
    match_state,
    evidence_count,
    plate_width_px: readNumber(raw["plate_width_px"]),
    vehicle_type: readString(raw["vehicle_type"]),
    snapshot_uri: readString(raw["snapshot_uri"]),
  };
}

export interface AdaptedJourney {
  plate: string;
  disclaimer: string;
  // The server's claim, carried verbatim when present. It is NOT reconciled
  // against sightings.length and never overwritten: if the two disagree, that
  // disagreement is information, and silently replacing one with the other
  // would hide it.
  //
  // Optional, and absence is NOT a discard trigger. The nullable-return
  // principle exists to stop the adapter inventing a value, not to throw away
  // values that are present: a response with usable sightings, real cameras,
  // real times and a verbatim disclaimer must not be discarded over a missing
  // cross-check number. Count is the server's claim about the evidence, not
  // the evidence.
  sighting_count?: number;
  // Usable only, sorted ascending by first_seen_at.
  sightings: VehicleSighting[];
  // Usable only. Positional nulls are NOT preserved -- see the matching note
  // below, position is not how connectors are matched.
  segments: AdaptedJourneySegment[];
  discardedSightingCount: number;
  discardedSegmentCount: number;
}

// NOTE FOR THE SCREEN, not implemented here.
//
// Connectors are matched to node pairs by CAMERA IDENTITY, never by array
// position. For each adjacent pair of sorted sightings, find the segment whose
// from_camera_id and to_camera_id equal that pair's camera_ids.
//   - A pair with no matching segment renders the gap marker.
//   - A segment matching no pair is counted and shown, never dropped.
// Position cannot be used: readSegment discards unusable segments, so indices
// no longer line up with node pairs, and a discarded segment is exactly the
// case the gap marker exists for.
export function readJourney(raw: unknown): AdaptedJourney | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "a journey was not a JSON object");
    return null;
  }

  const plate = readString(raw["plate"]);
  if (plate === null) {
    recordDiscard("plate", "a journey was discarded: plate was missing");
    return null;
  }

  // Canonical 6.3 makes disclaimer mandatory precisely so that no client can
  // render a journey without it. A response lacking it is violating the
  // contract, and rendering anyway is the exact failure the mandate prevents.
  // The screen becomes an error state; we never invent a disclaimer.
  const disclaimer = readString(raw["disclaimer"]);
  if (disclaimer === null || disclaimer === "") {
    recordDiscard(
      "disclaimer",
      "a journey was discarded: disclaimer was missing or empty, and a journey must never render without it",
    );
    return null;
  }

  // Absent is fine. Not defaulted to 0, which would put a count on screen the
  // server never claimed; not discarded either, which would throw away a whole
  // usable journey to avoid rendering a cross-check number. The key is simply
  // omitted and the screen shows nothing where the count would go.
  const sighting_count = readNumber(raw["sighting_count"]);

  const rawSightings: unknown[] = Array.isArray(raw["sightings"])
    ? (raw["sightings"] as unknown[])
    : [];
  const rawSegments: unknown[] = Array.isArray(raw["segments"])
    ? (raw["segments"] as unknown[])
    : [];

  const sightings: VehicleSighting[] = [];
  let discardedSightingCount = 0;
  for (const entry of rawSightings) {
    const sighting = readSighting(entry);
    if (sighting === null) discardedSightingCount += 1;
    else sightings.push(sighting);
  }

  // Sorted client-side, always. Nothing in the contract mirror or anywhere in
  // this repo enforces ordering, and TypeScript cannot express it. If the API
  // ever returns these unordered, the timeline draws nodes in one order while
  // connectors -- carrying their own from_time/to_time -- describe another,
  // and the two disagree silently: a journey that never happened. Date.parse
  // rather than string compare so an offset-bearing timestamp still sorts.
  sightings.sort(
    (a, b) => Date.parse(a.first_seen_at) - Date.parse(b.first_seen_at),
  );

  const segments: AdaptedJourneySegment[] = [];
  let discardedSegmentCount = 0;
  for (const entry of rawSegments) {
    const segment = readSegment(entry);
    if (segment === null) discardedSegmentCount += 1;
    else segments.push(segment);
  }

  const journey: AdaptedJourney = {
    plate,
    disclaimer,
    sightings,
    segments,
    discardedSightingCount,
    discardedSegmentCount,
  };
  if (sighting_count !== null) journey.sighting_count = sighting_count;
  return journey;
}

export function readSegment(raw: unknown): AdaptedJourneySegment | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "a segment was not a JSON object");
    return null;
  }

  const from_camera_id = readString(raw["from_camera_id"]);
  const to_camera_id = readString(raw["to_camera_id"]);
  const from_time = readString(raw["from_time"]);
  const to_time = readString(raw["to_time"]);
  const straight_line_km = readNumber(raw["straight_line_km"]);
  const elapsed_seconds = readNumber(raw["elapsed_seconds"]);

  // A distance or an interval we never measured cannot be defaulted to 0
  // without asserting a measurement. Drop and count.
  if (
    from_camera_id === null ||
    to_camera_id === null ||
    from_time === null ||
    to_time === null ||
    straight_line_km === null ||
    elapsed_seconds === null
  ) {
    const field =
      from_camera_id === null
        ? "from_camera_id"
        : to_camera_id === null
          ? "to_camera_id"
          : from_time === null
            ? "from_time"
            : to_time === null
              ? "to_time"
              : straight_line_km === null
                ? "straight_line_km"
                : "elapsed_seconds";
    recordDiscard(field, `a segment was discarded: ${field} was missing`);
    return null;
  }

  const segment: AdaptedJourneySegment = {
    from_camera_id,
    to_camera_id,
    from_time,
    to_time,
    straight_line_km,
    elapsed_seconds,
    note: readString(raw["note"]),
  };

  // Three states, not two. Canonical 6.3 spells this `feasible`.
  //
  // An `is_feasible` fallback used to sit here. Removed: that spelling appears
  // NOWHERE in the canonical contract -- not in 6.3's segment shape, not as a
  // database column, nowhere. It came from an execution manual the contract
  // supersedes, and a name only a superseded manual uses is not wire-format
  // risk, it is dead code that made the drift counters harder to trust.
  //
  // Absent leaves the key ABSENT. It used to default to true, which styled a
  // segment the backend never assessed identically to one it assessed and
  // passed -- asserting a check that never ran, the same class of error as a
  // solid connector. Absence records no fallback either: it is not a rename.
  const canonicalFeasible = readBoolean(raw["feasible"]);
  if (canonicalFeasible !== null) {
    segment.feasible = canonicalFeasible;
  }

  // Never fabricated. The key stays absent so the UI renders
  // "Speed unavailable" rather than a confident wrong number.
  //
  // No fallback is recorded for absence, for the same reason feasible:absent
  // records none. Absence is not a rename, there is no alternative spelling of
  // this field to detect, and it already renders as a visible state -- which
  // is a better signal than a counter. Counting it inflated the drift number
  // with something that is not drift.
  const required_speed_kmh = readNumber(raw["required_speed_kmh"]);
  if (required_speed_kmh !== null) {
    segment.required_speed_kmh = required_speed_kmh;
  }

  return segment;
}

// ---------------------------------------------------------------------------
// Benchmark report.
//
// Canonical 7.3 defines the bucket map as `by_plate_width` with these six
// keys. NOTE: 7.3's shape and what this reader expects differ in more than a
// name -- 7.3 carries a single rate per bucket and no event counts at all,
// while this reader wants {eligible_events, correct_events}. That is an open
// question for Parth, recorded rather than papered over; it is not something
// a fallback can fix.
const BENCHMARK_BUCKET_KEYS: Record<BenchmarkBucketKey, true> = {
  ">100": true,
  "80-100": true,
  "60-80": true,
  "40-60": true,
  "30-40": true,
  "<30": true,
};

// 7.3 sends one RATE per bucket, 0.0-1.0, or null where the bucket was not
// run. Both are meaningful and distinct: null is "not measured", 0.0 is
// "measured, nothing correct" -- and 0.0 in the <30 bucket IS the finding, so
// collapsing the two would delete the point of the screen.
//
// A rate outside [0,1] is refused rather than rendered: 7.1 defines it as
// correct events over eligible events, so 1.4 is not a number this metric can
// produce and showing it would launder a backend bug into a claim.
function readRate(value: unknown): number | null {
  const n = readNumber(value);
  if (n === null) return null;
  if (n < 0 || n > 1) {
    recordFallback(
      "benchmark_rate_range",
      `benchmark rate ${n} is outside [0,1]; Canonical 7.1 defines it as a ratio, value ignored`,
    );
    return null;
  }
  return n;
}

export function readBenchmark(raw: unknown): BenchmarkReport | null {
  if (!isRecord(raw)) return null;

  const run_id = readString(raw["run_id"]);
  if (run_id === null) {
    // Without a run id the report cannot be referred to, so it cannot be
    // checked. That is a discard, not a fallback.
    recordDiscard("run_id", "benchmark report has no run_id and cannot be cited");
    return null;
  }

  // NO event-count check. This used to require eligible_events and
  // correct_events and discard the report without them -- and 7.3 defines
  // neither, so every conforming response was thrown away with a discard
  // keyed "eligible_events". Measured, not assumed: a strictly 7.3-shaped
  // payload returned null and the panel rendered its refusal text.
  //
  // Canonical 7.3 spells the bucket map `by_plate_width`, and that is the only
  // spelling read.
  const rawBuckets = raw["by_plate_width"];

  const by_plate_width: Partial<Record<BenchmarkBucketKey, number | null>> = {};
  if (isRecord(rawBuckets)) {
    for (const key of BENCHMARK_BUCKET_ORDER) {
      // Present-and-null is recorded as null (not measured); absent stays
      // absent. The panel renders those differently.
      if (key in rawBuckets) by_plate_width[key] = readRate(rawBuckets[key]);
    }
    // A key we have never heard of is drift worth counting, not worth crashing
    // on. It is not rendered, because there is no column for it.
    for (const key of Object.keys(rawBuckets)) {
      if (!Object.hasOwn(BENCHMARK_BUCKET_KEYS, key)) {
        recordFallback(
          "benchmark_bucket_key",
          `unrecognised plate-width bucket "${key}", ignored`,
        );
      }
    }
  }

  // 7.3 spells this dataset_manifest_sha256. We previously read
  // `manifest_sha256` -- a name the contract does not use -- so the hash was
  // always absent and the panel always claimed the run was unreproducible.
  //
  // Absent remains a legitimate, reportable state: the panel says so on screen
  // rather than hiding it. Not a discard -- the rest of the report is still
  // worth reading, it just is not reproducible.
  const dataset_manifest_sha256 = readString(raw["dataset_manifest_sha256"]);

  // 7.3's headline scalar. Read, but deliberately rendered below the buckets
  // and labelled -- 7.2 forbids reporting an accuracy number as a single
  // average, and this is exactly one.
  const e2e_correct_plate_event_rate = readRate(raw["e2e_correct_plate_event_rate"]);

  return {
    run_id,
    dataset_manifest_sha256,
    e2e_correct_plate_event_rate,
    by_plate_width,
  };
}
// ---------------------------------------------------------------------------
// Cameras, alerts and search.
//
// Until now only journey, its segments and the benchmark had adapters. Cameras,
// alerts and search called apiGet with a type assertion, so NOTHING absorbed a
// rename on them -- and last session's hostile harness turned two of those
// assertions into white screens. These three close that gap, following
// readSighting's pattern exactly rather than inventing a second style.

const CAMERA_STATUSES: Record<CameraStatus, true> = {
  online: true,
  offline: true,
  degraded: true,
  unknown: true,
};

function readCameraStatus(value: unknown): CameraStatus | null {
  return typeof value === "string" && Object.hasOwn(CAMERA_STATUSES, value)
    ? (value as CameraStatus)
    : null;
}

export function readCamera(raw: unknown): Camera | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "a camera was not a JSON object");
    return null;
  }

  // camera_id is the ONLY discard here. Everything else about a camera can
  // degrade to an honest placeholder, but without an id the card cannot be
  // keyed and no sighting can be resolved against it.
  let camera_id = readString(raw["camera_id"]);
  if (camera_id === null) {
    const fallback = readString(raw["external_camera_id"]);
    if (fallback !== null) {
      recordFallback(
        "external_camera_id",
        "API sent external_camera_id; Canonical 6.4 calls this field camera_id",
      );
      camera_id = fallback;
    }
  }
  if (camera_id === null) {
    recordDiscard(
      "camera_id",
      "a camera was discarded: no id under either spelling, so nothing could reference it",
    );
    return null;
  }

  // Canonical 6.4 calls this `name`. Sightings and alerts call the same idea
  // `camera_name`, and at least one manual uses that spelling here too.
  let name = readString(raw["name"]);
  if (name === null) {
    const alt = readString(raw["camera_name"]);
    if (alt !== null) {
      recordFallback(
        "camera_name",
        "API sent camera_name on a camera; Canonical 6.4 calls this field name",
      );
      name = alt;
    }
  }
  if (name === null) {
    // Soft default: camera_id is a real observed value, not an invention.
    recordFallback("name", "camera name missing; falling back to camera_id");
    name = camera_id;
  }

  // Canonical 6.4 returns `status`, and 5.1 backs it with a CHECK constraint
  // on the same name. A `health_state` fallback was removed: that spelling is
  // in no part of the contract, wire or database.
  const statusValue = readString(raw["status"]);
  let status = readCameraStatus(statusValue);
  if (status === null) {
    // NOT a discard. A camera we cannot classify still exists, still appears in
    // the catalogue, and still has historical sightings pointing at it --
    // dropping it would lose a real camera to a vocabulary disagreement.
    // "unknown" is a real member of the union and means exactly this.
    recordFallback(
      "camera_status",
      `unrecognised camera status ${JSON.stringify(statusValue)}; rendering as unknown`,
    );
    status = "unknown";
  }

  // readNumber returns null for an ABSENT key as well as a null one, so absent
  // and null arrive here identically -- which is the whole point. Both mean
  // "not surveyed", both are counted as unplaced, neither is ever plotted and
  // neither is a discard.
  return {
    camera_id,
    name,
    lat: readNumber(raw["lat"]),
    lon: readNumber(raw["lon"]),
    status,
    last_seen_at: readString(raw["last_seen_at"]),
  };
}

// api.ts is a canonical mirror and off limits, but two of Alert's fields need a
// third state the mirror cannot express. Same technique as
// AdaptedJourneySegment: widen exactly what must widen, nothing else.
//
//   priority absent  -> we do not know how urgent this is. NOT "medium":
//                       inventing an assessment nobody made is worse than
//                       saying we lack one.
//   plate null       -> we could not read it. Alert.plate is a bare `string`,
//                       which cannot represent that, and a blank cell is
//                       forbidden. Rendered as "Unreadable".
export type AdaptedAlert = Omit<Alert, "priority" | "plate"> & {
  priority?: Alert["priority"];
  plate: string | null;
};

const ALERT_PRIORITIES: Record<Alert["priority"], true> = {
  low: true,
  medium: true,
  high: true,
  critical: true,
};

// Alert.match_state EXCLUDES low_confidence and unreadable, and Canonical 5.1
// backs that with a database CHECK. A value outside the permitted pair is not a
// new render path to build; it is a record the database says cannot exist.
const ALERT_MATCH_STATES: Record<Alert["match_state"], true> = {
  exact: true,
  probable: true,
};

export function readAlert(raw: unknown): AdaptedAlert | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "an alert was not a JSON object");
    return null;
  }

  const alert_id = readString(raw["alert_id"]);
  if (alert_id === null) {
    // Both dedup and acknowledgement key on this. Without it an alert cannot be
    // merged, cannot be acknowledged, and would duplicate on every refetch.
    recordDiscard(
      "alert_id",
      "an alert was discarded: no alert_id, so it cannot be deduped or acknowledged",
    );
    return null;
  }

  const matchValue = readString(raw["match_state"]);
  if (matchValue === null || !Object.hasOwn(ALERT_MATCH_STATES, matchValue)) {
    recordDiscard(
      "match_state",
      `an alert was discarded: match_state ${JSON.stringify(matchValue)} is outside the permitted set, which the database enforces with a CHECK`,
    );
    return null;
  }
  const match_state = matchValue as Alert["match_state"];

  let plate = readString(raw["plate"]);
  if (plate === null) {
    const fallback = readString(raw["plate_normalized"]);
    if (fallback !== null) {
      recordFallback(
        "plate_normalized",
        "API sent plate_normalized on an alert; Canonical 6.5 calls this field plate",
      );
      plate = fallback;
    }
  }

  let camera_id = readString(raw["camera_id"]);
  if (camera_id === null) {
    const fallback = readString(raw["external_camera_id"]);
    if (fallback !== null) {
      recordFallback(
        "external_camera_id",
        "API sent external_camera_id on an alert; Canonical 6.5 calls this field camera_id",
      );
      camera_id = fallback;
    }
  }
  if (camera_id === null) {
    // Not a discard: the alert is still actionable from its plate, time and
    // camera name. The id is what we lose, and the screen does not render it.
    recordFallback("camera_id", "alert camera_id missing under either spelling");
    camera_id = "";
  }

  let camera_name = readString(raw["camera_name"]);
  if (camera_name === null) {
    recordFallback("camera_name", "alert camera_name missing; falling back to camera_id");
    camera_name = camera_id;
  }

  // Canonical 6.5 spells this `priority`, and 5.7's alerts table uses the same
  // name with a CHECK on the same four values. A `severity` fallback was
  // removed: that spelling exists in no part of the contract.
  const priorityValue = readString(raw["priority"]);

  // acknowledged is a BOOLEAN. Canonical 6.5 sends one and 5.7 stores one
  // (`acknowledged BOOLEAN NOT NULL DEFAULT false`) -- there is no timestamp
  // column anywhere in the schema, so no serializer can leak one.
  //
  // An `acknowledged_at` fallback was removed for that reason. It was the most
  // defensible of the six on consequence -- misreading it means an operator
  // re-acknowledges work somebody already did -- but consequence is not
  // evidence, and the contract offers no path by which that field arrives.
  let acknowledged = readBoolean(raw["acknowledged"]);
  if (acknowledged === null) {
    recordFallback("acknowledged", "alert acknowledged missing; falling back to false");
    acknowledged = false;
  }

  const created_at = readString(raw["created_at"]);
  if (created_at === null) {
    recordDiscard(
      "created_at",
      "an alert was discarded: no created_at, so it cannot be ordered in the feed",
    );
    return null;
  }

  const alert: AdaptedAlert = {
    alert_id,
    plate,
    camera_id,
    camera_name,
    match_state,
    confidence: readNumber(raw["confidence"]),
    created_at,
    acknowledged,
    snapshot_uri: readString(raw["snapshot_uri"]),
  };

  if (priorityValue !== null && Object.hasOwn(ALERT_PRIORITIES, priorityValue)) {
    alert.priority = priorityValue as Alert["priority"];
  } else {
    // Left ABSENT rather than defaulted. The row renders "Unknown priority"
    // with no colour, because a colour is an assessment and nobody made one.
    recordFallback(
      "alert_priority",
      `unrecognised alert priority ${JSON.stringify(priorityValue)}; rendering as unknown`,
    );
  }

  return alert;
}

// The server's own count is carried through VERBATIM and never reconciled with
// results.length. If they disagree that is the API's claim against the API's
// data, and silently rewriting one to match the other would hide exactly the
// bug worth seeing. Discards are counted separately and reach the screen,
// because a quietly shorter list asserts fewer sightings than were sent.
export interface AdaptedSearchResponse {
  query: { plate: string; normalized: string; fuzzy: boolean };
  count: number;
  results: VehicleSighting[];
  candidates: Array<VehicleSighting & { distance: number }>;
  discardedResults: number;
  discardedCandidates: number;
}

export function readSearchResponse(raw: unknown): AdaptedSearchResponse | null {
  if (!isRecord(raw)) {
    recordDiscard("(not an object)", "a search response was not a JSON object");
    return null;
  }

  const rawQuery = isRecord(raw["query"]) ? raw["query"] : {};
  const fuzzy = readBoolean(rawQuery["fuzzy"]) ?? false;
  const query = {
    plate: readString(rawQuery["plate"]) ?? "",
    normalized: readString(rawQuery["normalized"]) ?? "",
    fuzzy,
  };

  // Every result goes through the EXISTING readSighting, so the
  // plate_normalized fallback that already rescues a journey sighting rescues a
  // search result too. One field, one rename, one outcome -- it must not depend
  // on which screen asked.
  const rawResults: unknown[] = Array.isArray(raw["results"]) ? (raw["results"] as unknown[]) : [];
  const results: VehicleSighting[] = [];
  let discardedResults = 0;
  for (const item of rawResults) {
    const sighting = readSighting(item);
    if (sighting === null) discardedResults += 1;
    else results.push(sighting);
  }

  const candidates: Array<VehicleSighting & { distance: number }> = [];
  let discardedCandidates = 0;
  const rawCandidates: unknown[] = Array.isArray(raw["candidates"])
    ? (raw["candidates"] as unknown[])
    : [];
  // Candidates on a non-fuzzy search are ignored entirely, not rendered as an
  // empty region. An exact search that shows a "Candidates" heading invites the
  // reading that fuzzy matching ran and found nothing, which is not what
  // happened.
  if (fuzzy) {
    for (const item of rawCandidates) {
      const sighting = readSighting(item);
      if (sighting === null) {
        discardedCandidates += 1;
        continue;
      }
      const distance = isRecord(item) ? readNumber(item["distance"]) : null;
      if (distance === null) {
        recordFallback("distance", "a fuzzy candidate had no distance; falling back to 0");
      }
      candidates.push({ ...sighting, distance: distance ?? 0 });
    }
  } else if (rawCandidates.length > 0) {
    recordFallback(
      "candidates",
      `${rawCandidates.length} candidates arrived on a non-fuzzy search and were ignored`,
    );
  }

  const count = readNumber(raw["count"]);
  if (count === null) {
    recordFallback("count", "search response had no count; falling back to the number of results");
  } else if (count !== results.length + discardedResults) {
    // Logged, never corrected. The server's claim stays on screen as the
    // server's claim.
    recordFallback(
      "count_mismatch",
      `server reported count ${count} but sent ${rawResults.length} results`,
    );
  }

  return {
    query,
    count: count ?? results.length,
    results,
    candidates,
    discardedResults,
    discardedCandidates,
  };
}
// Array wrappers, so the four camera call sites and the alert list do not each
// reimplement "map, drop the nulls, keep the counts". A non-array body is a
// discard rather than an empty list: "the API sent something that was not a
// list" and "the API sent no rows" are different facts.
export function readCameras(raw: unknown): Camera[] {
  if (!Array.isArray(raw)) {
    recordDiscard("(cameras not an array)", "the cameras response was not a JSON array");
    return [];
  }
  const out: Camera[] = [];
  for (const item of raw) {
    const camera = readCamera(item);
    if (camera !== null) out.push(camera);
  }
  return out;
}

export function readAlerts(raw: unknown): AdaptedAlert[] {
  if (!Array.isArray(raw)) {
    recordDiscard("(alerts not an array)", "the alerts response was not a JSON array");
    return [];
  }
  const out: AdaptedAlert[] = [];
  for (const item of raw) {
    const alert = readAlert(item);
    if (alert !== null) out.push(alert);
  }
  return out;
}