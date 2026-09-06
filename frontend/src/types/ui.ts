// NOT canonical. Derived from Canonical Contracts 6.4 (cameras response
// shape) and 5.1 (the status CHECK constraint). These types are ours,
// which is why they are not in api.ts.
import type { VehicleSighting } from "./api";

export type CameraStatus = "online" | "offline" | "degraded" | "unknown";

export interface Camera {
  camera_id: string;
  name: string;
  lat: number | null;
  lon: number | null;
  status: CameraStatus;
  last_seen_at: string | null;
}

export interface SearchResponse {
  query: { plate: string; normalized: string; fuzzy: boolean };
  count: number;
  results: VehicleSighting[];
  candidates?: Array<VehicleSighting & { distance: number }>;
}

export type ApiErrorCode =
  | "VALIDATION_FAILED"
  | "UNKNOWN_CAMERA"
  | "UNSUPPORTED_SCHEMA_VERSION"
  | "DUPLICATE_EVENT"
  | "NOT_FOUND"
  | "DEPENDENCY_UNAVAILABLE"
  | "INTERNAL_ERROR";

export interface ApiErrorBody {
  error: {
    code: ApiErrorCode;
    message: string;
    field?: string;
    request_id: string;
    details?: Record<string, unknown>;
  };
}

export type WsStatus = "connecting" | "connected" | "reconnecting" | "offline";

// Which set of mock payloads the handlers serve. "hostile" serves every
// unresolved Canonical-vs-manual field conflict at once, so the frontend can be
// measured against the API it might actually meet rather than the one the
// contract promises. Not a secret; a two-value switch.
export type MockShape = "canonical" | "hostile";

// NOT in Canonical 6.5. No Watchlist or WatchlistEntry type exists there, and
// api.ts is a mirror that must not be extended on our judgement, so the shape
// lives here until the contract documents it -- at which point this moves to
// api.ts and this comment goes with it.
//
// `priority` reuses Alert's four levels rather than inventing a parallel scale,
// because a watchlist entry's priority is what an alert raised from it will
// carry. `id` is here because endpoints.watchlistItem takes an id, not a plate,
// so the soft delete has to address the entry rather than the vehicle.
export type WatchlistPriority = "low" | "medium" | "high" | "critical";

export interface WatchlistEntry {
  id: string;
  plate: string;
  reason: string;
  priority: WatchlistPriority;
}

// What the add form sends. No id: the server assigns it.
export interface WatchlistDraft {
  plate: string;
  reason: string;
  priority: WatchlistPriority;
}

// Canonical 7.3 defines the benchmark report. This mirrors the subset of 7.3
// the UI renders; it lives here rather than api.ts because api.ts mirrors 6.5
// only, and 7.3 is a different section.
//
// The shape was previously INFERRED from an execution manual and was wrong in
// a way no fallback could fix: it expected {eligible_events, correct_events}
// counts per bucket, and 7.3 carries a single RATE per bucket and no event
// counts anywhere. A conforming response was discarded outright. Rewritten
// against 7.3 as written.
export type BenchmarkBucketKey = ">100" | "80-100" | "60-80" | "40-60" | "30-40" | "<30";

// Bucket order is fixed here, widest plate first, so the table reads the way the
// finding does: accuracy falls off as the plate gets smaller.
export const BENCHMARK_BUCKET_ORDER: BenchmarkBucketKey[] = [
  ">100",
  "80-100",
  "60-80",
  "40-60",
  "30-40",
  "<30",
];

export interface BenchmarkReport {
  run_id: string;
  // 7.3 spells this dataset_manifest_sha256. Nullable on purpose: a report
  // without the manifest hash is not reproducible, so it is not evidence --
  // but that is a fact to SHOW, not a reason to discard the record.
  dataset_manifest_sha256: string | null;
  // 7.3's headline scalar. Rendered small and BELOW the buckets, never as the
  // headline -- see 7.2, which forbids reporting an accuracy number as a
  // single average. A value per bucket; null where 7.3 sends null (not run).
  e2e_correct_plate_event_rate: number | null;
  by_plate_width: Partial<Record<BenchmarkBucketKey, number | null>>;
}