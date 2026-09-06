// Every path comes from src/api/endpoints.ts. Nothing here hardcodes a URL,
// so when a contested path is settled the change stays a one-line edit there.
//
// Handlers are registered against the resolved apiBaseUrl rather than a bare
// relative path: apiGet requests VITE_API_BASE_URL, which is a different
// origin from the dev server, and a relative pattern would silently fail to
// match it.
import { HttpResponse, delay, http } from "msw";
import { apiBaseUrl } from "../api/client";
import { endpoints } from "../api/endpoints";
import { appMode, isDemo } from "../lib/appMode";
import type { Alert } from "../types/api";
import type { ApiErrorBody, MockShape, SearchResponse } from "../types/ui";
import {
  driftedJourney,
  hostileAlerts,
  hostileCameras,
  hostileJourney,
  hostileSearch,
} from "./fixtures/drifted";
import {
  alerts,
  benchmarkCanonical,
  benchmarkNoManifest,
  cameras,
  journeyAmbiguousPair,
  journeyFourCamera,
  journeyMissingDisclaimer,
  journeyNoCount,
  journeyNoTimeMatch,
  journeyRepeatedPair,
  journeySingleSighting,
  journeyUndrawableConnector,
  searchFuzzy,
  sightings,
  systemStatus,
} from "./fixtures";

function at(path: string): string {
  return new URL(path, apiBaseUrl).toString();
}

// Which payload shape this session serves. Read once at module load: the shape
// is a deployment decision for a whole run, not something to flip mid-session.
//
// "hostile" serves the execution manuals' spelling for every unresolved
// conflict at once. The drifted spellings themselves live only in
// fixtures/drifted.ts, which is the guard's noncanon exemption; this file
// references them by name and never spells one.
const MOCK_SHAPE: MockShape =
  import.meta.env.VITE_MOCK_SHAPE === "hostile" ? "hostile" : "canonical";
const HOSTILE = MOCK_SHAPE === "hostile";

// Demo mode pins the payloads so a rehearsal is identical every run.
const DEMO = isDemo(appMode());

if (HOSTILE) {
  console.warn(
    "[trinetra] MOCK SHAPE = HOSTILE. Serving the execution manuals' field " +
      "names for every unresolved contract conflict. This is a rehearsal for " +
      "integration day, not a bug.",
  );
}

// Module-level so it survives across requests within one page session.
let alertsRequestCount = 0;

// Acknowledgement has to be STATEFUL or the verification is theatre: an
// optimistic flip followed by an invalidate would refetch acknowledged:false
// and the row would silently revert, which looks exactly like a rollback.
const acknowledgedIds = new Set<string>();

// Acknowledging THIS id always fails with a canonical 500, so the rollback path
// can be exercised from the browser without editing code. Chosen as a real
// fixture id rather than a synthetic one so it is visible in the list and can
// actually be clicked.
const ACK_FAILS_FOR = "alr-005";

// Watchlist. Soft delete: entries are marked, never spliced out, because the
// backend's DELETE is a soft delete and a mock that hard-deletes would hide the
// difference between "gone from the list" and "gone from the table".
interface MockWatchlistRow {
  id: string;
  plate: string;
  reason: string;
  priority: "low" | "medium" | "high" | "critical";
  deleted: boolean;
}

let watchlistSeq = 3;
const watchlistRows: MockWatchlistRow[] = [
  { id: "wl-001", plate: "GJ01AB1234", reason: "Stolen vehicle report", priority: "critical", deleted: false },
  { id: "wl-002", plate: "GJ05CD5678", reason: "Traffic violation follow-up", priority: "medium", deleted: false },
  { id: "wl-003", plate: "GJ18KL9012", reason: "Person of interest", priority: "high", deleted: false },
];

// Plate-selectable failures, so both error paths are reachable from the form
// with no code edit. Documented in WORKLOG and in the browser checklist.
const WATCHLIST_409_PLATE = "GJ09XX0409";
const WATCHLIST_500_PLATE = "GJ09XX0500";

function errorBody(
  code: ApiErrorBody["error"]["code"],
  message: string,
  requestId: string,
): ApiErrorBody {
  return { error: { code, message, request_id: requestId } };
}

export const handlers = [
  // Deliberately slow. Instant mocks let a loading state go unbuilt until the
  // day a real network makes it mandatory.
  http.get(at(endpoints.cameras()), async () => {
    await delay(120);
    if (HOSTILE) return HttpResponse.json(hostileCameras as object);
    return HttpResponse.json(cameras);
  }),

  http.get(at(endpoints.search()), async ({ request }) => {
    await delay(250);
    // Returned before any filtering: the point is the SHAPE the screen receives,
    // not which rows match.
    if (HOSTILE) return HttpResponse.json(hostileSearch as object);
    const params = new URL(request.url).searchParams;

    const plate = params.get("plate");
    const fuzzy = params.get("fuzzy") === "true";
    const cameraId = params.get("camera_id");

    // No plate filter means "everything we hold", which is what the map
    // needs. Canonical 6.2 fixes the response shape, so this stays a
    // SearchResponse rather than becoming a second, map-shaped payload.
    if (plate === null || plate === "") {
      const all = cameraId
        ? sightings.filter((s) => s.camera_id === cameraId)
        : sightings;
      const response: SearchResponse = {
        query: { plate: "", normalized: "", fuzzy },
        count: all.length,
        results: all,
      };
      return HttpResponse.json(response);
    }

    // Normalisation is done for real rather than echoed back unchanged, so
    // the screen can honestly show that "GJ 01 ab 1234" was searched as
    // "GJ01AB1234". A mock that skipped this would make the normalisation
    // line look like it works when it does not.
    const normalized = plate.toUpperCase().replace(/[^A-Z0-9]/g, "");

    let results = sightings.filter((s) => s.plate === normalized);
    if (cameraId) results = results.filter((s) => s.camera_id === cameraId);

    const response: SearchResponse = {
      query: { plate, normalized, fuzzy },
      count: results.length,
      results,
      // Canonical 4.6: fuzzy matching generates candidates only. They are a
      // separate array and are never merged into results.
      ...(fuzzy ? { candidates: searchFuzzy.candidates } : {}),
    };
    return HttpResponse.json(response);
  }),

  // The path still comes from endpoints.journey, never hand-written.
  // encodeURIComponent leaves "*" untouched, so passing it yields the MSW
  // wildcard while keeping the path shape owned by endpoints.ts. The plate is
  // then read back off the request URL.
  http.get(at(endpoints.journey("*")), async ({ request }) => {
    await delay(250);
    if (HOSTILE) return HttpResponse.json(hostileJourney as object);
    const segments = new URL(request.url).pathname.split("/");
    const plate = decodeURIComponent(segments[segments.length - 1] ?? "");

    const journeys: Record<string, unknown> = {
      GJ01AB1234: journeyFourCamera,
      GJ05CD6789: journeySingleSighting,
      GJ18EF4321: journeyMissingDisclaimer,
      GJ12ZZ0001: driftedJourney,
      GJ22KL0007: journeyRepeatedPair,
      GJ22KL0008: journeyAmbiguousPair,
      GJ22KL0009: journeyNoCount,
      GJ22KL0010: journeyNoTimeMatch,
      GJ22KL0011: journeyUndrawableConnector,
    };

    const journey = journeys[plate];
    if (journey === undefined) {
      const body: ApiErrorBody = {
        error: {
          code: "NOT_FOUND",
          message: `No journey for plate ${plate}`,
          request_id: "req_mock_journey_404",
        },
      };
      return HttpResponse.json(body, { status: 404 });
    }

    return HttpResponse.json(journey);
  }),

  // Deliberately stateful. From the SECOND request onward it returns one
  // additional alert with a distinct alert_id, so a reconnect-triggered
  // refetch is observable.
  //
  // Honest about what this proves: it proves the CLIENT invalidates and
  // refetches on reconnect. It proves nothing about server-side persistence
  // across an outage -- that needs the real backend and cannot be tested here.
  http.get(at(endpoints.alerts()), async () => {
    await delay(120);
    if (HOSTILE) return HttpResponse.json(hostileAlerts as object);
    alertsRequestCount += 1;

    // Applied on the way out, so an acknowledged row survives every later
    // refetch instead of reverting to the fixture's value.
    const withAcks = (list: Alert[]): Alert[] =>
      list.map((alert) =>
        acknowledgedIds.has(alert.alert_id) ? { ...alert, acknowledged: true } : alert,
      );

    // Demo mode is a rehearsal, so it must be identical every run. The
    // post-reconnect injection below carries a wall-clock created_at and only
    // appears from request 2 onward, which makes the list differ between a
    // rehearsal and the stage. Deterministic set only.
    if (DEMO || alertsRequestCount === 1) return HttpResponse.json(withAcks(alerts));
    const extra: Alert = {
      alert_id: "alr-post-reconnect",
      plate: "GJ01AB1234",
      camera_id: "cam01",
      camera_name: "Ashram Road junction",
      match_state: "exact",
      confidence: null,
      priority: "critical",
      created_at: new Date().toISOString(),
      acknowledged: false,
      snapshot_uri: null,
    };
    return HttpResponse.json(withAcks([extra, ...alerts]));
  }),

  http.post(at(endpoints.alertAcknowledge("*")), async ({ request }) => {
    await delay(150);
    const segments = new URL(request.url).pathname.split("/");
    // .../alerts/{id}/acknowledge -- the id is the second-to-last segment.
    const alertId = decodeURIComponent(segments[segments.length - 2] ?? "");

    if (alertId === ACK_FAILS_FOR) {
      return HttpResponse.json(
        errorBody(
          "INTERNAL_ERROR",
          "Could not record the acknowledgement.",
          "req_mock_ack_500",
        ),
        { status: 500 },
      );
    }

    acknowledgedIds.add(alertId);
    const found = alerts.find((a) => a.alert_id === alertId);
    if (!found) {
      // A live socket alert has no fixture row. The write still succeeds; a 204
      // is the honest answer to "recorded, nothing to return".
      return new HttpResponse(null, { status: 204 });
    }
    return HttpResponse.json({ ...found, acknowledged: true });
  }),

  http.get(at(endpoints.watchlist()), async () => {
    await delay(120);
    return HttpResponse.json(
      watchlistRows.filter((r) => !r.deleted).map(({ deleted: _deleted, ...row }) => row),
    );
  }),

  http.post(at(endpoints.watchlist()), async ({ request }) => {
    await delay(150);
    const draft = (await request.json()) as { plate?: string; reason?: string; priority?: string };
    const plate = draft.plate ?? "";

    if (plate === WATCHLIST_500_PLATE) {
      return HttpResponse.json(
        errorBody("INTERNAL_ERROR", "Could not add the plate to the watchlist.", "req_mock_wl_500"),
        { status: 500 },
      );
    }

    // A real duplicate, and a designated plate that always duplicates, so the
    // 409 copy is reachable on a clean list.
    const duplicate =
      plate === WATCHLIST_409_PLATE ||
      watchlistRows.some((r) => !r.deleted && r.plate === plate);
    if (duplicate) {
      return HttpResponse.json(
        errorBody("DUPLICATE_EVENT", `${plate} is already on the watchlist.`, "req_mock_wl_409"),
        { status: 409 },
      );
    }

    watchlistSeq += 1;
    const row: MockWatchlistRow = {
      id: `wl-${String(watchlistSeq).padStart(3, "0")}`,
      plate,
      reason: draft.reason ?? "",
      priority: (draft.priority as MockWatchlistRow["priority"]) ?? "medium",
      deleted: false,
    };
    watchlistRows.push(row);
    const { deleted: _deleted, ...created } = row;
    return HttpResponse.json(created, { status: 201 });
  }),

  http.delete(at(endpoints.watchlistItem("*")), async ({ request }) => {
    await delay(150);
    const segments = new URL(request.url).pathname.split("/");
    const id = decodeURIComponent(segments[segments.length - 1] ?? "");
    const row = watchlistRows.find((r) => r.id === id);
    if (!row) {
      return HttpResponse.json(
        errorBody("NOT_FOUND", `No watchlist entry ${id}`, "req_mock_wl_404"),
        { status: 404 },
      );
    }
    // Soft: the row stays, flagged, and simply stops being listed.
    row.deleted = true;
    return new HttpResponse(null, { status: 204 });
  }),

  // Every failure below is selectable by query string, so the error paths can
  // be exercised from the browser without editing code:
  //   /system/status?fail=1        -> 500, canonical envelope
  //   /health/live?fail=1          -> 503
  //   /health/ready?fail=1         -> 503
  //   /metrics/benchmark?shape=no_manifest
  // They are independent on purpose: "running" and "usable" are different
  // states, and a mock that could only fail both at once would hide that.
  http.get(at(endpoints.systemStatus()), async ({ request }) => {
    await delay(120);
    if (new URL(request.url).searchParams.get("fail") !== null) {
      return HttpResponse.json(
        errorBody("INTERNAL_ERROR", "Could not read system status.", "req_mock_status_500"),
        { status: 500 },
      );
    }
    return HttpResponse.json(systemStatus);
  }),

  http.get(at(endpoints.healthLive()), async ({ request }) => {
    await delay(60);
    if (new URL(request.url).searchParams.get("fail") !== null) {
      return HttpResponse.json({ status: "down" }, { status: 503 });
    }
    return HttpResponse.json({ status: "ok" });
  }),

  http.get(at(endpoints.healthReady()), async ({ request }) => {
    await delay(60);
    if (new URL(request.url).searchParams.get("fail") !== null) {
      return HttpResponse.json({ status: "not ready" }, { status: 503 });
    }
    return HttpResponse.json({ status: "ok" });
  }),

  http.get(at(endpoints.metricsBenchmark()), async ({ request }) => {
    await delay(150);
    // Same idiom as the journey handler: a Record<string, unknown> indexed and
    // then narrowed, because these are wire payloads typed `unknown` and
    // narrowing is what makes them serialisable without a cast.
    const shapes: Record<string, unknown> = {
      no_manifest: benchmarkNoManifest,
      canonical: benchmarkCanonical,
    };
    const shape = new URL(request.url).searchParams.get("shape") ?? "canonical";
    const body = shapes[shape] ?? shapes["canonical"];
    if (body === undefined) return new HttpResponse(null, { status: 204 });
    return HttpResponse.json(body);
  }),
];
