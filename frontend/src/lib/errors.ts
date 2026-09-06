// Maps anything thrown by the api layer to what a panel renders.
// requestId reaches the caller on every branch: the same value is in the
// backend log line, and it is the fastest debugging path during a demo.
import { TrinetraApiError } from "../api/client";

export interface DisplayError {
  // "empty" is a legitimate no-results state, not a failure. Rendering it as
  // an error panel would tell an officer something broke when nothing did.
  kind: "empty" | "error";
  title: string;
  detail: string;
  retry: boolean;
  requestId: string | null;
}

// What the failed call was trying to do. Almost every case reads the same
// either way; NOT_FOUND is the one that cannot.
export type Operation = "read" | "write";

function fromApiError(error: TrinetraApiError, operation: Operation): DisplayError {
  const requestId = error.requestId;

  switch (error.code) {
    case "VALIDATION_FAILED":
      return {
        kind: "error",
        // "filter" was read-only vocabulary. This code now also reaches a
        // rejected watchlist add, which has fields but no filters.
        title: "That request isn't valid",
        detail: error.field
          ? `Check the ${error.field} value and try again.`
          : "Check the values and try again.",
        retry: false,
        requestId,
      };

    case "UNKNOWN_CAMERA":
      return {
        kind: "error",
        title: "Camera not in the registry",
        detail: "Check the camera id and try again.",
        retry: false,
        requestId,
      };

    case "UNSUPPORTED_SCHEMA_VERSION":
      return {
        kind: "error",
        title: "Unsupported schema version",
        detail:
          "The API rejected this request's schema version. Report this with the request id.",
        retry: false,
        requestId,
      };

    case "DUPLICATE_EVENT":
      return {
        kind: "error",
        // "event" was ingestion vocabulary. The code also lands on a watchlist
        // add whose plate is already watched, which is not an event.
        title: "Already recorded",
        detail: "The API has already recorded this. No action is needed.",
        retry: false,
        requestId,
      };

    // The audit note left here last session said the fix needed the verb, which
    // this layer did not receive. It receives it now, so the fix is the verb --
    // NOT a rewritten string. A read that matches nothing is a legitimate empty
    // state and keeps its copy exactly; a write that cannot find its target is a
    // failure and must not render as a calm "No results".
    case "NOT_FOUND":
      if (operation === "write") {
        return {
          kind: "error",
          title: "Not found",
          detail:
            "The server has no record of this. Refresh to see the current state.",
          // Retrying a delete against an id the server does not have will fail
          // identically. Refreshing is the action that helps.
          retry: false,
          requestId,
        };
      }
      return {
        kind: "empty",
        title: "No results",
        detail: "Nothing matched this request. Widen the time range, or check the plate.",
        retry: false,
        requestId,
      };

    case "DEPENDENCY_UNAVAILABLE":
      return {
        kind: "error",
        title: "The database is unreachable",
        // Was "Search and history are unavailable. Live updates continue."
        // Two faults: it framed the outage as read-only when the same code can
        // reject an acknowledge or a watchlist write, and its second sentence
        // asserted the socket was healthy. This layer sees one failed HTTP
        // response and cannot observe the socket at all -- that claim belongs
        // to the status line, which measures it.
        detail: "Stored data cannot be read or written until it recovers.",
        retry: true,
        requestId,
      };

    case "INTERNAL_ERROR":
      return {
        kind: "error",
        title: "The server failed on this request",
        // Was "Retry, or narrow the time range." Narrowing a range is advice
        // for a slow query, not a server fault, so it was misplaced even for
        // reads; on a failed acknowledge there is no range and the sentence is
        // an instruction the operator cannot follow.
        //
        // Deliberately does NOT say whether anything was written. On a 500 from
        // a POST the write may have partly committed, and this layer sees only
        // the status code -- "nothing was changed" would be a fabricated
        // reassurance, which is the failure mode this whole app is built
        // against.
        detail: "The server reported an internal fault. Retry, and report this with the request id if it continues.",
        retry: true,
        requestId,
      };

    case "UNKNOWN":
      return {
        kind: "error",
        title: `Request failed (HTTP ${error.status})`,
        detail: "Retry. If it continues, report this with the request id.",
        retry: true,
        requestId,
      };

    // A 2xx whose body was not JSON. Rendering this as
    // "Request failed (HTTP 200)" pairs a failure with a success code and
    // reads as nonsense to anyone debugging in a hurry. Retrying cannot fix
    // a misrouted base URL, so this one does not offer it.
    case "NON_JSON":
      return {
        kind: "error",
        title: "The server returned a response that was not JSON",
        detail:
          "The request reached something other than the API, which usually means " +
          "VITE_API_BASE_URL is unset and requests are going to this page's own origin.",
        retry: false,
        requestId,
      };

    default: {
      // Unreachable: client.ts maps every unrecognised code to "UNKNOWN".
      // Typing it as never makes a new ApiErrorCode a compile error here
      // rather than a case that quietly falls through at a demo.
      const unhandled: never = error.code;
      return {
        kind: "error",
        title: `Request failed (HTTP ${error.status})`,
        detail: `The API returned an unrecognised error code (${String(unhandled)}). Report this with the request id.`,
        retry: true,
        requestId,
      };
    }
  }
}

// Defaults to "read", so every existing call site keeps its exact behaviour and
// the search empty state is untouched. Only the mutation panels pass "write".
export function toDisplayError(
  error: unknown,
  opts?: { operation?: Operation },
): DisplayError {
  const operation = opts?.operation ?? "read";
  if (error instanceof TrinetraApiError) {
    return fromApiError(error, operation);
  }

  // fetch rejects on DNS failure, connection refused, TLS problems and CORS.
  // There is no request id, because the request never reached the backend.
  return {
    kind: "error",
    title: "Cannot reach the server",
    detail: "Check that the API is running.",
    retry: true,
    requestId: null,
  };
}
