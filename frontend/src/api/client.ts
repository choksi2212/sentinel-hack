// Every request goes through here. One place to change when the base URL,
// the headers or the error envelope moves.
import type { ApiErrorBody, ApiErrorCode } from "../types/ui";

// An unset VITE_API_BASE_URL means "same origin as the page" -- the shape a
// reverse-proxied deployment takes. Falling back keeps `new URL` from
// throwing inside every request, which would look like a broken frontend.
// It is announced once at module load, because the silent version sends
// every request to the frontend's own origin, gets index.html back, and
// surfaces as a parse failure that looks like a backend fault.
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL;

if (!configuredBaseUrl) {
  console.error(
    `[trinetra] VITE_API_BASE_URL is not set. Using the page origin ` +
      `${window.location.origin} instead. In api mode this returns index.html ` +
      `for every request.`,
  );
}

// Exported so the System Status screen can show what the app is talking to.
// That screen answers "is this actually running", and the answer includes
// which backend.
export const apiBaseUrl: string = configuredBaseUrl || window.location.origin;

export type QueryValue = string | number | boolean | undefined;
export type QueryParams = Record<string, QueryValue>;

// Keyed on ApiErrorCode, so adding a code to types/ui.ts fails this file at
// compile time rather than silently degrading it to "UNKNOWN" at runtime.
const API_ERROR_CODES: Record<ApiErrorCode, true> = {
  VALIDATION_FAILED: true,
  UNKNOWN_CAMERA: true,
  UNSUPPORTED_SCHEMA_VERSION: true,
  DUPLICATE_EVENT: true,
  NOT_FOUND: true,
  DEPENDENCY_UNAVAILABLE: true,
  INTERNAL_ERROR: true,
};

export class TrinetraApiError extends Error {
  // `message` is deliberately not declared here. Under useDefineForClassFields
  // a bare field declaration would define it as undefined after super(),
  // wiping the message Error already set.
  // "NON_JSON" is a 2xx whose body was not JSON at all, which is a different
  // fault from "the server refused this request" and gets its own copy.
  code: ApiErrorCode | "UNKNOWN" | "NON_JSON";
  requestId: string | null;
  field: string | null;
  status: number;

  constructor(
    code: ApiErrorCode | "UNKNOWN" | "NON_JSON",
    message: string,
    status: number,
    requestId: string | null,
    field: string | null,
  ) {
    super(message);
    this.name = "TrinetraApiError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.field = field;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (!isRecord(value)) return false;
  const error = value["error"];
  if (!isRecord(error)) return false;
  return (
    typeof error["message"] === "string" &&
    typeof error["request_id"] === "string"
  );
}

function toErrorCode(value: unknown): ApiErrorCode | "UNKNOWN" {
  if (typeof value === "string" && Object.hasOwn(API_ERROR_CODES, value)) {
    return value as ApiErrorCode;
  }
  return "UNKNOWN";
}

function buildUrl(path: string, params?: QueryParams): string {
  const url = new URL(path, apiBaseUrl);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined) continue;
      const encoded = String(value);
      // An empty filter is an absent filter, not an empty-string filter.
      if (encoded === "") continue;
      url.searchParams.set(key, encoded);
    }
  }
  return url.toString();
}

// Never throws. A malformed error body still yields a usable error object,
// because an exception raised while parsing an error is indistinguishable
// from the frontend itself being broken.
async function toApiError(response: Response): Promise<TrinetraApiError> {
  const fallback = (): TrinetraApiError =>
    new TrinetraApiError(
      "UNKNOWN",
      `Request failed (HTTP ${response.status})`,
      response.status,
      null,
      null,
    );

  try {
    const body: unknown = await response.json();
    if (!isApiErrorBody(body)) return fallback();
    const { code, message, request_id, field } = body.error;
    return new TrinetraApiError(
      toErrorCode(code),
      message,
      response.status,
      request_id,
      field ?? null,
    );
  } catch {
    return fallback();
  }
}

// The success half of every verb, shared so a write can never drift from a
// read in how it reads a body. Resolves to null when the response carries no
// body, so a 204 is a legitimate "nothing here" rather than a thrown parse
// error -- which is the normal shape of a successful DELETE.
async function readBody<T>(response: Response): Promise<T | null> {
  if (!response.ok) {
    throw await toApiError(response);
  }

  if (response.status === 204) return null;

  const text = await response.text();
  if (text.trim() === "") return null;

  try {
    return JSON.parse(text) as T;
  } catch {
    // A 200 carrying HTML is what the same-origin fallback looks like when
    // VITE_API_BASE_URL is unset. Naming that beats a raw SyntaxError, which
    // errors.ts would render as "Cannot reach the server" and send us to
    // debug a backend that is fine.
    throw new TrinetraApiError(
      "NON_JSON",
      `The server returned a non-JSON body (HTTP ${response.status})`,
      response.status,
      null,
      null,
    );
  }
}

export async function apiGet<T>(path: string, params?: QueryParams): Promise<T | null> {
  const response = await fetch(buildUrl(path, params), {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  return readBody<T>(response);
}

// NO RETRY, here or in apiDelete, and none may be added later. A retried POST
// is a duplicate write: the first attempt may well have reached the backend and
// committed before the response was lost, so a retry acknowledges twice or adds
// the same plate twice. apiGet's retry policy lives on the QueryClient and is
// safe only because GET is idempotent; it must not be inherited by these.
//
// A write that fails silently is worse than a read that fails silently. A read
// that fails shows an empty panel; a write that fails leaves the operator
// believing an alert is acknowledged when the backend never heard about it.
// So every failure throws a TrinetraApiError carrying requestId -- the same
// value as the backend log line, and the fastest way to find out what happened
// while a demo is still running.
export async function apiPost<TBody, TResult>(
  path: string,
  body: TBody,
): Promise<TResult | null> {
  const response = await fetch(buildUrl(path), {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return readBody<TResult>(response);
}

export async function apiDelete<TResult>(path: string): Promise<TResult | null> {
  const response = await fetch(buildUrl(path), {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });

  return readBody<TResult>(response);
}
