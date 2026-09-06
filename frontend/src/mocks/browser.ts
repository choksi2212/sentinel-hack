import { setupWorker } from "msw/browser";
import { apiBaseUrl } from "../api/client";
import { handlers } from "./handlers";

export const worker = setupWorker(...handlers);

// "warn" warns on every unhandled request, which includes the app's own
// document and every asset Vite serves. That buries the one case worth
// seeing: an API call nobody wrote a handler for.
//
// It is also what surfaced the GET / warning. Chrome DevTools issues a
// cache-only request for the document, and public/mockServiceWorker.js:101
// only bypasses only-if-cached requests whose mode is NOT "same-origin" --
// the DevTools one is same-origin, so it falls through to handler matching.
// Scoping the warning to API traffic silences that. It does not stop the
// TypeError that follows it, which is not MSW's to cause or ours to fix:
// the same request throws identically with MSW never started.
export function onUnhandledRequest(
  request: Request,
  print: { warning: () => void; error: () => void },
): void {
  if (request.url.startsWith(apiBaseUrl)) {
    print.warning();
  }
}
