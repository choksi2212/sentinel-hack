# TRINETRA Frontend -- agent working rules

Project: a police-facing multi-camera vehicle intelligence dashboard for the
Gujarat Police "Sentinel" hackathon. Owner: Parth.

Stack: Vite + React + TypeScript (strict) + Tailwind v4 + MapLibre GL +
TanStack Query + zustand + MSW.

The normative specification is docs/TRINETRA_Canonical_Contracts.md.
Where any other document disagrees with it, it wins.

## Absolute rules -- violating any of these is a build failure

1. NEVER edit src/types/api.ts. It mirrors Canonical Contracts 6.5, which is
   normative. If a field looks wrong or missing, STOP and tell me. Do not
   "fix" it. Canonical field names, which differ from some of our manuals:
     plate            (NOT plate_normalized)
     feasible         (NOT is_feasible)
     priority         (NOT severity)
     acknowledged     (boolean, NOT acknowledged_at)
     status: "online" (NOT health_state / "healthy")
     camera_id        (NOT external_camera_id)
     lat, lon         are number | null
2. NEVER invent an API path. Every path lives in src/api/endpoints.ts.
3. NEVER render a raw confidence number to a user. Use
   <MatchStateChip state={...} observations={...} />. Confidences are
   relative evidence, not probabilities (Canonical 4.4).
4. NEVER draw a solid line between camera positions. Journey connectors are
   dashed in every code path, via src/map/journeyLineStyle.ts. A solid line
   asserts a route we did not observe. Infeasible segments get a warning
   treatment, never removal.
5. NEVER hide the journey disclaimer in a tooltip, accordion, modal or icon.
   It is a permanent visible footer rendered verbatim from response.disclaimer.
6. NEVER show a plate we did not read. plate === null renders the string
   "Unreadable". Never a guess, never a blank cell.
7. NEVER plot a sighting or camera whose lat or lon is null. List it instead.
   "Do not invent coordinates."
8. NEVER merge fuzzy candidates into the exact results list. Separate region,
   each row labelled "Candidate", distance shown.
9. NEVER reference cctv.corp8.cloud or any upstream stream URL. Camera
   previews go through the backend proxy at
   /api/v1/cameras/{camera_id}/preview.m3u8
10. NEVER use native <video> for HLS. Use hls.js, dynamically imported,
    mounted on click, destroyed on unmount, one player at a time.
11. NEVER use localStorage or sessionStorage.
12. NEVER put a secret in a VITE_ variable. They compile into the public
    bundle and the repository is public.
13. NEVER use `any` in src/types/ or src/api/.
14. NEVER add a screen, route, dependency or feature I did not ask for.

## Conventions

- Camera ids are lowercase cam01..cam30. CAM_001, CAM-001, Cam04 and cam4
  are all forbidden formats.
- A sighting is an interval: first_seen_at -> last_seen_at. There is no
  `timestamp` field anywhere.
- WebSocket messages are { type, data }; heartbeat is { type, ts } with ts at
  the top level. Tolerate `payload` but log when you see it.
- On WebSocket reconnect, always refetch alerts/status/cameras from REST.
  A socket-only client silently loses every event from the outage gap.
- Alerts are only ever "exact" or "probable". There is no low_confidence
  alert path; the database forbids it.
- Show error.request_id on every error panel. The same value is in the
  server log line.
- Every screen needs a loading, an empty and an error state. A blank panel
  is a bug -- on a projector nobody can tell it from a crash.
- Copy is sentence case. Errors say what failed and what to do. No
  apologies, no "Oops", no exclamation marks.
- Tailwind v4: no config file, no PostCSS file, no @tailwind directives.
  Design tokens live in src/styles/theme.css inside an @theme block.
- Directory is src/pages/, not src/screens/.

## Before you tell me you are done

Run: npm run guard && npm run typecheck && npm run build
Then describe what changed in one paragraph and flag anything you were
unsure about. Do not report success if any of the three failed.

## TypeScript config constraints (Vite 8 scaffold)

- erasableSyntaxOnly is ON: no enums, no namespaces, and no constructor
  parameter properties. `constructor(readonly code: string)` will fail the
  build; assign in the body instead.
- verbatimModuleSyntax is ON: type-only imports must use `import type`.
- noUnusedLocals and noUnusedParameters are ON: an unused import or
  parameter fails the build.
- strict and noUncheckedIndexedAccess are ON: array access returns
  `T | undefined`, so handle the empty case rather than asserting.
- Do NOT run `npx tsc --noEmit` at the repo root. tsconfig.json is
  solution-style with an empty file list, so it checks ZERO files and
  exits 0. It looks like a pass and is not one. Always use
  `npm run typecheck`, which is `tsc -b` and follows the project
  references.