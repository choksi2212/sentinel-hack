# RECOVERY — rebuilding this frontend from a clone

You are reading this because the laptop is gone, broken, or someone else has
to run the demo. Follow it top to bottom. Ten minutes.

Backup remote: `https://github.com/iamwinter1116-void/trinetra-frontend-backup`
(PRIVATE). It holds every commit. It does NOT hold four things — (a), (b) and
the two gaps at the end.

---

## a) `.env.local` — NOT in git. Create it by hand, verbatim.

At the repo root, a file named exactly `.env.local`:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_APP_MODE=mock
```

`.gitignore` line 13 (`*.local`) keeps it out of the repo. That is correct —
it is machine config — but it means a clone has NO env file at all, and Vite
does not warn.

Change `VITE_APP_MODE` to `api` to talk to a real backend, and point
`VITE_API_BASE_URL` / `VITE_WS_BASE_URL` at it. See INTEGRATION.md.

## b) `public/tiles/` — NOT in git. Optional.

2,064 placeholder tile images (2.3 MB), gitignored deliberately
(`.gitignore` line 33) because they are map DATA, not source, and because
committing fake tiles would make the offline basemap look finished when no
real tiles have ever been sourced.

**Its absence breaks nothing.** The default basemap is online OpenStreetMap
and works without it. Only `VITE_BASEMAP=offline` needs tiles, and without
them that mode renders an empty grid — degraded, not broken. If you need
placeholders back: `node scripts/make-placeholder-tiles.mjs --clean`.

## c) Clone to a serving build

```powershell
git clone https://github.com/iamwinter1116-void/trinetra-frontend-backup.git trinetra-frontend
cd trinetra-frontend
# create .env.local now, from section (a) above, BEFORE building
npm ci
npm run build          # runs guard + typecheck + vite build; must exit 0
npm run preview        # serves dist/ on http://localhost:4173
```

For a live-reload dev server instead of a static build: `npm run dev`
(http://localhost:5173).

Realtime alerts need the mock socket in a second terminal:
`npm run mock:ws` (listens on ws://localhost:8000/ws/alerts).

Verified: a fresh clone + `npm ci` + `npm run build` exits 0 — guard 43 files,
typecheck 0, build 0.

## d) Symptom to check FIRST if the app reaches no backend

**If the app cannot reach the API after a clone, check `.env.local` before
you suspect the backend.**

With no `.env.local`, `VITE_API_BASE_URL` is undefined and the client falls
back to `window.location.origin` — the app requests ITSELF, gets `index.html`
back, and fails as a JSON parse error that looks like a backend fault.

The tell is in the build output: the main JS chunk is **511.99 kB without
`.env.local` and 511.89 kB with it**. A 0.10 kB difference, because the URL
string is baked into the bundle. If you see 511.99, the env file is missing.

There is also a console line on boot naming the resolved base URL — read it:

```
[trinetra] mode=mock, mock layer ON, api base http://localhost:8000
```

If `api base` shows a `localhost:5173` / `localhost:4173` origin instead of
port 8000, the env file is missing or was added after the build.

## e) VITE_ variables are BUILD-TIME. A rescue must REBUILD, not re-export.

All five are inlined into the bundle by Vite at build time:

| Variable | Values | Effect |
|---|---|---|
| `VITE_API_BASE_URL` | URL | where every REST call goes |
| `VITE_WS_BASE_URL` | URL | where the alerts socket connects |
| `VITE_APP_MODE` | `mock` / `api` / `demo` / `auto` | `api` = no mock layer |
| `VITE_BASEMAP` | `osm` (default) / `offline` | tile source |
| `VITE_MOCK_SHAPE` | `canonical` (default) / `hostile` | integration rehearsal |

**Changing any of them requires `npm run build` again.** Editing `.env.local`
and restarting `npm run preview` does NOTHING — the old values are already
compiled into `dist/`. Under `npm run dev` Vite restarts on env change, but
still re-reads at startup, not live.

## f) Two GitHub accounts are in this machine's keyring

`gh auth status` on the build laptop lists **two** logged-in accounts:
`iamwinter1116-void` (active, owns the backup repo) and **`Tejaspatel1524`**.

A push from this machine could land under the wrong identity if the active
account is switched. Check `gh api user --jq .login` before any `gh` command
that writes. Parth is handling the account itself; this note exists so a
rescuer does not push Parth's work to a stranger's namespace.

---

## Two gaps a clone will also have — know before you need them

1. **`docs/TRINETRA_Canonical_Contracts.md` is UNTRACKED.** The normative
   contract sits in the working tree on the build laptop and is NOT in the
   backup. A clone has no contract. To fix, on a machine that has it:
   `git add -f docs/TRINETRA_Canonical_Contracts.md` then commit and push.
   Until then, get it from Manas.
2. **No `.env.example`.** Canonical §8.1 says commit one; this repo has none,
   which is why section (a) exists at all.
