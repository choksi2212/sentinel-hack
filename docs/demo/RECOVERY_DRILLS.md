# Recovery drills

Six failures that can happen mid-demo. For each: the trigger, what the operator
does, and the exact line to say.

**The principle.** Every one of these is survivable because the app already
says what is wrong. The operator's job is to *point at what the screen is
already telling the room*, not to apologise and not to explain around it. An
audience forgives a dependency failing. It does not forgive a dashboard that
kept claiming everything was fine.

Rehearse all six. Four of them can be triggered on purpose in under ten
seconds.

**Before rehearsing, read the two traps at the top of `DEMO_SCRIPT.md`.** Both
bite during these drills, not just during the script:

- **The watchlist demo plate is `GJ22KL0007` and no other.** `GJ01AB1234`,
  `GJ05CD5678` and `GJ18KL9012` are seeded and return 409; `GJ09XX0409` and
  `GJ09XX0500` are the designated failure fixtures. If you are practising
  drill 1 or drill 2 and want to add a watchlist row, use `GJ22KL0007`.
- **The alert feed shows watchlisted plates regardless of anything you do.**
  Three of the four plates `mock-ws` emits are already on the watchlist. Never
  narrate a drill in a way that implies the watchlist caused an alert.

The mock watchlist resets to its three seeded rows on a full page reload, so a
rehearsal does not poison the demo -- but **reload between rehearsal and the
real run**, or your second add of `GJ22KL0007` will 409.

---

## Drill 1 -- the backend dies

**Trigger.** Running in `VITE_APP_MODE=api` and the API stops answering:
process crash, port conflict, laptop sleeps and the connection drops.

**What the screen does.** The mode badge, top right, changes from `LIVE` or
`REPLAY` to **`MODE UNAVAILABLE`** (`src/layout/StatusBar.tsx:54-59`). It does
not fall back to a default badge, because a default there is a claim. Each
panel independently renders an error state with `error.request_id` on it, and a
Retry control.

**What the operator does.** Stop talking about the data. Point at the badge.
Then either switch to mock and carry on, or use the drill as the beat.

> "The backend has just gone. Notice what the badge did -- it says MODE
> UNAVAILABLE rather than holding the last thing it knew. Every panel is
> showing a request id you can grep in the server log. It would have been very
> easy to build this so it kept showing the last good screen, and that is
> exactly the failure we designed against."

**Recovery.** Restart the API and click Retry on any panel, or reload.

**Warning, and it is a real one.** If the browser tab is **not focused** when
the requests fail, panels can sit in a pending state showing skeletons
indefinitely rather than showing the error. TanStack pauses a retry whenever
the document is unfocused, and that check does not consult `networkMode`
(`src/main.tsx:29-33`). The health queries on `/status` use `retry: false` and
go straight to an error state; everything else uses `retry: 1` and can park
mid-retry. Whether refocusing the tab resolves it is **UNKNOWN** -- it has not
been reproduced in a real browser.

**Consequence for the demo: keep the tab focused.** Do not alt-tab to a
terminal and back while a panel is loading.

---

## Drill 2 -- the WebSocket disconnects

**Trigger.** `mock:ws` is killed, crashes, or the real socket drops. Also
happens if a NAT silently holds a dead connection open -- reproduce that
deliberately with `npm run mock:ws -- --go-quiet 10`, which stops sending
everything including heartbeats after ten seconds while keeping every socket
open.

**What the screen does.** The chip in the status bar walks through
`Live updates on` -> `Live updates reconnecting` -> `Live updates offline`
(`src/layout/StatusBar.tsx:95-100`). It never keeps saying "on" over a dead
socket -- a hardcoded "offline" string used to live there and was removed
precisely because a global indicator that disagrees with the screen below it
costs more trust than no indicator at all.

On reconnect the app **refetches alerts, status and cameras over REST** rather
than trusting the socket alone.

**What the operator does.** Let it reconnect. Say what it is doing.

> "The realtime feed just dropped and the app says so. Watch what happens when
> it comes back -- it re-fetches over HTTP rather than picking up where the
> socket left off, because a socket-only client silently loses every event from
> the gap. The dangerous version of this bug is the one where the indicator
> still says connected."

**Recovery.** Restart `npm run mock:ws`. The chip returns to `Live updates on`
within a few seconds and the list refills.

---

## Drill 3 -- no live feed, running on replay

**Trigger.** The Sentinel grid is unreachable, or you were always going to
present from recorded footage. This is the expected state for the whole demo,
not a failure.

**What the screen does.** The badge reads **`REPLAY`** followed by the source
mode spelled out -- "file replay", "frame replay", "synthetic"
(`src/layout/StatusBar.tsx:12-18`, `:77-86`). In demo mode the badge shows
REPLAY regardless of what `is_live` says in the payload, checked *before*
`is_live`, so a fixture edit can never put LIVE on a rehearsal.

**What the operator does.** Say it first, before anyone asks. The badge is
already visible; getting there second looks like a concession.

> "This is running on recorded footage and the badge says so -- REPLAY, with
> the source spelled out next to it. The same pipeline runs against the live
> grid; nothing below the media layer knows the difference. What I will not do
> is show you a screen that says LIVE when it is not."

**Note.** The badge never reads "ONLINE" in any state. That word is reserved
for camera reachability, and reusing it for a mode is the specific mistake the
status bar exists to prevent (`src/layout/StatusBar.tsx:66-68`). If someone
asks whether a camera is online, that is a different question with a different
answer on `/cameras`.

---

## Drill 4 -- map tiles fail

**Trigger.** Default is `VITE_BASEMAP=osm`, which fetches from
`tile.openstreetmap.org`. No network, blocked venue wifi, or a rate limit and
the tiles stop arriving.

**What the screen does.** Leaflet renders nothing for a failed tile and leaves
it **transparent**. Markers, journey connectors and every panel around the map
keep working -- only the basemap imagery is missing.

`errorTileUrl` is set on the OFFLINE config only, deliberately. On OSM it is
undefined, so a tile that fails to fetch stays transparent and you get an empty
area with correct markers on it. That is the right answer here: a network
failure is not a missing tile, it is a missing network, and drawing a neutral
placeholder thirty times would dress up an outage as a design choice. Say it
instead.

**What the operator does.** Name it immediately, then use the markers.

> "Basemap tiles come from the network and the network has gone. The markers
> are still exactly where the cameras are -- that data is ours, the map picture
> is not. Everything I am about to show you works off the positions, not the
> imagery."

**Recovery.** None mid-demo. Switching to `VITE_BASEMAP=offline` requires a
rebuild, because VITE_ variables are inlined at build time -- restarting
`preview` changes nothing.

**Do not switch to offline as a fix.** `public/tiles/` holds 2,064 *placeholder*
PNGs generated by `scripts/make-placeholder-tiles.mjs`, not map imagery. You
would swap a blank area for a grid of blank squares, which looks more broken,
not less. See `PROJECTOR_CHECK.md` item 6.

**Related, and worth knowing:** a sighting or camera whose `lat` or `lon` is
null is never plotted at an invented position. It is listed instead, and
Journey shows a count -- "N sightings not on the map". If someone asks why a
sighting is missing from the map, that is the answer.

---

## Drill 5 -- a plate reads wrong

**Trigger.** Someone in the audience spots a plate on screen that does not
match the vehicle in the snapshot, or two rows that are obviously the same
vehicle under different plates.

**What the screen already does.** This is the strongest drill, because the app
was built expecting it.

- A plate the system could not read renders the literal string
  **`Unreadable`** -- never blank, never a guess.
- Raw and normalised reads are distinct: `plate_raw` is what the OCR emitted,
  `plate` is normalised.
- Fuzzy matches are **never merged into results**. They render in a separate
  region, each row labelled `Candidate` with an edit `distance`. The shipped
  example is `GJ05CD5670` from a raw read of `GJ05CD567O` -- an O read as a
  zero, distance 1.
- Match state is a chip, never a bare confidence number.
- On `/journey`, an implausible segment is flagged rather than removed. The
  cam07 -> cam19 segment on `GJ01AB1234` reads
  `⚠ Not plausible — 212 km/h required` and carries the note: *"Requires 212
  km/h between these cameras. Check for an OCR error before treating this as a
  real movement."*

**What the operator does.** Agree, then show that the system already suspects
it. Go to `/status` and show the width buckets.

> "You are probably right, and look -- the system says so too. That segment
> needs 212 km/h, which is not a car, and the note tells the officer to check
> for an OCR error before treating it as movement. We flag it rather than
> deleting it, because deleting it hides the error. And here is why it happens:
> below thirty pixels of plate width our accuracy is zero. Not low. Zero. The
> sensor never resolved the plate and no model change recovers it."

**Do not** say the plate is correct, and do not silently navigate away.

---

## Drill 6 -- the laptop stutters

**Trigger.** The demo machine is running Postgres, Redis, a FastAPI backend and
a GPU worker alongside the browser. Frame rate drops, the map lags, typing
falls behind.

**What the app already does about it.** The caps are deliberate and they are
enforced in the store, not at the render layer:

- `ALERT_CAP = 100` in `src/lib/liveStore.ts:14`
- `ALERT_FEED_CAP = 100` in `src/lib/alertFeed.ts:10`, applied after dedup and
  sort so the newest hundred survive

An unbounded alert list becomes projector stutter, and an audience attributes
that stutter to the AI rather than to the browser.

`refetchOnWindowFocus` is **off** (`src/main.tsx:19`), so alt-tabbing does not
trigger a refetch storm.

**One thing to know before you tune anything.** The `/status` health queries are
the only ones in the app that poll in the background: `refetchInterval: 10_000`
with `refetchIntervalInBackground: true` (`src/pages/SystemStatus.tsx:138`,
`:150`). Every other query is idle unless the screen asks. So **a `/status` tab
left open in the background keeps polling every ten seconds forever.**

**What the operator does.**

1. Close any spare browser tab, especially a second `/status`.
2. Navigate away from `/status` when not presenting it.
3. If the map is the problem, stay on a list screen -- `/search`,
   `/alerts` and `/cameras` do not run Leaflet.
4. Do not restart the dev server mid-demo. Use the built bundle
   (`npm run build && npm run preview`), which is ~511 kB and does no
   transform work at runtime.

> Say nothing unless the audience notices. If they do:
> "That is this laptop running the whole stack, not the pipeline. The alert
> list is capped at a hundred for exactly this reason."

**Not a factor:** there is no video decoding in this build. hls.js and camera
preview are not implemented, so nothing here is decoding a stream.

---

## Quick reference

| Drill | Trigger it with | The screen already says |
|---|---|---|
| 1 Backend dies | stop the API in `api` mode | `MODE UNAVAILABLE` + request_id per panel |
| 2 WS disconnects | Ctrl+C `mock:ws`, or `--go-quiet 10` | `Live updates reconnecting` -> `offline` |
| 3 Replay not live | default state | `REPLAY` + source mode spelled out |
| 4 Tiles fail | disconnect network on `osm` | nothing -- transparent tiles, markers fine |
| 5 Plate wrong | audience spots it | `Unreadable`, `Candidate`, `⚠ Not plausible` |
| 6 Stutter | load the machine | nothing -- caps are silent by design |

Drills 1, 2, 4 and 6 have no visible recovery affordance beyond Retry. Drills 3
and 5 are not failures at all -- they are the system working, and they are the
two most worth rehearsing out loud.
