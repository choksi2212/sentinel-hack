# TRINETRA -- eight-minute demo script

Timed beats, exact routes, exact click paths, and one line to say per beat.

**Read the "Cannot be demonstrated" section before rehearsing.** Three of the
eleven beats do not work as specified against the current build. They are
marked BLOCKED inline and explained at the end. Nothing in this file describes
behaviour that has not been run.

---

# ⚠ READ THIS FIRST -- TWO TRAPS

## Trap 1: the alert feed looks like the watchlist causes it. It does not.

```
mock-ws alert plates : GJ01AB1234   GJ05CD5678   GJ18KL9012   GJ27MN3456
seeded watchlist     : GJ01AB1234   GJ05CD5678   GJ18KL9012
```

**Three of the four plates the live feed emits alerts for are already on the
watchlist**, so the alert list will show watchlisted plates arriving no matter
what you do or do not click -- **never say or imply that adding a plate caused
an alert.**

`scripts/mock-ws.mjs` is a separate process that picks from a fixed list on an
8-second timer. It has never read the watchlist and cannot. The coincidence is
convincing, which is exactly what makes it dangerous: the beat appears to work.
The real join -- a sighting whose plate matches a watchlist entry -- lives in
Mihir's backend and is not wired here. Narrate it, do not mime it.

## Trap 2: the plate you will reach for by reflex returns an error.

**`GJ01AB1234` is already on the watchlist.** Typing it into the watchlist form
returns **409 `DUPLICATE_EVENT` -- "GJ01AB1234 is already on the watchlist."**
It is the demo plate everywhere else in this script, so it is the one a
presenter types without thinking.

### The watchlist demo plate is `GJ22KL0007`. Use no other.

Verified by execution: not seeded, not in mock-ws's plate list, not one of the
designated failure plates, and a POST for it succeeds and lands in the
watchlist. Any other plate breaks the beat in one of four ways:

| Plate | What happens |
|---|---|
| `GJ01AB1234` `GJ05CD5678` `GJ18KL9012` | 409 -- already seeded on the watchlist |
| `GJ27MN3456` | Accepted, but it is in mock-ws's list, so an alert for it arrives on the timer and looks caused |
| `GJ09XX0409` | 409 by design -- the reachable-error fixture |
| `GJ09XX0500` | 500 by design -- the reachable-error fixture |

**Rehearsal note.** The mock watchlist is in-memory in the service worker and
**resets to the three seeded rows on a full page reload** -- verified. So a
rehearsal that adds `GJ22KL0007` does not poison the demo. But if you rehearse
and then present *without* reloading, your second add of `GJ22KL0007` will 409.
**Reload between rehearsal and the real run.**

---

## Before you start

```
VITE_APP_MODE=mock          # or demo; both serve fixtures, neither needs a backend
npm run dev                 # or npm run build && npm run preview
npm run mock:ws             # SEPARATE terminal. Required for beats 5:30 and 6:30.
```

`npm run mock:ws` binds port 8000, path `/ws/alerts`. Without it the status bar
reads **Live updates reconnecting** for roughly the first minute and then
**Live updates offline** for the rest -- `wsClient.ts` flips the label after six
failed attempts, and the backoff puts that at about 60 seconds. Both are honest,
and both undercut the 5:30 beat.

Confirm before you present: the badge top-right reads **REPLAY** followed by
the source mode. It must never read LIVE during a fixture demo, and it never
reads "ONLINE" in any state -- that word is reserved for camera reachability.

Demo plate throughout: **GJ01AB1234**. Its journey spans **five cameras** --
cam04, cam07, cam19, cam23, cam09 -- and it carries the infeasible segment used
at 4:45. It is the only plate with more than a two-camera journey: verified by
counting every journey fixture, and the next largest spans two.

---

## 0:00 -- 0:45 · The problem

**Route:** none. Do not have the app on screen yet, or the audience reads
instead of listening.

**Say:** "Thirty cameras across Ahmedabad, each recording independently. When a
vehicle of interest passes camera four and then camera nineteen, nobody knows
those were the same vehicle -- because nothing joins the two recordings. The
question an investigating officer actually asks is 'where has this plate been',
and today that question takes hours of manual review."

**Status:** demonstrable (spoken).

---

## 0:45 -- 1:15 · The insight

**Route:** none.

**Say:** "A plate read is not an identity, it is evidence with a confidence.
So we never merge two reads into a claim. We show what each camera observed,
when, and how strongly -- and we say plainly where we are inferring rather than
observing. The system is built to be wrong out loud rather than confident and
wrong."

**Status:** demonstrable (spoken).

---

## 1:15 -- 2:00 · Live Map, badge first

**Route:** `/map` (the app redirects `/` here).

**Click path:** open the app. Do not click anything yet. Point at the status
bar top-right *before* the map.

**On screen:** the status bar shows `REPLAY` on a coloured chip, then the
source mode spelled out ("file replay", "synthetic", etc.), then a live-updates
chip and a clock. The map fills the panel with camera markers.

**Say:** "First thing on this screen is what mode we are in. That says REPLAY,
so everything you are about to see is recorded, not live. If this said LIVE and
it were not, every claim on this screen would be false -- so the badge is the
one control we made impossible to get wrong."

**Note:** the badge shows `MODE --` while the status request is in flight and
`MODE UNAVAILABLE` if it fails. Neither is an error you need to explain away;
both are the app refusing to guess.

**Status:** demonstrable.

---

## 2:00 -- 2:45 · Camera Grid

**Route:** `/cameras` (left rail, "Cameras").

**Click path:** click **Cameras** in the left rail.

**On screen:** the camera list. The canonical fixture holds 12 cameras: 8
`online`, 2 `degraded`, 1 `offline`, 1 `unknown`.

**Say:** "Twelve cameras here, four different states. Degraded is not offline
and unknown is not offline -- an operator who cannot tell those apart will
either chase a camera that is fine or ignore one that is not."

**Status:** demonstrable.

---

## 2:45 -- 3:45 · Search

**BLOCKED as specified -- "partial plate" does not work. See below.** Run this
substitute, which demonstrates more and is true.

**Route:** `/search`.

**Click path:** click **Search**. Type `GJ 05 cd 5678` into the plate field --
lowercase, with spaces, deliberately messy. Tick the fuzzy option. Submit.

**On screen:** the query echoes back normalised to `GJ05CD5678`. Exact results
render in the main list. **Three** fuzzy candidates render in a **separate
region**, each row labelled `Candidate` with a `distance` value:

| Candidate | Raw read | Distance |
|---|---|---|
| `GJ05CD5670` | `GJ05CD567O` | 1 |
| `GJ05C05678` | `GJ05C05678` | 1 |
| `GJ06CD5678` | `GJ06CD5678` | 2 |

Only the first shows a raw read differing from the normalised plate -- an `O`
read as a zero. The other two are near-miss plates the index actually holds,
with no misread asserted. **Point at the first one**; it is the only row where
the OCR story is visible on screen.

**Say:** "I typed that badly on purpose -- wrong case, extra spaces -- and it
normalises. The important part is below: these are candidates, not matches.
They are in a separate list, each labelled, with the edit distance shown. We
never merge a maybe into a list of yes."

**Status:** demonstrable in this form. The literal "partial plate" beat is not.

---

## 3:45 -- 4:45 · Journey, dashed, and the disclaimer

**Route:** `/journey`.

**Click path:** click **Journey**. Enter `GJ01AB1234`. Submit.

**On screen — observed, not inferred.** The right-hand panel reads, verbatim:

```
This journey
4 sightings on the map
Server reports 5 sightings for this plate
1 sightings not on the map, location not surveyed
Segment data unusable — connection not shown (1)
1 segments could not be placed on this timeline
1 segments discarded as unusable

Sightings
GJ01AB1234 — Paldi circle 15:30:00
GJ01AB1234 — Maninagar station approach 15:34:30
Unreadable — cam19 15:39:10
GJ01AB1234 — cam23 15:44:45
GJ01AB1234 — Naroda industrial gate 15:51:05
```

and the permanent footer, verbatim from the response:

> Sightings are individual camera observations. Connecting lines are inferred,
> not observed routes.

**Three connectors are drawn, all dashed, with three different dash patterns
by feasibility state** (`6 6`, `10 5`, `3 6`) — shape, not just colour, so a
projector that flattens hue still separates them.

**Say:** "Five observations, three lines drawn between them. Every line is
dashed, and that is not decoration -- a solid line would assert a route we
never observed. The footer is not a tooltip and does not collapse; it is on
screen for as long as this journey is."

### Then use the panel. It is the best unscripted moment on this screen.

Four separate admissions, all true, all on screen without being asked for:

**Say:** "Look at what it is telling us about itself. It has four sightings on
the map and it says the server reported five -- it will not quietly show four
and call it five. One sighting has no surveyed location. One segment it could
not place. And there" — point at `Unreadable — cam19` — "is a plate it could
not read, and it says *Unreadable* rather than guessing. A dashboard that
hid any of those four would look cleaner and would be lying."

**Status:** OBSERVED in mock mode on `/journey?plate=GJ01AB1234`. Every string
above was read off the screen, not from source.

---

## 4:45 -- 5:30 · The infeasible segment

**Route:** `/journey`, same view. No navigation.

**Click path — you must CLICK the connector. Pointing shows nothing.** The
verdict lives in a Leaflet popup that only enters the page when opened. Click
the middle dashed line on the map, the one running north-east from Maninagar.

**Rehearse this click.** It is a thin line on a map, it is the single most
important thing you show, and there is no keyboard route to it.

**On screen — observed.** Clicking each of the three connectors opens a popup:

| Dash | Popup contents |
|---|---|
| `6 6` | Paldi circle → Maninagar station approach · 4.1 km straight line · `54.7 km/h required` |
| `10 5` | Maninagar station approach → cam19 · 16.4 km straight line · **`⚠ Not plausible — 212 km/h required`** · *"Requires 212 km/h between these cameras. Check for an OCR error before treating this as a real movement."* |
| `3 6` | cam19 → cam23 · 5.2 km straight line · `Plausibility not assessed` |

**Say:** "This one needs 212 kilometres per hour. That is not a car, it is
almost certainly a misread plate. We flag it and we keep it -- deleting it
would hide the error instead of surfacing it. And the note tells the officer
what to do about it: check for an OCR error before treating it as movement."

Then click the third connector.

**Say:** "And this one says 'not assessed', which is different from
'plausible'. The backend did not send us that field, so we do not have that
answer, and we do not invent one."

**Status:** OBSERVED. All three popups opened and their text read off the
screen. This is the strongest beat in the demo and the only one gated behind a
click on a thin target — practise it.

---

## 5:30 -- 6:30 · Watchlist to alert

**BLOCKED as a causal chain. See below.** What follows is what actually works.

**Route:** `/alerts`.

**Click path:** click **Alerts**. In the watchlist panel enter the plate
**`GJ22KL0007`** -- see Trap 2 at the top of this file, no other plate works --
a reason, and a priority (`low` / `medium` / `high` / `critical`). Submit. Then,
in the alert list, click **Acknowledge** on any unacknowledged alert.

**On screen:** the new watchlist row appears. The acknowledge button reads
`Acknowledging…`, then the row shows `Acknowledged`, dims, and **stays in the
list**.

**Say:** "That plate is on the watchlist now. And when an alert is acknowledged
it dims but does not disappear -- removing it would assert the matter is
resolved, and acknowledgement only means somebody saw it."

Then point at the alert list filling on its own and say the true thing:

> "These are arriving on the live feed. On the real backend the join is a
> sighting whose plate matches a watchlist entry -- that is what mints one of
> these. You are seeing both halves; the matching runs in Mihir's service."

**Say NOT:** anything implying the watchlist entry caused an alert. It did not,
three of the four feed plates are watchlisted already, and an audience that
works this out has been given a reason to doubt everything else on screen.

**If you typo and get an error panel, that is real and correct behaviour.**
A 409 renders "…is already on the watchlist"; the designated fixture plates
`GJ09XX0409` and `GJ09XX0500` produce a 409 and a 500 on purpose. Read the
message aloud, say "that is the app refusing a duplicate rather than silently
creating one", correct the plate, and move on. A recoverable error handled
calmly reads better than a beat that never risks anything.

**Status:** partially demonstrable. Watchlist add and acknowledge both work.
The watchlist-triggers-alert chain does not exist.

---

## 6:30 -- 7:15 · Resilience

**Route:** stay on `/alerts`.

**Click path:** in the `mock:ws` terminal, press Ctrl+C. Watch the status bar.

**On screen:** the live-updates chip moves from `Live updates on` to
`Live updates reconnecting`, then `Live updates offline`. Restart with
`npm run mock:ws` and it returns to `Live updates on`, and the app refetches
alerts over REST rather than trusting the socket alone.

**Say:** "I have just killed the realtime feed. The app says so -- it does not
keep showing a live indicator over a dead socket. And when it comes back it
re-fetches over HTTP, because a socket-only client silently loses every event
from the gap."

**BLOCKED:** do **not** demonstrate this by pulling the network cable. See
below.

**Status:** demonstrable via the socket. Not demonstrable as a full network
unplug.

---

## 7:15 -- 7:45 · Honest limits

**Route:** `/status` (left rail, "System status").

**Click path:** click **System status**. Scroll to the benchmark panel.

**On screen:** six plate-width buckets, largest first, each with a bar and a
number:

```
>100    0.98
80-100  0.93
60-80   0.83
40-60   0.64
30-40   0.38
<30     0.00   No software fix at this width -- recommend camera placement.
```

Below them, smaller: `Mean across all widths: 0.71 -- an average over the six
buckets above, not a system accuracy figure.` No percent sign appears anywhere
on this screen.

**Say:** "This is where the system fails. Above a hundred pixels of plate width
we are at 0.98. Below thirty we are at zero -- not low, zero, and no model
change fixes it because the sensor never resolved the plate. That is a camera
placement problem and we are saying so rather than quoting one average and
hoping nobody asks."

**Status:** demonstrable.

---

## 7:45 -- 8:00 · Close

**Route:** leave `/status` on screen.

**Say:** "Thirty cameras, one queryable record of which vehicle was seen where.
Everything you have seen runs with no backend at all -- it is fixtures behind a
service worker, and the badge said REPLAY the whole time. What it does not do
is guess."

**Status:** demonstrable.

---

# Beats that CANNOT be demonstrated, and why

## 1. 2:45 -- "Search, partial plate" does not work

`src/mocks/handlers.ts:151` filters with `s.plate === normalized`. That is an
**exact** match after normalisation, not a prefix or substring match. Typing a
partial plate returns **zero results and an empty state**.

Normalisation is real -- the handler does
`plate.toUpperCase().replace(/[^A-Z0-9]/g, "")` rather than echoing the input
back -- so messy input genuinely works. Fuzzy candidates are also real.
"Partial" is the only part that is not.

Fixing it would mean changing the mock handler to do prefix matching. That is a
mock change, not an app change, and **UNKNOWN** whether the real backend
supports partial search -- the contract defines the `plate` query parameter but
this document's author has not verified its matching semantics against Mihir's
implementation. Do not promise partial search on stage until that is confirmed.

## 2. 5:30 -- adding a watchlist entry does not produce an alert

`scripts/mock-ws.mjs` generates alerts from a fixed internal `PLATES` list on a
timer. It is a standalone process that never reads the watchlist and has no
connection to it. Adding `GJ01AB1234` to the watchlist and waiting will produce
alerts, but they are the scheduled ones and would have arrived anyway.

The individual halves are real: the watchlist add is a genuine optimistic
mutation with rollback, and acknowledge is a genuine write with sticky local
state. The **causal chain between them is not implemented anywhere in this
lane**, and whether the real backend implements it is **UNKNOWN**.

If the beat must show cause and effect, the honest option is to say "on the
real backend, a watchlist hit is what generates one of these" and not mime it.

## 3. 6:30 -- the network-unplug version of the resilience beat

Two separate problems.

**The basemap.** The default is `VITE_BASEMAP=osm`, which fetches tiles from
`tile.openstreetmap.org`. Pull the network and the map goes blank. Setting
`VITE_BASEMAP=offline` switches to `/tiles/{z}/{x}/{y}.png`, and
`public/tiles/` does contain 2,064 files across z10-z16 (49/56/72/99/180/432/
1176) -- but they are **placeholder PNGs generated by
`scripts/make-placeholder-tiles.mjs`, not map imagery**. Offline mode currently
renders a correctly-tiled grid with no streets on it. It proves the plumbing
works; it does not look like Ahmedabad, and on a projector it will read as
broken.

Real tiles have never been sourced. `src/map/basemap.ts:37` records the cost:
1,086 tiles for z0-16 of the demo bbox, 2,982 at z17, 47,094 at z19.

**Missing-tile handling exists now, and does not rescue this.** `errorTileUrl`
was declared and never wired; it is now set on OFFLINE and passed to
`L.tileLayer`, so a missing offline tile draws a flat neutral tile instead of
going transparent. That makes a gap look deliberate rather than half-rendered.
It does not put streets on the map, so offline mode is still not presentable.

**Do this instead:** kill `mock:ws` with Ctrl+C. It demonstrates the same
point -- the app tells the truth when a dependency dies -- using a mechanism
that actually works, and it recovers cleanly on restart.

---

# What each beat needs running

| Beat | Needs |
|---|---|
| 0:00, 0:45 | nothing |
| 1:15 Live map | dev server; network if `VITE_BASEMAP=osm` |
| 2:00 Cameras | dev server |
| 2:45 Search | dev server |
| 3:45, 4:45 Journey | dev server |
| 5:30 Alerts | dev server; `mock:ws` for live arrivals |
| 6:30 Resilience | `mock:ws`, to kill and restart |
| 7:15 System status | dev server |

No beat requires a backend, a database, or the AI worker. `VITE_APP_MODE=mock`
serves everything through MSW.
