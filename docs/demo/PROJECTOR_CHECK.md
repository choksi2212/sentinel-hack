# Projector check

Seven items. For each: how to test it, and what is known about this build
right now.

**Status vocabulary.** PASS means it has been checked. FAIL means it has been
checked and does not work. UNKNOWN means nobody has run the test -- it is not
a soft pass, and there are five of them here. Anything below marked UNKNOWN is
a thing to do before submission, not a thing to assume.

## Run this first

**Press Ctrl+Alt+P.** That toggles `html.projector`, which raises the root font
from 15px to 19px (`src/styles/theme.css:59`). Every size in the app is
rem-based, so one class scales the whole interface together.

The chord uses two modifiers so it cannot be produced by ordinary typing, and
the handler additionally ignores the event when focus is in a text input, a
textarea or a select -- that guard exists because on layouts where AltGr is
sent as Ctrl+Alt, AltGr+P is a printable character. **Consequence: with the
cursor in the plate filter or the camera dropdown the shortcut does nothing.
Click the page background first.**

It is not persisted -- CONVENTIONS forbids browser storage -- so it resets on
reload. Re-press after any refresh.

**Verify it works on the demo machine before demo day.** The toggle logic is
measured and correct: dispatching a Ctrl+Alt+P keydown flips the class and the
computed root font goes 15px -> 19px. But whether a *physical* Ctrl+Alt+P
reaches the page on this hardware and keyboard layout is **UNKNOWN** -- the
automation used to test this delivers no key events to the page at all, for any
key, so it produced no evidence either way. Press it yourself once. If nothing
happens, the console fallback still works:

```js
document.documentElement.classList.toggle('projector')
```

Every measurement below should be taken twice, with and without the class.

---

# The ten-minute physical pass

Five checks nobody can run from a desk. Do them on the demo machine, in order.
Each says what to look at, what counts as failure, and what to do about it --
and for four of the five the answer under time pressure is a **script change**,
not a code change.

### 1. Does Ctrl+Alt+P actually work? (2 min)

Open the app, **click the page background first** -- not a text field -- and
press **Ctrl+Alt+P**. Everything should visibly grow. Press again to revert.

Open the console (F12) and look for `[trinetra] projector mode ON, root 19px`.

- **Log line appears, nothing grows** -- the CSS is not matching. Unlikely.
- **No log line** -- the key is not reaching the app on this keyboard layout.

**If it fails:** paste `document.documentElement.classList.toggle('projector')`
into the console. Five seconds, works regardless of layout. That makes it a
*script change* -- add "open console, paste this" to your pre-flight -- and no
code needs touching. **This is the one check that is genuinely unknown**: the
toggle logic is measured, a physical keypress is not.

### 2. Both resolutions (3 min)

DevTools device toolbar (Ctrl+Shift+M), Responsive, type **1024 x 768**. Walk
all six routes. Then **1920 x 1080**.

**Look at `/journey` first** -- it has the most fixed chrome (184px rail +
260px aside = 444px, leaving 580px). Check: is the timeline clipped? Does the
aside overlap the map? Does the status bar's right cluster (badge, socket chip,
clock) wrap to a second line?

**If it fails:** *script change.* Present at 1920x1080 and do not switch. If
the venue forces 1024x768 and Journey breaks, run that one beat on the laptop
screen. A layout fix to a screen this close to submission is the wrong trade.

### 3. Horizontal scroll at 1024 (2 min)

At 1024x768, on each of the six routes, in the console:

```js
document.documentElement.scrollWidth > document.documentElement.clientWidth
```

**`false` on all six is the pass.** Then repeat with projector mode ON -- that
is the harder case and the one most likely to fail. Also just look: is there a
scrollbar along the bottom of the window?

**If it fails:** the culprit is one of the three fixed-width `shrink-0`
elements. A code fix exists (`overflow-x-auto` on the offending panel) but it
is a real change to a working screen. *Script change:* present wider, or leave
projector mode off for that screen.

### 4. Map hover dependence (2 min)

On `/map`, **click** each camera marker -- do not hover. On `/journey` with
`GJ01AB1234`, click each connector and each sighting node.

The question: **is there anything you can only learn by hovering?** A Leaflet
tooltip, a `title` attribute, a `:hover` reveal. Specifically confirm the
212 km/h `⚠ Not plausible` verdict is rendered as text beside the segment and
not only on hover.

**If it fails:** *script change.* Click it and read the value aloud. A
projector audience never sees a hover state, but they see you click and they
hear you say the number.

### 5. 125% zoom (1 min)

Ctrl+Plus twice from 100% at 1024x768. **`/journey` breaks first**: 1024px at
125% is 819 CSS px, minus 444px of fixed chrome leaves **375px** for the
timeline. Look for the timeline column collapsing, text wrapping mid-word, the
aside overlapping the map. Re-run the scrollWidth check from step 3. Then try
150%.

**If it fails:** **use projector mode instead of browser zoom.** They are
different mechanisms and this matters: projector mode raises the root font but
leaves the fixed pixel widths alone, so text grows without the layout viewport
shrinking. Browser zoom shrinks the viewport, which is what breaks Journey.
When someone at the back asks you to make it bigger, **Ctrl+Alt+P is the right
answer and Ctrl+Plus is the wrong one.**

---

## 1. 1024x768 and 1920x1080 both usable

**How to test.** DevTools device toolbar, set 1024x768 exactly, then 1920x1080.
Walk all six routes: `/map`, `/search`, `/journey`, `/alerts`, `/cameras`,
`/status`. At each size confirm nothing is clipped and no panel overlaps
another.

**What is known.** UNKNOWN -- never measured at either size.

What can be said from the source is that the layout uses **fixed, non-shrinking
widths** and the arithmetic is tight at 1024:

| Element | Width | Source |
|---|---|---|
| Left rail | 184px, `shrink-0` | `src/layout/LeftRail.tsx:93` |
| Live map aside | 220px, `shrink-0` | `src/pages/LiveMap.tsx:67` |
| Journey aside | 260px, `shrink-0` | `src/pages/Journey.tsx:196` |

At 1024px, Journey leaves `1024 - 184 - 260 = 580px` for the main panel, before
padding and borders. That is the narrowest case in the app and the one to test
first. At 1920 there is no plausible problem.

**Risk if untested:** the Journey timeline is the 3:45 and 4:45 beat, and 580px
is where it would break.

---

## 2. Contrast readable in a bright room

**How to test.** Project it in the actual room with the actual lights. No
substitute exists -- a laptop screen at desk distance tells you nothing about a
projector's black level. Failing that, sample the token pairs with a contrast
checker.

**What is known.** PARTIAL, by design rather than by measurement.

The palette is deliberately **light-on-white, not dark**.
`src/styles/theme.css:11` records the reason: projectors have poor black
levels, so a dark UI that looks sharp on a laptop washes out. The core text
tokens are `--color-ink: #0f1619` on `--color-paper`, with `--color-ink-2:
#45565d` and `--color-ink-3: #6b7c83` for secondary text.

`#6b7c83` is the one to check. It is the lightest text token in the palette and
is used for de-emphasised values. Its contrast against the paper background has
**not been computed** -- UNKNOWN, and it is the most likely single failure on
this item.

**What is definitely right:** no state anywhere is conveyed by colour alone.
Camera status, alert priority and match state each carry colour AND a text
label AND a shape (`src/components/CameraStatusChip.tsx:4`,
`src/pages/Alerts.tsx:25`, `src/pages/SystemStatus.tsx:21`). Projector gamma
flattens hue; it cannot flatten a word.

---

## 3. Legible from 5m

**How to test.** Project it, walk five metres back, and read the smallest text
on each screen aloud. Do it with `.projector` on and off.

**What is known.** PASS against the app's own floor, measured. UNKNOWN at five
actual metres, which needs a room.

`src/styles/theme.css:53` states the rule:

> never use text-xs (0.75rem = 11.25px). Nothing below 13px, and text-sm is
> 13.1px.

There used to be exactly one violation -- `src/pages/Search.tsx` rendered its
snapshot placeholder at `text-[0.7rem]`, **10.5px** at a 15px root, smaller
than the `text-xs` the rule bans by name. It is now `text-sm`, and the label
reads "No image" rather than "No snapshot" because the longer word does not fit
a 48px box at the larger size.

Measured in Chrome at 1464px, walking every leaf element with text content:

| | root | smallest text | elements below 13px |
|---|---|---|---|
| projector OFF | 15px | **13.125px** | **0** |
| projector ON | 19px | **16.625px** | **0** |

The placeholder does not overflow its 45px box at the new size
(`scrollWidth > clientWidth` is false).

**Still UNKNOWN:** whether 13.125px is readable from five metres on the actual
projector, in the actual room. Nothing about a computed font size answers that.
Projector mode nearly doubles the margin and is the mitigation if it is not.

---

## 4. No horizontal scroll

**How to test.** At 1024x768, on each of the six routes, run in the console:

```js
document.documentElement.scrollWidth > document.documentElement.clientWidth
```

`false` on all six is the pass. Repeat with `.projector` applied, which is the
harder case -- 19px root text in the same fixed-width containers.

**What is known.** PASS at 1464px, both with and without projector mode.
UNKNOWN at 1024px, which is the case that matters.

Measured in Chrome at 1464px on `/search` and `/status`:
`document.documentElement.scrollWidth > clientWidth` is **false** with the
projector class off *and* on.

Two things pull in opposite directions at narrower widths:

- **Against:** a search of `src/` for `overflow-x`, `overflow-x-auto` and
  `overflow-auto` returns **no matches**. Nothing puts wide content in its own
  scroll container, so anything too wide pushes the page rather than scrolling
  inside a panel.
- **For:** `src/layout/AppLayout.tsx:20` puts `min-w-0` on the `<main>` flex
  child, with a comment naming this exact problem -- "keeps a wide child from
  forcing the page to scroll sideways at 1024px". That is the standard fix for
  the flexbox min-content trap and it is already in place. An earlier draft of
  this document omitted it and was wrong to.

So the mitigation exists and is deliberate; what is untested is whether it
holds on Journey at 1024px with projector mode on, which is the tightest
combination in the app.

---

## 5. No hover-dependence

**How to test.** Navigate the entire eight-minute script using only the
keyboard and single clicks. Never let the pointer rest on anything. Anything
that only reveals information on hover is a failure, because a projector
audience never sees a hover state.

**What is known.** PASS for navigation. UNKNOWN for the map.

The left rail carries an icon **and** a visible text label for every route,
with no hamburger and no icon-only mode -- and
`src/layout/LeftRail.tsx:86-88` records that this was a deliberate choice for
exactly this reason. Hover only changes background tint, never content.

**Not verified:** whether any map marker, journey segment or chart bar exposes
information only in a Leaflet tooltip or a `title` attribute. The Journey
segment verdicts (`Plausibility not assessed`, `⚠ Not plausible — 212 km/h
required`) are rendered as text content rather than tooltips, which is the
important case and is fine. The rest is UNKNOWN.

---

## 6. Map tiles available offline

**How to test.** Set `VITE_BASEMAP=offline`, rebuild (VITE_ variables are
inlined at build time -- restarting `preview` changes nothing), disconnect the
network, load `/map` and `/journey`.

**What is known.** **FAIL.** This is the clearest failure on the list.

`public/tiles/` contains 2,064 files across z10-z16 (49 / 56 / 72 / 99 / 180 /
432 / 1176). The plumbing works: `src/map/basemap.ts:43` points offline mode at
`/tiles/{z}/{x}/{y}.png`, Vite copies `public/` verbatim into `dist/`, and the
zoom range matches `maxZoom: 16`.

But every one of those 2,064 files is a **placeholder generated by
`scripts/make-placeholder-tiles.mjs`**, not map imagery. Offline mode renders a
correctly-aligned grid with no streets on it. It proves the tile path resolves;
it does not look like Ahmedabad, and on a projector it will read as a broken
map rather than an offline one.

Real tiles have never been sourced. `src/map/basemap.ts:37` records the cost:
1,086 tiles for z0-16 of the demo bbox, 2,982 at z17, 47,094 at z19.

**Second, smaller failure -- now fixed.** `errorTileUrl` used to be declared on
`BasemapConfig`, never set on either config, and never read anywhere in `src/`,
while the comment beside it described a fallback that did not exist. It is now
set on OFFLINE and passed through to `L.tileLayer` in
`src/map/useMapInstance.ts`. A missing offline tile renders a flat neutral tile
with a faint diagonal hatch, inlined as a data URI so the fallback cannot
itself 404.

It carries **no text** on purpose: a viewport is thirty to forty tiles across,
and thirty copies of the words "no tile" is worse noise than the blank it
replaces. OSM leaves the field undefined, because a missing tile there means
the network died mid-demo -- drill 4 in `RECOVERY_DRILLS.md` handles that by
saying so out loud rather than by drawing something.

This does not make item 6 pass. It makes the gap look deliberate rather than
broken.

**Decision needed before the demo:** either source real tiles for the bbox, or
run with `VITE_BASEMAP=osm` and accept that the map needs network. The demo
script's 6:30 beat is written for the second option.

---

## 7. 125% zoom still functional

**How to test.** Browser zoom to 125% (Ctrl+Plus twice from 100% in Chrome:
110%, 125%). Walk all six routes at 1024x768. Then repeat at 150%, which is
what someone at the back will ask for if they cannot read it.

**What is known.** UNKNOWN -- never measured.

The arithmetic is worth knowing before testing. Browser zoom shrinks the CSS
viewport: a 1024px window at 125% is **819 CSS px**. The fixed chrome on
Journey is 184 + 260 = 444px, leaving **375px** for the timeline. That is
phone-width, in a layout with no `overflow-x` containers anywhere.

Note that browser zoom and `.projector` are different mechanisms and compound.
`.projector` raises the root font but not the fixed pixel widths; browser zoom
scales both. Test them separately before testing them together.

---

## Summary

| # | Item | Status |
|---|---|---|
| 1 | 1024x768 and 1920x1080 | UNKNOWN -- measured only at 1464px |
| 2 | Contrast in a bright room | PARTIAL -- palette chosen for it; lightest token uncomputed |
| 3 | Legible from 5m | PASS against the 13px floor (13.125 / 16.625 measured, 0 below); UNKNOWN at five real metres |
| 4 | No horizontal scroll | PASS at 1464px both modes; UNKNOWN at 1024px |
| 5 | No hover-dependence | PASS for navigation; UNKNOWN for map layers |
| 6 | Tiles offline | **FAIL** -- 2,064 placeholders, no real imagery |
| 7 | 125% zoom | UNKNOWN |

One failure left, down from two. Item 3's violation is fixed and measured.
Item 6 is not a code problem: it is a decision about whether to source real
tiles for the bbox or present with `VITE_BASEMAP=osm` and accept that the map
needs network. The demo script is written for the second option.

Four items still need a room, a projector and a 1024x768 window. None of them
can be closed from a desk.
