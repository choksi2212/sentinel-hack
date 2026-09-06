# DECISIONS.md

Judgement calls made under the ambiguity procedure in SESSIONS.md section 1.
Do not re-litigate these. Each entry records what was chosen, what was
rejected, and why.

Tags:
- **REVERSIBLE** — confident; if Parth disagrees it is a small edit.
- **NEEDS-PARTH** — not confident, or reversing it touches several files.
  Implemented anyway, listed at the top of WORKLOG.md.

---

## D-001 · Setup · MSW `GET /` passthrough warning and "Failed to fetch"
Chose:      Scope `onUnhandledRequest` to API traffic only; do NOT attempt to
            fix the TypeError. Already shipped in commit e4abe27.
Rejected:   Adding a `GET /` handler, or bypassing the request in our own
            code, to make the TypeError disappear.
Confidence: high
Tag:        REVERSIBLE
Reason:     Diagnosed before fixing, per instruction. The page loads
            correctly on first navigation and on hard reload (`pageLoaded:
            true`, `h1: "Get started"`). The request is Chrome DevTools'
            cache-only fetch for the document: `cache: "only-if-cached"`,
            `mode: "same-origin"`. `public/mockServiceWorker.js:101` bypasses
            only-if-cached requests only when their mode is NOT
            "same-origin", so this one falls through to handler matching.
            The TypeError is upstream noise, not ours: rebuilt with
            VITE_APP_MODE=api, where MSW never starts, the identical request
            throws the same TypeError with a completely empty console. A
            fetch with cache "only-if-cached" that misses the cache fails by
            specification. Inventing a fix in the mock layer would have been
            fixing someone else's browser behaviour. The warning was still
            worth scoping: `"warn"` warned on every document and asset,
            burying the one signal worth having, which is an API call with no
            handler. Verified after the change: no `GET /` warning, while an
            unmocked `/api/v1/watchlist` call still warns.

---

## D-002 · Session C · QueryClientProvider added in C rather than E
Chose:      Mount `QueryClientProvider` in `main.tsx` during Session C, with
            `staleTime: 5000` and `refetchOnWindowFocus: false`.
Rejected:   Deferring it to Session E and having the status bar fetch with a
            bare `useEffect` in the meantime.
Confidence: high
Tag:        REVERSIBLE
Reason:     Session C requires the status bar to read `SystemStatus` from the
            mock, so C already needs a fetch. SESSIONS.md section 2 rejects
            data-router APIs precisely because "two systems fetching means
            two caches disagreeing" -- a throwaway `useEffect` fetch would
            create exactly that second system for one session and then be
            deleted. `refetchOnWindowFocus: false` because alt-tabbing during
            a demo must not silently change what is on the projector.

---

## D-003 · Session C · Root font-size 15px, sizes in rem
Chose:      `html { font-size: 15px }` with `html.projector { 19px }`, and
            express the status bar height as `3.7333rem` (= 56px exactly at
            the 15px root).
Rejected:   Keeping the 16px root and setting `body { font-size: 15px }`,
            with the status bar as a hard `56px`.
Confidence: high
Tag:        REVERSIBLE
Reason:     SESSIONS.md asks for body 15px, a 56px status bar, and one class
            that "scales everything for a projector". Those only hold
            together if sizes are rem-relative to a 15px root; a hard 56px
            bar would stay 56px while its own text grew under .projector and
            overflowed. Measured in the browser at 55.99px. Consequence
            recorded in theme.css: `text-xs` is now 11.25px and must never be
            used, since nothing may go below 13px. `text-sm` is 13.1px.

---

## D-004 · Session C · index.css and App.css left on disk, unimported
Chose:      Stop importing `src/index.css` and `src/App.css`; strip the
            Tailwind import out of index.css so exactly one remains; leave
            both files in the repository.
Rejected:   Deleting them, which is the tidier result.
Confidence: high
Tag:        REVERSIBLE
Reason:     SESSIONS.md section 1 lists "deleting a file you did not create
            in that same session" as a hard stop. Both are Vite scaffold
            files from before this run. They are now dead code -- their
            `#root { width: 1126px; text-align: center }` would have fought
            the shell layout, which is why they are unimported rather than
            merely overridden. Safe for Parth to delete, along with
            src/assets/hero.png, react.svg and vite.svg, which the old
            App.tsx was the only consumer of.

---

## D-005 · Session C · Status bar mode badge while loading or failing
Chose:      Render `MODE —` while the query is pending and `MODE UNAVAILABLE`
            on error or a null body. Never LIVE, never REPLAY.
Rejected:   Defaulting to REPLAY while loading, on the grounds that it is the
            safer of the two.
Confidence: high
Tag:        REVERSIBLE
Reason:     Ambiguity procedure rule 1: refuse to invent data. Either badge
            is a claim about what the backend is doing, and we do not know it
            yet. REPLAY would be the safer lie but it is still a lie, and it
            would mask a genuinely broken status endpoint by looking normal.
            Verified in the browser that the mock's `is_live: false` renders
            REPLAY with `synthetic` spelled out beside it, not LIVE.

---

## D-006 · Session D · Local patch to public/mockServiceWorker.js
Chose:      Add a bypass to MSW's generated service worker for requests whose
            `destination` is `worker`, `sharedworker` or `serviceworker`.
Rejected:   Working around it in our own code (a blob URL for the worker
            script), which I tried first and which does not work in dev.
Confidence: high on the diagnosis, medium on the patch being the right home
Tag:        NEEDS-PARTH
Reason:     Measured, not reasoned. With MSW running, MapLibre's module worker
            script request never settles and no error is raised. With MSW
            stopped (VITE_APP_MODE=api, so the worker never starts) the
            identical `new Worker(url, {type:'module'})` boots immediately.
            MSW's fetch handler resolves a controlling window client before
            matching handlers; a worker-script request has no such client, so
            the respondWith promise never resolves. MSW's own file already
            bypasses navigation requests for a comparable reason, so this
            follows the file's existing shape.
            Why NEEDS-PARTH: it edits a vendored, generated file. `msw init`
            would silently revert it. The patch carries a marker comment.
            Parth may prefer to pin the file, report it upstream, or drop MSW's
            service worker in favour of a different mock transport.
            The blob-URL alternative was tried and rejected on evidence: Vite's
            dev transform turns the 18.5 kB worker into a 129 kB module that is
            not valid standalone, and loading it from a blob raises an error
            event.

---

## D-007 · Session D · Committed the map work despite the map not rendering
Chose:      Commit Session D's code under an honest message that does not
            claim markers render, mark the session BLOCKED rather than DONE,
            and stop the run.
Rejected:   (a) Committing under the prescribed message
            `feat(frontend): maplibre canvas with mocked sighting markers`.
            (b) Leaving the whole session uncommitted in the working tree.
Confidence: high
Tag:        NEEDS-PARTH
Reason:     The verify gate passes — guard, typecheck and build all exit 0 —
            but the deliverable does not work: MapLibre never finishes loading
            its style, so `load` never fires, the GeoJSON source and circle
            layer are never added, and no tile is ever requested. The
            prescribed commit message asserts that markers render. They do
            not, and writing that message would be exactly the kind of
            plausible-looking false claim this project exists to prevent.
            Leaving it uncommitted was rejected because the branch already
            contains two independently verified fixes and a full diagnosis;
            a dirty tree is harder for Parth to untangle than one revertible
            commit. `git revert` removes it cleanly.

**Update after the map decision tree:** the map now works, on Leaflet. This
entry stands as the record of why the intermediate commit was worded the way
it was.

---

## D-008 · Map decision tree · Leaflet, and the state of the MSW patch
Chose:      Step 3. Leaflet 1.9.4 with OpenStreetMap raster tiles, plain
            Leaflet driven imperatively from `useMapInstance`. maplibre-gl and
            pmtiles uninstalled.
Rejected:   Steps 1 and 2, both tested and both failed — see WORKLOG for the
            measurements.
Confidence: high
Tag:        REVERSIBLE
Reason:     Step 1 (production build, no dev transforms) failed identically to
            dev. Step 2 (maplibre-gl 5.24.0) failed identically to 6.6.0, in
            dev and in preview. WebGL, the MSW worker-script hang, worker-
            initiated fetches and our own component were each ruled out by
            direct measurement. Leaflet renders basemap and markers first try,
            in dev and in a production build, and drops the bundle by 900 kB.
            Leaflet uses neither WebGL nor a tile-parsing worker, so it
            sidesteps the whole unexplained area rather than betting on it.
            The tile URL is marked TEMPORARY in `useMapInstance.ts`: OSM
            raster tiles need the network and the demo has to survive the
            network being unplugged. Offline tiles remain a separate task, and
            the PMTiles plan now needs rethinking since pmtiles was removed
            with maplibre — that package was a maplibre protocol adapter.

**On D-006, which Parth deferred:** with maplibre gone the app no longer
starts any web worker, so the `public/mockServiceWorker.js` patch is now dead
weight rather than load-bearing. I have deliberately NOT reverted it, because
D-006 was explicitly deferred and reverting is a decision, not housekeeping.
It is harmless — bypassing worker-script requests is correct behaviour for a
mock service worker regardless — and it carries a marker comment. Reverting it
is now a one-line cleanup whenever Parth wants it, and the diagnosis it
records stays valuable: MSW 2.15 hangs worker-script requests, and anything
that later introduces a worker will hit it again.

---

## D-009 · Session E · Rendering the word "online" on a camera chip
Chose:      Camera status chips read "Online" in sentence case.
Rejected:   Suppressing the label for that one state, or substituting a
            different word such as "Live" or "Up".
Confidence: medium
Tag:        NEEDS-PARTH
Reason:     Two instructions collide. SESSIONS.md section 2 says "Never render
            the word ONLINE anywhere in the app", in a paragraph about the
            status bar's live/replay badge. SESSIONS.md section 4 requires
            that every camera status carry "colour AND a text label AND a
            shape", and `online` is one of the four canonical values in
            Canonical 5.1.
            Read as a ban on the uppercase system-level badge — claiming the
            system is ONLINE when it is replaying — the two reconcile: the
            badge never says it, and a camera chip says "Online" in sentence
            case, which CONVENTIONS.md's copy rules call for anyway. Suppressing
            the label would leave that state identifiable by colour and shape
            alone, which section 4 explicitly forbids.
            Grep confirms the uppercase string appears nowhere in `src/`
            except the comment in CameraStatusChip.tsx explaining this.
            Flagged because the instruction was emphatic and I may have read
            its scope too narrowly. Reversing it is one string in one file.

---

## D-010 · Guard · What the journey-line rule does and does not cover
Chose:      Trigger on connector CREATION (Leaflet polyline/polygon, SVG
            `<line>`, SVG `<polyline>`) in any .ts/.tsx file, requiring an
            import of journeyLineStyle. Dropped the filename filter and the
            dash-syntax trigger.
Rejected:   (a) The previous rule, which fired on dash syntax in files named
            /journey/i. (b) Including `<path>` in the trigger. (c) A global
            dash-syntax rule.
Confidence: high on what it covers, and it covers less than the phrase "no
            code path produces a solid line" implies.
Tag:        REVERSIBLE
Reason:     The old rule could not catch what it was written for. A file
            drawing a SOLID line uses no dash syntax, so it passed silently --
            the precise failure the rule exists to prevent -- and a file named
            TimelineTrack.tsx was never examined at all. Regex cannot detect
            "solid", so the trigger is inverted to line creation, which it can
            detect.

            **Residual holes, stated rather than papered over:**

            1. `<path>` is not a trigger. LeftRail.tsx draws eight nav icons
               with `<path>`; including it produced 8 false positives on the
               real tree. An SVG `<path>` used as a connector is therefore not
               caught. Rejected because a rule that fails the build on every
               run for legitimate code gets disabled, and a disabled rule
               protects nothing.
            2. A dash-syntax rule cannot be applied globally either:
               LeftRail.tsx:51 and CameraStatusChip.tsx both use
               strokeDasharray legitimately, on the journey nav icon and the
               "unknown" camera status.
            3. **A CSS-drawn connector is not caught at all.** A Tailwind
               `border-l-2 border-dashed` on a timeline row is invisible to
               this rule, and so is a solid `border-l-2`. This is the most
               likely way a timeline connector actually gets built, so the
               timeline must take its styling from journeyLineStyle by
               convention and by review, not because the guard forces it.
            4. Line-based, like every rule here: a construction split across
               lines is not matched.

            The escape hatch is an IMPORT, not a mention. The first version
            used `src.includes("journeyLineStyle")` and was defeated by its
            own test file, which named the module in a comment and was
            silently exempted. Now matched with
            /(?:from|import)\s*\(?\s*["'][^"']*journeyLineStyle["']/.

            Verified: 0 hits on the real tree; fires on `L.polyline`,
            `new L.Polyline`, `<line>`, `<polyline>`, `L.polygon`; silent on
            circleMarker, tileLayer, `<svg>`, `<circle>`, `<path>` icons,
            strokeDasharray on icons, `<linearGradient>`, and on a file that
            imports journeyLineStyle.

---

## D-011 · Journey · Contract answers supplied by Parth, 2026-09-03
Chose:      Record these as settled.
Confidence: Parth's, stated per item
Tag:        REVERSIBLE
Reason:     The normative document is not in this repository and will not be
            until Manas's bootstrap lands, so these are Parth's readings of
            Canonical 6.3, not anything an agent can verify from source.
            - Path parameter `{plate_normalized}`: CONFIRMED. Mihir's backend
              manual says `?plate=` instead; that conflict is escalated to the
              team. endpoints.ts follows canonical, which wins.
            - `sightings` + `sighting_count` alongside `segments`: CONFIRMED.
            - Ordering: explicitly NOT confident. Sort client-side by
              first_seen_at ascending and comment why.
            - `disclaimer` non-null: CONFIRMED, mandatory in the response body
              so that no client can render a journey without it.

---

## D-012 · Journey · A response missing `disclaimer` is an error, not a journey
Chose:      Treat a journey response with no disclaimer as unusable. Error
            state naming the missing field, counted as a discard. Never render
            the journey, never invent a disclaimer.
Rejected:   Rendering the journey with the footer omitted.
Confidence: high
Tag:        REVERSIBLE
Reason:     Canonical 6.3 makes the field mandatory precisely so a client
            cannot render a journey without it, so a response lacking it is
            violating the contract and rendering anyway is the exact failure
            the mandate prevents. This is where "no conditional rendering
            path" and "the wire is untrusted" reconcile: there is no
            conditional around the footer because a response that could not
            produce one never reaches the renderer.

---

## D-013 · Journey · Three feasibility states, not two
Chose:      `feasible` becomes optional in AdaptedJourneySegment. readSegment
            stops defaulting an absent value to true. Three treatments: true
            normal, false warning, absent neutral grey labelled "plausibility
            not assessed". All three dashed.
Rejected:   The default-to-true introduced in Session B.
Confidence: high
Tag:        REVERSIBLE
Reason:     An unassessed segment styled identically to an assessed-and-passed
            one asserts a check that never ran -- the same class of error as a
            solid line. Touches adapters.ts only; api.ts is untouched and
            JourneySegment.feasible stays `boolean` there.

---

## D-014 · Journey · A discarded segment renders a gap marker
Chose:      When readSegment returns null, render an explicit "Segment data
            unusable, connection not shown" marker between the two nodes it
            would have joined.
Rejected:   Silence, which is the behaviour today.
Confidence: high
Tag:        REVERSIBLE
Reason:     "Flagged, never filtered" applies to discards too. Two adjacent
            nodes with nothing between them asserts that they are unconnected.
            That is an assertion by omission, and omissions are invisible in a
            way that wrong values are not.

---

## D-015 · Journey · Plate normalisation duplicated client-side, deliberately
Chose:      Normalise in the UI before building the request path: uppercase,
            strip every non-alphanumeric character.
Rejected:   Passing the raw query-string value straight into the path.
Confidence: medium
Tag:        NEEDS-PARTH
Reason:     The path parameter is `{plate_normalized}` and the UI route
            carries a raw `?plate=`. A bookmarked URL containing spaces must
            not 404 mid-demo. This knowingly duplicates backend logic: if the
            backend normalises differently the two disagree and we see 404s.
            Recorded so that when it happens the cause is one lookup away
            rather than an afternoon. Search does not have this problem
            because it reads `query.normalized` off the response; Journey
            cannot, because it needs the value to build the request.

---

## D-016 · Journey · `feasible:absent` is no longer recorded as a fallback
Chose:      readSegment leaves the key absent and records nothing. The three
            states are carried in the data and rendered.
Rejected:   The previous behaviour, which defaulted absent to `true` and
            recorded a `feasible:absent` fallback.
Confidence: high
Tag:        REVERSIBLE
Reason:     Two separate errors in one place. Defaulting to true styled a
            segment the backend never assessed identically to one it assessed
            and passed, asserting a check that never ran -- the same class of
            error as a solid connector. And absence is not a rename: counting
            it as a fallback inflated the drift number with something that is
            not drift, which devalues the counter that exists to catch real
            contract drift. It is now a visible rendered state, which is a
            better signal than a counter nobody reads. Verified at runtime:
            after four journeys the summary carries is_feasible, plate_normalized
            and required_speed_kmh, and no feasible:absent.

---

## D-017 · Journey · Connectors matched by camera identity, not array position
### RESOLVED — implemented in src/lib/journeyTimeline.ts
Chose:      Identity match first. Where a pair has more than one candidate,
            tiebreak on from_time falling inside the inclusive interval
            [ nodes[i].first_seen_at , nodes[i+1].last_seen_at ]. A segment is
            consumed by at most one pair. If the tiebreak leaves zero or more
            than one survivor, attach nothing, render a gap with reason
            'ambiguous', increment ambiguousPairCount, and consume nothing so
            both candidates still surface as unplaced.
Rejected:   Matching segment[i] to node pair i. Also rejected: picking the
            first or the closest candidate when the tiebreak fails.
Confidence: high
Tag:        REVERSIBLE
Reason:     Position cannot work -- readJourney discards unusable segments, so
            indices stop lining up with node pairs after the first discard, and
            a discarded segment is exactly the case the gap marker exists for.
            Verified: fixture A sends five segments, one is discarded, and
            positional matching would misalign every connector after it.

            **Interval bounds, and why the lower one is first_seen_at.** A
            sighting is an interval, not an instant. A traversal can begin the
            moment the vehicle is first observed at the departure camera, so
            waiting for that sighting's last_seen_at would exclude legitimate
            segments. The upper bound is the arrival node's last_seen_at for
            the mirror-image reason: arrival is anywhere within the arrival
            sighting's interval. Both bounds inclusive. Compared as parsed
            times, never as strings, so an offset-bearing timestamp still
            orders correctly.

            **Consumed at most once**, or a single A->B segment would serve
            every repetition of A->B in one journey and the screen would claim
            the backend assessed traversals it never saw. Verified with
            fixture E (GJ22KL0007): cam04 -> cam07 -> cam04 -> cam07 produced
            three distinct connectors, connectors[0].from_time
            14:00:01.000Z (E1) and connectors[2].from_time 14:20:01.000Z (E3).

            **Ambiguity is counted, not guessed.** Verified with fixture F
            (GJ22KL0008): two candidates both inside the window gave
            [gap:ambiguous], ambiguousPairCount 1, unplacedSegments 2. A wrong
            connector is a false assertion; a gap is an honest one.

            Remaining known limit: two candidates that differ only outside the
            window both fail it and produce a gap rather than a match. That is
            the safe direction.

---

## D-017-ORIGINAL · superseded, kept for the record
Chose:      For each adjacent pair of sorted sightings, find the segment whose
            from_camera_id and to_camera_id equal that pair's camera_ids.
Rejected:   Matching segment[i] to node pair i.
Confidence: high on the rule, low on one edge case
Tag:        SUPERSEDED by D-017 above
Reason:     Position cannot work. readSegment discards unusable segments, so
            indices stop lining up with node pairs after the first discard --
            and a discarded segment is exactly the case the gap marker exists
            for. Verified: fixture A sends five segments, one is discarded, so
            positional matching would misalign every connector after it.
            **The edge case, which I cannot resolve alone:** if the same camera
            pair occurs twice in one journey -- a vehicle going A to B, then
            back, then A to B again -- identity matching is ambiguous and will
            attach the same segment to both pairs. Time-window matching
            (segment.from_time within the pair's interval) would disambiguate,
            but that is a second matching rule and I would rather Parth chose
            than have me invent one. No fixture exercises this today.

---

## D-018 · Journey · One copy string covers two different causes
### RESOLVED
Chose:      "Segment data unusable — connection not shown" renders both for a
            segment readSegment discarded and for an adjacent node pair the
            API sent no segment for. Both surface as
            { kind: 'gap', reason: 'no_segment' }.
Rejected:   Two distinct strings.
Confidence: high
Tag:        REVERSIBLE
Reason:     They are not distinguishable at render time. readSegment returns a
            bare `null` and the fields identifying which pair it belonged to --
            from_camera_id, to_camera_id -- are among the fields whose absence
            causes the discard. By the time the screen runs, "we received a
            broken segment" and "we received no segment" look identical. One
            honest string beats two that guess at a cause. If the distinction
            matters operationally, readSegment would have to return a richer
            failure value, which is a bigger change than this screen.

---

## D-019 · Guard · The journey-line rule checks the import, not the call site
Chose:      Accept that a file importing journeyLineStyle passes the rule even
            if a particular call site ignores it.
Rejected:   Attempting to verify that every connector call site spreads the
            returned options.
Confidence: high
Tag:        REVERSIBLE
Reason:     A regex cannot tell `L.polyline(pts, journeyLineStyle(s))` from
            `L.polyline(pts, { color: 'red' })` in a file that also imports the
            module somewhere. The import check is a tripwire, not a proof. The
            primary defence is the module's API: every branch returns a
            dashArray, the table is frozen, and there is no argument that
            produces a solid line -- so the easy path is the correct one and
            the incorrect path requires deliberately hand-writing options.
            Verified end to end this session: an SVG `<line>` with no import
            failed the build and named the file; the same line with the import
            passed.

---

## D-020 · Journey · sighting_count is optional, not a discard trigger
Chose:      `AdaptedJourney.sighting_count?: number`. Absent omits the key,
            records no fallback, records no discard, and does not return null.
            `disclaimer` and `plate` remain hard discard triggers.
Rejected:   My previous behaviour, which returned null and recorded a
            `sighting_count` discard.
Confidence: high (Parth corrected me)
Tag:        REVERSIBLE
Reason:     I over-applied the nullable-return principle. That principle exists
            to stop the adapter INVENTING a value, not to throw away values
            that are already present. A response with usable sightings, real
            cameras, real times and a verbatim disclaimer was being discarded
            entirely to avoid rendering one cross-check number. The technical
            reference marks a missing disclaimer as a whole-screen error state
            explicitly and gives no such rule for the count: count is the
            server's claim ABOUT the evidence, not the evidence. Verified with
            fixture G (GJ22KL0009): the journey adapts, sighting_count_present
            is false, and the timeline still builds one connector.

---

## D-021 · Journey · The required_speed_kmh fallback is removed
Chose:      Record nothing when `required_speed_kmh` is absent. The key stays
            off the object and the screen renders "Speed unavailable".
Rejected:   Keeping the fallback counter entry.
Confidence: high
Tag:        REVERSIBLE
Reason:     Checked what actually triggered it before touching it: the call
            fired on `readNumber(raw["required_speed_kmh"]) === null`, which is
            ABSENCE. There is no alternative spelling of this field anywhere in
            the adapter, so it was never detecting a rename. Same reasoning as
            `feasible:absent`: absence is not drift, counting it inflated the
            drift number with something that is not drift, and it already
            renders as a visible state, which is a better signal than a
            counter nobody reads. Verified: totalFallbacks dropped from 4 to 3
            across the same seven journeys, and the surviving fallbacks are
            camera_name, evidence_count, plate_normalized and is_feasible --
            two real substitutions and two real renames.

---

## D-022 · Journey · 'no_time_match' split out of 'ambiguous'
Chose:      `reason: 'no_segment' | 'ambiguous' | 'no_time_match'`, with
            `ambiguousPairCount` counting only 'ambiguous' and a new
            `noTimeMatchPairCount` counting the other.
Rejected:   One 'ambiguous' reason covering both outcomes.
Confidence: high
Tag:        REVERSIBLE
Reason:     Zero survivors and several survivors are different facts. Calling
            the zero case "ambiguous" is the same defect as the
            required_speed_kmh counter that fired on absence: a counter
            reporting one thing under the name of another, which makes the
            number useless the moment anyone tries to act on it. All three
            reasons still render the identical string on screen per D-018 --
            the split is for System Status and for debugging, not for the
            officer. Verified with fixture H (GJ22KL0010): two segments
            matching by camera identity, neither inside the window, giving
            noTimeMatchPairCount 1 and ambiguousPairCount 0.

---

## D-023 · Journey · A connector into an unplaced node is not drawn, and is counted
Chose:      When either endpoint sighting has a null lat or lon, the connector
            is omitted from the map layer entirely and
            `undrawableConnectorCount` is incremented.
Rejected:   Drawing a partial line; substituting the CAMERA's coordinates for
            the sighting's.
Confidence: high
Tag:        REVERSIBLE
Reason:     **This case appears in no existing document.** The null-coordinate
            rule was written for nodes; nothing said what happens to the line
            between them. A JourneySegment carries no coordinates of its own,
            so a connector's endpoints can only come from the two sightings it
            joins. Substituting the camera's position looks reasonable and is
            not: the camera is where the device is, the sighting is where the
            vehicle was seen, and a line drawn to the former asserts a
            journey leg between two places one of which we do not know.
            Counted rather than silent, because an omitted line is an
            assertion by omission.

            Note on fixture coverage, found by running it: fixture A does NOT
            exercise this path. Its unplaced node (cam09) sits in the single
            pair whose segment readSegment discards, so that pair is a gap and
            never becomes a candidate connector. Fixture I (GJ22KL0011) was
            added specifically to reach it: cam04 placed, cam09 unplaced, one
            well-formed segment joining them -> drawableConnectors 0,
            undrawableConnectorCount 1.

---

## D-025 · Realtime · A real killable ws server, not socket interception
Chose:      scripts/mock-ws.mjs, a standalone Node process using `ws`.
Rejected:   msw's WebSocket API, which IS available in 2.15.0 -- `./core/ws`
            is an export subpath and `ws` / `WebSocketHandler` are exported
            from core/index.d.ts. This was a choice, not a limitation.
Confidence: high
Tag:        REVERSIBLE
Reason:     The claim worth proving is "it reconnects after the server dies",
            and an intercepted socket cannot be Ctrl+C'd. Killing a real PID
            is the only way that claim gets tested rather than asserted. It
            was immediately vindicated -- see D-026, which was only findable
            because a real server could actually be absent.

---

## D-026 · Realtime · onopen is not evidence of a connection
Chose:      Promote to `connected`, and reset the backoff, on the FIRST FRAME
            received -- never on `onopen`.
Rejected:   The obvious `socket.onopen = () => { attempt = 0; status =
            'connected' }`, which is what I wrote first.
Confidence: high, measured
Tag:        REVERSIBLE
Reason:     Measured, not reasoned. With MSW running, `new WebSocket()` fires
            `onopen` even for a port nothing has ever listened on. A direct
            probe against ws://localhost:59999/nowhere returned "ONOPEN fired
            on a dead port". MSW's interceptor opens the client side first and
            attempts passthrough afterwards, so open-ness says nothing about
            whether a server exists.
            The consequence was severe and silent: resetting the backoff in
            onopen produced a permanent ~1s reconnect loop against a killed
            server -- observed as `retry 1 in 1065ms` repeating forever, 19
            "reconnects" in two minutes -- while the UI read `connected` with
            zero data flowing. That is precisely the lie the watchdog exists
            to prevent, arriving through a door the watchdog does not cover,
            because the socket closed at ~3s and the 30s watchdog never fired.
            The fix is the same principle the watchdog already encodes:
            traffic is evidence of a working connection, an open handle is
            not. After the change the backoff escalates correctly (1.1s, 2.3s,
            4.1s, 7.0s, 12.9s, 24.4s) and the status reaches `offline`.
            Cost: `connected` now lags the true open by up to one message
            interval. That is honest rather than optimistic.

---

## D-027 · Realtime · Watchdog at 30s, and why not lower
Chose:      30 seconds of silence force-closes the socket. Reset on ANY
            traffic, not only heartbeats.
Rejected:   15s (one heartbeat interval); resetting only on heartbeats.
Confidence: high
Tag:        REVERSIBLE
Reason:     Two missed beats at a 15s heartbeat. One missed beat is a late
            packet, not a dead link, and closing on it would produce a
            reconnect storm on a congested venue wifi. Never set below 25s for
            that reason. Resetting on any traffic because a socket delivering
            a sighting every 3s is demonstrably alive whether or not a
            heartbeat landed -- keying the watchdog to heartbeats alone would
            kill a healthy, busy socket.
            The underlying reason for having one at all: a socket held open by
            a NAT with nothing flowing is indistinguishable, from readyState,
            from a quiet night. "Connected" while nothing arrives is the same
            category of lie as labelling replay ONLINE.

---

## D-028 · Realtime · The envelope key disagreement is measured, not settled
Chose:      The mock emits `data` for every message EXCEPT every fifth alert,
            which uses `payload`. The client reads `msg.data ?? msg.payload`
            and logs which key arrived.
Rejected:   Emitting only `data` and trusting the canonical spelling.
Confidence: high
Tag:        REVERSIBLE
Reason:     Canonical says `data`, Mihir's backend manual says `payload`, and
            the disagreement is unresolved. Reading both is defensive; logging
            which one arrived turns the client into the instrument that
            settles it from real traffic rather than from a meeting. Observed
            in this session: `[ws] envelope key "payload" received for type
            "alert" (canonical is "data")`. heartbeat is exempt from the whole
            question -- it carries `ts` at the TOP level with no wrapper, and
            a parser that blindly reads msg.data.x throws on every beat. That
            shape exists to catch exactly that bug and is emitted deliberately.

---

## D-024 · Journey · The "Journey data incomplete" string
### RESOLVED — the string stays, and it does not name the field
Chose:      When readJourney returns null the screen renders:
            "Journey data incomplete. The response was missing required
            fields, so nothing is shown rather than a partial route."
Rejected:   Reusing a generic error string, or rendering the journey without
            its disclaimer.
Confidence: high (resolved by Parth)
Tag:        REVERSIBLE
Resolution: The operator-facing error deliberately does NOT name the missing
            field. "The disclaimer was missing" is not actionable to an
            officer, and the discard key already reaches the drift counters,
            which is where a contract problem belongs and where an engineer
            will look for it.
Reason:     D-012 makes a response without a disclaimer a whole-screen error
            state, and the copy contract has no string for it. This one was
            supplied by Parth in the session brief and is used verbatim. It is
            tagged NEEDS-PARTH because it is new copy in a police tool that has
            not been through whatever review the rest of the copy contract had,
            and because it deliberately does not name WHICH field was missing:
            the adapter knows, but telling an officer "disclaimer was missing"
            explains nothing they can act on. If that judgement is wrong, the
            adapter already records the field name as a discard key and the
            string could carry it.

---

## D-029 · Realtime · The watchdog abandons the socket itself, it does not wait for onclose
Chose:      When the watchdog fires it detaches the handlers, clears
            activeSocket, calls scheduleRetry() directly, and only then calls
            close(). The status change no longer depends on the close event
            arriving.
Rejected:   The original shape -- close(4000, 'watchdog') and let onclose call
            scheduleRetry, which is the tidier code and has one exit path.
Confidence: high (measured, not reasoned)
Tag:        REVERSIBLE
Reason:     Measured on the first run that ever made the watchdog fire. The
            watchdog fired correctly at 30958ms with readyState 1 (OPEN), but
            close() to onclose then took 13 seconds against an MSW-intercepted
            socket, and for all 13 of those seconds the UI still read "Live"
            over a connection this client had already declared dead. Total
            elapsed from last frame to the status changing was 44.0s, outside
            the 25-40s band the watchdog is supposed to guarantee.
            That is the exact lie the watchdog exists to prevent, arriving
            through the watchdog's own code path. Close delivery is not ours to
            control -- it crosses an interceptor, and in production it will
            cross a network -- so the user-visible status must not be gated on
            it. Handlers are detached BEFORE close() so the late onclose cannot
            schedule a second, competing retry.
            After the fix: fired at 30049ms, readyState 1, status left "Live"
            1ms later. Same run, same conditions.

---

## D-030 · Alerts · On an alert_id collision the fetched copy wins
Chose:      mergeAlertFeed writes live alerts into the map first and fetched
            alerts second, so a REST record overwrites a socket record with the
            same alert_id.
Rejected:   Socket-wins (newest arrival wins), and keeping both.
Confidence: medium
Tag:        REVERSIBLE
Reason:     The socket copy is a point-in-time emission; the REST copy is what
            the server currently holds. They differ exactly when something has
            changed since the emission -- an acknowledgement, a re-scored
            match -- and in every one of those cases the stored row is the
            truthful one. Showing the socket copy would contradict the backend
            while claiming to be live.
            This is not merely theoretical here: the reconnect refetch
            redelivers ids the socket already delivered, so the collision path
            runs on every reconnect. Proven under --burst 130: after a
            reconnect redelivered all 130 identical alr-burst-* ids plus a REST
            refetch of the same 9 ids, the list stayed at exactly 100 rows.

---

## D-031 · Alerts · The empty state omits a watchlist count
Chose:      The empty state is exactly "No alerts." with no second line.
Rejected:   "No alerts. N plates on the watchlist.", which reads better and
            tells the operator the system is armed rather than asleep.
Confidence: high
Tag:        REVERSIBLE
Reason:     There is no watchlist query in this client and no watchlist path in
            endpoints.ts. The number would have to be invented or hardcoded,
            and a hardcoded count next to the words "No alerts" is a claim
            about system state that nothing verifies. On a projector it is
            indistinguishable from a real one. Same principle as never
            rendering a plate we did not read: silence beats a plausible
            fabrication. Revisit when a watchlist endpoint exists.

---

## D-032 · Realtime · The status bar socket chip is driven by the store
Chose:      StatusBar's SocketChip reads wsStatus from liveStore.
Rejected:   Leaving it hardcoded and out of this session's scope.
Confidence: high
Tag:        CONFIRMED (Parth, write-paths session)
Resolution: Stays. Two indicators disagreeing on one screen is the failure this
            architecture exists to prevent, and a comment holding a truth that
            expires is not enforcement. Re-verified this session on a screen
            that is not Alerts: with the socket killed the chip on /cameras read
            "Live updates offline", and it returned to "Live updates on" when
            the server came back.
Reason:     It was hardcoded to "Live updates offline" with the comment
            "Hardcoded until D4 wires the socket. It reports offline because
            that is the truth today, not as a placeholder value." That
            reasoning was correct when written and expired when the socket
            landed. Caught during the Alerts regression pass: the header read
            "Live updates offline" while alerts were arriving over a healthy
            socket, and it sat directly above an Alerts status line reading
            "Live". Two indicators on one screen disagreeing about the same
            fact costs more trust than having no indicator, and the
            contradiction is most visible in precisely the situation the
            status bar exists for. This is a correction of a string that had
            become false, not a new feature.

---

## D-033 · Realtime · The Connecting…/Reconnecting… alternation stays as it is
Chose:      A fresh page retrying a dead server keeps alternating "Connecting…"
            and "Reconnecting…" until it settles on "Disconnected — showing last
            known data" at the sixth attempt.
Rejected:   Latching to one string, or suppressing the transitions behind a
            debounce so the header looks calmer.
Confidence: high (resolved by Parth)
Tag:        REVERSIBLE
Reason:     Each string is individually truthful at the moment it is shown:
            open() really is connecting, and scheduleRetry really is about to
            reconnect. Suppressing accurate transient state to look calmer is
            the wrong trade in a tool whose whole claim is that the status line
            can be believed. It settles on its own, and the settled state is the
            one an operator acts on.

---

## D-034 · Realtime · An inactive query defers its fetch rather than losing data
Chose:      No fix. The reconnect invalidation is a no-op for a screen that is
            not mounted, and the fetch happens when the screen mounts.
Rejected:   Forcing a refetch of inactive queries on reconnect.
Confidence: high (resolved by Parth)
Tag:        REVERSIBLE
Reason:     Measured in the alerts session: sitting on /journey across a kill and
            restart, the invalidation ran but the alerts fetch count stayed at 1,
            because invalidate only refetches queries with an active observer.
            Navigating to Alerts mounted the query stale and fetched immediately.
            The staleness recorded during the outage survives the outage, so the
            gap is deferred fetching, not silent data loss. Forcing inactive
            refetches would put every screen's queries on the wire on every
            reconnect for no benefit an operator can see. Worth remembering only
            because "it appeared when I clicked the tab" looks like a bug and is
            not one.

---

## D-035 · API · Writes do not retry, and no retry may be added later
Chose:      apiPost and apiDelete issue exactly one request. No retry, no
            backoff.
Rejected:   Inheriting the QueryClient's retry: 1, or adding a retry for
            "transient" failures.
Confidence: high
Tag:        PERMANENT
Reason:     A retried POST is a duplicate write. The first attempt may have
            reached the backend and committed before the response was lost, so a
            retry acknowledges twice, or adds the same plate twice. apiGet's
            retry policy is safe only because GET is idempotent, and it lives on
            the QueryClient rather than in the client module precisely so it does
            not follow the verb around. The failure mode a retry would paper over
            is one the operator must see, because a write that fails silently
            leaves them believing an alert is acknowledged when the backend never
            heard about it.

---

## D-036 · Alerts · Acknowledge is optimistic, and a rollback is always visible
Chose:      Flip the boolean immediately in both the query cache and the live
            store, roll both back on error, invalidate ['alerts'] on settle, and
            render an ErrorPanel carrying request_id whenever a rollback happens.
Rejected:   A pending spinner with no optimistic flip; or rolling back silently.
Confidence: high
Tag:        REVERSIBLE
Reason:     acknowledged is a BOOLEAN, so the optimistic write flips a boolean and
            never invents a timestamp the server has not confirmed. The rollback
            snapshots the previous value rather than assuming the flip can be
            reversed, because an alert may already have been acknowledged by
            someone else.
            A silently reverted checkbox is a lie: the operator watched it change.
            Measured in 4b against a forced 500 -- at 80ms the row read
            "Acknowledged", at 1.5s it read "Acknowledge" again, and the panel
            showed "The server failed on this request ... Request id
            req_mock_ack_500". The flip, the rollback and the reason are all
            observable, which is the whole requirement.
            The live store is written as well as the query cache because an alert
            that arrived only over the socket has no row in the REST list, and a
            cache-only optimistic write would not show on it at all.

---

## D-037 · Plate · One normaliser, in src/lib/plate.ts
Chose:      normalisePlate lives in src/lib/plate.ts. Journey imports it, and the
            watchlist form imports it through src/lib/watchlist.ts.
Rejected:   Leaving Journey's inline copy and writing a second one for the form.
Confidence: high
Tag:        REVERSIBLE
Reason:     Two normalisers that agree today drift the first time one of them
            learns about a new separator, and the disagreement only ever shows up
            against the real API, as a 404 on a plate the user can see is correct.
            Verified by execution: "gj 01 ab 1234", "GJ-01-AB-1234" and
            " gj01ab1234 " all yield GJ01AB1234 through both call paths, and the
            three inputs collapse to a single TanStack query key.
            Deliberately NOT shared with src/mocks/handlers.ts, which normalises
            the same way inline. That code simulates the SERVER's normalisation,
            and a mock importing the client's rule could never disagree with it --
            which is precisely the disagreement worth catching.

---

## D-038 · Alerts · A socket redelivery UN-acknowledges a socket-only alert
Chose:      Reported, not patched. mergeAlertFeed is unchanged.
Rejected:   Making acknowledged sticky in the merge, or in liveStore.addAlert,
            on my own judgement.
Confidence: high (the measurement); the fix is Parth's call
Tag:        RESOLVED (Parth, offline-basemap session)
Resolution: Sticky acknowledgement, implemented as directed.
            liveStore gains locallyAcknowledged, a Set of ids written ONLY on a
            confirmed 2xx from the acknowledge mutation and removed on rollback.
            mergeAlertFeed takes it as a third argument, stays pure, and applies
            it last so it survives dedup, sort and cap. It can only ever force
            acknowledged TRUE -- a set of confirmed writes must never be able to
            un-acknowledge something the server itself reported as acknowledged.
            The set is pruned on every merge to the ids that survived the cap,
            so it cannot grow across a long shift. Memory only: the storage rule
            bans persistence, and rightly, since an acknowledgement outliving the
            tab would be a claim about server state that nothing reconciled.
            The reasoning that settles it: the server confirmed the write with a
            204. A socket frame arriving afterwards carrying acknowledged:false
            PREDATES that write. It is stale, not a correction, and letting it
            win asserts the operator's action did not happen.
            Re-ran session G's 4c. Acknowledged alr-burst-0000 (204), forced a
            reconnect so the mock redelivered the same alert_id with
            acknowledged:false, and the row still read "Acknowledged" -- one live
            acknowledged row, where session G measured zero.
            Also verified the flag is not written optimistically: during the
            forced-500 run the row flipped to "Acknowledged" at 120ms while
            locallyAcknowledged stayed EMPTY, then both rolled back. A prediction
            never becomes sticky; only a confirmation does.
Reason:     4c, run deliberately. Acknowledged alr-burst-0000 (a socket-only
            alert, POST returned 204), confirmed the row read "Acknowledged",
            then forced a reconnect so the mock redelivered the SAME alert_id.
            After the reconnect, zero live rows were acknowledged. The row
            un-acknowledges.
            mergeAlertFeed's fetched-wins rule (D-030) does not protect this case,
            because a socket-only alert has no fetched copy to win: it exists only
            in the live store, and liveStore.addAlert replaces the stored entry
            wholesale with the fresh socket copy, whose acknowledged is false.
            An alert that IS in the REST list stays acknowledged, because the
            fetched copy wins and the mock's acknowledge state is stateful.
            Whether this bites in production depends on whether the backend's
            /alerts includes recently-pushed alerts. If it does, the next refetch
            heals it. If it does not, an operator can watch an acknowledgement
            undo itself, which is the same category of lie as a silent rollback.
            Two candidate fixes, both one-liners, neither taken without a
            decision: have addAlert preserve the existing acknowledged value when
            replacing a known id, or have the merge treat acknowledged as sticky.

---

## D-039 · Watchlist · The Watchlist types live in ui.ts, not the canonical mirror
Chose:      WatchlistEntry, WatchlistDraft and WatchlistPriority are declared in
            src/types/ui.ts.
Rejected:   Adding them to src/types/api.ts.
Confidence: high
Tag:        REVERSIBLE
Reason:     No Watchlist or WatchlistEntry type exists in Canonical 6.5, and
            api.ts is a mirror that must not be extended on our judgement. The
            shape came from the session brief, not from the contract, so it lives
            with the other non-canonical types until the contract documents it --
            at which point it moves and the comment moves with it.
            WatchlistEntry carries an `id` because endpoints.watchlistItem takes
            an id rather than a plate, so the soft delete has to address the entry
            rather than the vehicle. That is an inference from the path signature,
            not from a documented response body, and it is the first thing to
            check against the real API.

---

## D-040 · Watchlist · It stays a panel on Alerts, not a seventh route
Chose:      The watchlist add form and list stay inside the Alerts screen.
Rejected:   A /watchlist route with its own rail entry.
Confidence: high (resolved by Parth)
Tag:        CONFIRMED
Reason:     Six screens is the fixed surface. An eight-minute demo cannot spend
            a beat on a seventh route, and the rail is a navigational promise --
            every entry added is a place the presenter has to explain or
            conspicuously skip. The watchlist is also where the alerts come
            from, so it reads better beside them than behind another click.
            NUMBERING NOTE: the session brief asked for this as D-039, but D-039
            was already taken last session by the Watchlist types decision. Filed
            as D-040 rather than overwriting a live entry.

---

## D-041 · Errors · INTERNAL_ERROR copy made true for writes, and a full audit
Chose:      Rewrote four cases in the exhaustive switch in src/lib/errors.ts.
Rejected:   Changing only INTERNAL_ERROR; and rewriting NOT_FOUND.
Confidence: high
Tag:        REVERSIBLE
Reason:     The switch was written when every path was a read. The first write
            paths landed last session and immediately produced copy that an
            operator cannot act on. Before and after, verbatim:

            INTERNAL_ERROR detail
              was: "Retry, or narrow the time range."
              now: "The server reported an internal fault. Retry, and report
                    this with the request id if it continues."
              Narrowing a range is advice for a slow query, not a server fault,
              so it was misplaced for reads too; on a failed acknowledge there
              is no range at all. It deliberately does NOT say whether anything
              was written: on a 500 from a POST the write may have partly
              committed, and "nothing was changed" would be a fabricated
              reassurance.

            VALIDATION_FAILED title
              was: "That filter isn't valid"
              now: "That request isn't valid"
            VALIDATION_FAILED detail (no-field branch)
              was: "Check the filter values and try again."
              now: "Check the values and try again."
              "filter" is read-only vocabulary; the same code now rejects a
              watchlist add, which has fields and no filters.

            DUPLICATE_EVENT title
              was: "Duplicate event"
              now: "Already recorded"
            DUPLICATE_EVENT detail
              was: "The API has already recorded this event. No action is needed."
              now: "The API has already recorded this. No action is needed."
              "event" is ingestion vocabulary. The code lands on a watchlist add
              whose plate is already watched, which is not an event.

            DEPENDENCY_UNAVAILABLE detail
              was: "Search and history are unavailable. Live updates continue."
              now: "Stored data cannot be read or written until it recovers."
              Two faults, not one. It framed the outage as read-only, and its
              second sentence asserted the socket was healthy -- a claim this
              layer cannot make, since it sees one failed HTTP response and
              cannot observe the socket at all. That claim belongs to the status
              line, which measures it.

            AUDIT, cases left alone and why:
              UNKNOWN_CAMERA, UNSUPPORTED_SCHEMA_VERSION, UNKNOWN, NON_JSON,
              default, and the network fallback are all verb-neutral already.
              NOT_FOUND is NOT neutral and was deliberately not changed. Its
              detail, "Nothing matched this request. Widen the time range, or
              check the plate", is read-only advice, and kind:"empty" renders a
              404 as a calm no-results panel -- wrong for a DELETE of a watchlist
              entry that is not there, which is a real failure. Fixing it
              properly needs to know the verb, which this layer never receives,
              and a verb-neutral string would strip the search empty state of its
              only useful advice to serve a rare write edge. Left alone with an
              audit note in the source rather than made worse. The fix is a
              decision about plumbing, not a rewrite.

---

## D-042 · Map · Offline basemap is pre-seeded raster tiles, not PMTiles
Chose:      Candidate B. A selectable offline source reading
            public/tiles/{z}/{x}/{y}.png, with the online OSM raster source
            unchanged and still the default, switched by VITE_BASEMAP.
Rejected:   Candidate A, protomaps-leaflet@5.1.0 reading a .pmtiles archive.
Confidence: high on the measurements; medium on the licence conclusion, which
            depends on where Parth actually sources tiles
Tag:        REVERSIBLE
Reason:     Measured, both candidates installed and built.

            BUNDLE. Baseline index js is 492.65 kB (NOT the ~426 kB the brief
            assumed -- that number predates the write-paths session, and is
            reported here rather than repeated).
              Candidate A: 619.25 kB, +126.60 kB raw / +35.87 kB gzip, measured
                by actually importing leafletLayer so Rollup could not shake it.
              Candidate B: 492.96 kB, +0.31 kB. It is a URL string; public/ is
                copied, never bundled.

            DISK. Zoom range measured from the code, not assumed: the map opens
            at z11 (useMapInstance ZOOM), fitBounds caps at z15 (MapCanvas
            maxZoom), and the layer allows z19 with scrollWheelZoom on. Demo bbox
            from the fixture coordinates is lat 22.9871..23.0521, lon
            72.5115..72.6647. Tiles to cover it, padded:
              z10-15   294 tiles     ~5.7 MB
              z10-16 1,086 tiles    ~21.2 MB
              z17     2,982 tiles   ~58 MB
              z19    47,094 tiles  ~920 MB
            Capped at z16: one level of scroll headroom, and z17 upward is where
            it stops being a demo asset and starts being a download.

            LICENCE, the deciding constraint, stated plainly.
              Bulk downloading tiles from OSM's public tile servers VIOLATES the
              OSM Tile Usage Policy. Any plan that seeds public/tiles by
              scraping tile.openstreetmap.org is BLOCKED and must not be run.
              Candidate A's data source, a Protomaps daily basemap build, is
              ODbL and is published expressly to be downloaded whole, so on
              licence A is the cleaner path. That is the one axis A wins.
              Candidate B is licence-neutral: it renders whatever is in the
              directory, so it is exactly as licensed as its source. Real tiles
              must come from somewhere that permits bulk export -- a
              self-rendered set from OSM data, or a provider whose terms allow
              offline caching. The WORKLOG procedure says so, and the placeholder
              generator exists so the plumbing can be proven without touching
              anyone's tile server.

            VERIFIABILITY, which decided it. Candidate A could not be rendered
            in this session at all. Producing a .pmtiles extract needs the Go
            `pmtiles` CLI; the npm package ships no bin (checked: bin is absent),
            and downloading a binary is not something I will do unprompted. So A
            would have been committed unrendered, on the strength of a bundle
            measurement and nothing else. MapLibre already cost a day in this
            environment by failing in a way nobody could explain, and shipping a
            second unverified map renderer into a hackathon demo is the same bet
            twice. B rendered: 15/15 tiles from localhost, 12 markers,
            attribution correct, journey connectors still three dashed lines.

            GLYPHS, SPRITES, FONTS. B needs none -- raster PNGs carry their own
            labels. A would need glyphs and sprites, which @protomaps/basemaps
            bundles locally, so A would also have been self-contained. Neither
            candidate fails on this axis. The app's own fonts were already local
            @fontsource packages, with no remote font host anywhere.

---

## D-043 · Process · The mojibake gate has a fixed, stated scan set
Chose:      scripts/mojibake.mjs, run as `npm run mojibake`. Scan set: src/ and
            scripts/ recursively, plus root-level *.md; extensions .ts .tsx .css
            .mjs .js .md .json; node_modules, dist and dot-directories skipped;
            nothing excluded by name.
Rejected:   Continuing to retype a PowerShell one-liner each session.
Confidence: high
Tag:        REVERSIBLE
Reason:     The denominator moved without explanation: session F reported 45
            files, session G reported 43, while src/ grew by two. Reconciled --
            F additionally scanned four root .md files and excluded WORKLOG.md,
            against a src/ that was two files smaller (41 code + 4 docs = 45);
            G scanned 43 code files and no docs. Nothing was ever dropped
            silently. But nobody could establish that from the numbers alone,
            and a gate whose denominator moves is a gate that cannot tell you it
            passed.
            Session F's set also named a `docs` root that does not exist, under
            -ErrorAction SilentlyContinue, so it contributed zero files and said
            nothing about it. CONVENTIONS.md points at docs/TRINETRA_Canonical_
            Contracts.md, and that file is not in this repository.
            Lines carrying the token MOJIBAKE-ALLOW are skipped, because three
            WORKLOG passages and the pattern list itself legitimately contain the
            sequences. The exemption is line-scoped, not file-scoped, so real
            corruption elsewhere in the same file is still caught, and the
            pattern list is written as \u escapes so the gate cannot exempt
            itself by name.

---

## D-044 · Errors · NOT_FOUND resolved by passing the verb, not by rewriting copy
Chose:      toDisplayError(error, { operation?: 'read' | 'write' }), defaulting
            to 'read'. ErrorPanel forwards an optional `operation` prop. Only the
            acknowledge and watchlist mutation panels pass 'write'.
Rejected:   Rewriting the NOT_FOUND string into something verb-neutral.
Confidence: high (resolved by Parth)
Tag:        REVERSIBLE
Reason:     D-041 recorded that this could not be fixed because the layer never
            received the verb. It receives it now, so the fix is the verb.
            Rewriting the string would have cost the search empty state its only
            useful advice in order to serve a rare write edge -- paying the
            common path to fix the uncommon one.
            NOT_FOUND, read (unchanged, and every existing call site is on this
            path because the parameter defaults):
              kind    "empty"
              title   "No results"
              detail  "Nothing matched this request. Widen the time range, or
                       check the plate."
            NOT_FOUND, write (new):
              kind    "error"
              title   "Not found"
              detail  "The server has no record of this. Refresh to see the
                       current state."
              retry   false -- retrying a delete against an id the server does
                      not have fails identically; refreshing is what helps.
            A read that matches nothing is a legitimate empty state. A write
            that cannot find its target is a failure, and must not render as a
            calm no-results panel.

---

## D-045 · Status · Drift counters are labelled since page load, and by read
Chose:      The Contract drift section is labelled "Counted since page load, in
            memory, reset on reload -- not cumulative system figures. Counts
            adapter reads, not distinct records."
Rejected:   Presenting the numbers bare, as system totals.
Confidence: high
Tag:        REVERSIBLE
Reason:     Two separate false claims were available here and both are refused.
            First, the counters are module-level in adapters.ts and reset on
            every reload. Presenting them as cumulative system figures would
            invite "the API dropped 6 records" when the truth is "this tab saw
            6 discards since you opened it".
            Second, and found by measurement rather than reading: the counter
            increments once per adapter READ, not once per distinct bad record.
            journeyFourCamera contains exactly one deliberately malformed
            segment (S4, to_time omitted), and opening that journey three times
            showed to_time 2, then 4, then 6 -- two per visit, because React
            StrictMode double-invokes in development, and again on every
            revisit. The original heading, "a record was dropped entirely",
            would have read as six lost records. It now reads "a record was
            unusable and dropped on read", and the section says what it counts.
            Deduplicating the counter itself was NOT done: that changes adapter
            semantics, which is contract-adjacent, and it is a decision rather
            than a wording fix. Flagged for Parth.

---

## D-046 · Status · apiBaseUrl is labelled build-time configuration
Chose:      The Build section is titled "Build-time configuration" and says the
            values are inlined when the bundle is built.
Rejected:   Showing apiBaseUrl bare, which reads as live configuration.
Confidence: high
Tag:        REVERSIBLE
Reason:     import.meta.env.VITE_API_BASE_URL is substituted at build time and
            baked into the artifact. An operator reading "API base URL:
            http://localhost:8000" on a status screen would reasonably conclude
            the running system can be pointed elsewhere; it cannot, without a
            rebuild. The same artifact is not repointable, and a status screen
            that implies otherwise sends someone to edit an env file that the
            running bundle will never read.

---

## D-047 · RISK · public/tiles/ is gitignored, so a clean clone has no basemap
Chose:      Recorded as a risk. Nothing attempted this session.
Rejected:   Committing the placeholder tiles, or committing real tiles, to make
            a clone self-sufficient.
Confidence: high on the risk; the resolution is Parth's
Tag:        NEEDS-PARTH
Reason:     D-042 gitignored public/tiles/ for good reasons -- 2,064 fake PNGs
            in the history would look like a finished offline basemap, and real
            tiles are a ~21 MB deployment artifact rather than source. The
            consequence is unresolved and needs stating plainly: a fresh clone
            on any machine but Parth's has NO offline basemap. Running with
            VITE_BASEMAP=offline there renders an empty grid.
            This blocks G6 anywhere but the machine where the tiles were
            generated. Options, none taken: commit real tiles despite the size;
            publish them as a release artifact with a fetch script; document
            regeneration as a setup step; or drop offline mode and accept the
            network dependency. Each is a trade Parth should make, not me.
            Compounding it: the tiles are still UNSOURCED. Bulk download from
            OSM's public tile servers is blocked by their usage policy, so the
            only lawful sources are a self-render from OSM data or a provider
            whose terms permit offline caching. Until that is settled there is
            nothing to commit even if the size were acceptable.

---

## D-048 · Mocks · The drift smoke test was removed from startup
Chose:      Deleted the runDriftSmoke() call from main.tsx. src/mocks/smoke.ts
            is left on disk, unimported.
Rejected:   Leaving it running; and deleting the file.
Confidence: high
Tag:        REVERSIBLE
Reason:     Its own comment said "TEMPORARY: remove with src/mocks/smoke.ts once
            System Status renders driftSummary() for real", and that screen now
            does. It had to go rather than merely could: it fed the DRIFTED
            fixtures through the adapters at boot, so a fresh load already read
            is_feasible 2, plate_normalized 1, disclaimer 1, source_mode 1,
            to_time 1 before any journey was opened. A status screen reporting
            that as observed API drift would have been reporting its own
            diagnostic back to itself -- a fabricated system claim on the one
            screen whose entire job is not making them. The zero state could
            never have been seen either.
            Worth recording precisely because those five numbers are the ones
            the session brief expected the JOURNEY to produce. They were the
            smoke test's, not the journey's. Opening
            /journey?plate=GJ01AB1234 -- the CANONICAL fixture -- actually
            produces no fallbacks and to_time discards only.
            The file is left on disk under the standing rule against deleting a
            file the session did not create. It is dead code and safe to remove.

---

## D-049 · Status · Benchmark types live in ui.ts, and the shape is inferred
Chose:      BenchmarkReport, BenchmarkBucket, BenchmarkBucketKey and
            BENCHMARK_BUCKET_ORDER declared in src/types/ui.ts. readBenchmark in
            adapters.ts tolerates by_plate_width and by_width_bucket, counting
            the latter as a fallback.
Rejected:   Adding them to src/types/api.ts.
Confidence: high on the placement; LOW on the shape
Tag:        NEEDS-PARTH
Reason:     Same terms as the watchlist types in D-039: no Benchmark type exists
            in Canonical 6.5, and api.ts is a mirror that must not be extended
            on our judgement.
            The SHAPE is inferred from the session brief's description, not from
            a response body anyone has seen. eligible_events, correct_events,
            failure_buckets and manifest_sha256 are all guesses at names. That is
            the first thing to check against the real API, and the reason
            readBenchmark discards rather than fabricates when run_id or the
            event counts are missing.
            manifest_sha256 is nullable on purpose. A report without the manifest
            hash cannot be reproduced, so it is not evidence -- but that is a
            fact to display, not a reason to drop the record or hide the panel.
            The screen says "Absent -- this run cannot be reproduced and is not
            evidence" and renders everything else.

---

## D-050 · Mocks · src/mocks/smoke.ts is deleted
Chose:      Deleted the file. Nothing in src/ imported it; only comments named it.
Rejected:   Leaving it unimported on disk, which is what D-048 did.
Confidence: high (resolved by Parth)
Tag:        RESOLVED
Reason:     D-048 removed the call and left the file, under the standing rule
            against deleting a file the session did not create. That was the
            wrong balance here. This is a file that WROTE INTO A PRODUCTION
            COUNTER: it fed drifted fixtures through the adapters at boot, and
            the System Status screen presents those counters as observed API
            drift. It is also precisely the file someone re-imports during
            integration to "check the adapters quickly", at which point the
            status screen starts reporting the diagnostic back to itself again.
            Guard file count 42 -> 41 on deletion, then 42 again once
            src/lib/appMode.ts landed. Reported as measured rather than
            reconciled to an expectation.

---

## D-051 · Status · Liveness polls in the background; nothing else does
Chose:      refetchIntervalInBackground: true on the /health/live and
            /health/ready queries only.
Rejected:   Adding it to any other query; changing refetchOnWindowFocus.
Confidence: high
Tag:        REVERSIBLE
Reason:     TanStack pauses refetchInterval whenever the document is hidden, so
            a minimised or backgrounded projector tab stopped polling and left
            the last "Passing" frozen on screen. That is the same category of
            lie as the dead-port onopen in D-026: a UI asserting health it is no
            longer measuring. Liveness is the one thing that must keep checking
            when nobody is watching it.
            Everything else stays foreground-only on purpose. refetchOnWindowFocus
            remains false everywhere, so alt-tabbing back mid-demo cannot fire a
            refetch storm across six screens, and no other query gains a
            background timer -- the socket already covers live data, and polling
            it as well would be two systems disagreeing about the same facts.

---

## D-052 · Status · D-045 resolved by labelling, not by deduping
Chose:      Keep the counter as-is. The heading reads "Discards -- a record was
            unusable and dropped on read" and the section states that it counts
            adapter reads, not distinct records.
Rejected:   Deduplicating the counter by record identity.
Confidence: high (resolved by Parth)
Tag:        RESOLVED
Reason:     Deduping means a pure read function carries identity state across
            calls -- driftSummary and the read* family stop being pure, and every
            call site inherits a cache whose lifetime nobody has specified. That
            is a larger change than the problem justifies.
            And the problem largely disappears against the real API: each
            response is read once. The inflation measured in D-045 came from
            React StrictMode double-invoking in development plus re-opening the
            same journey, neither of which describes a production read. Naming
            what the number counts costs one sentence and is true in both worlds.

---

## D-053 · Contract · The conflict harness, and what it found
Chose:      VITE_MOCK_SHAPE=hostile serves the execution manuals' spelling for
            every unresolved conflict at once. The drifted spellings live only in
            src/mocks/fixtures/drifted.ts, the guard's noncanon exemption.
Rejected:   Testing the conflicts one at a time, or reasoning about them.
Confidence: high -- every row below was measured by execution
Tag:        REVERSIBLE

            THREE CRASHES, all white-screening the ENTIRE app, all fixed:
              1. health_state instead of status -> STATUS[undefined].text threw
                 in CameraStatusChip. Now falls back to "Unknown".
              2. lat/lon ABSENT (not null) -> hasCoordinates tested `!== null`,
                 so undefined passed and reached L.latLng(undefined, undefined),
                 which throws during render. Now tests Number.isFinite.
              3. severity instead of priority -> PRIORITY[undefined].bar threw in
                 the alert row. Now falls back to "Unknown priority".
            Each was ONE renamed field in ONE endpoint taking down every screen,
            because an unguarded render error unmounts the whole React tree. The
            lesson is not the three fixes; it is that a lookup table indexed by a
            wire value is a crash waiting for a rename, and there may be more.

            THE STRUCTURAL FINDING: only journey, its segments and the benchmark
            go through an adapter. Cameras, alerts, search and system status call
            apiGet directly with a type assertion, so NOTHING absorbs a rename on
            those four. readSighting exists and is exported, but is used only
            inside readJourney -- search results never pass through it, so the
            plate fallback that works on a journey does not work on a search.

            An asymmetry worth one line of Mihir's time: readSighting falls back
            from plate_normalized to plate, but readJourney does NOT. The hostile
            journey was discarded at the envelope on `plate` before its sightings
            were ever read.

            THE TABLE (verbatim, as handed over):

            | Conflict | What happens today | Verdict |
            |---|---|---|
            | plate_normalized (journey sightings) | readSighting falls back, counts a fallback | SAFE |
            | plate_normalized (journey envelope) | readJourney has no fallback; whole journey discarded, screen reads "Journey data incomplete" | BREAKING |
            | plate_normalized (search results) | no adapter; plate cell renders EMPTY, not "Unreadable" | DEGRADED |
            | plate_normalized (alerts) | no adapter; plate renders empty | DEGRADED |
            | is_feasible | readSegment falls back, counts a fallback | SAFE |
            | severity | no adapter; CRASHED the app, now "Unknown priority" | DEGRADED (was BREAKING) |
            | acknowledged_at | no adapter; acknowledged is undefined so an ALREADY-ACKNOWLEDGED alert shows an Acknowledge button | DEGRADED, and the most dangerous row here |
            | health_state | no adapter; CRASHED the app, now every camera "Unknown" | DEGRADED (was BREAKING) |
            | external_camera_id (cameras) | camera_id undefined; ids absent, React key warnings | DEGRADED |
            | external_camera_id (journey sightings) | readSighting discards the sighting | BREAKING |
            | lat/lon absent vs null | CRASHED Leaflet; now counted as unplaced and listed | SAFE |
            | WS envelope payload vs data | wsClient reads data ?? payload and logs which arrived | SAFE |
            | journey as query param | canonical path 404s; a real 404 renders "No results -- nothing matched this request", blaming the PLATE rather than the path | BREAKING |
            | /stats/system vs /system/status | Service health and Source degrade; the other six sections keep working | BREAKING for those two sections |

            The four DEGRADED rows all share one shape: data silently missing
            rather than wrong. acknowledged_at is the one to settle first, because
            an operator re-acknowledging an alert somebody already handled is a
            real-world error, not a cosmetic one.

---

## D-054 · Modes · api mode runs without MSW; auto has no defined rule
Chose:      src/lib/appMode.ts owns the four values. usesMockLayer() answers the
            one question the startup path asks. `auto` resolves to `mock` and
            logs that the rule is undefined.
Rejected:   Inventing a resolution for `auto`.
Confidence: high
Tag:        NEEDS-PARTH (for the auto rule only)
Reason:     Measured: mode=api logs "mock layer OFF" and no MSW line follows it,
            and with nothing on port 8000 every one of the six screens settles to
            "Cannot reach the server / Check that the API is running" in 5.7 to
            9.7 seconds. No screen hangs on a skeleton.
            One caveat that matters for the venue: those errors only appear when
            the tab is FOCUSED. Hidden, retries pause and /map sat on "Loading
            sightings..." for the full 30s budget. The pause is TanStack's
            documented behaviour and D-051 fixes it only for liveness.
            `auto` is declared in Canonical 8.1 with no resolution rule anywhere
            in the contract or this repo. Resolving it to mock would fake data
            against a real backend; resolving it to api would break an offline
            demo. Both fail silently, so it resolves to mock and SAYS SO in the
            console. One constant changes when Mihir settles it.
            Render stays unconditional: worker.start() rejects outside a secure
            context, and http://192.168.x.x on the venue LAN is not one, so an
            MSW-dependent mode must never gate the shell.

---

## D-055 · RISK · public/tiles/ remains the outstanding G6 blocker
Chose:      Restated, not fixed. Nothing attempted this session.
Rejected:   Committing tiles, or sourcing them under time pressure.
Confidence: high
Tag:        NEEDS-PARTH
Reason:     Unchanged from D-047 and still open. public/tiles/ is gitignored, so
            a cold clone on any machine but Parth's has NO offline basemap and
            VITE_BASEMAP=offline renders an empty grid. The tiles are also still
            UNSOURCED: bulk download from OSM's public tile servers is blocked by
            their usage policy, so the lawful options remain a self-render from
            OSM data or a provider whose terms permit offline caching.
            Restated here because it is the one item that blocks G6 on hardware
            other than the machine it was built on, and it has now survived two
            sessions without an owner.

---

## D-056 · Contract · The absent-vs-null rule, and the full audit
Chose:      Every place that branches on a wire field being null now treats
            ABSENT identically. Signatures widened to `T | null | undefined`
            where the wire can omit the key.
Rejected:   Trusting the type. Canonical types are honest about null and silent
            about absent, and the execution manuals omit keys.
Confidence: high
Tag:        REVERSIBLE
Reason:     The lat/lon crash was the general case, not a one-off: the contract
            says `number | null`, the code tested `!== null`, the manual omitted
            the key, and `undefined !== null` is true.

            THE FULL AUDIT. Every null test in src/ was read. Most are SAFE and
            stay as they are, for one of three reasons:
              - inside adapters.ts they test the RESULT of readString/readNumber,
                which already collapse absent to null. That is the design, and
                it is why the adapters were never the crash site.
              - `??` catches absent and null alike (MapCanvas plate, Journey
                node plate, plate_width_px, every `data ?? []`).
              - they test internal state, not wire data (socket refs, timers,
                query results, router params).

            FOUR were genuinely wrong, all at the RENDER layer, all fixed:
              src/lib/time.ts:17   formatRelativeTime(iso === null)
              src/lib/time.ts:46   formatAbsoluteTime(iso === null)
                An absent timestamp fell through to `new Date(undefined)` and
                rendered "Time unreadable" -- which says the clock is broken
                when the truth is the camera has never been seen. Two different
                facts, one wrong label. Not a crash, which is why it survived
                this long.
              src/pages/Cameras.tsx:33  camera.last_seen_at === null
                Same conflation, one layer up.
              src/pages/Journey.tsx:111 segment.note !== null
                `textContent = undefined` writes the literal string "undefined"
                into the timeline. An officer would read that as data.
              src/pages/Search.tsx:37   sighting.plate === null
                The blank plate cell. See D-058.

            hasCoordinates was NOT changed again. Absent and null both mean not
            placeable, both are counted as unplaced, neither is ever plotted.

---

## D-057 · API · readCamera, readAlert, readSearchResponse
Chose:      Three adapters in src/api/adapters.ts, following readSighting's
            pattern. All seven call sites rewired, INCLUDING the WebSocket path.
Rejected:   Adapting only the REST paths.
Confidence: high
Tag:        REVERSIBLE
Reason:     Before this, only journey, its segments and the benchmark had
            adapters; cameras, alerts, search and system status used a bare type
            assertion, and last session two of those assertions became white
            screens.

            The socket matters as much as REST. wsClient did
            `addAlert(body as unknown as Alert)`, and the socket feed MERGES with
            the REST feed into one list -- so without this a rename would have
            been absorbed or not depending on which way the alert arrived, in the
            same list, on the same screen. It now goes through readAlert too.

            WHY A STATUS OUTSIDE THE UNION IS "unknown" BUT A MISSING alert_id
            IS A DISCARD. The test is not severity, it is whether the record can
            still do its job.
              A camera whose status we cannot classify is still a real camera. It
              is in the catalogue, historical sightings reference it, and it has
              a name and an id. "unknown" is a real member of CameraStatus and
              means precisely "we do not know" -- so the record survives with one
              honest field. Dropping it would lose a camera to a vocabulary
              disagreement, and the map would silently stop showing a site that
              exists.
              An alert with no alert_id cannot be deduped, cannot be
              acknowledged, and duplicates on every refetch. There is no honest
              value to substitute: any id we invent is one the server will never
              recognise when the acknowledge POST goes back. The record cannot do
              its job, so it is dropped and counted.
              Same test for match_state: Canonical 5.1 backs the permitted pair
              with a database CHECK, so a value outside it is a row the database
              says cannot exist. That is a discard, not a new render path.

            Alert needed two widenings the canonical mirror cannot express, done
            with the AdaptedJourneySegment technique rather than touching api.ts:
            `priority?` (absent means no assessment was made, and no colour is
            chosen by default) and `plate: string | null` (Alert.plate is a bare
            `string`, which cannot say "unreadable").

            Search runs every result and candidate through the EXISTING
            readSighting, so one field with one rename cannot have two outcomes
            depending on which screen asked. The server's `count` is carried
            verbatim and never reconciled with results.length; discards are
            counted separately and shown, because a quietly shorter list asserts
            fewer sightings than were sent. Candidates arriving on a non-fuzzy
            search are ignored entirely rather than rendered as an empty region,
            which would imply fuzzy matching ran and found nothing.

---

## D-058 · Search · The blank plate cell, and the third surface
Chose:      Absent plate takes the same path as null on all three surfaces:
            search rows, alert rows and map popups. All render "Unreadable".
Rejected:   Leaving alerts to render `{alert.plate}` bare.
Confidence: high
Tag:        REVERSIBLE
Reason:     CONVENTIONS.md rule 6 is "never a guess, never a blank cell", and a blank
            is what an absent plate produced. The search fix also made the
            fallback chain terminate in 'Unreadable' rather than possibly
            undefined -- a blank cell should be impossible by construction, not
            by argument about which branch can be reached.
            The alert row was the third surface and had the same hole; it now
            renders "Unreadable" in the dimmed style the search rows use.

---

## D-059 · Layout · The error boundary sits INSIDE the layout
Chose:      One ScreenErrorBoundary wrapping <Outlet /> inside AppLayout, keyed
            on the router's pathname. No dependency added.
Rejected:   A boundary around the whole app, or around <App /> in main.tsx.
Confidence: high
Tag:        REVERSIBLE
Reason:     An unguarded render error unmounts the WHOLE React tree -- that is
            how three separate field renames each produced a blank projector. A
            boundary at the root would catch the crash and still take the
            StatusBar and the left rail down with it, which is the worst
            outcome: the badge, the clock and the navigation are exactly what a
            presenter needs when one screen has failed. Inside the layout, a
            crashed screen is a broken panel under a working shell, and the demo
            has a line for that.
            Keyed on useLocation().pathname, not window.location.pathname: the
            global does not change on a client-side navigation, so keying on it
            would leave the operator stuck on a crashed screen after navigating
            away. Caught during review, before it was measured.
            It is IN ADDITION to the guarded lookups, never instead of them.
            Anything reaching this boundary is a bug to go and find.

---

## D-060 · Contract · The updated handover table
Tag:        REFERENCE

            | Conflict | What happens now | Verdict |
            |---|---|---|
            | plate_normalized (journey sightings) | readSighting falls back | SAFE |
            | plate_normalized (journey envelope) | readJourney still has no fallback; journey discarded, "Journey data incomplete" | BREAKING |
            | plate_normalized (search results) | readSearchResponse runs readSighting; fallback applies | SAFE |
            | plate_normalized (alerts, REST and socket) | readAlert falls back | SAFE |
            | is_feasible | readSegment falls back | SAFE |
            | severity | readAlert falls back; priorities render correctly | SAFE |
            | acknowledged_at | readAlert derives the boolean: null = not acknowledged, non-null = acknowledged. Verified both | SAFE |
            | health_state (key) | readCamera falls back | SAFE |
            | health_state (VALUES) | "healthy"/"down" are outside CameraStatus; render "Unknown", counted | DEGRADED, and NEW -- the key rename and the vocabulary are two separate disagreements |
            | external_camera_id (cameras) | readCamera falls back; ids resolve | SAFE |
            | external_camera_id (alerts) | readAlert falls back | SAFE |
            | external_camera_id (journey/search sightings) | readSighting has NO fallback; sighting discarded and counted, surfaced on screen | DEGRADED |
            | lat/lon absent vs null | identical; counted unplaced, never plotted | SAFE |
            | WS envelope payload vs data | read and logged | SAFE |
            | journey as query param | canonical path 404s; renders "No results... check the plate", blaming the plate not the path | BREAKING |
            | /stats/system vs /system/status | Service health and Source degrade; other six sections keep working | BREAKING |

            Eight rows moved to SAFE. What remains needs Mihir, not an adapter:
            the two ENDPOINT conflicts cannot be absorbed because a path is not
            a field, and readJourney's envelope plate plus readSighting's
            camera_id are two one-line fallbacks deliberately NOT added this
            session because they were not in scope -- both are visible, counted
            and surfaced rather than silent.

---

## D-061 · RISK · public/tiles/ is still the outstanding G6 blocker
Chose:      Restated. Nothing attempted; not in scope this session.
Confidence: high
Tag:        NEEDS-PARTH
Reason:     Unchanged from D-047 and D-055, now surviving a third session
            without an owner. public/tiles/ is gitignored, so a cold clone on
            any machine but Parth's has NO offline basemap and
            VITE_BASEMAP=offline renders an empty grid. The tiles are also still
            UNSOURCED: bulk download from OSM's public tile servers is blocked by
            their usage policy, leaving a self-render from OSM data or a provider
            whose terms permit offline caching.

---

## D-062 · Realtime · The WS `payload` tolerance is a DELIBERATE exception — do not remove it for consistency
Chose:      wsClient keeps reading `envelope.data ?? envelope.payload` and
            logging which key arrived, even though `payload` appears NOWHERE in
            the canonical contract.
Rejected:   Removing it alongside the other six contract-less tolerances.
Confidence: high (resolved by Parth)
Tag:        CONFIRMED
Reason:     Six tolerances were removed in the same session precisely because
            the contract does not use those names (D-063). This one is kept,
            and the difference is not the evidence -- it is the CONSEQUENCE OF
            BEING WRONG, and the asymmetry is severe enough to override the
            rule that removed the others.
            Every removed fallback sat on a REST path. If the backend surprises
            us there, the record still arrives: a field degrades, the adapter
            counts it, the screen shows "Unknown priority" or "not assessed",
            and the drift panel names the spelling. The failure is visible and
            partial.
            The socket has NO such floor. `handleMessage` requires a usable
            body; a frame whose payload it cannot find is counted as malformed
            and DROPPED. If Mihir's socket emits `payload` and we removed the
            tolerance, every alert frame would be silently discarded mid-demo,
            with a live-looking status line and an empty list -- the exact
            class of lie this app is built against. There is no REST refetch
            behind an individual frame to recover it.
            The cost of keeping it is one `if` and one console line. The cost
            of being wrong without it is the alerts screen going quiet on
            stage while claiming to be connected.
            So: do not delete this for tidiness. If a future session removes
            contract-less tolerances again, this one is exempt by decision,
            and the reason is the missing fallback path, not the field name.

---

## D-063 · Contract · Six tolerances removed; two kept as database-column leaks
Chose:      Removed the is_feasible, health_state, severity, acknowledged_at,
            coords_placeholder and by_width_bucket fallbacks, and the four of
            those that were in the guard's noncanon ban list. Kept
            plate_normalized and external_camera_id.
Rejected:   Keeping all eight "just in case".
Confidence: high (resolved by Parth)
Tag:        RESOLVED
Reason:     The canonical contract arrived and settled it. The four removed
            guard names and six removed fallbacks appear NOWHERE in it -- not
            on the wire, not as a database column. They came from execution
            manuals the contract supersedes, and Parth's own note records the
            origin: manual-vs-canonical disagreements were being treated as
            wire-format risk.
            The surviving two are different in kind. `plate_normalized` is the
            vehicle_sightings column (5.5) whose wire field is `plate` (6.5);
            `external_camera_id` is the cameras column (5.1), immutable once
            seeded, whose wire field is `camera_id` (6.4). Both are real names
            for real things one layer down, and an ORM serializing a row
            straight to JSON emits the column name. That is a specific,
            predictable failure rather than a hypothetical.
            The test that separates them is not "is this name plausible" but
            "does the contract describe a mechanism by which it reaches us".
            Two do. Six do not.
            The guard's noncanon message now names what the rule actually
            catches -- a serializer leaking the schema -- rather than the
            generic "non-canonical field name" it claimed before.

---

## D-064 · Metrics · The benchmark panel is built against 7.3, with 7.2 enforced by ORDER
Chose:      readBenchmark and the panel read Canonical 7.3 exactly:
            `dataset_manifest_sha256`, `e2e_correct_plate_event_rate`, and
            `by_plate_width` as a SCALAR RATE per bucket. Dropped
            eligible_events, correct_events and failure_buckets.
Rejected:   Tolerating both shapes; waiting for Akshat to emit counts.
Confidence: high (resolved by Parth)
Tag:        RESOLVED
Reason:     The old shape was inferred from an execution manual and was wrong
            in a way no fallback could repair: it required per-bucket
            {eligible_events, correct_events}, and 7.3 carries one rate per
            bucket and no event counts anywhere. Measured before changing
            anything -- a strictly 7.3-conforming payload returned null,
            recorded a discard keyed "eligible_events", and the panel rendered
            "The benchmark report could not be read." Not empty. Not partial.
            A refusal, on the strongest screen in the demo.
            Tolerating both was rejected for the reason D-063 exists: it
            re-adds the habit that session removed, and it would mean two
            render paths on that screen, one of them unrehearsable until
            Akshat's output exists.
            THE 7.2 PROBLEM AND ITS RESOLUTION. 7.2 says "No accuracy number
            may be reported as a single average", and 7.3 none the less ships
            exactly one -- e2e_correct_plate_event_rate. Conforming to 7.3 by
            rendering that scalar as the headline would violate 7.2. The
            resolution is ORDERING, not omission: the six buckets are rendered
            first with bar plus number, and the mean appears BELOW them, small,
            phrased "Mean across all widths: 0.71 -- an average over the six
            buckets above, not a system accuracy figure". Reading order enforces
            7.2 structurally; a reader reaches the collapse before the average,
            and the average cannot be lifted out as "the accuracy number".
            Rates render as the decimals 7.3 sends (0.38), never as "38%".
            No percent sign appears anywhere on that screen -- verified.
            `null` and `0.0` are kept distinct: null is "not run", 0.0 is
            "measured, nothing correct", and 0.0 in the <30 bucket IS the
            finding. Collapsing them would delete the point of the screen.
            A rate outside [0,1] is refused and counted rather than rendered:
            7.1 defines it as a ratio, so 1.4 is not a value this metric can
            produce and drawing it would launder a backend bug into a claim.
            WHAT IS NOT RENDERED, and why: task, git_commit, weights_sha256,
            machine, runtime, source_mode, by_condition, diagnostics, notes.
            All are in 7.3 and all are omitted deliberately. 7.1 is explicit
            that everything except the primary rate is a DIAGNOSTIC, "used to
            explain the primary number, never reported as the headline" -- and
            this panel is an operator's status screen, not a model report.
            fps, VRAM and CER belong in Akshat's leaderboard CSV. run_id and
            dataset_manifest_sha256 ARE rendered, because a run nobody can
            identify or reproduce is not evidence.
            The cost accepted knowingly: "598 of 842" is gone. It was the
            better rhetorical form -- a count answers "out of what?" before it
            is asked -- but 7.3 provides no counts, and inventing them was what
            broke the panel in the first place.
            failure_buckets is gone with it. Its one irreplaceable line, that a
            plate below ~30px was never resolved by the sensor and no model
            change recovers it, now sits beside the <30 bucket as a STATIC
            caption -- a standing engineering finding, not data we pretend the
            API sent.

---

## D-065 · Mocks · drifted.ts narrowed to what the contract makes plausible
Chose:      Kept drifted payloads for plate_normalized and external_camera_id
            only, plus lat/lon ABSENT and one camera status outside 5.1's
            CHECK. Rewrote the header to say what the file now tests.
Rejected:   Deleting the file; leaving the six spellings in place.
Confidence: high (resolved by Parth)
Tag:        RESOLVED
Reason:     After D-063 the file was documentation of a superseded belief: its
            header explained six live conflicts, four of which the contract
            says cannot occur. A harness that exercises impossible inputs
            reports confidence it has not earned, and a smaller honest harness
            is worth more than a large misleading one.
            The status-outside-CHECK case is kept DELIBERATELY and is now
            called out in the file rather than left implicit, because it is
            the one input here the contract forbids: 5.1 constrains status to
            online/offline/degraded/unknown at the database level, so a
            conforming backend cannot emit "healthy". It stays because a CHECK
            constraint protects the DATABASE, not the JSON encoder -- an enum
            widened in code but not in the migration, or a value mapped on the
            way out, reaches us without violating anything the database
            enforced. We degrade it to "unknown" and count it, and proving
            that costs one fixture.
            WHAT THE HARNESS STILL PROVES: column names on the wire are
            absorbed and named in the drift panel; lat/lon absent is counted as
            unplaced rather than crashing Leaflet; an unrecognised camera
            status degrades instead of throwing; a sighting with no source_mode
            is discarded and counted.
            WHAT IT NO LONGER PROVES, and should not be claimed: that the app
            survives severity/acknowledged_at/is_feasible/health_state
            spellings. It does not handle them any more, by decision, and the
            fixtures no longer pretend to test it.

---

## D-066 · RISK · api.ts is hand-mirrored, not generated — an additive field is invisible to us
Chose:      Recorded as a known gap. No codegen.
Rejected:   Building a generator three days before submission.
Confidence: high (resolved by Parth)
Tag:        NEEDS-PARTH
Reason:     Canonical 6.8 states "Parth's types are generated from this
            contract, and a silent rename breaks the build at the worst
            possible moment." They are NOT generated. src/types/api.ts is
            hand-written from 6.5 and was verified field-by-field against the
            real document -- it is currently a perfect mirror, which is care
            plus luck, not a mechanism.
            The consequence is specific and one-directional. 6.8 permits
            ADDITIVE change: "Adding an optional field is fine." A rename or a
            type change would break our build, exactly as 6.8 promises. An
            ADDITION would not: Mihir adds a field, the contract stays valid,
            our types stay valid, and the frontend silently ignores data that
            now exists. Nothing tells us. No test fails, no counter moves --
            the drift counters only fire on names we already look for.
            Codegen is the right long-term answer and the wrong trade this
            week: it is new tooling on the critical path days before
            submission, and it would have caught nothing that has actually
            happened.
            The mitigation is social, and it is Parth's to run: ask Mihir
            directly whether anything has been added since the contract was
            written, and ask again after any backend change. Recorded here so
            the gap is not rediscovered from scratch in a later session and
            mistaken for a new finding.
