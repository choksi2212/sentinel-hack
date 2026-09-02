# Test fixtures

The twelve files named in Canonical Contracts §9, plus `expectations.json`, which
records what each one is supposed to do. Filenames are locked by the contract — if you
need a different case, add a test, not a thirteenth fixture.

They are **contract** fixtures. §9: *"Fixtures validate contracts. They never justify an
accuracy claim."* Every `model.detector_weights_sha256` in here is `null`, deliberately,
so that `ModelProvenance.is_citeable` is false for all of them and no number derived from
this directory can be quoted in a submission. `tests/test_contracts.py` asserts that.

## What each file is

| File | Kind | Expected |
|---|---|---|
| `ai_event_high_confidence.json` | one event | accepted, alert raised |
| `ai_event_low_confidence.json` | one event | accepted, **no** alert |
| `ai_event_unreadable.json` | one event | accepted, `plate: null` is correct |
| `ai_event_duplicate.json` | one event | **duplicate** — POST `high_confidence` first |
| `ai_event_bad_timestamp.json` | one event | 422 `VALIDATION_FAILED` on `observed_at` |
| `ai_event_unknown_camera.json` | one event | 422 `UNKNOWN_CAMERA` on `camera_id` |
| `camera_reconnect.json` | 2 events | 2 rows in `vehicle_tracks`, 2 in `stream_sessions` |
| `scene_discontinuity.json` | 2 events | 2 sessions, first ends `discontinuity` |
| `journey_four_cameras.json` | 4 events + response | 3 segments, all feasible |
| `journey_implausible.json` | 2 events + response | 1 segment, `feasible: false`, both sightings kept |
| `watchlist_match.json` | seed + events | one `exact` alert, and only one |
| `search_response.json` | response shape | not POSTable — the frontend's mock |

Machine-readable form of that table: `expectations.json`. Read it rather than retyping
it, so the AI lane's tests and the backend lane's ingest tests cannot drift apart.

## Three things that look like mistakes and are not

**`ai_event_unknown_camera.json` passes local validation.** `cam99` is a well-formed
Sentinel catalogue ID, and `ai/contracts/ids.py` checks shape only — existence is the
`cameras` table's business. So `validate_payload()` returns no errors and ingest still
has to reject it, with `UNKNOWN_CAMERA` and not `VALIDATION_FAILED`. This is the only
fixture that separates the two, which is why it exists.

**`ai_event_duplicate.json` is byte-identical to `ai_event_high_confidence.json`.** That
is what a retry looks like. The worker builds the envelope once, including its
`event_id`, and re-POSTs the same bytes when a response is lost — see
`ai/emit/http_sink.py`. A duplicate that differed in content would be testing something
that cannot happen.

**`ai_event_unreadable.json` has `plate: null`, not `match_state: "unreadable"`.** Those
are two different states and §9 names this one by its payload, not by its enum value:

- `plate: null` — no plate was located on this vehicle at all.
- `plate: {normalized: null, match_state: "unreadable"}` — a plate *was* located and no
  usable text came out of it.

The second is covered inline in `tests/test_contracts.py`, because §9 fixes this
directory at twelve files and the enum case does not need one of its own.

## Camera catalogue

The lat/lon in the journey and search fixtures are internally consistent and the
distances are haversine on WGS84 with `R = 6371.0088 km`, per §4.5. The real catalogue is
the `cameras` table; these are here so the feasibility arithmetic is reproducible.

| `camera_id` | Name | lat | lon |
|---|---|---|---|
| `cam04` | Paldi Junction | 23.012 | 72.558 |
| `cam07` | Ashram Road Chowk | 23.0295 | 72.5715 |
| `cam14` | Ring Road North | 23.0512 | 72.589 |
| `cam21` | Sabarmati Bridge East | 23.073026 | 72.605273 |
| `cam23` | Naroda Patiya | `null` | `null` |

`cam21` sits where it does because `cam04 → cam21` is then 8.333 km, which over two
seconds is 15,000.0 km/h — the number in the §6.6 error-envelope example.
`cam23` has no coordinates on purpose: §6.5 allows `null` until a camera is surveyed, and
the map has to skip that marker without dropping the list row.

Distances and speeds in the fixtures are the haversine result rounded to 3 and 1 decimal
places respectively, and `test_contracts.py` recomputes every one of them from the lat/lon
in the same file. A straight-line distance is a deliberate under-estimate of a road
distance, so reporting it to fifteen significant figures would be claiming precision the
method does not have.

## Loading them together

The twelve are consistent as a set: no two of them put the same vehicle in two places at
once, and `search_response.json` returns exactly the three `GJ01AB1234` sightings that
`ai_event_high_confidence.json`, `camera_reconnect.json` and `watchlist_match.json` put
in the database. So they can all be loaded into one instance and the search and journey
responses will match these files.

`ai_event_duplicate.json` is the one ordering constraint: POST `ai_event_high_confidence.json`
first, or the duplicate is a new event and the idempotency test proves nothing.

## Known unresolved: which path is ingest?

`docs/TRINETRA_Canonical_Contracts.md` §6.1 and
`docs/manuals/TRINETRA_Mihir_Backend_Data_Platform_Execution_Manual.md` §5.1 disagree,
and have since both were written:

| | Contracts §6.1 | Backend manual §5.1 |
|---|---|---|
| Path | `/api/v1/events/vehicle-sighting` | `/api/v1/ingest/events` |
| New event | `200 {"status":"accepted"}` | `201 {"status":"accepted"}` |

The fixtures are bodies, so they do not care. The worker does: it posts to
`/api/v1/ingest/events`, because that is the path in the document the endpoint is being
implemented from, and `ai/emit/http_sink.py` classifies both `201` and
`200 {"status":"accepted"}` as accepted so either backend works without a code change.
Needs one decision, recorded in Contracts §6.1. Tracked in `expectations.json` under
`ingest_path_disagreement`.
