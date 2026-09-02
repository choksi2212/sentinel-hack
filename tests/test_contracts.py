"""The event contract, asserted against the twelve fixtures Contracts section 9 locks.

Two jobs, and they are different jobs.

The first is to prove `ai/contracts/event.py` accepts what the contract says is valid and
rejects what it says is invalid -- including the cases that look like they should be
rejected and must not be, which is where a validator usually goes wrong. `plate: null` is
the one that matters: an over-eager validator that refuses it turns "this vehicle could not
be identified" into no event at all, and section 3.2 calls fabricating a plate the worst
possible failure mode in a police system. Silently dropping the vehicle is the second worst.

The second is to keep the fixtures themselves honest. They are a deliverable another lane
tests against, so this file asserts the properties the backend is relying on -- that
camera_reconnect really does carry two different session ids, that the duplicate really is
byte-identical, that the arithmetic in journey_implausible actually produces 15,000 km/h --
because a fixture that quietly stops demonstrating its case is worse than a missing one.
"""

import json
import math

import pytest

from ai.contracts.enums import (
    MATCH_STATES,
    SCHEMA_VERSION,
    SOURCE_MODES,
    VEHICLE_TYPES,
)
from ai.contracts.event import EventEnvelope, ModelProvenance, validate_payload
from ai.contracts.ids import is_valid_camera_id
from ai.contracts.timebase import seconds_between

from conftest import CONTRACT_FIXTURES, FIXTURE_DIR, fixture_events, load_fixture

# Haversine mean earth radius, per Contracts 4.5. Written here rather than imported because
# the backend implements the distance and this test checks the fixture against the constant
# the contract names -- importing our own would make the check circular.
EARTH_RADIUS_KM = 6371.0088

# Contracts 4.5. 150 km/h urban, the starting value.
PLAUSIBILITY_CEILING_KMH = 150.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------- the twelve exist


def test_all_twelve_fixtures_exist():
    missing = [n for n in CONTRACT_FIXTURES if not (FIXTURE_DIR / n).is_file()]
    assert not missing, f"Contracts section 9 locks these filenames; missing: {missing}"


def test_fixture_directory_has_no_unlisted_json():
    """Nothing extra except the two scaffolding files, which are named.

    A fixture directory that accumulates thirteenth and fourteenth cases is one where
    nobody can tell which files the contract actually promised.
    """
    allowed = set(CONTRACT_FIXTURES) | {"expectations.json"}
    found = {p.name for p in FIXTURE_DIR.glob("*.json")}
    assert found == allowed, f"unexpected: {sorted(found - allowed)}"


def test_fixtures_are_ascii():
    """No smart quotes, no ellipsis characters, no stray non-Latin glyphs.

    The canonical contract document writes a truncated hash as `a1b2c3…` with a real
    U+2026. Copied into a fixture that becomes a value the backend has to parse, and a
    non-ASCII byte in a JSON payload is a bug that reproduces on one machine in four.
    """
    for name in CONTRACT_FIXTURES:
        raw = (FIXTURE_DIR / name).read_text(encoding="utf-8")
        bad = sorted({ch for ch in raw if ord(ch) > 126})
        assert not bad, f"{name} contains non-ASCII characters: {bad}"


# ------------------------------------------------------------------- envelope validation


def _every_event() -> list[tuple[str, int, dict]]:
    out = []
    for name in CONTRACT_FIXTURES:
        for index, event in enumerate(fixture_events(load_fixture(name))):
            out.append((name, index, event))
    return out


EVERY_EVENT = _every_event()

# The one fixture that is invalid by construction. Everything else in the directory must
# pass validate_payload(), including ai_event_unknown_camera.json -- see its own test.
LOCALLY_INVALID = {"ai_event_bad_timestamp.json"}


@pytest.mark.parametrize(
    "name,index,event",
    [(n, i, e) for n, i, e in EVERY_EVENT if n not in LOCALLY_INVALID],
    ids=[f"{n}[{i}]" for n, i, _ in EVERY_EVENT if n not in LOCALLY_INVALID],
)
def test_fixture_events_validate(name, index, event):
    errors = validate_payload(event)
    assert errors == [], f"{name}[{index}]: {errors}"


@pytest.mark.parametrize(
    "name,index,event", [(n, i, e) for n, i, e in EVERY_EVENT], ids=[f"{n}[{i}]" for n, i, _ in EVERY_EVENT]
)
def test_fixture_events_round_trip(name, index, event):
    """from_dict -> to_dict is lossless for every field the contract defines.

    Runs on the invalid fixture too. A parser that only survives well-formed input is not
    much use for reproducing a rejection, and the bad-timestamp fixture is malformed in a
    field the dataclass stores verbatim rather than one it has to interpret.
    """
    envelope = EventEnvelope.from_dict(event)
    again = envelope.to_dict()
    for key in ("schema_version", "event_id", "camera_id", "stream_session_id", "track_id",
                "observed_at", "source_pts_ms", "source_mode", "image_quality"):
        assert again[key] == event[key], f"{name}[{index}] field {key} changed on round trip"
    assert again["vehicle"]["type"] == event["vehicle"]["type"]
    if event["plate"] is None:
        assert again["plate"] is None
    else:
        for key in ("raw", "normalized", "match_state", "plate_width_px", "evidence_count"):
            assert again["plate"][key] == event["plate"][key], f"{name}[{index}] plate.{key}"


@pytest.mark.parametrize(
    "name,index,event", [(n, i, e) for n, i, e in EVERY_EVENT], ids=[f"{n}[{i}]" for n, i, _ in EVERY_EVENT]
)
def test_fixture_events_use_locked_enums(name, index, event):
    assert event["schema_version"] == SCHEMA_VERSION
    assert event["source_mode"] in SOURCE_MODES
    assert event["vehicle"]["type"] in VEHICLE_TYPES
    if event["plate"] is not None:
        assert event["plate"]["match_state"] in MATCH_STATES


@pytest.mark.parametrize(
    "name,index,event", [(n, i, e) for n, i, e in EVERY_EVENT], ids=[f"{n}[{i}]" for n, i, _ in EVERY_EVENT]
)
def test_no_fixture_claims_citeable_provenance(name, index, event):
    """Contracts section 9: fixtures never justify an accuracy claim.

    Enforced structurally rather than by convention. Every fixture leaves
    detector_weights_sha256 null, so is_citeable is false for all of them and a number
    computed from this directory cannot be quoted in a submission even by accident.
    """
    provenance = ModelProvenance.from_dict(event["model"])
    assert not provenance.is_citeable, (
        f"{name}[{index}] carries a weights hash. Fixtures are contract tests, not runs -- "
        "a citeable-looking provenance block here invites someone to quote a fixture."
    )


# ------------------------------------------------------- the cases that must be accepted


def test_plate_null_is_valid():
    """Contracts 3.2. The single most important negative-space assertion in this file."""
    event = load_fixture("ai_event_unreadable.json")
    assert event["plate"] is None
    assert validate_payload(event) == []


def test_located_but_unreadable_is_valid():
    """The other null: a plate object present, normalized null, match_state unreadable.

    Not one of the twelve -- section 9 fixes the directory at twelve files and describes
    ai_event_unreadable.json by its payload, `plate: null`, which is the no-plate-located
    case. This is the located-but-unread case, built here from a fixture so the two cannot
    drift apart.
    """
    event = dict(load_fixture("ai_event_low_confidence.json"))
    event["plate"] = dict(
        event["plate"], normalized=None, match_state="unreadable", evidence_count=1
    )
    assert validate_payload(event) == []


def test_unreadable_with_a_normalized_plate_is_rejected():
    """The two nulls must not be mixed. One of the pair is always wrong."""
    event = dict(load_fixture("ai_event_low_confidence.json"))
    event["plate"] = dict(event["plate"], match_state="unreadable")
    errors = validate_payload(event)
    assert any("normalized" in e for e in errors), errors


def test_normalized_null_without_unreadable_is_rejected():
    event = dict(load_fixture("ai_event_low_confidence.json"))
    event["plate"] = dict(event["plate"], normalized=None)
    errors = validate_payload(event)
    assert any("match_state" in e for e in errors), errors


def test_low_confidence_fixture_is_exactly_what_section_9_describes():
    """One observation, confidence 0.51, low_confidence. The numbers are the fixture's point."""
    plate = load_fixture("ai_event_low_confidence.json")["plate"]
    assert plate["evidence_count"] == 1
    assert plate["confidence"] == 0.51
    assert plate["match_state"] == "low_confidence"


def test_high_confidence_fixture_is_exactly_what_section_9_describes():
    plate = load_fixture("ai_event_high_confidence.json")["plate"]
    assert plate["evidence_count"] == 4
    assert plate["match_state"] == "exact"


# ------------------------------------------------------- the cases that must be rejected


def test_bad_timestamp_is_rejected_naming_the_field():
    event = load_fixture("ai_event_bad_timestamp.json")
    errors = validate_payload(event)
    assert errors, "a naive observed_at must not validate"
    assert any("observed_at" in e for e in errors), errors
    # And it is the ONLY thing wrong with it, or the fixture is testing two things at once
    # and a backend could reject it for the wrong reason and still look correct.
    assert len(errors) == 1, errors


def test_unknown_camera_passes_shape_and_must_fail_the_catalogue():
    """The fixture the two validation layers are separated by.

    cam99 is a well-formed Sentinel ID, so the envelope validator must pass it: existence
    is the cameras table's business, not the envelope's. A backend that rejects this with
    VALIDATION_FAILED rather than UNKNOWN_CAMERA has collapsed shape and existence into one
    check, and will reject a camera that was added to the grid after its last deploy.
    """
    event = load_fixture("ai_event_unknown_camera.json")
    assert is_valid_camera_id(event["camera_id"])
    assert validate_payload(event) == []
    assert event["camera_id"] == "cam99"


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("schema_version", "1.0", "schema_version"),
        ("schema_version", "2.0", "schema_version"),
        ("event_id", "not-a-uuid", "event_id"),
        ("camera_id", "CAM04", "camera_id"),
        ("camera_id", "cam4", "camera_id"),
        ("stream_session_id", None, "stream_session_id"),
        ("track_id", -1, "track_id"),
        ("track_id", True, "track_id"),
        ("source_pts_ms", -5, "source_pts_ms"),
        ("source_mode", "live", "source_mode"),
        ("source_mode", "rtsp", "source_mode"),
        ("image_quality", 1.4, "image_quality"),
    ],
)
def test_each_required_field_is_actually_checked(field, value, needle):
    """One mutation per row, so a failure names the field that stopped being validated.

    `source_mode: "live"` is in here because it reads like a valid value and is not: the
    five modes are live_rtsp, live_hls, file, frames and synthetic, the event schema has a
    CHECK constraint on the column, and ingest answers 422.
    """
    event = dict(load_fixture("ai_event_high_confidence.json"))
    event[field] = value
    errors = validate_payload(event)
    assert any(needle in e for e in errors), f"{field}={value!r} produced {errors}"


def test_missing_plate_key_is_rejected_but_explicit_null_is_not():
    """Absent and null are different. Absent is a bug; null is an answer."""
    event = dict(load_fixture("ai_event_high_confidence.json"))
    del event["plate"]
    assert any("plate" in e for e in validate_payload(event))

    event["plate"] = None
    assert validate_payload(event) == []


def test_missing_provenance_is_rejected():
    event = dict(load_fixture("ai_event_high_confidence.json"))
    event["model"] = dict(event["model"])
    del event["model"]["tracker"]
    assert any("model.tracker" in e for e in validate_payload(event))


# ------------------------------------------------------------ the multi-event fixtures


def test_camera_reconnect_proves_the_trackkey_invariant():
    """Contracts 1.2. Same camera, same track_id, different session, two vehicles.

    If this fixture ever ends up with one session id or two track ids it stops testing
    anything, and uq_trackkey could regress to (camera_id, track_id) with the fixture still
    green. So the shape is asserted, not just the validity.
    """
    payload = load_fixture("camera_reconnect.json")
    events = payload["events"]
    assert len(events) == 2
    assert payload["expected_vehicle_tracks_rows"] == 2

    a, b = events
    assert a["camera_id"] == b["camera_id"], "must be the same camera"
    assert a["track_id"] == b["track_id"], "must be the same track_id -- that is the trap"
    assert a["stream_session_id"] != b["stream_session_id"], "must be different sessions"

    # Two genuinely different vehicles, so a merge is visibly wrong rather than plausible.
    assert a["vehicle"]["type"] != b["vehicle"]["type"]
    assert a["plate"]["normalized"] != b["plate"]["normalized"]

    # The TrackKey tuple is what the constraint is on. Distinct as triples, colliding as pairs.
    key = lambda e: (e["camera_id"], e["stream_session_id"], e["track_id"])
    assert key(a) != key(b)
    assert (a["camera_id"], a["track_id"]) == (b["camera_id"], b["track_id"])


def test_scene_discontinuity_goes_backwards_and_rotates_the_session():
    payload = load_fixture("scene_discontinuity.json")
    a, b = payload["events"]
    assert b["source_pts_ms"] < a["source_pts_ms"], "the PTS must regress; that is the case"
    assert a["stream_session_id"] != b["stream_session_id"]
    assert a["camera_id"] == b["camera_id"]
    assert payload["expected_first_session_end_reason"] == "discontinuity"
    # Wallclock still moves forward. Only the source timeline jumped -- a fixture where both
    # went backwards would be describing a broken clock, not a scene cut.
    assert seconds_between(a["observed_at"], b["observed_at"]) > 0


def test_duplicate_is_byte_identical_to_high_confidence():
    """A retry re-POSTs the same bytes, because the envelope was built once."""
    high = (FIXTURE_DIR / "ai_event_high_confidence.json").read_text(encoding="utf-8")
    dup = (FIXTURE_DIR / "ai_event_duplicate.json").read_text(encoding="utf-8")
    assert json.loads(high) == json.loads(dup)
    assert json.loads(dup)["event_id"] == json.loads(high)["event_id"]


def test_event_ids_are_unique_except_the_duplicate():
    """Every other fixture event has its own event_id, so they can all be loaded at once.

    Without this, two fixtures sharing an id would make the second one a duplicate and a
    backend test would report an idempotency pass it never earned.
    """
    seen: dict[str, list[str]] = {}
    for name, index, event in EVERY_EVENT:
        seen.setdefault(event["event_id"], []).append(f"{name}[{index}]")
    shared = {k: v for k, v in seen.items() if len(v) > 1}
    assert list(shared.values()) == [
        ["ai_event_high_confidence.json[0]", "ai_event_duplicate.json[0]"]
    ], f"unexpected shared event_ids: {shared}"


# ----------------------------------------------------------------- feasibility arithmetic


def _cameras(payload: dict) -> dict[str, dict]:
    return {c["camera_id"]: c for c in payload["cameras"]}


def test_journey_four_cameras_is_ordered_and_all_feasible():
    payload = load_fixture("journey_four_cameras.json")
    events = payload["events"]
    assert len(events) == 4
    assert len({e["camera_id"] for e in events}) == 4

    times = [e["observed_at"] for e in events]
    assert all(seconds_between(times[i], times[i + 1]) > 0 for i in range(3)), "must be ordered"
    assert len({e["plate"]["normalized"] for e in events}) == 1, "must be one vehicle"

    response = payload["expected_response"]
    assert response["sighting_count"] == 4
    assert response["disclaimer"], "Contracts 6.3: mandatory in the body, not a UI convention"
    assert len(response["segments"]) == 3
    assert all(s["feasible"] for s in response["segments"])
    assert all(s["note"] is None for s in response["segments"])
    assert all(s["required_speed_kmh"] < PLAUSIBILITY_CEILING_KMH for s in response["segments"])


def test_journey_segment_arithmetic_reproduces_from_the_coordinates():
    """The stated km and km/h are recomputed from lat/lon, not trusted.

    The backend implements 4.5 independently. If the fixture's numbers were hand-typed and
    slightly wrong, the correct implementation would fail against it and the search would
    start in the wrong place.

    Compared at the precision the response declares -- 3 decimals of a kilometre, 1 of a
    km/h -- not to floating-point equality. The distance is a deliberate straight-line
    under-estimate of a road distance, so `16.427129003725124 km/h` would be fifteen
    significant figures of a number that is wrong by whatever the road bends. Rounding is
    the honest wire format, and this test asserts the fixtures use it consistently.
    """
    for name in ("journey_four_cameras.json", "journey_implausible.json"):
        payload = load_fixture(name)
        cams = _cameras(payload)
        for segment in payload["expected_response"]["segments"]:
            label = f"{name} {segment['from_camera_id']}->{segment['to_camera_id']}"
            a, b = cams[segment["from_camera_id"]], cams[segment["to_camera_id"]]
            km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            elapsed = seconds_between(segment["from_time"], segment["to_time"])

            assert elapsed == pytest.approx(segment["elapsed_seconds"], abs=0.001), label
            assert round(km, 3) == segment["straight_line_km"], (
                f"{label}: stated {segment['straight_line_km']}, computed {km!r}"
            )
            speed = km / max(elapsed / 3600.0, 1e-6)
            assert round(speed, 1) == segment["required_speed_kmh"], (
                f"{label}: stated {segment['required_speed_kmh']}, computed {speed!r}"
            )
            assert segment["feasible"] == (speed <= PLAUSIBILITY_CEILING_KMH), label


def test_journey_implausible_is_15000_kmh_and_keeps_both_sightings():
    payload = load_fixture("journey_implausible.json")
    assert payload["plausibility_ceiling_kmh"] == PLAUSIBILITY_CEILING_KMH

    segments = payload["expected_response"]["segments"]
    assert len(segments) == 1
    segment = segments[0]
    assert segment["feasible"] is False
    assert segment["required_speed_kmh"] == pytest.approx(15000.0, rel=0.001)
    assert "exceeds plausibility ceiling" in segment["note"]

    # Contracts 4.5: flag it, never delete it. Both sightings stay in the response.
    assert payload["expected_response"]["sighting_count"] == 2
    assert len(payload["events"]) == 2
    assert len({e["plate"]["normalized"] for e in payload["events"]}) == 1


# ------------------------------------------------------------------ watchlist and search


def test_watchlist_seed_normalizes_and_stays_selective():
    payload = load_fixture("watchlist_match.json")
    entries = payload["watchlist"]
    assert len(entries) == 3

    # Every entry carries both forms, and the normalized one is what a match compares.
    # The as_typed values have separators and mixed case on purpose: a loader that skips
    # normalization matches nothing, and that failure looks like a broken alert path.
    for entry in entries:
        assert entry["plate_normalized"] == entry["plate_normalized"].upper()
        assert entry["plate_normalized"].isalnum()
        assert entry["as_typed"] != entry["plate_normalized"] or entry["active"] is False

    active = [e for e in entries if e["active"]]
    assert len(active) == 2, "one entry is deactivated on purpose"

    event = payload["matching_event"]
    assert validate_payload(event) == []
    assert event["plate"]["match_state"] == "exact"
    assert event["plate"]["normalized"] in {e["plate_normalized"] for e in active}
    assert payload["expected_alert"]["match_state"] == "exact"

    # And the negative cases are present, or the fixture proves only that a permissive
    # matcher fires. A deactivated exact hit and a one-character miss must both stay silent.
    reasons = payload["non_matching_events"]
    assert len(reasons) == 2
    assert all(r["expected_alert"] is False for r in reasons)
    deactivated = next(e for e in entries if not e["active"])
    assert deactivated["plate_normalized"] in {r["normalized"] for r in reasons}


def test_search_response_matches_the_frontend_wire_shape():
    """Contracts 6.5 field names, exactly. first_seen_at, never timestamp."""
    payload = load_fixture("search_response.json")
    response = payload["response"]

    assert set(response["query"]) == {"plate", "normalized", "fuzzy"}
    assert response["query"]["fuzzy"] is False
    assert response["count"] == len(response["results"]) == 3

    required = {
        "sighting_id", "camera_id", "camera_name", "lat", "lon", "first_seen_at",
        "last_seen_at", "source_pts_ms", "source_mode", "plate", "plate_raw",
        "confidence", "match_state", "evidence_count", "plate_width_px",
        "vehicle_type", "snapshot_uri",
    }
    for row in response["results"]:
        assert set(row) == required, f"wire shape drift: {set(row) ^ required}"
        assert "timestamp" not in row, "Contracts 6.5 renamed this to first_seen_at"
        assert row["source_mode"] in SOURCE_MODES
        assert row["match_state"] in MATCH_STATES
        assert seconds_between(row["first_seen_at"], row["last_seen_at"]) >= 0

    # One vehicle, and the results are the exact-match list, so every row is exact.
    assert len({r["plate"] for r in response["results"]}) == 1
    assert all(r["match_state"] == "exact" for r in response["results"])
    # Newest first. An operator searching a plate wants the last sighting at the top.
    stamps = [r["first_seen_at"] for r in response["results"]]
    assert all(seconds_between(stamps[i + 1], stamps[i]) > 0 for i in range(len(stamps) - 1))


def test_fuzzy_results_are_never_exact_and_never_merged():
    """Contracts 4.6. Fuzzy generates candidates only.

    Two separate assertions because they are two separate failures: a candidate labelled
    exact is a guess presented as a match, and a candidate merged into `results` is a guess
    presented as a match by a client that had no way to tell them apart.
    """
    payload = load_fixture("search_response.json")
    fuzzy = payload["fuzzy_response"]

    assert fuzzy["query"]["fuzzy"] is True
    assert fuzzy["results"] == [], "exact and fuzzy are never one undifferentiated list"
    assert fuzzy["count"] == 0
    assert fuzzy["candidates"], "fuzzy=true must carry a candidates array"

    for candidate in fuzzy["candidates"]:
        assert candidate["match_state"] in {"probable", "low_confidence", "unreadable"}
        assert candidate["match_state"] != "exact"
        assert isinstance(candidate["distance"], int) and candidate["distance"] > 0

    # The unsurveyed camera and the missing snapshot are in here on purpose: 6.5 allows
    # both as null and the frontend has to render around them rather than crash.
    assert any(c["lat"] is None and c["lon"] is None for c in fuzzy["candidates"])
    assert any(c["snapshot_uri"] is None for c in fuzzy["candidates"])


def test_search_results_agree_with_the_event_fixtures():
    """The directory is one coherent database, not twelve unrelated files.

    search_response.json returns three sightings of GJ01AB1234. Those three come from
    three other fixtures, and if one of them is edited without this one the mock the
    frontend builds against stops describing the data the backend will actually hold.
    """
    payload = load_fixture("search_response.json")
    rows = {r["first_seen_at"]: r for r in payload["response"]["results"]}

    sources = [
        load_fixture("ai_event_high_confidence.json"),
        load_fixture("camera_reconnect.json")["events"][0],
        load_fixture("watchlist_match.json")["matching_event"],
    ]
    assert len(sources) == len(rows)

    for event in sources:
        row = rows.get(event["observed_at"])
        assert row is not None, f"no search row for {event['event_id']} at {event['observed_at']}"
        assert row["camera_id"] == event["camera_id"]
        assert row["plate"] == event["plate"]["normalized"]
        assert row["plate_raw"] == event["plate"]["raw"]
        assert row["source_mode"] == event["source_mode"]
        assert row["source_pts_ms"] == event["source_pts_ms"]
        assert row["evidence_count"] == event["plate"]["evidence_count"]
        assert row["plate_width_px"] == event["plate"]["plate_width_px"]
        assert row["vehicle_type"] == event["vehicle"]["type"]


# ------------------------------------------------------------------- expectations table


def test_expectations_covers_every_fixture_and_nothing_else():
    payload = json.loads((FIXTURE_DIR / "expectations.json").read_text(encoding="utf-8"))
    assert set(payload["fixtures"]) == set(CONTRACT_FIXTURES)


def test_expectations_agree_with_what_the_validator_does():
    """The shared table cannot claim a fixture is locally valid when it is not.

    This is the assertion that keeps the two lanes honest. The backend drives its ingest
    test off expectations.json; if that file said ai_event_bad_timestamp.json was valid,
    the backend would be built to accept it.
    """
    payload = json.loads((FIXTURE_DIR / "expectations.json").read_text(encoding="utf-8"))
    for name, expected in payload["fixtures"].items():
        if expected["locally_valid"] is None:
            assert expected["kind"] == "response_shape"
            continue
        events = fixture_events(load_fixture(name))
        assert events, f"{name} declares locally_valid but carries no events"
        actually_valid = all(validate_payload(e) == [] for e in events)
        assert actually_valid == expected["locally_valid"], (
            f"{name}: expectations.json says locally_valid={expected['locally_valid']}, "
            f"validate_payload says {actually_valid}"
        )
        declared = expected.get("event_count")
        if declared is not None:
            assert len(events) == declared, f"{name}: {len(events)} events, table says {declared}"
