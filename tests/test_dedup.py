"""Sighting dedup. Contracts 4.7, LOCKED.

    dedupe_key = sha256(f"{camera_id}|{stream_session_id}|{track_id}|{normalized_plate or ''}")
    DEDUP_WINDOW_SECONDS = 10

The key is enforced by a UNIQUE constraint in the database, which is where it actually
matters, so the first job of this file is to prove the AI side computes the same string the
backend will. A disagreement about the key does not produce an error -- it produces a
UNIQUE violation on a value nobody expected, or worse, two rows the constraint never saw as
duplicates. So the expected digest is hard-coded from the formula rather than recomputed
with the same code under test.

The second job is the pipe-delimited format itself, which has a genuine weakness worth
having a test for: `4.7` concatenates fields with a separator that could in principle
appear inside a field. `test_the_field_separator_is_not_ambiguous_for_real_inputs` states
exactly why that is safe here and what would break the reasoning.
"""

import hashlib

import pytest

from ai.contracts.ids import TrackKey
from ai.dedup.key import DEDUP_WINDOW_SECONDS, SightingDeduper, dedupe_key, dedupe_key_for

CAMERA = "cam04"
SESSION_A = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05"
SESSION_B = "8c4b91d6-2a70-4e35-b8f1-59d3c6e04a27"
PLATE = "GJ01AB1234"

BASE = "2026-09-01T10:03:21.234Z"


def at(seconds: float) -> str:
    """An ISO instant `seconds` after BASE, to millisecond precision."""
    total_ms = 21_234 + int(round(seconds * 1000))
    minute, ms = divmod(total_ms, 60_000)
    return f"2026-09-01T10:{3 + minute:02d}:{ms // 1000:02d}.{ms % 1000:03d}Z"


# ------------------------------------------------------------------------------ the key


def test_the_window_is_ten_seconds():
    assert DEDUP_WINDOW_SECONDS == 10


def test_the_key_is_the_sha256_the_contract_specifies():
    """Computed independently from the formula, not by calling the code under test.

    A test that recomputes the digest with `dedupe_key`'s own logic would pass no matter
    what the format string was, which is exactly the failure that matters: the backend has
    its own implementation, and a silent format change here means the two stop agreeing
    while both keep working.
    """
    expected = hashlib.sha256(
        f"{CAMERA}|{SESSION_A}|42|{PLATE}".encode("utf-8")
    ).hexdigest()
    assert dedupe_key(CAMERA, SESSION_A, 42, PLATE) == expected

    # And the digest itself, written out, so a change to the format string above cannot
    # move both sides of the comparison together.
    assert expected == "461968dd6ee1e2fbcad6e3596afd65450bf3bcb9cd52bd8088629a6ff91254b9", (
        "the key format changed; the backend's UNIQUE constraint is now computing a "
        "different digest for the same vehicle and neither side will report an error"
    )


def test_the_key_is_a_64_character_hex_digest():
    key = dedupe_key(CAMERA, SESSION_A, 42, PLATE)
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_the_key_is_stable_across_calls():
    """It is a database key. Nondeterminism would create one row per emission."""
    assert len({dedupe_key(CAMERA, SESSION_A, 42, PLATE) for _ in range(50)}) == 1


def test_an_unreadable_plate_uses_the_empty_string_not_the_word_none():
    """`normalized_plate or ''`. Interpolating None would put the literal 'None' in the key.

    Which still dedups -- consistently, even -- right up until the backend interpolates an
    SQL NULL as '' and the two implementations produce different digests for the same
    vehicle.
    """
    expected = hashlib.sha256(f"{CAMERA}|{SESSION_A}|42|".encode("utf-8")).hexdigest()
    assert dedupe_key(CAMERA, SESSION_A, 42, None) == expected
    assert dedupe_key(CAMERA, SESSION_A, 42, "") == expected
    assert "None" not in f"{CAMERA}|{SESSION_A}|42|"
    assert expected == "700d1eb6929de40057ec79693d54b76af8b0aa91f024a3231b2261c6d241ee97"


def test_two_unidentified_vehicles_on_one_camera_get_different_keys():
    """Because track_id is in the key.

    Dropping track_id would collapse every unreadable vehicle on a camera into a single
    row -- and unreadable vehicles are the common case at 30 px, so the search would report
    one unidentified vehicle per camera per session for a whole shift.
    """
    a = dedupe_key(CAMERA, SESSION_A, 42, None)
    b = dedupe_key(CAMERA, SESSION_A, 43, None)
    assert a != b


def test_the_same_plate_on_two_cameras_is_two_sightings():
    """The entire point of the product. Merging these would erase the journey."""
    assert dedupe_key("cam04", SESSION_A, 42, PLATE) != dedupe_key("cam07", SESSION_A, 42, PLATE)


def test_the_same_track_id_in_two_sessions_is_two_sightings():
    """Contracts 1.2 again, in the dedup key this time.

    ByteTrack restarts numbering at 1 after a reconnect, so without the session in the key
    the first vehicle of the new session would be deduped against the first vehicle of the
    old one and silently dropped.
    """
    assert dedupe_key(CAMERA, SESSION_A, 1, PLATE) != dedupe_key(CAMERA, SESSION_B, 1, PLATE)


def test_a_different_plate_on_the_same_track_is_a_different_key():
    """A track whose fused plate changed between finalizations is a different sighting.

    Which is correct and slightly counter-intuitive: the same vehicle gets two rows. Better
    two rows an operator can compare than one row that silently overwrote the earlier
    reading with the later one.
    """
    assert dedupe_key(CAMERA, SESSION_A, 42, PLATE) != dedupe_key(CAMERA, SESSION_A, 42, "GJ01AB1284")


def test_dedupe_key_for_matches_the_positional_form():
    key = TrackKey(CAMERA, SESSION_A, 42)
    assert dedupe_key_for(key, PLATE) == dedupe_key(CAMERA, SESSION_A, 42, PLATE)
    assert dedupe_key_for(key, None) == dedupe_key(CAMERA, SESSION_A, 42, None)


def test_the_field_separator_is_not_ambiguous_for_real_inputs():
    """A pipe-joined key is only injective if no field can contain a pipe.

    Here none can: camera_id is `cam\\d{2}` by `is_valid_camera_id`, stream_session_id is a
    UUID, track_id is an int, and normalized_plate is `[^A-Z0-9]`-stripped by
    `normalize_plate`. So the four fields are recoverable and two distinct sightings cannot
    collide.

    Asserted because it is a property of *other* modules that this one silently depends on.
    If camera_id ever became a free-text name, or normalize_plate stopped stripping
    punctuation, the collision would look like dedup working rather than like a bug.
    """
    assert "|" not in CAMERA
    assert "|" not in SESSION_A
    assert "|" not in PLATE

    # track_id and normalized_plate are adjacent in the format string, so this pair is the
    # collision a separator-less key would actually produce: both would join to the same
    # `...421234`. With the pipe they are distinguishable.
    assert dedupe_key(CAMERA, SESSION_A, 42, "1234") != dedupe_key(CAMERA, SESSION_A, 421, "234")
    assert "42" + "1234" == "421" + "234", "the hazard is real: those two joins collide"
    assert "42|1234" != "421|234", "the separator is what makes the join injective"


# ----------------------------------------------------------------------------- the window


def test_the_first_sighting_always_emits():
    deduper = SightingDeduper()
    assert deduper.should_emit("k", BASE, 0.8) is True
    assert deduper.stats()["suppressed"] == 0


def test_a_repeat_inside_the_window_with_worse_evidence_is_suppressed():
    """Without this, one row per frame per vehicle lands in the database and the search
    becomes unusable within minutes."""
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.8)
    assert deduper.should_emit("k", at(3), 0.5) is False
    assert deduper.should_emit("k", at(9), 0.79) is False
    assert deduper.stats()["suppressed"] == 2


def test_equal_evidence_inside_the_window_is_suppressed():
    """Strictly better, not merely as good. Otherwise an identical re-finalization -- which
    the max-duration cap produces on a vehicle stopped at a signal -- emits twice."""
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.8)
    assert deduper.should_emit("k", at(1), 0.8) is False


def test_better_evidence_inside_the_window_re_emits():
    """Contracts 4.7: keep the best evidence, not the first.

    Matching the database's behaviour here keeps the two from disagreeing about which
    snapshot is canonical -- a disagreement an operator would see as a search result whose
    thumbnail does not match its confidence.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.60)
    assert deduper.should_emit("k", at(4), 0.85) is True
    assert deduper.stats()["upgraded"] == 1
    assert deduper.stats()["suppressed"] == 0


def test_an_upgrade_raises_the_bar_rather_than_replacing_it():
    """After a 0.85 upgrade, a later 0.70 must not re-emit.

    If the remembered weight were overwritten rather than maxed, evidence quality could
    walk downhill one emission at a time and the final row would hold the worst crop.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.60)
    deduper.should_emit("k", at(2), 0.85)
    assert deduper.should_emit("k", at(4), 0.70) is False
    assert deduper.should_emit("k", at(6), 0.90) is True


def test_a_sighting_past_the_window_emits_again():
    """The same vehicle passing the same camera twenty seconds later is a real second
    sighting, not a duplicate."""
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.8)
    assert deduper.should_emit("k", at(DEDUP_WINDOW_SECONDS + 0.001), 0.1) is True


def test_the_window_boundary_is_inclusive_of_suppression():
    """Exactly 10.000 s is still inside the window; 10.001 s is outside.

    Contracts 4.7 says "within DEDUP_WINDOW_SECONDS", and the strict comparison is what
    makes exactly-10 a duplicate rather than a coin flip on float rounding.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.8)
    assert deduper.should_emit("k", at(DEDUP_WINDOW_SECONDS), 0.1) is False

    fresh = SightingDeduper()
    fresh.should_emit("k", BASE, 0.8)
    assert fresh.should_emit("k", at(DEDUP_WINDOW_SECONDS + 0.001), 0.1) is True


def test_the_window_is_anchored_to_the_last_emission_not_the_last_observation():
    """A suppressed repeat does not push the window forward.

    Consequence, and it is the behaviour wanted: a vehicle stopped at a signal for a minute
    produces roughly one event per window, not one per frame and not one for the whole
    minute. The alternative -- calling `_remember` on the suppression path too -- would let
    a vehicle that keeps being re-finalized suppress itself indefinitely, and it would
    disappear from the last-seen timeline for as long as it stayed in view. A vehicle
    parked in front of a camera is exactly the one an operator is looking for.
    """
    deduper = SightingDeduper()
    assert deduper.should_emit("k", at(0), 0.8) is True
    assert deduper.should_emit("k", at(9), 0.1) is False
    assert deduper.should_emit("k", at(18), 0.1) is True, (
        "18s after the last emission is outside the window; the suppressed call at 9s "
        "must not have extended it"
    )
    assert deduper.stats()["suppressed"] == 1


def test_an_upgrade_does_slide_the_window():
    """The asymmetry against the test above, and it is deliberate.

    An upgrade emitted, so the window restarts from it -- otherwise the improved event and
    the next one could land inside ten seconds of each other. Suppression emitted nothing,
    so there is nothing for a window to run from.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", at(0), 0.60)
    assert deduper.should_emit("k", at(8), 0.90) is True
    # 14s after the first emission, but only 6s after the upgrade, so still suppressed.
    assert deduper.should_emit("k", at(14), 0.10) is False


def test_different_keys_do_not_interfere():
    deduper = SightingDeduper()
    assert deduper.should_emit("a", BASE, 0.8) is True
    assert deduper.should_emit("b", BASE, 0.1) is True
    assert deduper.should_emit("a", at(1), 0.1) is False
    assert deduper.should_emit("b", at(1), 0.05) is False


def test_the_window_is_configurable():
    """Tuning it is a benchmark exercise on real footage, so it cannot be a constant only."""
    deduper = SightingDeduper(window_seconds=60)
    deduper.should_emit("k", BASE, 0.8)
    assert deduper.should_emit("k", at(30), 0.1) is False
    assert deduper.stats()["window_seconds"] == 60


def test_an_unparseable_timestamp_fails_open():
    """The event still reaches ingest, which rejects it naming the field.

    Failing closed would drop it inside the deduper, where the only trace is a counter --
    and "the plate never appeared in search" is a much harder bug to find than a 422 that
    says `observed_at`. Deliberately the same choice the naive-timestamp fixture tests
    from the other side.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.9)
    assert deduper.should_emit("k", "2026-09-01T10:03:21", 0.1) is True
    assert deduper.should_emit("k", "not a timestamp", 0.1) is True
    assert deduper.stats()["suppressed"] == 0


def test_the_cache_is_bounded():
    """A long run must not grow the cache without limit.

    Oldest-first eviction is safe because the window is 10 seconds: anything old enough to
    be evicted under load is old enough that re-emitting it is correct anyway.
    """
    deduper = SightingDeduper()
    for i in range(25_000):
        deduper.should_emit(f"key-{i}", BASE, 0.5)
    assert deduper.stats()["tracked_keys"] <= 20_000


def test_eviction_keeps_the_most_recent_keys():
    deduper = SightingDeduper()
    for i in range(20_050):
        deduper.should_emit(f"key-{i}", BASE, 0.5)
    # The newest key is still remembered, so a repeat of it is suppressed.
    assert deduper.should_emit("key-20049", at(1), 0.1) is False
    # The oldest was evicted, so a repeat of it emits again -- safe, not silent.
    assert deduper.should_emit("key-0", at(1), 0.1) is True


def test_stats_report_the_shape_the_worker_logs():
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.5)
    deduper.should_emit("k", at(1), 0.9)
    deduper.should_emit("k", at(2), 0.1)
    assert deduper.stats() == {
        "tracked_keys": 1,
        "suppressed": 1,
        "upgraded": 1,
        "window_seconds": DEDUP_WINDOW_SECONDS,
    }


# --------------------------------------------------------- what dedup must never do


def test_suppression_is_a_decision_not_a_deletion():
    """`should_emit` returns a bool. It holds no payload and can discard nothing.

    Contracts 4.7 requires that a repeat update `last_seen_at` and increment
    `observation_count` while never discarding plate_observations. That is the backend's
    job, and this API shape is what keeps it possible: a deduper that swallowed events
    could not hand the backend the repeat it needs to count.
    """
    deduper = SightingDeduper()
    deduper.should_emit("k", BASE, 0.9)
    result = deduper.should_emit("k", at(1), 0.1)
    assert result is False
    assert isinstance(result, bool)
    assert deduper.stats()["suppressed"] == 1, "the repeat is counted, not forgotten"
