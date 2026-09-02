"""The vehicle quality gate. Owner's manual 5.4 -- where the GPU budget is won.

Four rules, and every one of them is a decision to spend no plate-detection compute on a
vehicle. Rejecting cheaply is the whole point, so the tests here are mostly about the
boundary: a gate one pixel too tight throws away real vehicles, and a gate that never
rejects anything is a stage that costs latency and buys nothing.

The rejection counter matters as much as the decisions. It is the evidence for the claim
that the gate is worth having, and it is the first place to look when recall collapses --
a gate tuned too tight looks exactly like a bad detector from the outside.

One test in here documents something uncomfortable rather than asserting a behaviour:
`test_the_confidence_rule_is_inert_under_bytetrack_defaults`. The gate's 0.35 confidence
floor can never fire on ByteTrack output, because ByteTrack only starts tracks from
detections above 0.5. It is not dead code -- other trackers and other configs reach it --
but a stats() line reading `low_detector_confidence: 0` is not evidence the rule works.
"""

import pytest

from ai.contracts.ids import TrackKey
from ai.contracts.stages import TrackResult
from ai.quality.gate import (
    DEPARTING_HEIGHT_PX,
    DEPARTING_SHRINK_RATIO,
    EDGE_MARGIN_PX,
    MAX_EDGE_CONTACTS,
    MIN_DETECTOR_CONFIDENCE,
    MIN_VEHICLE_HEIGHT_PX,
    GateDecision,
    VehicleGate,
)
from ai.track.base import DEFAULT_HIGH_THRESHOLD

FRAME_W, FRAME_H = 1920, 1080
CAMERA = "cam04"
SESSION_A = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05"
SESSION_B = "8c4b91d6-2a70-4e35-b8f1-59d3c6e04a27"


def vehicle(
    *,
    box=(800, 400, 1000, 560),
    confidence: float = 0.93,
    track_id: int = 42,
    session: str = SESSION_A,
    pts_ms: int = 0,
) -> TrackResult:
    return TrackResult(
        camera_id=CAMERA,
        stream_session_id=session,
        track_id=track_id,
        bbox_xyxy=box,
        class_name="car",
        confidence=confidence,
        frame_index=pts_ms // 100,
        pts_ms=pts_ms,
    )


def centred(height: int, width: int = 200) -> tuple[int, int, int, int]:
    """A box of the given size well away from every frame edge."""
    x1, y1 = 800, 400
    return (x1, y1, x1 + width, y1 + height)


# ---------------------------------------------------------------------------- the happy path


def test_an_ordinary_vehicle_passes():
    gate = VehicleGate()
    decision = gate.check(vehicle(), FRAME_W, FRAME_H)
    assert decision.passed
    assert decision.reason is None
    assert bool(decision) is True


def test_a_decision_is_truthy_or_falsy_so_callers_can_use_it_directly():
    """`if gate.check(...)` has to mean what it looks like it means.

    Without __bool__ a GateDecision is always truthy, so `if gate.check(t, w, h):` would
    admit every vehicle -- a gate that silently stops gating while its stats() still
    reports rejections it computed and discarded.
    """
    assert bool(GateDecision(passed=True)) is True
    assert bool(GateDecision(passed=False, reason="vehicle_too_small")) is False


# --------------------------------------------------------------------------- too small


def test_the_height_floor_is_60_px():
    """Manual 5.4: below 60 px the plate cannot exceed roughly 20 px.

    Indian plates are about a third of vehicle height at best, and 20 px is under every
    Contracts 7.2 reporting bucket -- so this is not a guess about OCR, it is arithmetic.
    """
    assert MIN_VEHICLE_HEIGHT_PX == 60


def test_a_short_vehicle_is_rejected_with_the_arithmetic_in_the_detail():
    gate = VehicleGate()
    decision = gate.check(vehicle(box=centred(40)), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "vehicle_too_small"
    assert "40px" in decision.detail and "60px" in decision.detail
    assert "13px" in decision.detail, "the implied plate width belongs in the reason"


def test_the_height_boundary_is_inclusive_of_the_floor():
    """Exactly 60 px passes, 59 does not. Stated because an off-by-one here throws away
    real vehicles at the far end of every camera, which is where the interesting ones are."""
    gate = VehicleGate()
    assert gate.check(vehicle(box=centred(MIN_VEHICLE_HEIGHT_PX)), FRAME_W, FRAME_H).passed
    assert not gate.check(
        vehicle(box=centred(MIN_VEHICLE_HEIGHT_PX - 1), track_id=43), FRAME_W, FRAME_H
    ).passed


# ------------------------------------------------------------------------- edge clipping


def test_one_edge_contact_is_allowed_and_two_is_not():
    """A vehicle entering frame touches one edge and its plate is usually intact.

    Two contacts means a corner, and a corner vehicle is normally half outside the frame.
    Rejecting on one contact would discard every vehicle at the moment it appears.
    """
    assert MAX_EDGE_CONTACTS == 1
    gate = VehicleGate()

    left_only = (0, 400, 300, 600)
    assert gate.check(vehicle(box=left_only), FRAME_W, FRAME_H).passed

    corner = (0, 0, 300, 200)
    decision = gate.check(vehicle(box=corner, track_id=43), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "edge_clipped"
    assert "2 frame edges" in decision.detail


def test_edge_contact_is_measured_with_a_margin():
    """A box two pixels off the edge is touching it as far as a plate is concerned."""
    assert EDGE_MARGIN_PX == 2
    gate = VehicleGate()
    near_corner = (EDGE_MARGIN_PX, EDGE_MARGIN_PX, 300, 200)
    assert not gate.check(vehicle(box=near_corner), FRAME_W, FRAME_H).passed


def test_the_right_and_bottom_edges_count_too():
    """Easy to check only x1 and y1 and never notice: vehicles leaving frame on the right
    would sail through the gate and produce clipped plates all day."""
    gate = VehicleGate()
    bottom_right = (FRAME_W - 300, FRAME_H - 200, FRAME_W, FRAME_H)
    decision = gate.check(vehicle(box=bottom_right), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "edge_clipped"


def test_a_vehicle_filling_the_frame_touches_four_edges():
    gate = VehicleGate()
    decision = gate.check(vehicle(box=(0, 0, FRAME_W, FRAME_H)), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "edge_clipped"
    assert "4 frame edges" in decision.detail
    assert gate.rejections["edge_clipped"] == 1


# ---------------------------------------------------------------------------- departing


def test_a_shrinking_small_vehicle_is_rejected_as_departing():
    """Better frames were already captured, so there is nothing left to gain.

    Needs two frames: the rule is a comparison against history, which is why the gate is
    stateful at all.
    """
    gate = VehicleGate()
    assert gate.check(vehicle(box=centred(100, 250), pts_ms=100), FRAME_W, FRAME_H).passed

    decision = gate.check(vehicle(box=centred(70, 150), pts_ms=200), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "departing"
    assert "better frames already captured" in decision.detail


def test_a_large_shrinking_vehicle_still_gets_a_plate_attempt():
    """Above the departing height a vehicle is worth trying even while receding.

    A vehicle crossing the frame diagonally shrinks the whole way; rejecting on shrinkage
    alone would drop it before it ever got close enough to read.
    """
    assert DEPARTING_HEIGHT_PX == 110
    gate = VehicleGate()
    gate.check(vehicle(box=centred(400, 700), pts_ms=100), FRAME_W, FRAME_H)
    assert gate.check(vehicle(box=centred(200, 350), pts_ms=200), FRAME_W, FRAME_H).passed


def test_detector_jitter_is_not_departure():
    """A box that wobbled by 5% is the same vehicle at the same distance.

    Without the shrink ratio the departing rule would fire on ordinary detector noise and
    reject roughly half of all frames of every vehicle.
    """
    assert DEPARTING_SHRINK_RATIO == 0.85
    gate = VehicleGate()
    gate.check(vehicle(box=centred(100, 200), pts_ms=100), FRAME_W, FRAME_H)
    assert gate.check(vehicle(box=centred(98, 196), pts_ms=200), FRAME_W, FRAME_H).passed


def test_a_growing_vehicle_is_never_departing():
    gate = VehicleGate()
    gate.check(vehicle(box=centred(70, 150), pts_ms=100), FRAME_W, FRAME_H)
    assert gate.check(vehicle(box=centred(100, 220), pts_ms=200), FRAME_W, FRAME_H).passed


def test_history_is_updated_after_evaluation_not_before():
    """Otherwise every frame is compared against itself and the rule can never fire.

    The bug is invisible: the gate keeps working, just never reports a single departing
    rejection, which reads as "no vehicles departed" rather than "the rule is broken".
    """
    gate = VehicleGate()
    gate.check(vehicle(box=centred(100, 250), pts_ms=100), FRAME_W, FRAME_H)
    gate.check(vehicle(box=centred(70, 150), pts_ms=200), FRAME_W, FRAME_H)
    assert gate.rejections["departing"] == 1


def test_the_first_frame_of_a_track_has_no_history_to_compare_against():
    """A small vehicle on its first frame is arriving, not departing."""
    gate = VehicleGate()
    assert gate.check(vehicle(box=centred(70, 150)), FRAME_W, FRAME_H).passed


# ------------------------------------------------------------------- per-session history


def test_history_is_keyed_on_the_full_trackkey():
    """Keyed on (camera_id, track_id) the gate would compare one car's area before a
    reconnect against a different car's after it, and admit or reject a vehicle on the
    strength of a comparison between two vehicles."""
    gate = VehicleGate()
    gate.check(vehicle(box=centred(400, 700), session=SESSION_A, pts_ms=100), FRAME_W, FRAME_H)

    # Same camera, same track_id 42, new session. A small first sighting, not a departure.
    fresh = gate.check(vehicle(box=centred(70, 150), session=SESSION_B, pts_ms=100), FRAME_W, FRAME_H)
    assert fresh.passed
    assert gate.rejections["departing"] == 0


def test_flush_session_drops_only_that_session():
    gate = VehicleGate()
    gate.check(vehicle(box=centred(400, 700), session=SESSION_A, pts_ms=100), FRAME_W, FRAME_H)
    gate.check(vehicle(box=centred(400, 700), session=SESSION_B, pts_ms=100), FRAME_W, FRAME_H)

    gate.flush_session(SESSION_A)

    # SESSION_A's history is gone, so this small box is a first sighting and passes.
    assert gate.check(
        vehicle(box=centred(70, 150), session=SESSION_A, pts_ms=200), FRAME_W, FRAME_H
    ).passed
    # SESSION_B's history survived, so the same box there is still a departure.
    assert not gate.check(
        vehicle(box=centred(70, 150), session=SESSION_B, pts_ms=200), FRAME_W, FRAME_H
    ).passed


def test_reset_clears_everything():
    gate = VehicleGate()
    gate.check(vehicle(box=centred(400, 700), pts_ms=100), FRAME_W, FRAME_H)
    gate.reset()
    assert gate.check(vehicle(box=centred(70, 150), pts_ms=200), FRAME_W, FRAME_H).passed


# ----------------------------------------------------------------- the confidence rule


def test_a_low_confidence_detection_is_rejected_when_it_reaches_the_gate():
    gate = VehicleGate()
    decision = gate.check(vehicle(confidence=0.20), FRAME_W, FRAME_H)
    assert not decision.passed
    assert decision.reason == "low_detector_confidence"
    assert "0.20" in decision.detail and "0.35" in decision.detail


def test_the_confidence_boundary_is_inclusive():
    gate = VehicleGate()
    assert gate.check(vehicle(confidence=MIN_DETECTOR_CONFIDENCE), FRAME_W, FRAME_H).passed
    assert not gate.check(
        vehicle(confidence=MIN_DETECTOR_CONFIDENCE - 0.01, track_id=43), FRAME_W, FRAME_H
    ).passed


def test_the_confidence_rule_is_inert_under_bytetrack_defaults():
    """Documented rather than fixed, because it is not a bug and it is not dead code.

    ByteTrack only *starts* a track from a detection above its high threshold, so no track
    it emits can ever have peaked below 0.5. The gate's floor is 0.35. Under the default
    configuration the rule therefore cannot fire, and a stats() line reading
    `low_detector_confidence: 0` is not evidence that the rule works -- it is evidence that
    nothing reached it.

    It stays because the tracker is swappable: the IoU tracker has no such floor, a config
    may lower high_threshold, and a scripted tracker in a test can emit anything. What must
    not happen is someone reading that zero as a tuning signal and raising the gate's floor
    to "make it do something", which would start rejecting real vehicles for the first time.
    """
    assert MIN_DETECTOR_CONFIDENCE < DEFAULT_HIGH_THRESHOLD, (
        "if this ever inverts, the gate's floor starts rejecting tracks ByteTrack accepted, "
        "and the two thresholds need reconciling rather than one of them nudging"
    )


def test_cheapest_test_first_so_an_obvious_reject_costs_nothing():
    """A vehicle failing several rules is reported by the cheapest one.

    Ordering is the reason the stage pays for itself: confidence is a float compare, the
    edge test reads four coordinates, and the departing test needs a dict lookup.
    """
    gate = VehicleGate()
    hopeless = vehicle(box=(0, 0, 20, 20), confidence=0.05)
    assert gate.check(hopeless, FRAME_W, FRAME_H).reason == "low_detector_confidence"


def test_rule_order_puts_size_before_edges():
    gate = VehicleGate()
    small_and_cornered = vehicle(box=(0, 0, 30, 30), confidence=0.9)
    assert gate.check(small_and_cornered, FRAME_W, FRAME_H).reason == "vehicle_too_small"


# ---------------------------------------------------------------------------------- stats


def test_stats_count_every_decision_by_reason():
    """The evidence for the claim that the gate is worth having."""
    gate = VehicleGate()
    gate.check(vehicle(track_id=1), FRAME_W, FRAME_H)
    gate.check(vehicle(track_id=2, box=centred(40)), FRAME_W, FRAME_H)
    gate.check(vehicle(track_id=3, box=(0, 0, 300, 200)), FRAME_W, FRAME_H)
    gate.check(vehicle(track_id=4, confidence=0.1), FRAME_W, FRAME_H)

    stats = gate.stats()
    assert stats["evaluated"] == 4
    assert stats["passed"] == 1
    assert stats["rejected"] == 3
    assert stats["pass_rate"] == 0.25
    assert stats["by_reason"] == {
        "vehicle_too_small": 1,
        "edge_clipped": 1,
        "low_detector_confidence": 1,
    }


def test_evaluated_always_balances_against_passed_and_rejected():
    """stats() must read as a balance. A decision counted in neither column is a decision
    nobody can account for."""
    gate = VehicleGate()
    for i in range(20):
        gate.check(vehicle(track_id=i, box=centred(40 + i * 10), confidence=0.1 + i * 0.05), FRAME_W, FRAME_H)
    stats = gate.stats()
    assert stats["evaluated"] == stats["passed"] + stats["rejected"]
    assert sum(stats["by_reason"].values()) == stats["rejected"]


def test_pass_rate_is_none_before_anything_is_evaluated():
    """Not 0.0. A gate that has seen nothing has no pass rate, and reporting 0% would read
    as "the gate rejected everything" in exactly the summary someone skims."""
    assert VehicleGate().stats() == {
        "evaluated": 0, "passed": 0, "rejected": 0, "pass_rate": None, "by_reason": {}
    }


def test_thresholds_are_overridable_for_a_benchmark_sweep():
    """Tuning the gate is a benchmark exercise, so the values have to be injectable.

    Without this, tuning means editing the constant, which means the number that justified
    the change cannot be reproduced afterwards.
    """
    loose = VehicleGate(min_height_px=30, min_confidence=0.1, departing_height_px=50)
    assert loose.check(vehicle(box=centred(35), confidence=0.15), FRAME_W, FRAME_H).passed

    strict = VehicleGate(min_height_px=200)
    assert not strict.check(vehicle(box=centred(150)), FRAME_W, FRAME_H).passed
