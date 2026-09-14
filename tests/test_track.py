"""Tracking -- stage 4, and the stage whose failures are silent.

Every stage after this one groups by TrackKey. Quality ranking picks the best frame
per track, fusion votes across frames of one track, one sighting event is emitted per
track. So a tracking mistake does not look like a tracking mistake downstream: it
looks like a plate. Two vehicles merged into one track produce one event whose plate
belongs to neither, and that event validates, dedupes, geocodes and renders on the
map. Nothing throws.

That is why this module spends most of its assertions on identity rather than on
recall. Recall is the number that looks like tracking quality; identity is the number
that decides whether the answer is right. The ByteTracker docstring makes the point
with the IoU baseline row -- higher recall, 9 identities for 6 vehicles, 2 switches --
and the test for that row is here.

Two things are routinely mistaken for bugs and are not:

  * **Track ids restart at 1 after a session change.** That is correct and is the whole
    reason TrackKey has three parts. See test_the_third_part_of_the_key.
  * **The motion gate changes nothing on the synthetic fixture.** Also correct: four
    well-separated lanes with no crossing trajectories do not contain the failure the
    gate prevents. It is kept on an argument, not on a measurement, and the module
    says so.

The measured figures asserted here come from ai/track/bytetrack.py's own docstrings.
Every run is cached at module level -- each configuration is ~0.7 s and several tests
read the same one.
"""

import numpy as np
import pytest

from ai.contracts.stages import DetectorResult, TrackResult
from ai.detect import build_detector
from ai.detect.base import DEFAULT_CONFIDENCE_THRESHOLD
from ai.media import build_source
from ai.track import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MIN_HITS,
    DEFAULT_TRACK_BUFFER,
    MAX_IOU_COST,
    SHIPPABLE_TRACKERS,
    TRACKER_NAMES,
    ByteTracker,
    IOUTracker,
    KalmanBoxFilter,
    OracleTracker,
    ScriptedTracker,
    SessionMismatchError,
    Track,
    TrackerConfigError,
    TrackerRegistry,
    build_registry,
    build_tracker,
    cxcyah_to_xyxy,
    describe_tracker,
    fuse_detection_score,
    gate_cost,
    iou_cost,
    iou_matrix,
    normalize_tracker_config,
    solve,
    tracker_factory,
    tracker_ships,
    xyxy_to_cxcyah,
)
from ai.track import assignment, kalman
from ai.track import bytetrack as bytetrack_module
from ai.track.base import check_detector_threshold
from ai.track.kalman import (
    CHI2_INV95,
    CHI2_INV99,
    MEASUREMENT_DIMS,
    POSITION_DIMS,
    POSITION_SIZE_DIMS,
    shared_filter,
)
from ai.track.track import CONFIRMED, LOST, REMOVED, TENTATIVE

CAMERA = "cam04"
SESSION = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05"
OTHER_SESSION = "9b2c4d61-0e7a-4f35-91d8-6a3b5c8e2f40"


# --------------------------------------------------------------------- fixtures
#
# The three detector configurations from ai/track/bytetrack.py's measurements. Held
# here verbatim rather than imported because they are the experiment, and an
# experiment whose inputs can be changed elsewhere proves nothing.

PERFECT = {"name": "oracle", "miss_rate": 0.0}

# Flat 0.92 confidence on every box. Named because it is the fixture that does NOT
# pose the second-stage question: nothing ever lands in [0.1, 0.5), so matched_low is
# 0 and an ablation would show the second stage costing nothing.
FLAT = {
    "name": "oracle",
    "miss_rate": 0.15,
    "jitter_px": 4,
    "false_positives_per_frame": 0.1,
}

# The one the numbers are quoted against. confidence_threshold=0.05 is load-bearing:
# the detector default of 0.35 is above the tracker's low threshold and would starve
# stage 2 -- which is the misconfiguration check_detector_threshold exists to catch.
FALLOFF = {
    "name": "oracle",
    "miss_rate": 0.15,
    "jitter_px": 4,
    "false_positives_per_frame": 0.1,
    "confidence_threshold": 0.05,
    "confidence_full_width_px": 140,
    "min_confidence": 0.08,
    "low_confidence_rate": 0.25,
    "low_confidence_value": 0.3,
}

# The same fixture with the detector's own filter left at its default. Not a separate
# experiment -- it is FALLOFF misconfigured, and the point is what that costs.
STARVED = dict(FALLOFF, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD)

SOURCE = {
    "mode": "synthetic",
    "camera_id": "cam01",
    "seed": 42,
    "total_frames": 200,
    "target_interval_ms": 120,
}

_MATCH_IOU = 0.5


def _age_bucket(age: int) -> str:
    if age <= 2:
        return "0-2"
    if age <= 4:
        return "3-4"
    if age <= 9:
        return "5-9"
    if age <= 19:
        return "10-19"
    return "20+"


_CACHE: dict[tuple, dict] = {}


def measure(detector_config, tracker_config=None, *, gate_dims=None):
    """Run the corpus once and return every figure any test in this file needs.

    Cached on the configuration. Each run is ~0.7 s and six tests read the perfect
    fixture, so without this the module would spend most of its time re-deriving
    numbers it already had.

    Two matchings are computed from the same pass because they answer different
    questions. Identity accounting matches each truth vehicle to the best EMITTED box
    at IoU >= 0.5 -- what a consumer of TrackResult sees. The age table matches each
    truth vehicle to the live track whose filter ESTIMATE overlaps it most, which
    counts a coasting track as "being followed" and is how report_bbox_xyxy and
    bbox_xyxy get compared on identical frames.
    """
    tracker_config = dict(tracker_config or {"name": "bytetrack"})
    key = (
        tuple(sorted(detector_config.items())),
        tuple(sorted(tracker_config.items())),
        gate_dims,
    )
    if key in _CACHE:
        return _CACHE[key]

    saved = (bytetrack_module.GATING_DIMS, bytetrack_module.GATING_THRESHOLD)
    if gate_dims is not None:
        bytetrack_module.GATING_DIMS = gate_dims
        bytetrack_module.GATING_THRESHOLD = CHI2_INV99[len(gate_dims)]
    try:
        result = _run(detector_config, tracker_config)
    finally:
        bytetrack_module.GATING_DIMS, bytetrack_module.GATING_THRESHOLD = saved

    _CACHE[key] = result
    return result


def _run(detector_config, tracker_config):
    source = build_source(SOURCE)
    source.open()
    detector = build_detector(detector_config, source=source)
    detector.load()
    tracker = build_tracker(
        tracker_config, "cam01", source.session_id, source=source
    )

    truth_to_tracks: dict[str, set[int]] = {}
    track_to_truths: dict[int, set[str]] = {}
    truth_frames = matched = 0
    report_iou: dict[str, list[float]] = {}
    estimate_iou: dict[str, list[float]] = {}
    followed_not_emitted = 0

    for envelope in source:
        detections = detector.detect_envelope(envelope)
        results = tracker.update(
            detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
        )
        truth = source.truth_at_pts(envelope.pts_ms)
        if truth is None or not truth.vehicles:
            continue
        truth_boxes = [v.vehicle_bbox_xyxy for v in truth.vehicles]
        truth_frames += len(truth_boxes)

        if results:
            grid = iou_matrix(truth_boxes, [r.bbox_xyxy for r in results])
            for index, vehicle in enumerate(truth.vehicles):
                best = int(np.argmax(grid[index]))
                if grid[index, best] < _MATCH_IOU:
                    continue
                matched += 1
                track_id = results[best].track_id
                truth_to_tracks.setdefault(str(vehicle.vehicle_id), set()).add(track_id)
                track_to_truths.setdefault(track_id, set()).add(str(vehicle.vehicle_id))

        if not hasattr(tracker, "tracks"):
            continue
        live = {t.track_id: t for t in tracker.tracks()}
        emitted = {r.track_id: r for r in results}
        if not live:
            continue
        ids = list(live)
        grid = iou_matrix(truth_boxes, [live[i].bbox_xyxy for i in ids])
        for index, box in enumerate(truth_boxes):
            best = int(np.argmax(grid[index]))
            if grid[index, best] <= 0.0:
                continue
            track = live[ids[best]]
            bucket = _age_bucket(track.age)
            if track.track_id not in emitted:
                followed_not_emitted += 1
                continue
            report_iou.setdefault(bucket, []).append(
                float(iou_matrix([box], [emitted[track.track_id].bbox_xyxy])[0, 0])
            )
            estimate_iou.setdefault(bucket, []).append(
                float(iou_matrix([box], [track.bbox_xyxy])[0, 0])
            )

    stats = tracker.stats()
    source.close()
    detector.close()

    return {
        "recall": matched / max(1, truth_frames),
        "truth_frames": truth_frames,
        "vehicles": len(truth_to_tracks),
        "ids": len(track_to_truths),
        "fragments": sum(len(v) - 1 for v in truth_to_tracks.values()),
        "switches": sum(len(v) - 1 for v in track_to_truths.values()),
        "vetoes": stats.get("gate_vetoes"),
        "stats": stats,
        "report_iou": report_iou,
        "estimate_iou": estimate_iou,
        "followed_not_emitted": followed_not_emitted,
    }


def detection(x1, y1, x2, y2, confidence=0.9, class_name="car"):
    return DetectorResult(
        bbox_xyxy=(x1, y1, x2, y2), class_name=class_name, confidence=confidence
    )


def moving(x, confidence=0.9, class_name="car"):
    """One 100x100 vehicle at horizontal position x. The workhorse of the unit tests."""
    return [detection(x, 300, x + 100, 400, confidence, class_name)]


# ============================================================ the three-part key
#
# ai/track/base.py's module docstring is entirely about this, and it is the one claim
# in the package whose failure mode is a wrong answer that looks like a working
# feature.


def test_the_third_part_of_the_key_is_the_session():
    tracker = build_tracker({"name": "bytetrack", "min_hits": 1}, CAMERA, SESSION)
    result = tracker.update(moving(100), frame_index=0, pts_ms=0)[0]

    assert result.track_key == (CAMERA, SESSION, 1)
    assert len(result.track_key) == 3


def test_the_same_track_number_in_two_sessions_is_two_different_keys():
    """The failure the third part prevents, demonstrated rather than described.

    Both sessions produce track_id 1 -- correctly, since numbering restarts. With a
    two-part key these two vehicles would share an identity, and fusion would vote
    their plate characters together into one plate belonging to neither.
    """
    tracker = build_tracker({"name": "bytetrack", "min_hits": 1}, CAMERA, SESSION)
    first = tracker.update(moving(100), frame_index=0, pts_ms=0)[0]

    tracker.reset(stream_session_id=OTHER_SESSION)
    second = tracker.update(moving(700), frame_index=0, pts_ms=0)[0]

    assert first.track_id == second.track_id == 1
    assert first.track_key != second.track_key
    assert len({first.track_key, second.track_key}) == 2


def test_track_numbering_restarts_at_one_and_that_is_deliberate():
    tracker = build_tracker({"name": "bytetrack", "min_hits": 1}, CAMERA, SESSION)
    for index in range(4):
        tracker.update(moving(100 + 200 * index), frame_index=index, pts_ms=100 * index)
    assert tracker.stats()["tracks_started"] == 4

    tracker.reset(stream_session_id=OTHER_SESSION)
    result = tracker.update(moving(100), frame_index=0, pts_ms=0)[0]

    assert result.track_id == 1
    # Cumulative across the boundary; only the allocator restarts.
    assert tracker.stats()["tracks_started"] == 5
    assert tracker.stats()["sessions_seen"] == 2


def test_a_tracker_cannot_be_built_without_a_session():
    with pytest.raises(ValueError, match="stream_session_id"):
        build_tracker({"name": "bytetrack"}, CAMERA, "")
    with pytest.raises(ValueError, match="camera_id"):
        build_tracker({"name": "bytetrack"}, "", SESSION)


def test_the_refusal_explains_the_consequence_not_just_the_rule():
    """A message that says "required" teaches nothing. This one has to say why."""
    with pytest.raises(ValueError) as caught:
        build_tracker({"name": "bytetrack"}, CAMERA, "")
    text = str(caught.value).lower()
    assert "reconnect" in text
    assert "merges" in text


@pytest.mark.parametrize(
    "config",
    [
        {"name": "bytetrack", "min_hits": 1},
        {"name": "iou", "min_hits": 1, "confidence_threshold": 0.1},
        {"name": "scripted", "script": {0: [(7, (10, 20, 110, 120), "bus", 0.8)]}},
    ],
)
def test_every_tracker_stamps_its_own_camera_and_session(config):
    """BaseTracker._emit is the only place a TrackResult is built.

    Tested behaviourally across the implementations rather than by reading the code,
    because the guarantee that matters is that no subclass can construct one itself.
    """
    tracker = build_tracker(config, CAMERA, SESSION)
    results = tracker.update(moving(100), frame_index=3, pts_ms=300)
    assert results, f"{config['name']} emitted nothing to check"
    for result in results:
        assert isinstance(result, TrackResult)
        assert result.camera_id == CAMERA
        assert result.stream_session_id == SESSION
        assert result.track_key[:2] == (CAMERA, SESSION)


# ================================================ the misconfiguration next door
#
# ai/track/base.py: "the default configuration has this bug", and its only symptom is
# a counter sitting at 0.


def test_the_two_shipped_defaults_are_misconfigured_against_each_other():
    """The detector's default filter is above the tracker's low threshold.

    Not a hypothetical the helper guards against -- the state the repository ships in.
    Asserted so that changing either default without the other fails here.
    """
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.35
    assert DEFAULT_LOW_THRESHOLD == 0.1
    assert DEFAULT_CONFIDENCE_THRESHOLD > DEFAULT_LOW_THRESHOLD

    warning = check_detector_threshold(DEFAULT_CONFIDENCE_THRESHOLD)
    assert warning is not None


@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.1])
def test_a_detector_at_or_below_the_low_threshold_is_sound(threshold):
    assert check_detector_threshold(threshold) is None


@pytest.mark.parametrize("threshold", [0.11, 0.35, 0.5])
def test_a_detector_above_the_low_threshold_is_reported(threshold):
    assert check_detector_threshold(threshold) is not None


def test_the_warning_names_both_numbers_and_the_fix():
    warning = check_detector_threshold(0.35, 0.1)
    assert "0.35" in warning and "0.1" in warning
    assert "second association stage" in warning
    # The actionable half. A warning that describes a problem without naming the
    # value to set is a warning that gets acknowledged and not fixed.
    assert "0.1 or below" in warning


def test_starving_the_detector_is_not_a_degradation_but_the_ablation():
    """The sharpest available statement of the cost, and it is exact.

    Setting the detector's threshold to its own default produces results
    indistinguishable from constructing the tracker with use_low_stage=False: the same
    recall, the same identity count, the same fragments, the same tracks started. The
    misconfiguration does not weaken ByteTrack's contribution, it deletes it -- by
    changing a number in a different stage's config file, with no error anywhere.
    """
    starved = measure(STARVED)
    ablated = measure(FALLOFF, {"name": "bytetrack", "use_low_stage": False})

    for field in ("recall", "ids", "fragments", "switches"):
        assert starved[field] == ablated[field], field
    assert starved["stats"]["tracks_started"] == ablated["stats"]["tracks_started"] == 35
    assert starved["stats"]["matched_low"] == ablated["stats"]["matched_low"] == 0

    # And it is the row base.py quotes.
    assert starved["recall"] == pytest.approx(0.342, abs=0.002)
    assert starved["ids"] == 8
    assert starved["fragments"] == 2


def test_the_starved_tracker_reports_nothing_dropped():
    """Why the misconfiguration is silent: nothing is discarded *by the tracker*.

    dropped_below_low counts detections the tracker itself refused. The starved run
    has none, because the boxes were already gone when it was called. Every counter
    the tracker owns looks healthy.
    """
    stats = measure(STARVED)["stats"]
    assert stats["dropped_below_low"] == 0
    assert stats["low_stage_enabled"] is True
    assert stats["matched_low"] == 0


# ================================================================ the main table
#
# ByteTracker's class docstring. Its module docstring promises this file asserts them.


def test_a_perfect_detector_gives_one_identity_per_vehicle():
    result = measure(PERFECT)
    assert result["truth_frames"] == 240
    assert result["vehicles"] == 6
    assert result["recall"] == pytest.approx(0.904, abs=0.002)
    assert result["ids"] == 6
    assert result["fragments"] == 0
    assert result["switches"] == 0


def test_the_naive_baseline_beats_the_motion_model_when_nothing_goes_wrong():
    """Reported because it is true, not because it is flattering.

    With a perfect detector a Kalman filter has nothing to recover, and the filter's
    lag costs a little recall. A module that only published the rows where its choice
    wins is not measuring, it is advertising.
    """
    bytetrack = measure(PERFECT)
    baseline = measure(PERFECT, {"name": "iou"})

    assert baseline["recall"] == pytest.approx(0.938, abs=0.002)
    assert baseline["recall"] > bytetrack["recall"]
    # And it is a fair fight on identity here -- both are clean.
    assert baseline["ids"] == bytetrack["ids"] == 6
    assert baseline["fragments"] == baseline["switches"] == 0


def test_the_second_association_stage_is_most_of_the_tracker():
    full = measure(FALLOFF)
    without = measure(FALLOFF, {"name": "bytetrack", "use_low_stage": False})

    assert full["recall"] == pytest.approx(0.588, abs=0.002)
    assert without["recall"] == pytest.approx(0.342, abs=0.002)
    # 0.246 of recall, which is 42% of what the tracker was recovering.
    assert full["recall"] - without["recall"] == pytest.approx(0.246, abs=0.004)
    assert full["ids"] == 6 and full["fragments"] == 0
    assert without["ids"] == 8 and without["fragments"] == 2


def test_the_second_stage_is_actually_exercised_on_this_fixture():
    """An ablation only means something if the ablated path ran.

    41 matched frames came from boxes below the high threshold. Without this
    assertion, a fixture change that stopped producing marginal boxes would leave the
    ablation test above passing while measuring nothing.
    """
    stats = measure(FALLOFF)["stats"]
    assert stats["matched_low"] == 41
    assert stats["matched_high"] == 94
    assert stats["low_stage_enabled"] is True


def test_the_flat_confidence_fixture_never_poses_the_question():
    """Which is why FALLOFF exists, and it is worth pinning as a negative.

    With flat 0.92 confidence nothing lands between the two thresholds, so the second
    stage is live and idle. Measured on this fixture, disabling it would look free.
    """
    stats = measure(FLAT)["stats"]
    assert stats["matched_low"] == 0
    assert stats["matched_high"] == 169


def test_the_motion_gate_earns_nothing_here_and_is_kept_anyway():
    full = measure(FALLOFF)
    without = measure(FALLOFF, {"name": "bytetrack", "use_gating": False})

    for field in ("recall", "ids", "fragments", "switches"):
        assert full[field] == without[field], field
    # It ran. "Free" and "not wired up" are different claims.
    assert full["stats"]["gate_vetoes"] > 0
    assert without["stats"]["gate_vetoes"] == 0


def test_the_baseline_wins_on_recall_and_is_still_the_wrong_tracker():
    """The row that decides the choice, and it is not decided on recall.

    The primary metric is correct-plate events per eligible vehicle. A switch pools
    two vehicles' plate evidence into one answer that is wrong for both, and no amount
    of recall compensates. 9 identities for 6 vehicles, 5 fragments, 2 switches.
    """
    bytetrack = measure(FALLOFF)
    baseline = measure(FALLOFF, {"name": "iou"})

    assert baseline["recall"] == pytest.approx(0.667, abs=0.002)
    assert baseline["recall"] > bytetrack["recall"]

    assert baseline["ids"] == 9
    assert baseline["fragments"] == 5
    assert baseline["switches"] == 2
    assert bytetrack["switches"] == 0
    assert bytetrack["fragments"] == 0


def test_the_baseline_has_no_motion_model_and_says_so():
    stats = measure(FALLOFF, {"name": "iou"})["stats"]
    assert stats["motion_model"] is None
    assert stats["tracker"] == "iou"


# ======================================================== the gate's real choice
#
# GATING_DIMS in ai/track/bytetrack.py: the dimension count is the load-bearing
# decision and the confidence level is immaterial. Both halves are asserted, because
# the file's earlier comments had that backwards.


def test_the_gate_threshold_is_indexed_by_the_dimensions_it_gates_on():
    """The mismatch this guards against is a looser gate that looks deliberate."""
    assert bytetrack_module.GATING_DIMS == POSITION_SIZE_DIMS
    assert len(bytetrack_module.GATING_DIMS) == 3
    assert bytetrack_module.GATING_THRESHOLD == CHI2_INV99[3]
    assert bytetrack_module.GATING_THRESHOLD == pytest.approx(11.3449)
    assert bytetrack_module.GATING_THRESHOLD != CHI2_INV99[4]


def test_the_mismatch_would_loosen_the_gate_rather_than_tighten_it():
    """Which is why it is the dangerous direction: the failure is permissive.

    A tightened gate loses tracks and someone investigates. A loosened one accepts
    associations it should have vetoed, and the result is a plausible-looking journey.
    """
    for table in (CHI2_INV95, CHI2_INV99):
        assert table[4] > table[3] > table[2]


def test_gating_on_aspect_ratio_is_the_one_change_that_does_damage():
    """Six vehicles into fifteen identities, with a PERFECT detector.

    The largest effect measured anywhere in this module, and it comes from adding one
    dimension to the gate. Included because it is the reason GATING_THRESHOLD is
    derived from GATING_DIMS instead of written as a number.
    """
    shipped = measure(PERFECT)
    with_aspect = measure(PERFECT, gate_dims=MEASUREMENT_DIMS)

    assert with_aspect["ids"] == 15
    assert with_aspect["fragments"] == 9
    assert with_aspect["recall"] == pytest.approx(0.821, abs=0.003)

    assert shipped["ids"] == 6
    assert shipped["fragments"] == 0
    assert with_aspect["recall"] < shipped["recall"]


def test_height_is_the_dimension_that_earns_nothing():
    """The shipped gate is three dimensions of which two do the work.

    Dropping height at the shipped confidence level changes nothing: same recall, same
    identities, same fragments, same veto count. Asserted because the comment block
    over GATING_DIMS used to claim the opposite, on the strength of a row measured at
    95% -- where two dimensions is the *stricter* gate and does fragment.
    """
    shipped = measure(PERFECT)
    position_only = measure(PERFECT, gate_dims=POSITION_DIMS)

    assert position_only["ids"] == shipped["ids"] == 6
    assert position_only["fragments"] == shipped["fragments"] == 0
    assert position_only["recall"] == pytest.approx(shipped["recall"])
    assert position_only["vetoes"] == shipped["vetoes"] == 961


def test_no_veto_in_the_corpus_comes_from_the_height_dimension():
    """The mechanism behind the row above, checked pair by pair rather than in summary.

    Two gates agreeing on recall and identity count could still be disagreeing on
    individual associations and cancelling out. They are not: across every
    track/detection pair the corpus produces, the set vetoed by (cx, cy, h) is a strict
    subset of the set vetoed by (cx, cy). Adding a dimension raises the threshold
    faster than it adds distance to these trajectories, so the wider gate is the
    three-dimensional one.
    """
    source = build_source(SOURCE)
    source.open()
    detector = build_detector(PERFECT, source=source)
    detector.load()
    tracker = build_tracker({"name": "bytetrack"}, SOURCE["camera_id"],
                            source.session_id, source=source)

    pairs = height_only = position_only = 0
    for envelope in source:
        detections = detector.detect_envelope(envelope)
        live = list(tracker.tracks())
        if live and detections:
            measurements = np.array(
                [xyxy_to_cxcyah(d.bbox_xyxy) for d in detections], dtype=float
            )
            for track in live:
                three = track.gating_distance(measurements, dims=POSITION_SIZE_DIMS)
                two = track.gating_distance(measurements, dims=POSITION_DIMS)
                vetoed_by_three = three > CHI2_INV99[3]
                vetoed_by_two = two > CHI2_INV99[2]
                pairs += len(measurements)
                height_only += int(np.sum(vetoed_by_three & ~vetoed_by_two))
                position_only += int(np.sum(vetoed_by_two & ~vetoed_by_three))
        tracker.update(detections, frame_index=envelope.frame_index,
                       pts_ms=envelope.pts_ms)
    source.close()
    detector.close()

    assert pairs > 1000, "the corpus should offer plenty of pairs to disagree over"
    assert height_only == 0
    assert position_only > 0


def test_the_confidence_level_makes_no_difference_at_three_dimensions():
    """The null result, asserted so nobody re-derives it.

    95% and 99% produce identical recall, identities, fragments and veto counts. The
    level is kept at 99% on an argument about asymmetric costs; anyone preferring the
    conventional 95% has the same measurement behind them.
    """
    saved = bytetrack_module.GATING_THRESHOLD
    bytetrack_module.GATING_THRESHOLD = CHI2_INV95[3]
    try:
        at_95 = _run(FALLOFF, {"name": "bytetrack"})
    finally:
        bytetrack_module.GATING_THRESHOLD = saved
    at_99 = measure(FALLOFF)

    for field in ("recall", "ids", "fragments", "switches"):
        assert at_95[field] == at_99[field], field
    assert at_95["stats"]["gate_vetoes"] == at_99["stats"]["gate_vetoes"] == 570


def test_the_gate_vetoes_pairs_that_are_nowhere_near_the_threshold():
    """Why the level is immaterial: the pairs it stops are not marginal.

    A track and a detection two lanes apart sit at squared distances in the hundreds.
    Moving the line from 7.81 to 11.34 lets none of them through.
    """
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    track.predict()
    far = xyxy_to_cxcyah((600, 300, 700, 400))
    distance = float(track.gating_distance(np.array([far]), dims=POSITION_SIZE_DIMS)[0])

    assert distance > 100
    assert distance > CHI2_INV95[3] and distance > CHI2_INV99[3]


def test_a_correct_next_box_is_well_inside_the_gate_from_the_first_frame():
    """The tentative stage runs ungated, and this is what it would have cost.

    A clean target moving 30 px per frame is never near the threshold, so gating the
    tentative stage would veto nothing here. The stage stays ungated on the argument
    in bytetrack.py -- a tentative track dies on its first miss, so a veto is
    unrecoverable -- and not on this measurement.
    """
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    distances = []
    for frame in range(1, 8):
        x = 100 + 30 * frame
        box = (x, 300, x + 100, 400)
        track.predict()
        distances.append(
            float(
                track.gating_distance(
                    np.array([xyxy_to_cxcyah(box)]), dims=POSITION_SIZE_DIMS
                )[0]
            )
        )
        track.update(box, "car", 0.9, frame, 100 * frame, min_hits=3)

    assert max(distances) < bytetrack_module.GATING_THRESHOLD
    # And it converges as the velocity estimate settles.
    assert distances[-1] < distances[0]


# ================================================================ the lifecycle
#
# ai/track/track.py. The state machine is where the subtle bugs live.


def test_a_single_frame_false_positive_never_becomes_a_vehicle():
    """The reason min_hits exists: wet-tarmac headlight glare on the night frames."""
    tracker = build_tracker({"name": "bytetrack"}, CAMERA, SESSION)
    assert tracker.update(moving(100), frame_index=0, pts_ms=0) == []
    assert tracker.update([], frame_index=1, pts_ms=100) == []

    assert tracker.tracks() == []
    assert tracker.stats()["tracks_started"] == 1
    assert tracker.stats()["tracks_removed"] == 1


def test_a_track_is_not_reported_until_it_has_persisted():
    tracker = build_tracker({"name": "bytetrack"}, CAMERA, SESSION)
    emitted = [
        len(tracker.update(moving(100 + 10 * i), frame_index=i, pts_ms=100 * i))
        for i in range(4)
    ]
    assert emitted == [0, 0, 1, 1]
    assert DEFAULT_MIN_HITS == 3


def test_a_tentative_track_dies_on_its_first_miss():
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    assert track.state == TENTATIVE
    track.mark_missed(max_age=DEFAULT_TRACK_BUFFER)
    assert track.state == REMOVED


def test_a_confirmed_track_gets_the_whole_buffer():
    """The asymmetry: no evidence of being real means no patience, and vice versa."""
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    for frame in range(1, DEFAULT_MIN_HITS):
        track.update((100, 300, 200, 400), "car", 0.9, frame, 100 * frame,
                     min_hits=DEFAULT_MIN_HITS)
    assert track.state == CONFIRMED

    for _ in range(DEFAULT_TRACK_BUFFER):
        track.predict()
        track.mark_missed(max_age=DEFAULT_TRACK_BUFFER)
        assert track.state == LOST

    track.predict()
    track.mark_missed(max_age=DEFAULT_TRACK_BUFFER)
    assert track.state == REMOVED


def test_a_lost_track_is_not_reported_while_it_is_being_guessed():
    """Emitting a sighting for a vehicle nobody can see writes a guess to the DB."""
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    for frame in (1, 2):
        track.update((100, 300, 200, 400), "car", 0.9, frame, 100 * frame, min_hits=3)
    assert track.is_active

    track.predict()
    track.mark_missed(max_age=DEFAULT_TRACK_BUFFER)
    assert track.state == LOST
    assert track.is_confirmed is False
    assert track.is_active is False


def test_a_vehicle_that_reappears_keeps_its_identity():
    """Without this, a two-second occlusion is two events for one car."""
    tracker = build_tracker(
        {"name": "bytetrack", "min_hits": 2, "track_buffer": 5}, CAMERA, SESSION
    )
    for index in range(3):
        first = tracker.update(
            moving(100 + 10 * index), frame_index=index, pts_ms=100 * index
        )
    original = first[0].track_id

    for index in range(3, 6):
        assert tracker.update([], frame_index=index, pts_ms=100 * index) == []

    again = tracker.update(moving(160), frame_index=6, pts_ms=600)
    assert len(again) == 1
    assert again[0].track_id == original
    assert tracker.stats()["reacquisitions"] == 1


def test_a_reacquisition_outlives_the_track_that_earned_it():
    """reacquisitions is cumulative, and it used to be recomputed from live tracks.

    That made it decrement: a vehicle that came back from behind a bus and then left
    the junction took its reacquisition with it. The counter sits beside
    tracks_started and tracks_removed in stats(), both of which only count up, and it
    is the only evidence the lost buffer does anything at all -- so "never occluded a
    vehicle" and "buffer is broken" become indistinguishable.
    """
    tracker = build_tracker(
        {"name": "bytetrack", "min_hits": 2, "track_buffer": 3}, CAMERA, SESSION
    )
    for index in range(3):
        tracker.update(moving(100 + 10 * index), frame_index=index, pts_ms=100 * index)
    for index in range(3, 5):
        tracker.update([], frame_index=index, pts_ms=100 * index)
    tracker.update(moving(150), frame_index=5, pts_ms=500)
    assert tracker.stats()["reacquisitions"] == 1

    for index in range(6, 12):
        tracker.update([], frame_index=index, pts_ms=100 * index)

    assert tracker.tracks() == []
    assert tracker.stats()["tracks_removed"] == 1
    assert tracker.stats()["reacquisitions"] == 1


def test_a_session_change_does_not_erase_the_reacquisition_count():
    """The boundary that most often produces them must not be the one that eats them."""
    tracker = build_tracker(
        {"name": "bytetrack", "min_hits": 2, "track_buffer": 3}, CAMERA, SESSION
    )
    for index in range(3):
        tracker.update(moving(100 + 10 * index), frame_index=index, pts_ms=100 * index)
    for index in range(3, 5):
        tracker.update([], frame_index=index, pts_ms=100 * index)
    tracker.update(moving(150), frame_index=5, pts_ms=500)
    assert tracker.stats()["reacquisitions"] == 1

    tracker.reset(stream_session_id=OTHER_SESSION)
    assert tracker.stats()["reacquisitions"] == 1

    tracker.update(moving(100), frame_index=0, pts_ms=0)
    assert tracker.stats()["reacquisitions"] == 1


def test_the_counters_that_read_as_totals_only_ever_increase():
    """The general property. The specific bug above was one instance of it.

    Anything reported next to a cumulative counter is read as cumulative. This walks a
    full lifecycle -- confirm, occlude, re-acquire, remove, change session, start again
    -- and checks monotonicity every frame.
    """
    cumulative = (
        "frames_seen",
        "tracks_started",
        "tracks_removed",
        "reacquisitions",
        "sessions_seen",
        "matched_high",
        "matched_low",
        "matched_tentative",
        "gate_vetoes",
        "dropped_below_low",
    )
    tracker = build_tracker(
        {"name": "bytetrack", "min_hits": 2, "track_buffer": 2, "low_threshold": 0.05},
        CAMERA,
        SESSION,
    )
    previous = {key: 0 for key in cumulative}

    script = (
        [moving(100), moving(110), moving(120)]          # confirm
        + [[], []]                                       # occlude -> lost
        + [moving(150)]                                  # re-acquire
        + [[], [], [], []]                               # let it die
        + [moving(400, 0.2), moving(410), moving(420)]    # a low box, then a new track
    )
    for index, detections in enumerate(script):
        tracker.update(detections, frame_index=index, pts_ms=100 * index)
        stats = tracker.stats()
        for key in cumulative:
            assert stats[key] >= previous[key], f"{key} decreased at frame {index}"
            previous[key] = stats[key]

    tracker.reset(stream_session_id=OTHER_SESSION)
    stats = tracker.stats()
    for key in cumulative:
        assert stats[key] >= previous[key], f"{key} decreased across the session change"


def test_the_class_is_a_confidence_weighted_vote_over_the_whole_track():
    """A vehicle's first detection is its smallest and worst.

    One confident motorcycle vote loses to three weak car votes, which is the point:
    the class on a 40 px box is close to a coin flip and should not decide the answer.
    """
    track = Track(1, (100, 300, 200, 400), "motorcycle", 0.9, 0, 0)
    assert track.class_name == "motorcycle"

    for frame in range(1, 4):
        track.update((100, 300, 200, 400), "car", 0.4, frame, 100 * frame, min_hits=3)

    assert track.class_votes["motorcycle"] == pytest.approx(0.9)
    assert track.class_votes["car"] == pytest.approx(1.2)
    assert track.class_name == "car"


def test_a_tied_vote_breaks_alphabetically_so_two_runs_agree():
    """Determinism, not fairness. A fixture whose answer depends on dict order
    cannot prove anything."""
    track = Track(1, (100, 300, 200, 400), "truck", 0.5, 0, 0)
    track.update((100, 300, 200, 400), "bus", 0.5, 1, 100, min_hits=3)

    assert track.class_votes == {"truck": 0.5, "bus": 0.5}
    assert track.class_name == "bus"


def test_velocity_is_a_direction_and_never_a_speed():
    """Pixels per frame is not km/h without calibration, and 80,000 heterogeneous
    cameras do not have one. The sign survives having none."""
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    for frame in range(1, 6):
        x = 100 + 30 * frame
        track.predict()
        track.update((x, 300, x + 100, 400), "car", 0.9, frame, 100 * frame, min_hits=3)

    vx, vy = track.velocity_px_per_frame
    assert vx > 0
    assert abs(vy) < abs(vx)


def test_duration_comes_from_pts_and_not_from_a_frame_count():
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 10, 1200)
    track.update((110, 300, 210, 400), "car", 0.9, 14, 1680, min_hits=3)
    assert track.duration_ms == 480


# ============================================== which box goes downstream, and why


def test_the_reported_box_is_the_observation_when_there_was_one():
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    track.predict()
    track.update((130, 300, 230, 400), "car", 0.9, 1, 100, min_hits=3)

    assert track.time_since_update == 0
    assert track.report_bbox_xyxy == (130, 300, 230, 400)
    # The estimate is a compromise between the observation and an extrapolation.
    assert track.bbox_xyxy != track.report_bbox_xyxy


def test_the_reported_box_is_the_estimate_when_nothing_was_seen():
    track = Track(1, (100, 300, 200, 400), "car", 0.9, 0, 0)
    for frame in range(1, 4):
        x = 100 + 30 * frame
        track.predict()
        track.update((x, 300, x + 100, 400), "car", 0.9, frame, 100 * frame, min_hits=3)
    last_seen = track.report_bbox_xyxy

    track.predict()
    assert track.time_since_update == 1
    assert track.report_bbox_xyxy == track.bbox_xyxy
    # It moved on rather than staying where it was last seen -- which is the whole
    # difference between this and the IoU baseline.
    assert track.report_bbox_xyxy[0] > last_seen[0]


def test_the_emitted_box_is_the_detection_at_every_track_age():
    """1.000 by construction, and the construction is the design decision.

    report_bbox_xyxy returns last_bbox whenever time_since_update == 0, and is_active
    -- the condition for being emitted -- is exactly that plus CONFIRMED. So an
    emitted box is the detector's own box, and against a perfect detector it is the
    truth box. All 217 of them, min and max.
    """
    report = measure(PERFECT)["report_iou"]
    pooled = [value for values in report.values() for value in values]

    assert len(pooled) == 217
    assert min(pooled) == pytest.approx(1.0)
    assert max(pooled) == pytest.approx(1.0)
    for bucket in ("3-4", "5-9", "10-19", "20+"):
        values = report[bucket]
        assert sum(values) / len(values) == pytest.approx(1.0), bucket


def test_the_filter_estimate_is_what_a_plate_crop_would_have_cost():
    """The other column, on the identical frames: 0.475 at age 3-4.

    A crop taken from the estimate misses more than half the vehicle, and it is worst
    exactly where it hurts most -- early track is when a vehicle is furthest away and
    its plate smallest. The lag decays as velocity converges and never fully closes.
    """
    estimate = measure(PERFECT)["estimate_iou"]
    means = {
        bucket: sum(values) / len(values) for bucket, values in estimate.items()
    }

    assert means["3-4"] == pytest.approx(0.475, abs=0.005)
    assert means["5-9"] == pytest.approx(0.465, abs=0.005)
    assert means["10-19"] == pytest.approx(0.693, abs=0.005)
    assert means["20+"] == pytest.approx(0.900, abs=0.005)

    assert means["3-4"] < 0.5
    assert means["20+"] < 1.0
    assert means["20+"] > means["10-19"] > means["5-9"]


def test_nothing_is_emitted_before_the_track_is_confirmed():
    """Where the perfect fixture's missing 9.6% of recall goes.

    23 of 240 truth vehicle-frames, every one of them a track still below min_hits.
    Not filter lag -- the cost of requiring persistence, paid knowingly.
    """
    result = measure(PERFECT)
    assert "0-2" not in result["report_iou"]
    assert result["followed_not_emitted"] == 23
    assert result["truth_frames"] - 217 == 23


# ==================================================================== assignment
#
# Pure functions, so they get exact assertions rather than statistical ones.


def test_iou_is_zero_and_not_nan_for_a_degenerate_box():
    """A nan propagates into the cost matrix and makes the solver skip a whole row.

    Detectors do return zero-area boxes on a frame edge, so this is reached.
    """
    grid = iou_matrix([(10, 10, 10, 10)], [(0, 0, 100, 100)])
    assert grid.shape == (1, 1)
    assert grid[0, 0] == 0.0
    assert not np.isnan(grid).any()


def test_an_empty_side_gives_an_empty_matrix_of_the_right_shape():
    assert iou_matrix([], [(0, 0, 10, 10)]).shape == (0, 1)
    assert iou_matrix([(0, 0, 10, 10)], []).shape == (1, 0)
    assert iou_matrix([], []).shape == (0, 0)


def test_identical_boxes_cost_nothing_and_disjoint_boxes_cost_everything():
    box = (0, 0, 100, 100)
    assert iou_cost([box], [box])[0, 0] == pytest.approx(0.0)
    assert iou_cost([box], [(500, 500, 600, 600)])[0, 0] == pytest.approx(1.0)


def test_half_overlap_is_a_third_by_iou():
    """Guards the union arithmetic, which is the half that is easy to get wrong."""
    grid = iou_matrix([(0, 0, 100, 100)], [(50, 0, 150, 100)])
    assert grid[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_an_infeasible_pair_is_marked_with_a_finite_cost():
    """scipy raises on an all-inf row, and a track whose vehicle left the frame is
    exactly that. It must produce an unmatched track, not an exception mid-run."""
    cost = np.array([[0.95, 0.99]], dtype=np.float64)
    gated = gate_cost(cost, max_cost=MAX_IOU_COST)

    assert np.isfinite(gated).all()
    assert (gated == assignment.INFEASIBLE).all()

    matches, unmatched_rows, unmatched_cols = solve(gated, max_cost=MAX_IOU_COST)
    assert matches == []
    assert unmatched_rows == [0]
    assert unmatched_cols == [0, 1]


def test_an_explicit_infeasible_mask_is_honoured():
    cost = np.array([[0.1, 0.2]], dtype=np.float64)
    mask = np.array([[True, False]])
    gated = gate_cost(cost, max_cost=MAX_IOU_COST, infeasible=mask)

    assert gated[0, 0] == assignment.INFEASIBLE
    assert gated[0, 1] == pytest.approx(0.2)


def test_gate_cost_does_not_mutate_its_input():
    cost = np.array([[0.95]], dtype=np.float64)
    gate_cost(cost, max_cost=MAX_IOU_COST)
    assert cost[0, 0] == pytest.approx(0.95)


def test_a_pair_the_solver_needed_but_cannot_keep_returns_both_members():
    """Rejection happens after solving, deliberately.

    The solver may take a bad pair to unlock two good ones. Once it has served that
    purpose the pair is undone -- and both its members have to come back as unmatched,
    or a track silently disappears for a frame.
    """
    cost = np.array(
        [
            [0.10, 0.99],
            [0.20, 0.99],
        ],
        dtype=np.float64,
    )
    matches, unmatched_rows, unmatched_cols = solve(cost, max_cost=MAX_IOU_COST)

    assert matches == [(0, 0)]
    assert unmatched_rows == [1]
    assert unmatched_cols == [1]


def test_an_empty_problem_returns_everything_unmatched():
    matches, rows, cols = solve(np.zeros((0, 3)), max_cost=MAX_IOU_COST)
    assert matches == [] and rows == [] and cols == [0, 1, 2]

    matches, rows, cols = solve(np.zeros((2, 0)), max_cost=MAX_IOU_COST)
    assert matches == [] and rows == [0, 1] and cols == []


def test_score_fusion_prefers_the_confident_detection_at_equal_overlap():
    """ByteTrack's score fusion. A cost, never a probability -- Contracts section 8
    forbids multiplying uncalibrated confidences, and this output never leaves the
    assignment."""
    cost = np.array([[0.5, 0.5]], dtype=np.float32)
    fused = fuse_detection_score(cost, [0.9, 0.3])

    assert fused[0, 0] < fused[0, 1]
    assert fused[0, 0] == pytest.approx(1.0 - 0.5 * 0.9, abs=1e-6)
    assert fused[0, 1] == pytest.approx(1.0 - 0.5 * 0.3, abs=1e-6)


def test_score_fusion_on_an_empty_matrix_is_a_no_op():
    empty = np.zeros((0, 0), dtype=np.float32)
    assert fuse_detection_score(empty, []).shape == (0, 0)


def test_the_greedy_fallback_is_deterministic_and_breaks_ties_in_index_order():
    """A track id that depends on tie-break order makes a fixture non-reproducible."""
    cost = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.2],
        ],
        dtype=np.float64,
    )
    first = assignment._greedy_assign(cost)
    second = assignment._greedy_assign(cost)

    assert first == second
    assert list(zip(*first)) == [(0, 0), (1, 1)]


def test_greedy_and_hungarian_agree_on_the_problems_the_tracker_actually_poses():
    """Near-diagonal, well-separated. Which is why the fallback is acceptable."""
    cost = np.array(
        [
            [0.05, 0.90, 0.95],
            [0.92, 0.08, 0.93],
            [0.96, 0.91, 0.11],
        ],
        dtype=np.float64,
    )
    greedy = sorted(zip(*assignment._greedy_assign(cost)))
    combined = sorted(zip(*assignment._assign(cost)))
    assert greedy == combined == [(0, 0), (1, 1), (2, 2)]


def test_greedy_is_not_optimal_and_the_module_says_so():
    """Pinned so the docstring's caveat stays honest rather than defensive.

    Greedy takes (0, 0) at 0.10 and is then forced into 0.90; the optimal pairing
    gives up 0.10 to save 0.70.
    """
    cost = np.array(
        [
            [0.10, 0.20],
            [0.15, 0.90],
        ],
        dtype=np.float64,
    )
    greedy = list(zip(*assignment._greedy_assign(cost)))
    greedy_total = sum(cost[r, c] for r, c in greedy)
    optimal_total = cost[0, 1] + cost[1, 0]

    assert greedy_total > optimal_total
    if assignment._HAVE_SCIPY:
        chosen = list(zip(*assignment._assign(cost)))
        assert sum(cost[r, c] for r, c in chosen) == pytest.approx(optimal_total)


def test_the_solver_in_use_is_reported():
    assert assignment.solver_name() in {"hungarian", "greedy"}
    assert measure(PERFECT)["stats"]["assignment_solver"] == assignment.solver_name()


def test_the_iou_ceiling_is_loose_on_purpose():
    """A motorcycle's box 100 ms later can barely overlap its predecessor.

    Tightening this instead of trusting the motion model loses exactly the vehicles
    the system most needs to read.
    """
    assert MAX_IOU_COST == 0.8
    assert bytetrack_module.MATCH_THRESH_HIGH == 0.8
    # Stage 2 is stricter: a marginal box plus a loose gate is how a plate gets
    # attached to the wrong car.
    assert bytetrack_module.MATCH_THRESH_LOW < bytetrack_module.MATCH_THRESH_HIGH
    assert bytetrack_module.MATCH_THRESH_TENTATIVE < bytetrack_module.MATCH_THRESH_HIGH


# ======================================================================= kalman


def test_the_box_round_trip_holds():
    box = (100, 300, 240, 400)
    assert cxcyah_to_xyxy(xyxy_to_cxcyah(box)) == box


def test_a_zero_height_box_does_not_poison_the_filter():
    """An infinite aspect would corrupt the covariance for the rest of the track."""
    measurement = xyxy_to_cxcyah((50, 50, 50, 50))
    assert np.isfinite(measurement).all()
    assert measurement[2] == pytest.approx(1.0)
    assert measurement[3] == pytest.approx(1.0)


def test_an_absurd_aspect_is_clamped_before_it_becomes_a_box():
    """An unclamped filter can drift during a long occlusion, and the box it reports
    then spans the frame and matches everything."""
    wide = cxcyah_to_xyxy(np.array([500.0, 300.0, 500.0, 100.0]))
    assert wide[2] - wide[0] == 20 * 100

    negative = cxcyah_to_xyxy(np.array([500.0, 300.0, -3.0, 100.0]))
    assert negative[2] > negative[0]


def test_the_filter_holds_no_track_state_and_is_shared():
    assert shared_filter() is shared_filter()

    filt = KalmanBoxFilter()
    mean, cov = filt.initiate(xyxy_to_cxcyah((100, 300, 200, 400)))
    other_mean, other_cov = filt.initiate(xyxy_to_cxcyah((600, 300, 700, 400)))

    assert mean[0] != other_mean[0]
    assert np.allclose(cov, other_cov)


def test_a_new_track_has_zero_velocity_with_large_uncertainty():
    """One box gives no information about motion. Pretending otherwise makes the
    first prediction confidently wrong."""
    mean, cov = shared_filter().initiate(xyxy_to_cxcyah((100, 300, 200, 400)))
    assert np.allclose(mean[4:], 0.0)
    assert cov[4, 4] > 0.0
    assert cov[6, 6] < cov[4, 4]


def test_uncertainty_grows_while_a_track_is_unobserved():
    """Which is what lets the gate widen for a vehicle behind a bus."""
    filt = shared_filter()
    mean, cov = filt.initiate(xyxy_to_cxcyah((100, 300, 200, 400)))
    before = cov[0, 0]
    for _ in range(5):
        mean, cov = filt.predict(mean, cov)
    assert cov[0, 0] > before


def test_a_stationary_track_does_not_teleport_when_its_covariance_collapses():
    """The reason update() factorises instead of inverting.

    Feeding one box repeatedly drives the projected covariance towards singular.
    Explicitly inverting it produces a gain full of enormous values and the track
    jumps across the frame.
    """
    filt = shared_filter()
    box = (100, 300, 200, 400)
    mean, cov = filt.initiate(xyxy_to_cxcyah(box))
    for _ in range(200):
        mean, cov = filt.predict(mean, cov)
        mean, cov = filt.update(mean, cov, xyxy_to_cxcyah(box))

    assert np.isfinite(mean).all()
    assert np.isfinite(cov).all()
    assert cxcyah_to_xyxy(mean) == pytest.approx(box, abs=2)
    assert np.all(np.linalg.eigvalsh(cov[:4, :4]) > -1e-6)


def test_the_gate_ignores_the_dimensions_it_is_told_to_ignore():
    """The mechanism behind the whole GATING_DIMS table.

    A box in the right place with a badly wrong shape is far away in four dimensions
    and near in three. Which is why including aspect scatters identities.

    "The right place" has to mean the same centre AND the same height, since those are
    the three dimensions the shipped gate judges -- so the wide box is built outward
    from x=150 rather than rightward from x=100. Widening one edge moves the centre by
    half the width added, and a first draft of this test did exactly that: (100,300)
    to (400,400) puts the centre 100 px downstream, which the position dims see as a
    genuine displacement and the three-dimensional distance came out at 52.9, ten times
    over its own threshold. The test then proved nothing about aspect ratio.
    """
    filt = shared_filter()
    mean, cov = filt.initiate(xyxy_to_cxcyah((100, 300, 200, 400)))
    mean, cov = filt.predict(mean, cov)

    # Centre (150, 350) and height 100, both unchanged. Width 100 -> 300, so aspect
    # goes 1.0 -> 3.0: a car-shaped box where a car-shaped box is expected, three
    # times too wide.
    same_place_wrong_shape = np.array([xyxy_to_cxcyah((0, 300, 300, 400))])
    three = filt.gating_distance(mean, cov, same_place_wrong_shape,
                                dims=POSITION_SIZE_DIMS)[0]
    four = filt.gating_distance(mean, cov, same_place_wrong_shape,
                                dims=MEASUREMENT_DIMS)[0]

    assert three == pytest.approx(0.0, abs=1e-9)
    assert three < CHI2_INV99[3]
    assert four > three
    assert four > CHI2_INV99[4], "aspect alone should be enough to veto this"


def test_a_position_only_gate_ignores_height_too():
    filt = shared_filter()
    mean, cov = filt.initiate(xyxy_to_cxcyah((100, 300, 200, 400)))
    mean, cov = filt.predict(mean, cov)

    right_place_wrong_height = np.array([xyxy_to_cxcyah((100, 250, 200, 450))])
    two = filt.gating_distance(mean, cov, right_place_wrong_height, dims=POSITION_DIMS)[0]
    three = filt.gating_distance(
        mean, cov, right_place_wrong_height, dims=POSITION_SIZE_DIMS
    )[0]
    assert three > two


def test_the_numpy_linear_algebra_fallback_agrees_with_scipy(monkeypatch):
    """The tracker is the one stage that must run on a machine with nothing installed.

    Exercised explicitly because CI may only ever take one of the two branches, and a
    fallback nobody runs is a fallback that is broken.
    """
    if not kalman._HAVE_SCIPY_LINALG:
        pytest.skip("scipy.linalg absent; the fallback is already the live path")

    filt = KalmanBoxFilter()
    box = xyxy_to_cxcyah((100, 300, 200, 400))
    measurement = np.array([xyxy_to_cxcyah((130, 305, 232, 402))])

    mean, cov = filt.initiate(box)
    mean, cov = filt.predict(mean, cov)
    scipy_mean, scipy_cov = filt.update(mean, cov, measurement[0])
    scipy_distance = filt.gating_distance(mean, cov, measurement,
                                          dims=POSITION_SIZE_DIMS)

    monkeypatch.setattr(kalman, "_HAVE_SCIPY_LINALG", False)
    assert kalman.solver_name() == "numpy"
    numpy_mean, numpy_cov = filt.update(mean, cov, measurement[0])
    numpy_distance = filt.gating_distance(mean, cov, measurement,
                                          dims=POSITION_SIZE_DIMS)

    assert np.allclose(scipy_mean, numpy_mean, atol=1e-8)
    assert np.allclose(scipy_cov, numpy_cov, atol=1e-8)
    assert np.allclose(scipy_distance, numpy_distance, atol=1e-8)


def test_the_triangular_solve_fallback_is_correct_in_both_directions():
    lower = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, -1.0, 4.0]])
    rhs = np.array([[4.0, 2.0], [9.0, 3.0], [8.0, 1.0]])

    solved = kalman._numpy_solve_triangular(lower, rhs, lower=True)
    assert np.allclose(lower @ solved, rhs)

    upper = lower.T
    solved = kalman._numpy_solve_triangular(upper, rhs, lower=False)
    assert np.allclose(upper @ solved, rhs)


def test_the_triangular_solve_fallback_handles_a_single_column():
    lower = np.array([[2.0, 0.0], [1.0, 4.0]])
    rhs = np.array([6.0, 7.0])
    solved = kalman._numpy_solve_triangular(lower, rhs, lower=True)

    assert solved.shape == (2,)
    assert np.allclose(lower @ solved, rhs)


def test_the_chi_squared_tables_are_the_published_values():
    """Verbatim from the DeepSORT reference so a comparison against published
    tracker numbers is comparing the same gate."""
    assert CHI2_INV95[4] == pytest.approx(9.4877)
    assert CHI2_INV99[3] == pytest.approx(11.3449)
    assert set(CHI2_INV95) == set(CHI2_INV99) == set(range(1, 10))
    for dof in range(1, 10):
        assert CHI2_INV99[dof] > CHI2_INV95[dof]


def test_the_solver_choice_is_reported_so_a_cross_machine_difference_is_explained():
    assert kalman.solver_name() in {"scipy", "numpy"}
    assert measure(PERFECT)["stats"]["linalg_solver"] == kalman.solver_name()


# ===================================================================== registry
#
# ai/track/registry.py. The ordering requirement is the expensive one.


class FakeSessionChange:
    def __init__(self, camera_id, stream_session_id):
        self.camera_id = camera_id
        self.stream_session_id = stream_session_id


class LegacySessionChange:
    """Carries session_id rather than stream_session_id. The registry reads all three
    names via getattr so it does not have to import the media package."""

    def __init__(self, camera_id, session_id):
        self.camera_id = camera_id
        self.session_id = session_id


class MediaSessionChange:
    """Shaped like the real ai.media SessionChange: camera_id and new_session_id.

    Deliberately not the real dataclass. The registry must stay usable with no media
    package present -- that is what makes the AI stages testable from a fixture -- so
    the test that proves it reads this shape must not import the thing it is imitating.
    test_the_open_event_alone_is_enough_to_expose_the_field_mismatch uses the real one.
    """

    def __init__(self, camera_id, new_session_id):
        self.camera_id = camera_id
        self.new_session_id = new_session_id
        self.previous_session_id = None
        self.reason = "reconnect"


class FakeSource:
    def __init__(self, events=()):
        self._events = list(events)
        self.listeners = []

    def drain_session_events(self):
        events, self._events = self._events, []
        return events

    def add_session_listener(self, listener):
        self.listeners.append(listener)


def registry(**kwargs):
    return TrackerRegistry(
        lambda camera_id, session_id: build_tracker(
            {"name": "bytetrack", "min_hits": 1}, camera_id, session_id
        ),
        **kwargs,
    )


def test_a_frame_from_an_undrained_session_raises_rather_than_mixing_vehicles():
    """The mistake: drain after tracking instead of before.

    A loud failure in a test run is worth an unbounded amount of quiet wrongness in a
    demo, because the quiet version produces one track id and one plate for two cars.
    """
    reg = registry()
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)

    with pytest.raises(SessionMismatchError) as caught:
        reg.update(CAMERA, OTHER_SESSION, moving(100), frame_index=0, pts_ms=0)

    text = str(caught.value)
    assert "drained" in text
    assert "on_session_change" in text


def test_draining_before_tracking_resets_the_tracker():
    reg = registry()
    source = FakeSource([FakeSessionChange(CAMERA, OTHER_SESSION)])
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)

    assert reg.drain(source) == 1
    results = reg.update(CAMERA, OTHER_SESSION, moving(700), frame_index=0, pts_ms=0)

    assert results[0].stream_session_id == OTHER_SESSION
    assert results[0].track_id == 1
    assert reg.stats()["session_changes_handled"] == 1


def test_draining_an_empty_source_is_free():
    reg = registry()
    assert reg.drain(FakeSource()) == 0
    assert reg.stats()["session_changes_handled"] == 0


def test_a_lenient_registry_resets_instead_of_raising():
    """For a long-running worker where a crash is worse than a reset. Off by default:
    the reset happens on first sight of a frame that is already too late to associate
    correctly, so it is the safety net rather than the mechanism."""
    reg = registry(strict_sessions=False)
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)

    results = reg.update(CAMERA, OTHER_SESSION, moving(700), frame_index=0, pts_ms=0)
    assert results[0].stream_session_id == OTHER_SESSION
    assert reg.stats()["session_changes_handled"] == 1
    assert reg.stats()["strict_sessions"] is False


def test_a_session_change_for_an_unseen_camera_is_recorded_not_dropped():
    """Otherwise the camera's first frame looks like a mismatch and raises."""
    reg = registry()
    reg.on_session_change(FakeSessionChange("cam09", OTHER_SESSION))
    assert len(reg) == 0

    results = reg.update("cam09", OTHER_SESSION, moving(100), frame_index=0, pts_ms=0)
    assert results[0].stream_session_id == OTHER_SESSION


def test_a_session_change_is_accepted_under_any_of_the_three_names():
    """stream_session_id downstream, session_id historically, new_session_id upstream.

    The third is the one the media layer actually emits, and reading only the first two
    is what made drain() unusable against a real source.
    """
    for fake in (FakeSessionChange, LegacySessionChange, MediaSessionChange):
        reg = registry()
        reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)
        reg.on_session_change(fake(CAMERA, OTHER_SESSION))
        reg.expect_session(CAMERA, OTHER_SESSION)


def test_a_session_change_missing_either_half_is_refused():
    reg = registry()
    with pytest.raises(ValueError, match="needs camera_id and a session id"):
        reg.on_session_change(FakeSessionChange("", OTHER_SESSION))
    with pytest.raises(ValueError, match="needs camera_id and a session id"):
        reg.on_session_change(FakeSessionChange(CAMERA, ""))
    with pytest.raises(ValueError, match="needs camera_id and a session id"):
        reg.on_session_change(object())


def test_expect_session_catches_the_ordering_mistake_at_the_call_site():
    reg = registry()
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)

    reg.expect_session(CAMERA, SESSION)
    with pytest.raises(SessionMismatchError):
        reg.expect_session(CAMERA, OTHER_SESSION)
    # An unknown camera is not an error; there is nothing to be wrong about yet.
    reg.expect_session("cam17", SESSION)


def test_one_tracker_per_camera_with_no_state_between_them():
    """A worker serving several cameras cannot leak identity across them."""
    reg = registry()
    first = reg.update("cam01", SESSION, moving(100), frame_index=0, pts_ms=0)
    second = reg.update("cam02", SESSION, moving(100), frame_index=0, pts_ms=0)

    assert first[0].track_id == second[0].track_id == 1
    assert first[0].track_key != second[0].track_key
    assert reg.cameras() == ["cam01", "cam02"]
    assert len(reg) == 2
    assert reg.stats()["trackers_created"] == 2


def test_attach_subscribes_and_is_documented_as_the_wrong_tool_for_ordering():
    reg = registry()
    source = FakeSource()
    reg.attach(source)

    assert source.listeners == [reg.on_session_change]
    # The push path works; it just does not guarantee the ordering drain() does.
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)
    source.listeners[0](FakeSessionChange(CAMERA, OTHER_SESSION))
    reg.expect_session(CAMERA, OTHER_SESSION)


def test_reset_all_keeps_each_camera_on_its_own_session():
    reg = registry()
    reg.update("cam01", SESSION, moving(100), frame_index=0, pts_ms=0)
    reg.update("cam02", OTHER_SESSION, moving(100), frame_index=0, pts_ms=0)

    reg.reset_all()

    assert reg.update("cam01", SESSION, moving(100), frame_index=1, pts_ms=100)
    assert reg.update("cam02", OTHER_SESSION, moving(100), frame_index=1, pts_ms=100)


def test_a_forgotten_camera_leaves_no_trace():
    reg = registry()
    reg.update(CAMERA, SESSION, moving(100), frame_index=0, pts_ms=0)
    reg.forget(CAMERA)

    assert reg.cameras() == []
    # Including the session record -- otherwise re-adding it would raise a mismatch.
    reg.update(CAMERA, OTHER_SESSION, moving(100), frame_index=0, pts_ms=0)
    assert reg.cameras() == [CAMERA]


def test_forgetting_an_unknown_camera_is_not_an_error():
    reg = registry()
    reg.forget("cam99")


def test_the_registry_reports_per_camera_stats():
    reg = registry()
    reg.update("cam01", SESSION, moving(100), frame_index=0, pts_ms=0)
    reg.update("cam02", SESSION, moving(100), frame_index=0, pts_ms=0)

    stats = reg.stats()
    assert stats["cameras"] == 2
    assert stats["frames_routed"] == 2
    assert sorted(stats["per_camera"]) == ["cam01", "cam02"]
    assert stats["per_camera"]["cam01"]["camera_id"] == "cam01"


def test_iterating_a_registry_is_sorted_and_deterministic():
    reg = registry()
    for camera in ("cam09", "cam02", "cam05"):
        reg.update(camera, SESSION, moving(100), frame_index=0, pts_ms=0)
    assert [camera for camera, _ in reg] == ["cam02", "cam05", "cam09"]


def test_the_registry_drains_a_real_media_source():
    """The fake source above tests the registry; this tests the two together.

    Which is the test that matters, because it is the one the fakes could not fail.
    on_session_change accepted camera_id plus stream_session_id, and the real
    SessionChange carries neither -- it says camera_id (now) and new_session_id -- so
    drain() raised ValueError on the "open" event every source emits before its first
    frame. Every fake in this module happened to be built to the registry's spec rather
    than to the media layer's, so nothing caught it.
    """
    source = build_source(dict(SOURCE, total_frames=25))
    source.open()
    reg = build_registry({"name": "bytetrack", "min_hits": 1})
    drained = 0
    routed = 0
    try:
        for envelope in source:
            drained += reg.drain(source)
            reg.expect_session(envelope.camera_id, envelope.stream_session_id)
            reg.update(
                envelope.camera_id,
                envelope.stream_session_id,
                [],
                frame_index=envelope.frame_index,
                pts_ms=envelope.pts_ms,
            )
            routed += 1
    finally:
        source.close()

    # 25 raw frames at SYNTHETIC_STEP_MS=40 span 0..960 ms, sampled at the fixture's
    # 120 ms interval. The count is the sampler's business, not the registry's; what
    # this test cares about is that every frame that arrived was routed and that the
    # session events arrived with it rather than raising.
    assert routed > 0
    assert reg.stats()["frames_routed"] == routed
    assert drained >= 1, "the open event alone should have been drained"
    assert reg.cameras() == ["cam01"]


def test_the_open_event_alone_is_enough_to_expose_the_field_mismatch():
    """One drain call against one freshly opened source. No frames needed."""
    source = build_source(dict(SOURCE, total_frames=2))
    source.open()
    try:
        events = source.drain_session_events()
        assert events, "opening a source announces its first session"
        change = events[0]
        assert change.camera_id == SOURCE["camera_id"]
        assert change.new_session_id == source.session_id
        assert change.previous_session_id is None
        assert not hasattr(change, "stream_session_id")

        reg = build_registry({"name": "bytetrack"})
        reg.on_session_change(change)
        reg.expect_session(source.camera_id, source.session_id)
    finally:
        source.close()


# ====================================================================== factory


def test_an_unknown_tracker_name_is_refused_with_the_alternatives():
    with pytest.raises(TrackerConfigError) as caught:
        build_tracker({"name": "sort"}, CAMERA, SESSION)
    assert "sort" in str(caught.value)
    for name in TRACKER_NAMES:
        assert name in str(caught.value)


def test_a_config_without_a_name_is_refused():
    with pytest.raises(TrackerConfigError, match="no 'name'"):
        build_tracker({}, CAMERA, SESSION)


def test_a_config_that_is_not_a_mapping_is_refused():
    with pytest.raises(TrackerConfigError, match="must be a mapping"):
        build_tracker(["bytetrack"], CAMERA, SESSION)


def test_a_misspelled_key_is_refused_rather_than_ignored():
    """`track_bufer:` would otherwise run on the default and produce a benchmark row
    that silently describes different settings than the one beside it."""
    with pytest.raises(TrackerConfigError) as caught:
        build_tracker({"name": "bytetrack", "track_bufer": 60}, CAMERA, SESSION)
    assert "track_bufer" in str(caught.value)
    assert "track_buffer" in str(caught.value)


def test_a_key_belonging_to_another_tracker_is_refused():
    with pytest.raises(TrackerConfigError, match="min_iou"):
        build_tracker({"name": "bytetrack", "min_iou": 0.3}, CAMERA, SESSION)
    with pytest.raises(TrackerConfigError, match="high_threshold"):
        build_tracker(
            {"name": "iou", "high_threshold": 0.5}, CAMERA, SESSION
        )


@pytest.mark.parametrize(
    "low,high",
    [(0.6, 0.5), (-0.1, 0.5), (0.1, 1.5)],
)
def test_the_thresholds_must_be_ordered_and_in_range(low, high):
    config = {"name": "bytetrack", "low_threshold": low, "high_threshold": high}
    with pytest.raises((TrackerConfigError, ValueError), match="low_threshold"):
        build_tracker(config, CAMERA, SESSION)
    with pytest.raises(TrackerConfigError, match="low_threshold"):
        normalize_tracker_config(config)


def test_the_defaults_are_the_ones_the_measurements_used():
    tracker = build_tracker({"name": "bytetrack"}, CAMERA, SESSION)
    assert tracker.high_threshold == DEFAULT_HIGH_THRESHOLD == 0.5
    assert tracker.low_threshold == DEFAULT_LOW_THRESHOLD == 0.1
    assert tracker.track_buffer == DEFAULT_TRACK_BUFFER == 30
    assert tracker.min_hits == DEFAULT_MIN_HITS == 3
    assert tracker.use_low_stage and tracker.use_gating and tracker.fuse_score


def test_the_oracle_needs_a_source_that_carries_truth():
    with pytest.raises(TrackerConfigError, match="ground truth"):
        build_tracker({"name": "oracle"}, CAMERA, SESSION)

    class Bare:
        pass

    with pytest.raises(TypeError, match="truth_at_pts"):
        OracleTracker(CAMERA, SESSION, Bare())


def test_the_scripted_tracker_needs_a_script_and_it_must_be_a_mapping():
    with pytest.raises(TrackerConfigError, match="requires a 'script'"):
        build_tracker({"name": "scripted"}, CAMERA, SESSION)
    with pytest.raises(TrackerConfigError, match="must be a mapping"):
        build_tracker({"name": "scripted", "script": [(0, [])]}, CAMERA, SESSION)


@pytest.mark.parametrize("name", TRACKER_NAMES)
def test_only_bytetrack_may_be_published(name):
    assert tracker_ships(name) is (name == "bytetrack")
    assert (name in SHIPPABLE_TRACKERS) is (name == "bytetrack")


@pytest.mark.parametrize("name", ["iou", "oracle", "scripted"])
def test_publication_refuses_a_tracker_the_submission_does_not_use(name):
    config = {"name": name}
    assert normalize_tracker_config(config) == config

    with pytest.raises(TrackerConfigError) as caught:
        normalize_tracker_config(config, for_publication=True)
    assert name in str(caught.value)
    assert "bytetrack" in str(caught.value)


def test_an_ablation_is_legal_to_run_and_illegal_to_publish_as_bytetrack():
    """Turning off the second stage produces a SORT variant. Its numbers must not be
    filed under the name of the method the submission claims."""
    config = {"name": "bytetrack", "use_low_stage": False}
    assert normalize_tracker_config(config) == config
    assert build_tracker(config, CAMERA, SESSION).use_low_stage is False

    with pytest.raises(TrackerConfigError, match="whole method"):
        normalize_tracker_config(config, for_publication=True)


def test_the_gating_ablation_is_publishable_because_it_is_a_tuning_choice():
    """Unlike the second stage, the gate is not what ByteTrack is. Drawing the line
    somewhere is the point; drawing it here is the judgement."""
    config = {"name": "bytetrack", "use_gating": False}
    assert normalize_tracker_config(config, for_publication=True) == config


def test_the_provenance_string_names_the_thresholds_that_change_the_output():
    """Two runs with the same footage, the same detector and different thresholds are
    not comparable, and a report recording them identically invites the comparison."""
    tracker = build_tracker({"name": "bytetrack"}, CAMERA, SESSION)
    assert tracker.model_name == "bytetrack@0.5/0.1"

    other = build_tracker(
        {"name": "bytetrack", "high_threshold": 0.6, "low_threshold": 0.2},
        CAMERA,
        SESSION,
    )
    assert other.model_name == "bytetrack@0.6/0.2"
    assert other.model_name != tracker.model_name


@pytest.mark.parametrize(
    "flag,label",
    [
        ("use_low_stage", "no-low-stage"),
        ("use_gating", "no-gating"),
        ("fuse_score", "no-score-fusion"),
    ],
)
def test_each_ablation_is_visible_in_the_provenance_string(flag, label):
    tracker = build_tracker({"name": "bytetrack", flag: False}, CAMERA, SESSION)
    assert label in tracker.model_name


def test_the_common_case_stays_short_so_a_deviation_is_loud():
    tracker = build_tracker({"name": "bytetrack"}, CAMERA, SESSION)
    assert "no-" not in tracker.model_name


def test_describe_tracker_summarises_without_constructing_one():
    described = describe_tracker({"name": "bytetrack"})
    assert described == {
        "name": "bytetrack",
        "ships": True,
        "track_buffer": DEFAULT_TRACK_BUFFER,
        "min_hits": DEFAULT_MIN_HITS,
        "two_stage": True,
    }

    assert describe_tracker({"name": "bytetrack", "use_low_stage": False})[
        "two_stage"
    ] is False
    # Only ByteTrack has two stages to report.
    assert describe_tracker({"name": "iou"})["two_stage"] is False
    assert describe_tracker({"name": "iou"})["ships"] is False


def test_build_registry_freezes_the_config_so_a_late_camera_matches_an_early_one():
    """Otherwise a camera added mid-run picks up different settings and the
    multi-camera benchmark is not comparable with itself."""
    config = {"name": "bytetrack", "min_hits": 1, "track_buffer": 7}
    reg = build_registry(config)
    reg.update("cam01", SESSION, moving(100), frame_index=0, pts_ms=0)

    config["track_buffer"] = 99
    reg.update("cam02", SESSION, moving(100), frame_index=0, pts_ms=0)

    per_camera = reg.stats()["per_camera"]
    assert per_camera["cam01"]["tracker"] == per_camera["cam02"]["tracker"]
    assert reg._trackers["cam02"].track_buffer == 7


def test_build_registry_validates_once_rather_than_on_the_first_camera():
    """A bad config should fail at build time, not several minutes into a run."""
    with pytest.raises(TrackerConfigError, match="unknown key"):
        build_registry({"name": "bytetrack", "buffer": 10})


def test_tracker_factory_binds_the_config_and_the_source():
    factory = tracker_factory({"name": "bytetrack", "min_hits": 1})
    first = factory("cam01", SESSION)
    second = factory("cam02", SESSION)

    assert first is not second
    assert first.camera_id == "cam01"
    assert second.camera_id == "cam02"


# ============================================== the diagnostic trackers, briefly


def test_the_oracle_never_switches_an_identity_which_is_its_whole_purpose():
    """It separates two questions that otherwise get answered together: is the plate
    reading wrong, or is the tracking wrong?"""
    result = measure(FALLOFF, {"name": "oracle"})
    assert result["switches"] == 0
    assert result["ids"] == result["vehicles"] == 6
    assert result["fragments"] == 0


def test_the_oracle_declares_itself_so_the_benchmark_refuses_to_publish():
    source = build_source(dict(SOURCE, total_frames=3))
    source.open()
    try:
        tracker = build_tracker({"name": "oracle"}, CAMERA, SESSION, source=source)
        assert tracker.is_oracle is True
        assert tracker.stats()["is_oracle"] is True
    finally:
        source.close()


def test_the_oracle_is_silent_rather_than_inventive_when_truth_is_missing():
    """Returning the detections with invented ids would look like it worked."""

    class NoTruth:
        def truth_at_pts(self, pts_ms):
            return None

    tracker = OracleTracker(CAMERA, SESSION, NoTruth())
    assert tracker.update(moving(100), frame_index=0, pts_ms=0) == []
    assert tracker.stats()["unresolved_frames"] == 1


def test_the_scripted_tracker_replays_by_call_order_not_by_frame_index():
    """Surprising and deliberate: the cursor advances per update() call, and
    frame_index only lands in the emitted row. Pinned because a reader will assume
    otherwise."""
    script = {
        0: [(7, (10, 20, 110, 120), "bus", 0.8)],
        1: [(7, (20, 20, 120, 120), "bus", 0.9)],
    }
    tracker = ScriptedTracker(CAMERA, SESSION, script)

    first = tracker.update([], frame_index=100, pts_ms=1000)
    assert len(first) == 1
    assert first[0].track_id == 7
    assert first[0].frame_index == 100
    assert first[0].bbox_xyxy == (10, 20, 110, 120)

    second = tracker.update([], frame_index=200, pts_ms=2000)
    assert second[0].bbox_xyxy == (20, 20, 120, 120)

    assert tracker.update([], frame_index=300, pts_ms=3000) == []


def test_the_scripted_tracker_rewinds_on_reset():
    script = {0: [(1, (10, 20, 110, 120), "car", 0.8)]}
    tracker = ScriptedTracker(CAMERA, SESSION, script)
    assert tracker.update([], frame_index=0, pts_ms=0)
    assert tracker.update([], frame_index=1, pts_ms=100) == []

    tracker.reset(stream_session_id=OTHER_SESSION)
    replayed = tracker.update([], frame_index=0, pts_ms=0)
    assert replayed[0].stream_session_id == OTHER_SESSION


def test_the_baseline_loses_a_track_the_moment_it_is_occluded():
    """Its stated weakness, asserted rather than discovered. No prediction means the
    box stays where it was last seen. Use it where occlusion is rare, not at a
    junction."""
    tracker = IOUTracker(
        CAMERA, SESSION, min_iou=0.35, min_hits=2, track_buffer=10,
        confidence_threshold=0.1,
    )
    for index in range(3):
        tracker.update(moving(100 + 30 * index), frame_index=index, pts_ms=100 * index)

    # Gone for three frames, then back where it would actually be.
    for index in range(3, 6):
        tracker.update([], frame_index=index, pts_ms=100 * index)
    again = tracker.update(moving(250), frame_index=6, pts_ms=600)

    assert again == [] or again[0].track_id != 1
    assert tracker.stats()["motion_model"] is None


def test_the_baseline_drops_the_low_confidence_boxes_bytetrack_keeps():
    tracker = IOUTracker(CAMERA, SESSION, min_hits=1, confidence_threshold=0.3)
    assert tracker.update(moving(100, confidence=0.2), frame_index=0, pts_ms=0) == []
    assert tracker.stats()["tracks_started"] == 0


# ============================================================= no attribution
#
# The repo-wide scan lives in tests/test_no_attribution.py. This is the local guard
# for the package this module covers.


def test_the_tracking_package_carries_no_authorship_attribution():
    from conftest import REPO_ROOT

    # Assembled from fragments so this guard file holds no whole marker word
    # (the repo-wide scan in tests/test_no_attribution.py relies on that).
    needles = ("co-auth" "ored-by", "cla" "ude", "anthro" "pic", "chat" "gpt",
               "copi" "lot", "generated " "with", "ai-" "generated")
    for path in sorted((REPO_ROOT / "ai" / "track").rglob("*.py")):
        lowered = path.read_text(encoding="utf-8").lower()
        for needle in needles:
            assert needle not in lowered, f"{path.name} contains {needle!r}"
