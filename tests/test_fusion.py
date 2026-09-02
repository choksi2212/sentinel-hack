"""Temporal OCR consensus. Contracts 4.3 and 4.4.

The centrepiece here is `test_the_walkthrough_example_reproduces`. Four frames, three
reading `GJ01AB1234` and one reading `GJ01A81234`, fusing to `GJ01AB1234` at confidence
0.876 with evidence_count 3. That number is on a slide, in the contracts document, and in
`ai/fusion/consensus.py`'s own docstring. If the arithmetic drifts, the slide becomes a
false claim in front of judges who can do the division themselves.

The rest of the file guards the properties around it. Fusion is per-TrackKey and mixing
tracks invents a vehicle. Zero readable observations is `None`, which means `plate: null`,
which is a correct answer. The confidence is a share of total evidence rather than a
probability, and the tests assert the consequences of that -- notably that a single
unanimous reading scores 1.0, which is what a share means and is *not* what a probability
would mean.
"""

import numpy as np
import pytest

from ai.contracts.ids import TrackKey
from ai.contracts.stages import PlateCandidate, PlateObservation, TrackResult
from ai.fusion.accumulator import (
    DEFAULT_MAX_TRACK_DURATION_MS,
    DEFAULT_TOP_K,
    DEFAULT_TRACK_IDLE_MS,
    CropBuffer,
    EvidenceAccumulator,
    TrackCrop,
)
from ai.fusion.consensus import (
    best_agreeing_observation,
    consensus_gain,
    fuse,
    fuse_observations,
)

CAMERA = "cam04"
SESSION_A = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05"
SESSION_B = "8c4b91d6-2a70-4e35-b8f1-59d3c6e04a27"


def observation(
    text: str,
    ocr_confidence: float,
    image_quality: float,
    *,
    track_id: int = 42,
    session: str = SESSION_A,
    camera: str = CAMERA,
    frame_index: int = 0,
    pts_ms: int = 0,
    width: int = 62,
) -> PlateObservation:
    return PlateObservation(
        camera_id=camera,
        stream_session_id=session,
        track_id=track_id,
        plate_bbox_xyxy=(100, 200, 100 + width, 224),
        plate_width_px=width,
        plate_raw=text,
        ocr_confidence=ocr_confidence,
        image_quality=image_quality,
        frame_index=frame_index,
        pts_ms=pts_ms,
        observed_at="2026-09-01T10:03:21.234Z",
    )


# The four frames from Contracts 4.3, in order, with the weights the document states.
WALKTHROUGH = [
    observation("GJ01AB1234", 0.91, 0.90, frame_index=1, pts_ms=100),   # 0.819
    observation("GJ01AB1234", 0.94, 0.92, frame_index=2, pts_ms=200),   # 0.8648
    observation("GJ01AB1234", 0.88, 0.87, frame_index=3, pts_ms=300),   # 0.7656
    observation("GJ01A81234", 0.63, 0.55, frame_index=4, pts_ms=400),   # 0.3465
]


# ------------------------------------------------------------------- the worked example


def test_the_walkthrough_weights_are_what_the_document_says():
    """Each frame's weight is ocr_confidence * image_quality, and nothing else.

    Asserted separately from the fused result so that if the fusion total moves, the
    failure says whether the inputs or the aggregation changed.
    """
    weights = [round(o.fusion_weight, 4) for o in WALKTHROUGH]
    assert weights == [0.819, 0.8648, 0.7656, 0.3465]


def test_the_walkthrough_example_reproduces():
    """Contracts 4.3, verbatim: GJ01AB1234, evidence_count 3, confidence 0.876.

    2.4494 of evidence for GJ01AB1234 against 0.3465 for GJ01A81234, so the majority
    reading takes 2.4494 / 2.7959 = 0.8761 of the total. A single-frame system that
    happened to sample frame 4 would have emitted GJ01A81234 at 0.63 and been confidently
    wrong, which in a police system is worse than reporting nothing.
    """
    result = fuse(
        [
            {"text": o.plate_raw, "ocr_confidence": o.ocr_confidence, "image_quality": o.image_quality}
            for o in WALKTHROUGH
        ]
    )
    assert result["normalized"] == "GJ01AB1234"
    assert result["evidence_count"] == 3
    assert round(result["confidence"], 3) == 0.876


def test_the_typed_wrapper_reproduces_the_same_numbers():
    """fuse_observations must not quietly change the answer the copied block gives.

    Grammar downgrade off, because GJ01AB1234 passes the grammar anyway -- the point is
    that the wrapper is a wrapper.
    """
    fused = fuse_observations(WALKTHROUGH, apply_grammar_downgrade=False)
    assert fused is not None
    assert fused.normalized == "GJ01AB1234"
    assert fused.evidence_count == 3
    assert round(fused.confidence, 3) == 0.876
    assert fused.total_observations == 4
    assert fused.grammar_ok is True


def test_the_walkthrough_keeps_the_best_agreeing_crop_not_the_best_crop():
    """Frame 2 is the best agreeing observation, and frame 2 is also the best overall.

    Here they coincide. `test_best_agreeing_ignores_a_stronger_outlier` is the case where
    they do not, which is the one that matters.
    """
    fused = fuse_observations(WALKTHROUGH)
    assert fused.best_observation.frame_index == 2
    assert fused.best_observation.plate_raw == "GJ01AB1234"


def test_the_walkthrough_is_the_high_calibration_band():
    """3 observations at 0.876 -- Contracts 4.4's HIGH band, reported as a band."""
    assert fuse_observations(WALKTHROUGH, apply_grammar_downgrade=False).calibration_band == "HIGH"


def test_consensus_gain_reports_no_change_on_the_walkthrough():
    """Fusion agreed with the single best frame here. That is worth measuring too.

    The diagnostic is only interesting if it reports both outcomes honestly; a `changed`
    flag that is always True would make the before-versus-after number meaningless.
    """
    gain = consensus_gain(WALKTHROUGH)
    assert gain["single_frame_pick"] == "GJ01AB1234"
    assert gain["fused"] == "GJ01AB1234"
    assert gain["changed"] is False
    assert gain["distinct_readings"] == 2
    assert gain["observations"] == 4


def test_consensus_gain_reports_the_case_fusion_exists_for():
    """One loud wrong frame, two quiet right ones. The single-frame pick is wrong.

    This is the number to put on the slide: consensus overruled the highest-confidence
    individual read and got the right answer.
    """
    observations = [
        observation("GJ01A81234", 0.95, 0.90),  # weight 0.855, the loudest single frame
        observation("GJ01AB1234", 0.70, 0.70),  # 0.49
        observation("GJ01AB1234", 0.68, 0.72),  # 0.4896
    ]
    gain = consensus_gain(observations)
    assert gain["single_frame_pick"] == "GJ01A81234"
    assert gain["fused"] == "GJ01AB1234"
    assert gain["changed"] is True
    assert gain["evidence_count"] == 2


# --------------------------------------------------------------- what a share of evidence means


def test_a_single_unanimous_reading_scores_one_point_zero():
    """Not a bug. Contracts 4.4: the confidence is a share of total evidence.

    One observation is 100% of the evidence there is. It is emphatically NOT a claim that
    the plate is certainly right -- which is exactly why evidence_count travels with it and
    why the band for this is LOW, not HIGH. A reader who wants "how sure are we" reads the
    band and the count, never the number alone.
    """
    fused = fuse_observations([observation("GJ01AB1234", 0.40, 0.30)])
    assert fused.confidence == 1.0
    assert fused.evidence_count == 1
    assert fused.calibration_band == "LOW"


def test_confidence_is_not_the_ocr_confidence():
    """The fused number and the per-frame OCR score are different quantities.

    Reporting the OCR score as the event confidence is the single easiest way to make the
    whole fusion stage decorative.
    """
    fused = fuse_observations(WALKTHROUGH, apply_grammar_downgrade=False)
    assert fused.confidence not in {o.ocr_confidence for o in WALKTHROUGH}


def test_unanimous_agreement_beats_a_split_at_the_same_total_weight():
    """Same evidence mass, different consensus. The split reading scores lower.

    This is the property that makes the number worth reporting at all.
    """
    unanimous = fuse_observations(
        [observation("GJ01AB1234", 0.80, 0.80), observation("GJ01AB1234", 0.80, 0.80)],
        apply_grammar_downgrade=False,
    )
    split = fuse_observations(
        [observation("GJ01AB1234", 0.80, 0.80), observation("GJ99XY9999", 0.80, 0.80)],
        apply_grammar_downgrade=False,
    )
    assert unanimous.confidence == 1.0
    assert split.confidence == pytest.approx(0.5)
    assert unanimous.confidence > split.confidence


def test_confidence_is_always_in_the_unit_interval():
    """The event schema has a CHECK constraint on this column.

    A share of a positive total cannot leave [0, 1], but the grammar penalty multiplies it
    afterwards and a future penalty above 1.0 would break the invariant silently until
    ingest started rejecting events.
    """
    cases = [
        [observation("GJ01AB1234", 1.0, 1.0)],
        [observation("22BH1234AA", 1.0, 1.0)],   # grammar penalty path
        WALKTHROUGH,
        [observation("GJ01AB1234", 0.01, 0.01), observation("GJ99XY9999", 0.99, 0.99)],
    ]
    for observations in cases:
        fused = fuse_observations(observations)
        assert 0.0 <= fused.confidence <= 1.0, fused


def test_calibration_bands_are_the_three_contracts_4_4_defines():
    assert fuse_observations([observation("GJ01AB1234", 0.9, 0.9)]).calibration_band == "LOW"

    two = [observation("GJ01AB1234", 0.9, 0.9)] * 2
    assert fuse_observations(two, apply_grammar_downgrade=False).calibration_band == "MEDIUM"

    three = [observation("GJ01AB1234", 0.9, 0.9)] * 3
    assert fuse_observations(three, apply_grammar_downgrade=False).calibration_band == "HIGH"


def test_a_grammar_failure_downgrades_the_fused_confidence():
    unusual = [observation("22BH1234AA", 0.9, 0.9)] * 3
    penalized = fuse_observations(unusual)
    honest = fuse_observations(unusual, apply_grammar_downgrade=False)
    assert penalized.confidence < honest.confidence
    assert penalized.grammar_ok is False
    assert penalized.normalized == "22BH1234AA", "the string is never rewritten"


# ------------------------------------------------------------------------ the null cases


def test_no_observations_fuses_to_none():
    assert fuse_observations([]) is None


def test_observations_with_no_usable_characters_fuse_to_none():
    """Every reading normalizes to the empty string, so there is no answer.

    Returning None here is what makes `plate: null` reachable. The alternative -- picking
    the least-bad garbage -- is the fabrication Contracts 3.2 calls the worst failure mode.
    """
    assert fuse_observations([observation("", 0.9, 0.9), observation("...", 0.8, 0.8)]) is None
    assert fuse(
        [{"text": "!!!", "ocr_confidence": 0.9, "image_quality": 0.9}]
    ) is None


def test_unreadable_observations_are_skipped_not_counted_as_a_reading():
    """An empty read must not dilute the confidence of a real one.

    If the empty string were a candidate, a track with one good read and three blanks
    would report 25% confidence in a plate it read perfectly well.
    """
    fused = fuse_observations(
        [observation("GJ01AB1234", 0.9, 0.9), observation("", 0.9, 0.9), observation("   ", 0.9, 0.9)],
        apply_grammar_downgrade=False,
    )
    assert fused.confidence == 1.0
    assert fused.evidence_count == 1
    assert fused.total_observations == 3, "the blanks are still counted as observations"


def test_the_copied_block_divides_by_zero_and_the_wrapper_is_where_that_is_handled():
    """The canonical block from Contracts 4.3 raises when the total weight is zero.

    Asserted rather than fixed, because that block is copied verbatim from the contract and
    editing it here would silently fork the document's own arithmetic. So the guard lives in
    `fuse_observations`, and this test exists so that nobody discovers the crash by
    "tidying up" the wrapper.
    """
    with pytest.raises(ZeroDivisionError):
        fuse([{"text": "GJ01AB1234", "ocr_confidence": 0.0, "image_quality": 0.9}])


def test_a_track_with_no_evidence_weight_is_unreadable_not_a_crash():
    """Every reading scored zero, so there is no evidence to take a share of.

    Reachable in practice: an OCR engine can report 0.0 on text it emitted, and a fully
    black or one-pixel crop can score 0.0 on quality. Before the guard this raised inside
    the frame loop.
    """
    assert fuse_observations([observation("GJ01AB1234", 0.0, 0.9)]) is None
    assert fuse_observations([observation("GJ01AB1234", 0.9, 0.0)]) is None
    assert fuse_observations(
        [observation("GJ01AB1234", 0.0, 0.0), observation("GJ01AB1234", 0.0, 0.5)]
    ) is None


def test_weightless_agreement_is_not_corroboration():
    """The reason zero-weight readings are dropped rather than merely tolerated.

    evidence_count promotes a plate to `probable` at 2 and to the HIGH band at 3. If a
    reading that contributed nothing to the weighted vote still incremented the count, one
    real read plus two worthless ones that happened to agree would be reported to an
    operator as three-frame corroboration.
    """
    observations = [
        observation("GJ01AB1234", 0.90, 0.90),  # the only real evidence
        observation("GJ01AB1234", 0.00, 0.90),  # agrees, contributes nothing
        observation("GJ01AB1234", 0.90, 0.00),  # agrees, contributes nothing
    ]
    fused = fuse_observations(observations, apply_grammar_downgrade=False)
    assert fused.normalized == "GJ01AB1234"
    assert fused.evidence_count == 1, "only the weighted read corroborates"
    assert fused.calibration_band == "LOW"
    assert fused.total_observations == 3, "but all three were still observed"


def test_a_weightless_outlier_cannot_outvote_a_weighted_reading():
    """Symmetric case: the noise must not win either, and must not dilute the answer."""
    fused = fuse_observations(
        [observation("GJ01AB1234", 0.80, 0.80), observation("MH12DE9812", 0.00, 0.90)],
        apply_grammar_downgrade=False,
    )
    assert fused.normalized == "GJ01AB1234"
    assert fused.confidence == 1.0


# ------------------------------------------------------------------ per-TrackKey isolation


def test_fusion_refuses_more_than_one_trackkey():
    """Contracts 4.3 is per-TrackKey. Two tracks in one buffer is a fabricated vehicle.

    This is the one mistake in the module that produces plausible output rather than a
    crash -- passing the camera's whole buffer instead of one track's -- so it raises.
    """
    mixed = [observation("GJ01AB1234", 0.9, 0.9), observation("GJ01AB1234", 0.9, 0.9, track_id=43)]
    with pytest.raises(ValueError, match="distinct TrackKeys"):
        fuse_observations(mixed)


def test_the_same_track_id_in_two_sessions_is_two_tracks():
    """The reconnect case. Same camera, same track_id 42, different session.

    A TrackKey missing the session would make this one track and fuse a car's plate with a
    bus's, which is where the four-seconds-across-Ahmedabad journey comes from.
    """
    mixed = [
        observation("GJ01AB1234", 0.9, 0.9, session=SESSION_A),
        observation("GJ18GS6620", 0.9, 0.9, session=SESSION_B),
    ]
    with pytest.raises(ValueError, match="distinct TrackKeys"):
        fuse_observations(mixed)


def test_best_agreeing_ignores_a_stronger_outlier():
    """The crop we keep must show the plate we reported.

    Here the outlier read has the highest weight of any single frame. Attaching its crop to
    the consensus string would produce an event whose own evidence contradicts its plate
    field -- the detail that unravels a demo under questioning.
    """
    observations = [
        observation("GJ01AB1234", 0.70, 0.70, frame_index=1),
        observation("GJ01AB1234", 0.72, 0.71, frame_index=2),
        observation("GJ01A81234", 0.99, 0.99, frame_index=9),  # strongest, and wrong
    ]
    fused = fuse_observations(observations, apply_grammar_downgrade=False)
    assert fused.normalized == "GJ01AB1234"
    assert fused.best_observation.frame_index == 2
    assert max(observations, key=lambda o: o.fusion_weight).frame_index == 9


def test_best_agreeing_returns_none_when_nothing_agrees():
    assert best_agreeing_observation([observation("GJ01AB1234", 0.9, 0.9)], "MH12DE9812") is None


def test_best_agreeing_compares_normalized_not_raw():
    """`GJ 01 AB 1234` and `gj-01-ab-1234` are the same reading."""
    observations = [observation("GJ 01 AB 1234", 0.9, 0.9), observation("gj-01-ab-1234", 0.5, 0.5)]
    best = best_agreeing_observation(observations, "GJ01AB1234")
    assert best is not None and best.ocr_confidence == 0.9


def test_a_tie_is_broken_deterministically():
    """Two readings with identical evidence must not produce a different answer per run.

    Nondeterminism here would make a benchmark number unreproducible, and an
    unreproducible number cannot be published under Contracts 7.3.
    """
    observations = [observation("GJ01AB1234", 0.8, 0.8), observation("MH12DE9812", 0.8, 0.8)]
    picks = {fuse_observations(observations, apply_grammar_downgrade=False).normalized for _ in range(20)}
    assert len(picks) == 1


# ------------------------------------------------------------------------ evidence buffers


def crop(quality: float, *, frame_index: int = 0, pts_ms: int = 0, width: int = 62) -> TrackCrop:
    return TrackCrop(
        quality=quality,
        crop_bgr=np.zeros((24, width, 3), dtype=np.uint8),
        candidate=PlateCandidate(plate_bbox_xyxy=(0, 0, width, 24), detector_confidence=0.8),
        frame_index=frame_index,
        pts_ms=pts_ms,
        observed_at="2026-09-01T10:03:21.234Z",
        vehicle_bbox_xyxy=(0, 0, 200, 150),
        vehicle_class="car",
        vehicle_confidence=0.93,
    )


def track(track_id: int = 42, *, pts_ms: int = 0, session: str = SESSION_A, box=(0, 0, 200, 150)) -> TrackResult:
    return TrackResult(
        camera_id=CAMERA,
        stream_session_id=session,
        track_id=track_id,
        bbox_xyxy=box,
        class_name="car",
        confidence=0.93,
        frame_index=pts_ms // 100,
        pts_ms=pts_ms,
    )


def test_top_k_default_is_in_the_contract_range():
    """Contracts 4.1 says keep the top 3 to 5 crops per track."""
    assert 3 <= DEFAULT_TOP_K <= 5


def test_the_buffer_keeps_the_best_k_crops_out_of_many():
    """The cost argument for the whole stage.

    At 10 fps a vehicle in frame for three seconds offers 30 crops. OCR is the expensive
    stage; the 26 worst crops buy nothing but latency.
    """
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42), top_k=4)
    for i in range(30):
        buf.offer(crop(i / 100.0, frame_index=i))

    assert len(buf.crops) == 4
    assert [round(c.quality, 2) for c in buf.crops] == [0.29, 0.28, 0.27, 0.26]
    assert buf.crops_offered == 30


def test_crops_offered_balances_against_kept_and_rejected():
    """stats() should read as a balance rather than three unrelated tallies."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42), top_k=3)
    for i in range(12):
        buf.offer(crop((i % 5) / 10.0, frame_index=i))
    assert buf.crops_offered == len(buf.crops) + buf.crops_rejected


def test_the_buffer_is_sorted_best_first():
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42), top_k=4)
    for q in (0.4, 0.9, 0.1, 0.7, 0.8):
        buf.offer(crop(q))
    qualities = [c.quality for c in buf.crops]
    assert qualities == sorted(qualities, reverse=True)


def test_a_worse_crop_is_rejected_once_the_buffer_is_full():
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42), top_k=2)
    assert buf.offer(crop(0.9)) is True
    assert buf.offer(crop(0.8)) is True
    assert buf.offer(crop(0.1)) is False
    assert [c.quality for c in buf.crops] == [0.9, 0.8]


def test_note_track_keeps_the_largest_view_of_the_vehicle():
    """Largest box, not highest confidence.

    Detector confidence saturates on a near vehicle, so it cannot rank the two useful
    frames apart -- and the biggest view is the more useful snapshot for an operator.
    """
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42))
    buf.note_track(track(box=(0, 0, 100, 80)), "2026-09-01T10:03:21.234Z")
    buf.note_track(track(box=(0, 0, 400, 300)), "2026-09-01T10:03:21.334Z")
    buf.note_track(track(box=(0, 0, 120, 90)), "2026-09-01T10:03:21.434Z")
    assert buf.best_vehicle_bbox == (0, 0, 400, 300)


def test_a_vehicle_with_no_plate_evidence_is_still_reportable():
    """"A vehicle passed and could not be identified" is information worth keeping.

    Without the vehicle summary this track could not produce an event at all, and the
    pipeline would silently under-report every vehicle whose plate was unreadable.
    """
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42))
    buf.note_track(track(pts_ms=100), "2026-09-01T10:03:21.234Z")
    assert buf.has_plate_evidence is False
    assert buf.best_vehicle_bbox is not None
    assert buf.vehicle_class == "car"
    assert buf.best_vehicle_confidence == 0.93


def test_duration_is_measured_on_the_source_timeline():
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION_A, 42))
    assert buf.duration_ms == 0, "an empty buffer has no duration, not a negative one"
    buf.note_track(track(pts_ms=1000), "2026-09-01T10:03:21.234Z")
    buf.note_track(track(pts_ms=3400), "2026-09-01T10:03:23.634Z")
    assert buf.duration_ms == 2400


def test_the_accumulator_keys_on_the_full_trackkey():
    """Same camera, same track_id, two sessions, two buffers. Contracts 1.2."""
    acc = EvidenceAccumulator()
    acc.note_track(track(42, session=SESSION_A, pts_ms=100), "2026-09-01T10:03:21.234Z")
    acc.note_track(track(42, session=SESSION_B, pts_ms=100), "2026-09-01T10:41:07.120Z")
    assert len(acc) == 2
    assert TrackKey(CAMERA, SESSION_A, 42) in acc
    assert TrackKey(CAMERA, SESSION_B, 42) in acc


def test_take_finished_uses_pts_not_wallclock():
    """During a 5x replay a wallclock timeout finalizes every track immediately.

    That would destroy the consensus the pipeline depends on, and it would do it only under
    replay -- so it would pass every live test and fail the benchmark.
    """
    acc = EvidenceAccumulator(track_idle_ms=DEFAULT_TRACK_IDLE_MS)
    acc.note_track(track(42, pts_ms=1000), "2026-09-01T10:03:21.234Z")

    assert acc.take_finished(1000 + DEFAULT_TRACK_IDLE_MS - 1) == []
    finished = acc.take_finished(1000 + DEFAULT_TRACK_IDLE_MS)
    assert [b.track_key.track_id for b in finished] == [42]
    assert len(acc) == 0, "a finished buffer is removed, not returned twice"


def test_a_track_open_too_long_is_finalized_anyway():
    """A vehicle stopped at a signal must still produce an event."""
    acc = EvidenceAccumulator()
    acc.note_track(track(42, pts_ms=0), "2026-09-01T10:03:21.234Z")
    acc.note_track(track(42, pts_ms=DEFAULT_MAX_TRACK_DURATION_MS), "2026-09-01T10:03:36.234Z")
    finished = acc.take_finished(DEFAULT_MAX_TRACK_DURATION_MS)
    assert len(finished) == 1


def test_take_session_drains_one_session_and_leaves_the_other():
    """Contracts 1.2: flush evidence buffers when a session ends.

    Flush means finalize and emit, then drop. The vehicles were really observed, so the
    evidence is returned to the caller -- what must not happen is the next session
    appending to it.
    """
    acc = EvidenceAccumulator()
    acc.note_track(track(42, session=SESSION_A, pts_ms=100), "2026-09-01T10:03:21.234Z")
    acc.note_track(track(7, session=SESSION_A, pts_ms=100), "2026-09-01T10:03:21.234Z")
    acc.note_track(track(42, session=SESSION_B, pts_ms=100), "2026-09-01T10:41:07.120Z")

    drained = acc.take_session(SESSION_A)
    assert len(drained) == 2
    assert all(b.track_key.stream_session_id == SESSION_A for b in drained)
    assert len(acc) == 1
    assert TrackKey(CAMERA, SESSION_B, 42) in acc


def test_a_drained_session_cannot_be_appended_to():
    """The buffer is gone, so the next note_track for that key starts a fresh one."""
    acc = EvidenceAccumulator()
    acc.note_track(track(42, session=SESSION_A, pts_ms=100), "2026-09-01T10:03:21.234Z")
    acc.note_track(track(42, session=SESSION_A, pts_ms=200), "2026-09-01T10:03:21.334Z")
    drained = acc.take_session(SESSION_A)[0]
    assert drained.frames_seen == 2

    acc.note_track(track(42, session=SESSION_A, pts_ms=300), "2026-09-01T10:03:21.434Z")
    assert acc.buffer_for(TrackKey(CAMERA, SESSION_A, 42)).frames_seen == 1


def test_take_all_drains_everything_for_eof_and_shutdown():
    acc = EvidenceAccumulator()
    for tid in (1, 2, 3):
        acc.note_track(track(tid, pts_ms=100), "2026-09-01T10:03:21.234Z")
    assert len(acc.take_all()) == 3
    assert len(acc) == 0
    assert acc.take_all() == []


def test_top_k_below_one_is_refused():
    """A buffer that keeps zero crops reads every frame and reports no plates."""
    with pytest.raises(ValueError, match="top_k"):
        EvidenceAccumulator(top_k=0)


def test_accumulator_stats_report_what_is_held():
    acc = EvidenceAccumulator(top_k=4)
    acc.note_track(track(42, pts_ms=100), "2026-09-01T10:03:21.234Z")
    acc.offer_crop(TrackKey(CAMERA, SESSION_A, 42), crop(0.8))
    acc.offer_crop(TrackKey(CAMERA, SESSION_A, 42), crop(0.7))
    assert acc.stats() == {"open_tracks": 1, "crops_held": 2, "top_k": 4}
