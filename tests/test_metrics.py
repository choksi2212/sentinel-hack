"""Prove the honest number is the only one this module will hand out.

ai/metrics.py exists to make a flattering-but-false accuracy figure structurally hard to
produce. These tests are weighted, heavily, toward the four defenses that do that work,
because every one of them is a place where a plausible refactor quietly inflates the headline:

  1. Fabricating a plate and abstaining (plate: null) cost the primary rate exactly the same,
     but they are NOT the same failure and the module refuses to let them be added together
     unseen -- a null is a vehicle nobody identified, a wrong string is a vehicle somebody
     else gets accused of being.
  2. The denominator is driven by ground truth, not by what the pipeline emitted. Iterating
     events instead computes "of the vehicles I found, how many did I read", a different and
     much prettier number. A vehicle the detector missed entirely still sits in the
     denominator.
  3. An oracle stage's accuracy is a statement about a config file, so it cannot reach
     benchmark/reports/. The refusal is structural (a different directory), not a warning.
  4. No number is ever reported as a single average: the per-bucket table is primary and an
     empty bucket reads `n=0` / null, never 0.0.

Everything here runs against synthetic ground truth and dict-shaped events -- the same shape a
replayed spool file has. No real registration plate and no real credential appears; the plate
strings are invented and the module never touches the network or a GPU on this path.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from ai import metrics as M
from ai.contracts.event import PlateBlock


# --------------------------------------------------------------------------- helpers


def _gt(uid, *, readable=True, plate="GJ01AB1234", width=90, condition="day"):
    """A GroundTruthVehicle with sane defaults; override only what a test is about."""
    return M.GroundTruthVehicle(
        vehicle_uid=uid,
        readable=readable,
        plate=plate,
        plate_width_px=width,
        condition=condition,
    )


def _ev(uid, normalized=None, *, raw=None, null=False):
    """A dict-shaped event -- the loaded-fixture / replayed-spool shape metrics accepts."""
    if null:
        plate = None
    else:
        plate = {"normalized": normalized}
        if raw is not None:
            plate["raw"] = raw
    return {"uid": uid, "plate": plate}


def _link(known):
    """A link that returns the uid for a known vehicle and None otherwise.

    None is the contract for "this event matched no labelled vehicle"; a uid that matches no
    truth would simply never be looked up, so the None path is what feeds unmatched_events.
    """
    known = set(known)
    return lambda e: (e.get("uid") if e.get("uid") in known else None)


# ============================================================================ TIER 1
# The fabrication / abstention distinction -- the whole honesty argument, per vehicle.
# ============================================================================


def test_a_single_wrong_plate_is_fabricated_not_merely_incorrect():
    """The failure that matters: a real registration number attached to a vehicle that does
    not carry it. It must be its own outcome, not folded into a generic miss."""
    outcome, predicted = M._outcome_for(_gt("v"), [_ev("v", "GJ99XX0000")])
    assert outcome == M.OUTCOME_FABRICATED
    assert predicted == "GJ99XX0000"


def test_a_null_plate_is_abstained_and_earns_no_point():
    """Honest, and not correct. Contracts 3.2 calls a null a valid event -- a statement about
    the event, not a credit in this metric."""
    outcome, predicted = M._outcome_for(_gt("v"), [_ev("v", null=True)])
    assert outcome == M.OUTCOME_ABSTAINED
    assert predicted is None


def test_abstaining_and_fabricating_cost_the_primary_rate_the_same():
    """Both leave `correct` unmoved, so the headline rate cannot distinguish them -- which is
    exactly why fabrication_rate exists as a separate number below."""
    abstain = M.ScoreCard.score([_gt("v")], [_ev("v", null=True)], link=_link(["v"]))
    fabricate = M.ScoreCard.score([_gt("v")], [_ev("v", "WRONG0000")], link=_link(["v"]))
    assert abstain.rate == 0.0
    assert fabricate.rate == 0.0
    # ...and yet they are not the same event, and the module keeps them apart:
    assert abstain.fabricated == 0
    assert fabricate.fabricated == 1


def test_fabrication_rate_is_not_the_complement_of_the_correct_rate():
    """A clip of one correct, one null, one wrong: rate is 1/3 and fabrication is 1/3, and the
    missing third is the abstention -- which neither number counts. Adding rate+fabrication
    would imply a vehicle that does not exist."""
    gts = [_gt("a"), _gt("b"), _gt("c")]
    evs = [_ev("a", "GJ01AB1234"), _ev("b", null=True), _ev("c", "WRONG0000")]
    card = M.ScoreCard.score(gts, evs, link=_link(["a", "b", "c"]))
    tally = card.buckets[M.width_bucket(90)]
    assert tally.rate == pytest.approx(1 / 3)
    assert tally.fabrication_rate == pytest.approx(1 / 3)


def test_a_fragmented_track_one_right_one_wrong_is_contradicted_and_counts_against():
    """One vehicle, two events, one correct and one wrong. Crediting the correct half would
    score fragmentation as free -- and fragmentation is what TrackKey and the session id exist
    to prevent. The system has published a wrong identification; a search for the wrong plate
    returns this vehicle."""
    outcome, predicted = M._outcome_for(
        _gt("v"), [_ev("v", "GJ01AB1234"), _ev("v", "GJ99XX0000")]
    )
    assert outcome == M.OUTCOME_CONTRADICTED
    assert "GJ01AB1234" in predicted and "GJ99XX0000" in predicted  # both surfaced to argue


def test_a_contradiction_is_counted_among_fabrications_in_the_headline_count():
    """ScoreCard.fabricated pools fabricated + contradicted: a contradiction contains a wrong
    plate, so a report that says '0 fabrications' while a contradiction stands would be lying
    by category."""
    card = M.ScoreCard.score(
        [_gt("v")], [_ev("v", "GJ01AB1234"), _ev("v", "GJ99XX0000")], link=_link(["v"])
    )
    assert card.fabricated == 1
    assert card.correct == 0  # the correct half earns nothing


def test_two_agreeing_correct_reads_collapse_to_a_single_correct():
    """Consensus, not contradiction. Distinct reads is a set, so agreement is one value."""
    outcome, _ = M._outcome_for(
        _gt("v"), [_ev("v", "GJ01AB1234"), _ev("v", "GJ01AB1234")]
    )
    assert outcome == M.OUTCOME_CORRECT


def test_two_events_same_wrong_value_is_a_single_fabrication():
    outcome, predicted = M._outcome_for(
        _gt("v"), [_ev("v", "GJ99XX0000"), _ev("v", "GJ99XX0000")]
    )
    assert outcome == M.OUTCOME_FABRICATED
    assert predicted == "GJ99XX0000"


def test_events_that_all_carry_null_are_abstained_not_missed():
    """`missed` means no event at all; two null events is a different fact -- the vehicle was
    seen and not identified -- and the outcomes must not conflate them."""
    outcome, _ = M._outcome_for(_gt("v"), [_ev("v", null=True), _ev("v", null=True)])
    assert outcome == M.OUTCOME_ABSTAINED


# ============================================================================ TIER 1
# The denominator is ground truth, never the events. Reversing the loop is the prettier lie.
# ============================================================================


def test_a_vehicle_with_no_event_at_all_is_missed_and_stays_in_the_denominator():
    """The detector missing a vehicle entirely is the failure most tempting to hide, because
    the vehicle leaves no trace in the event stream. It sits in the denominator regardless."""
    card = M.ScoreCard.score([_gt("ghost")], [], link=_link(["ghost"]))
    assert card.eligible == 1
    assert card.correct == 0
    assert card.rate == 0.0
    assert card.buckets[M.width_bucket(90)].missed == 1


def test_events_for_unlabelled_vehicles_do_not_touch_the_denominator():
    """A run that emits events for vehicles ground truth never labelled cannot inflate the
    rate: those events are counted as unmatched and excluded from the primary metric by its
    definition."""
    card = M.ScoreCard.score(
        [_gt("a")],
        [_ev("a", "GJ01AB1234"), _ev("x", "WHATEVER"), _ev("y", "ALSO")],
        link=_link(["a"]),
    )
    assert card.eligible == 1
    assert card.rate == 1.0
    assert card.unmatched_events == 2


def test_an_ineligible_vehicle_is_out_of_the_denominator_entirely():
    """readable=False means no human could read the plate in any sampled frame -- a property
    of the footage. It is counted so the clip can be described, never scored."""
    card = M.ScoreCard.score(
        [_gt("a"), _gt("bad", readable=False, plate=None, width=0)],
        [_ev("a", "GJ01AB1234")],
        link=_link(["a", "bad"]),
    )
    assert card.eligible == 1
    assert card.ineligible == 1
    assert card.rate == 1.0  # the ineligible vehicle did not drag it to 1/2


def test_readable_true_with_no_plate_string_is_rejected_at_construction():
    """An eligible vehicle with no label can never be scored correct, so it would lower the
    metric merely by existing -- which is not a measurement. Caught where it is written."""
    with pytest.raises(ValueError, match="readable=True requires a plate string"):
        M.GroundTruthVehicle(vehicle_uid="v", readable=True, plate=None)


def test_a_condition_outside_the_locked_set_is_rejected():
    with pytest.raises(ValueError, match="condition"):
        _gt("v", condition="monsoon")


def test_duplicate_ground_truth_uid_is_rejected_before_it_double_counts():
    """Two rows for one vehicle would count it twice in the denominator."""
    with pytest.raises(ValueError, match="duplicate ground truth"):
        M.ScoreCard.score([_gt("v"), _gt("v", plate="GJ02CD5678")], [], link=_link(["v"]))


# ============================================================================ TIER 1
# _plate_string refuses to guess. A silent None here scores every vehicle as an abstention.
# ============================================================================


def test_an_unrecognised_event_shape_raises_rather_than_scoring_a_fictional_zero():
    """The subtle catastrophe: an object metrics does not understand, read as 'no plate',
    scores every vehicle abstained and reports a clean, plausible, entirely fictional 0%."""
    with pytest.raises(TypeError, match="cannot read a plate"):
        M._plate_string(1234)


def test_plate_string_reads_the_three_supported_shapes():
    assert M._plate_string({"plate": {"normalized": "N", "raw": "R"}}) == "N"
    assert M._plate_string({"plate": {"raw": "R"}}) == "R"  # normalized absent -> raw

    class _Evt:  # the worker's in-hand shape: an object with a .plate
        plate = {"normalized": "N2"}

    assert M._plate_string(_Evt()) == "N2"

    class _Dumpable:  # anything with to_dict()
        def to_dict(self):
            return {"plate": {"normalized": "N3"}}

    assert M._plate_string(_Dumpable()) == "N3"


def test_plate_string_reads_a_real_plateblock_normalized_then_raw():
    """Locks the attribute contract against the real contracts type: if PlateBlock ever renames
    .normalized, this fails here rather than silently scoring every read as an abstention."""

    class _Holder:
        def __init__(self, plate):
            self.plate = plate

    pb = PlateBlock(
        raw="GJ 01 AB 1234",
        normalized="GJ01AB1234",
        confidence=0.9,
        match_state="confirmed",
        plate_width_px=90,
        evidence_count=3,
        bbox_xyxy=(10, 20, 110, 60),
    )
    assert M._plate_string(_Holder(pb)) == "GJ01AB1234"
    pb_unreadable = PlateBlock(
        raw="GJ0",
        normalized=None,
        confidence=0.0,
        match_state="unreadable",
        plate_width_px=40,
        evidence_count=1,
        bbox_xyxy=(10, 20, 50, 40),
    )
    assert M._plate_string(_Holder(pb_unreadable)) == "GJ0"  # falls through to raw


def test_plate_string_returns_none_only_for_a_genuinely_absent_plate():
    assert M._plate_string({"plate": None}) is None
    assert M._plate_string({}) is None


# ============================================================================ TIER 1
# An empty bucket is null, never 0.0. 0.0 means "we got every one wrong"; null means "none
# of this size was in the clip", and printing the first when the second is true hides that
# the benchmark has no coverage there.
# ============================================================================


def test_empty_buckets_are_null_not_zero():
    eb = M.empty_buckets()
    assert tuple(eb.keys()) == M.BUCKET_KEYS
    assert all(v is None for v in eb.values())


def test_a_bucket_with_no_eligible_vehicles_reports_none_not_zero():
    card = M.ScoreCard.score([_gt("a", width=90)], [_ev("a", "GJ01AB1234")], link=_link(["a"]))
    widths = card.by_plate_width()
    assert widths["80-100"] == 1.0
    assert widths["<30"] is None  # nothing that small in the clip -- not a 0% score
    assert widths[">100"] is None


def test_coverage_gaps_name_the_untested_buckets():
    """A clean table with empty rows is a benchmark that did not test the hard cases, and that
    is indistinguishable from one that passed them unless somebody says so."""
    card = M.ScoreCard.score([_gt("a", width=90)], [_ev("a", "GJ01AB1234")], link=_link(["a"]))
    gaps = card.coverage_gaps()
    assert "80-100" not in gaps
    assert {">100", "60-80", "40-60", "30-40", "<30"} == set(gaps)


# ============================================================================ TIER 1
# An oracle number cannot reach benchmark/reports/. The refusal is structural.
# ============================================================================


def _stage(name, *, ships=True, is_oracle=False, sha="deadbeef"):
    return M.StageIdentity(
        name=name,
        model_name=f"{name}-model",
        model_version="1",
        ships=ships,
        is_oracle=is_oracle,
        weights_sha256=sha,
    )


def _shipping_stages():
    return {
        "detect": _stage("detect", sha="d"),
        "plate": _stage("plate", sha="p"),
        "ocr": _stage("ocr", sha="o"),
    }


def _good_card():
    return M.ScoreCard.score(
        [_gt("a", width=90)], [_ev("a", "GJ01AB1234")], link=_link(["a"])
    )


def test_a_shipping_run_with_a_manifest_is_publishable():
    rep = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="r1",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    assert rep.refusals() == []
    assert rep.publishable is True


def test_an_oracle_stage_makes_the_run_unpublishable():
    stages = _shipping_stages()
    stages["ocr"] = _stage("ocr", is_oracle=True)
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="r2",
        stages=stages, dataset_manifest_sha256="ds",
    )
    assert rep.publishable is False
    assert any("oracle" in reason for reason in rep.refusals())


def test_a_non_shipping_stage_makes_the_run_unpublishable():
    stages = _shipping_stages()
    stages["detect"] = _stage("detect", ships=False)
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="r3",
        stages=stages, dataset_manifest_sha256="ds",
    )
    assert rep.publishable is False
    assert any("does not ship" in reason for reason in rep.refusals())


def test_no_stage_identities_is_a_refusal():
    """Nothing attests that the models measured are the models that ship."""
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="r4", dataset_manifest_sha256="ds"
    )
    assert rep.publishable is False


def test_a_missing_dataset_manifest_is_a_refusal():
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="r5", stages=_shipping_stages()
    )
    assert any("dataset manifest" in reason for reason in rep.refusals())


def test_zero_eligible_vehicles_is_a_refusal():
    empty = M.ScoreCard.score([], [], link=_link([]))
    rep = M.BenchmarkReport.build(
        empty, None, task="e2e", run_id="r6",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    assert any("no eligible" in reason for reason in rep.refusals())


def test_an_unpublishable_report_nulls_the_headline_and_demotes_the_figure_to_a_note():
    """The number still exists -- it is useful for debugging -- but it is not in the headline
    field and it is labelled diagnostic-only, so a screenshot two days later cannot misread
    it as a model claim."""
    stages = _shipping_stages()
    stages["ocr"] = _stage("ocr", is_oracle=True)
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="r7",
        stages=stages, dataset_manifest_sha256="ds",
    )
    d = rep.to_dict()
    assert d["e2e_correct_plate_event_rate"] is None
    assert all(v is None for v in d["by_plate_width"].values())
    assert all(v is None for v in d["by_condition"].values())
    assert any("NOT PUBLISHABLE" in note for note in d["notes"])
    assert any("diagnostic-only" in note for note in d["notes"])


def test_a_publishable_report_carries_the_real_headline():
    rep = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="r8",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    assert rep.to_dict()["e2e_correct_plate_event_rate"] == 1.0


def test_write_sends_a_publishable_run_to_reports_and_the_rest_to_diagnostics():
    """The destination carries the meaning: a file outside benchmark/reports/ cannot be
    misread as a leaderboard result, and the full figure is still in runs/diagnostics/."""
    pub = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="pub",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    oracle_stages = _shipping_stages()
    oracle_stages["ocr"] = _stage("ocr", is_oracle=True)
    diag = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="diag",
        stages=oracle_stages, dataset_manifest_sha256="ds",
    )
    with tempfile.TemporaryDirectory() as tmp:
        p_pub = pub.write(root=tmp)
        p_diag = diag.write(root=tmp)
        assert M.REPORT_DIR in str(p_pub).replace("\\", "/")
        assert M.DIAGNOSTIC_DIR in str(p_diag).replace("\\", "/")
        # the diagnostic file carries the full scorecard, precisely because it is not a claim
        payload = json.loads(Path(p_diag).read_text(encoding="utf-8"))
        assert payload["scorecard"] is not None
        assert "counters" in payload


def test_append_leaderboard_refuses_an_unpublishable_run():
    """The leaderboard is the file the submission claim is read off; a row on it is a claim
    whether or not anybody meant it as one."""
    oracle_stages = _shipping_stages()
    oracle_stages["ocr"] = _stage("ocr", is_oracle=True)
    pub = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="pub",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    diag = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="diag",
        stages=oracle_stages, dataset_manifest_sha256="ds",
    )
    with tempfile.TemporaryDirectory() as tmp:
        assert diag.append_leaderboard(root=tmp) is None
        appended = pub.append_leaderboard(root=tmp)
        assert appended is not None
        text = Path(appended).read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("run_id")  # header written once


def test_small_plate_recall_pools_the_two_bottom_buckets():
    """The contract asks for one number for 'small plates', and B6 (<30) alone is often n=0 on
    a clip -- reporting it alone would publish null for the diagnostic that matters most. So
    <30 and 30-40 are pooled: one correct of two below 40 px is 0.5."""
    gts = [_gt("s1", width=25), _gt("s2", width=35)]
    evs = [_ev("s1", "GJ01AB1234"), _ev("s2", "WRONG0000")]
    card = M.ScoreCard.score(gts, evs, link=_link(["s1", "s2"]))
    rep = M.BenchmarkReport.build(
        card, None, task="e2e", run_id="sp",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    assert rep._diagnostics()["small_plate_recall"] == 0.5


# ============================================================================ TIER 2
# width_bucket -- an off-by-one moves samples between the small buckets that decide the claim.
# ============================================================================


@pytest.mark.parametrize(
    "width,expected",
    [
        (None, "<30"), (0, "<30"), (29, "<30"),
        (30, "30-40"), (39, "30-40"),
        (40, "40-60"), (59, "40-60"),
        (60, "60-80"), (79, "60-80"),
        (80, "80-100"), (100, "80-100"),
        (101, ">100"), (500, ">100"),
    ],
)
def test_width_bucket_boundaries(width, expected):
    """Lower-inclusive, upper-exclusive everywhere except the strict '>100'. 80 is B2, 100 is
    B2, 101 is B1. None/0 land in the hardest bucket -- a plate never located has no width and
    excluding it would remove the misses from exactly the bucket where misses happen."""
    assert M.width_bucket(width) == expected


# ============================================================================ TIER 2
# BucketTally arithmetic.
# ============================================================================


def test_bucket_tally_rate_is_none_only_when_empty():
    t = M.BucketTally()
    assert t.rate is None
    assert t.fabrication_rate is None
    t.add(M.OUTCOME_MISSED)
    assert t.rate == 0.0  # now populated and genuinely zero


def test_bucket_tally_counts_each_outcome_into_its_own_field():
    t = M.BucketTally()
    for outcome in (
        M.OUTCOME_CORRECT, M.OUTCOME_CORRECT, M.OUTCOME_FABRICATED,
        M.OUTCOME_CONTRADICTED, M.OUTCOME_MISSED, M.OUTCOME_ABSTAINED,
    ):
        t.add(outcome)
    assert (t.eligible, t.correct, t.fabricated, t.contradicted, t.missed, t.abstained) == (
        6, 2, 1, 1, 1, 1,
    )
    assert t.rate == pytest.approx(2 / 6)
    # fabrication pools fabricated + contradicted, deliberately not the complement of rate
    assert t.fabrication_rate == pytest.approx(2 / 6)


def test_bucket_tally_rejects_an_unknown_outcome():
    """Reachable only by a direct caller -- _outcome_for only ever returns the five OUTCOMES --
    but it raises rather than silently miscounting."""
    with pytest.raises(ValueError, match="unknown outcome"):
        M.BucketTally().add("banana")


def test_bucket_tally_to_dict_rounds_the_rate():
    t = M.BucketTally()
    for _ in range(2):
        t.add(M.OUTCOME_CORRECT)
    for _ in range(4):
        t.add(M.OUTCOME_MISSED)
    assert t.to_dict()["rate"] == round(2 / 6, 4)


# ============================================================================ TIER 2
# _normalize_for_comparison -- forgiving on purpose, so a CSV space is not a wrong read.
# ============================================================================


def test_normalize_for_comparison_is_transcription_forgiving():
    assert M._normalize_for_comparison("GJ 01 AB 1234") == "GJ01AB1234"
    assert M._normalize_for_comparison("gj-01-ab-1234") == "GJ01AB1234"
    assert M._normalize_for_comparison(None) == ""
    assert M._normalize_for_comparison("") == ""


def test_a_spacing_style_difference_is_scored_correct_not_wrong():
    """Ground truth is written by hand; 'GJ 01 AB 1234' vs 'GJ01AB1234' is a style, not an
    error. Being strict here would score a correct pipeline as wrong over a space in a CSV."""
    outcome, _ = M._outcome_for(
        _gt("v", plate="GJ 01 AB 1234"), [_ev("v", "GJ01AB1234")]
    )
    assert outcome == M.OUTCOME_CORRECT


# ============================================================================ TIER 2
# _percentile -- hand-rolled so importing metrics costs nothing on the worker's startup path.
# ============================================================================


def test_percentile_edges_and_interpolation():
    assert M._percentile([], 0.5) is None
    assert M._percentile([42.0], 0.95) == 42.0  # single sample is its own percentile
    assert M._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert M._percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0  # p0 == min
    assert M._percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0  # p100 == max
    assert M._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


# ============================================================================ TIER 2
# RunCounters -- warm-up touches latency and fps, never accuracy (they are separate objects).
# ============================================================================


def test_warmup_frames_are_discarded_from_latency_and_the_caller_is_told():
    """observe_frame returns False for a discarded frame so a caller can log 'frame 0
    discarded (warm-up)' rather than leaving a run whose frame count and latency sample size
    silently disagree."""
    rc = M.RunCounters(warmup_frames=2)
    assert rc.warming_up is True
    assert rc.observe_frame(latency_ms=100.0) is False  # frame 0, slow, discarded
    assert rc.observe_frame(latency_ms=100.0) is False  # frame 1, discarded
    assert rc.warming_up is False
    assert rc.observe_frame(latency_ms=10.0) is True  # counted
    assert rc.observe_frame(latency_ms=20.0) is True
    assert rc.frames_discarded_warmup == 2
    assert rc.frames_sampled == 4
    assert rc.frame_ms == [10.0, 20.0]  # only the counted frames land in the sample
    assert rc.latency_p50_ms == 15.0


def test_observe_stage_obeys_the_same_warmup_rule_as_the_frame_total():
    rc = M.RunCounters(warmup_frames=1)
    rc.observe_frame(latency_ms=1.0)  # warm-up
    rc.observe_stage("detect", 5.0)  # ignored: still warming
    rc.observe_frame(latency_ms=1.0)  # counted
    rc.observe_stage("detect", 7.0)  # recorded
    assert rc.stage_ms["detect"] == [7.0]


def test_stream_seconds_comes_from_pts_and_is_none_without_it():
    rc = M.RunCounters(warmup_frames=0)
    rc.observe_frame(latency_ms=1.0, pts_ms=1000)
    rc.observe_frame(latency_ms=1.0, pts_ms=3000)
    assert rc.first_pts_ms == 1000 and rc.last_pts_ms == 3000
    assert rc.stream_seconds == 2.0
    assert M.RunCounters(warmup_frames=0).stream_seconds is None


def test_fps_is_measured_from_the_first_counted_frame_not_the_whole_run():
    """fps and the latency percentiles must describe the same frames. Dividing warm-up-excluded
    frames by warm-up-included time skews low, because the discarded frames are the slowest and
    their time would land in the denominator while their frames do not land in the numerator."""
    rc = M.RunCounters(warmup_frames=0)
    rc.start()
    rc.observe_frame(latency_ms=1.0, pts_ms=0)
    time.sleep(0.01)
    rc.observe_frame(latency_ms=1.0, pts_ms=1000)
    rc.stop()
    assert rc.measured_seconds is not None and rc.measured_seconds > 0
    assert rc.fps is not None and rc.fps > 0
    assert rc.real_time_factor is not None


def test_derived_timings_are_none_before_a_run_starts():
    rc = M.RunCounters()
    assert rc.wall_seconds is None
    assert rc.measured_seconds is None
    assert rc.fps is None
    assert rc.real_time_factor is None


def test_located_but_unread_rate_divides_by_tracks_not_crops():
    """Against tracks_with_plate_crops, never crops_offered: a vehicle in frame for seconds
    offers dozens of crops and has a few read, so dividing the per-track numerator by the
    per-crop denominator would report a stage that failed on every vehicle as a few-percent
    shortfall."""
    rc = M.RunCounters()
    assert rc.located_but_unread_rate is None  # nothing offered a crop yet
    rc.tracks_with_plate_crops = 10
    rc.plate_located_no_read = 2
    assert rc.located_but_unread_rate == pytest.approx(0.2)


# ============================================================================ TIER 2
# _weights_sha256 -- one field, one deterministic answer, per task.
# ============================================================================


def test_single_stage_task_uses_that_stages_own_hash():
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="ocr", run_id="w1",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    assert rep._weights_sha256() == "o"  # the ocr stage's hash, verifiable against the file
    assert rep._weights_note() is None


def test_multi_stage_task_uses_a_composite_hash_and_spells_it_out():
    """e2e has three checkpoints and one field, so the value is a hash of the per-stage hashes
    -- deterministic, and explicitly not matchable against any single file, with the components
    named in a note so nobody tries."""
    rep = M.BenchmarkReport.build(
        _good_card(), None, task="e2e", run_id="w2",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    composite = rep._weights_sha256()
    assert isinstance(composite, str) and len(composite) == 64
    assert composite not in {"d", "p", "o"}
    assert "composite" in rep._weights_note()


# ============================================================================ TIER 3
# Rendering and provenance -- shape and non-crashiness, no claim rides on these.
# ============================================================================


def test_format_table_always_prints_all_six_buckets_including_empty_ones():
    """An empty row reads n=0, never omitted -- a missing row reads as 'nothing to see there',
    the opposite of what it means."""
    card = M.ScoreCard.score([_gt("a", width=90)], [_ev("a", "GJ01AB1234")], link=_link(["a"]))
    table = card.format_table()
    for key in M.BUCKET_KEYS:
        assert key in table
    assert "ALL" in table
    assert "NO COVERAGE" in table  # the empty buckets are flagged, not hidden


def test_format_table_on_an_empty_card_still_shows_every_bucket():
    table = M.ScoreCard.score([], [], link=_link([])).format_table()
    for key in M.BUCKET_KEYS:
        assert key in table


def test_scorecard_to_dict_has_the_locked_shape():
    card = M.ScoreCard.score([_gt("a", width=90)], [_ev("a", "GJ01AB1234")], link=_link(["a"]))
    d = card.to_dict()
    assert d["task"] == "e2e"
    assert d["rate"] == 1.0
    assert set(d["by_plate_width"].keys()) == set(M.BUCKET_KEYS)
    assert set(d["by_condition"].keys()) == set(M.CONDITIONS)
    assert isinstance(d["coverage_gaps"], list)


def test_report_to_dict_key_order_matches_the_contract_document():
    rep = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="k1",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    keys = list(rep.to_dict().keys())
    assert keys[0] == "run_id"
    assert keys[1] == "task"
    assert "e2e_correct_plate_event_rate" in keys
    assert "diagnostics" in keys and "notes" in keys


def test_diagnostics_carries_every_locked_key_even_when_unmeasured():
    """A consumer reading these must not have to test for presence; unmeasured is null, absent
    is never allowed."""
    rep = M.BenchmarkReport.build(None, None, task="e2e", run_id="k2")
    diag = rep.to_dict()["diagnostics"]
    for key in (
        "precision", "recall", "map50", "small_plate_recall", "ocr_exact_accuracy",
        "cer", "fps", "latency_p50_ms", "latency_p95_ms", "vram_peak_mb", "real_time_factor",
    ):
        assert key in diag


def test_runcounters_to_dict_and_stage_table():
    rc = M.RunCounters(warmup_frames=0)
    rc.observe_frame(latency_ms=10.0, pts_ms=0)
    rc.observe_stage("detect", 5.0)
    d = rc.to_dict()
    assert "fps" in d and "located_but_unread_rate" in d
    assert d["stages"]["detect"]["n"] == 1
    assert "detect" in rc.stage_table()
    assert "no stage timings" in M.RunCounters().stage_table()


def test_stage_identity_from_stage_and_to_dict():
    class _Stage:
        model_name = "rfdetr-small"
        model_version = "1.0"
        ships = True
        is_oracle = False
        weights_sha256 = "abc"

    si = M.StageIdentity.from_stage("detect", _Stage())
    assert si.model_name == "rfdetr-small"
    assert si.to_dict()["model"] == "rfdetr-small@1.0"
    # a bare identity renders an empty model rather than a stray '@'
    assert M.StageIdentity(name="x").to_dict()["model"] == ""


def test_provenance_helpers_return_sane_types_without_a_gpu_or_network():
    assert M.git_commit() is None or isinstance(M.git_commit(), str)
    assert isinstance(M.git_is_dirty(), (bool, type(None)))
    assert M.sha256_file("this-file-does-not-exist.bin") is None
    assert isinstance(M.machine_description(), str)
    assert isinstance(M.runtime_description(), str)
    assert M.vram_peak_mb() is None or isinstance(M.vram_peak_mb(), float)


def test_sha256_file_hashes_a_real_file(tmp_path):
    from hashlib import sha256

    p = tmp_path / "weights.bin"
    p.write_bytes(b"synthetic-checkpoint-bytes")
    assert M.sha256_file(str(p)) == sha256(b"synthetic-checkpoint-bytes").hexdigest()


def test_safe_makes_a_filename_windows_accepts():
    assert M._safe("a/b:c*d") == "a_b_c_d"
    assert M._safe("///") == "run"  # never empty


def test_format_summary_names_the_run_and_survives_missing_pieces():
    rep = M.BenchmarkReport.build(
        _good_card(), M.RunCounters(), task="e2e", run_id="sum1",
        stages=_shipping_stages(), dataset_manifest_sha256="ds",
    )
    summary = rep.format_summary()
    assert "sum1" in summary
    # a report with neither scorecard nor counters still renders a line, not a crash
    assert "bare" in M.BenchmarkReport.build(None, None, task="e2e", run_id="bare").format_summary()
