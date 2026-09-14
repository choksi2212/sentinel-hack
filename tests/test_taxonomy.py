"""The failure taxonomy -- owner's manual section 6, the deliverable that gates
the A100.

Two things are pinned here that the rest of the project leans on:

  1. The ten buckets are exactly the manual's ten, in the manual's order, and
     exactly one of them (`ocr_wrong`) is the miss this lane's fine-tune could
     fix. If a bucket is renamed, added, or its remedy re-pointed, a test here
     fails rather than the deliverable quietly drifting from the manual.

  2. The verdict is conservative by construction. Naming one dominant bucket
     unlocks a paid run, so the logic refuses to name one on a tie, on too few
     misses, or on a leader that is not clearly ahead -- because "the analysis
     is not finished" is the correct and cheap answer, and a wrong "train this"
     is a guess with a credit card attached.
"""

import json

import pytest

from ai.quality.taxonomy import (
    BUCKETS_BY_KEY,
    CO_DOMINANT,
    DEFAULT_MARGIN,
    DEFAULT_MIN_SAMPLE,
    DOMINANT,
    FAILURE_BUCKETS,
    INSUFFICIENT,
    POINTS_AT_NO_SOFTWARE_FIX,
    POINTS_AT_OCR_FINETUNE,
    POINTS_AT_OTHER_LANE,
    FailureTaxonomy,
)

# The manual's section-6 table, in order. Copied here so the test is the manual
# and a drift in either direction is a failure, not a silent divergence.
MANUAL_BUCKETS = (
    "vehicle_miss",
    "plate_miss",
    "plate_too_small",
    "ocr_wrong",
    "ocr_partial",
    "fusion_wrong",
    "track_broken",
    "track_merged",
    "duplicate",
    "dropped_frame",
)


# --------------------------------------------------------------- the ten buckets


def test_the_buckets_are_exactly_the_manuals_ten_in_order():
    """Order matters: the report renders in this order, so a reordering reads as
    a change to the deliverable when nothing changed."""
    assert tuple(b.key for b in FAILURE_BUCKETS) == MANUAL_BUCKETS


def test_there_are_ten_buckets_and_no_more():
    """A eleventh bucket would split the histogram and could turn a real dominant
    bucket into an apparent tie -- the exact false 'not finished' that stops a
    justified run."""
    assert len(FAILURE_BUCKETS) == 10
    assert len(BUCKETS_BY_KEY) == 10


def test_every_bucket_has_a_symptom_and_a_remedy():
    for bucket in FAILURE_BUCKETS:
        assert bucket.symptom.strip(), f"{bucket.key} has no symptom"
        assert bucket.remedy.strip(), f"{bucket.key} has no remedy"


def test_exactly_one_bucket_points_at_this_lanes_fine_tune_and_it_is_ocr_wrong():
    """The whole reason the verdict is more than argmax: only `ocr_wrong` is a
    miss the OCR recogniser fine-tune could fix, so only it may unlock the A100."""
    pointing = [b.key for b in FAILURE_BUCKETS if b.points_at == POINTS_AT_OCR_FINETUNE]
    assert pointing == ["ocr_wrong"]


def test_exactly_one_bucket_has_no_software_fix_and_it_is_plate_too_small():
    """`plate_too_small` is a legitimate finding with no code remedy. If it were
    mislabelled as fixable, a dominant `plate_too_small` would wrongly justify a
    training run instead of a camera-placement recommendation."""
    no_fix = [b.key for b in FAILURE_BUCKETS if b.points_at == POINTS_AT_NO_SOFTWARE_FIX]
    assert no_fix == ["plate_too_small"]


def test_ocr_wrong_remedy_is_training_and_plate_too_small_remedy_is_not_software():
    assert "OCR training" in BUCKETS_BY_KEY["ocr_wrong"].remedy
    assert "Nothing in software" in BUCKETS_BY_KEY["plate_too_small"].remedy


def test_ocr_partial_points_elsewhere_not_at_the_fine_tune():
    """Uncomfortable but load-bearing: partly-right OCR is a temporal-consensus
    problem (fusion, more frames), not a reason to spend the budget on the
    recogniser. If this ever re-points at the fine-tune, a dominant `ocr_partial`
    would open the gate on the wrong evidence."""
    assert BUCKETS_BY_KEY["ocr_partial"].points_at == POINTS_AT_OTHER_LANE


def test_every_points_at_value_is_one_of_the_three_known_classes():
    known = {POINTS_AT_OCR_FINETUNE, POINTS_AT_NO_SOFTWARE_FIX, POINTS_AT_OTHER_LANE}
    assert all(b.points_at in known for b in FAILURE_BUCKETS)


# --------------------------------------------------------------- recording misses


def test_recording_an_unknown_bucket_is_a_hard_error():
    tax = FailureTaxonomy()
    with pytest.raises(ValueError, match="unknown failure bucket"):
        tax.record("ocr_broken")  # not a real bucket


def test_a_negative_count_is_refused():
    tax = FailureTaxonomy()
    with pytest.raises(ValueError, match="count must be"):
        tax.record("ocr_wrong", -1)


def test_recording_accumulates_and_totals():
    tax = FailureTaxonomy()
    tax.record("ocr_wrong")
    tax.record("ocr_wrong", 4)
    tax.record("plate_miss", 2)
    assert tax.counts["ocr_wrong"] == 5
    assert tax.counts["plate_miss"] == 2
    assert tax.total == 7


def test_record_many_takes_a_whole_histogram():
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 10, "plate_too_small": 3})
    assert tax.total == 13
    assert tax.counts["ocr_wrong"] == 10


def test_shares_are_fractions_of_the_total():
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 3, "plate_miss": 1})
    shares = tax.shares()
    assert shares["ocr_wrong"] == pytest.approx(0.75)
    assert shares["plate_miss"] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_shares_are_empty_before_anything_is_recorded():
    """Not ten zeros. A share of a total of zero is undefined, and 0.0 across the
    board would read as ten equal findings rather than as no data."""
    assert FailureTaxonomy().shares() == {}


# ------------------------------------------------------------------- the verdict


def test_too_few_misses_is_insufficient_not_a_guess():
    tax = FailureTaxonomy()
    tax.record("ocr_wrong", DEFAULT_MIN_SAMPLE - 1)
    verdict = tax.verdict()
    assert verdict.status == INSUFFICIENT
    assert verdict.dominant is None
    assert verdict.unlocks_ocr_finetune is False
    assert str(DEFAULT_MIN_SAMPLE) in verdict.recommendation


def test_a_clear_ocr_wrong_leader_is_dominant_and_unlocks_the_fine_tune():
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 40, "plate_too_small": 8, "vehicle_miss": 5, "ocr_partial": 4})
    verdict = tax.verdict()
    assert verdict.status == DOMINANT
    assert verdict.dominant == "ocr_wrong"
    assert verdict.points_at == POINTS_AT_OCR_FINETUNE
    assert verdict.unlocks_ocr_finetune is True
    assert verdict.leader_share == pytest.approx(40 / 57)
    assert "config/training.yaml" in verdict.recommendation


def test_a_dominant_plate_too_small_unlocks_nothing_and_recommends_camera_placement():
    tax = FailureTaxonomy()
    tax.record_many({"plate_too_small": 35, "ocr_wrong": 6, "vehicle_miss": 4})
    verdict = tax.verdict()
    assert verdict.status == DOMINANT
    assert verdict.dominant == "plate_too_small"
    assert verdict.points_at == POINTS_AT_NO_SOFTWARE_FIX
    assert verdict.unlocks_ocr_finetune is False
    assert "camera-placement" in verdict.recommendation


def test_a_dominant_other_lane_bucket_is_a_real_fix_but_not_this_lanes_a100():
    tax = FailureTaxonomy()
    tax.record_many({"vehicle_miss": 30, "plate_miss": 5, "ocr_wrong": 3})
    verdict = tax.verdict()
    assert verdict.status == DOMINANT
    assert verdict.dominant == "vehicle_miss"
    assert verdict.points_at == POINTS_AT_OTHER_LANE
    assert verdict.unlocks_ocr_finetune is False
    assert "not this lane" in verdict.recommendation


def test_a_tie_is_co_dominant_and_refuses_to_train():
    """The manual's rule: two co-dominant buckets means the analysis is not
    finished. A tie can never clear the margin, so it must land here."""
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 20, "plate_too_small": 20})
    verdict = tax.verdict()
    assert verdict.status == CO_DOMINANT
    assert set(verdict.co_dominant) == {"ocr_wrong", "plate_too_small"}
    assert verdict.dominant is None
    assert verdict.unlocks_ocr_finetune is False
    assert "not finished" in verdict.recommendation


def test_a_leader_just_short_of_the_margin_is_co_dominant():
    """24 vs 20 is not a quarter clear, so it is not a mandate."""
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 24, "plate_too_small": 20})
    verdict = tax.verdict()
    assert verdict.status == CO_DOMINANT
    assert set(verdict.co_dominant) == {"ocr_wrong", "plate_too_small"}


def test_a_leader_exactly_at_the_margin_is_dominant():
    """25 vs 20 is a quarter clear -- the boundary is inclusive so the rule has a
    definite edge rather than a floating-point coin toss."""
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 25, "plate_too_small": 20})
    verdict = tax.verdict()
    assert verdict.status == DOMINANT
    assert verdict.dominant == "ocr_wrong"


def test_a_lone_bucket_over_the_floor_is_dominant():
    """One bucket with no runner-up is as clear as it gets: the runner-up count
    is zero, and any positive leader clears a margin over zero."""
    tax = FailureTaxonomy()
    tax.record("ocr_wrong", DEFAULT_MIN_SAMPLE)
    verdict = tax.verdict()
    assert verdict.status == DOMINANT
    assert verdict.dominant == "ocr_wrong"
    assert verdict.leader_share == pytest.approx(1.0)


def test_the_floor_is_overridable_for_a_sweep():
    """Below the floor the same histogram is insufficient; lower the floor and it
    resolves -- so the floor is a knob, not a hard-coded verdict."""
    tax = FailureTaxonomy()
    tax.record("ocr_wrong", 10)
    assert tax.verdict().status == INSUFFICIENT
    assert tax.verdict(min_sample=5).status == DOMINANT


def test_the_margin_is_overridable():
    """A looser margin turns a near-tie into a dominant call; the point is that
    the threshold is explicit and reproducible, not that 1.25 is sacred."""
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 24, "plate_too_small": 20})
    assert tax.verdict().status == CO_DOMINANT
    assert tax.verdict(margin=1.1).status == DOMINANT


def test_the_default_margin_and_floor_are_what_the_docs_claim():
    assert DEFAULT_MARGIN == 1.25
    assert DEFAULT_MIN_SAMPLE == 30


def test_an_empty_taxonomy_is_insufficient():
    verdict = FailureTaxonomy().verdict()
    assert verdict.status == INSUFFICIENT
    assert verdict.total == 0


def test_the_verdict_is_json_ready():
    """The report embeds this and a run log records it, so it has to serialise
    without a custom encoder."""
    tax = FailureTaxonomy()
    tax.record_many({"ocr_wrong": 40, "plate_miss": 5})
    payload = tax.verdict().to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored["status"] == DOMINANT
    assert restored["dominant"] == "ocr_wrong"
    assert restored["unlocks_ocr_finetune"] is True
    assert restored["counts"]["ocr_wrong"] == 40
