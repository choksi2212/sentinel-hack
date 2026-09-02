"""The locked plate quality score and its four components. Contracts 4.1.

    score = 0.30 * sharpness + 0.25 * resolution + 0.25 * detector_confidence
          + 0.20 * exposure

One function, two uses: it ranks crops for the top-K buffer AND it is the temporal-fusion
weight. That is why the weights are asserted here as data rather than trusted. A silent
reweighting would change every fused plate in the project without touching another line of
code, and every accuracy number computed before the change would become uncomparable with
every number after it -- with nothing in the diff to say so.

The other thing this file pins is the width-vs-quality relationship, because Contracts 7.2
forbids reporting accuracy as a single average and requires it bucketed by plate width. If
resolution stopped tracking width, the buckets would still be produced and would still look
plausible while measuring nothing.
"""

import numpy as np
import pytest

from ai.quality.metrics import (
    CLIP_HIGH,
    CLIP_LOW,
    EXPOSURE_CLIP_TOLERANCE,
    RESOLUTION_REFERENCE_PX,
    SHARPNESS_REFERENCE,
    contrast_norm,
    exposure_norm,
    laplacian_variance,
    resolution_norm,
    sharpness_norm,
    to_gray,
)
from ai.quality.score import (
    QUALITY_WEIGHTS,
    plate_quality,
    plate_quality_breakdown,
    rank_crops,
)


def flat(value: int, h: int = 24, w: int = 62) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def checkerboard(h: int = 24, w: int = 62, high: int = 200, low: int = 40) -> np.ndarray:
    """A maximally busy crop -- the sharpest thing an 8-bit image can be."""
    ys, xs = np.mgrid[0:h, 0:w]
    plane = np.where((ys + xs) % 2 == 0, high, low).astype(np.uint8)
    return np.stack([plane] * 3, axis=2)


def plate_like(h: int = 24, w: int = 62) -> np.ndarray:
    """Dark bars on a light ground, roughly what a plate crop looks like."""
    img = np.full((h, w, 3), 210, dtype=np.uint8)
    for start in range(4, w - 4, 8):
        img[4 : h - 4, start : start + 3] = 30
    return img


# ------------------------------------------------------------------------- locked weights


def test_the_four_weights_are_the_contract_values():
    assert QUALITY_WEIGHTS == {
        "sharpness": 0.30,
        "resolution": 0.25,
        "detector_confidence": 0.25,
        "exposure": 0.20,
    }


def test_the_weights_sum_to_one():
    """Otherwise a perfect crop cannot score 1.0, or an ordinary one can exceed it.

    Both failure modes are quiet: the score is clamped, so an over-unity sum shows up as
    every decent crop saturating at 1.0 and the ranking losing all resolution.
    """
    assert sum(QUALITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_sharpness_is_the_heaviest_component():
    """Deliberate. A blurred plate is unreadable at any size, and motion blur is the
    dominant failure on a 25 fps traffic camera."""
    assert QUALITY_WEIGHTS["sharpness"] == max(QUALITY_WEIGHTS.values())


def test_the_score_is_the_weighted_sum_of_the_reported_components():
    """The breakdown must reconstruct the score, or the failure taxonomy lies.

    A crop rejected at 0.31 is only actionable if the four numbers shown alongside it are
    the four numbers that produced it.
    """
    crop = plate_like()
    parts = plate_quality_breakdown(crop, plate_width_px=62, detector_confidence=0.81)
    recomputed = sum(QUALITY_WEIGHTS[k] * parts[k] for k in QUALITY_WEIGHTS)
    assert parts["score"] == pytest.approx(recomputed)
    assert plate_quality(crop, plate_width_px=62, detector_confidence=0.81) == pytest.approx(
        parts["score"]
    )


def test_the_score_is_always_in_the_unit_interval():
    """It is stored on the event and the event schema constrains the column."""
    cases = [
        (checkerboard(), 999, 1.0),
        (flat(0), 0, 0.0),
        (flat(255), 10_000, 1.0),
        (plate_like(), 62, 0.5),
        (np.zeros((3, 3, 3), dtype=np.uint8), 3, 0.5),
    ]
    for crop, width, confidence in cases:
        score = plate_quality(crop, plate_width_px=width, detector_confidence=confidence)
        assert 0.0 <= score <= 1.0, (score, width, confidence)


def test_detector_confidence_is_clamped_not_trusted():
    """A detector reporting 1.4 must not push the score over 1.0.

    Not hypothetical -- a logit that skipped its sigmoid, or a scripted stage in a test,
    both produce out-of-range confidences, and the failure would surface as a CHECK
    constraint violation at ingest rather than here.
    """
    crop = plate_like()
    assert plate_quality(crop, plate_width_px=62, detector_confidence=1.4) <= 1.0
    assert plate_quality(crop, plate_width_px=62, detector_confidence=-0.5) == pytest.approx(
        plate_quality(crop, plate_width_px=62, detector_confidence=0.0)
    )
    parts = plate_quality_breakdown(crop, detector_confidence=1.4)
    assert parts["detector_confidence"] == 1.0


def test_an_empty_crop_scores_zero_rather_than_raising():
    """A zero-area crop happens: a plate box clipped entirely outside the frame.

    Scoring it 0.0 keeps it out of every top-K slot, which is the outcome wanted. Raising
    would drop the whole frame over one bad box.
    """
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert plate_quality(empty) == 0.0
    assert plate_quality_breakdown(empty)["score"] == 0.0


def test_the_breakdown_of_an_empty_crop_has_every_key():
    """Callers write these four values into the taxonomy unconditionally."""
    parts = plate_quality_breakdown(np.zeros((0, 0, 3), dtype=np.uint8))
    assert set(parts) == set(QUALITY_WEIGHTS) | {"score"}


# ---------------------------------------------------------------------------- resolution


def test_the_resolution_reference_is_100_px():
    """Contracts 4.1: "plate pixel width vs 100px reference"."""
    assert RESOLUTION_REFERENCE_PX == 100.0
    assert resolution_norm(100) == 1.0


def test_resolution_tracks_width_across_the_reporting_buckets():
    """Contracts 7.2 buckets accuracy by plate width. Those buckets have to mean something.

    30 px is the width that decides whether this works on real infrastructure, and it must
    score materially lower than 100 px rather than both saturating.
    """
    widths = [10, 30, 40, 60, 80, 100]
    scores = [resolution_norm(w) for w in widths]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores), "no two buckets may score the same"
    assert resolution_norm(30) == pytest.approx(0.3)


def test_resolution_saturates_rather_than_rewarding_a_billboard():
    """A 400 px plate is not four times as readable as a 100 px one."""
    assert resolution_norm(400) == 1.0
    assert resolution_norm(100) == resolution_norm(400)


def test_resolution_of_a_nonpositive_width_is_zero():
    assert resolution_norm(0) == 0.0
    assert resolution_norm(-5) == 0.0


def test_resolution_scores_the_frame_width_not_the_upscaled_crop():
    """The crop handed to OCR may already be 4x upscaled.

    Scoring the array would rank a 30 px plate stretched to 120 px above a real 120 px
    plate -- precisely backwards, and invisible unless someone thinks to check.
    """
    upscaled = plate_like(h=96, w=248)
    honest = plate_quality(upscaled, plate_width_px=62, detector_confidence=0.8)
    naive = plate_quality(upscaled, detector_confidence=0.8)
    assert honest < naive
    assert plate_quality_breakdown(upscaled, plate_width_px=62)["resolution"] == pytest.approx(0.62)


# ----------------------------------------------------------------------------- sharpness


def test_a_flat_crop_has_no_sharpness():
    """No edges, so nothing to be sharp. The honest answer is zero."""
    assert laplacian_variance(flat(128)) == 0.0
    assert sharpness_norm(flat(128)) == 0.0


def test_a_busy_crop_saturates_sharpness():
    assert sharpness_norm(checkerboard()) == 1.0


def test_blurring_reduces_sharpness():
    """The property the component exists for, tested with a real blur.

    A 3x3 box blur on a checkerboard destroys most of the high-frequency content, which is
    what motion blur on a 25 fps camera does to a moving plate.
    """
    sharp = checkerboard().astype(np.float32)
    blurred = sharp.copy()
    for _ in range(2):
        padded = np.pad(blurred, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blurred = sum(
            padded[dy : dy + blurred.shape[0], dx : dx + blurred.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ) / 9.0
    assert sharpness_norm(blurred.astype(np.uint8)) < sharpness_norm(sharp.astype(np.uint8))


def test_a_crop_too_small_to_have_an_interior_has_no_measurable_sharpness():
    """A 2 px tall crop has no interior pixels, so the Laplacian is undefined on it.

    Returning zero keeps it out of the top-K, which is right: returning something nonzero
    would let a 2 px sliver compete for a slot against a real plate.
    """
    assert laplacian_variance(np.full((2, 40, 3), 128, dtype=np.uint8)) == 0.0
    assert laplacian_variance(np.full((40, 2, 3), 128, dtype=np.uint8)) == 0.0
    assert laplacian_variance(np.full((3, 3, 3), 128, dtype=np.uint8)) == 0.0


def test_sharpness_saturates_at_the_reference():
    """Otherwise one very sharp crop wins the ranking on sharpness alone."""
    assert SHARPNESS_REFERENCE == 500.0
    assert sharpness_norm(checkerboard(high=255, low=0)) == 1.0
    assert sharpness_norm(checkerboard(high=200, low=40)) == 1.0


# ------------------------------------------------------------------------------ exposure


def test_a_well_exposed_crop_scores_one():
    assert exposure_norm(flat(128)) == 1.0
    assert exposure_norm(plate_like()) == 1.0


def test_a_blown_out_crop_scores_zero():
    """Direct afternoon glare on the Sentinel feeds. The characters are gone."""
    assert exposure_norm(flat(255)) == 0.0
    assert exposure_norm(flat(CLIP_HIGH)) == 0.0


def test_a_crushed_crop_scores_zero():
    """A plate under a sodium lamp at night, crushed to black.

    Measured at both ends deliberately -- an exposure metric that only caught highlights
    would rate every night crop as perfect.
    """
    assert exposure_norm(flat(0)) == 0.0
    assert exposure_norm(flat(CLIP_LOW)) == 0.0


def test_exposure_degrades_smoothly_up_to_the_tolerance():
    """A quarter clipped is the zero point, and everything below it is graded."""
    assert EXPOSURE_CLIP_TOLERANCE == 0.25
    h, w = 20, 20
    previous = 1.0
    for clipped_rows in (0, 1, 2, 3, 4, 5, 10):
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        img[:clipped_rows] = 255
        score = exposure_norm(img)
        assert score <= previous
        previous = score
        expected = max(0.0, 1.0 - (clipped_rows / h) / EXPOSURE_CLIP_TOLERANCE)
        assert score == pytest.approx(expected)
    assert previous == 0.0


def test_exposure_of_an_empty_crop_is_zero():
    assert exposure_norm(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


# -------------------------------------------------------------------------------- to_gray


def test_to_gray_uses_bt601_in_bgr_order():
    """Matching cv2.cvtColor(COLOR_BGR2GRAY), so a crop scored here and a crop scored in a
    notebook with OpenCV agree. Getting the channel order backwards is silent: it produces
    a plausible grayscale image with the red and blue contributions swapped."""
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = (255, 0, 0)  # pure blue in BGR
    assert to_gray(bgr)[0, 0] == pytest.approx(0.114 * 255, abs=0.01)

    bgr[0, 0] = (0, 0, 255)  # pure red in BGR
    assert to_gray(bgr)[0, 0] == pytest.approx(0.299 * 255, abs=0.01)

    bgr[0, 0] = (0, 255, 0)
    assert to_gray(bgr)[0, 0] == pytest.approx(0.587 * 255, abs=0.01)


def test_to_gray_accepts_grayscale_and_single_channel():
    assert to_gray(np.full((4, 4), 90, dtype=np.uint8)).shape == (4, 4)
    assert to_gray(np.full((4, 4, 1), 90, dtype=np.uint8)).shape == (4, 4)


def test_to_gray_rejects_an_unexpected_shape():
    """A 4-channel BGRA array is the realistic mistake, and averaging alpha into luma would
    produce a quietly wrong score rather than an error."""
    with pytest.raises(ValueError, match="expected HxW or HxWx3"):
        to_gray(np.zeros((4, 4, 4), dtype=np.uint8))


def test_contrast_is_a_diagnostic_and_not_in_the_score():
    """Present for the failure taxonomy, absent from the locked weights."""
    assert "contrast" not in QUALITY_WEIGHTS
    assert contrast_norm(flat(128)) == 0.0
    assert contrast_norm(checkerboard()) > 0.0
    assert 0.0 <= contrast_norm(plate_like()) <= 1.0
    assert contrast_norm(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


# ------------------------------------------------------------------------- crop ranking


def test_rank_crops_keeps_the_best_k_best_first():
    crops = [{"quality": q, "plate_width_px": 60} for q in (0.2, 0.9, 0.5, 0.7, 0.1)]
    ranked = rank_crops(crops, top_k=3)
    assert [c["quality"] for c in ranked] == [0.9, 0.7, 0.5]


def test_rank_crops_breaks_ties_on_width_so_a_benchmark_is_reproducible():
    """A run that reorders equal scores is not reproducible, and Contracts 7.3 will not
    publish a number that cannot be reproduced."""
    crops = [
        {"quality": 0.8, "plate_width_px": 40, "id": "narrow"},
        {"quality": 0.8, "plate_width_px": 90, "id": "wide"},
    ]
    for _ in range(10):
        assert [c["id"] for c in rank_crops(list(reversed(crops)), top_k=2)] == ["wide", "narrow"]


def test_rank_crops_tolerates_missing_keys():
    """A crop that never got scored ranks last rather than crashing the frame."""
    ranked = rank_crops([{"quality": 0.5}, {}, {"plate_width_px": 80}], top_k=3)
    assert ranked[0]["quality"] == 0.5


def test_rank_crops_refuses_a_zero_top_k():
    """top_k=0 reads every frame and reports no plates -- an expensive way to do nothing."""
    with pytest.raises(ValueError, match="top_k"):
        rank_crops([{"quality": 0.5}], top_k=0)


def test_rank_crops_does_not_mutate_its_input():
    """The caller's buffer is the evidence. Sorting it in place is a surprise."""
    crops = [{"quality": 0.2}, {"quality": 0.9}]
    snapshot = [dict(c) for c in crops]
    rank_crops(crops, top_k=2)
    assert crops == snapshot


# ------------------------------------------------------------- the ordering that matters


def test_a_sharp_wide_well_exposed_crop_outranks_every_degradation_of_it():
    """The end-to-end property. Each degradation must cost something.

    If any single degradation were free, the top-K buffer would be filling with crops that
    OCR cannot read while a readable one was evicted.
    """
    good = plate_quality(checkerboard(), plate_width_px=100, detector_confidence=0.95)
    assert good > plate_quality(checkerboard(), plate_width_px=30, detector_confidence=0.95)
    assert good > plate_quality(checkerboard(), plate_width_px=100, detector_confidence=0.40)
    assert good > plate_quality(flat(128), plate_width_px=100, detector_confidence=0.95)
    assert good > plate_quality(flat(255), plate_width_px=100, detector_confidence=0.95)


def test_a_narrow_plate_can_still_beat_a_blurred_wide_one():
    """Width alone does not decide it, and that is the point of a weighted score.

    A 40 px sharp plate is often readable where a 100 px smear is not, so the ranking has
    to be able to prefer it.
    """
    narrow_sharp = plate_quality(checkerboard(w=40), plate_width_px=40, detector_confidence=0.9)
    wide_flat = plate_quality(flat(128, w=100), plate_width_px=100, detector_confidence=0.9)
    assert narrow_sharp > wide_flat
