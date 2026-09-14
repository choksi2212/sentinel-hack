"""Plate detection. Owner's manual 5.5, Contracts sections 3 and 8.

Three things are being tested here and they are not equally dangerous.

**The coordinate frame, which is the one that can ship a silent lie.** This stage runs on
a vehicle crop and Contracts section 3 requires `plate_bbox_xyxy` in full-frame
coordinates. A crop-local box that escapes is a plausible box of a plausible size: it
passes every type check, lands in the event, and produces a snapshot URI pointing at a
patch of road tens of metres from the vehicle along with a `plate_width_px` that happens
to be correct. Nothing downstream can detect it, so the only place it can be caught is
here. Several tests below therefore check the *mapping* rather than the detection, and
one of them (`test_a_backend_that_returns_frame_coordinates_is_wrong_by_the_offset`)
deliberately builds the bug to show what it looks like and why it hides near the origin.

**The measured claims in the module's own comments.** `ai/plate/stub.py` justifies four
constants with numbers. A constant defended by a number nobody re-runs is a constant
defended by a story, so the numbers are pinned: the oracle-vs-edge table, the four width
buckets, the threshold/contrast diagonal, the `col_quantile` trade, and the erosion.
Every figure in here was re-measured after five of them turned out to be wrong -- see
the tests' own docstrings, which record what the file used to claim.

Two numbers in that file are deliberately *not* pinned: 0.70 ms median and 1.00 ms p95.
They are machine-dependent, and a timing assertion in CI is a flake with a ticket
attached.

**The honesty properties.** `ships` is False on all three stub backends -- including the
edge one, whose licence is clean -- because the flag means "may appear in a published
accuracy claim". The oracle respects `plate_legible`, so it cannot beat a real detector
by reading a plate that was never visible. A missing key in the returned dict is a
first-class answer meaning "no plate", never an error.

The fixture is the one every measured claim in ai/plate is stated against: source mode
synthetic, cam01, seed 42, total_frames 200 -- which *emits* 67 frames at 120 ms. That
distinction is not pedantry; reading it as 200 emitted frames is the error that made
five of those claims unreproducible, and `test_the_fixture_is_the_one_the_claims_name`
pins the shape so the next reader cannot repeat it.
"""

import numpy as np
import pytest

from ai.contracts.stages import PlateCandidate, TrackResult
from ai.detect import build_detector
from ai.media import build_source
from ai.metrics import width_bucket
from ai.plate import (
    CROP_PAD_FRACTION,
    DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
    PLATE_ASPECT_MAX,
    PLATE_ASPECT_MIN,
    PLATE_DETECTOR_NAMES,
    SHIPPABLE_PLATE_DETECTORS,
    BasePlateDetector,
    PlateConfigError,
    aspect_ratio,
    build_plate_detector,
    crop_vehicle,
    describe_plate_detector,
    map_to_frame,
    normalize_plate_config,
    plate_detector_ships,
    plausible_plate_box,
    region_prior,
)
from ai.plate.geometry import (
    MIN_PLATE_BOX_PX,
    PLATE_REGION_TOP_FRACTION,
    best_by_confidence,
    clip_to_crop,
)
from ai.plate.stub import (
    EDGE_COL_QUANTILE,
    EDGE_GRADIENT_THRESHOLD,
    EDGE_MIN_COLS,
    EDGE_MIN_ROWS,
    EDGE_ROW_MIN_FILL,
    ORACLE_MATCH_IOU,
    EdgePlateDetector,
    ScriptedPlateDetector,
    _aspect_agreement,
    _contiguous_band,
    _iou,
    _running_max,
)
from ai.track import build_tracker

# --------------------------------------------------------------------------- fixture

SOURCE = {
    "mode": "synthetic",
    "camera_id": "cam01",
    "seed": 42,
    "total_frames": 200,
    "target_interval_ms": 120,
}
CAMERA = SOURCE["camera_id"]
HIT_IOU = 0.30

# What the fixture is, so a test can assert it rather than assume it.
EXPECTED_EMITTED_FRAMES = 67
EXPECTED_VEHICLE_FRAMES = 240
EXPECTED_LEGIBLE = 226
EXPECTED_DISTINCT_VEHICLES = 6
FRAME_W, FRAME_H = 1280, 720


class Corpus:
    """The fixture decoded, detected and tracked once, then reused.

    Building it costs about 0.7 s and one plate pass over it costs 30 ms, so a
    session-scoped cache is what makes it affordable to pin a five-cell diagonal and a
    parameter trade rather than one summary number. Perturbed copies are memoised for
    the same reason.

    Tracking runs on the *clean* frame and plate detection on the perturbed one, which
    is the protocol the measured claims were taken under: the perturbation is a
    statement about what this stage can see, not about what the tracker can follow.
    """

    def __init__(self) -> None:
        source = build_source(dict(SOURCE))
        source.open()
        detector = build_detector({"name": "oracle", "miss_rate": 0.0}, source=source)
        detector.load()
        tracker = build_tracker(
            {"name": "bytetrack"}, CAMERA, source.session_id, source=source
        )

        self.session_id = source.session_id
        self.frames: list[np.ndarray] = []
        self.tracks: list[tuple[TrackResult, ...]] = []
        self.truth: list = []
        self.envelopes: list = []

        for envelope in source:
            detections = detector.detect_envelope(envelope)
            tracks = tracker.update(
                detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
            )
            self.frames.append(envelope.frame_bgr.copy())
            self.tracks.append(tuple(tracks))
            self.truth.append(source.truth_at_pts(envelope.pts_ms))
            self.envelopes.append(envelope)

        source.close()
        detector.close()
        self._perturbed: dict[tuple[int, float], list[np.ndarray]] = {}

    def pixels(self, *, grain: int = 0, contrast: float = 1.0) -> list[np.ndarray]:
        key = (grain, contrast)
        if key in self._perturbed:
            return self._perturbed[key]
        if grain == 0 and contrast == 1.0:
            return self.frames
        rng = np.random.default_rng(7)
        out = []
        for frame in self.frames:
            work = frame
            if contrast != 1.0:
                work = np.clip(
                    (work.astype(np.float32) - 128.0) * contrast + 128.0, 0, 255
                ).astype(np.uint8)
            if grain:
                noise = rng.integers(-grain, grain + 1, size=work.shape, dtype=np.int16)
                work = np.clip(work.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            out.append(work)
        self._perturbed[key] = out
        return out

    def legible_plates(self, index: int) -> list:
        truth = self.truth[index]
        if truth is None:
            return []
        return [
            v
            for v in truth.vehicles
            if v.plate_bbox_xyxy is not None and v.plate_legible
        ]

    def plates_on_frame(self, index: int) -> list:
        truth = self.truth[index]
        if truth is None:
            return []
        return [v for v in truth.vehicles if v.plate_bbox_xyxy is not None]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus()


class Score:
    """Recall, mean IoU and precision for one plate backend over the corpus.

    Mean IoU is over every returned box, matching the figure in EdgePlateDetector's
    docstring. `iou_over_hits` is kept beside it because the two differ, and a setting
    that trades tight boxes for more of them moves them in opposite directions -- which
    is the entire col_quantile argument.
    """

    def __init__(self, corpus: Corpus, config, *, grain: int = 0, contrast: float = 1.0):
        detector = build_plate_detector(dict(config), source=_OracleSource(corpus))
        detector.load()
        pixels = corpus.pixels(grain=grain, contrast=contrast)

        self.legible = self.hits = self.boxes = self.boxes_correct = 0
        self.by_bucket: dict[str, list[int]] = {}
        hit_ious: list[float] = []
        box_ious: list[float] = []

        for index, (frame, tracks) in enumerate(zip(pixels, corpus.tracks)):
            detector._begin_frame(corpus.envelopes[index])
            found = detector.detect_plates(frame, tracks)
            self.boxes += len(found)
            on_frame = corpus.plates_on_frame(index)

            for vehicle in corpus.legible_plates(index):
                self.legible += 1
                bucket = self.by_bucket.setdefault(
                    width_bucket(vehicle.plate_width_px), [0, 0]
                )
                bucket[1] += 1
                best = max(
                    (
                        _iou(c.plate_bbox_xyxy, vehicle.plate_bbox_xyxy)
                        for c in found.values()
                    ),
                    default=0.0,
                )
                if best >= HIT_IOU:
                    self.hits += 1
                    hit_ious.append(best)
                    bucket[0] += 1

            for candidate in found.values():
                best = max(
                    (
                        _iou(candidate.plate_bbox_xyxy, v.plate_bbox_xyxy)
                        for v in on_frame
                    ),
                    default=0.0,
                )
                box_ious.append(best)
                if best >= HIT_IOU:
                    self.boxes_correct += 1

        self.stats = detector.stats()
        detector.close()
        self.recall = self.hits / self.legible if self.legible else 0.0
        self.mean_iou = float(np.mean(box_ious)) if box_ious else 0.0
        self.iou_over_hits = float(np.mean(hit_ious)) if hit_ious else 0.0
        self.box_ious = box_ious

    def bucket(self, key: str) -> tuple[int, int]:
        hits, total = self.by_bucket.get(key, [0, 0])
        return hits, total

    def bucket_recall(self, key: str) -> float:
        hits, total = self.bucket(key)
        return hits / total if total else 0.0


class _OracleSource:
    """Just enough source for OraclePlateDetector: truth keyed by envelope.

    A real SyntheticReplaySource would work, but it would decode the clip a second time
    per test. This exposes the one method the oracle's _load checks for.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._by_pts = {
            env.pts_ms: truth for env, truth in zip(corpus.envelopes, corpus.truth)
        }

    def truth_for_envelope(self, envelope):
        return self._by_pts.get(envelope.pts_ms)


class _NoTruth:
    """A source that satisfies the factory's duck-type check and knows nothing.

    Needed because the oracle's source requirement is enforced at *build* time, so any
    test that merely wants to construct one has to hand it something with the method.
    """

    def truth_for_envelope(self, envelope):
        return None


def track(
    *,
    box=(600, 400, 800, 560),
    track_id: int = 7,
    class_name: str = "car",
    confidence: float = 0.91,
    frame_index: int = 0,
    pts_ms: int = 0,
    session: str = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05",
) -> TrackResult:
    return TrackResult(
        camera_id=CAMERA,
        stream_session_id=session,
        track_id=track_id,
        bbox_xyxy=box,
        class_name=class_name,
        confidence=confidence,
        frame_index=frame_index,
        pts_ms=pts_ms,
    )


def blank(width: int = FRAME_W, height: int = FRAME_H, value: int = 40) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


# ================================================================ the fixture itself


def test_the_fixture_is_the_one_the_claims_name(corpus: Corpus) -> None:
    """200 raw frames emit 67. Reading that as 200 emitted is the recorded error.

    Pinned because five measured claims in ai/plate/stub.py were unreproducible for
    exactly this reason: the docstring said "200 emitted frames at 120 ms", and a
    reader who builds a fixture that emits 200 frames gets total_frames=600, 894
    legible plate-frames, and not one matching cell. The numbers were right; the
    protocol line was wrong.
    """
    assert len(corpus.frames) == EXPECTED_EMITTED_FRAMES
    assert corpus.frames[0].shape == (FRAME_H, FRAME_W, 3)

    vehicle_frames = sum(len(t.vehicles) for t in corpus.truth if t is not None)
    legible = sum(len(corpus.legible_plates(i)) for i in range(len(corpus.frames)))
    distinct = {
        v.plate for t in corpus.truth if t is not None for v in t.vehicles
    }

    assert vehicle_frames == EXPECTED_VEHICLE_FRAMES
    assert legible == EXPECTED_LEGIBLE
    assert len(distinct) == EXPECTED_DISTINCT_VEHICLES


def test_the_clip_holds_six_vehicles_not_six_per_frame(corpus: Corpus) -> None:
    """3.2 vehicles on screen per frame, 6 in the whole clip.

    The docstring's throughput line used to read "around six vehicles a frame", which
    conflated the two and made the per-frame cost look twice what it is.
    """
    seen = sum(len(t) for t in corpus.tracks)
    per_frame = seen / len(corpus.frames)
    assert 3.0 <= per_frame <= 3.5
    assert seen == 217


# ============================================================== coordinate frames


def test_the_crop_origin_is_returned_because_it_cannot_be_recomputed() -> None:
    """A vehicle at the frame edge gets a clamped origin, not the one padding asked for."""
    frame = blank()
    crop, origin = crop_vehicle(frame, (0, 0, 100, 80), pad_fraction=0.10)
    assert origin == (0, 0)                      # padding wanted (-10, -8)
    assert crop.shape[:2] == (88, 110)           # and only the inside half was taken

    crop, origin = crop_vehicle(frame, (600, 400, 700, 480), pad_fraction=0.10)
    assert origin == (590, 392)
    assert crop.shape[:2] == (96, 120)


def test_the_crop_is_a_view_not_a_copy() -> None:
    """Documented as deliberate: 80 crops a second per camera, and nobody writes to them."""
    frame = blank()
    crop, _ = crop_vehicle(frame, (600, 400, 700, 480))
    assert crop.base is frame or crop.base is frame.base
    frame[420, 620] = (7, 7, 7)
    assert tuple(crop[420 - 394, 620 - 592]) == (7, 7, 7)


def test_a_degenerate_box_returns_an_empty_crop_rather_than_raising() -> None:
    """One bad tracker box on one frame must not end a run. The caller counts it."""
    frame = blank()
    for box in ((500, 400, 500, 400), (-40, -40, -10, -10), (2000, 900, 2100, 1000)):
        crop, origin = crop_vehicle(frame, box)
        assert crop.size == 0
        assert crop.shape == (0, 0, 3)
        assert origin[0] >= 0 and origin[1] >= 0


def test_the_empty_crop_is_counted_not_swallowed() -> None:
    detector = ScriptedPlateDetector({0: [((0, 0, 40, 10), 0.9)]})
    detector.load()
    found = detector.detect_plates(blank(), [track(box=(2000, 900, 2100, 1000))])
    assert found == {}
    assert detector.stats()["crops_empty"] == 1
    assert detector.stats()["vehicles_without_plate"] == 1


def test_map_to_frame_is_the_inverse_of_the_crop_origin() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    crop, origin = crop_vehicle(frame, vehicle)
    local = (30, 100, 90, 118)
    mapped = map_to_frame(local, origin, frame.shape)
    assert mapped == (origin[0] + 30, origin[1] + 100, origin[0] + 90, origin[1] + 118)
    # And the box is inside the crop's footprint in the frame.
    assert origin[0] <= mapped[0] and mapped[2] <= origin[0] + crop.shape[1]


def test_map_to_frame_clamps_at_the_frame_edge() -> None:
    """Otherwise plate_width_px disagrees with the snapshot the coordinates produce."""
    mapped = map_to_frame((0, 0, 400, 60), (FRAME_W - 100, FRAME_H - 30), (FRAME_H, FRAME_W))
    assert mapped == (FRAME_W - 100, FRAME_H - 30, FRAME_W, FRAME_H)
    assert mapped[2] - mapped[0] == 100
    assert mapped[3] - mapped[1] == 30


def test_a_backend_that_returns_frame_coordinates_is_wrong_by_the_offset() -> None:
    """The bug _detect_in_crop's docstring warns about, built so its shape is on record.

    A backend that "helpfully" returns full-frame coordinates gets the crop origin added
    a second time, so the error is exactly the vehicle's distance from the top-left
    corner. Near the origin that is a few pixels and the output looks fine; at the far
    corner of a 1280x720 frame it is most of the frame away. This is why the mapping
    lives in one function and why the contract on _detect_in_crop is crop-local.
    """
    frame = blank()
    true_plate = (660, 520, 740, 539)

    near = track(box=(20, 30, 220, 190))
    far = track(box=(600, 400, 800, 560))

    errors = {}
    for label, t in (("near", near), ("far", far)):
        _, origin = crop_vehicle(frame, t.bbox_xyxy)
        # A correct backend would return this box minus the origin.
        already_frame_coords = true_plate
        doubled = map_to_frame(already_frame_coords, origin, frame.shape)
        errors[label] = doubled[0] - true_plate[0]

    assert errors["near"] < 20                   # looks almost right
    assert errors["far"] > 500                   # and is half a frame out
    assert errors["far"] > 25 * errors["near"]


def test_clip_to_crop_refuses_a_box_that_leaves_nothing() -> None:
    """A rounding overshoot that survives becomes a negative plate_width_px in an event."""
    assert clip_to_crop((10, 5, 50, 20), (100, 200)) == (10, 5, 50, 20)
    assert clip_to_crop((-5, -5, 50, 20), (100, 200)) == (0, 0, 50, 20)
    assert clip_to_crop((190, 5, 260, 20), (100, 200)) == (190, 5, 200, 20)
    assert clip_to_crop((210, 5, 260, 20), (100, 200)) is None
    assert clip_to_crop((10, 5, 10, 20), (100, 200)) is None


def test_the_clip_rejection_is_counted_separately_from_the_shape_rejection() -> None:
    """Two different failures. Merging them hides which one a config change caused."""
    detector = ScriptedPlateDetector(
        {0: [((900, 10, 950, 30), 0.9), ((5, 5, 60, 8), 0.9)]}
    )
    detector.load()
    found = detector.detect_plates(blank(), [track(box=(600, 400, 800, 560))])
    stats = detector.stats()
    assert stats["boxes_proposed"] == 2
    assert stats["boxes_rejected_clip"] == 1     # x beyond the 232 px crop
    assert stats["boxes_rejected_shape"] == 1    # 55x3, too thin to be a plate
    assert found == {}


# ==================================================================== shape gate


def test_the_aspect_range_admits_every_indian_plate_format() -> None:
    """PLATE_ASPECT_MIN exists so a 4:1 filter does not reject every motorcycle.

    The four CMVR formats, as the geometry module's comment lists them.
    """
    formats = {
        "car single row": 500 / 120,
        "car two row": 340 / 200,
        "motorcycle two row": 285 / 200,
        "motorcycle single row": 200 / 100,
    }
    for label, ratio in formats.items():
        assert PLATE_ASPECT_MIN <= ratio <= PLATE_ASPECT_MAX, label
        height = 40
        box = (0, 0, int(round(ratio * height)), height)
        assert plausible_plate_box(box), label


def test_the_shape_gate_rejects_a_grille_strip_and_a_bumper_shadow() -> None:
    assert not plausible_plate_box((0, 0, 400, 20))      # aspect 20, a grille strip
    assert not plausible_plate_box((0, 0, 20, 60))       # aspect 0.33, taller than wide
    assert not plausible_plate_box((0, 0, 6, 6))         # below MIN_PLATE_BOX_PX


def test_the_shape_gate_boundaries_are_inclusive() -> None:
    height = 100
    assert plausible_plate_box((0, 0, int(PLATE_ASPECT_MIN * height), height))
    assert plausible_plate_box((0, 0, int(PLATE_ASPECT_MAX * height), height))
    assert not plausible_plate_box((0, 0, int(PLATE_ASPECT_MAX * height) + 2, height))


def test_the_minimum_box_is_asymmetric_in_width_and_height() -> None:
    """A plate 8 px wide is nothing; a plate 8 px tall at 40 px wide is a distant plate.

    plausible_plate_box takes min(min_size_px, 6) for the height, so a wide thin box
    survives where a small square does not.
    """
    assert MIN_PLATE_BOX_PX == 8
    assert plausible_plate_box((0, 0, 30, 6))     # aspect 5.0, inside the range
    assert not plausible_plate_box((0, 0, 30, 5))  # one pixel shorter, gone
    assert not plausible_plate_box((0, 0, 7, 6))   # narrower than the width bar
    # 40x6 fails, but on aspect (6.67) rather than size -- the two gates are separate.
    assert not plausible_plate_box((0, 0, 40, 6))
    assert aspect_ratio((0, 0, 40, 6)) > PLATE_ASPECT_MAX


def test_aspect_ratio_survives_a_zero_height_box() -> None:
    assert aspect_ratio((0, 0, 40, 0)) == 40.0
    assert aspect_ratio((0, 0, 0, 0)) == 0.0


# =================================================================== region prior


def test_the_region_prior_is_a_penalty_not_a_filter() -> None:
    """0.4 at the top of the box rather than 0.0, so a badly bounded plate is demoted."""
    vehicle = (600, 400, 800, 560)                # 160 px tall
    at_bumper = (660, 530, 740, 550)
    at_windscreen = (660, 402, 740, 414)
    above_the_box = (660, 380, 740, 396)          # possible: the crop is padded

    assert region_prior(at_bumper, vehicle) == 1.0
    assert region_prior(at_windscreen, vehicle) == pytest.approx(0.486, abs=0.002)
    assert region_prior(above_the_box, vehicle) == pytest.approx(0.4, abs=1e-9)
    assert region_prior(above_the_box, vehicle) > 0.0, "demoted, never deleted"


def test_the_region_prior_turns_over_at_the_documented_fraction() -> None:
    vehicle = (0, 0, 200, 100)
    boundary_centre = PLATE_REGION_TOP_FRACTION * 100
    just_below = (50, int(boundary_centre) + 1, 150, int(boundary_centre) + 3)
    assert region_prior(just_below, vehicle) == 1.0
    higher = (50, 8, 150, 12)
    assert region_prior(higher, vehicle) < 1.0


def test_the_region_prior_ranks_two_boxes_on_one_vehicle() -> None:
    """The behaviour it exists for: keep the bumper box when both are found."""
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    windscreen_local = (60, 405 - origin[1], 140, 420 - origin[1])
    bumper_local = (60, 530 - origin[1], 140, 545 - origin[1])

    detector = ScriptedPlateDetector(
        {0: [(windscreen_local, 0.90), (bumper_local, 0.60)]},
        confidence_threshold=0.0,
    )
    detector.load()
    found = detector.detect_plates(frame, [track(box=vehicle)])
    chosen = found[7].plate_bbox_xyxy
    assert chosen[1] > 500, "the lower box should win despite lower raw confidence"


def test_the_region_prior_can_be_switched_off() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    windscreen_local = (60, 405 - origin[1], 140, 420 - origin[1])
    bumper_local = (60, 530 - origin[1], 140, 545 - origin[1])
    script = {0: [(windscreen_local, 0.90), (bumper_local, 0.60)]}

    off = ScriptedPlateDetector(dict(script), apply_region_prior=False,
                                confidence_threshold=0.0)
    off.load()
    chosen = off.detect_plates(frame, [track(box=vehicle)])[7].plate_bbox_xyxy
    assert chosen[1] < 500, "with the prior off, raw confidence decides"


# ============================================================== the base pipeline


def test_a_missing_key_is_the_answer_for_no_plate() -> None:
    """Contracts section 6: plate null is correct. Not an exception, not a sentinel box."""
    detector = ScriptedPlateDetector({})
    detector.load()
    found = detector.detect_plates(blank(), [track(track_id=1), track(track_id=2)])
    assert found == {}
    assert isinstance(found, dict)
    assert detector.stats()["vehicles_without_plate"] == 2


def test_the_result_is_keyed_by_track_id_not_by_index() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    local = (60, 530 - origin[1], 140, 545 - origin[1])
    detector = ScriptedPlateDetector({0: [(local, 0.9)]}, confidence_threshold=0.0)
    detector.load()
    found = detector.detect_plates(frame, [track(box=vehicle, track_id=4242)])
    assert set(found) == {4242}
    assert isinstance(found[4242], PlateCandidate)


def test_detect_plates_before_load_raises_rather_than_returning_nothing() -> None:
    """A missing checkpoint must fail at startup, not look like a detector that found none."""
    detector = ScriptedPlateDetector({})
    with pytest.raises(RuntimeError, match="before load"):
        detector.detect_plates(blank(), [track()])


def test_load_and_close_are_idempotent() -> None:
    detector = ScriptedPlateDetector({})
    detector.load()
    detector.load()
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="before load"):
        detector.detect_plates(blank(), [track()])


def test_the_context_manager_loads_and_releases() -> None:
    with ScriptedPlateDetector({}) as detector:
        assert detector.detect_plates(blank(), [track()]) == {}
    with pytest.raises(RuntimeError, match="before load"):
        detector.detect_plates(blank(), [track()])


def test_the_best_box_wins_on_score_after_the_prior_not_before() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    low_but_placed = (60, 530 - origin[1], 140, 545 - origin[1])
    high_but_high_up = (60, 390 - origin[1], 140, 398 - origin[1])

    detector = ScriptedPlateDetector(
        {0: [(high_but_high_up, 0.99), (low_but_placed, 0.45)]},
        confidence_threshold=0.0,
    )
    detector.load()
    candidate = detector.detect_plates(frame, [track(box=vehicle)])[7]
    # 0.99 * 0.4 = 0.396 against 0.45 * 1.0 = 0.45.
    assert candidate.plate_bbox_xyxy[1] > 500
    assert candidate.detector_confidence == pytest.approx(0.45, abs=1e-3)


def test_the_confidence_is_the_post_prior_score_and_is_capped_at_one() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    local = (60, 530 - origin[1], 140, 545 - origin[1])
    detector = ScriptedPlateDetector({0: [(local, 4.0)]}, confidence_threshold=0.0)
    detector.load()
    candidate = detector.detect_plates(frame, [track(box=vehicle)])[7]
    assert candidate.detector_confidence == 1.0


def test_a_box_below_the_threshold_is_counted_and_dropped() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    local = (60, 530 - origin[1], 140, 545 - origin[1])
    detector = ScriptedPlateDetector({0: [(local, 0.10)]}, confidence_threshold=0.25)
    detector.load()
    assert detector.detect_plates(frame, [track(box=vehicle)]) == {}
    assert detector.stats()["boxes_below_threshold"] == 1


def test_the_stage_does_not_filter_small_plates() -> None:
    """DEFAULT_PLATE_CONFIDENCE_THRESHOLD's comment: the <30 px bucket must report attempts.

    A stage that silently dropped tiny plates would make the smallest width bucket read
    "0 attempts, 0 errors", which looks like no problem rather than never tried.
    """
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    tiny = (60, 530 - origin[1], 60 + 22, 530 - origin[1] + 8)
    detector = ScriptedPlateDetector({0: [(tiny, 0.9)]}, confidence_threshold=0.0)
    detector.load()
    found = detector.detect_plates(frame, [track(box=vehicle)])
    assert 7 in found
    x1, _, x2, _ = found[7].plate_bbox_xyxy
    assert width_bucket(x2 - x1) == "<30"


def test_min_vehicle_width_defaults_to_zero_and_is_opt_in() -> None:
    """This stage does not decide what is worth trying; config does, visibly."""
    detector = ScriptedPlateDetector({})
    assert detector.min_vehicle_width_px == 0

    frame = blank()
    small = track(box=(600, 400, 640, 432))
    gated = ScriptedPlateDetector({}, min_vehicle_width_px=80)
    gated.load()
    gated.detect_plates(frame, [small])
    assert gated.stats()["vehicles_skipped_small"] == 1
    assert gated.stats()["crops_empty"] == 0     # skipped before the crop


def test_recall_proxy_is_named_a_proxy_and_is_plates_over_vehicles() -> None:
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    local = (60, 530 - origin[1], 140, 545 - origin[1])
    detector = ScriptedPlateDetector({0: [(local, 0.9)]}, confidence_threshold=0.0)
    detector.load()
    detector.detect_plates(frame, [track(box=vehicle), track(box=vehicle, track_id=8)])
    stats = detector.stats()
    assert stats["vehicles_seen"] == 2
    assert stats["plates_emitted"] == 2
    assert stats["recall_proxy"] == 1.0
    assert "recall" not in stats, "the honest name is the only name offered"


def test_stats_reports_every_reason_a_plate_was_not_produced() -> None:
    """Each counter is a different config mistake. One aggregate number diagnoses none."""
    expected = {
        "vehicles_seen",
        "vehicles_skipped_small",
        "crops_empty",
        "boxes_proposed",
        "boxes_below_threshold",
        "boxes_rejected_shape",
        "boxes_rejected_clip",
        "plates_emitted",
        "vehicles_without_plate",
    }
    stats = ScriptedPlateDetector({}).stats()
    assert expected <= set(stats)
    assert {"model_name", "model_version", "license", "ships"} <= set(stats)


def test_best_by_confidence_returns_none_on_an_empty_list() -> None:
    assert best_by_confidence([], []) is None
    assert best_by_confidence([(0, 0, 1, 1), (0, 0, 2, 2)], [0.1, 0.9]) == 1


# ================================================================= the oracle


def test_the_oracle_needs_a_synthetic_source_and_refuses_at_build_time() -> None:
    """The check is in the factory, not in load(), and that is the useful place for it.

    A misconfigured oracle discovered at load() is a run that has already opened a
    stream; discovered by build_plate_detector it is a config error, which is what
    scripts/validate_config.py can catch before anything starts.
    """
    with pytest.raises(PlateConfigError, match="truth_for_envelope"):
        build_plate_detector({"name": "oracle"}, source=object())
    with pytest.raises(PlateConfigError, match="needs the media source"):
        build_plate_detector({"name": "oracle"})

    # And the same guard exists on the class, for anyone constructing it directly.
    from ai.plate.stub import OraclePlateDetector

    direct = OraclePlateDetector(object())
    with pytest.raises(RuntimeError, match="truth_for_envelope"):
        direct.load()


def test_the_seed_42_fixture_cannot_exercise_the_legibility_guard(
    corpus: Corpus,
) -> None:
    """All 14 non-legible vehicle-frames have no plate box, so require_legible never fires.

    Recorded rather than worked around. A test that asserted illegible_skipped > 0 on
    this fixture would fail, and the tempting reading of that failure -- "the oracle
    ignores legibility" -- is wrong. The guard is real; the fixture just never presents
    the case, so it is tested against a constructed truth frame below.
    """
    total = sum(len(t.vehicles) for t in corpus.truth if t is not None)
    no_box = sum(
        1
        for t in corpus.truth
        if t is not None
        for v in t.vehicles
        if v.plate_bbox_xyxy is None
    )
    box_but_illegible = sum(
        1
        for t in corpus.truth
        if t is not None
        for v in t.vehicles
        if v.plate_bbox_xyxy is not None and not v.plate_legible
    )
    assert (total, no_box, box_but_illegible) == (240, 14, 0)
    assert total - no_box == EXPECTED_LEGIBLE

    score = Score(corpus, {"name": "oracle"})
    assert score.stats["illegible_skipped"] == 0


def test_the_oracle_respects_legibility() -> None:
    """A perfect detector that read an invisible plate would measure the unachievable."""
    from ai.media.synthetic_source import FrameTruth, VehicleTruth
    from ai.plate.stub import OraclePlateDetector

    vehicle_box = (600, 400, 800, 560)
    illegible = VehicleTruth(
        vehicle_id=1,
        plate="GJ01AB1234",
        vehicle_type="car",
        vehicle_bbox_xyxy=vehicle_box,
        plate_bbox_xyxy=(660, 520, 740, 539),
        plate_width_px=80,
        plate_legible=False,
    )
    frame_truth = FrameTruth(frame_index=0, pts_ms=0, vehicles=(illegible,))

    class _Fixed:
        def truth_for_envelope(self, envelope):
            return frame_truth

    strict = OraclePlateDetector(_Fixed())
    strict.load()
    strict._begin_frame(object())
    assert strict.detect_plates(blank(), [track(box=vehicle_box)]) == {}
    assert strict.stats()["illegible_skipped"] == 1

    permissive = OraclePlateDetector(_Fixed(), require_legible=False)
    permissive.load()
    permissive._begin_frame(object())
    found = permissive.detect_plates(blank(), [track(box=vehicle_box)])
    assert 7 in found, "with the guard off it reads the invisible plate"
    assert found[7].plate_bbox_xyxy == (660, 520, 740, 539)
    assert permissive.stats()["illegible_skipped"] == 0


def test_the_oracle_returns_frame_coordinates_after_a_round_trip() -> None:
    """It converts truth *back* to crop-local so the base class's one mapping applies.

    Deliberately perverse and worth pinning: the alternative is a second code path for
    backends that already know frame coordinates, and then the mapping Contracts
    section 3 warns about exists twice.
    """
    from ai.media.synthetic_source import FrameTruth, VehicleTruth
    from ai.plate.stub import OraclePlateDetector

    vehicle_box = (600, 400, 800, 560)
    plate_box = (660, 520, 740, 539)
    truth = FrameTruth(
        frame_index=0,
        pts_ms=0,
        vehicles=(
            VehicleTruth(
                vehicle_id=1,
                plate="GJ01AB1234",
                vehicle_type="car",
                vehicle_bbox_xyxy=vehicle_box,
                plate_bbox_xyxy=plate_box,
                plate_width_px=80,
                plate_legible=True,
            ),
        ),
    )

    class _Fixed:
        def truth_for_envelope(self, envelope):
            return truth

    detector = OraclePlateDetector(_Fixed())
    detector.load()
    detector._begin_frame(object())
    found = detector.detect_plates(blank(), [track(box=vehicle_box)])
    assert found[7].plate_bbox_xyxy == plate_box


def test_the_oracle_reaches_its_ceiling_and_the_ceiling_is_not_one(
    corpus: Corpus,
) -> None:
    """0.956, and the 10 missing sit in one bucket. Loss before this stage, not in it."""
    score = Score(corpus, {"name": "oracle"})
    assert score.legible == EXPECTED_LEGIBLE
    assert score.recall == pytest.approx(0.956, abs=0.002)
    assert score.mean_iou == pytest.approx(1.000, abs=0.001)
    assert (score.boxes_correct, score.boxes) == (216, 216)

    for key in (">100", "80-100", "60-80"):
        assert score.bucket_recall(key) == 1.0, key
    assert score.bucket("40-60") == (67, 77)
    assert score.legible - score.hits == 10


def test_the_oracle_ceiling_is_the_trackers_confirmation_latency(
    corpus: Corpus,
) -> None:
    """All 10 of the oracle's misses are in one bucket, and it is not this stage's fault.

    The obvious explanation is ORACLE_MATCH_IOU rejecting a tracker box that has drifted
    off its truth vehicle, and that explanation is wrong: no_truth_match is 0 for the
    whole run. What actually happens is that the 10 legible plates have no emitted track
    at all on the frame in question, because ByteTrack needs DEFAULT_MIN_HITS = 3 hits
    before a track is CONFIRMED and BaseTracker.update only returns active tracks.

    Dropping min_hits to 1 recovers 7 of the 10, which is the confirmation cost made
    visible. And they land in 40-60 px for a mechanical reason worth stating: a vehicle
    enters the frame far away, so its plate is small exactly while its track is still
    TENTATIVE, and by the time the track confirms the plate has grown past 60 px. The
    small-plate bucket therefore carries the tracker's warm-up on top of its own
    difficulty, which is a reason to read the oracle row as the ceiling for the
    *pipeline* rather than for the plate stage alone.
    """
    detector = build_plate_detector({"name": "oracle"}, source=_OracleSource(corpus))
    detector.load()
    for index, (frame, tracks) in enumerate(zip(corpus.frames, corpus.tracks)):
        detector._begin_frame(corpus.envelopes[index])
        detector.detect_plates(frame, tracks)
    stats = detector.stats()
    detector.close()

    assert ORACLE_MATCH_IOU == 0.30
    assert stats["no_truth_match"] == 0, "not a box-match failure"
    assert stats["unresolved_frames"] == 0
    assert stats["crops_empty"] == 0
    assert stats["boxes_rejected_shape"] == stats["boxes_rejected_clip"] == 0

    # 217 tracks emitted, 216 plates: one emitted track has no truth plate box.
    assert (stats["vehicles_seen"], stats["plates_emitted"]) == (217, 216)

    score = Score(corpus, {"name": "oracle"})
    assert score.legible - score.hits == 10
    assert score.bucket("40-60") == (67, 77)
    for key in (">100", "80-100", "60-80"):
        assert score.bucket_recall(key) == 1.0, key


def test_lowering_min_hits_recovers_most_of_that_ceiling() -> None:
    """0.956 -> 0.987 with min_hits=1. The cost is attributed, not just measured.

    Not a recommendation to ship min_hits=1 -- three hits is what suppresses a
    single-frame detector artefact from becoming a track and then a sighting event.
    This exists so the 0.956 is not read as a plate-stage ceiling.
    """
    from ai.plate.stub import _iou as iou

    results = {}
    for min_hits in (3, 1):
        source = build_source(dict(SOURCE))
        source.open()
        detector = build_detector({"name": "oracle", "miss_rate": 0.0}, source=source)
        detector.load()
        tracker = build_tracker(
            {"name": "bytetrack", "min_hits": min_hits},
            CAMERA,
            source.session_id,
            source=source,
        )
        plate = build_plate_detector({"name": "oracle"}, source=source)
        plate.load()

        legible = hits = 0
        for envelope in source:
            tracks = tracker.update(
                detector.detect_envelope(envelope),
                frame_index=envelope.frame_index,
                pts_ms=envelope.pts_ms,
            )
            plate._begin_frame(envelope)
            found = plate.detect_plates(envelope.frame_bgr, tracks)
            truth = source.truth_at_pts(envelope.pts_ms)
            if truth is None:
                continue
            for vehicle in truth.vehicles:
                if vehicle.plate_bbox_xyxy is None or not vehicle.plate_legible:
                    continue
                legible += 1
                best = max(
                    (
                        iou(c.plate_bbox_xyxy, vehicle.plate_bbox_xyxy)
                        for c in found.values()
                    ),
                    default=0.0,
                )
                if best >= HIT_IOU:
                    hits += 1
        source.close()
        detector.close()
        plate.close()
        results[min_hits] = (legible, hits)

    assert results[3] == (EXPECTED_LEGIBLE, 216)
    assert results[1] == (EXPECTED_LEGIBLE, 223)
    assert results[1][1] - results[3][1] == 7


def test_the_oracle_miss_rate_is_deterministic(corpus: Corpus) -> None:
    """Hashed on (plate, frame_index), so two runs of one config agree exactly."""
    first = Score(corpus, {"name": "oracle", "miss_rate": 0.30})
    second = Score(corpus, {"name": "oracle", "miss_rate": 0.30})
    assert first.hits == second.hits
    assert 0 < first.hits < Score(corpus, {"name": "oracle"}).hits


def test_the_oracle_never_ships() -> None:
    detector = build_plate_detector({"name": "oracle"}, source=_NoTruth())
    assert detector.ships is False
    assert detector.model_name == "oracle-plate"
    assert detector.model_version == "truth"
    assert detector.license_name == "not-applicable"


# ============================================================== the edge detector


def test_the_edge_detector_needs_no_weights_and_no_network() -> None:
    """The property the offline-to-live swap rests on: this stage loads with nothing."""
    detector = build_plate_detector({"name": "edge"})
    detector.load()
    assert detector.detect_plates(blank(), []) == {}
    detector.close()


def test_the_edge_detector_reproduces_its_headline_numbers(corpus: Corpus) -> None:
    """recall 0.664, mean IoU 0.736, 150 of 151 boxes above the 0.3 bar."""
    score = Score(corpus, {"name": "edge"})
    assert score.legible == EXPECTED_LEGIBLE
    assert score.recall == pytest.approx(0.664, abs=0.002)
    assert score.mean_iou == pytest.approx(0.736, abs=0.002)
    assert (score.boxes_correct, score.boxes) == (150, 151)


def test_mean_iou_over_boxes_and_over_hits_are_different_numbers(
    corpus: Corpus,
) -> None:
    """0.736 against 0.740. Stated because a setting can move the two in opposite ways."""
    score = Score(corpus, {"name": "edge"})
    assert score.mean_iou == pytest.approx(0.736, abs=0.002)
    assert score.iou_over_hits == pytest.approx(0.740, abs=0.002)
    assert score.mean_iou < score.iou_over_hits


def test_the_edge_detector_is_usable_above_eighty_px_and_not_below_sixty(
    corpus: Corpus,
) -> None:
    """The single row that is the argument for the trained model: 0.221 at 40-60 px.

    And 40-60 px is where real junction plates live, which is why this breakdown exists
    and why Contracts section 7.2 forbids reporting one average.
    """
    score = Score(corpus, {"name": "edge"})
    assert score.bucket("40-60") == (17, 77)
    assert score.bucket("60-80") == (71, 86)
    assert score.bucket("80-100") == (54, 54)
    assert score.bucket(">100") == (8, 9)

    assert score.bucket_recall("40-60") == pytest.approx(0.221, abs=0.002)
    assert score.bucket_recall("80-100") == 1.0
    assert score.bucket_recall("40-60") < 0.30 < 0.80 < score.bucket_recall("60-80")


def test_the_edge_detector_loses_the_small_bucket_the_oracle_keeps(
    corpus: Corpus,
) -> None:
    """The comparison the oracle exists to make: 0.870 available, 0.221 achieved."""
    edge = Score(corpus, {"name": "edge"})
    oracle = Score(corpus, {"name": "oracle"})
    assert oracle.bucket_recall("40-60") == pytest.approx(0.870, abs=0.002)
    assert edge.bucket_recall("40-60") == pytest.approx(0.221, abs=0.002)
    gap = oracle.bucket_recall("40-60") - edge.bucket_recall("40-60")
    assert gap > 0.6, "most of the small-plate loss is this stage, not the tracker"


def test_the_edge_detector_never_ships_despite_a_clean_licence() -> None:
    """`ships` means "may appear in a published accuracy claim", not "may be distributed"."""
    detector = build_plate_detector({"name": "edge"})
    assert detector.license_name == "Apache-2.0"
    assert detector.ships is False
    assert "edge" not in SHIPPABLE_PLATE_DETECTORS
    assert plate_detector_ships("edge") is False


def test_the_edge_detector_reports_its_own_dead_ends(corpus: Corpus) -> None:
    """rows_empty and cols_empty separate "no dense rows" from "no dense columns"."""
    score = Score(corpus, {"name": "edge"})
    for key in ("crops_too_small", "rows_empty", "cols_empty", "gradient_threshold"):
        assert key in score.stats
    assert score.stats["gradient_threshold"] == EDGE_GRADIENT_THRESHOLD


# ================================================== the erosion, which is load-bearing


class _NoErosion(EdgePlateDetector):
    """EdgePlateDetector with the closing's second half removed. One line changed.

    Written out rather than parameterised because the erosion is not a setting -- it is
    the `+ radius` / `- radius` on the column span, and the claim being tested is about
    what happens when that specific line is absent.
    """

    def _detect_in_crop(self, crop_bgr, track_result):
        height, width = crop_bgr.shape[:2]
        if height < 12 or width < 16:
            self.crops_too_small += 1
            return ()
        y_offset = int(height * (1.0 - self.search_lower_fraction))
        region = crop_bgr[y_offset:, :]
        if region.shape[0] < EDGE_MIN_ROWS:
            self.crops_too_small += 1
            return ()
        grey = region.astype(np.int16).mean(axis=2)
        strokes = np.abs(np.diff(grey, axis=1)) > self.gradient_threshold
        row_floor = max(2.0, self.row_min_fill * strokes.shape[1])
        band = _contiguous_band(strokes.sum(axis=1), row_floor, EDGE_MIN_ROWS)
        if band is None:
            self.rows_empty += 1
            return ()
        row_start, row_end = band
        radius = max(1, (row_end - row_start) // 2)
        col_counts = _running_max(strokes[row_start:row_end].sum(axis=0), radius)
        peak = float(col_counts.max()) if col_counts.size else 0.0
        span = _contiguous_band(
            col_counts, peak * self.col_quantile, EDGE_MIN_COLS + 2 * radius
        )
        if span is None:
            self.cols_empty += 1
            return ()
        col_start, col_end = span                # the only difference: no +/- radius
        local = (int(col_start), int(row_start + y_offset),
                 int(col_end + 1), int(row_end + y_offset))
        area = max(1, (row_end - row_start) * (col_end - col_start))
        density = float(strokes[row_start:row_end, col_start:col_end].sum()) / area
        confidence = min(0.85, max(0.0, density * 1.6)) * _aspect_agreement(
            aspect_ratio(local)
        )
        return ((local, round(confidence, 4)),)


def _shape_breakdown(corpus: Corpus, detector: BasePlateDetector) -> dict:
    detector.load()
    aspects = []
    for frame, tracks in zip(corpus.frames, corpus.tracks):
        for candidate in detector.detect_plates(frame, tracks).values():
            aspects.append(aspect_ratio(candidate.plate_bbox_xyxy))
    stats = detector.stats()
    detector.close()
    return {
        "proposed": stats["boxes_proposed"],
        "rejected_shape": stats["boxes_rejected_shape"],
        "emitted": stats["plates_emitted"],
        "near_ceiling": sum(1 for a in aspects if 4.2 < a <= PLATE_ASPECT_MAX),
    }


def test_without_the_erosion_the_detector_all_but_stops(corpus: Corpus) -> None:
    """198 of 216 rejected for shape, 18 surviving. It is not a refinement.

    The comment on this line used to claim 117 rejected and 63 surviving, which
    understated the damage threefold and made a load-bearing step read as a polish pass.
    """
    shipped = _shape_breakdown(corpus, EdgePlateDetector())
    without = _shape_breakdown(corpus, _NoErosion())

    assert shipped["proposed"] == without["proposed"] == 216
    assert shipped["rejected_shape"] == 65
    assert shipped["emitted"] == 151
    assert without["rejected_shape"] == 198
    assert without["emitted"] == 18
    assert shipped["emitted"] - without["emitted"] == 133


def test_the_erosion_failure_is_entirely_a_width_failure(corpus: Corpus) -> None:
    """Every one of the 198 is rejected for being too wide, which is the dilation's shape.

    Dilation adds radius on each side, so a 4.17 plate returns at about 5.17 and one
    whose row band under-measured the height clears 6.0 and is thrown out.
    """
    detector = _NoErosion()
    detector.load()
    too_wide = other = 0
    original = _NoErosion._detect_in_crop

    for index, (frame, tracks) in enumerate(zip(corpus.frames, corpus.tracks)):
        for t in tracks:
            crop, origin = crop_vehicle(frame, t.bbox_xyxy,
                                        pad_fraction=detector.pad_fraction)
            if crop.size == 0:
                continue
            for local, _ in original(detector, crop, t):
                clipped = clip_to_crop(local, crop.shape)
                if clipped is None:
                    continue
                mapped = map_to_frame(clipped, origin, frame.shape)
                if plausible_plate_box(mapped):
                    continue
                if aspect_ratio(mapped) > PLATE_ASPECT_MAX:
                    too_wide += 1
                else:
                    other += 1
    detector.close()
    assert too_wide == 198
    assert other == 0


def test_the_surviving_boxes_pile_up_below_the_aspect_ceiling(corpus: Corpus) -> None:
    """13 of the 18 survivors sit in 4.2-6.0, pressed against the gate that let them past."""
    without = _shape_breakdown(corpus, _NoErosion())
    assert without["near_ceiling"] == 13
    assert without["near_ceiling"] > without["emitted"] * 0.7


# ================================================= threshold against contrast


CONTRAST_LEVELS = (1.00, 0.60, 0.40, 0.25, 0.15, 0.10)

# The measured table. Every cell was re-run; the diagonal is the finding.
THRESHOLD_CONTRAST = {
    20: (0.637, 0.637, 0.664, 0.717, 0.726, 0.726),
    34: (0.637, 0.664, 0.717, 0.726, 0.000, 0.000),
    50: (0.664, 0.717, 0.726, 0.726, 0.000, 0.000),
    80: (0.717, 0.726, 0.726, 0.000, 0.000, 0.000),
    120: (0.726, 0.726, 0.000, 0.000, 0.000, 0.000),
}
DIAGONAL = ((120, 1.00), (80, 0.60), (50, 0.40), (34, 0.25), (20, 0.15))


@pytest.mark.parametrize("threshold,contrast", DIAGONAL)
def test_the_same_detector_appears_at_five_points_on_one_diagonal(
    corpus: Corpus, threshold: int, contrast: float
) -> None:
    """0.726 at (120,100%), (80,60%), (50,40%), (34,25%), (20,15%).

    The effective parameter is threshold/contrast, because the threshold is compared
    against gradients that scale linearly with contrast. Halving both is the same
    detector, and this is the claim in ai/plate/stub.py that reproduced exactly on
    every cell.
    """
    score = Score(corpus, {"name": "edge", "gradient_threshold": threshold},
                  contrast=contrast)
    assert score.recall == pytest.approx(0.726, abs=0.002)


@pytest.mark.parametrize("threshold", sorted(THRESHOLD_CONTRAST))
def test_each_threshold_dies_one_step_past_its_working_range(
    corpus: Corpus, threshold: int
) -> None:
    """Not a taper. The row goes to 0.000 in one step, because the band vanishes at once.

    A gradual decline would be a detector getting worse; this is _contiguous_band
    finding no run at all once every stroke falls under the threshold, which is why the
    parameter has to be chosen against the worst contrast expected rather than the best.
    """
    expected = THRESHOLD_CONTRAST[threshold]
    measured = [
        Score(corpus, {"name": "edge", "gradient_threshold": threshold},
              contrast=c).recall
        for c in CONTRAST_LEVELS
    ]
    for level, want, got in zip(CONTRAST_LEVELS, expected, measured):
        assert got == pytest.approx(want, abs=0.002), f"th={threshold} at {level}"

    zeros = [i for i, v in enumerate(measured) if v == 0.0]
    if zeros:
        assert zeros == list(range(zeros[0], len(measured))), "dies once, stays dead"


def test_fifty_beats_thirty_four_across_the_shared_range_and_ties_at_the_bottom(
    corpus: Corpus,
) -> None:
    """Why 50 is the default. Higher at 100/60/40% contrast, equal at 25%.

    The docstring claimed "strictly higher recall at every point inside it", which
    overstated the last cell -- they meet at 0.726.
    """
    for level in (1.00, 0.60, 0.40):
        low = Score(corpus, {"name": "edge", "gradient_threshold": 34}, contrast=level)
        default = Score(corpus, {"name": "edge", "gradient_threshold": 50},
                        contrast=level)
        assert default.recall > low.recall, level

    at_bottom = [
        Score(corpus, {"name": "edge", "gradient_threshold": t}, contrast=0.25).recall
        for t in (34, 50)
    ]
    assert at_bottom[0] == pytest.approx(at_bottom[1], abs=0.002)
    assert at_bottom[0] == pytest.approx(0.726, abs=0.002)


def test_the_default_threshold_is_the_shipped_one() -> None:
    assert EDGE_GRADIENT_THRESHOLD == 50
    detector = build_plate_detector({"name": "edge"})
    assert detector.gradient_threshold == 50


# ============================================================ grain, the other axis


def test_the_default_threshold_barely_notices_grain(corpus: Corpus) -> None:
    """0.664 / 0.664 / 0.659 / 0.650 at +/-0, 6, 10, 24. Chosen partly for this."""
    measured = [
        Score(corpus, {"name": "edge"}, grain=g).recall for g in (0, 6, 10, 24)
    ]
    for got, want in zip(measured, (0.664, 0.664, 0.659, 0.650)):
        assert got == pytest.approx(want, abs=0.003)
    assert measured[0] - measured[-1] < 0.02


def test_the_best_fixture_score_is_the_one_that_cannot_survive_a_real_camera(
    corpus: Corpus,
) -> None:
    """Threshold 5: recall 0.779 with perfect precision, then 0.058, then nothing.

    The generator flat-shades bodywork, so 5/255 measures a vehicle with no texture. The
    docstring claimed 0.863 / IoU 0.780 collapsing to 0.018; re-measured it is 0.779 /
    0.754 collapsing to 0.058. The collapse -- which is the point -- is unchanged.
    """
    clean = Score(corpus, {"name": "edge", "gradient_threshold": 5})
    assert clean.recall == pytest.approx(0.779, abs=0.003)
    assert clean.mean_iou == pytest.approx(0.754, abs=0.003)
    assert clean.boxes_correct == clean.boxes == 176

    default = Score(corpus, {"name": "edge"})
    assert clean.recall > default.recall, "it really is the better fixture score"

    grainy = Score(corpus, {"name": "edge", "gradient_threshold": 5}, grain=6)
    dead = Score(corpus, {"name": "edge", "gradient_threshold": 5}, grain=10)
    assert grainy.recall == pytest.approx(0.058, abs=0.005)
    assert dead.recall == 0.0

    survivor = Score(corpus, {"name": "edge"}, grain=10)
    assert survivor.recall > 0.60, "the default is untouched where threshold 5 is dead"


def test_the_dead_threshold_still_returns_boxes_which_is_worse_than_silence(
    corpus: Corpus,
) -> None:
    """At +/-10 grain, threshold 5 returns 212 boxes and not one of them is a plate.

    Worth pinning because a config that returns nothing is obviously broken, whereas one
    that returns a full complement of confident wrong boxes looks healthy from every
    counter this stage exposes -- recall_proxy included.
    """
    dead = Score(corpus, {"name": "edge", "gradient_threshold": 5}, grain=10)
    assert dead.boxes == 212
    assert dead.boxes_correct == 0
    assert dead.stats["recall_proxy"] > 0.9, "the proxy is blind to this, by construction"


# ====================================================== the col_quantile trade


def test_the_looser_quantile_buys_recall_and_pays_in_box_tightness(
    corpus: Corpus,
) -> None:
    """0.60 scores +0.022 recall and drops mean IoU from 0.736 to 0.577."""
    default = Score(corpus, {"name": "edge"})
    looser = Score(corpus, {"name": "edge", "col_quantile": 0.60})

    assert EDGE_COL_QUANTILE == 0.45
    assert looser.recall - default.recall == pytest.approx(0.022, abs=0.003)
    assert default.mean_iou == pytest.approx(0.736, abs=0.002)
    assert looser.mean_iou == pytest.approx(0.577, abs=0.003)


def test_the_twenty_loose_boxes_are_all_on_a_plate(corpus: Corpus) -> None:
    """The corrected finding, and a stronger argument than the one it replaces.

    The comment on EDGE_COL_QUANTILE used to call these "20 boxes that overlap no plate
    at all". The count is right and the description was not: all 20 overlap a real
    plate, the lowest at 0.187 and none at zero. So 0.60 does not hallucinate plates
    elsewhere in the crop -- it finds the same plates with looser boxes. That is the
    worse failure of the two: a box off the plate reads as unreadable, whereas a box on
    the plate but bounded too wide feeds OCR a clipped character and yields a confident
    wrong string, which Contracts section 12 names as the worst outcome available.
    """
    looser = Score(corpus, {"name": "edge", "col_quantile": 0.60})
    below = [v for v in looser.box_ious if v < HIT_IOU]
    assert len(below) == 20
    assert all(v > 0.0 for v in below)
    assert min(below) == pytest.approx(0.187, abs=0.005)

    default = Score(corpus, {"name": "edge"})
    assert len([v for v in default.box_ious if v < HIT_IOU]) == 1


# ============================================================= internal helpers


def test_the_contiguous_band_takes_the_longest_run_not_the_outer_span() -> None:
    """A plate under a bright grille: the outer span covers both and the aspect fails."""
    counts = np.array([9, 9, 0, 0, 0, 9, 9, 9, 9, 0])
    assert _contiguous_band(counts, 5.0, 2) == (5, 9)
    assert _contiguous_band(counts, 5.0, 5) is None
    assert _contiguous_band(np.array([]), 5.0, 1) is None
    assert _contiguous_band(counts, 0.0, 1) is None


def test_the_contiguous_band_finds_a_run_that_reaches_the_end() -> None:
    counts = np.array([0, 0, 7, 7, 7])
    assert _contiguous_band(counts, 5.0, 2) == (2, 5)


def test_the_running_max_turns_a_comb_into_a_plateau() -> None:
    """A plate's column profile alternates stroke and gap; the extent needs the gaps filled."""
    comb = np.array([0, 9, 0, 9, 0, 9, 0, 0, 0, 0])
    plateau = _running_max(comb, 1)
    assert list(plateau[:6]) == [9, 9, 9, 9, 9, 9]
    assert plateau[8] == 0
    assert _contiguous_band(comb, 5.0, 5) is None
    assert _contiguous_band(plateau, 5.0, 5) is not None


def test_the_aspect_agreement_is_a_plateau_not_a_peak() -> None:
    """Peaked at 4.2 it would score every motorcycle plate as a near miss."""
    assert _aspect_agreement(1.43) == _aspect_agreement(4.17) == 1.0
    assert _aspect_agreement(2.0) == 1.0
    assert _aspect_agreement(6.0) < 1.0
    assert _aspect_agreement(0.5) < 1.0
    assert _aspect_agreement(20.0) >= 0.2


def test_iou_is_zero_for_disjoint_boxes_and_one_for_identical() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_the_row_bar_is_absolute_so_the_gradient_threshold_can_move_it() -> None:
    """A peak-relative bar made the threshold inert: lowering it raises the peak too.

    Measured on the fixture, the median count of rows clearing 80% of the peak is 6 at
    gradient threshold 34, 6 at 20 and 6 at 10 -- a three-fold threshold change moving
    nothing. The absolute bar's responsiveness is the contrast table above, not this
    triple, since 34/20/10 all sit far below a clean glyph's gradient.
    """
    assert EDGE_ROW_MIN_FILL == 0.06
    profile = np.array([2, 3, 30, 32, 31, 33, 30, 3, 2])
    bar = max(2.0, EDGE_ROW_MIN_FILL * 100)
    assert int((profile >= 0.8 * profile.max()).sum()) == 5
    assert int((profile >= bar).sum()) == 5

    # Now weaken every gradient sixfold, as reduced contrast does. The peak-relative
    # count is unchanged, because the peak weakened by the same factor. The absolute
    # bar notices immediately, and that difference is the whole reason for the choice.
    weaker = profile // 6
    assert int((weaker >= 0.8 * weaker.max()).sum()) == 5
    assert int((weaker >= bar).sum()) == 0


# ==================================================== the scripted backend


def test_the_scripted_backend_is_keyed_on_frame_index() -> None:
    """So a test describes a whole frame in one line, and a frame with no entry is silent."""
    frame = blank()
    vehicle = (600, 400, 800, 560)
    _, origin = crop_vehicle(frame, vehicle)
    local = (60, 530 - origin[1], 140, 545 - origin[1])
    detector = ScriptedPlateDetector({3: [(local, 0.9)]}, confidence_threshold=0.0)
    detector.load()
    assert detector.detect_plates(frame, [track(box=vehicle, frame_index=2)]) == {}
    assert detector.detect_plates(frame, [track(box=vehicle, frame_index=3)]) != {}


def test_the_scripted_backend_does_not_ship() -> None:
    detector = ScriptedPlateDetector({})
    assert detector.ships is False
    assert detector.model_name == "scripted-plate"
    assert detector.license_name == "not-applicable"


# ==================================================================== the factory


def test_the_four_backend_names_are_the_registry() -> None:
    assert PLATE_DETECTOR_NAMES == ("rtdetr", "edge", "oracle", "scripted")
    assert SHIPPABLE_PLATE_DETECTORS == frozenset({"rtdetr"})


def test_an_unknown_backend_name_is_refused_with_the_alternatives() -> None:
    with pytest.raises(PlateConfigError, match="unknown plate detector"):
        build_plate_detector({"name": "yolo-plate"})
    with pytest.raises(PlateConfigError, match="no 'name'"):
        build_plate_detector({"confidence_threshold": 0.4})
    with pytest.raises(PlateConfigError, match="must be a mapping"):
        build_plate_detector(["edge"])


def test_a_misspelled_key_is_refused_rather_than_ignored() -> None:
    """A silently ignored key is a config that does not do what it says it does."""
    with pytest.raises(PlateConfigError, match="gradient_treshold"):
        build_plate_detector({"name": "edge", "gradient_treshold": 40})
    with pytest.raises(PlateConfigError, match="unknown key"):
        build_plate_detector({"name": "edge", "miss_rate": 0.1})


def test_per_backend_keys_do_not_leak_across_backends() -> None:
    """miss_rate belongs to the oracle, gradient_threshold to edge, and neither travels."""
    build_plate_detector({"name": "oracle", "miss_rate": 0.1}, source=_NoTruth())
    build_plate_detector({"name": "edge", "gradient_threshold": 40})
    with pytest.raises(PlateConfigError):
        build_plate_detector({"name": "oracle", "gradient_threshold": 40},
                             source=_NoTruth())


def test_the_common_keys_work_on_every_backend() -> None:
    for name, extra in (
        ("edge", {}),
        ("oracle", {}),
        ("scripted", {"script": {}}),
    ):
        detector = build_plate_detector(
            {
                "name": name,
                "confidence_threshold": 0.4,
                "pad_fraction": 0.12,
                "apply_region_prior": False,
                "min_vehicle_width_px": 30,
                **extra,
            },
            source=_NoTruth(),
        )
        assert detector.confidence_threshold == 0.4
        assert detector.pad_fraction == 0.12
        assert detector.apply_region_prior is False
        assert detector.min_vehicle_width_px == 30


def test_a_script_without_a_threshold_is_not_filtered() -> None:
    """An explicit instruction is not a proposal. The default 0.25 would drop test rows."""
    detector = build_plate_detector({"name": "scripted", "script": {0: [((0, 0, 40, 12), 0.05)]}})
    assert detector.confidence_threshold == 0.0

    explicit = build_plate_detector(
        {"name": "scripted", "script": {}, "confidence_threshold": 0.25}
    )
    assert explicit.confidence_threshold == 0.25


def test_a_malformed_script_says_which_entry_is_wrong() -> None:
    with pytest.raises(PlateConfigError, match="requires a 'script'"):
        build_plate_detector({"name": "scripted"})
    with pytest.raises(PlateConfigError, match="must be a mapping"):
        build_plate_detector({"name": "scripted", "script": [1, 2]})
    with pytest.raises(PlateConfigError, match="missing bbox_xyxy"):
        build_plate_detector({"name": "scripted", "script": {0: [{"confidence": 0.5}]}})
    with pytest.raises(PlateConfigError, match=r"script\[0\]"):
        build_plate_detector({"name": "scripted", "script": {0: ["nonsense"]}})


def test_a_script_accepts_both_the_pair_and_the_mapping_form() -> None:
    pairs = build_plate_detector(
        {"name": "scripted", "script": {0: [((1, 2, 41, 14), 0.7)]}}
    )
    mappings = build_plate_detector(
        {
            "name": "scripted",
            "script": {0: [{"bbox_xyxy": (1, 2, 41, 14), "confidence": 0.7}]},
        }
    )
    assert pairs._script == mappings._script


def test_publication_refuses_every_backend_but_the_trained_one() -> None:
    """A run that cannot be published must fail in the first second, not after the numbers."""
    for name, extra in (("edge", {}), ("oracle", {}), ("scripted", {"script": {}})):
        with pytest.raises(PlateConfigError, match="must not appear in a published"):
            normalize_plate_config({"name": name, **extra}, for_publication=True)
        assert normalize_plate_config({"name": name, **extra}) == {
            "name": name,
            **extra,
        }
    assert normalize_plate_config({"name": "rtdetr"}, for_publication=True) == {
        "name": "rtdetr"
    }


def test_normalize_refuses_the_same_keys_build_refuses() -> None:
    """Otherwise validate_config passes a block the run then rejects at startup."""
    bad = {"name": "edge", "gradient_treshold": 40}
    with pytest.raises(PlateConfigError):
        normalize_plate_config(bad)
    with pytest.raises(PlateConfigError):
        build_plate_detector(bad)


def test_describe_does_not_construct_or_download() -> None:
    """scripts/validate_config.py must not pull 171 MB of weights to check a YAML file."""
    described = describe_plate_detector({"name": "rtdetr"})
    assert described == {
        "name": "rtdetr",
        "ships": True,
        "confidence_threshold": DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
        "batched": True,
    }
    assert describe_plate_detector({"name": "edge"})["ships"] is False
    assert describe_plate_detector({"name": "edge"})["batched"] is False


def test_the_hugging_face_token_is_read_from_three_names_in_order(monkeypatch) -> None:
    """HUGGINGFACE_TOKEN, HF_TOKEN, HUGGING_FACE_HUB_TOKEN. All three are in the wild."""
    from ai.plate.factory import _env_token

    for name in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert _env_token() is None

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "third")
    assert _env_token() == "third"
    monkeypatch.setenv("HF_TOKEN", "second")
    assert _env_token() == "second"
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "first")
    assert _env_token() == "first"


# ================================================ the padding's premise


def test_the_plate_stage_is_never_handed_a_coasting_box(corpus: Corpus) -> None:
    """CROP_PAD_FRACTION was justified against a case the shipped path cannot produce.

    The comment claimed the padding covers the Kalman estimate's lag while a track
    coasts through a missed detection, "measured at IoU 0.98 against truth while
    coasting". Both halves were wrong. The estimate scores mean IoU 0.776 over 37
    coasting track-frames at miss_rate 0.25 and 0.819 over 66 at 0.50, worst case
    0.129 -- an order of magnitude more error than claimed. And BaseTracker.update
    returns only tracks where is_active holds, which requires time_since_update == 0,
    so a coasting box never reaches this stage at all.

    This test pins the premise rather than the number, because the premise is what a
    future change can break: widening is_active, or reading the tracker's live track
    list directly to get "more" vehicles to crop, starts feeding crop_vehicle boxes
    that can be 87% wrong, and the resulting crop of empty road looks like a plate
    detector problem.
    """
    source = build_source(dict(SOURCE))
    source.open()
    detector = build_detector({"name": "oracle", "miss_rate": 0.40}, source=source)
    detector.load()
    tracker = build_tracker(
        {"name": "bytetrack"}, CAMERA, source.session_id, source=source
    )

    coasting_internally = 0
    emitted = 0
    for envelope in source:
        detections = detector.detect_envelope(envelope)
        reported = tracker.update(
            detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
        )
        emitted += len(reported)
        coasting_internally += sum(
            1 for t in tracker._tracks if t.time_since_update > 0
        )
        # Whatever the tracker is carrying internally, nothing it hands out is stale.
        assert len(reported) == sum(1 for t in tracker._tracks if t.is_active)
    source.close()
    detector.close()

    assert coasting_internally > 0, "the fixture must actually produce coasting tracks"
    assert emitted > 0


def test_the_padding_is_small_enough_to_stay_on_the_vehicle() -> None:
    """Every padded pixel is road that can hold a false positive."""
    assert CROP_PAD_FRACTION == 0.08
    frame = blank()
    vehicle = (600, 400, 800, 560)
    crop, origin = crop_vehicle(frame, vehicle)
    vehicle_area = (vehicle[2] - vehicle[0]) * (vehicle[3] - vehicle[1])
    crop_area = crop.shape[0] * crop.shape[1]
    assert crop_area / vehicle_area < 1.40


def test_a_larger_pad_moves_the_origin_and_the_mapping_follows() -> None:
    """The origin is returned, not recomputed, so pad changes cannot desynchronise it."""
    frame = blank()
    vehicle = (600, 400, 800, 560)
    local = (30, 100, 90, 118)
    boxes = []
    for pad in (0.0, 0.08, 0.25):
        _, origin = crop_vehicle(frame, vehicle, pad_fraction=pad)
        boxes.append(map_to_frame(local, origin, frame.shape))
    assert boxes[0] != boxes[1] != boxes[2]
    assert boxes[0][0] > boxes[1][0] > boxes[2][0]


# ================================================ source independence


def test_the_edge_backend_is_indifferent_to_where_the_pixels_came_from() -> None:
    """Contracts 2.3: swapping source.mode must not change this stage's configuration.

    detect_plates takes an array. Only the oracle needs frame identity, and it needs it
    because it reads an answer key -- which is also why it never ships.
    """
    detector = build_plate_detector({"name": "edge"})
    detector.load()
    frame = blank(value=90)
    frame[520:540, 660:740] = 240
    for _ in range(2):
        assert isinstance(detector.detect_plates(frame, []), dict)
    assert detector.detect_plates(frame, [track()]) is not None
    detector.close()


def test_detect_plates_envelope_is_a_thin_wrapper(corpus: Corpus) -> None:
    """It exists for the oracle. For every other backend it must add nothing."""
    detector = build_plate_detector({"name": "edge"})
    detector.load()
    envelope = corpus.envelopes[10]
    direct = detector.detect_plates(corpus.frames[10], corpus.tracks[10])
    detector.close()

    again = build_plate_detector({"name": "edge"})
    again.load()
    wrapped = again.detect_plates_envelope(envelope, corpus.tracks[10])
    again.close()

    assert set(direct) == set(wrapped)
    for key in direct:
        assert direct[key].plate_bbox_xyxy == wrapped[key].plate_bbox_xyxy
