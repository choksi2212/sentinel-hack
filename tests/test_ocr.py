"""OCR -- stage 8 of the 14. Owner's manual 5.6, Contracts sections 3.3 and 12.

Three tiers of risk, and the tests are ordered by them rather than by module layout.

**Fabrication, which is the failure Contracts section 12 names as the worst this
pipeline can produce.** A fabricated plate does not lose information, it points at a
real registration number belonging to a real person at a place and time it was never
at. So the first thing tested is the refusal machinery: the width floor, that it is
checked against the *candidate* and not the crop, that the 24-29 px band the floor
deliberately does not cover is measured rather than assumed safe, and that the four
invented strings this corpus produces are still there. They are. This module pins the
fabrication rate rather than asserting it away, because the honest position -- stated in
TemplateOCR's docstring -- is that this stage does not prevent invention, it only makes
it visible and leaves the filtering to grammar and to cross-frame agreement.

**The variant loop's honesty.** `confidence` is a max over N preprocessing variants and
max-of-N is biased upward by construction; `agreement` exists to disclose that. Two
traps live here. `variants_tried` is the *configured* count, not the count that
returned something -- a test asserting on `agreement` under the default template config
is asserting on 1/1 and proves nothing, which is why every agreement test below passes
`variants` explicitly. And `ai/ocr/base.py:355` promises in prose that a test checks no
variant writes into its input; `test_no_variant_writes_into_its_input` is that test, and
without it a variant that wrote in place would corrupt the frame for every stage after
this one including the snapshot.

**The measured tables.** `ai/ocr/stub.py` justifies its constants and its one-variant
default with numbers, and a constant defended by a number nobody re-runs is a constant
defended by a story. Every figure in TemplateOCR's docstring is re-measured here: the
seven-row width table in both of its accuracy columns, the precision ladder, the two
confidence distributions, the 12-of-12 per-track result, the fabrication rows, and the
standalone width sweep that set MIN_TEMPLATE_SCORE. Several of those figures were wrong
before this module existed -- the table was stated at a non-default `min_hits`, the
100 px boundary disagreed with `ai/metrics.py`, and one number had been measured against
an implementation that no longer exists -- so the tests carry what the file used to
claim in their docstrings.

Two columns of that docstring are deliberately **not** pinned: 95.0 ms and 16.8 ms per
plate. They are this machine under this load, and a timing assertion in CI is a flake
with a ticket attached. The machine-independent form of the same claim is pinned
instead -- six variants perform six reads per plate and one performs one -- because that
is the number the stage is budgeted on and it cannot drift with the hardware.

The fixture is the one every measured claim in `ai/ocr` is stated against: source mode
synthetic, cam04, seed 42, total_frames 400 -- which *emits* 134 frames at 120 ms. It is
built once with the plate detector's `require_legible` off, giving 569 crops of which
550 are the legible ones the width table is about and 19 are the occluded ones the
fabrication section is about. `test_one_pass_serves_both_tables` checks that the legible
subset is pixel-identical to what a `require_legible=True` run offers, because a
six-variant pass over this corpus costs about a minute and doing it twice is not worth
the same numbers.
"""

import numpy as np
import pytest

from ai.contracts.stages import PlateCandidate
from ai.detect import build_detector
from ai.media import build_source
from ai.media.glyphs import GLYPHS, text_extent, text_mask
from ai.media.synthetic_source import (
    MIN_LEGIBLE_PLATE_WIDTH_PX,
    PLATE_ASPECT,
    FrameTruth,
    VehicleTruth,
    _draw_plate,
)
from ai.metrics import BUCKET_KEYS, width_bucket
from ai.ocr import (
    DEFAULT_VARIANTS,
    MIN_OCR_PLATE_WIDTH_PX,
    OCR_ENGINE_NAMES,
    PLATE_CROP_PAD_PX,
    SHIPPABLE_OCR_ENGINES,
    BaseOCR,
    OCRConfigError,
    OCREngine,
    OCRRead,
    apply_variant,
    build_ocr_engine,
    check_ocr_width_floor,
    default_variants_for,
    describe_ocr_engine,
    normalize_ocr_config,
    ocr_engine_ships,
    variant_names,
)
from ai.ocr.base import FrameRef
from ai.ocr.preprocess import (
    ADAPTIVE_OFFSET,
    STRETCH_HIGH_PERCENTILE,
    STRETCH_LOW_PERCENTILE,
    _resize_bilinear,
)
from ai.ocr.stub import (
    CELL_H,
    CELL_W,
    CONFUSIONS,
    MIN_PX_PER_CHAR,
    MIN_TEMPLATE_SCORE,
    ORACLE_PLATE_MATCH_IOU,
    TEMPLATE_DEFAULT_VARIANTS,
    TEMPLATE_INK_LEVEL,
    TEMPLATE_MAX_CHARS,
    TEMPLATE_MIN_CHARS,
    OracleOCR,
    ScriptedOCR,
    TemplateOCR,
    _resample_cell,
    _unit_hash,
)
from ai.plate import build_plate_detector
from ai.plate.stub import ORACLE_MATCH_IOU, _iou
from ai.track import build_tracker

# --------------------------------------------------------------------------- fixture

SOURCE = {
    "mode": "synthetic",
    "camera_id": "cam04",
    "seed": 42,
    "total_frames": 400,
    "target_interval_ms": 120,
}
CAMERA = SOURCE["camera_id"]

EXPECTED_EMITTED_FRAMES = 134
EXPECTED_VEHICLES = 12
EXPECTED_CROPS = 569
EXPECTED_LEGIBLE = 550
EXPECTED_ILLEGIBLE = 19

# Seven distinct strings across twelve vehicles: five drawn twice, two once. The
# docstring's 12-of-12 per-track result is a sanity check that fusion is possible on
# this corpus, not a measurement of how well fusion works, and this is why.
CORPUS_PLATES = (
    "22BH1234AA",
    "GJ01AB1234",
    "GJ05JK4521",
    "GJ18XY7788",
    "GJ27AA0001",
    "GJ3C4567",
    "MH12DE9812",
)

# TemplateOCR's docstring table, per width bucket:
#   crops, exact reads, positional char accuracy, edit-distance char accuracy, refusals.
# Buckets are ai/metrics.py's, which is why the single crop of exactly 100 px is in the
# 80-100 row: width_bucket's ">100" is strict. An earlier version of this table used
# >= 100 and reported 28/110 instead of 27/111.
WIDTH_TABLE = {
    ">100": (27, 17, 0.911, 0.911, 0),
    "80-100": (111, 42, 0.711, 0.731, 2),
    "60-80": (210, 40, 0.582, 0.602, 7),
    "40-60": (184, 23, 0.505, 0.526, 17),
    "30-40": (10, 0, 0.080, 0.130, 4),
    "<30": (8, 0, 0.051, 0.051, 6),
}
ALL_ROW = (550, 122, 0.583, 0.602, 36)

# Reads, and correct reads, at each confidence floor. Precision climbs monotonically
# even though the two confidence distributions overlap across almost their whole range,
# which is the entire empirical case for weighting by confidence rather than
# thresholding on it.
PRECISION_LADDER = (
    (0.42, 122, 514),
    (0.50, 114, 325),
    (0.55, 98, 213),
    (0.60, 63, 119),
    (0.70, 27, 32),
    (0.75, 6, 6),
)

# Every crop the six-variant configuration is measured on. A full six-variant pass over
# 550 crops costs about a minute; every twentieth crop costs three seconds and still
# spans 21 to 111 px.
SAMPLE_STRIDE = 20

FRAME_W, FRAME_H = 1280, 720


class Row:
    """One plate crop offered to the OCR stage, plus what truth says about it."""

    __slots__ = (
        "frame_index",
        "pts_ms",
        "track_id",
        "candidate",
        "crop",
        "width",
        "plate",
        "vehicle_id",
        "visible",
        "legible",
    )

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def frame_ref(self) -> FrameRef:
        return FrameRef(
            camera_id=CAMERA,
            stream_session_id="",
            frame_index=self.frame_index,
            pts_ms=self.pts_ms,
        )


class Corpus:
    """The fixture decoded, detected, tracked and cropped once, then reused.

    **Crops are cached, not frames.** 134 frames of 1280x720 is 370 MB; 569 plate crops
    is under 6 MB, and `cut_crop` is deterministic in `(frame, candidate, pad_px)`, so
    `read_crop` on a cached crop reproduces `read` on the live frame exactly at the
    default pad. Each crop is copied out because `cut_crop` returns a view and a view
    keeps its whole frame alive as its base -- caching views would retain all 370 MB
    while looking like a decoder leak.

    A test that needs a whole frame, or a non-default `pad_px`, renders a plate
    standalone with the generator's own `_draw_plate` instead.

    `require_legible` is off on the plate detector, so the 19 occluded plates truth
    marks unreadable are handed over as well. `legible` and `illegible` split them.
    """

    def __init__(self) -> None:
        source = build_source(dict(SOURCE))
        source.open()
        detector = build_detector({"name": "oracle", "miss_rate": 0.0}, source=source)
        detector.load()
        tracker = build_tracker(
            {"name": "bytetrack"}, CAMERA, source.session_id, source=source
        )
        plates = build_plate_detector(
            {"name": "oracle", "require_legible": False}, source=source
        )
        plates.load()
        cutter = build_ocr_engine({"name": "template"})
        cutter.load()

        self.source = source
        self.session_id = source.session_id
        self.rows: list[Row] = []
        self.frames_emitted = 0
        self.vehicle_ids: set[int] = set()

        for envelope in source:
            self.frames_emitted += 1
            truth = source.truth_for_envelope(envelope)
            detections = detector.detect_envelope(envelope)
            tracks = tracker.update(
                detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
            )
            if truth is not None:
                self.vehicle_ids.update(v.vehicle_id for v in truth.vehicles)

            found = plates.detect_plates_envelope(envelope, tracks)
            for track_id, candidate in found.items():
                match = _best_truth_plate(candidate, truth)
                crop = cutter.cut_crop(envelope.frame_bgr, candidate)
                self.rows.append(
                    Row(
                        frame_index=envelope.frame_index,
                        pts_ms=envelope.pts_ms,
                        track_id=track_id,
                        candidate=candidate,
                        crop=None if crop is None else crop.copy(),
                        width=candidate.plate_width_px,
                        plate=None if match is None else match.plate,
                        vehicle_id=None if match is None else match.vehicle_id,
                        visible=None if match is None else match.plate_visible_fraction,
                        legible=None if match is None else match.plate_legible,
                    )
                )

        source.close()
        detector.close()
        plates.close()
        cutter.close()

        self.legible = [r for r in self.rows if r.legible]
        self.illegible = [r for r in self.rows if r.legible is False]
        self._reads: dict[tuple[str, ...], list] = {}

    def reads(self, variants=None) -> list:
        """One read per row, in row order, from a default TemplateOCR.

        Memoised per variant tuple. The default (one variant) pass over all 569 crops is
        the expensive thing this module does; every measured table is derived from that
        single pass rather than from a pass per assertion.
        """
        key = tuple(variants) if variants is not None else TEMPLATE_DEFAULT_VARIANTS
        if key in self._reads:
            return self._reads[key]
        config = {"name": "template"}
        if variants is not None:
            config["variants"] = list(variants)
        engine = build_ocr_engine(config)
        engine.load()
        out = [engine.read_crop(r.crop, r.candidate) for r in self.rows]
        engine.close()
        self._reads[key] = out
        return out

    def legible_reads(self, variants=None) -> list[tuple[Row, object]]:
        """(row, read) for the 550 legible crops, refusals included as None reads."""
        return [
            (row, read)
            for row, read in zip(self.rows, self.reads(variants))
            if row.legible
        ]

    def sample(self) -> list[Row]:
        return self.legible[::SAMPLE_STRIDE]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus()


def _best_truth_plate(candidate: PlateCandidate, truth):
    """Highest-IoU truth plate for a candidate box, or None."""
    best, best_iou = None, 0.0
    for vehicle in truth.vehicles if truth is not None else ():
        if vehicle.plate_bbox_xyxy is None:
            continue
        score = _iou(candidate.plate_bbox_xyxy, vehicle.plate_bbox_xyxy)
        if score > best_iou:
            best, best_iou = vehicle, score
    return best


def _positional_matches(text, truth: str) -> int:
    """Matching characters in matching positions. The table's char-acc numerator."""
    return sum(1 for a, b in zip(text or "", truth) if a == b)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _edit_matches(text, truth: str) -> int:
    """Truth length minus edit distance, floored at zero. The softer char column."""
    return max(0, len(truth) - _levenshtein(text or "", truth))


def _render_plate(plate: str, width_px: int, background: int = 128):
    """One plate on a flat background, drawn by the generator's own renderer.

    Returns (frame, box). The box is what `_draw_plate` reports, so a candidate built
    from it is exactly what the plate stage would have produced on a perfect detection.
    """
    height = max(4, int(width_px * PLATE_ASPECT))
    frame = np.full((height + 40, width_px + 40, 3), background, dtype=np.uint8)
    box = _draw_plate(frame, plate, (20, 20, width_px, height))
    return frame, box


def _candidate(box, confidence: float = 0.9) -> PlateCandidate:
    return PlateCandidate(plate_bbox_xyxy=tuple(box), detector_confidence=confidence)


def _read_standalone(engine, plate: str, width_px: int, background: int = 128):
    frame, box = _render_plate(plate, width_px, background)
    if box is None:
        return None
    return engine.read(frame, _candidate(box))


class _TruthSource:
    """Minimum viable ground-truth source for the oracle backend.

    Hand-built rather than generated, so a test can put a plate at exactly 24 px, mark
    exactly one vehicle illegible, or place two plates one pixel apart. The real
    synthetic source is exercised too -- see the tests using the corpus fixture -- but it
    cannot be asked for a specific width on demand.
    """

    def __init__(self, frames: dict[int, FrameTruth]) -> None:
        self._frames = frames

    def truth_for_envelope(self, envelope):
        pts = getattr(envelope, "pts_ms", None)
        if pts is None:
            return None
        return self._frames.get(int(pts))


def _vehicle(
    plate: str,
    box,
    *,
    vehicle_id: int = 1,
    legible: bool = True,
    visible: float = 1.0,
) -> VehicleTruth:
    return VehicleTruth(
        vehicle_id=vehicle_id,
        plate=plate,
        vehicle_type="car",
        vehicle_bbox_xyxy=(box[0] - 20, box[1] - 40, box[2] + 20, box[3] + 10),
        plate_bbox_xyxy=tuple(box),
        plate_width_px=box[2] - box[0],
        plate_legible=legible,
        plate_visible_fraction=visible,
    )


class _Envelope:
    """Just enough of a FrameEnvelope for the backends that only read identity."""

    def __init__(self, frame_index: int, pts_ms, frame_bgr=None) -> None:
        self.frame_index = frame_index
        self.pts_ms = pts_ms
        self.frame_bgr = frame_bgr


class _FlakyOCR(BaseOCR):
    """Reads only on the variants it is told to, so the loop's counters are visible.

    The point of it: `variants_tried` must report the *configured* count and not the
    number of variants that came back with a string. A backend that fails on the
    grayscale variant and succeeds on the others is a real and very quiet failure mode
    -- it shows up only as that variant never appearing in `variant_wins` -- and
    `agreement` is only interpretable if the denominator is the count that was asked
    for.
    """

    def __init__(self, answers: dict[str, tuple[str, float]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.answers = answers
        self.seen: list[str] = []

    def _load(self) -> None:
        pass

    def _read_crop(self, crop_bgr, candidate):
        # Which variant is running is not passed in; the shape it produced identifies it
        # well enough for a stub, so answers are consumed in configured order instead.
        index = len(self.seen)
        name = self.variants[index] if index < len(self.variants) else "?"
        self.seen.append(name)
        answer = self.answers.get(name)
        if answer is None:
            return None
        text, confidence = answer
        return OCRRead(text=text, confidence=confidence, variant="ignored")

    @property
    def model_name(self) -> str:
        return "flaky"

    @property
    def model_version(self) -> str:
        return "0"

    @property
    def license_name(self) -> str:
        return "not-applicable"


# ------------------------------------------------------------------- the fixture itself


def test_the_fixture_is_the_one_every_measured_claim_names(corpus: Corpus) -> None:
    """400 generated frames, 134 emitted, 12 vehicles, 7 distinct plate strings.

    Read as 400 emitted frames rather than 400 generated ones, this fixture produces a
    different table -- which is how the plate stage's claims came to be unreproducible.
    """
    assert corpus.frames_emitted == EXPECTED_EMITTED_FRAMES
    assert len(corpus.vehicle_ids) == EXPECTED_VEHICLES
    assert sorted({r.plate for r in corpus.rows}) == list(CORPUS_PLATES)


def test_the_corpus_offers_569_crops_split_550_legible_and_19_illegible(
    corpus: Corpus,
) -> None:
    assert len(corpus.rows) == EXPECTED_CROPS
    assert len(corpus.legible) == EXPECTED_LEGIBLE
    assert len(corpus.illegible) == EXPECTED_ILLEGIBLE


def test_every_crop_matched_a_truth_plate(corpus: Corpus) -> None:
    """The oracle plate detector's boxes come from truth, so nothing should be unmatched.

    If this fails the width table is being computed against the wrong vehicle and every
    accuracy figure below is meaningless rather than merely wrong.
    """
    assert [r for r in corpus.rows if r.plate is None] == []


def test_one_pass_serves_both_tables(corpus: Corpus) -> None:
    """The legible subset is pixel-identical to what require_legible=True offers.

    This is what licenses deriving the width table and the fabrication rows from one
    pass. A six-variant pass over this corpus costs about a minute, so the alternative
    is not free.
    """
    source = build_source(dict(SOURCE))
    source.open()
    detector = build_detector({"name": "oracle", "miss_rate": 0.0}, source=source)
    detector.load()
    tracker = build_tracker(
        {"name": "bytetrack"}, CAMERA, source.session_id, source=source
    )
    plates = build_plate_detector(
        {"name": "oracle", "require_legible": True}, source=source
    )
    plates.load()
    cutter = build_ocr_engine({"name": "template"})
    cutter.load()

    strict: list[tuple[int, int, tuple, np.ndarray]] = []
    for envelope in source:
        detections = detector.detect_envelope(envelope)
        tracks = tracker.update(
            detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
        )
        for track_id, candidate in plates.detect_plates_envelope(
            envelope, tracks
        ).items():
            crop = cutter.cut_crop(envelope.frame_bgr, candidate)
            strict.append(
                (
                    envelope.frame_index,
                    track_id,
                    candidate.plate_bbox_xyxy,
                    None if crop is None else crop.copy(),
                )
            )
    source.close()
    detector.close()
    plates.close()
    cutter.close()

    assert len(strict) == EXPECTED_LEGIBLE
    for row, (frame_index, track_id, box, crop) in zip(corpus.legible, strict):
        assert (row.frame_index, row.track_id) == (frame_index, track_id)
        assert row.candidate.plate_bbox_xyxy == box
        assert np.array_equal(row.crop, crop)


def test_the_illegible_plates_are_occluded_rather_than_small(corpus: Corpus) -> None:
    """All 19 are 59-69 px, well above the legibility floor. Occlusion, not size.

    Which is what makes them the right fabrication test: a plate refused for being 20 px
    wide tests the width floor, whereas a 65 px plate that truth says is not visible
    tests whether the matcher will invent a string for pixels that contain no plate.
    """
    widths = sorted(r.width for r in corpus.illegible)
    assert widths[0] == 59
    assert widths[-1] == 69
    assert all(w > MIN_LEGIBLE_PLATE_WIDTH_PX for w in widths)


# ------------------------------------------------------------------- OCRRead, FrameRef


def test_ocrread_defaults_to_one_variant_tried_and_agreeing() -> None:
    read = OCRRead(text="GJ01AB1234", confidence=0.7, variant="raw")
    assert (read.variants_tried, read.variants_agreeing) == (1, 1)
    assert read.agreement == 1.0


def test_agreement_is_the_share_of_variants_that_produced_the_same_string() -> None:
    read = OCRRead(
        text="GJ01AB1234",
        confidence=0.7,
        variant="upscale_2x",
        variants_tried=6,
        variants_agreeing=5,
    )
    assert read.agreement == pytest.approx(5 / 6)


def test_agreement_is_zero_rather_than_a_division_error_when_nothing_was_tried() -> None:
    read = OCRRead(text="", confidence=0.0, variant="raw", variants_tried=0)
    assert read.agreement == 0.0


def test_ocrread_is_frozen() -> None:
    read = OCRRead(text="GJ01AB1234", confidence=0.7, variant="raw")
    with pytest.raises(Exception):
        read.text = "MH12DE9812"  # type: ignore[misc]


def test_char_confidences_are_optional() -> None:
    """PaddleOCR builds that expose per-character scores are not universal.

    Optional rather than required so the stage degrades to whole-string voting on an
    older build instead of failing.
    """
    assert OCRRead(text="A", confidence=0.5, variant="raw").char_confidences is None


def test_frame_ref_carries_identity_and_no_pixels() -> None:
    """Four integers' worth of information, deliberately not the envelope.

    Holding the envelope would pin frame_bgr: four crops per track across eight open
    tracks is eight whole frames retained, 48 MB at 1920x1080, invisible and looking
    like a decoder leak.
    """
    ref = FrameRef(
        camera_id="cam04", stream_session_id="s", frame_index=3, pts_ms=360
    )
    assert set(ref.__dataclass_fields__) == {
        "camera_id",
        "stream_session_id",
        "frame_index",
        "pts_ms",
    }
    with pytest.raises(Exception):
        ref.pts_ms = 400  # type: ignore[misc]


def test_ocrread_is_lane_internal_and_not_in_the_contract_module() -> None:
    """The contract boundary carries what other lanes depend on.

    No other lane needs to know this stage tries six preprocessing variants, so OCRRead
    stays out of ai/contracts/stages.py. If it moves there, that is a contract change
    and needs the document changed with it.
    """
    import ai.contracts.stages as stages

    assert not hasattr(stages, "OCRRead")
    assert not hasattr(stages, "FrameRef")


# ------------------------------------------------------------------------ the width floor


def test_the_floor_is_24_px_and_the_pad_is_3_px() -> None:
    assert MIN_OCR_PLATE_WIDTH_PX == 24
    assert PLATE_CROP_PAD_PX == 3


def test_the_floor_sits_below_the_renderers_legibility_threshold() -> None:
    """24 against 30, and the gap is deliberate rather than an oversight.

    24 px is where the signal is absent for any backend; 30 px is a property of this
    particular renderer, which a trained recogniser on real frames may well beat.
    Hard-coding 30 into ai/ocr would bake a fixture's limitation into the shipping
    stage, so the gap stays and check_ocr_width_floor marks it.
    """
    assert MIN_OCR_PLATE_WIDTH_PX < MIN_LEGIBLE_PLATE_WIDTH_PX


def test_a_plate_below_the_floor_is_refused_before_any_backend_runs() -> None:
    engine = _FlakyOCR({"raw": ("GJ01AB1234", 0.9)}, variants=("raw",))
    engine.load()
    crop = np.full((12, 29, 3), 200, dtype=np.uint8)
    assert engine.read_crop(crop, _candidate((0, 0, 23, 12))) is None
    assert engine.refused_small == 1
    assert engine.reads_attempted == 0
    assert engine.seen == []


def test_a_plate_at_exactly_the_floor_is_read() -> None:
    engine = _FlakyOCR({"raw": ("GJ01AB1234", 0.9)}, variants=("raw",))
    engine.load()
    crop = np.full((12, 30, 3), 200, dtype=np.uint8)
    read = engine.read_crop(crop, _candidate((0, 0, 24, 12)))
    assert read is not None and read.text == "GJ01AB1234"
    assert engine.refused_small == 0


def test_the_floor_is_checked_against_the_candidate_not_the_crop() -> None:
    """A 20 px plate stays refused however large the crop handed in.

    The candidate carries the plate's width in the *scene*; the crop's width is that
    plus padding, or double it after upscale_2x. Checking the crop would let a 20 px
    plate through as soon as anything enlarged it, which is the exact failure the floor
    exists to prevent -- the characters are not in the data and interpolation cannot put
    them there.
    """
    engine = _FlakyOCR({"raw": ("GJ01AB1234", 0.9)}, variants=("raw",))
    engine.load()
    huge = np.full((200, 800, 3), 200, dtype=np.uint8)
    assert engine.read_crop(huge, _candidate((0, 0, 20, 6))) is None
    assert engine.refused_small == 1


def test_an_empty_crop_is_counted_separately_from_a_small_plate() -> None:
    """Two different failures. A refusal is a decision; an empty crop is a bug upstream."""
    engine = _FlakyOCR({"raw": ("X", 0.9)}, variants=("raw",))
    engine.load()
    assert engine.read_crop(None, _candidate((0, 0, 60, 15))) is None
    assert engine.read_crop(np.empty((0, 0, 3), np.uint8), _candidate((0, 0, 60, 15))) is None
    assert (engine.crops_empty, engine.refused_small) == (2, 0)


def test_check_ocr_width_floor_is_silent_off_the_synthetic_corpus() -> None:
    """The invariant is about ground truth, and only the synthetic source has any."""
    assert check_ocr_width_floor({"name": "template"}, synthetic=False) is None


def test_check_ocr_width_floor_warns_at_the_default_on_synthetic() -> None:
    message = check_ocr_width_floor({"name": "template"}, synthetic=True)
    assert message is not None
    assert str(MIN_OCR_PLATE_WIDTH_PX) in message
    assert str(MIN_LEGIBLE_PLATE_WIDTH_PX) in message


def test_check_ocr_width_floor_is_silent_when_raised_to_the_legibility_floor() -> None:
    config = {"name": "template", "min_plate_width_px": MIN_LEGIBLE_PLATE_WIDTH_PX}
    assert check_ocr_width_floor(config, synthetic=True) is None


def test_the_warning_does_not_promise_the_engines_refuse_that_band() -> None:
    """It used to, and the corpus contradicts it.

    The earlier text told the operator that score_too_low and ink_not_found "should
    account for every plate in that band". Measured, both of the two crops this corpus
    offers in the 24-29 px band came back with a string. The warning now says the
    engines do not reliably refuse it, which is what the next test measures.
    """
    message = check_ocr_width_floor({"name": "template"}, synthetic=True)
    assert message is not None
    assert "do not reliably refuse" in message


def test_the_band_the_warning_names_is_not_protected_by_the_engine(
    corpus: Corpus,
) -> None:
    """Both crops in the 24-29 px band returned a string, and neither was correct.

    27 px reads 22BH for 22BH1234AA at 0.478 -- a truncation -- and 28 px reads FHE1R for
    GJ05JK4521 at 0.437, which is an invention. The score floor rejects neither. This is
    the measured content of the warning above and the reason it is a warning rather than
    a refusal: reading the band is a legitimate experiment, and measuring how often a
    backend fabricates is exactly how you find out whether its refusal threshold is set
    right.
    """
    band = [
        (row, read)
        for row, read in zip(corpus.rows, corpus.reads())
        if 24 <= row.width < MIN_LEGIBLE_PLATE_WIDTH_PX
    ]
    assert len(band) == 2
    assert all(read is not None for _, read in band)
    assert all(read.text != row.plate for row, read in band)
    texts = {read.text for _, read in band}
    assert texts == {"22BH", "FHE1R"}


def test_the_six_crops_below_the_floor_are_all_refused(corpus: Corpus) -> None:
    below = [r for r in corpus.rows if r.width < MIN_OCR_PLATE_WIDTH_PX]
    assert len(below) == 6
    assert all(r.width >= 20 for r in below)
    reads = {id(r): read for r, read in zip(corpus.rows, corpus.reads())}
    assert all(reads[id(r)] is None for r in below)


# ------------------------------------------------------------------------------ cropping


def test_cut_crop_pads_by_pad_px_and_clamps_to_the_frame() -> None:
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = engine.cut_crop(frame, _candidate((50, 40, 110, 55)))
    assert crop.shape[:2] == (15 + 2 * PLATE_CROP_PAD_PX, 60 + 2 * PLATE_CROP_PAD_PX)
    edge = engine.cut_crop(frame, _candidate((0, 0, 60, 15)))
    assert edge.shape[:2] == (15 + PLATE_CROP_PAD_PX, 60 + PLATE_CROP_PAD_PX)


def test_cut_crop_returns_a_view_so_a_deferred_caller_must_copy() -> None:
    """Documented, and the reason the corpus fixture copies every crop it caches."""
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = engine.cut_crop(frame, _candidate((50, 40, 110, 55)))
    assert np.shares_memory(crop, frame)


def test_cut_crop_returns_none_only_when_the_box_misses_the_frame() -> None:
    """A zero-area box still yields pixels, because the pad rescues it.

    Worth pinning rather than assuming the opposite: a 0x0 box at (50,40) crops the 6x6
    square the padding describes, which is a real array that a backend would happily
    read. Nothing in the cropping path rejects it -- the width floor does, against the
    candidate rather than the crop, which is the test two above this one.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame = np.zeros((100, 200, 3), np.uint8)
    assert engine.cut_crop(frame, _candidate((50, 40, 50, 40))).shape[:2] == (
        2 * PLATE_CROP_PAD_PX,
        2 * PLATE_CROP_PAD_PX,
    )
    assert engine.cut_crop(frame, _candidate((500, 500, 560, 515))) is None


def test_read_crop_reproduces_read_at_the_default_pad() -> None:
    """The deferred read must equal the immediate one or the two cannot be compared.

    Which is why cut_crop lives on the engine rather than in the pipeline: a caller that
    guessed 4 px of padding where the backend uses 3 would produce reads that look fine
    and are not comparable with a non-deferred run.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame, box = _render_plate("GJ01AB1234", 90)
    candidate = _candidate(box)
    immediate = engine.read(frame, candidate)
    deferred = engine.read_crop(engine.cut_crop(frame, candidate).copy(), candidate)
    assert immediate is not None
    assert (immediate.text, immediate.confidence) == (deferred.text, deferred.confidence)


# ------------------------------------------------------------------------------ variants


def test_the_six_variants_are_the_six_the_document_lists() -> None:
    assert DEFAULT_VARIANTS == (
        "raw",
        "grayscale",
        "contrast_stretch",
        "adaptive_threshold",
        "upscale_2x",
        "sharpen",
    )
    assert set(variant_names()) == set(DEFAULT_VARIANTS)


def test_variants_are_applied_singly_and_never_composed() -> None:
    """Six singly. Composed in every order and combination it would be 1957.

    At 0.7 ms a read on eight vehicles a frame that is no longer a preprocessing step,
    it is the frame budget -- and composition also destroys the diagnostic value of
    variant_wins, which is the whole reason the winner is reported.
    """
    assert len(variant_names()) == 6


@pytest.mark.parametrize("name", DEFAULT_VARIANTS)
def test_no_variant_writes_into_its_input(name: str) -> None:
    """The test ai/ocr/base.py:355 promises exists.

    cut_crop returns a view into the frame. A variant that wrote in place would corrupt
    the frame for every stage after this one -- including the snapshot that ends up in
    the event -- and the symptom would be "the second vehicle in each frame reads worse
    than the first", which is nobody's first guess.
    """
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(40, 120, 3), dtype=np.uint8)
    before = frame.copy()
    crop = frame[10:30, 20:100]

    out = apply_variant(crop, name)
    assert out is not None
    assert not np.shares_memory(out, frame)
    out[:] = 0
    assert np.array_equal(frame, before)


def test_raw_copies_rather_than_returning_a_view() -> None:
    """One wasted memcpy on a 60x20 crop, against a whole class of bug."""
    crop = np.full((20, 60, 3), 128, np.uint8)
    out = apply_variant(crop, "raw")
    assert np.array_equal(out, crop)
    assert not np.shares_memory(out, crop)


def test_an_unknown_variant_raises_rather_than_passing_through() -> None:
    """A typo must not become "raw was tried twice", which reads as a normal run."""
    crop = np.full((20, 60, 3), 128, np.uint8)
    with pytest.raises(KeyError) as excinfo:
        apply_variant(crop, "upscale_4x")
    assert "upscale_4x" in str(excinfo.value)


@pytest.mark.parametrize("name", DEFAULT_VARIANTS)
def test_a_zero_size_crop_returns_none_from_every_variant(name: str) -> None:
    assert apply_variant(np.empty((0, 0, 3), np.uint8), name) is None


def test_grayscale_uses_bt601_weights_in_bgr_order() -> None:
    crop = np.zeros((4, 4, 3), np.uint8)
    crop[..., 0], crop[..., 1], crop[..., 2] = 100, 150, 200
    out = apply_variant(crop, "grayscale")
    expected = int(0.114 * 100 + 0.587 * 150 + 0.299 * 200)
    assert out[0, 0, 0] == expected


def test_grayscale_returns_three_channels() -> None:
    """So a backend expecting colour does not fail on exactly one variant.

    That failure shows up only as this variant never appearing in variant_wins, which is
    very hard to notice and looks like the variant simply not being useful.
    """
    out = apply_variant(np.full((6, 20, 3), 90, np.uint8), "grayscale")
    assert out.ndim == 3 and out.shape[2] == 3
    assert np.array_equal(out[..., 0], out[..., 2])


def test_contrast_stretch_maps_the_percentiles_to_the_full_range() -> None:
    rng = np.random.default_rng(3)
    grey = rng.integers(90, 150, size=(20, 60), dtype=np.uint8)
    crop = np.repeat(grey[:, :, None], 3, axis=2)
    out = apply_variant(crop, "contrast_stretch")
    assert out.min() == 0
    assert out.max() == 255


def test_contrast_stretch_falls_back_to_grayscale_on_a_flat_crop() -> None:
    """Dividing by a sub-unit range turns sensor noise into full-scale speckle.

    The result reads as confident nonsense rather than as unreadable, which is the wrong
    direction for every rule in this package.
    """
    crop = np.full((20, 60, 3), 128, np.uint8)
    assert np.array_equal(
        apply_variant(crop, "contrast_stretch"), apply_variant(crop, "grayscale")
    )


def test_the_stretch_percentiles_are_not_zero_and_a_hundred() -> None:
    """One hot pixel setting the contrast of the whole image is how a stretch harms."""
    assert STRETCH_LOW_PERCENTILE == 2.0
    assert STRETCH_HIGH_PERCENTILE == 98.0


def test_adaptive_threshold_emits_only_black_and_white() -> None:
    rng = np.random.default_rng(5)
    crop = rng.integers(0, 256, size=(20, 60, 3), dtype=np.uint8)
    out = apply_variant(crop, "adaptive_threshold")
    assert set(np.unique(out).tolist()) <= {0, 255}


def test_adaptive_threshold_keeps_ink_dark_on_a_light_background() -> None:
    """Every OCR engine and the glyph templates both assume dark-on-light."""
    crop = np.full((20, 60, 3), 200, np.uint8)
    crop[8:12, 20:26] = 20
    out = apply_variant(crop, "adaptive_threshold")
    assert out[10, 23, 0] == 0
    assert out[2, 2, 0] == 255


def test_adaptive_threshold_needs_a_pixel_meaningfully_darker_than_its_neighbours() -> None:
    """Without the offset, flat regions split half and half on noise: a field of speckle."""
    assert ADAPTIVE_OFFSET > 0
    flat = np.full((20, 60, 3), 128, np.uint8)
    assert (apply_variant(flat, "adaptive_threshold") == 255).all()


def test_upscale_2x_doubles_both_axes() -> None:
    out = apply_variant(np.full((10, 30, 3), 128, np.uint8), "upscale_2x")
    assert out.shape[:2] == (20, 60)


def test_upscale_2x_uses_the_half_pixel_centre_convention() -> None:
    """The naive dst*scale convention shifts by half a pixel times the scale factor.

    On a 30 px plate doubled to 60 that lands the whole read half a character to the
    left -- and it is invisible on any test image without a hard edge in it.
    """
    crop = np.zeros((1, 2, 3), np.uint8)
    crop[0, 1] = 255
    out = _resize_bilinear(crop, 4, 2)
    assert out[0, :, 0].tolist() == [0, 63, 191, 255]
    # The naive convention would give this instead, and both rows would still match.
    assert out[0, :, 0].tolist() != [0, 127, 255, 255]


def test_sharpen_leaves_a_flat_crop_alone() -> None:
    """A mild unsharp mask on nothing is nothing. Ringing is what "mild" buys."""
    crop = np.full((20, 60, 3), 100, np.uint8)
    out = apply_variant(crop, "sharpen")
    assert (out == 100).all()


def test_sharpen_raises_contrast_across_an_edge() -> None:
    crop = np.full((20, 60, 3), 120, np.uint8)
    crop[:, 30:] = 160
    out = apply_variant(crop, "sharpen")
    assert out[10, 29, 0] < 120
    assert out[10, 30, 0] > 160


# -------------------------------------------------------------------------- the variant loop


def test_variants_tried_is_the_configured_count_not_the_successful_count() -> None:
    """Six asked for, two answered, and the denominator is six.

    Reporting two would make a read that failed on four variants look unanimous.
    """
    engine = _FlakyOCR(
        {"raw": ("GJ01AB1234", 0.6), "upscale_2x": ("GJ01AB1234", 0.8)},
        variants=DEFAULT_VARIANTS,
    )
    engine.load()
    read = engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))
    assert read is not None
    assert read.variants_tried == 6
    assert read.variants_agreeing == 2
    assert read.agreement == pytest.approx(2 / 6)
    assert engine.reads_attempted == 6
    assert engine.reads_empty == 4


def test_the_highest_confidence_variant_wins_and_is_credited() -> None:
    engine = _FlakyOCR(
        {"raw": ("AAA", 0.6), "grayscale": ("BBB", 0.9), "sharpen": ("CCC", 0.7)},
        variants=("raw", "grayscale", "sharpen"),
    )
    engine.load()
    read = engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))
    assert (read.text, read.confidence, read.variant) == ("BBB", 0.9, "grayscale")
    assert engine.variant_wins == {"grayscale": 1}


def test_the_caller_owns_the_variant_field_not_the_backend() -> None:
    """A backend cannot label its own read, and _FlakyOCR labels every read "ignored".

    Worth pinning because the three stubs all set this field to their own name and none
    of those names ever reaches an event -- variant_wins is a histogram over
    *preprocessing* variants, which is the only reading under which it is actionable.
    A backend allowed to name the winner could report "template" 523 times and the
    resolution-starved finding in ai/ocr/preprocess.py's docstring would be invisible.
    """
    engine = _FlakyOCR({"sharpen": ("GJ01AB1234", 0.6)}, variants=("raw", "sharpen"))
    engine.load()
    read = engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))
    assert read.variant == "sharpen"
    assert engine.variant_wins == {"sharpen": 1}


def test_agreeing_counts_only_the_variants_that_produced_the_winning_string() -> None:
    engine = _FlakyOCR(
        {"raw": ("AAA", 0.6), "grayscale": ("AAA", 0.5), "sharpen": ("BBB", 0.9)},
        variants=("raw", "grayscale", "sharpen"),
    )
    engine.load()
    read = engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))
    assert read.text == "BBB"
    assert read.variants_agreeing == 1


def test_a_backend_that_reads_nothing_returns_none_and_counts_it() -> None:
    """plate: null is a valid answer, so this is a result rather than an error."""
    engine = _FlakyOCR({}, variants=("raw", "grayscale"))
    engine.load()
    assert engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15))) is None
    assert (engine.reads_attempted, engine.reads_empty, engine.reads_returned) == (2, 2, 0)


def test_an_empty_string_is_treated_as_no_read() -> None:
    engine = _FlakyOCR({"raw": ("", 0.9)}, variants=("raw",))
    engine.load()
    assert engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15))) is None
    assert engine.reads_empty == 1


def test_the_engine_never_rescales_the_backends_confidence() -> None:
    """ai/fusion weights by this number directly, so a rescale here is a fabricated
    calibration that propagates all the way to the event."""
    engine = _FlakyOCR({"raw": ("GJ01AB1234", 0.3141)}, variants=("raw",))
    engine.load()
    read = engine.read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))
    assert read.confidence == 0.3141


def test_the_template_default_is_one_variant_so_agreement_is_trivially_full(
    corpus: Corpus,
) -> None:
    """Which is exactly why every agreement assertion above passes variants explicitly.

    TEMPLATE_DEFAULT_VARIANTS' comment says so in as many words: a test that needs the
    variant loop exercised non-trivially must configure it.
    """
    assert TEMPLATE_DEFAULT_VARIANTS == ("upscale_2x",)
    reads = [read for read in corpus.reads() if read is not None]
    assert reads
    assert {r.variants_tried for r in reads} == {1}
    assert {r.agreement for r in reads} == {1.0}


def test_a_six_variant_template_never_agrees_at_two_three_or_four(
    corpus: Corpus,
) -> None:
    """Measured on every twentieth legible crop: 1 in 9 reads, 5 in 10, 6 in 7.

    Never 2, 3 or 4, and that is the reason agreement is worth reporting at all rather
    than a curiosity. The four greyscale-derived variants always agree with each other,
    so the vote is really raw against upscale_2x with four abstentions attached to
    whichever of them wins -- an agreement of 5/6 therefore means much less than it
    looks like. The full-corpus histogram has the same shape: 1 in 178, 5 in 250, 6 in
    109.
    """
    sample = corpus.sample()
    assert len(sample) == 28
    engine = build_ocr_engine({"name": "template", "variants": list(DEFAULT_VARIANTS)})
    engine.load()
    reads = [engine.read_crop(r.crop, r.candidate) for r in sample]
    stats = engine.stats()
    engine.close()

    got = [read for read in reads if read is not None]
    assert len(got) == 26
    assert {r.variants_tried for r in got} == {6}
    histogram: dict[int, int] = {}
    for read in got:
        histogram[read.variants_agreeing] = histogram.get(read.variants_agreeing, 0) + 1
    assert histogram == {1: 9, 5: 10, 6: 7}
    assert stats["variant_wins"] == {"upscale_2x": 16, "raw": 10}


def test_only_raw_and_upscale_2x_ever_win_a_plate(corpus: Corpus) -> None:
    """The other four cost two thirds of the stage's time and win nothing.

    Over the full corpus: of the 537 reads the six-variant configuration returns,
    upscale_2x wins 276 and raw 261; grayscale, contrast_stretch, adaptive_threshold and
    sharpen win zero. That upscale_2x dominates is the diagnostic ai/ocr/preprocess.py
    predicted -- this stage is resolution-starved, not model-starved -- and it says
    nothing about PaddleOCR on real frames, where adaptive_threshold exists for glare
    this renderer never draws.
    """
    sample = corpus.sample()
    engine = build_ocr_engine({"name": "template", "variants": list(DEFAULT_VARIANTS)})
    engine.load()
    for row in sample:
        engine.read_crop(row.crop, row.candidate)
    wins = engine.stats()["variant_wins"]
    engine.close()
    assert set(wins) <= {"raw", "upscale_2x"}


def test_the_four_greyscale_derived_variants_agree_with_each_other(
    corpus: Corpus,
) -> None:
    """The mechanism behind the 5/6 agreements, tested directly rather than inferred.

    grayscale, contrast_stretch, adaptive_threshold and sharpen all reduce the crop to
    one channel at the same size, and the ink mask is taken relative to the plate face's
    own range -- so a monotone rescale of the grey values cannot move it. Zero
    disagreements over the sample.
    """
    grey_derived = ("grayscale", "contrast_stretch", "adaptive_threshold", "sharpen")
    engines = {}
    for name in grey_derived:
        engine = build_ocr_engine({"name": "template", "variants": [name]})
        engine.load()
        engines[name] = engine

    for row in corpus.sample()[::4]:
        texts = set()
        for engine in engines.values():
            read = engine.read_crop(row.crop, row.candidate)
            texts.add(None if read is None else read.text)
        assert len(texts) == 1, f"{row.width} px split the greyscale variants: {texts}"

    for engine in engines.values():
        engine.close()


def test_one_variant_costs_one_read_per_plate_and_six_cost_six(
    corpus: Corpus,
) -> None:
    """The machine-independent form of the cost table.

    TemplateOCR's docstring reports 95.0 ms per plate for six variants against 16.8 ms
    for one, a 5.7x ratio on this machine under this load. Those two columns are
    deliberately not pinned -- a timing assertion in CI is a flake with a ticket
    attached -- but reads per plate is the quantity the ratio comes from and it cannot
    drift with the hardware.
    """
    sample = corpus.sample()
    counts = {}
    for label, variants in (("one", TEMPLATE_DEFAULT_VARIANTS), ("six", DEFAULT_VARIANTS)):
        engine = build_ocr_engine({"name": "template", "variants": list(variants)})
        engine.load()
        for row in sample:
            engine.read_crop(row.crop, row.candidate)
        counts[label] = engine.stats()["reads_attempted"]
        engine.close()
    assert counts["six"] == 6 * counts["one"]


def test_upscale_2x_alone_beats_raw_alone_on_the_sample(corpus: Corpus) -> None:
    """Directional, because the full-corpus figures are 122 against 94.

    Adding raw back to upscale_2x is what *loses* a track: raw occasionally returns a
    higher-confidence wrong string than upscale_2x's correct one, which is max-of-N
    selection bias behaving exactly as OCRRead.agreement was written to expose.
    """
    sample = corpus.sample()
    exact = {}
    for name in ("upscale_2x", "raw"):
        engine = build_ocr_engine({"name": "template", "variants": [name]})
        engine.load()
        exact[name] = sum(
            1
            for row in sample
            if (read := engine.read_crop(row.crop, row.candidate)) is not None
            and read.text == row.plate
        )
        engine.close()
    assert exact == {"upscale_2x": 9, "raw": 6}


# ------------------------------------------------------------------- the measured table


@pytest.mark.parametrize("bucket", list(WIDTH_TABLE))
def test_the_width_table_reproduces(corpus: Corpus, bucket: str) -> None:
    """Crops, exact reads and refusals per width bucket.

    This table was previously stated at ByteTrack's min_hits=1, which offers 564 crops
    rather than 550 -- an undocumented non-default rig behind a documented number. It is
    now the shipped tracker config, so a plain build reproduces it.
    """
    crops, exact, _, _, refused = WIDTH_TABLE[bucket]
    rows = [
        (row, read)
        for row, read in corpus.legible_reads()
        if width_bucket(row.width) == bucket
    ]
    assert len(rows) == crops
    assert sum(1 for row, read in rows if read is not None and read.text == row.plate) == exact
    assert sum(1 for _, read in rows if read is None) == refused


@pytest.mark.parametrize("bucket", list(WIDTH_TABLE))
def test_positional_character_accuracy_reproduces(corpus: Corpus, bucket: str) -> None:
    """Matching characters in matching positions, over the truth string's length.

    Stated, because it was not: the earlier table gave a char-acc column with no
    definition and two rows reading 0.000, which no positional definition can produce
    for buckets that contain reads. A refused crop contributes zero matches and its
    truth length to the denominator, which is why this is not simply a softer version of
    the exact column.
    """
    _, _, positional, _, _ = WIDTH_TABLE[bucket]
    rows = [
        (row, read)
        for row, read in corpus.legible_reads()
        if width_bucket(row.width) == bucket
    ]
    matched = sum(
        _positional_matches(None if read is None else read.text, row.plate)
        for row, read in rows
    )
    chars = sum(len(row.plate) for row, _ in rows)
    assert matched / chars == pytest.approx(positional, abs=0.001)


@pytest.mark.parametrize("bucket", list(WIDTH_TABLE))
def test_the_edit_distance_column_reproduces(corpus: Corpus, bucket: str) -> None:
    """The softer column, and the gap between the two is truncation specifically.

    Positional scoring gives a truncation almost nothing: "J3C4567" for GJ3C4567 matches
    in no position at all, while its edit distance is 1. Both columns are reported
    because the difference between them is a measurement of how much of this stage's
    error is dropped leading characters rather than wrong ones.
    """
    _, _, _, edit, _ = WIDTH_TABLE[bucket]
    rows = [
        (row, read)
        for row, read in corpus.legible_reads()
        if width_bucket(row.width) == bucket
    ]
    matched = sum(
        _edit_matches(None if read is None else read.text, row.plate) for row, read in rows
    )
    chars = sum(len(row.plate) for row, _ in rows)
    assert matched / chars == pytest.approx(edit, abs=0.001)


def test_the_all_row_reproduces(corpus: Corpus) -> None:
    """550 crops, 122 exact, 0.222 overall, 36 refused.

    Reported per bucket and never as the single 0.222, which is a statement about this
    corpus's width distribution rather than about the matcher. Contracts section 7.2:
    no accuracy number may be reported as a single average.
    """
    crops, exact, positional, edit, refused = ALL_ROW
    rows = corpus.legible_reads()
    assert len(rows) == crops
    assert sum(1 for row, read in rows if read is not None and read.text == row.plate) == exact
    assert sum(1 for _, read in rows if read is None) == refused
    chars = sum(len(row.plate) for row, _ in rows)
    assert (
        sum(
            _positional_matches(None if read is None else read.text, row.plate)
            for row, read in rows
        )
        / chars
    ) == pytest.approx(positional, abs=0.001)
    assert (
        sum(_edit_matches(None if read is None else read.text, row.plate) for row, read in rows)
        / chars
    ) == pytest.approx(edit, abs=0.001)


def test_the_table_is_monotonic_in_width(corpus: Corpus) -> None:
    """The only shape this table is allowed to have.

    It came out non-monotonic twice during development and both times that was a bug in
    _ink_mask, not a small sample -- once from a fractional cap on ring stripping and
    once from assuming the plate is the brightest thing in the crop.
    """
    order = [key for key in BUCKET_KEYS if key in WIDTH_TABLE]
    rates = []
    positional = []
    for bucket in order:
        crops, exact, acc, _, _ = WIDTH_TABLE[bucket]
        rates.append(exact / crops)
        positional.append(acc)
    assert rates == sorted(rates, reverse=True)
    assert positional == sorted(positional, reverse=True)


def test_the_crop_of_exactly_100_px_sits_in_the_lower_bucket(corpus: Corpus) -> None:
    """ai/metrics.py's ">100" is strict, and this is the crop that proves it matters.

    Frame 78, GJ01AB1234 read as GJ01AB1Z34. Counting it in the top row gave 28/110
    where the metrics module gives 27/111, which is how a table can disagree with the
    bucketing function it claims to use.
    """
    assert width_bucket(100) == "80-100"
    exact = [row for row in corpus.legible if row.width == 100]
    assert len(exact) == 1
    assert exact[0].frame_index == 78
    assert exact[0].plate == "GJ01AB1234"


def test_per_frame_precision_is_0_237(corpus: Corpus) -> None:
    """122 correct of 514 reads. The number that matters is not in the width table."""
    rows = [(row, read) for row, read in corpus.legible_reads() if read is not None]
    correct = [1 for row, read in rows if read.text == row.plate]
    assert len(rows) == 514
    assert len(correct) == 122
    assert len(correct) / len(rows) == pytest.approx(0.237, abs=0.001)


def test_the_best_read_of_each_track_is_right_for_all_twelve(corpus: Corpus) -> None:
    """Frame accuracy 24%, track accuracy 100%, from the same reads.

    That gap is the entire empirical case for temporal fusion existing. It comes with a
    caveat and the caveat is in the docstring: twelve vehicles is seven distinct plate
    strings, five of them drawn twice, so this is a sanity check that fusion is possible
    on this corpus rather than a measurement of how well fusion works.
    """
    best: dict[int, tuple] = {}
    for row, read in corpus.legible_reads():
        if read is None:
            continue
        current = best.get(row.track_id)
        if current is None or read.confidence > current[1].confidence:
            best[row.track_id] = (row, read)
    assert len(best) == 12
    assert sum(1 for row, read in best.values() if read.text == row.plate) == 12


def test_the_confidence_distributions_overlap_across_almost_their_whole_range(
    corpus: Corpus,
) -> None:
    """Confidence ranks without separating, which is all a weighted vote needs.

        correct   n=122   min 0.469   p10 0.506   median 0.602   max 0.772
        wrong     n=392   min 0.420   p10 0.436   median 0.511   max 0.717

    A single threshold anywhere in that overlap discards correct reads to remove wrong
    ones, which is why ai/fusion weights by this number and does not threshold on it.
    """
    rows = [(row, read) for row, read in corpus.legible_reads() if read is not None]
    correct = sorted(read.confidence for row, read in rows if read.text == row.plate)
    wrong = sorted(read.confidence for row, read in rows if read.text != row.plate)
    assert (len(correct), len(wrong)) == (122, 392)

    def p10(values):
        return values[int(0.10 * (len(values) - 1))]

    def median(values):
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    assert correct[0] == pytest.approx(0.469, abs=0.001)
    assert p10(correct) == pytest.approx(0.506, abs=0.001)
    assert median(correct) == pytest.approx(0.602, abs=0.001)
    assert correct[-1] == pytest.approx(0.772, abs=0.001)
    assert wrong[0] == pytest.approx(0.420, abs=0.001)
    assert p10(wrong) == pytest.approx(0.436, abs=0.001)
    assert median(wrong) == pytest.approx(0.511, abs=0.001)
    assert wrong[-1] == pytest.approx(0.717, abs=0.001)
    # Wrong reads start below and end below, but the ranges overlap almost entirely.
    assert wrong[0] < correct[0] < wrong[-1] < correct[-1]


@pytest.mark.parametrize("floor,correct,kept", PRECISION_LADDER)
def test_precision_climbs_monotonically_with_the_threshold(
    corpus: Corpus, floor: float, correct: int, kept: int
) -> None:
    rows = [
        (row, read)
        for row, read in corpus.legible_reads()
        if read is not None and read.confidence >= floor
    ]
    assert len(rows) == kept
    assert sum(1 for row, read in rows if read.text == row.plate) == correct


def test_the_ladder_is_monotonic() -> None:
    precisions = [correct / kept for _, correct, kept in PRECISION_LADDER]
    assert precisions == sorted(precisions)


def test_above_0_75_there_are_six_reads_and_all_are_correct(corpus: Corpus) -> None:
    """Too few to claim a clean region, which is why the docstring says so.

    Under the six-variant configuration the two distributions' maxima did cross -- 0.862
    correct against 0.895 wrong -- and that crossing was not a property of the matcher:
    taking the best of six means a variant that scores a wrong string unusually high
    gets selected. Max-of-N inflates the tail of the wrong distribution specifically.
    """
    rows = [
        (row, read)
        for row, read in corpus.legible_reads()
        if read is not None and read.confidence >= 0.75
    ]
    assert len(rows) == 6
    assert all(read.text == row.plate for row, read in rows)


def test_the_corpus_has_seven_plate_strings_across_twelve_vehicles(
    corpus: Corpus,
) -> None:
    by_plate: dict[str, set[int]] = {}
    for row in corpus.rows:
        if row.plate:
            by_plate.setdefault(row.plate, set()).add(row.vehicle_id)
    assert len(by_plate) == 7
    counts = sorted(len(ids) for ids in by_plate.values())
    assert counts == [1, 1, 2, 2, 2, 2, 2]


def test_the_stage_counters_add_up(corpus: Corpus) -> None:
    """Every crop is read, refused for width, or refused for score. Nothing vanishes.

    ink_not_found and grid_rejected are both zero on this corpus, which is worth pinning
    rather than ignoring: the ink mask is derived from the known plate box by arithmetic
    rather than searched for, so failing to find ink would mean the box itself was
    wrong.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    for row in corpus.rows:
        engine.read_crop(row.crop, row.candidate)
    stats = engine.stats()
    engine.close()

    assert stats["plates_seen"] == EXPECTED_CROPS
    assert stats["refused_small"] == 6
    assert stats["score_too_low"] == 40
    assert stats["ink_not_found"] == 0
    assert stats["grid_rejected"] == 0
    assert stats["crops_empty"] == 0
    assert stats["reads_returned"] == 523
    assert (
        stats["refused_small"] + stats["score_too_low"] + stats["reads_returned"]
        == EXPECTED_CROPS
    )
    assert stats["variant_wins"] == {"upscale_2x": 523}
    assert stats["read_proxy"] == pytest.approx(523 / 569, abs=0.0001)


# ------------------------------------------------------------------------- fabrication


def test_nine_of_the_nineteen_occluded_plates_still_returned_a_string(
    corpus: Corpus,
) -> None:
    """0 correct, confidences 0.442 to 0.568 -- entirely inside the unseparated band.

    The honest position, and it is in the docstring: this stage does not prevent
    invention. It is caught downstream by grammar and by cross-frame agreement, and both
    of those are documented as weak.
    """
    rows = [
        (row, read)
        for row, read in zip(corpus.rows, corpus.reads())
        if row.legible is False
    ]
    assert len(rows) == EXPECTED_ILLEGIBLE
    got = [(row, read) for row, read in rows if read is not None]
    assert len(got) == 9
    assert all(read.text != row.plate for row, read in got)
    confidences = [read.confidence for _, read in got]
    assert min(confidences) == pytest.approx(0.442, abs=0.001)
    assert max(confidences) == pytest.approx(0.568, abs=0.001)
    assert all(c < MIN_TEMPLATE_SCORE + 0.15 for c in confidences)


def test_four_reads_are_inventions_at_zero_visible_fraction(corpus: Corpus) -> None:
    """Nothing of the plate was drawn and a string came back anyway.

    4 of the 569 crops this run offers survive the score floor with an invented plate.
    That is the failure Contracts section 12 names as the worst this pipeline can
    produce, stated as a rate rather than asserted away.
    """
    zero = [row for row in corpus.rows if row.legible is False and row.visible == 0.0]
    assert len(zero) == 10
    invented = [
        (row, read)
        for row, read in zip(corpus.rows, corpus.reads())
        if row.legible is False and row.visible == 0.0 and read is not None
    ]
    assert len(invented) == 4


@pytest.mark.parametrize(
    "visible,width,text,confidence",
    [
        (0.83, 59, "J3C4567", 0.552),
        (0.74, 69, "GJ3C45", 0.550),
        (0.00, 65, "RWJT5T9T", 0.442),
        (0.00, 65, "ETJT74", 0.471),
    ],
)
def test_the_named_truncations_and_inventions_reproduce(
    corpus: Corpus, visible: float, width: int, text: str, confidence: float
) -> None:
    """Two distinct failures, and the difference matters to what recovers them.

    A truncation reads the visible characters correctly and stops, so cross-frame
    agreement recovers it -- a substring agrees with the full string on every character
    it has. An invention agrees with nothing, so it never accumulates weight.
    """
    matches = [
        (row, read)
        for row, read in zip(corpus.rows, corpus.reads())
        if row.legible is False and read is not None and read.text == text
    ]
    assert len(matches) == 1
    row, read = matches[0]
    assert row.width == width
    assert row.visible == pytest.approx(visible, abs=0.005)
    assert read.confidence == pytest.approx(confidence, abs=0.001)
    assert row.plate == "GJ3C4567"


def test_truncations_are_substrings_of_truth_and_inventions_are_not() -> None:
    """The property the recovery argument rests on, spelled out."""
    assert "J3C4567" in "GJ3C4567"
    assert "GJ3C45" in "GJ3C4567"
    assert "RWJT5T9T" not in "GJ3C4567"
    assert "ETJT74" not in "GJ3C4567"


def test_the_occluded_plates_that_were_refused_outnumber_the_ones_that_read(
    corpus: Corpus,
) -> None:
    """10 of 19 refused. The score floor is a filter, not a guarantee."""
    rows = [
        read
        for row, read in zip(corpus.rows, corpus.reads())
        if row.legible is False
    ]
    assert sum(1 for read in rows if read is None) == 10


# ------------------------------------------------------------------- template internals


def test_the_template_constants_are_the_documented_ones() -> None:
    assert TEMPLATE_INK_LEVEL == 0.45
    assert TEMPLATE_MIN_CHARS == 4
    assert TEMPLATE_MAX_CHARS == 12
    assert MIN_TEMPLATE_SCORE == 0.42
    assert MIN_PX_PER_CHAR == 2
    assert (CELL_W, CELL_H) == (5, 7)


@pytest.mark.parametrize(
    "width,low,high,survivors",
    [
        (24, 0.2159, 0.2436, 0),
        (26, 0.4189, 0.5220, 5),
        (28, 0.4087, 0.5048, 6),
    ],
)
def test_the_standalone_sweep_fabricates_below_the_legibility_floor(
    width: int, low: float, high: float, survivors: int
) -> None:
    """Seven plate strings, seven fabrications, none correct, at every width from 24 up.

    This is what set MIN_TEMPLATE_SCORE, and the measurement has to be standalone
    because the corpus offers only two crops in the 24-29 px band. The floor removes the
    24 px band outright and does *not* cover 26-29 px, where five and then six of the
    seven survive it and land squarely inside the range correct reads occupy.

    An earlier version of this claim said all five test plates returned "ZZZZ" at 0.604.
    That measurement was real but it was of the *deleted* ring-stripping ink mask
    recorded in _ink_mask's docstring: the shipped matcher returns seven different
    strings, none reaching 0.25 at 24 px. The conclusion survived the correction -- the
    matcher fabricates confidently on plates with no legible characters, and that is not
    fixable by tuning it, because the matcher is doing its job.
    """
    loose = build_ocr_engine({"name": "template", "min_score": 0.0})
    loose.load()
    reads = [_read_standalone(loose, plate, width) for plate in CORPUS_PLATES]
    loose.close()

    assert all(read is not None for read in reads)
    assert all(read.text != plate for plate, read in zip(CORPUS_PLATES, reads))
    confidences = [read.confidence for read in reads]
    assert min(confidences) == pytest.approx(low, abs=0.001)
    assert max(confidences) == pytest.approx(high, abs=0.001)
    assert sum(1 for c in confidences if c >= MIN_TEMPLATE_SCORE) == survivors

    tight = build_ocr_engine({"name": "template"})
    tight.load()
    kept = [_read_standalone(tight, plate, width) for plate in CORPUS_PLATES]
    tight.close()
    assert sum(1 for read in kept if read is not None) == survivors


def test_the_standalone_reads_are_background_independent() -> None:
    """Byte-identical on a dark, mid-grey and bright surround.

    Ink is measured against the plate face's own range rather than the crop's, so the
    bodywork in the pad cannot set the threshold. The previous implementation located
    the brightest connected region instead and failed on light-coloured vehicles -- buses
    read 0 of 13 exact against cars' 7 of 15 -- because it assumed the plate is the
    brightest thing in the crop.
    """
    loose = build_ocr_engine({"name": "template", "min_score": 0.0})
    loose.load()
    per_background = {}
    for background in (40, 128, 210):
        per_background[background] = [
            (read.text, read.confidence)
            for read in (_read_standalone(loose, p, 24, background) for p in CORPUS_PLATES)
        ]
    loose.close()
    assert per_background[40] == per_background[128] == per_background[210]


def test_min_score_zero_lets_the_fabrications_straight_through() -> None:
    """The floor is the only thing standing between the matcher and a 24 px invention."""
    tight = build_ocr_engine({"name": "template"})
    tight.load()
    loose = build_ocr_engine({"name": "template", "min_score": 0.0})
    loose.load()
    assert _read_standalone(tight, "GJ01AB1234", 24) is None
    assert _read_standalone(loose, "GJ01AB1234", 24) is not None
    tight.close()
    loose.close()


def test_a_flat_crop_is_refused_for_want_of_ink() -> None:
    """No contrast inside the plate is an honest "nothing here", not a guess."""
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame = np.full((120, 300, 3), 140, np.uint8)
    assert engine.read(frame, _candidate((100, 50, 180, 68))) is None
    assert engine.stats()["ink_not_found"] == 1
    engine.close()


def test_an_inner_box_too_narrow_for_min_chars_is_refused() -> None:
    """Checked before the contrast test, so a high-contrast crop still fails on size."""
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    crop = np.zeros((6, 6, 3), np.uint8)
    crop[:, 2] = 255
    assert engine._ink_mask(crop, _candidate((0, 0, 4, 6))) is None
    engine.close()


def test_fit_grid_refuses_below_two_source_pixels_per_character() -> None:
    """And the obvious stricter test was measured wrong.

    One pixel per glyph *column* would reject the true 10-character hypothesis at 60 px
    plate width, where the ink box is 56 px against 59 cells -- so the read came back one
    character short with everything after the second character shifted. The 60 px bucket
    is where a real junction plate lives.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    ink = np.zeros((7, 7), bool)
    ink[3, :] = True
    assert engine._fit_grid(ink, 4, 0, 0) is None
    wider = np.zeros((7, 8), bool)
    wider[3, :] = True
    assert engine._fit_grid(wider, 4, 0, 0) is not None
    engine.close()


def test_best_glyph_scores_an_empty_cell_zero() -> None:
    """The whole reason the matcher uses Jaccard rather than mean absolute difference.

    Under MAD an all-background cell scores 1 - 10/35 = 0.714 against the sparsest glyph,
    and 0.714 sat inside the range real reads were producing -- so an empty cell was
    competitive with a correct character and a wrong character count could win outright.
    Directly observed: MH12DE1433 read as YLJLE1433 at 200 px, a 9-character fit to a
    10-character plate with every substitution a sparse glyph.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    _, score = engine._best_glyph(np.zeros((CELL_H, CELL_W), np.float32))
    assert score == 0.0
    engine.close()


def test_best_glyph_returns_the_matching_glyph_at_full_score() -> None:
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    for char in ("A", "7", "W", "3"):
        found, score = engine._best_glyph(GLYPHS[char].astype(np.float32))
        assert score == pytest.approx(1.0)
        assert np.array_equal(GLYPHS[found], GLYPHS[char])
    engine.close()


def test_an_all_ink_cell_scores_below_any_real_match() -> None:
    """20/35 against the densest glyph, which is the right answer rather than a residual.

    A saturated cell genuinely is more consistent with a dense glyph than a sparse one,
    and 0.571 is still below any real match. Normalised correlation would also fix the
    empty-cell case and introduces a different error -- it is invariant to ink density,
    so a cell 80% covered correlates perfectly with a glyph 30% covered and 8 reads as B.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    _, score = engine._best_glyph(np.ones((CELL_H, CELL_W), np.float32))
    densest = max(int(mask.sum()) for char, mask in GLYPHS.items() if char != " ")
    assert score == pytest.approx(densest / (CELL_W * CELL_H))
    assert score < 0.6
    engine.close()


def test_the_glyph_stack_drops_space() -> None:
    """Leaving it in would let the matcher explain any low-ink cell as a space.

    Every fit score would be inflated, and a plate's spaces are separators anyway --
    Contracts section 4.2 normalizes them out.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    assert " " not in engine._chars
    assert len(engine._chars) == 36
    assert engine._stack.shape == (36, CELL_H, CELL_W)
    engine.close()


def test_resample_cell_area_averages_rather_than_sampling() -> None:
    """A 1 px stroke in a 12 px cell must not depend on where the sample point lands.

    Nearest sampling makes the same character read differently depending on sub-pixel
    alignment, so the read becomes a function of where the plate happens to sit in the
    frame. Averaging turns a partly-covered cell into an intermediate value, which is
    both stable and the correct representation of the uncertainty.
    """
    cell = np.zeros((CELL_H, 10), bool)
    cell[:, 0] = True
    out = _resample_cell(cell, CELL_W, CELL_H)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(0.0)


def test_resample_cell_survives_an_empty_cell() -> None:
    out = _resample_cell(np.zeros((0, 0), bool), CELL_W, CELL_H)
    assert out.shape == (CELL_H, CELL_W)
    assert not out.any()


def test_implied_char_count_is_exact_on_the_glyph_extent() -> None:
    """(7a + 1) / 6 against text_extent is exact: 59x7, aspect 8.429, implied 10.00."""
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    mask = text_mask("GJ01AB1234")
    width, height = text_extent("GJ01AB1234")
    assert mask.shape == (height, width) == (7, 59)
    assert engine._implied_char_count(mask) == pytest.approx(10.0, abs=0.01)
    engine.close()


def test_implied_char_count_does_not_survive_rendering(corpus: Corpus) -> None:
    """Which is why it is diagnostic only, and the negative result is worth keeping.

    The generator stretches the 59x7 mask to fill the plate's whole inner rectangle
    rather than preserving the glyph aspect, so a rendered 10-character plate implies
    five to seven characters depending on width. Used as a search window it forced
    6-character fits onto 10-character plates and took the exact-read rate to zero.

    The deeper point survives the fixture: a plate is a fixed physical size and a
    6-character registration is set in wider characters rather than on a shorter plate,
    so any length prior taken from the box is measuring the plate and not the string.
    """
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    frame, box = _render_plate("GJ01AB1234", 100)
    candidate = _candidate(box)
    ink = engine._ink_mask(engine.cut_crop(frame, candidate), candidate)
    assert ink is not None
    implied = engine._implied_char_count(ink)
    assert implied is not None
    assert implied < 8.0
    engine.close()


def test_reading_before_load_raises_rather_than_returning_none() -> None:
    """A missing load is a programming error, not an unreadable plate."""
    engine = TemplateOCR()
    with pytest.raises(RuntimeError, match="load"):
        engine._read_crop(np.full((20, 66, 3), 128, np.uint8), _candidate((0, 0, 60, 15)))


def test_the_template_backend_never_ships_and_the_reason_is_not_the_licence() -> None:
    """Apache-2.0 and genuinely classical OCR. What disqualifies it is the corpus.

    The templates are the same masks the fixtures are rendered with, so any accuracy
    figure from it is circular -- it would be the best number in the project and would
    mean nothing.
    """
    engine = build_ocr_engine({"name": "template"})
    assert engine.license_name == "Apache-2.0"
    assert engine.ships is False
    assert engine.model_name == "glyph-template"


# ------------------------------------------------------------------------------- oracle


def _oracle_rig(**kwargs):
    """An oracle over one hand-built frame: GJ01AB1234 at 90 px, legible."""
    box = (100, 200, 190, 222)
    truth = FrameTruth(frame_index=7, pts_ms=840, vehicles=[_vehicle("GJ01AB1234", box)])
    source = _TruthSource({840: truth})
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"], **kwargs}, source=source
    )
    engine.load()
    return engine, _candidate(box), _Envelope(7, 840)


def test_the_oracle_returns_truth_at_the_default_error_rate() -> None:
    engine, candidate, envelope = _oracle_rig()
    read = engine.read_crop(np.full((28, 96, 3), 128, np.uint8), candidate, frame_ref=envelope)
    assert read is not None
    assert read.text == "GJ01AB1234"
    assert read.confidence == pytest.approx(0.95)
    engine.close()


def test_the_oracle_ignores_the_pixels_entirely() -> None:
    """Which is the point of it: substituting this answers "how much of the end-to-end
    rate is OCR" in one run, and if the rate barely moves a day on the OCR model would
    have been a day wasted."""
    engine, candidate, envelope = _oracle_rig()
    noise = np.random.default_rng(1).integers(0, 256, (28, 96, 3), dtype=np.uint8)
    flat = np.zeros((28, 96, 3), np.uint8)
    first = engine.read_crop(noise, candidate, frame_ref=envelope)
    second = engine.read_crop(flat, candidate, frame_ref=envelope)
    assert first.text == second.text == "GJ01AB1234"
    engine.close()


def test_a_deferred_oracle_read_without_a_frame_ref_is_a_silent_zero() -> None:
    """Documented as a silent failure rather than a loud one, so it is pinned here.

    Read at flush time with no reference, the oracle answers against whichever frame
    happened to be last, finds no vehicle whose box matches, and returns None. Every
    plate comes back unread, every event carries plate: null, and nothing raises -- so
    the offline run that exists to verify the pipeline reports a clean, plausible zero.
    """
    engine, candidate, _ = _oracle_rig()
    assert engine.read_crop(np.full((28, 96, 3), 128, np.uint8), candidate) is None
    assert engine.plates_seen == 1
    engine.close()


def test_a_frame_ref_resolves_truth_because_only_pts_is_needed() -> None:
    """FrameRef is enough; the envelope's pixels are not part of the lookup."""
    engine, candidate, _ = _oracle_rig()
    ref = FrameRef(camera_id=CAMERA, stream_session_id="s", frame_index=7, pts_ms=840)
    read = engine.read_crop(np.zeros((28, 96, 3), np.uint8), candidate, frame_ref=ref)
    assert read is not None and read.text == "GJ01AB1234"
    engine.close()


def test_an_unresolvable_frame_is_counted_rather_than_guessed() -> None:
    """A faulted run's PTS does not resolve, and pretending otherwise would be a guess."""
    engine, candidate, _ = _oracle_rig()
    engine.read_crop(
        np.zeros((28, 96, 3), np.uint8), candidate, frame_ref=_Envelope(9, 99999)
    )
    assert engine.stats()["unresolved_frames"] == 1
    engine.close()


def test_the_oracle_refuses_an_illegible_plate_and_counts_it() -> None:
    """An oracle that read those would measure a capability no real backend can have."""
    box = (100, 200, 190, 222)
    truth = FrameTruth(
        frame_index=1,
        pts_ms=120,
        vehicles=[_vehicle("GJ01AB1234", box, legible=False, visible=0.0)],
    )
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"]}, source=_TruthSource({120: truth})
    )
    engine.load()
    read = engine.read_crop(
        np.zeros((28, 96, 3), np.uint8), _candidate(box), frame_ref=_Envelope(1, 120)
    )
    assert read is None
    assert engine.stats()["illegible_skipped"] == 1
    engine.close()


def test_require_legible_false_reads_the_illegible_plate_anyway() -> None:
    """The switch that makes the fabrication measurement possible at all."""
    box = (100, 200, 190, 222)
    truth = FrameTruth(
        frame_index=1,
        pts_ms=120,
        vehicles=[_vehicle("GJ01AB1234", box, legible=False)],
    )
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"], "require_legible": False},
        source=_TruthSource({120: truth}),
    )
    engine.load()
    read = engine.read_crop(
        np.zeros((28, 96, 3), np.uint8), _candidate(box), frame_ref=_Envelope(1, 120)
    )
    assert read is not None and read.text == "GJ01AB1234"
    assert engine.stats()["illegible_skipped"] == 0
    engine.close()


def test_a_box_matching_no_truth_plate_counts_no_truth_match() -> None:
    engine, _, envelope = _oracle_rig()
    elsewhere = _candidate((600, 600, 690, 622))
    assert engine.read_crop(np.zeros((28, 96, 3), np.uint8), elsewhere, frame_ref=envelope) is None
    assert engine.stats()["no_truth_match"] == 1
    engine.close()


def test_the_oracle_plate_match_threshold_is_looser_than_the_vehicle_one() -> None:
    """0.20 against the plate stage's 0.30, and the reason is geometry.

    A plate box is small and a few pixels of error is a large IoU change: a 50x12 box
    shifted 3 px horizontally has IoU 0.79 with itself. The oracle should not start
    refusing to read plates because the detector was 3 px out.
    """
    assert ORACLE_PLATE_MATCH_IOU == 0.20
    assert ORACLE_PLATE_MATCH_IOU < ORACLE_MATCH_IOU

    box = (100, 200, 150, 212)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"], "min_plate_width_px": 10},
        source=_TruthSource({120: truth}),
    )
    engine.load()
    shifted = _candidate((103, 200, 153, 212))
    assert _iou(shifted.plate_bbox_xyxy, box) > 0.7
    read = engine.read_crop(
        np.zeros((18, 56, 3), np.uint8), shifted, frame_ref=_Envelope(1, 120)
    )
    assert read is not None
    engine.close()


@pytest.mark.parametrize(
    "width,expected",
    [(90, 0.95), (120, 0.95), (24, 0.45), (57, 0.45 + 0.5 * 0.5)],
)
def test_oracle_confidence_falls_linearly_with_width(width: int, expected: float) -> None:
    """Linear rather than anything shaped.

    The point is only that confidence and width move together; a fitted curve here would
    invent a calibration the oracle has no basis for, and ai/quality is where the real
    relationship between width and reliability belongs.
    """
    box = (100, 200, 100 + width, 224)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"]}, source=_TruthSource({120: truth})
    )
    engine.load()
    read = engine.read_crop(
        np.zeros((30, width + 6, 3), np.uint8), _candidate(box), frame_ref=_Envelope(1, 120)
    )
    assert read.confidence == pytest.approx(expected, abs=0.001)
    engine.close()


def test_oracle_corruption_is_deterministic_and_keyed_on_plate_position_and_frame() -> None:
    """Reproducible, different between frames of one track, order-independent.

    Different between frames is what gives fusion something to do; order independence is
    what stops the corruption from depending on which vehicle happens to be processed
    first.
    """
    box = (100, 200, 190, 222)
    frames = {
        120: FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)]),
        240: FrameTruth(frame_index=2, pts_ms=240, vehicles=[_vehicle("GJ01AB1234", box)]),
    }
    reads = {}
    for _ in range(2):
        engine = build_ocr_engine(
            {"name": "oracle", "variants": ["raw"], "char_error_rate": 0.5},
            source=_TruthSource(frames),
        )
        engine.load()
        for pts, index in ((120, 1), (240, 2)):
            read = engine.read_crop(
                np.zeros((28, 96, 3), np.uint8),
                _candidate(box),
                frame_ref=_Envelope(index, pts),
            )
            reads.setdefault(pts, []).append(read.text)
        engine.close()

    assert reads[120][0] == reads[120][1]
    assert reads[240][0] == reads[240][1]
    assert reads[120][0] != reads[240][0]
    assert _unit_hash("GJ01AB1234", 0, 1) == _unit_hash("GJ01AB1234", 0, 1)
    assert _unit_hash("GJ01AB1234", 0, 1) != _unit_hash("GJ01AB1234", 0, 2)


def test_oracle_corruption_only_substitutes_confusable_characters() -> None:
    """A random character error is trivially outvoted because the wrong answers differ.

    A systematic confusion produces the SAME wrong character on every frame and can win
    a majority vote outright, so fusion has to be tested against the failure that can
    actually beat it.
    """
    box = (100, 200, 190, 222)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"], "char_error_rate": 5.0},
        source=_TruthSource({120: truth}),
    )
    engine.load()
    read = engine.read_crop(
        np.zeros((28, 96, 3), np.uint8), _candidate(box), frame_ref=_Envelope(1, 120)
    )
    engine.close()
    assert read.text != "GJ01AB1234"
    for got, want in zip(read.text, "GJ01AB1234"):
        assert got == want or got == CONFUSIONS[want]


def test_oracle_corruption_gets_worse_as_the_plate_shrinks() -> None:
    """The rate scales as (full_width / width): a 45 px plate at twice the configured rate.

    That inverse relationship is the one real property being modelled -- OCR error is
    driven by pixels per character, and pixels per character is linear in plate width.

    Measured over 140 characters rather than 10. A single 10-character plate on a single
    frame corrupts 2 characters at either width, because the per-character decision is a
    fixed hash rather than a draw: the rate is a property of the ensemble and asserting
    it on one plate is asserting on one sample of it.
    """
    corrupted = {}
    for width in (45, 90):
        box = (100, 200, 100 + width, 224)
        frames = {}
        for index, plate in enumerate(CORPUS_PLATES):
            for repeat in range(2):
                pts = 120 * (index * 2 + repeat + 1)
                frames[pts] = FrameTruth(
                    frame_index=index * 2 + repeat,
                    pts_ms=pts,
                    vehicles=[_vehicle(plate, box)],
                )
        engine = build_ocr_engine(
            {"name": "oracle", "variants": ["raw"], "char_error_rate": 0.3},
            source=_TruthSource(frames),
        )
        engine.load()
        for pts, truth in frames.items():
            engine.read_crop(
                np.zeros((30, width + 6, 3), np.uint8),
                _candidate(box),
                frame_ref=_Envelope(truth.frame_index, pts),
            )
        corrupted[width] = engine.stats()["chars_corrupted"]
        engine.close()
    assert corrupted[45] > corrupted[90]


def test_the_oracle_agrees_with_itself_across_every_variant() -> None:
    """Correct rather than a quirk: with no pixel dependence there is no disagreement.

    A downstream stage weighting by agreement should see maximum agreement here, which
    is also why an oracle cannot substitute for the template matcher when the thing under
    test is fusion -- five identical votes are one vote counted five times.
    """
    box = (100, 200, 190, 222)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    engine = build_ocr_engine(
        {"name": "oracle", "variants": list(DEFAULT_VARIANTS)},
        source=_TruthSource({120: truth}),
    )
    engine.load()
    read = engine.read_crop(
        np.zeros((28, 96, 3), np.uint8), _candidate(box), frame_ref=_Envelope(1, 120)
    )
    engine.close()
    assert (read.variants_tried, read.variants_agreeing) == (6, 6)
    assert read.agreement == 1.0


def test_the_oracle_reads_the_real_corpus_perfectly(corpus: Corpus) -> None:
    """Every legible crop, right, from a FrameRef alone.

    This is the run that isolates everything downstream from OCR quality, and the reason
    it has to work on the deferred path: the pipeline reads a track's top-K crops when
    the track finishes, by which time the frames are gone.
    """
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"], "min_plate_width_px": 10},
        source=corpus.source,
    )
    engine.load()
    wrong = []
    for row in corpus.legible:
        read = engine.read_crop(row.crop, row.candidate, frame_ref=row.frame_ref)
        if read is None or read.text != row.plate:
            wrong.append((row.width, None if read is None else read.text))
    stats = engine.stats()
    engine.close()
    assert wrong == []
    assert stats["no_truth_match"] == 0
    assert stats["unresolved_frames"] == 0
    assert stats["ships"] is False


def test_the_oracle_never_ships() -> None:
    engine, _, _ = _oracle_rig()
    assert engine.ships is False
    assert engine.model_name == "oracle-ocr"
    assert engine.model_version == "truth"
    assert engine.license_name == "not-applicable"
    engine.close()


# ----------------------------------------------------------------------------- scripted


def test_the_script_describes_plates_not_variants() -> None:
    """Two variants, two plates, and the script still has one entry per plate.

    _read_crop runs once per variant, so the cursor is divided by the variant count. Get
    that wrong and a two-variant run consumes the script twice as fast, silently.
    """
    engine = build_ocr_engine(
        {
            "name": "scripted",
            "variants": ["raw", "grayscale"],
            "script": {3: [("GJ01AB1234", 0.9), ("MH12DE9812", 0.8)]},
        }
    )
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    reads = engine.read_all_envelope(
        _Envelope(3, 360, frame),
        {
            1: _candidate((50, 50, 140, 72)),
            2: _candidate((200, 100, 290, 122)),
        },
    )
    engine.close()
    assert reads[1].text == "GJ01AB1234"
    assert reads[2].text == "MH12DE9812"
    assert reads[1].variants_tried == 2


def test_a_frame_with_no_script_entry_produces_no_read() -> None:
    """How the "plate found, not readable" path -- the one that must emit plate: null --
    gets exercised deterministically."""
    engine = build_ocr_engine({"name": "scripted", "script": {3: ["GJ01AB1234"]}})
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    reads = engine.read_all_envelope(_Envelope(4, 480, frame), {1: _candidate((50, 50, 140, 72))})
    engine.close()
    assert reads == {}


def test_running_past_the_end_of_a_frames_entries_returns_none() -> None:
    engine = build_ocr_engine({"name": "scripted", "script": {0: ["GJ01AB1234"]}})
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    reads = engine.read_all_envelope(
        _Envelope(0, 0, frame),
        {1: _candidate((50, 50, 140, 72)), 2: _candidate((200, 100, 290, 122))},
    )
    engine.close()
    assert set(reads) == {1}


def test_the_script_cursor_resets_at_each_frame() -> None:
    engine = build_ocr_engine({"name": "scripted", "script": {0: ["A1"], 1: ["B2"]}})
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    candidates = {1: _candidate((50, 50, 140, 72))}
    first = engine.read_all_envelope(_Envelope(0, 0, frame), candidates)
    second = engine.read_all_envelope(_Envelope(1, 120, frame), candidates)
    again = engine.read_all_envelope(_Envelope(0, 0, frame), candidates)
    engine.close()
    assert first[1].text == "A1"
    assert second[1].text == "B2"
    assert again[1].text == "A1"


def test_a_bare_string_in_the_script_is_a_full_confidence_read() -> None:
    """Convenient in YAML, where the common case is "frame 7 reads GJ01AB1234"."""
    engine = build_ocr_engine({"name": "scripted", "script": {0: ["GJ01AB1234"]}})
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    reads = engine.read_all_envelope(_Envelope(0, 0, frame), {1: _candidate((50, 50, 140, 72))})
    engine.close()
    assert reads[1].confidence == 1.0


def test_the_scripted_backend_never_ships() -> None:
    engine = build_ocr_engine({"name": "scripted", "script": {}})
    assert engine.ships is False
    assert engine.model_name == "scripted-ocr"


# ------------------------------------------------------------------------------ factory


def test_the_registry_names_are_the_four_documented_ones() -> None:
    assert OCR_ENGINE_NAMES == ("paddle", "template", "oracle", "scripted")


def test_only_paddle_may_appear_in_a_published_claim() -> None:
    """A literal set rather than a derivation from the ships property.

    Adding a backend forces a decision here instead of inheriting one, and the two
    reasons for exclusion are different: oracle never ships because it reads the answer
    key, template because the fonts it matches are the fonts the fixtures are drawn with.
    Only one of them is excluded for cheating.
    """
    assert SHIPPABLE_OCR_ENGINES == frozenset({"paddle"})
    assert ocr_engine_ships("paddle") is True
    for name in ("template", "oracle", "scripted"):
        assert ocr_engine_ships(name) is False


def test_every_registered_name_builds() -> None:
    box = (100, 200, 190, 222)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    configs = {
        "paddle": ({"name": "paddle"}, None),
        "template": ({"name": "template"}, None),
        "oracle": ({"name": "oracle"}, _TruthSource({120: truth})),
        "scripted": ({"name": "scripted", "script": {}}, None),
    }
    for name, (config, source) in configs.items():
        engine = build_ocr_engine(config, source=source)
        assert isinstance(engine, BaseOCR)
        assert isinstance(engine, OCREngine)


def test_building_paddle_needs_no_paddleocr_installed() -> None:
    """The lazy import is what lets this environment run all 14 stages.

    ai/ocr/factory.py imports ai.ocr.paddle inside _build_paddle, and ai/ocr/paddle.py
    imports paddleocr inside _load, so the ImportError only fires if the branch is
    actually taken. It also matters for cost: importing paddlepaddle loads a CUDA
    context and costs several seconds.
    """
    engine = build_ocr_engine({"name": "paddle"})
    assert engine.ships is True
    assert engine.license_name == "Apache-2.0"


def test_a_missing_name_is_refused() -> None:
    with pytest.raises(OCRConfigError, match="no 'name'"):
        build_ocr_engine({})


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(OCRConfigError, match="unknown ocr engine"):
        build_ocr_engine({"name": "tesseract"})


def test_a_non_mapping_config_is_refused() -> None:
    with pytest.raises(OCRConfigError, match="must be a mapping"):
        build_ocr_engine(["template"])  # type: ignore[arg-type]


def test_an_unknown_key_is_refused_rather_than_defaulted() -> None:
    """A config with `drop_scor:` runs at the default and nothing in the output says so.

    Which turns a typo into a benchmark row that is quietly about a different
    configuration than the one it claims.
    """
    with pytest.raises(OCRConfigError, match="drop_scor"):
        build_ocr_engine({"name": "paddle", "drop_scor": 0.2})


def test_a_key_belonging_to_another_backend_is_refused() -> None:
    with pytest.raises(OCRConfigError, match="min_score"):
        build_ocr_engine({"name": "paddle", "min_score": 0.5})


def test_a_bare_string_variants_value_is_refused() -> None:
    """"raw" is a Sequence of characters, and every one of them is an unknown variant."""
    with pytest.raises(OCRConfigError, match="must be a list"):
        build_ocr_engine({"name": "template", "variants": "raw"})


def test_an_empty_variants_list_is_refused() -> None:
    with pytest.raises(OCRConfigError, match="reads nothing"):
        build_ocr_engine({"name": "template", "variants": []})


def test_an_unknown_variant_is_refused_at_build_time() -> None:
    """Not at first use. A misspelled variant raises KeyError inside the per-plate loop,
    so unchecked it surfaces a thousand frames into a benchmark rather than in the first
    second."""
    with pytest.raises(OCRConfigError, match="unknown preprocessing variant"):
        build_ocr_engine({"name": "template", "variants": ["upscale_3x"]})


def test_the_template_backend_refuses_min_chars_above_max_chars() -> None:
    """It admits no length hypothesis at all and would refuse every plate."""
    with pytest.raises(OCRConfigError, match="min_chars"):
        build_ocr_engine({"name": "template", "min_chars": 9, "max_chars": 4})


def test_the_oracle_refuses_to_build_without_a_source() -> None:
    with pytest.raises(OCRConfigError, match="synthetic"):
        build_ocr_engine({"name": "oracle"})


def test_the_oracle_refuses_a_source_that_carries_no_truth() -> None:
    with pytest.raises(OCRConfigError, match="truth_for_envelope"):
        build_ocr_engine({"name": "oracle"}, source=object())


def test_the_scripted_backend_accepts_three_row_shapes() -> None:
    engine = build_ocr_engine(
        {
            "name": "scripted",
            "script": {
                0: ["GJ01AB1234"],
                1: [("MH12DE9812", 0.4)],
                2: [{"text": "GJ05JK4521", "confidence": 0.6}],
            },
        }
    )
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    candidates = {1: _candidate((50, 50, 140, 72))}
    got = [
        engine.read_all_envelope(_Envelope(index, index * 120, frame), candidates)[1]
        for index in (0, 1, 2)
    ]
    engine.close()
    assert [(r.text, r.confidence) for r in got] == [
        ("GJ01AB1234", 1.0),
        ("MH12DE9812", 0.4),
        ("GJ05JK4521", 0.6),
    ]


def test_the_scripted_backend_refuses_a_non_mapping_script() -> None:
    with pytest.raises(OCRConfigError, match="must be a mapping"):
        build_ocr_engine({"name": "scripted", "script": ["GJ01AB1234"]})


def test_the_scripted_backend_refuses_an_entry_without_text() -> None:
    with pytest.raises(OCRConfigError, match="missing text"):
        build_ocr_engine({"name": "scripted", "script": {0: [{"confidence": 0.5}]}})


def test_the_scripted_backend_refuses_an_unusable_row() -> None:
    with pytest.raises(OCRConfigError, match="pairs"):
        build_ocr_engine({"name": "scripted", "script": {0: [3.5]}})


def test_default_variants_for_narrows_only_the_template_backend() -> None:
    """Kept here as well as in the backend's __init__ so describe can report true cost.

    Letting describe fall back to DEFAULT_VARIANTS made it report six reads per plate
    for a config that performs one: a validator's cost estimate off by 6x, in the number
    the stage is budgeted on.
    """
    assert default_variants_for("template") == TEMPLATE_DEFAULT_VARIANTS
    for name in ("paddle", "oracle", "scripted"):
        assert default_variants_for(name) == DEFAULT_VARIANTS


def test_describe_reports_the_true_cost_without_constructing() -> None:
    """Config validation must not import paddle or download recogniser weights."""
    template = describe_ocr_engine({"name": "template"})
    assert template["reads_per_plate"] == 1
    assert template["variants"] == list(TEMPLATE_DEFAULT_VARIANTS)
    assert template["ships"] is False
    assert template["min_plate_width_px"] == MIN_OCR_PLATE_WIDTH_PX
    assert template["pad_px"] == PLATE_CROP_PAD_PX

    paddle = describe_ocr_engine({"name": "paddle"})
    assert paddle["reads_per_plate"] == 6
    assert paddle["ships"] is True


def test_describe_honours_an_explicit_variant_list() -> None:
    described = describe_ocr_engine({"name": "template", "variants": ["raw", "sharpen"]})
    assert described["reads_per_plate"] == 2


def test_normalize_refuses_a_non_shipping_backend_for_publication() -> None:
    """The primary metric is a plate-string rate, so an unshippable OCR backend does not
    taint one diagnostic -- it invalidates the headline number."""
    for name in ("template", "oracle", "scripted"):
        with pytest.raises(OCRConfigError, match="published benchmark"):
            normalize_ocr_config({"name": name}, for_publication=True)


def test_normalize_accepts_paddle_for_publication() -> None:
    assert normalize_ocr_config({"name": "paddle"}, for_publication=True) == {
        "name": "paddle"
    }


def test_normalize_still_validates_when_publication_is_not_required() -> None:
    assert normalize_ocr_config({"name": "template"})["name"] == "template"
    with pytest.raises(OCRConfigError, match="unknown key"):
        normalize_ocr_config({"name": "template", "nonsense": 1})


def test_normalize_validates_the_variant_list() -> None:
    with pytest.raises(OCRConfigError, match="unknown preprocessing variant"):
        normalize_ocr_config({"name": "template", "variants": ["blur"]})


# ------------------------------------------------------------------------ base plumbing


def test_stats_reports_the_backend_identity_and_the_counters() -> None:
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    stats = engine.stats()
    engine.close()
    assert stats["model_name"] == "glyph-template"
    assert stats["model_version"] == "1"
    assert stats["license"] == "Apache-2.0"
    assert stats["ships"] is False
    assert stats["variants"] == list(TEMPLATE_DEFAULT_VARIANTS)
    for key in (
        "plates_seen",
        "refused_small",
        "crops_empty",
        "reads_attempted",
        "reads_empty",
        "reads_returned",
        "variant_wins",
        "read_proxy",
        "ink_not_found",
        "grid_rejected",
        "score_too_low",
    ):
        assert key in stats


def test_read_proxy_is_named_a_proxy_because_of_its_denominator() -> None:
    """It counts every plate box handed to this stage, including the ones it was right
    to refuse. Calling it a read rate would make refusing correctly look like failing."""
    engine = _FlakyOCR({"raw": ("GJ01AB1234", 0.9)}, variants=("raw",))
    engine.load()
    crop = np.full((20, 66, 3), 128, np.uint8)
    engine.read_crop(crop, _candidate((0, 0, 60, 15)))
    engine.read_crop(crop, _candidate((0, 0, 20, 6)))
    assert engine.stats()["read_proxy"] == pytest.approx(0.5)


def test_read_proxy_is_zero_rather_than_undefined_before_any_plate() -> None:
    engine = _FlakyOCR({}, variants=("raw",))
    engine.load()
    assert engine.stats()["read_proxy"] == 0.0


def test_load_is_idempotent_and_close_is_safe_twice() -> None:
    engine = build_ocr_engine({"name": "template"})
    engine.load()
    engine.load()
    engine.close()
    engine.close()
    assert engine.stats()["model_name"] == "glyph-template"


def test_read_all_omits_the_plates_it_could_not_read() -> None:
    """A missing key means no plate, as in ai/plate. It is never an error."""
    engine = build_ocr_engine({"name": "scripted", "script": {0: ["GJ01AB1234"]}})
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    reads = engine.read_all_envelope(
        _Envelope(0, 0, frame),
        {7: _candidate((50, 50, 140, 72)), 9: _candidate((200, 100, 290, 122))},
    )
    engine.close()
    assert list(reads) == [7]


def test_read_all_envelope_begins_the_frame_and_read_all_does_not() -> None:
    """Only the backends that need frame identity get it, and they get it explicitly.

    A real backend's signature must not advertise access to information it may never use,
    which is why this is a separate entry point rather than an envelope threaded through
    read().
    """
    box = (100, 200, 190, 222)
    truth = FrameTruth(frame_index=1, pts_ms=120, vehicles=[_vehicle("GJ01AB1234", box)])
    engine = build_ocr_engine(
        {"name": "oracle", "variants": ["raw"]}, source=_TruthSource({120: truth})
    )
    engine.load()
    frame = np.full((300, 400, 3), 128, np.uint8)
    candidates = {1: _candidate(box)}
    assert engine.read_all(frame, candidates) == {}
    assert engine.read_all_envelope(_Envelope(1, 120, frame), candidates)[1].text == (
        "GJ01AB1234"
    )
    engine.close()


def test_the_default_variant_property_is_the_full_six() -> None:
    """Only TemplateOCR narrows it, and it does so in its own __init__."""
    engine = _FlakyOCR({})
    assert engine.variants == DEFAULT_VARIANTS
    assert TemplateOCR().variants == TEMPLATE_DEFAULT_VARIANTS


def test_an_explicit_variant_list_survives_construction() -> None:
    engine = TemplateOCR(variants=("raw", "sharpen"))
    assert engine.variants == ("raw", "sharpen")


@pytest.mark.parametrize(
    "engine_factory",
    [
        lambda: TemplateOCR(),
        lambda: ScriptedOCR({}),
        lambda: OracleOCR(_TruthSource({})),
    ],
)
def test_no_stub_backend_claims_to_ship(engine_factory) -> None:
    """ships means "may appear in a published accuracy claim", not "the licence is fine".

    TemplateOCR's licence is genuinely clean; what disqualifies it is that the only
    corpus it can be measured on is drawn with its own templates.
    """
    assert engine_factory().ships is False


def test_the_pipeline_can_swap_backends_without_changing_its_calls(
    corpus: Corpus,
) -> None:
    """Source independence, at this stage's boundary.

    The same three calls -- cut_crop, read_crop, stats -- serve every backend, which is
    what makes substituting the oracle a one-line change to a config rather than an edit
    to the pipeline.
    """
    row = next(r for r in corpus.legible if r.width > 80)
    engines = [
        build_ocr_engine({"name": "template"}),
        build_ocr_engine(
            {"name": "oracle", "variants": ["raw"]}, source=corpus.source
        ),
        build_ocr_engine(
            {"name": "scripted", "script": {row.frame_index: [(row.plate, 0.5)]}}
        ),
    ]
    for engine in engines:
        engine.load()
        read = engine.read_crop(row.crop, row.candidate, frame_ref=row.frame_ref)
        assert read is not None
        assert isinstance(read.text, str) and read.text
        assert 0.0 <= read.confidence <= 1.0
        assert engine.stats()["plates_seen"] == 1
        engine.close()
