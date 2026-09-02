"""OCR backends that need no weights: an oracle, a template matcher, a script.

    oracle    reads the synthetic source's ground-truth plate string, with an optional
              width-dependent character error model. Never ships.
    template  real OCR. Fits a character grid to the ink and correlates each cell
              against the 5x7 glyph masks in ai/media/glyphs.py. Never ships, and the
              reason is circularity rather than licensing -- see the class docstring.
    scripted  fixed strings per frame, for tests that need an exact read.

The template matcher is the one that makes the rest of the pipeline measurable. An
oracle returns the same string on every frame of a track, so temporal fusion has
nothing to fuse -- five identical votes prove nothing about a mechanism whose entire
job is resolving disagreement. The template matcher makes real per-frame character
errors that get worse as the plate shrinks, which is exactly the input ai/fusion,
ai/normalize and ai/quality were written against. With it, the primary metric --
correct final plate events over eligible vehicle events -- is computable end to end on
a machine with no downloads at all.
"""

import hashlib
from typing import Any, Optional, Sequence

import numpy as np

from ai.contracts.stages import BBox, PlateCandidate
from ai.ocr.base import OCRRead, BaseOCR

# ------------------------------------------------------------------------- oracle

# Minimum IoU between a plate candidate's box and a truth plate box for the oracle to
# treat them as the same plate. Lower than the vehicle-level match in ai/plate/stub.py
# because a plate box is small and a few pixels of error is a large IoU change: a 50x12
# box shifted 3 px horizontally has IoU 0.79 with itself, and the oracle should not
# start refusing to read plates because the detector was 3 px out.
ORACLE_PLATE_MATCH_IOU = 0.20

# Characters that OCR actually confuses, as pairs. Not random substitution: a random
# character error is trivially outvoted by temporal fusion because the wrong answers
# are all different, whereas a systematic confusion produces the SAME wrong character
# on every frame and can win a majority vote outright. Fusion has to be tested against
# the failure that can actually beat it.
#
# Chosen for the 5x7 shapes in ai/media/glyphs.py, which is also roughly what a real
# engine confuses at low resolution: pairs differing by one or two stroke cells.
CONFUSIONS: dict[str, str] = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
    "D": "0",
    "Q": "O",
    "7": "T", "T": "7",
    "4": "A",
}


class OracleOCR(BaseOCR):
    """The true plate string, optionally degraded by a width-dependent error model.

    Only works with source mode "synthetic". Two uses, and they are different:

    With defaults (no errors) it isolates everything downstream from OCR quality. When
    the end-to-end correct-plate rate is 0.62, substituting this answers "how much of
    that is OCR" in one run -- and if the rate barely moves, a day spent on the OCR
    model would have been a day wasted.

    With char_error_rate above zero it becomes a fusion test rig: real disagreement
    across frames of one track, with a realistic confusion structure, at a rate this
    file controls exactly. That is the only way to know that fuse() recovers the right
    plate from 5 votes with 2 errors, rather than merely that it returns something.

    Respects plate_legible, so it will not read a plate that is turned away. An oracle
    that read those would be measuring a capability no real backend can have.
    """

    def __init__(
        self,
        source: Any,
        *,
        char_error_rate: float = 0.0,
        error_full_width_px: int = 90,
        confidence: float = 0.95,
        min_confidence: float = 0.45,
        require_legible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source = source
        self._truth: Optional[Any] = None
        self.char_error_rate = float(char_error_rate)
        self.error_full_width_px = int(error_full_width_px)
        self.confidence = float(confidence)
        self.min_confidence = float(min_confidence)
        self.require_legible = bool(require_legible)
        self.no_truth_match = 0
        self.illegible_skipped = 0
        self.chars_corrupted = 0
        self.unresolved_frames = 0

    def _load(self) -> None:
        if not hasattr(self._source, "truth_for_envelope"):
            raise RuntimeError(
                f"OracleOCR needs a source exposing truth_for_envelope(); "
                f"{type(self._source).__name__} does not. Use source mode 'synthetic'."
            )

    def _begin_frame(self, envelope: Any) -> None:
        self._truth = self._source.truth_for_envelope(envelope)
        if self._truth is None:
            self.unresolved_frames += 1

    def _read_crop(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        """Ignores the pixels entirely, which is the point of an oracle.

        Runs once per preprocessing variant like every other backend and returns the
        same answer each time, so variants_agreeing comes out at the full count. That
        is correct rather than a quirk: with no pixel dependence there is genuinely no
        disagreement between variants, and a downstream stage weighting by agreement
        should see maximum agreement here.
        """
        if self._truth is None:
            return None

        vehicle = self._match_plate(candidate.plate_bbox_xyxy)
        if vehicle is None:
            self.no_truth_match += 1
            return None
        if self.require_legible and not vehicle.plate_legible:
            self.illegible_skipped += 1
            return None

        width = candidate.plate_width_px
        text = vehicle.plate
        confidence = self._confidence_for(width)

        if self.char_error_rate > 0.0:
            text = self._corrupt(text, width, vehicle.plate)

        return OCRRead(text=text, confidence=round(confidence, 4), variant="oracle")

    def _confidence_for(self, width_px: int) -> float:
        """Falls with plate width, linearly between the floor and full width.

        Linear rather than anything shaped, because the point is only that confidence
        and width move together -- a fitted curve here would be inventing a calibration
        the oracle has no basis for, and ai/quality is where the real relationship
        between width and reliability belongs.
        """
        if width_px >= self.error_full_width_px:
            return self.confidence
        span = max(1, self.error_full_width_px - self.min_plate_width_px)
        share = max(0.0, (width_px - self.min_plate_width_px) / span)
        return self.min_confidence + share * (self.confidence - self.min_confidence)

    def _corrupt(self, text: str, width_px: int, key: str) -> str:
        """Substitute confusable characters at a width-dependent rate.

        The rate scales as (full_width / width), so a 45 px plate is corrupted at twice
        the configured rate and a 90 px plate at exactly it. That inverse relationship
        is the one real property being modelled here: OCR error is driven by pixels per
        character, and pixels per character is linear in plate width.

        Keyed on the plate string, the character position and the frame index, so the
        errors are reproducible, differ between frames of the same track -- which is
        what gives fusion something to do -- and are independent of the order vehicles
        happen to be processed in.
        """
        scale = self.error_full_width_px / max(1, width_px)
        rate = min(0.9, self.char_error_rate * scale)
        frame = self._frame_index()

        out = []
        for position, char in enumerate(text):
            swap = CONFUSIONS.get(char)
            if swap is not None and _unit_hash(key, position, frame) < rate:
                out.append(swap)
                self.chars_corrupted += 1
            else:
                out.append(char)
        return "".join(out)

    def _frame_index(self) -> int:
        return int(getattr(self._truth, "frame_index", 0) or 0)

    def _match_plate(self, bbox: BBox) -> Optional[Any]:
        best, best_iou = None, 0.0
        for vehicle in self._truth.vehicles:
            if vehicle.plate_bbox_xyxy is None:
                continue
            iou = _iou(bbox, vehicle.plate_bbox_xyxy)
            if iou > best_iou:
                best, best_iou = vehicle, iou
        return best if best_iou >= ORACLE_PLATE_MATCH_IOU else None

    @property
    def model_name(self) -> str:
        return "oracle-ocr"

    @property
    def model_version(self) -> str:
        return "truth"

    @property
    def license_name(self) -> str:
        return "not-applicable"

    @property
    def ships(self) -> bool:
        """Never. It reads the answer key."""
        return False

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "no_truth_match": self.no_truth_match,
                "illegible_skipped": self.illegible_skipped,
                "chars_corrupted": self.chars_corrupted,
                "unresolved_frames": self.unresolved_frames,
                "char_error_rate": self.char_error_rate,
            }
        )
        return base


# ----------------------------------------------------------------------- template

# Ink is anything darker than this share of the way from the crop's darkest to its
# brightest pixel. Relative rather than absolute, so it survives the contrast_stretch
# and adaptive_threshold variants -- which move the absolute levels a long way -- and
# so it works on a night plate without a separate threshold.
TEMPLATE_INK_LEVEL = 0.45

# Character counts to hypothesise. Indian formats run from 6 (older two-letter state
# codes) to 10 characters; 4 to 12 brackets that with room either side, because the
# fit score is what selects, and giving it a hypothesis it can reject is cheaper than
# being unable to represent a plate that exists.
TEMPLATE_MIN_CHARS = 4
TEMPLATE_MAX_CHARS = 12

# Best-fit score below which the read is refused outright.
#
# Necessary because the grid always produces *something*: it fits the best available
# character to every cell whether or not there is a character there. Measured directly,
# by rendering all seven of the corpus's plate strings standalone with the generator's
# own _draw_plate and reading them with this floor removed:
#
#      24 px    7 reads    0 correct    0.216 - 0.244    all 7 refused by this floor
#      26 px    7 reads    0 correct    0.419 - 0.522    5 of 7 survive it
#      28 px    7 reads    0 correct    0.409 - 0.505    6 of 7 survive it
#
# So the floor removes the 24 px band outright and does *not* cover 26-29 px, where
# fabrications land squarely inside the correct-read range below. The corpus corroborates
# it: both of the two crops it offers in the 24-29 px band came back with a string, at
# 0.478 and 0.437, neither correct.
#
# An earlier version of this comment claimed all five test plates returned the same
# string "ZZZZ" at confidence 0.604 at 24 px. That measurement was real but it was of the
# *deleted* ring-stripping implementation recorded in _ink_mask's docstring, not of this
# one, and it does not describe this one: the seven fabrications are all different from
# each other, none reaches 0.604, and the band this floor fails to cover is one step
# higher than 24 px. The conclusion it was quoted for survives -- the matcher fabricates
# confidently on plates with no legible characters, and that is not fixable by tuning it,
# because the matcher is doing its job. Only the numbers belonged to other code.
#
# 0.42 is set from the measured distribution rather than guessed, and it does not
# separate the two populations. Measured on 514 reads of the synthetic road scenes in the
# default configuration:
#
#     correct   n=122   min 0.469   p10 0.506   median 0.602   max 0.772
#     wrong     n=392   min 0.420   p10 0.436   median 0.511   max 0.717
#
# The distributions overlap across almost their whole range, so no value of this constant
# separates them; the overlap is a property of the matcher, not of where the line is
# drawn. 0.42 sits just under the floor of the wrong-read distribution, which is the most
# it can honestly do -- it removes reads that are bad by the matcher's own reckoning and
# nothing more.
#
# **Deliberately not fitted to the lowest correct read.** That would have meant 0.49 when
# the six-variant configuration was measured, whose lowest correct read was 0.491. The
# default then narrowed to one variant and the lowest correct read moved to 0.469 -- so a
# constant fitted to the old minimum would now be discarding correct reads, silently, and
# the width-bucket table would have quietly lost accuracy with no code change to blame.
# The constant is deliberately looser than any observed minimum for exactly that reason:
# 122 samples of seven plate strings in one bitmap font do not locate a distribution's
# floor to three decimal places.
#
# So confident-looking garbage in the 0.42-0.60 band gets through -- measured at 9 of 19
# occluded plates, all wrong, at 0.442 to 0.568, four of them with no plate visible at
# all. That band is what ai/normalize/plate.py's grammar check and ai/fusion's cross-frame
# agreement exist to catch, and stacking three weak filters is the honest design here.
# Cross-frame agreement is the one that carries it: per-frame precision at this floor is
# 0.237, and taking each track's best read lifts it to 12 of 12 vehicles.
MIN_TEMPLATE_SCORE = 0.42

# Source pixels per character below which a length hypothesis is refused. See _fit_grid
# -- this replaced a one-pixel-per-glyph-column test that rejected correct answers at
# exactly the plate width that matters most.
MIN_PX_PER_CHAR = 2

# Width of the plate's painted border, in *source* pixels, from
# ai/media/synthetic_source.py's _draw_plate -- which strokes exactly one pixel on each
# edge. Insetting by it plus base.py's pad puts the crop's sampling window strictly
# inside the plate face, which is the whole reason the read no longer depends on whether
# the vehicle behind the plate is darker or brighter than the plate. Scaled by the
# variant's own magnification at use, so upscale_2x insets two.
PLATE_BORDER_PX = 1

# The variants this backend runs when a config does not say otherwise -- one, not the
# six in preprocess.py. Measured: of the 537 reads the six-variant configuration returns,
# upscale_2x wins 276 and raw wins 261, and the other four win nothing at all. Adding raw
# to upscale_2x then *loses* a track to max-of-N selection bias while roughly doubling the
# time. See TemplateOCR's docstring for the full table. Narrowed only for this backend,
# which exists to make CI cheap and is fixture-only anyway; DEFAULT_VARIANTS stays at
# six for ai/ocr/paddle.py, where there is no measurement yet and real-world glare is a
# failure mode this renderer cannot produce.
#
# A test that needs the variant loop exercised non-trivially -- anything asserting on
# OCRRead.agreement or variants_agreeing -- must pass variants explicitly.
TEMPLATE_DEFAULT_VARIANTS: tuple[str, ...] = ("upscale_2x",)

# Glyph cell geometry, matching ai/media/glyphs.py: 5 wide, 7 tall, 1 px gap. A string
# of L characters is therefore 6L-1 cells wide, which is what turns a hypothesised L
# into exact cell boundaries without needing to find the gaps.
CELL_W = 5
CELL_H = 7
CELL_GAP = 1


class TemplateOCR(BaseOCR):
    """Grid-fit template matching against the 5x7 glyph masks. numpy only.

    **Why this ships is False, and it is not about the licence.** The templates are the
    same 37 masks ai/media/synthetic_source.py renders the plates *with*. Measuring this
    backend on the synthetic corpus therefore measures a matcher against its own font,
    and any accuracy figure from it is circular -- it would be the best number in the
    project and would mean nothing. The code is honest, Apache-2.0 and genuinely
    classical OCR; what disqualifies it is the corpus it can be evaluated on. On real
    Sentinel frames it would read close to nothing, since real plates are not a 5x7
    bitmap font.

    **What it is for**, and it is worth more than its accuracy: it produces real,
    pixel-driven, frame-varying character errors that get worse as the plate shrinks.
    That makes temporal fusion, plate grammar normalisation and the confidence bands
    testable end to end with no model downloads. An oracle cannot do this -- it agrees
    with itself on every frame, so five votes are one vote counted five times.

    **Method: fit, do not segment.** Locate the ink bounding box, then for each
    hypothesised character count L, lay a 6L-1 cell grid across it, resample each 5x7
    cell, and correlate against all 37 glyphs. Score the hypothesis by mean best-match
    similarity and take the winning L.

    Segmenting on the gaps between characters was the obvious alternative and is
    fragile exactly where it matters: at 40 px plate width the inter-character gap is
    under one pixel, so after any resampling the characters touch and a projection-based
    segmenter returns one blob. Grid fitting has no gaps to find -- it degrades by
    getting the cell boundaries slightly wrong, which lowers every correlation a
    little and shows up as reduced confidence. Graceful, and honestly signalled.

    **Measured on the synthetic road scenes**, in the configuration below -- one variant,
    which is what TEMPLATE_DEFAULT_VARIANTS selects, so these figures reproduce from a
    plain build_ocr_engine({"name": "template"}). 400 generated frames of cam04 at seed
    42 and target_interval_ms 120, which emits 134 sampled frames carrying 12 vehicles;
    the shipped tracker config; and plate boxes from the oracle plate detector, so that
    localisation is exact and every figure here is about OCR alone. 550 plate crops reach
    this stage. Exact means the whole string matched.

        plate width    crops    exact    rate     char acc    refused
        > 100 px          27       17    0.630       0.911          0
        80 - 100         111       42    0.378       0.711          2
        60 - 80          210       40    0.190       0.582          7
        40 - 60          184       23    0.125       0.505         17
        30 - 40           10        0    0.000       0.080          4
        < 30               8        0    0.000       0.051          6
        all              550      122    0.222       0.583         36

    Buckets are ai/metrics.py's, so the boundary is the metrics module's: the one crop of
    exactly 100 px sits in 80-100, not in the row above it. Char acc is positional against
    truth -- matching characters in matching positions, over the truth string's length,
    with a refused crop contributing zero matches -- which is why it is not simply a
    softer version of the exact column. Positional scoring gives a truncation almost
    nothing: "J3C4567" for GJ3C4567 matches in no position at all. The edit-distance
    version of the same column reads 0.911 / 0.731 / 0.602 / 0.526 / 0.130 / 0.051, and
    the gap between the two is truncation specifically.

    Reported per bucket and never as the single 0.222, which is a statement about this
    corpus's width distribution rather than about the matcher. Monotonic in width in both
    columns, which is the only shape this table is allowed to have -- it came out
    non-monotonic twice during development and both times that was a bug in _ink_mask, not
    a small sample.

    The two smallest rows carry the tracker's warm-up as well as their own difficulty, in
    the same way ai/plate/stub.py's oracle row does, and the size of that effect is
    measured: with ByteTrack's min_hits dropped to 1 this stage is offered 564 crops
    instead of 550, and all 14 extra crops land in the three smallest buckets. They
    produce 7 more reads, 7 more refusals, and **zero** more correct reads. A track's
    warm-up frames are the frames where the vehicle is furthest away, so they are worth
    nothing to OCR -- which is the useful half of that measurement.

    **The number that matters is not in that table.** Per-frame precision at the score
    floor is 0.237 -- of 514 reads, 122 correct and 392 wrong. Taking each track's
    highest-confidence read instead gives the right plate for 12 of 12 vehicles. Frame
    accuracy 24%, track accuracy 100%, from the same reads. That gap is the entire
    empirical case for temporal fusion existing, and it works because confidence *ranks*
    without *separating*:

        correct reads   n=122   min 0.469   p10 0.506   median 0.602   max 0.772
        wrong reads     n=392   min 0.420   p10 0.436   median 0.511   max 0.717

    The two overlap across almost their whole range, but precision climbs monotonically
    with the threshold -- 0.237 at 0.42, 0.351 at 0.50, 0.460 at 0.55, 0.529 at 0.60,
    0.844 at 0.70 -- which is all a weighted vote needs. ai/fusion therefore weights by
    this number and does not threshold on it.

    The 12/12 and the non-crossing maxima both deserve a caveat. 12 vehicles is seven
    distinct plate strings, five of them drawn twice; it is a sanity check that fusion is
    possible, not a measurement of how well it works. And above 0.75 there are 6 reads in
    total, all correct -- too few to claim a clean region. Under the *six*-variant
    configuration the maxima did cross (correct max 0.862, wrong max 0.895), and that
    crossing was not a property of the matcher: taking the best of six reads means a
    variant that scores a wrong string unusually high gets selected. Max-of-N inflates the
    tail of the wrong distribution specifically. Second independent reason to narrow the
    default.

    **Fabrication rate, measured directly.** Re-run with the plate detector's
    require_legible off, so plates truth marks unreadable are handed over anyway: 19 such
    plates, all 59-69 px, illegible from *occlusion* rather than size. OCR returned a
    string for 9 of them, 0 correct, confidences 0.442 to 0.568 -- entirely inside the
    band MIN_TEMPLATE_SCORE documents as unseparated. Two distinct failures in there:

        vis 0.83   truth GJ3C4567   read J3C4567     0.552    truncation
        vis 0.74   truth GJ3C4567   read GJ3C45      0.550    truncation
        vis 0.00   truth GJ3C4567   read RWJT5T9T    0.442    invention
        vis 0.00   truth GJ3C4567   read ETJT74      0.471    invention

    The truncations read the visible characters correctly and stopped; cross-frame
    agreement recovers those, because a substring agrees with the full string on every
    character it has. The four inventions are 4 of the 10 crops at
    plate_visible_fraction 0.00 -- nothing of the plate was drawn and a string came back
    anyway. That is the failure Contracts section 12 names as the worst this pipeline can
    produce, and the honest position is that this stage does not prevent it: 4 of the 569
    crops this run offers survive the score floor with an invented plate. It is caught
    downstream, by grammar (RWJT5T9T fits no Indian plate format) and by agreement (an
    invention agrees with nothing, so it never accumulates weight). Three weak filters,
    each documented as weak.

    **Cost, and why the default is one variant.** Measured per plate on this corpus,
    550 crops in each row:

        6 default variants        119 exact   11/12 tracks    95.0 ms   p95 103.3
        raw + upscale_2x          119 exact   11/12 tracks    32.2 ms   p95  34.9
        upscale_2x alone          122 exact   12/12 tracks    16.8 ms   p95  19.1
        raw alone                  94 exact   11/12 tracks    15.7 ms   p95  17.3

    The millisecond figures are this machine under this load and are the two columns no
    test pins; the 5.7x ratio between the first and third rows is the number the decision
    rests on, and it is stable.

    Across all six, only raw and upscale_2x ever win a plate -- of the 537 reads the
    six-variant configuration returns, upscale_2x wins 276 and raw 261, so grayscale,
    contrast_stretch, adaptive_threshold and sharpen won nothing at all while costing two
    thirds of the stage's time. Worse, adding raw to upscale_2x *loses* a track: raw
    occasionally returns a higher-confidence wrong string than upscale_2x's correct one,
    which is max-of-N selection bias behaving exactly as OCRRead.agreement was written to
    expose. The accuracy difference is 3 crops in 550 and should not be read as
    significant; the latency difference should. Hence TEMPLATE_DEFAULT_VARIANTS.

    One more thing that table hides, and it is the reason agreement is worth reporting at
    all: under six variants variants_agreeing is 1 in 178 reads, 5 in 250, and 6 in 109 --
    never 2, 3 or 4. The four greyscale-derived variants always agree with each other, so
    the vote is really raw against upscale_2x with four abstentions attached to whichever
    of them wins. An agreement of 5/6 therefore means much less than it looks like.

    That upscale_2x dominates is the diagnostic ai/ocr/preprocess.py predicted: this
    stage is resolution-starved, not model-starved. It says nothing about PaddleOCR on
    real frames, where adaptive_threshold exists for glare this renderer never draws --
    which is why DEFAULT_VARIANTS keeps all six and only this backend narrows them.
    """

    def __init__(
        self,
        *,
        ink_level: float = TEMPLATE_INK_LEVEL,
        min_chars: int = TEMPLATE_MIN_CHARS,
        max_chars: int = TEMPLATE_MAX_CHARS,
        min_score: float = MIN_TEMPLATE_SCORE,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("variants", TEMPLATE_DEFAULT_VARIANTS)
        super().__init__(**kwargs)
        self.ink_level = float(ink_level)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.min_score = float(min_score)
        self._chars: tuple[str, ...] = ()
        self._stack: Optional[np.ndarray] = None
        self.ink_not_found = 0
        self.grid_rejected = 0
        self.score_too_low = 0

    def _load(self) -> None:
        from ai.media.glyphs import GLYPHS

        # Space is dropped: a plate's spaces are separators, and Contracts section 4.2
        # normalizes them out anyway. Leaving it in would let the matcher explain any
        # low-ink cell as a space and inflate every fit score.
        pairs = [(c, m) for c, m in sorted(GLYPHS.items()) if c != " "]
        self._chars = tuple(c for c, _ in pairs)
        # One (36, 7, 5) array rather than 36 separate ones, so matching a cell is four
        # numpy calls instead of a 36-iteration Python loop. The loop version cost
        # 8.6 ms a plate once the trim search below was added, which is most of a frame
        # budget for a backend that exists to make CI cheap.
        self._stack = np.stack([m.astype(np.float32) for _, m in pairs])

    def _read_crop(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        if self._stack is None:
            raise RuntimeError("TemplateOCR.load() was not called")

        ink = self._ink_mask(crop_bgr, candidate)
        if ink is None:
            self.ink_not_found += 1
            return None

        # Pass 1: the character count, with the grid assumed to span the ink box exactly.
        best: Optional[tuple[float, str, tuple[float, ...], int]] = None
        for count in range(self.min_chars, self.max_chars + 1):
            fit = self._fit_grid(ink, count, 0, 0)
            if fit is None:
                continue
            if best is None or fit[0] > best[0]:
                best = (*fit, count)

        if best is None:
            self.grid_rejected += 1
            return None

        # Pass 2: the edge trim, with the count fixed. The ink box is the *ink's* extent,
        # not the text's, and they differ when the first or last glyph has an empty edge
        # column -- '1' has both edge columns empty in this font. That makes the box up
        # to two cells of 6L-1 narrower than the grid assumes, so the cell pitch is
        # overestimated by 3.4% and the error accumulates leftward across the plate: a
        # third of a character by the far end, which is enough to lose the last two or
        # three. Measured as DL8CAF5031 -- the one plate of five ending in '1' -- reading
        # DL8CAF5Z2J at *every* width including 200 px, while the other four were exact.
        #
        # Two passes rather than searching count and trim jointly. The trim is a
        # sub-character correction and does not change which count wins, since the count
        # is fixed by the coarse periodicity of the strokes; searching jointly costs 36
        # grid fits per variant against 13 for this, for the same answer.
        count = best[3]
        for left in (0, 1):
            for right in (0, 1):
                if left == 0 and right == 0:
                    continue
                fit = self._fit_grid(ink, count, left, right)
                if fit is not None and fit[0] > best[0]:
                    best = (*fit, count)

        score, text, per_char, _ = best
        if score < self.min_score:
            # A fabricated string is the worst output this pipeline can produce, so a
            # fit this poor is reported as no read at all. See MIN_TEMPLATE_SCORE.
            self.score_too_low += 1
            return None

        # The similarity is already 0..1 and is not rescaled. A matcher that reports a
        # 0.71 mean cell similarity as a 0.9 confidence is asserting a calibration it
        # has not earned, and ai/fusion weights by this number directly.
        return OCRRead(
            text=text,
            confidence=round(score, 4),
            variant="template",
            char_confidences=per_char,
        )

    def _ink_mask(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[np.ndarray]:
        """Boolean ink mask of the text, taken from inside the known plate rectangle.

        The plate's position is not searched for. It is already known -- the candidate
        carries plate_bbox_xyxy and base.py built this crop by padding that box by
        pad_px -- so the face is found by arithmetic, not by looking at pixels.

        **Two pixel-based alternatives were implemented and both failed, in ways worth
        recording because each looked sound.**

        *Stripping the dark ring by scanning inward from each edge* needs a depth cap or
        text touching the crop edge gets eaten. But the ring is a fixed pixel count --
        pad plus 1-2 px of painted border -- while the crop shrinks with the plate, so
        the ring's *share* of the crop grows without limit. At 24 px plate width the crop
        is 11 rows of which 8 are ring: 73%. A cap of 25% left two ink rows standing, the
        grid fitted them, and all five test plates returned "ZZZZ" at confidence 0.604 --
        one fabricated string, identical across five different vehicles. No fractional cap
        fixes that, because the quantity being capped is not fractional.

        That "ZZZZ" figure is of *this deleted implementation*, and it escaped: it was
        quoted in MIN_TEMPLATE_SCORE's comment and in ai/ocr/factory.py's
        check_ocr_width_floor as though it described the shipped matcher, which it never
        did. The shipped one fabricates seven different strings at 24 px, at 0.216-0.244.
        Both sites now carry the measured figures. Worth naming the mechanism, because it
        is the cheapest error in this file to make: a number measured against code that
        no longer exists reads exactly like a number measured against code that does.

        *Locating the brightest connected region and calling it the face* has no depth
        parameter, passed every isolated-plate test, and then failed on the road scenes
        for one specific reason: **it assumes the plate is the brightest thing in the
        crop, and on a light-coloured vehicle it is not.** Measured on the synthetic
        corpus at 100-112 px plate width, oracle plate boxes, exact:

            pad ring darker than plate (cars, motorcycles)   7 of 15 exact, conf 0.58-0.72
            pad ring brighter than plate (buses)             0 of 13 exact, conf 0.43-0.47

        The bus reads all begin 'E' or 'B' where truth has 'G'. That is the signature: the
        face box had grown to the whole crop, so the plate's own painted border landed
        inside the leftmost grid cell and a bordered 'G' matches 'E'. Overall exact rate
        was 0.135; with pad_px forced to 0 -- which removes the bodywork from the crop
        entirely -- it was 0.213 on the same corpus, and tracks 8, 11 and 12 went from
        0 exact to 21, 8 and 11. The pad was the whole difference, so the fix is to
        exclude it by construction rather than to remove it: a real plate detector's box
        is approximate and the pad is why the crop survives that.

        Those two rates are of the deleted implementation, measured when the default was
        six variants, and are here as the record of why this code changed rather than as
        figures to compare against the class docstring's table.

        The border allowance is one *source* pixel, which is what
        ai/media/synthetic_source.py strokes. Scale is recovered from the crop's size
        against the known box size, so upscale_2x's doubled crop gets a doubled
        allowance. That calibration is specific to this renderer, which is consistent
        with this backend being fixture-only -- ai/ocr/paddle.py does its own
        localisation and needs none of it.
        """
        grey = (
            crop_bgr.astype(np.float32).mean(axis=2)
            if crop_bgr.ndim == 3
            else crop_bgr.astype(np.float32)
        )
        crop_h, crop_w = grey.shape
        box = candidate.plate_bbox_xyxy
        box_w = max(1, box[2] - box[0])
        box_h = max(1, box[3] - box[1])

        # The uniform scale this variant applied. Taken as the larger of the two ratios
        # because a crop clipped at the frame edge is smaller than the padded box in one
        # or both axes, which biases each ratio downward but never upward.
        scale = max(crop_w / (box_w + 2 * self.pad_px), crop_h / (box_h + 2 * self.pad_px))
        scale = max(1.0, scale)
        inset = int(round((self.pad_px + PLATE_BORDER_PX) * scale))

        # Clamped so the inset can never consume the plate. At 24 px the pad and border
        # really are most of the crop, and the right answer there is to hand the matcher
        # whatever interior exists and let the score floor refuse it -- not to inset into
        # nothing and call that a refusal for the wrong reason.
        max_inset_x = max(0, (crop_w - self.min_chars * MIN_PX_PER_CHAR) // 2)
        max_inset_y = max(0, (crop_h - 3) // 2)
        inset_x = min(inset, max_inset_x)
        inset_y = min(inset, max_inset_y)

        inner = grey[
            inset_y : crop_h - inset_y if inset_y else crop_h,
            inset_x : crop_w - inset_x if inset_x else crop_w,
        ]
        if inner.shape[0] < 3 or inner.shape[1] < self.min_chars * MIN_PX_PER_CHAR:
            return None

        # Ink is measured against the plate face's own range, not the whole crop's. The
        # crop's darkest pixel may be bodywork in the pad, which would put the threshold
        # below the text and find nothing.
        inner_low, inner_high = float(inner.min()), float(inner.max())
        if inner_high - inner_low < 12.0:
            # No contrast inside the plate. Either there is no text there -- which is the
            # honest answer below the renderer's legibility floor -- or the variant
            # flattened it. A refusal, not a guess.
            return None
        ink = inner < (inner_low + (inner_high - inner_low) * self.ink_level)

        rows = np.flatnonzero(ink.any(axis=1))
        cols = np.flatnonzero(ink.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return None
        boxed = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if boxed.shape[0] < 3 or boxed.shape[1] < self.min_chars * MIN_PX_PER_CHAR:
            return None
        return boxed

    def _fit_grid(
        self, ink: np.ndarray, count: int, left_trim: int, right_trim: int
    ) -> Optional[tuple[float, str, tuple[float, ...]]]:
        """Lay a count-character grid over the ink box and match every cell.

        Cell k spans the fraction [6k, 6k+5] of the 6*count-1 cell columns, which is
        the same arithmetic ai/media/glyphs.py uses to lay text out. Derived from the
        glyph geometry rather than measured from this image, so it holds for any
        rendering of the same font at any size.

        left_trim and right_trim say how many cell columns of the nominal extent are
        *missing* from the ink box because the first or last glyph has an empty edge
        column. They shift and rescale the grid rather than changing the cell count.
        """
        assert self._stack is not None
        height, width = ink.shape
        if width < count * MIN_PX_PER_CHAR:
            # Fewer than two source pixels per character. Below that the grid's cell
            # boundaries fall inside single pixels, so adjacent cells resample the same
            # values and the fit is arithmetic on one number repeated -- and 2 px per
            # character is also where ai/ocr/base.py says 0, O, D, 8 and B stop being
            # distinguishable, so nothing is lost by refusing.
            #
            # The obvious stricter test -- width < total_cells, one pixel per glyph
            # *column* -- is wrong and was measured wrong: at 60 px plate width the ink
            # box is 56 px against 59 cells, so the true 10-character hypothesis was
            # rejected while 9 was accepted, and the read came back one character short
            # with everything after the second character shifted. The 60 px bucket is
            # where a real junction plate lives; refusing the correct answer there
            # because it is 3 px under a round number is the expensive kind of wrong.
            return None

        total_cells = CELL_W * count + CELL_GAP * (count - 1)
        # The ink box covers cells [left_trim, total_cells - right_trim), so a cell
        # column maps to pixels at this pitch and offset.
        visible = total_cells - left_trim - right_trim
        if visible <= 0:
            return None
        pitch = width / visible

        chars: list[str] = []
        scores: list[float] = []
        for index in range(count):
            cell_start = index * (CELL_W + CELL_GAP) - left_trim
            start = int(round(cell_start * pitch))
            end = int(round((cell_start + CELL_W) * pitch))
            start = max(0, min(start, width - 1))
            end = max(start + 1, min(end, width))
            cell = _resample_cell(ink[:, start:end], CELL_W, CELL_H)

            char, score = self._best_glyph(cell)
            chars.append(char)
            scores.append(score)

        # Mean over cells, so a hypothesis is not rewarded for having fewer of them.
        # Without the mean, count=4 would win every time by matching four cells well and
        # ignoring the rest of the plate.
        return float(np.mean(scores)), "".join(chars), tuple(round(s, 4) for s in scores)

    def _best_glyph(self, cell: np.ndarray) -> tuple[str, float]:
        """Closest of the 36 glyph masks, by soft Jaccard overlap of the ink.

        Intersection over union of ink, not mean absolute difference over all 35 cells,
        and the difference decides whether this backend works at all.

        **Measured.** Ink is rare in a 5x7 glyph -- 10 cells of 35 for '1' and 'Y', 20
        for 'B'. Under mean absolute difference an all-background cell therefore scores
        1 - 10/35 = 0.714 against the sparsest glyph, and 0.714 sat *inside* the range of
        scores real reads were producing (0.688 to 0.849). An empty cell was competitive
        with a correct character. That inverted the length search: a wrong character
        count puts cell boundaries in the inter-character gaps, those cells come back
        mostly empty, they score 0.71 each as '1' or 'L' or 'J', and the hypothesis wins.
        Directly observed -- MH12DE1433 read as YLJLE1433 at 200 px plate width, a
        9-character fit to a 10-character plate, with every substitution a sparse glyph.

        Jaccard scores that same empty cell 0.000, because an empty intersection is an
        empty intersection regardless of how little ink the template has. An all-*ink*
        cell scores 20/35 = 0.571 against 'B', which is the right answer rather than a
        residual problem: a saturated cell genuinely is more consistent with a dense
        glyph than a sparse one, and 0.571 is still below any real match.

        Normalised correlation would also fix the empty-cell case and introduces a
        different error -- it is invariant to ink density, so a cell 80% covered in ink
        correlates perfectly with a glyph 30% covered, and 8 reads as B.
        """
        assert self._stack is not None
        inter = np.minimum(cell, self._stack).sum(axis=(1, 2))
        union = np.maximum(cell, self._stack).sum(axis=(1, 2))
        scores = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        index = int(scores.argmax())
        return self._chars[index], float(scores[index])

    def _implied_char_count(self, ink: np.ndarray) -> Optional[float]:
        """Character count the ink box's aspect ratio would imply. Diagnostic only.

        **Not used to select the length, and the reason is a measured negative result
        worth keeping.** A string of L characters is 6L-1 glyph cells wide and 7 tall, so
        an ink box of aspect a should imply L = (7a + 1) / 6 -- and against
        ai/media/glyphs.text_extent that formula is exact, a 10-character string
        measuring 59x7, aspect 8.429, implied L 10.00.

        It does not survive rendering. ai/media/synthetic_source.py stretches the 59x7
        mask to fill the plate's whole inner rectangle rather than preserving the glyph
        aspect, so the ink box of a rendered 10-character plate measures aspect 4.18 at
        200 px plate width, 4.80 at 100 px and 5.86 at 45 px. Those imply 5.4, 6.1 and
        7.2 characters. Used as a search window it forced 6-character fits onto
        10-character plates and took the exact-read rate to zero.

        The deeper point survives the fixture: plate aspect carries no length
        information, because a plate is a fixed physical size -- 500x120 mm for an Indian
        single-row car plate -- and a 6-character registration is set in wider characters
        rather than on a shorter plate. Any length prior taken from the box is measuring
        the plate, not the string.

        Kept because the number is useful when debugging a misaligned grid: if this and
        the winning count disagree by more than about a factor of the stretch, the ink
        box has caught something that is not text.
        """
        height, width = ink.shape
        if height <= 0:
            return None
        aspect = width / height
        return (CELL_H * aspect + CELL_GAP) / (CELL_W + CELL_GAP)

    @property
    def model_name(self) -> str:
        return "glyph-template"

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    @property
    def ships(self) -> bool:
        """False for circularity, not licensing. See the class docstring."""
        return False

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "ink_not_found": self.ink_not_found,
                "grid_rejected": self.grid_rejected,
                "score_too_low": self.score_too_low,
            }
        )
        return base


# ----------------------------------------------------------------------- scripted


class ScriptedOCR(BaseOCR):
    """Fixed reads per frame index. For tests that need an exact string.

    Keyed on frame index rather than on the plate box, so a test describes a whole
    frame's reads in one line. A frame with no entry produces no read at all, which is
    how the "plate found, not readable" path -- the one that must emit plate: null --
    gets exercised deterministically.
    """

    def __init__(
        self,
        script: dict[int, Sequence[tuple[str, float]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._script = {int(k): list(v) for k, v in script.items()}
        self._frame_index = 0
        self._cursor = 0

    def _load(self) -> None:
        """Nothing to load."""

    def _begin_frame(self, envelope: Any) -> None:
        self._frame_index = int(getattr(envelope, "frame_index", 0))
        self._cursor = 0

    def _read_crop(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        entries = self._script.get(self._frame_index)
        if not entries:
            return None
        # One entry per plate in the frame, consumed in the order read_all iterates.
        # Divided by the variant count because _read_crop is called once per variant
        # and the script describes plates, not variants.
        index = self._cursor // max(1, len(self.variants))
        self._cursor += 1
        if index >= len(entries):
            return None
        text, confidence = entries[index]
        return OCRRead(text=text, confidence=float(confidence), variant="scripted")

    @property
    def model_name(self) -> str:
        return "scripted-ocr"

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def license_name(self) -> str:
        return "not-applicable"

    @property
    def ships(self) -> bool:
        return False


# ---------------------------------------------------------------------- internals


def _resample_cell(cell: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average a boolean cell down to width x height floats in 0..1.

    Area averaging rather than nearest-neighbour sampling, because a 1 px glyph stroke
    in a 12 px cell is hit by nearest sampling only if a sample point lands on it --
    so the same character reads differently depending on sub-pixel alignment, and the
    read becomes a function of where the plate happens to sit in the frame. Averaging
    turns a partly-covered cell into an intermediate value, which is both stable and
    the correct representation of the uncertainty.
    """
    src_h, src_w = cell.shape
    if src_h == 0 or src_w == 0:
        return np.zeros((height, width), dtype=np.float32)

    values = cell.astype(np.float32)
    row_edges = np.linspace(0, src_h, height + 1)
    col_edges = np.linspace(0, src_w, width + 1)

    out = np.zeros((height, width), dtype=np.float32)
    for r in range(height):
        r0, r1 = int(np.floor(row_edges[r])), max(int(np.ceil(row_edges[r + 1])), int(np.floor(row_edges[r])) + 1)
        for c in range(width):
            c0, c1 = int(np.floor(col_edges[c])), max(int(np.ceil(col_edges[c + 1])), int(np.floor(col_edges[c])) + 1)
            patch = values[r0 : min(r1, src_h), c0 : min(c1, src_w)]
            out[r, c] = patch.mean() if patch.size else 0.0
    return out


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _unit_hash(key: str, position: int, frame_index: int) -> float:
    """Deterministic 0..1 from (plate, character position, frame)."""
    digest = hashlib.sha256(f"ocr-err|{key}|{position}|{frame_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
