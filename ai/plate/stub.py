"""Plate detectors that need no weights: an oracle, an edge detector, a script.

    oracle    reads the synthetic source's ground truth. Measures everything
              downstream of plate detection without plate detection being a
              variable. Never ships.
    edge      real numpy detector. Vertical-edge density projection. Ships in the
              sense that it is honest code with a stated licence, but it is a
              fallback, not a competitor to a trained model.
    scripted  fixed answers per frame, for tests that need an exact plate box.

The oracle exists for one specific measurement. When the end-to-end plate accuracy
on a clip is 0.62, that number is the product of five stages, and the useful question
is which one to spend the next day on. Substituting a perfect plate detector answers
it directly: if accuracy goes to 0.94, plate detection is the bottleneck; if it goes
to 0.66, the plate detector was never the problem and a day spent on it would have
been wasted. That is worth more than any single accuracy figure.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ai.contracts.stages import BBox, TrackResult
from ai.plate.base import BasePlateDetector
from ai.plate.geometry import aspect_ratio

# ------------------------------------------------------------------------- oracle

# Minimum IoU between a tracker box and a truth vehicle box for the oracle to accept
# them as the same vehicle. Low, because the tracker box is allowed to be imperfect --
# the oracle is standing in for the plate detector, not for the tracker, and holding
# it to a tight box match would make plate-stage results depend on tracker quality,
# which is the coupling this backend exists to remove.
ORACLE_MATCH_IOU = 0.30


class OraclePlateDetector(BasePlateDetector):
    """Perfect plate boxes, read from the synthetic generator's ground truth.

    Only works with source mode "synthetic", and refuses to construct otherwise
    rather than silently returning nothing.

    Respects plate_legible. A vehicle whose plate is turned away or occluded past the
    legibility threshold gets no candidate, because a perfect plate *detector* still
    cannot find a plate that is not visible -- and an oracle that returned a box
    anyway would be measuring something no real detector could achieve, which makes
    the comparison it exists for meaningless.
    """

    def __init__(
        self,
        source: Any,
        *,
        miss_rate: float = 0.0,
        confidence: float = 0.95,
        require_legible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source = source
        self._truth: Optional[Any] = None
        self.miss_rate = float(miss_rate)
        self.confidence = float(confidence)
        self.require_legible = bool(require_legible)
        self.unresolved_frames = 0
        self.illegible_skipped = 0
        self.no_truth_match = 0

    def _load(self) -> None:
        if not hasattr(self._source, "truth_for_envelope"):
            raise RuntimeError(
                f"OraclePlateDetector needs a source exposing truth_for_envelope(); "
                f"{type(self._source).__name__} does not. Use source mode 'synthetic'."
            )

    def _begin_frame(self, envelope: Any) -> None:
        self._truth = self._source.truth_for_envelope(envelope)
        if self._truth is None:
            self.unresolved_frames += 1

    def _detect_in_crop(
        self, crop_bgr: np.ndarray, track: TrackResult
    ) -> Sequence[tuple[BBox, float]]:
        """Truth plate box for whichever vehicle this track is following.

        Returns crop-local coordinates like every other backend, which for this one
        means converting *back* from the full-frame truth box. Slightly perverse, and
        deliberate: the alternative is a second code path through the base class for
        backends that already know frame coordinates, and then the mapping that
        Contracts section 3 warns about exists in two places instead of one.
        """
        if self._truth is None:
            return ()

        vehicle = self._match_vehicle(track.bbox_xyxy)
        if vehicle is None:
            self.no_truth_match += 1
            return ()
        if vehicle.plate_bbox_xyxy is None:
            return ()
        if self.require_legible and not vehicle.plate_legible:
            self.illegible_skipped += 1
            return ()
        if self.miss_rate > 0.0 and _unit_hash(vehicle.plate, track.frame_index) < self.miss_rate:
            return ()

        # Full frame -> crop-local. The crop origin is not passed to this method, so
        # it is recovered from the difference between the track box and the crop the
        # base class took. Both are derived from the same bbox and the same pad
        # fraction, so recomputing is exact rather than approximate.
        from ai.plate.geometry import crop_vehicle

        _, origin = crop_vehicle(
            np.empty((10_000, 10_000, 3), dtype=np.uint8),
            track.bbox_xyxy,
            pad_fraction=self.pad_fraction,
        )
        px1, py1, px2, py2 = vehicle.plate_bbox_xyxy
        local = (px1 - origin[0], py1 - origin[1], px2 - origin[0], py2 - origin[1])
        return ((local, self.confidence),)

    def _match_vehicle(self, bbox: BBox) -> Optional[Any]:
        """The truth vehicle this tracker box is following, by best IoU."""
        best, best_iou = None, 0.0
        for vehicle in self._truth.vehicles:
            iou = _iou(bbox, vehicle.vehicle_bbox_xyxy)
            if iou > best_iou:
                best, best_iou = vehicle, iou
        return best if best_iou >= ORACLE_MATCH_IOU else None

    @property
    def model_name(self) -> str:
        return "oracle-plate"

    @property
    def model_version(self) -> str:
        return "truth"

    @property
    def license_name(self) -> str:
        return "not-applicable"

    @property
    def ships(self) -> bool:
        """Never. This reads the answer key.

        Hard-coded False rather than derived from the licence, because the licence
        string is not the reason -- a permissively licensed oracle would be just as
        disqualified. ai/metrics.py refuses to publish a run using it.
        """
        return False

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "unresolved_frames": self.unresolved_frames,
                "illegible_skipped": self.illegible_skipped,
                "no_truth_match": self.no_truth_match,
            }
        )
        return base


# --------------------------------------------------------------------------- edge

# A plate's characters are near-vertical strokes on a contrasting background, so the
# horizontal gradient over a plate is far denser than over bodywork. These numbers turn
# that into a box.
#
# The two projections need *different kinds* of threshold, and using the same kind for
# both is what made the first version of this detector return nothing at all on every
# frame at every parameter setting. See _contiguous_band and the class docstring.
EDGE_GRADIENT_THRESHOLD = 50      # per-pixel |d/dx| counted as a stroke edge

# Rows: absolute, expressed as a fraction of the search region's width. A plate row
# crosses roughly two stroke edges per character, so ten characters across put 20-odd
# stroke pixels in a row -- 0.20 of a 100 px-wide region. The bar sits below that.
#
# Absolute rather than relative to the peak row, which is what the first version used.
# Peak-relative made gradient_threshold inert: lowering it raises every row's count
# including the peak's, so the ratio between them barely moves. Measured directly on the
# fixture the class docstring describes -- the median number of rows clearing 80% of the
# peak row is 6 at gradient threshold 34, 6 at 20 and 6 at 10 (means 7.3, 7.3, 7.4), so
# a three-fold change in the threshold moves the row band not at all, and every
# parameter configuration of the first version returned the same boxes because of it.
#
# The absolute bar is flat across those same three thresholds too (median 12 rows at
# each), for the mundane reason that 34, 20 and 10 all sit far below the stroke gradient
# on a clean synthetic glyph. Its responsiveness shows up across the full threshold
# range and under reduced contrast, which is what the table in the class docstring
# measures: 0.637 to 0.726 to 0.000 as threshold and contrast move against each other.
# A fixed bar gives the gradient threshold something to move against; a peak-relative
# one gives it nothing anywhere.
EDGE_ROW_MIN_FILL = 0.06
EDGE_MIN_ROWS = 3                 # a 40 px plate is 10 rows tall; 3 is the noise floor

# Columns: relative to the peak, but only *after* smoothing, and the order matters.
# Within the row band a column through a character stroke counts ~band_height and a
# column in the gap between characters counts ~0, so the raw column profile of a plate
# is a comb, not a plateau. Banding it directly asks for a contiguous run of dense
# columns, which is the one thing a plate does not have -- hence the running maximum
# below, which merges the teeth into the plateau the band logic is looking for.
#
# 0.45 rather than the 0.60 that scores 0.022 higher recall (0.664 -> 0.686), because
# 0.60 returns 20 boxes that fail the IoU >= 0.3 bar against 0.45's one, and drops mean
# IoU over all returned boxes from 0.736 to 0.577.
#
# Those 20 are not spurious detections somewhere else in the crop. Every one of them
# overlaps a real plate -- the lowest scores 0.187 and none score zero -- so what 0.60
# buys is not extra plates found but the same plates found with looser boxes, some of
# them loose enough to fall under the bar. An earlier version of this comment described
# them as boxes that "overlap no plate at all", which is both false and a weaker
# argument than the truth: a box that has drifted off the plate entirely is a
# false positive the OCR stage will fail to read and report as unreadable, whereas a box
# that is on the plate but bounded 20% too wide feeds OCR a clipped first character and
# yields a confident wrong string. Contracts section 12 names that as the worst outcome
# the pipeline can produce -- worse than the plate: null a missing box produces.
EDGE_COL_QUANTILE = 0.45
EDGE_MIN_COLS = 8                 # narrower than this is one character, not a plate


class EdgePlateDetector(BasePlateDetector):
    """Vertical-edge density projection. numpy only, no weights, no downloads.

    **What this is for.** The pipeline must be runnable end to end on a machine with
    no checkpoints -- that is what makes it possible to test the other thirteen stages
    on a CI box, and what makes the offline-to-live swap demonstrably a config change
    rather than a code change. A stage that can only run with a 90 MB download is a
    stage that cannot be tested that way.

    **What this is not.** A trained plate detector. It finds high-contrast regions of
    dense vertical structure in the lower half of a vehicle, which on a real vehicle
    also describes a grille, a radiator mesh, a bull bar, and the lettering across a
    truck tailgate. Expect it to work on clean synthetic frames and to degrade badly
    on a wet night junction. It exists so the pipeline runs, and its output must never
    appear in a published accuracy claim -- ships is False for that reason, even
    though the code carries no licence encumbrance at all.

    **Method**, deliberately the oldest one in the literature rather than anything
    clever: convert to grey, take the absolute horizontal difference, count per-row how
    many pixels exceed a stroke threshold, take the contiguous band of dense rows, then
    do the same across columns within that band. The two projections need different
    kinds of threshold and different pre-processing, and getting either wrong makes the
    detector return nothing at all -- see the notes on each constant above.

    **Measured** on the synthetic road scenes, and re-runnable: source mode synthetic,
    camera cam01, seed 42, total_frames 200, target_interval_ms 120. That fixture emits
    67 frames at 1280x720 -- 200 is the *raw* frame count, and an earlier version of
    this docstring reported it as "200 emitted frames", which sent anyone reproducing
    the table to a fixture three times the size with 894 legible plate-frames instead
    of 226. The clip contains 6 distinct vehicles across 240 truth vehicle-frames, of
    which 226 have a legible plate; around 3.2 vehicles are on screen per emitted frame.
    The vehicle detector is the oracle at miss_rate 0, so the numbers describe this
    stage rather than the detector's. A hit is a returned box overlapping the true plate
    box by IoU >= 0.3, and the precision column applies the same 0.3 bar to each
    returned box. Compared against OraclePlateDetector, which reads the truth boxes and
    therefore fixes the ceiling this can be measured against:

        oracle    recall 0.956    mean IoU 1.000    216/216 boxes correct
        edge      recall 0.664    mean IoU 0.736    150/151 boxes correct

    Mean IoU is over every returned box, not over the hits alone. The distinction is
    small here and worth stating because it is not always -- edge scores 0.740 over its
    150 hits and 0.736 once its one sub-threshold box is included, and a setting that
    trades tight boxes for more of them moves the two figures in opposite directions
    (see EDGE_COL_QUANTILE above, where 0.60 raises recall and drops mean IoU to 0.577).

    The oracle's own 0.956 is not a bug, and it is not this stage: 10 of the 226 have no
    emitted track at all on the frame in question, because ByteTrack needs
    DEFAULT_MIN_HITS = 3 hits before a track is CONFIRMED and BaseTracker.update returns
    only active tracks. Dropping min_hits to 1 recovers 7 of the 10. An earlier version
    of this line blamed the tracker's box match failing against ORACLE_MATCH_IOU, which
    is measurably false -- no_truth_match is 0 across the whole run.

    All 10 land in 40-60 px for a mechanical reason: a vehicle enters the frame far away,
    so its plate is small exactly while its track is still TENTATIVE, and by the time the
    track confirms the plate has grown past 60 px. The small-plate bucket therefore
    carries the tracker's warm-up on top of its own difficulty, so the oracle row is the
    ceiling for the *pipeline* rather than for this stage alone.

    By plate width, which is the only breakdown that means anything here:

        40-60 px     17/ 77   0.221      (oracle 67/77  0.870)
        60-80 px     71/ 86   0.826      (oracle 1.000)
        80-100 px    54/ 54   1.000      (oracle 1.000)
        >100 px       8/  9   0.889      (oracle 1.000)

    So it is usable above 80 px and nearly useless below 60 -- and 40-60 px is where
    real junction plates mostly live. That single row is the argument for the trained
    model, and it is a stronger argument than any average would have been.

    0.70 ms median per frame, 1.00 ms p95, at 3.2 vehicles a frame on 1280x720. Three
    orders of magnitude cheaper than the vehicle detector, so on the edge-device
    fallback path this stage is free. These two are the only numbers here that are
    machine-dependent, and no test pins them.

    **gradient_threshold: the effective parameter is threshold/contrast, and both
    failure directions are measured.** Sweeping the threshold against reduced crop
    contrast produces one table with a diagonal in it. Every cell below is measured, and
    the diagonal is the finding:

        recall        100%     60%     40%     25%     15%     10%
          th= 20     0.637   0.637   0.664   0.717   0.726   0.726
          th= 34     0.637   0.664   0.717   0.726   0.000   0.000
          th= 50     0.664   0.717   0.726   0.726   0.000   0.000
          th= 80     0.717   0.726   0.726   0.000   0.000   0.000
          th=120     0.726   0.726   0.000   0.000   0.000   0.000

    0.726 appears at (120, 100%), (80, 60%), (50, 40%), (34, 25%) and (20, 15%) -- five
    cells along one diagonal, exactly, and 0.000 immediately past each one. That is the
    expected shape: the threshold is compared against gradients that scale linearly with
    contrast, so halving the contrast and halving the threshold is the same detector.

    50 is the default because it has the same working contrast range as 34 and higher
    recall at every point inside it except the bottom, where the two meet at 0.726 --
    and because grain moves it almost not at all (0.664 at +/-0, 0.664 at +/-6, 0.659 at
    +/-10, 0.650 at +/-24).

    **Two settings scored better on the fixture and are not the default.** Threshold 5
    gives recall 0.779 and IoU 0.754 with perfect precision -- 176 boxes, all 176 above
    the 0.3 bar, the best recall in this file -- and collapses to 0.058 under +/-6 grain
    and to 0.000 under +/-10. The generator flat-shades vehicle bodywork, so a threshold
    of 5/255 is measuring a vehicle with no texture, which no real camera can promise.
    Threshold 120 is the fixture optimum at full contrast and fails outright below 40%,
    because synthetic glyphs are rendered black-on-white at maximum contrast and a dirty
    plate under sodium light is not. Both are the same mistake in opposite directions,
    and neither was visible until the fixture was perturbed in the axis each one
    depends on.
    """

    def __init__(
        self,
        *,
        gradient_threshold: int = EDGE_GRADIENT_THRESHOLD,
        row_min_fill: float = EDGE_ROW_MIN_FILL,
        col_quantile: float = EDGE_COL_QUANTILE,
        search_lower_fraction: float = 0.45,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.gradient_threshold = int(gradient_threshold)
        self.row_min_fill = float(row_min_fill)
        self.col_quantile = float(col_quantile)
        # Search only the lower part of the crop. Unlike region_prior in geometry.py
        # this IS a hard cut, and the difference is justified: the prior demotes a
        # box a trained model proposed, whereas here the whole detection is a
        # projection profile, and windscreen reflections plus roof edges produce
        # gradient bands strong enough to win the projection outright.
        self.search_lower_fraction = float(search_lower_fraction)
        self.crops_too_small = 0
        self.rows_empty = 0
        self.cols_empty = 0

    def _load(self) -> None:
        """Nothing to load. Stated explicitly so the empty body is not read as a stub."""

    def _detect_in_crop(
        self, crop_bgr: np.ndarray, track: TrackResult
    ) -> Sequence[tuple[BBox, float]]:
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
        gradient = np.abs(np.diff(grey, axis=1))
        strokes = gradient > self.gradient_threshold

        # Rows against an absolute bar: a plate row is dense in stroke pixels in a way
        # bodywork is not, and the density it needs is a property of how many characters
        # fit across a plate, not of how the rest of this crop happens to look.
        row_counts = strokes.sum(axis=1)
        row_floor = max(2.0, self.row_min_fill * strokes.shape[1])
        band = _contiguous_band(row_counts, row_floor, EDGE_MIN_ROWS)
        if band is None:
            self.rows_empty += 1
            return ()
        row_start, row_end = band
        band_height = row_end - row_start

        # Columns, after merging the comb. The radius comes from the band height rather
        # than being a fixed pixel count, because the quantity it has to bridge is the
        # character pitch and that scales with the plate: an Indian single-row plate is
        # 4.17 times as wide as it is tall and carries ten characters, so the pitch is
        # about 0.42 of the plate height and the gap within it is smaller again. Half
        # the band height bridges the gap at every plate size, which is the point --
        # a fixed radius that works on an 80 px plate splits a 200 px one into
        # single characters.
        radius = max(1, band_height // 2)
        col_counts = _running_max(
            strokes[row_start:row_end].sum(axis=0), radius
        )
        col_peak = float(col_counts.max()) if col_counts.size else 0.0

        # Dilate to bridge, then erode by the same radius to undo the growth. This is a
        # morphological closing and the second half is not optional: dilation widens the
        # span by radius on each side, so a correctly located plate of aspect 4.17 comes
        # back at 4.17h + 2*(h/2) = 5.17h, and one whose row band under-measures the
        # height by a third arrives above 6.0 and is thrown out by the aspect gate in
        # geometry.py.
        #
        # Measured on the fixture in the class docstring, with the dilation but without
        # this erosion: 198 of 216 proposals rejected for shape, all 198 of them for
        # being too wide, leaving 18 -- of which 13 sit in the 4.2-6.0 bucket pressed up
        # against the ceiling. Shipped, the same run rejects 65 (64 too wide, 1 too tall)
        # and emits 151. So the erosion is worth 133 plates out of 216, and an earlier
        # version of this comment put the damage at "117 rejected ... and the 63 that
        # survived", which understated it by a factor of three and made the line look
        # like a refinement rather than the thing holding the detector up.
        # The dilation is a device for finding the extent, not part of the answer.
        span = _contiguous_band(
            col_counts, col_peak * self.col_quantile, EDGE_MIN_COLS + 2 * radius
        )
        if span is None:
            self.cols_empty += 1
            return ()
        col_start, col_end = span[0] + radius, span[1] - radius

        local: BBox = (
            int(col_start),
            int(row_start + y_offset),
            int(col_end + 1),  # +1: diff shortened the axis by one
            int(row_end + y_offset),
        )

        # Confidence from stroke density and shape agreement, not from a model. Named
        # honestly in the docstring above: this is a heuristic score on a 0..1 scale so
        # it can flow through the same ranking as a real detector's confidence, and it
        # is not a probability of anything.
        area = max(1, (row_end - row_start) * (col_end - col_start))
        density = float(strokes[row_start:row_end, col_start:col_end].sum()) / area
        shape_fit = _aspect_agreement(aspect_ratio(local))
        confidence = min(0.85, max(0.0, density * 1.6)) * shape_fit
        return ((local, round(confidence, 4)),)

    @property
    def model_name(self) -> str:
        return "edge-projection"

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    @property
    def ships(self) -> bool:
        """False, despite a clean licence. It carries no trained weights.

        The flag means "may appear in a published accuracy claim", not "may be
        distributed". A projection profile is a legitimate fallback for keeping a
        low-value camera online and is not a plate detector whose numbers belong in a
        submission.
        """
        return False

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "crops_too_small": self.crops_too_small,
                "rows_empty": self.rows_empty,
                "cols_empty": self.cols_empty,
                "gradient_threshold": self.gradient_threshold,
            }
        )
        return base


# ----------------------------------------------------------------------- scripted


class ScriptedPlateDetector(BasePlateDetector):
    """Fixed crop-local boxes per frame index. For tests that need an exact answer.

    Keyed on frame_index rather than track_id so a test can describe a whole frame in
    one line. A frame with no entry produces no plates, which is how the "vehicle
    found, plate not" path gets exercised deterministically.
    """

    def __init__(
        self,
        script: dict[int, Sequence[tuple[BBox, float]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._script = dict(script)

    def _load(self) -> None:
        """Nothing to load."""

    def _detect_in_crop(
        self, crop_bgr: np.ndarray, track: TrackResult
    ) -> Sequence[tuple[BBox, float]]:
        return self._script.get(track.frame_index, ())

    @property
    def model_name(self) -> str:
        return "scripted-plate"

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


def _contiguous_band(
    counts: np.ndarray, floor: float, min_extent: int
) -> Optional[tuple[int, int]]:
    """Longest run of consecutive indices whose count is at least `floor`.

    `floor` is an absolute count, and the caller decides what it means -- the row pass
    derives it from the region width, the column pass from the peak of the smoothed
    profile. Taking a threshold rather than computing a peak-relative one internally is
    the whole reason this function is usable for both: those two criteria behave very
    differently and burying either one here made one of the two call sites silently
    wrong.

    Longest run rather than "first qualifying index to last" because the latter spans
    from the first to the last, and on a vehicle with both a plate and a bright grille
    above it that single span covers both plus the bodywork between them -- one box,
    wrong aspect, rejected downstream, and the plate lost.
    """
    if counts.size == 0 or floor <= 0:
        return None
    above = counts >= floor

    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    if start is not None:
        if best is None or (len(above) - start) > (best[1] - best[0]):
            best = (start, len(above))

    if best is None or (best[1] - best[0]) < min_extent:
        return None
    return best


def _running_max(counts: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 1-D profile: each index becomes the max over +/- radius.

    Turns the comb-shaped column profile of a plate into a plateau. A plate's column
    density alternates between stroke edges and inter-character gaps, so the only way
    a contiguous-run test can find the plate's *extent* is if the gaps are filled
    first. Cheap enough to be irrelevant: the profile is one row of at most a few
    hundred values.

    The dilation grows the box by up to `radius` on each side, which is a real cost
    and an acceptable one -- an outward error keeps the whole plate inside the box,
    where an inward error crops characters off it, and the OCR stage recovers from
    padding far better than from a missing first character.
    """
    if radius <= 0 or counts.size == 0:
        return counts
    padded = np.pad(counts, radius, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return windows.max(axis=1)


def _aspect_agreement(aspect: float) -> float:
    """1.0 for a plate-shaped box, tapering off for anything else.

    Peaks across the whole 1.4-4.2 range rather than at a single value, because both
    ends of that range are real Indian plate formats -- see PLATE_ASPECT_MIN in
    ai/plate/geometry.py. A function peaked at 4.2 would score every motorcycle plate
    as half-confident.
    """
    if 1.4 <= aspect <= 4.2:
        return 1.0
    if aspect < 1.4:
        return max(0.2, aspect / 1.4)
    return max(0.2, 1.0 - (aspect - 4.2) / 4.0)


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


def _unit_hash(plate: str, frame_index: int) -> float:
    """Deterministic 0..1 from (plate, frame). Same approach as ai/detect/stub.py."""
    import hashlib

    digest = hashlib.sha256(f"plate-miss|{plate}|{frame_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
