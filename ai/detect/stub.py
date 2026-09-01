"""Detector backends that need no weights, no GPU and no cv2.

Three of them, for three different jobs. Keeping them distinct matters, because
the failure they each protect against is different.

**ScriptedDetector** -- returns exactly what it was told to, per frame. For unit
tests that assert an exact track count or an exact fused plate. A test whose
input is itself approximate cannot prove anything about the code under test.

**OracleDetector** -- reads ground truth from a SyntheticReplaySource, with
optional deterministic degradation. This is how every stage *below* detection gets
built and debugged: with boxes known to be correct, a wrong plate is unambiguously
a plate-stage bug. It also answers a question the benchmark needs -- "with a
perfect detector, what would the end-to-end rate be?" -- which separates detection
loss from OCR loss instead of reporting one number and guessing. Never ships;
ships is False and is_oracle is True so ai/metrics.py can refuse to publish it.

**MotionBlobDetector** -- background subtraction with numpy. Genuinely runnable on
real footage from a static camera, which describes most of the grid. Not a
vehicle detector: it finds things that moved. That distinction is stated in its
docstring, in its stats, and in the class name it reports, because a motion mask
labelled "detections" on a slide is a false claim.

None of the three is a substitute for RF-DETR on real footage, and none is
presented as one.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ai.contracts.frame import FrameEnvelope
from ai.contracts.stages import DetectorResult
from ai.detect.base import BaseDetector
from ai.detect.blobs import blobs_from_mask, suppress_overlaps

# --------------------------------------------------------------------- scripted


class ScriptedDetector(BaseDetector):
    """Replays a fixed table of detections. The detector for exact assertions.

    The table is keyed by emitted frame index, not by PTS: a unit test constructs
    both the frames and the expected detections, so the two cannot drift, and the
    frame index is the more readable key to write a fixture in.

    A frame index absent from the table yields no detections, which is how a test
    expresses "the vehicle left" or "this frame is empty".
    """

    def __init__(
        self,
        script: Mapping[int, Sequence[DetectorResult]],
        *,
        confidence_threshold: float = 0.0,
        allowed_classes: Optional[frozenset[str]] = None,
    ) -> None:
        # Threshold defaults to 0.0 rather than the usual 0.35: a test that writes
        # a detection at confidence 0.2 means to see it, and having the base class
        # silently filter it would make the test fail for a reason that is not the
        # thing under test.
        super().__init__(
            confidence_threshold=confidence_threshold,
            allowed_classes=allowed_classes,
        )
        self._script = {int(k): list(v) for k, v in script.items()}
        self._cursor = 0

    def _load(self) -> None:
        return None

    def _detect(self, frame_bgr: np.ndarray) -> Sequence[DetectorResult]:
        """Detections for the next frame in sequence.

        Sequential rather than content-addressed, because a scripted test feeds
        frames in order by construction. warmup() is disabled below so the
        cursor is not advanced by frames the test never accounted for.
        """
        detections = self._script.get(self._cursor, [])
        self._cursor += 1
        return detections

    def warmup(self, frame_bgr: Optional[np.ndarray] = None) -> None:
        """No-op. There is nothing to warm, and warming would eat script rows."""
        self.load()
        self._warmed = True

    def reset(self) -> None:
        self._cursor = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def model_version(self) -> str:
        return "test-fixture"

    @property
    def license_name(self) -> str:
        return "MIT"

    @property
    def ships(self) -> bool:
        return False


# ----------------------------------------------------------------------- oracle


@dataclass
class OracleDegradation:
    """Deterministic imperfection, so an oracle run can be made realistic.

    All of it is a pure function of (plate, frame index, seed) -- never of a live
    RNG -- so the same run degrades identically every time. An imperfection that
    varies between runs cannot be used to test recovery from it.
    """

    #: Fraction of vehicle-frames dropped entirely. Simulates a detector miss.
    miss_rate: float = 0.0
    #: Uniform box jitter in pixels, applied per edge.
    jitter_px: int = 0
    #: Extra boxes per frame that correspond to no vehicle.
    false_positives_per_frame: float = 0.0
    #: Confidence assigned to true detections.
    confidence: float = 0.92
    #: Confidence assigned to injected false positives.
    false_positive_confidence: float = 0.55
    seed: int = 1

    @property
    def is_perfect(self) -> bool:
        return (
            self.miss_rate <= 0.0
            and self.jitter_px <= 0
            and self.false_positives_per_frame <= 0.0
        )


class OracleDetector(BaseDetector):
    """Perfect (or deliberately imperfect) detections read from the generator.

    Requires the envelope, not just the frame, because it identifies the frame by
    PTS -- see the truth API note in ai/media/synthetic_source.py for why frame
    index is the wrong key. Calling detect() with a bare array raises rather than
    returning nothing: a diagnostic backend that silently reports zero detections
    would read as a broken pipeline and cost an hour.
    """

    warmup_frames = 0

    def __init__(
        self,
        source: Any,
        *,
        degradation: Optional[OracleDegradation] = None,
        confidence_threshold: float = 0.0,
        allowed_classes: Optional[frozenset[str]] = None,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            allowed_classes=allowed_classes,
        )
        if not hasattr(source, "truth_for_envelope"):
            raise TypeError(
                "OracleDetector needs a source exposing truth_for_envelope(); "
                f"{type(source).__name__} does not. Only the synthetic source "
                "carries ground truth."
            )
        self.source = source
        self.degradation = degradation or OracleDegradation()
        self.unresolved_frames = 0
        self.missed_on_purpose = 0
        self.false_positives_injected = 0

    def _load(self) -> None:
        return None

    def _detect(self, frame_bgr: np.ndarray) -> Sequence[DetectorResult]:
        raise TypeError(
            "OracleDetector cannot work from pixels -- it reads ground truth and "
            "needs the frame's identity. Call detect_envelope(envelope)."
        )

    def _detect_envelope(self, envelope: FrameEnvelope) -> Sequence[DetectorResult]:
        truth = self.source.truth_for_envelope(envelope)
        if truth is None:
            # No truth for this frame: either it came from another source, or a
            # fault suppressed or corrupted its PTS. Counted, not hidden.
            self.unresolved_frames += 1
            return []

        height, width = envelope.frame_bgr.shape[:2]
        degradation = self.degradation
        detections: list[DetectorResult] = []

        for vehicle in truth.vehicles:
            if degradation.miss_rate > 0.0 and _unit_hash(
                degradation.seed, "miss", vehicle.plate, truth.frame_index
            ) < degradation.miss_rate:
                self.missed_on_purpose += 1
                continue

            bbox = vehicle.vehicle_bbox_xyxy
            if degradation.jitter_px > 0:
                bbox = _jitter_box(
                    bbox,
                    degradation.jitter_px,
                    width,
                    height,
                    degradation.seed,
                    vehicle.plate,
                    truth.frame_index,
                )
                if bbox is None:
                    continue

            detections.append(
                DetectorResult(
                    bbox_xyxy=bbox,
                    class_name=vehicle.vehicle_type,
                    confidence=degradation.confidence,
                )
            )

        detections.extend(self._inject_false_positives(truth, width, height))
        return detections

    def _inject_false_positives(
        self, truth: Any, width: int, height: int
    ) -> list[DetectorResult]:
        rate = self.degradation.false_positives_per_frame
        if rate <= 0.0:
            return []

        # Fractional rates are honoured probabilistically but deterministically:
        # 0.3 means a false positive on 30% of frames, always the same 30%.
        whole = int(rate)
        draw = _unit_hash(self.degradation.seed, "fp-count", "", truth.frame_index)
        count = whole + (1 if draw < (rate - whole) else 0)
        if count <= 0:
            return []

        injected: list[DetectorResult] = []
        for n in range(count):
            key = f"fp{n}"
            box_w = 40 + int(120 * _unit_hash(self.degradation.seed, key + "w", "", truth.frame_index))
            box_h = 30 + int(90 * _unit_hash(self.degradation.seed, key + "h", "", truth.frame_index))
            x1 = int((width - box_w) * _unit_hash(self.degradation.seed, key + "x", "", truth.frame_index))
            y1 = int((height - box_h) * _unit_hash(self.degradation.seed, key + "y", "", truth.frame_index))
            injected.append(
                DetectorResult(
                    bbox_xyxy=(x1, y1, x1 + box_w, y1 + box_h),
                    class_name="car",
                    confidence=self.degradation.false_positive_confidence,
                )
            )
        self.false_positives_injected += len(injected)
        return injected

    @property
    def model_name(self) -> str:
        return "oracle" if self.degradation.is_perfect else "oracle-degraded"

    @property
    def model_version(self) -> str:
        return f"seed{self.degradation.seed}"

    @property
    def license_name(self) -> str:
        return "MIT"

    @property
    def is_oracle(self) -> bool:
        """Reads the answer. Any metric derived from it is a ceiling, not a result."""
        return True

    @property
    def ships(self) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "is_oracle": True,
                "unresolved_frames": self.unresolved_frames,
                "missed_on_purpose": self.missed_on_purpose,
                "false_positives_injected": self.false_positives_injected,
            }
        )
        return base


# ------------------------------------------------------------------ motion blobs

# Per-channel absolute difference from the background model above which a pixel is
# called foreground.
#
# Measured on 100 emitted synthetic frames, seed 42, alpha 0.08:
#     background-only pixels, diff max per frame   p50 34   p95 52   max 57
#     vehicle pixels, fraction exceeding 30        p5 0.82  p50 0.99
#
# 45 sits above the background band's bulk and well below the vehicle band. The
# background peaks come from motion trails the exponential model has not yet
# absorbed; they survive the threshold but are thin and sparse, so the area and
# fill filters below remove them. Thresholding alone is not sufficient here, and
# that is why the geometry filters are not optional.
MOTION_DIFF_THRESHOLD = 45

# Background learning rate. 0.08 absorbs a lighting change over roughly a dozen
# emitted frames -- about 1.5 s at the 100 ms sampling interval. Faster and a
# slow-moving vehicle dissolves into the background; slower and a cloud passing
# over the scene reads as traffic for half a minute.
MOTION_BACKGROUND_ALPHA = 0.08

# Frames used to build the model before any detection is reported. During these
# the detector returns nothing, which is honest: it has no background yet, so it
# cannot say what moved.
MOTION_WARMUP_FRAMES = 6

MOTION_MIN_AREA = 900        # 30x30 px. Below this a plate cannot be read anyway.
MOTION_MIN_WIDTH = 24
MOTION_MIN_HEIGHT = 20
MOTION_MIN_FILL = 0.35       # a vehicle fills its box; a motion trail does not


class MotionBlobDetector(BaseDetector):
    """Foreground blobs from an exponential background model. numpy only.

    **What this is.** A real, dependency-free fallback for a static camera. In a
    grid of 80,000 cameras, most are low-value and permanently pointed at the same
    scene; running a GPU model on all of them is neither affordable nor necessary
    to know that something moved and roughly where.

    **What this is not.** A vehicle detector. It reports regions that differ from
    the background. A pedestrian, a swaying branch and a bus all qualify. The
    class it assigns comes from box geometry, not from recognition, so it is
    reported as "other" unless the geometry is unambiguous -- and per-class
    accuracy must never be quoted from this backend.

    **Measured on the synthetic road scenes** (seed 42, 100 emitted frames, 281
    ground-truth vehicle boxes, IoU 0.3 matching):

        recall     0.883      precision  0.947
        latency    41 ms median, 53 ms p95 on 1280x720, pure numpy, one core
        classes    other 242, motorcycle 19, truck 1

    The 33 misses break down as: 20 vehicles merged into a neighbour's box, 6
    below the minimum width as they crossed the frame edge, 2 during the
    background warm-up, 5 unexplained. Merging dominates, which is the honest
    headline: this backend loses vehicles in traffic, exactly where it matters.

    41 ms per frame is roughly 40% of one core at the 10 fps sampling rate, so it
    is affordable per camera and not affordable across thirty of them on one box.

    **Where it fails, stated rather than discovered.**
      - Two vehicles that overlap in the frame merge into one box. This is the
        largest single loss above, and no threshold change fixes it.
      - A camera that pans invalidates the model completely. reset() is wired to
        session changes for exactly this reason.
      - A vehicle stopped at a signal is absorbed into the background within a
        couple of seconds and disappears.
      - Headlight glare at night produces large bright blobs with no vehicle in
        them.
    """

    warmup_frames = 0  # the background model is the warm-up; see MOTION_WARMUP_FRAMES

    def __init__(
        self,
        *,
        diff_threshold: int = MOTION_DIFF_THRESHOLD,
        alpha: float = MOTION_BACKGROUND_ALPHA,
        background_frames: int = MOTION_WARMUP_FRAMES,
        min_area: int = MOTION_MIN_AREA,
        min_width: int = MOTION_MIN_WIDTH,
        min_height: int = MOTION_MIN_HEIGHT,
        min_fill_ratio: float = MOTION_MIN_FILL,
        max_detections: int = 32,
        confidence_threshold: float = 0.0,
        allowed_classes: Optional[frozenset[str]] = None,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            allowed_classes=allowed_classes,
        )
        self.diff_threshold = diff_threshold
        self.alpha = alpha
        self.background_frames = background_frames
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.min_fill_ratio = min_fill_ratio
        self.max_detections = max_detections

        self._background: Optional[np.ndarray] = None
        self._frames_modelled = 0
        self.frames_suppressed_warmup = 0
        self.frames_truncated = 0

    def _load(self) -> None:
        return None

    def reset(self) -> None:
        """Discard the background model. Call on every session change.

        A new session means the source timeline broke -- a reconnect, a loop
        boundary, a re-aimed camera. In all three cases the old background
        describes a scene that is no longer there, and every pixel of the new one
        would read as foreground.
        """
        self._background = None
        self._frames_modelled = 0

    def _detect(self, frame_bgr: np.ndarray) -> Sequence[DetectorResult]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR, got shape {frame.shape}")

        as_float = frame.astype(np.float32)

        if self._background is None:
            self._background = as_float.copy()
            self._frames_modelled = 1
            self.frames_suppressed_warmup += 1
            return []

        difference = np.abs(as_float - self._background).max(axis=2)

        # Update before the early return, so the warm-up frames actually build
        # the model rather than being thrown away.
        self._background *= 1.0 - self.alpha
        self._background += self.alpha * as_float
        self._frames_modelled += 1

        if self._frames_modelled <= self.background_frames:
            self.frames_suppressed_warmup += 1
            return []

        mask = difference >= self.diff_threshold
        blobs = blobs_from_mask(
            mask,
            min_area=self.min_area,
            min_width=self.min_width,
            min_height=self.min_height,
            min_fill_ratio=self.min_fill_ratio,
        )
        if not blobs:
            return []

        boxes = [(b.x1, b.y1, b.x2, b.y2) for b in blobs]
        # Confidence is the mean foreground strength inside the box, squashed into
        # 0..1. It is a *motion* score and not a class probability, which is why it
        # must never be multiplied into a plate confidence -- Contracts section 8
        # forbids combining uncalibrated numbers, and this one is not calibrated
        # against anything.
        scores = [
            min(1.0, float(difference[b.y1 : b.y2, b.x1 : b.x2].mean()) / 128.0)
            for b in blobs
        ]

        kept = suppress_overlaps(boxes, scores)
        if len(kept) > self.max_detections:
            self.frames_truncated += 1
            kept = kept[: self.max_detections]

        return [
            DetectorResult(
                bbox_xyxy=boxes[i],
                class_name=_class_from_geometry(blobs[i].width, blobs[i].height),
                confidence=round(scores[i], 4),
            )
            for i in kept
        ]

    @property
    def model_name(self) -> str:
        return "motion-blobs"

    @property
    def model_version(self) -> str:
        return f"ema{self.alpha}-t{self.diff_threshold}"

    @property
    def license_name(self) -> str:
        return "MIT"

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "classifies_by_appearance": False,
                "frames_modelled": self._frames_modelled,
                "frames_suppressed_warmup": self.frames_suppressed_warmup,
                "frames_truncated": self.frames_truncated,
            }
        )
        return base


def _class_from_geometry(width: int, height: int) -> str:
    """Geometry-based guess. Returns "other" whenever the shape is ambiguous.

    This is not classification and the thresholds are not learned. It exists so a
    motion backend can populate a required schema field without lying: "other"
    means "a vehicle we could not classify", which is exactly the situation.
    """
    if height <= 0:
        return "other"
    aspect = width / float(height)

    if aspect >= 2.6 and width >= 160:
        return "truck"          # long and large: a lorry or a bus body
    if aspect <= 0.85 and width <= 60:
        return "motorcycle"     # tall, narrow, small
    return "other"


# ---------------------------------------------------------------------- helpers


def _unit_hash(seed: int, key: str, plate: str, frame_index: int) -> float:
    """A stable pseudo-random float in [0, 1) from the inputs.

    A hash rather than an RNG because the value must not depend on how many
    frames have been processed. An oracle degraded by a live RNG produces a
    different run every time it is invoked, and a flaky fixture is worse than no
    fixture.
    """
    material = f"{seed}|{key}|{plate}|{frame_index}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _jitter_box(
    bbox: tuple[int, int, int, int],
    amount: int,
    width: int,
    height: int,
    seed: int,
    plate: str,
    frame_index: int,
) -> Optional[tuple[int, int, int, int]]:
    """Perturb each edge independently, clip to frame, drop if it collapses.

    Independent per edge, not a whole-box translation: a real detector's error is
    a slightly wrong extent, not a perfectly-shaped box in the wrong place, and
    the difference matters to the plate crop that comes next.
    """
    x1, y1, x2, y2 = bbox
    offsets = [
        int(round((_unit_hash(seed, f"j{edge}", plate, frame_index) * 2 - 1) * amount))
        for edge in range(4)
    ]
    nx1 = max(0, min(width - 1, x1 + offsets[0]))
    ny1 = max(0, min(height - 1, y1 + offsets[1]))
    nx2 = max(0, min(width, x2 + offsets[2]))
    ny2 = max(0, min(height, y2 + offsets[3]))

    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None
    return (nx1, ny1, nx2, ny2)
