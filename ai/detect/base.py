"""The vehicle detector interface. Owner's manual section 5.1.

Every detector behind this Protocol is interchangeable, and the pipeline never
learns which one it has. That matters for two reasons.

**Licensing.** RF-DETR is Apache-2.0 and ships. Ultralytics YOLO is AGPL-3.0 and
does not -- it may be run for a benchmark comparison and must never be in the
submitted path. A swappable backend is what makes that distinction enforceable in
config rather than dependent on remembering. Contracts section 11.

**The 12 GB constraint.** The development machine has an RTX 5070 Ti with 12 GB,
not an A100. Model size is a configuration decision made against measured VRAM,
and switching Nano to Small has to be a one-line change or it will not get tried.

Backends resolve their heavy imports inside load(), never at module import. The
contract tests, the fusion tests and the whole synthetic path must keep running
on a machine with numpy and nothing else.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from ai.contracts.enums import VEHICLE_TYPES
from ai.contracts.frame import FrameEnvelope
from ai.contracts.stages import DetectorResult

# The classes we keep. Exactly the VehicleType literal in ai/contracts/enums.py,
# imported rather than retyped -- a second copy of this list is a second thing to
# forget to update, and a class_name outside the contract is a 422 at ingest.
#
# "other" is in here on purpose. It is the honest answer for a vehicle a backend
# genuinely could not classify, and the motion fallback in ai/detect/stub.py
# reports it for almost every box. Excluding it would silently discard those
# detections and read as a detector that finds nothing.
#
# What gets dropped is a label outside the contract entirely -- a pedestrian, a
# traffic light, a potted plant. That is map_class_name's job, and it returns None
# rather than folding the label into "other": a detector firing on a pedestrian is
# not a vehicle sighting, and recording it as one inflates the denominator of the
# primary metric.
VEHICLE_CLASSES: frozenset[str] = frozenset(VEHICLE_TYPES)

# The subset that means "we identified the vehicle". Used by reporting, not by
# filtering: a run where 90% of detections are "other" has working detection and
# no classification, and those are different results that must not average.
CLASSIFIED_VEHICLE_CLASSES: frozenset[str] = VEHICLE_CLASSES - {"other"}

# COCO names -> our vocabulary. Most Apache-2.0 detectors are COCO-pretrained and
# COCO has no auto-rickshaw class, which is why a Gujarat deployment needs
# fine-tuning and why that gap is stated rather than hidden behind a mapping.
COCO_TO_VEHICLE_CLASS: dict[str, str] = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "motorcycle",
}

# Detections below this are dropped before anything downstream sees them.
#
# 0.35 is a reasonable *standalone* detector threshold and the wrong thing to run in
# front of a two-stage tracker. ByteTrack's low band starts at 0.1, so this filter
# removes the entire [0.1, 0.35) range that its second association stage exists to
# consume, and the tracker becomes single-threshold without saying so. When a config
# sets a tracker, its detector threshold should be at or below the tracker's
# low_threshold; ai/track/base.py:check_detector_threshold states the invariant and
# records what the misconfiguration costs, and scripts/validate_config.py enforces it.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35


class VehicleDetector(Protocol):
    """Frame in, boxes out. No identity, no session, no state."""

    def load(self) -> None: ...
    def detect(self, frame_bgr: np.ndarray) -> list[DetectorResult]: ...
    def detect_envelope(self, envelope: FrameEnvelope) -> list[DetectorResult]: ...
    def close(self) -> None: ...
    @property
    def model_name(self) -> str: ...
    @property
    def model_version(self) -> str: ...


class BaseDetector(ABC):
    """Shared bookkeeping: lazy load, warm-up, timing, class filtering.

    Warm-up is not a nicety. The first inference on a CUDA device includes kernel
    compilation and allocator setup and can take twenty times the steady-state
    latency. Reporting it inside a mean makes a 25 ms model look like a 60 ms one,
    and the number gets copied onto a slide. Contracts section 7 discards the
    first two frames of every measurement; this is where that starts.
    """

    #: Frames to run and throw away before timing means anything.
    warmup_frames = 2

    def __init__(
        self,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        allowed_classes: Optional[frozenset[str]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes or VEHICLE_CLASSES

        self._loaded = False
        self._warmed = False
        self.frames_seen = 0
        self.detections_emitted = 0
        self.detections_dropped_class = 0
        self.detections_dropped_confidence = 0
        self._latencies_ms: list[float] = []

    # ---------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def warmup(self, frame_bgr: Optional[np.ndarray] = None) -> None:
        """Run and discard a few inferences so later timings are steady-state."""
        self.load()
        if self._warmed:
            return

        sample = (
            frame_bgr
            if frame_bgr is not None
            else np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        for _ in range(self.warmup_frames):
            self._detect(sample)
        self._warmed = True
        self._latencies_ms.clear()

    def detect(self, frame_bgr: np.ndarray) -> list[DetectorResult]:
        """Detect from pixels alone. What a real backend implements."""
        return self._instrumented(lambda: self._detect(frame_bgr))

    def detect_envelope(self, envelope: "FrameEnvelope") -> list[DetectorResult]:
        """Detect from a frame envelope. What the pipeline always calls.

        Real backends ignore everything but the pixels, and the default below
        forwards to detect(). The diagnostic oracle in ai/detect/stub.py needs the
        frame's identity and overrides _detect_envelope instead.

        Two entry points rather than widening the Protocol with optional pts_ms
        and frame_index arguments that every real detector would have to accept
        and ignore. A detector that needs to know which frame it is looking at is
        by definition not doing detection, and the type system should say so.
        """
        return self._instrumented(lambda: self._detect_envelope(envelope))

    def close(self) -> None:
        if self._loaded:
            self._close()
        self._loaded = False
        self._warmed = False

    def _instrumented(
        self, run: "Callable[[], Sequence[DetectorResult]]"
    ) -> list[DetectorResult]:
        """Time it, count it, filter it. Shared by both entry points."""
        self.load()

        started = time.perf_counter()
        raw = run()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.frames_seen += 1
        if self._warmed:
            self._latencies_ms.append(elapsed_ms)

        kept = self._filter(raw)
        self.detections_emitted += len(kept)
        return kept

    # ------------------------------------------------------------- subclass API

    @abstractmethod
    def _load(self) -> None:
        """Acquire weights. Heavy imports belong here, not at module level."""

    @abstractmethod
    def _detect(self, frame_bgr: np.ndarray) -> Sequence[DetectorResult]:
        """Raw detections, unfiltered. Confidence and class filtering is shared."""

    def _detect_envelope(self, envelope: "FrameEnvelope") -> Sequence[DetectorResult]:
        """Default: ignore everything but the pixels. Right for every real model."""
        return self._detect(envelope.frame_bgr)

    def _close(self) -> None:
        """Release weights. Default is nothing, which is right for most backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        ...

    @property
    def license_name(self) -> str:
        """SPDX identifier. Contracts section 11 requires one per model in use.

        Abstract in spirit: a backend that cannot state its licence has no
        business being in a submitted pipeline.
        """
        return "unknown"

    @property
    def ships(self) -> bool:
        """False for a backend that may be benchmarked but never submitted.

        The AGPL-3.0 question, answered in code. A run whose detector reports
        False here must not appear in a submission claim -- see
        ModelProvenance.is_citeable in ai/contracts/event.py.
        """
        return self.license_name.startswith("Apache") or self.license_name == "MIT"

    # ----------------------------------------------------------------- internals

    def _filter(self, raw: Sequence[DetectorResult]) -> list[DetectorResult]:
        kept: list[DetectorResult] = []
        for detection in raw:
            if detection.confidence < self.confidence_threshold:
                self.detections_dropped_confidence += 1
                continue
            if detection.class_name not in self.allowed_classes:
                self.detections_dropped_class += 1
                continue
            kept.append(detection)
        return kept

    def stats(self) -> dict[str, Any]:
        latencies = sorted(self._latencies_ms)
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "license": self.license_name,
            "ships": self.ships,
            "frames_seen": self.frames_seen,
            "detections_emitted": self.detections_emitted,
            "dropped_low_confidence": self.detections_dropped_confidence,
            "dropped_wrong_class": self.detections_dropped_class,
            "latency_ms": _latency_summary(latencies),
        }

    def __enter__(self) -> "BaseDetector":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _latency_summary(sorted_ms: list[float]) -> Optional[dict[str, float]]:
    """Median and p95, never a bare mean.

    A mean hides the tail, and the tail is what an operator experiences. Contracts
    section 7 asks for p95 for exactly this reason.
    """
    if not sorted_ms:
        return None
    count = len(sorted_ms)
    return {
        "count": count,
        "median": round(sorted_ms[count // 2], 3),
        "p95": round(sorted_ms[min(count - 1, int(count * 0.95))], 3),
        "max": round(sorted_ms[-1], 3),
    }


def map_class_name(raw_name: str) -> Optional[str]:
    """Map a model's class label into our vocabulary, or None to drop it.

    Returns None rather than "other" for unrecognised labels. "other" exists in
    the schema for a vehicle we genuinely could not classify, not for a traffic
    light the detector was confident about.
    """
    key = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in VEHICLE_CLASSES:
        return key
    return COCO_TO_VEHICLE_CLASS.get(key)


def is_shippable_class(class_name: str) -> bool:
    """True if this label may appear in a submitted event. Contracts section 1.3."""
    return class_name in VEHICLE_CLASSES
