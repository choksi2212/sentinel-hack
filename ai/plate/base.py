"""What every plate detector must do, and the shared work none of them should repeat.

The stage takes a frame plus the tracked vehicle boxes on it, and returns at most one
PlateCandidate per vehicle in full-frame coordinates. One, not several: downstream
consumes a plate per vehicle per frame, and a stage that returns three boxes for one
car has moved the choice of which to believe into the OCR stage, which has less
information to make it with than this one does.

Shared here rather than per backend, because each of these has exactly one right
answer and three chances to get it wrong:

  * cropping the vehicle and remembering where the crop came from
  * mapping every box back to full-frame coordinates
  * rejecting boxes that cannot be plates on shape alone
  * applying the vertical-position prior
  * picking the best remaining box per vehicle
  * counting what was rejected and why

The last one is not bookkeeping. "The plate stage found nothing" has at least four
distinct causes -- no box proposed, boxes proposed but all below threshold, boxes
above threshold but rejected on shape, vehicle crop empty -- and they call for four
different fixes. A stage that reports only a count of successes leaves that
diagnosis to guesswork on the day it matters.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from ai.contracts.stages import BBox, PlateCandidate, TrackResult
from ai.plate.geometry import (
    CROP_PAD_FRACTION,
    clip_to_crop,
    crop_vehicle,
    map_to_frame,
    plausible_plate_box,
    region_prior,
)

# A plate box below this is kept, not dropped.
#
# It is tempting to filter small plates here, and wrong: the plate stage does not know
# what the OCR stage can read, and Contracts section 8 requires accuracy reported per
# width bucket including the <30 px bucket. If this stage silently dropped everything
# under 30 px then the <30 px bucket would report 0 attempts and 0 errors, which reads
# as "no problem here" rather than "we never tried". The quality gate and the top-K
# crop ranking decide what is worth OCR time; this stage reports what is there.
DEFAULT_PLATE_CONFIDENCE_THRESHOLD = 0.25


class PlateDetector(Protocol):
    """What the pipeline requires. Any backend satisfying this is swappable."""

    def detect_plates(
        self, frame_bgr: np.ndarray, tracks: Sequence[TrackResult]
    ) -> dict[int, PlateCandidate]: ...

    def load(self) -> None: ...

    def close(self) -> None: ...

    @property
    def model_name(self) -> str: ...


class BasePlateDetector(ABC):
    """Crop, detect, map home, filter, rank. Subclasses supply only the middle step.

    Keyed by track_id in the returned dict rather than returned as a list, because
    every consumer wants "the plate for this vehicle" and a list forces each of them
    to build the same index. The key is the track_id alone rather than the full
    TrackKey since a single call cannot span sessions -- one frame belongs to one
    session by construction.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
        pad_fraction: float = CROP_PAD_FRACTION,
        apply_region_prior: bool = True,
        min_vehicle_width_px: int = 0,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.pad_fraction = float(pad_fraction)
        self.apply_region_prior = bool(apply_region_prior)
        # 0 by default: this stage does not decide what is worth trying. A caller that
        # wants to save GPU time on 20 px vehicles sets it in config, where the
        # decision is visible in the run record.
        self.min_vehicle_width_px = int(min_vehicle_width_px)

        self._loaded = False

        self.vehicles_seen = 0
        self.vehicles_skipped_small = 0
        self.crops_empty = 0
        self.boxes_proposed = 0
        self.boxes_below_threshold = 0
        self.boxes_rejected_shape = 0
        self.boxes_rejected_clip = 0
        self.plates_emitted = 0
        self.vehicles_without_plate = 0

    # ---------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def close(self) -> None:
        if not self._loaded:
            return
        self._close()
        self._loaded = False

    def __enter__(self) -> "BasePlateDetector":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -------------------------------------------------------------------- stage

    def detect_plates_envelope(
        self, envelope: Any, tracks: Sequence[TrackResult]
    ) -> dict[int, PlateCandidate]:
        """detect_plates, for a backend that needs to know *which* frame this is.

        Every real model needs only the pixels, so detect_plates is the primary entry
        point and this is a thin wrapper. The oracle backend is the exception: it
        reads ground truth, which is keyed on frame identity, and pixels alone cannot
        tell it which frame it is looking at. Mirrors
        ai/detect/base.py:detect_envelope for the same reason.
        """
        self._begin_frame(envelope)
        return self.detect_plates(envelope.frame_bgr, tracks)

    def _begin_frame(self, envelope: Any) -> None:
        """Hook for backends that need frame identity. Default is to ignore it."""

    def detect_plates(
        self, frame_bgr: np.ndarray, tracks: Sequence[TrackResult]
    ) -> dict[int, PlateCandidate]:
        """One plate per track, in full-frame coordinates. Absent means none found.

        A missing key is a first-class answer and the calling code must treat it as
        one. Contracts section 6 is explicit that plate: null is correct for a vehicle
        whose plate was never legible, and that inventing a string is the worst
        outcome the pipeline can produce -- worse than silence, because a wrong plate
        in a police search result is acted on.
        """
        if not self._loaded:
            raise RuntimeError(
                f"{type(self).__name__}.detect_plates called before load(). The "
                "pipeline calls load() once at startup so a missing checkpoint fails "
                "immediately rather than on the first frame with a vehicle in it."
            )

        found: dict[int, PlateCandidate] = {}
        for track in tracks:
            self.vehicles_seen += 1
            candidate = self._detect_one(frame_bgr, track)
            if candidate is None:
                self.vehicles_without_plate += 1
                continue
            found[track.track_id] = candidate
            self.plates_emitted += 1
        return found

    def _detect_one(
        self, frame_bgr: np.ndarray, track: TrackResult
    ) -> Optional[PlateCandidate]:
        vehicle_bbox = track.bbox_xyxy
        if (
            self.min_vehicle_width_px
            and (vehicle_bbox[2] - vehicle_bbox[0]) < self.min_vehicle_width_px
        ):
            self.vehicles_skipped_small += 1
            return None

        crop, origin = crop_vehicle(
            frame_bgr, vehicle_bbox, pad_fraction=self.pad_fraction
        )
        if crop.size == 0:
            self.crops_empty += 1
            return None

        proposals = self._detect_in_crop(crop, track)
        self.boxes_proposed += len(proposals)
        if not proposals:
            return None

        best: Optional[tuple[float, BBox]] = None
        for local_bbox, confidence in proposals:
            clipped = clip_to_crop(local_bbox, crop.shape)
            if clipped is None:
                self.boxes_rejected_clip += 1
                continue

            frame_bbox = map_to_frame(clipped, origin, frame_bgr.shape)
            if not plausible_plate_box(frame_bbox):
                self.boxes_rejected_shape += 1
                continue

            score = float(confidence)
            if self.apply_region_prior:
                score *= region_prior(frame_bbox, vehicle_bbox)
            if score < self.confidence_threshold:
                self.boxes_below_threshold += 1
                continue

            if best is None or score > best[0]:
                best = (score, frame_bbox)

        if best is None:
            return None
        return PlateCandidate(
            plate_bbox_xyxy=best[1],
            detector_confidence=round(min(1.0, best[0]), 4),
        )

    # ------------------------------------------------------------- subclass API

    @abstractmethod
    def _load(self) -> None:
        """Acquire weights. Heavy imports belong here, not at module level."""

    @abstractmethod
    def _detect_in_crop(
        self, crop_bgr: np.ndarray, track: TrackResult
    ) -> Sequence[tuple[BBox, float]]:
        """Plate boxes in CROP-LOCAL coordinates, with confidences. Unfiltered.

        Crop-local is the contract for this method and full-frame is the contract for
        everything above it. The conversion happens in _detect_one and must not be
        anticipated here -- a backend that helpfully returns frame coordinates gets
        the crop origin added a second time, which puts the plate off by exactly the
        vehicle's position in the frame. That grows with distance from the top-left
        corner, so it looks correct on a vehicle near the origin.
        """

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
        """SPDX identifier. Contracts section 11 requires one per model in use."""
        return "unknown"

    @property
    def ships(self) -> bool:
        """False for a backend that may be benchmarked but never submitted."""
        return self.license_name.startswith("Apache") or self.license_name == "MIT"

    # ----------------------------------------------------------------- metadata

    def stats(self) -> dict[str, Any]:
        """Counters, including every reason a plate was not produced.

        recall_proxy is named as a proxy and not as recall. It is plates emitted over
        vehicles seen, which is only recall if every vehicle in view actually has a
        legible plate -- and on real footage most do not, because they are facing away
        or too far off. The real number needs ground truth and comes from the
        synthetic corpus or a labelled clip, not from here.
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "license": self.license_name,
            "ships": self.ships,
            "confidence_threshold": self.confidence_threshold,
            "vehicles_seen": self.vehicles_seen,
            "vehicles_skipped_small": self.vehicles_skipped_small,
            "crops_empty": self.crops_empty,
            "boxes_proposed": self.boxes_proposed,
            "boxes_below_threshold": self.boxes_below_threshold,
            "boxes_rejected_shape": self.boxes_rejected_shape,
            "boxes_rejected_clip": self.boxes_rejected_clip,
            "plates_emitted": self.plates_emitted,
            "vehicles_without_plate": self.vehicles_without_plate,
            "recall_proxy": round(
                self.plates_emitted / self.vehicles_seen, 4
            ) if self.vehicles_seen else 0.0,
        }
