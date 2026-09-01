"""Per-TrackKey evidence buffers.

Two jobs:

1. Keep only the top K plate crops per track, so OCR runs 3-5 times per vehicle
   instead of once per frame. This is the single largest avoidable cost in the
   pipeline -- at 10 fps a vehicle in frame for three seconds offers 30 crops,
   and the 25 worst ones contribute nothing but latency.

2. Hold nothing across a session boundary. Contracts section 1.2 requires
   flushing evidence buffers when a session ends, because a buffer keyed on the
   wrong thing is where the track-merge bug reappears.

Note what "flush" means here: finalize and emit what has been observed, then
drop the state. A vehicle seen for two seconds before a reconnect really did
pass the camera, and discarding that evidence loses real information. What must
not happen is carrying the buffer into the new session and appending to it.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from ai.contracts.ids import TrackKey
from ai.contracts.stages import PlateCandidate, TrackResult

# Contracts section 4.1: keep the top K = 3..5 crops per track.
DEFAULT_TOP_K = 4

# A track with no update for this long on the SOURCE timeline is finished and
# its evidence is ready to fuse. Roughly 8 sampled frames at the 100 ms
# interval -- long enough to survive a brief occlusion, short enough that a
# vehicle's event lands while it is still relevant to an operator.
DEFAULT_TRACK_IDLE_MS = 800

# Hard cap so a vehicle stopped at a signal still produces an event instead of
# accumulating forever. Its buffer is finalized and a fresh one starts.
DEFAULT_MAX_TRACK_DURATION_MS = 15_000


@dataclass
class TrackCrop:
    """One plate crop kept as evidence, with everything needed to build an event.

    The vehicle fields travel with the crop because by the time fusion runs the
    track is gone, and reconstructing which vehicle a crop belonged to after
    the fact is exactly the kind of bookkeeping that goes wrong quietly.
    """

    quality: float
    crop_bgr: np.ndarray
    candidate: PlateCandidate
    frame_index: int
    pts_ms: int
    observed_at: str
    vehicle_bbox_xyxy: tuple[int, int, int, int]
    vehicle_class: str
    vehicle_confidence: float
    frame_quality: float = 0.0

    @property
    def plate_width_px(self) -> int:
        return self.candidate.plate_width_px


@dataclass
class CropBuffer:
    """Top-K crops for one TrackKey, plus the track's own summary."""

    track_key: TrackKey
    top_k: int = DEFAULT_TOP_K
    crops: list[TrackCrop] = field(default_factory=list)

    first_pts_ms: Optional[int] = None
    last_pts_ms: Optional[int] = None
    first_observed_at: Optional[str] = None
    last_observed_at: Optional[str] = None
    frames_seen: int = 0
    crops_offered: int = 0
    crops_rejected: int = 0

    # Best vehicle observation for the track, used when no plate is ever read.
    # Without it an unreadable vehicle could not be reported at all, and "a
    # vehicle passed and could not be identified" is information worth keeping.
    best_vehicle_confidence: float = 0.0
    best_vehicle_bbox: Optional[tuple[int, int, int, int]] = None
    vehicle_class: str = "other"
    best_frame_quality: float = 0.0

    def note_track(self, track: TrackResult, observed_at: str) -> None:
        """Record that the track was seen, whether or not a plate was found."""
        self.frames_seen += 1
        if self.first_pts_ms is None:
            self.first_pts_ms = track.pts_ms
            self.first_observed_at = observed_at
        self.last_pts_ms = track.pts_ms
        self.last_observed_at = observed_at

        # Largest box wins the tie rather than highest confidence: the biggest
        # view of the vehicle is the most useful snapshot, and detector
        # confidence on a near vehicle is saturated anyway.
        area = max(0, track.width) * max(0, track.height)
        best_area = 0
        if self.best_vehicle_bbox is not None:
            bx1, by1, bx2, by2 = self.best_vehicle_bbox
            best_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        if self.best_vehicle_bbox is None or area > best_area:
            self.best_vehicle_bbox = track.bbox_xyxy
            self.best_vehicle_confidence = track.confidence
            self.vehicle_class = track.class_name

    def offer(self, crop: TrackCrop) -> bool:
        """Add a crop if it beats the worst one held. Returns True if kept."""
        self.crops_offered += 1
        self.best_frame_quality = max(self.best_frame_quality, crop.quality)

        if len(self.crops) < self.top_k:
            self.crops.append(crop)
            self.crops.sort(key=lambda c: c.quality, reverse=True)
            return True

        worst = self.crops[-1]
        if crop.quality > worst.quality:
            self.crops[-1] = crop
            self.crops.sort(key=lambda c: c.quality, reverse=True)
            self.crops_rejected += 1
            return True

        self.crops_rejected += 1
        return False

    @property
    def duration_ms(self) -> int:
        if self.first_pts_ms is None or self.last_pts_ms is None:
            return 0
        return max(0, self.last_pts_ms - self.first_pts_ms)

    @property
    def has_plate_evidence(self) -> bool:
        return bool(self.crops)

    def stats(self) -> dict[str, Any]:
        return {
            "track_key": str(self.track_key),
            "frames_seen": self.frames_seen,
            "crops_offered": self.crops_offered,
            "crops_kept": len(self.crops),
            "duration_ms": self.duration_ms,
            "best_crop_quality": round(self.best_frame_quality, 4),
        }


class EvidenceAccumulator:
    """All open CropBuffers, keyed by TrackKey.

    The key is the whole point. A dict keyed on (camera_id, track_id) merges
    the car that left cam04 before a reconnect with a different car that
    entered after it, and produces a journey showing a vehicle crossing
    Ahmedabad in four seconds. TrackKey carries the session, so the correct key
    is also the easy key.
    """

    def __init__(
        self,
        *,
        top_k: int = DEFAULT_TOP_K,
        track_idle_ms: int = DEFAULT_TRACK_IDLE_MS,
        max_track_duration_ms: int = DEFAULT_MAX_TRACK_DURATION_MS,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.track_idle_ms = track_idle_ms
        self.max_track_duration_ms = max_track_duration_ms
        self._buffers: dict[TrackKey, CropBuffer] = {}

    def __len__(self) -> int:
        return len(self._buffers)

    def __contains__(self, key: object) -> bool:
        return key in self._buffers

    def __iter__(self) -> Iterator[TrackKey]:
        return iter(self._buffers)

    def buffer_for(self, track_key: TrackKey) -> CropBuffer:
        buf = self._buffers.get(track_key)
        if buf is None:
            buf = CropBuffer(track_key=track_key, top_k=self.top_k)
            self._buffers[track_key] = buf
        return buf

    def note_track(self, track: TrackResult, observed_at: str) -> CropBuffer:
        buf = self.buffer_for(track.track_key)
        buf.note_track(track, observed_at)
        return buf

    def offer_crop(self, track_key: TrackKey, crop: TrackCrop) -> bool:
        return self.buffer_for(track_key).offer(crop)

    def take_finished(self, current_pts_ms: int) -> list[CropBuffer]:
        """Remove and return buffers whose track is finished.

        Finished means either idle for track_idle_ms on the source timeline, or
        open longer than max_track_duration_ms. Uses PTS, not wallclock: during
        a 5x replay a wallclock timeout would finalize every track immediately
        and destroy the consensus the pipeline depends on.
        """
        finished: list[TrackKey] = []
        for key, buf in self._buffers.items():
            if buf.last_pts_ms is None:
                continue
            idle = current_pts_ms - buf.last_pts_ms
            if idle >= self.track_idle_ms or buf.duration_ms >= self.max_track_duration_ms:
                finished.append(key)
        return [self._buffers.pop(key) for key in finished]

    def take_session(self, stream_session_id: str) -> list[CropBuffer]:
        """Remove and return every buffer belonging to one session.

        Called when a session ends for any reason. The caller finalizes these
        -- the vehicles were really observed -- and then the state is gone, so
        nothing can be appended to them from the next session.
        """
        keys = [k for k in self._buffers if k.stream_session_id == stream_session_id]
        return [self._buffers.pop(key) for key in keys]

    def take_all(self) -> list[CropBuffer]:
        """Drain everything, for shutdown and end of file."""
        buffers = list(self._buffers.values())
        self._buffers.clear()
        return buffers

    def stats(self) -> dict[str, Any]:
        return {
            "open_tracks": len(self._buffers),
            "crops_held": sum(len(b.crops) for b in self._buffers.values()),
            "top_k": self.top_k,
        }
