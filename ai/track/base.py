"""The tracker interface. Stage 4 of the fourteen.

Its whole job is to turn a list of DetectorResult -- boxes with no identity -- into
a list of TrackResult, where each box carries a TrackKey that is stable across
frames. Everything after this point in the pipeline groups by that key: quality
ranking picks the best frame *per track*, fusion votes across frames *of one
track*, and one sighting event is emitted *per track*.

Which makes the key the single most consequential thing in this module, and it is
three parts:

    TrackKey = (camera_id, stream_session_id, track_id)

Not (camera_id, track_id). ByteTrack -- any tracker -- restarts numbering at 1 for
a fresh session, so after a reconnect on cam04 the next vehicle is track 1 again.
With a two-part key it inherits every frame of evidence from the *previous*
track 1: a different vehicle, minutes earlier, possibly a different plate. Fusion
then votes across two vehicles' characters and emits one plate that belongs to
neither, attached to a journey showing a car in two places it never was.

That failure is silent. Nothing throws, the event validates, the map renders, and
the only symptom is a wrong answer that looks like a working feature. The session
id is not defensive programming; it is the difference between a system that is
right and one that is confidently wrong.

Session boundaries reach the tracker through ai/track/registry.py, which listens
for SessionChange from the media layer. The tracker itself never inspects a
timeline -- it is told.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Sequence

from ai.contracts.stages import DetectorResult, TrackResult

# Frames a confirmed track survives with no matching detection before it is
# dropped. At the 100 ms sampling interval, 30 frames is three seconds.
#
# Three seconds is enough to cross behind a stopped bus, which is the occlusion
# that actually happens at a junction. Much longer starts re-acquiring the wrong
# vehicle: after five or six seconds the filter's uncertainty has grown until its
# gate covers most of a lane, and a different car entering that lane matches.
DEFAULT_TRACK_BUFFER = 30

# Consecutive hits before a track is reported. Three at 100 ms is 300 ms of
# evidence.
#
# The night frames from cam04 are full of headlight glare on wet tarmac, and a
# single-frame detector false positive is common. Requiring persistence costs 300 ms
# of latency on a genuine vehicle -- which nothing downstream notices -- and removes
# essentially all of them, because glare does not move like a vehicle for three
# consecutive frames.
DEFAULT_MIN_HITS = 3

# ByteTrack's two thresholds. Detections above the high threshold may start a new
# track; detections between low and high may only continue an existing one.
#
# This is the paper's actual contribution and it is worth stating plainly: a
# partially occluded vehicle's confidence drops, and a single-threshold tracker
# discards those boxes and loses the track. ByteTrack keeps them for association
# only. A marginal box is weak evidence that *something new* is there and strong
# evidence about *where a known vehicle went*, and the two thresholds encode
# exactly that asymmetry.
DEFAULT_HIGH_THRESHOLD = 0.5
DEFAULT_LOW_THRESHOLD = 0.1


def check_detector_threshold(
    detector_confidence_threshold: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
) -> Optional[str]:
    """Warn when the detector's own filter has swallowed the tracker's low band.

    The invariant is one line: **a detector must not filter above the tracker's low
    threshold.** If it does, every box in [low_threshold, detector_threshold) is gone
    before the tracker is called, stage 2 receives nothing, and a two-stage tracker
    silently degrades into a one-stage tracker.

    This is not hypothetical. ai/detect/base.py's DEFAULT_CONFIDENCE_THRESHOLD is
    0.35 and DEFAULT_LOW_THRESHOLD here is 0.1, so *the default configuration has
    this bug*, and it produces no error, no warning and no obviously wrong number --
    only a matched_low counter sitting at 0. Measured cost of the misconfiguration on
    the synthetic fixture (200 frames, 6 vehicles, 15% miss, 4 px jitter, size-dependent
    confidence and a 25% mid-track dip):

        detector 0.05, stage 2 live      recall 0.588   6 track ids   0 fragments
        detector 0.35, stage 2 starved   recall 0.342   8 track ids   2 fragments

    The second row is bit-identical to running the tracker with use_low_stage=False --
    same recall, same 8 ids, same 2 fragments, same 35 tracks started. That equivalence
    is the sharpest way to state the cost: this misconfiguration is not a degradation of
    ByteTrack, it *is* the ablation that deletes ByteTrack's contribution, arrived at by
    changing a number in a different stage's config file.

    Returns None when the configuration is sound, or the warning text. Called by
    scripts/validate_config.py, so the mistake surfaces at config-load time rather
    than as a tracking quality shortfall nobody attributes to the detector.
    """
    if detector_confidence_threshold <= low_threshold:
        return None
    return (
        f"detector confidence_threshold={detector_confidence_threshold} is above the "
        f"tracker's low_threshold={low_threshold}, so detections in "
        f"[{low_threshold}, {detector_confidence_threshold}) are discarded before the "
        f"tracker sees them. ByteTrack's second association stage will receive no "
        f"input and the tracker degrades to single-threshold association. Set the "
        f"detector's confidence_threshold to {low_threshold} or below."
    )


class VehicleTracker(Protocol):
    """What the pipeline requires. Any tracker satisfying this is swappable."""

    def update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]: ...

    def reset(self, *, stream_session_id: Optional[str] = None) -> None: ...

    @property
    def tracker_name(self) -> str: ...


class BaseTracker(ABC):
    """Shared bookkeeping: session identity, ID allocation, counters.

    Subclasses implement _update and nothing else. In particular they never build
    a TrackResult themselves -- _emit does it, so the three-part key is constructed
    in exactly one place in the codebase and cannot be got wrong in a second one.
    """

    def __init__(
        self,
        camera_id: str,
        stream_session_id: str,
        *,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        min_hits: int = DEFAULT_MIN_HITS,
    ) -> None:
        if not camera_id:
            raise ValueError("tracker needs a camera_id; every track is scoped to one")
        if not stream_session_id:
            raise ValueError(
                "tracker needs a stream_session_id. Omitting it merges unrelated "
                "vehicles across a reconnect -- see the module docstring."
            )
        self.camera_id = camera_id
        self.stream_session_id = stream_session_id
        self.track_buffer = int(track_buffer)
        self.min_hits = int(min_hits)

        self._next_track_id = 1
        self.frames_seen = 0
        self.tracks_started = 0
        self.tracks_removed = 0
        self.reacquisitions = 0
        self.sessions_seen = 1

    def update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        self.frames_seen += 1
        return self._update(detections, frame_index=frame_index, pts_ms=pts_ms)

    def reset(self, *, stream_session_id: Optional[str] = None) -> None:
        """Drop all state. Called on every session change.

        Track IDs restart at 1, which is correct and is precisely why the session id
        is part of the key. Carrying IDs forward instead would look tidier and would
        be a lie: the tracker has no idea whether the vehicle it was following is
        still in frame after a thirty-second reconnect, and asserting that it is the
        same one is the fabrication this design exists to prevent.
        """
        if stream_session_id:
            if stream_session_id != self.stream_session_id:
                self.sessions_seen += 1
            self.stream_session_id = stream_session_id
        self._next_track_id = 1
        self._reset()

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self.tracks_started += 1
        return track_id

    def _emit(
        self,
        *,
        bbox_xyxy: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        track_id: int,
        frame_index: int,
        pts_ms: int,
    ) -> TrackResult:
        """The only place a TrackResult is constructed. See the class docstring."""
        return TrackResult(
            camera_id=self.camera_id,
            stream_session_id=self.stream_session_id,
            track_id=track_id,
            bbox_xyxy=bbox_xyxy,
            class_name=class_name,
            confidence=round(float(confidence), 4),
            frame_index=frame_index,
            pts_ms=pts_ms,
        )

    @abstractmethod
    def _update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]: ...

    @abstractmethod
    def _reset(self) -> None: ...

    @property
    @abstractmethod
    def tracker_name(self) -> str: ...

    @property
    def active_track_count(self) -> int:
        return 0

    def stats(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker_name,
            "camera_id": self.camera_id,
            "stream_session_id": self.stream_session_id,
            "sessions_seen": self.sessions_seen,
            "frames_seen": self.frames_seen,
            "tracks_started": self.tracks_started,
            "tracks_removed": self.tracks_removed,
            "reacquisitions": self.reacquisitions,
            "active_tracks": self.active_track_count,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(camera={self.camera_id!r}, "
            f"session={self.stream_session_id[:8]}, "
            f"started={self.tracks_started})"
        )
