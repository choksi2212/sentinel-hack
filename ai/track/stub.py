"""Two trackers that are not ByteTrack, for two different reasons.

IOUTracker is a real fallback. Published as "High-Speed Tracking-by-Detection
Without Using Image Information" (Bochinski et al., 2017) and it is genuinely what
it sounds like: greedy IoU association, no motion model, no filter. On a low-value
static camera at 10 fps it is close enough to ByteTrack to be worth its near-zero
cost, and across a grid of 80,000 cameras "close enough for almost free" is a real
architectural option rather than a compromise. It also runs with numpy alone, which
makes it the tracker that proves the pipeline works on a machine with nothing
installed.

Its weakness is exactly what the Kalman filter fixes, and it is stated rather than
discovered: no prediction means an occluded vehicle's box stays where it was last
seen, so anything longer than a frame or two of occlusion loses the track. Use it
where occlusion is rare, not at a junction.

OracleTracker reads identity from the synthetic source's ground truth. It exists to
separate two questions that otherwise get answered together: is the *plate reading*
wrong, or is the *tracking* wrong? Run the pipeline with a real detector and a
perfect tracker and any remaining error is downstream of tracking. Without it, a
poor end-to-end number has at least four possible causes and no way to choose
between them.

Neither may appear in a published benchmark. Both report ships=False and the
factory refuses them for publication.
"""

from typing import Any, Optional, Sequence

from ai.contracts.stages import DetectorResult, TrackResult
from ai.track.assignment import gate_cost, iou_cost, solve
from ai.track.base import DEFAULT_MIN_HITS, DEFAULT_TRACK_BUFFER, BaseTracker

# Minimum IoU for the fallback tracker to continue a track. Stricter than
# ByteTrack's stage 1 (0.2) because there is no motion model to fall back on: IoU
# is the only evidence available, so it has to carry the whole decision.
IOU_TRACKER_MIN_IOU = 0.35


class _SimpleTrack:
    __slots__ = (
        "track_id",
        "bbox",
        "class_votes",
        "confidence",
        "hits",
        "misses",
        "first_pts_ms",
        "last_pts_ms",
    )

    def __init__(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        pts_ms: int,
    ) -> None:
        self.track_id = track_id
        self.bbox = bbox
        self.class_votes: dict[str, float] = {class_name: float(confidence)}
        self.confidence = float(confidence)
        self.hits = 1
        self.misses = 0
        self.first_pts_ms = pts_ms
        self.last_pts_ms = pts_ms

    @property
    def class_name(self) -> str:
        return max(sorted(self.class_votes), key=lambda name: self.class_votes[name])


class IOUTracker(BaseTracker):
    """Greedy IoU association, no motion model. numpy only.

    Reports ships=False through the factory. It is a legitimate technique and it is
    not the technique this system claims to use, so a benchmark row produced by it
    would misdescribe the submission.
    """

    def __init__(
        self,
        camera_id: str,
        stream_session_id: str,
        *,
        min_iou: float = IOU_TRACKER_MIN_IOU,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        min_hits: int = DEFAULT_MIN_HITS,
        confidence_threshold: float = 0.3,
    ) -> None:
        super().__init__(
            camera_id,
            stream_session_id,
            track_buffer=track_buffer,
            min_hits=min_hits,
        )
        self.min_iou = float(min_iou)
        self.confidence_threshold = float(confidence_threshold)
        self._tracks: list[_SimpleTrack] = []

    def _update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        usable = [d for d in detections if d.confidence >= self.confidence_threshold]

        max_cost = 1.0 - self.min_iou
        matches, unmatched_tracks, unmatched_dets = solve(
            gate_cost(
                iou_cost([t.bbox for t in self._tracks], [d.bbox_xyxy for d in usable]),
                max_cost=max_cost,
            ),
            max_cost=max_cost,
        )

        for track_idx, det_idx in matches:
            track, detection = self._tracks[track_idx], usable[det_idx]
            track.bbox = detection.bbox_xyxy
            track.class_votes[detection.class_name] = track.class_votes.get(
                detection.class_name, 0.0
            ) + float(detection.confidence)
            track.confidence = float(detection.confidence)
            track.hits += 1
            track.misses = 0
            track.last_pts_ms = pts_ms

        for index in unmatched_tracks:
            self._tracks[index].misses += 1

        for index in unmatched_dets:
            detection = usable[index]
            self._tracks.append(
                _SimpleTrack(
                    self._allocate_track_id(),
                    detection.bbox_xyxy,
                    detection.class_name,
                    detection.confidence,
                    pts_ms,
                )
            )

        # A tentative track is dropped on its first miss, matching ByteTrack, so
        # the two are comparable. Differing lifecycle rules would make an
        # A/B measurement between them measure the rules rather than the trackers.
        survivors: list[_SimpleTrack] = []
        for track in self._tracks:
            confirmed = track.hits >= self.min_hits
            limit = self.track_buffer if confirmed else 0
            if track.misses > limit:
                self.tracks_removed += 1
            else:
                survivors.append(track)
        self._tracks = survivors

        return [
            self._emit(
                bbox_xyxy=track.bbox,
                class_name=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
                frame_index=frame_index,
                pts_ms=pts_ms,
            )
            for track in self._tracks
            if track.hits >= self.min_hits and track.misses == 0
        ]

    def _reset(self) -> None:
        self._tracks = []

    @property
    def tracker_name(self) -> str:
        return "iou"

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.hits >= self.min_hits)

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({"min_iou": self.min_iou, "motion_model": None})
        return base


class OracleTracker(BaseTracker):
    """Identity from the synthetic source's ground truth. Diagnostic only.

    Matches each detection to the ground-truth vehicle whose box overlaps it most,
    then uses that vehicle's stable id as the track id. So it is a perfect tracker
    given a perfect detector, and given an imperfect detector it still never
    switches an identity -- which is what makes it useful for isolating where an
    end-to-end error came from.

    Ground truth is looked up by PTS. Never by frame index: sampling means emitted
    frame 14 is generator frame 42, so indexing by the emitted counter compares
    against a different frame and yields an accuracy figure that is wrong while
    looking entirely plausible. The same trap is documented at length in
    ai/media/synthetic_source.py.
    """

    def __init__(
        self,
        camera_id: str,
        stream_session_id: str,
        source: Any,
        *,
        min_iou: float = 0.1,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        min_hits: int = 1,
    ) -> None:
        if not hasattr(source, "truth_at_pts"):
            raise TypeError(
                "OracleTracker needs a source exposing truth_at_pts(); "
                f"{type(source).__name__} does not. Use source mode 'synthetic'."
            )
        super().__init__(
            camera_id,
            stream_session_id,
            track_buffer=track_buffer,
            min_hits=min_hits,
        )
        self.source = source
        self.min_iou = float(min_iou)
        self._ids: dict[str, int] = {}
        self.unresolved_frames = 0
        self.unmatched_detections = 0

    def _update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        truth = self.source.truth_at_pts(pts_ms)
        if truth is None:
            # Loud in the stats, silent in the output. Returning the detections with
            # invented ids would look like it worked and would be measuring nothing.
            self.unresolved_frames += 1
            return []

        vehicles = list(truth.vehicles)
        if not vehicles or not detections:
            self.unmatched_detections += len(detections)
            return []

        max_cost = 1.0 - self.min_iou
        matches, _, unmatched_dets = solve(
            gate_cost(
                iou_cost(
                    [v.vehicle_bbox_xyxy for v in vehicles],
                    [d.bbox_xyxy for d in detections],
                ),
                max_cost=max_cost,
            ),
            max_cost=max_cost,
        )
        self.unmatched_detections += len(unmatched_dets)

        results: list[TrackResult] = []
        for vehicle_idx, det_idx in matches:
            vehicle = vehicles[vehicle_idx]
            detection = detections[det_idx]
            key = str(vehicle.vehicle_id)
            if key not in self._ids:
                self._ids[key] = self._allocate_track_id()
            results.append(
                self._emit(
                    bbox_xyxy=detection.bbox_xyxy,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    track_id=self._ids[key],
                    frame_index=frame_index,
                    pts_ms=pts_ms,
                )
            )
        return results

    def _reset(self) -> None:
        self._ids = {}

    @property
    def tracker_name(self) -> str:
        return "oracle"

    @property
    def is_oracle(self) -> bool:
        return True

    @property
    def active_track_count(self) -> int:
        return len(self._ids)

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "unresolved_frames": self.unresolved_frames,
                "unmatched_detections": self.unmatched_detections,
                "is_oracle": True,
            }
        )
        return base


class ScriptedTracker(BaseTracker):
    """Replays a fixed table of TrackResult rows. For exact downstream assertions.

    A test for temporal fusion needs a specific sequence of tracked frames and does
    not care how they were produced. Driving it through a real tracker makes the
    fusion test fail when the tracker changes, which is a test that reports the
    wrong thing.

    The table is keyed by frame index, not PTS: a test constructs both sides, so the
    two cannot drift, and the index is the more readable key to write a fixture in.
    """

    def __init__(
        self,
        camera_id: str,
        stream_session_id: str,
        script: dict[int, Sequence[tuple[int, tuple[int, int, int, int], str, float]]],
    ) -> None:
        super().__init__(camera_id, stream_session_id)
        self._script = dict(script)
        self._cursor = 0

    def _update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        rows = self._script.get(self._cursor, ())
        self._cursor += 1
        return [
            self._emit(
                bbox_xyxy=bbox,
                class_name=class_name,
                confidence=confidence,
                track_id=track_id,
                frame_index=frame_index,
                pts_ms=pts_ms,
            )
            for track_id, bbox, class_name, confidence in rows
        ]

    def _reset(self) -> None:
        self._cursor = 0

    @property
    def tracker_name(self) -> str:
        return "scripted"
