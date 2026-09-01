"""ByteTrack. Contracts section 11 locks it; MIT licensed; implemented here directly.

**Why implemented rather than imported.** The usual route is Roboflow's supervision
package, which is MIT and perfectly good. It is not installed on this machine, and
adding a dependency to get a two-stage association loop and a Kalman filter --
about three hundred lines, both of which are needed anyway -- buys less than it
costs. What it does buy is real and is stated honestly: a widely-used
implementation has been exercised by many more people than this one has.

The mitigation is that this one is *measured against ground truth* rather than
trusted. The synthetic source knows which vehicle is which on every frame, so the
ID-switch rate and the fragmentation rate are numbers, not hopes. The figures and the
ablations are in the ByteTracker docstring below; tests/test_track.py asserts them.

**The algorithm**, and specifically the part that is ByteTrack rather than SORT:

    1. Split detections at the high threshold.
    2. Predict every track forward one frame.
    3. Associate HIGH detections with confirmed+lost tracks, by IoU weighted by
       detection score, vetoed by Mahalanobis distance.
    4. Associate LOW detections with the tracks that came out of step 3 unmatched.
       <-- this step is ByteTrack
    5. Associate leftover HIGH detections with tentative tracks.
    6. Unmatched tracks age; unmatched HIGH detections start new tracks.

Step 4 is the whole paper. A vehicle passing behind a pole has its confidence
collapse for two or three frames; a single-threshold tracker throws those boxes
away, loses the track, and starts a new one when the vehicle emerges. That single
event -- one vehicle becoming two -- is the dominant tracking failure on junction
footage, and it costs a duplicate sighting and a broken journey each time.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ai.contracts.stages import DetectorResult, TrackResult
from ai.track.assignment import (
    fuse_detection_score,
    gate_cost,
    iou_cost,
    solve,
    solver_name as assignment_solver,
)
from ai.track.base import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MIN_HITS,
    DEFAULT_TRACK_BUFFER,
    BaseTracker,
)
from ai.track.kalman import (
    CHI2_INV99,
    POSITION_SIZE_DIMS,
    solver_name as linalg_solver,
    xyxy_to_cxcyah,
)
from ai.track.track import CONFIRMED, LOST, Track

# Which dimensions of (cx, cy, a, h) the motion gate judges, and the matching
# squared Mahalanobis threshold. The two are written on adjacent lines and the
# threshold is indexed BY the dimension count, because setting a 4-DOF threshold on
# a 3-DOF distance loosens the gate by about 18% while looking entirely deliberate.
#
# Aspect ratio is excluded and the level is 99% rather than the DeepSORT-conventional
# 95%; _gate explains the first and ai/track/kalman.py the second, both with the
# measurements behind them.
GATING_DIMS = POSITION_SIZE_DIMS
GATING_THRESHOLD = CHI2_INV99[len(GATING_DIMS)]

# Cost ceilings per association stage, all as 1 - IoU.
#
# Stage 2 is looser (0.5 -> IoU >= 0.5) than stage 1 (0.8 -> IoU >= 0.2) which
# reads backwards and is not. Stage 2 is matching *low-confidence* boxes, where the
# detector is already unsure the object is there; requiring strong geometric
# agreement is the only remaining evidence that it is the same vehicle. Being
# permissive with a marginal box is how a plate gets attached to the wrong car.
MATCH_THRESH_HIGH = 0.8
MATCH_THRESH_LOW = 0.5
MATCH_THRESH_TENTATIVE = 0.7


class ByteTracker(BaseTracker):
    """Two-stage association with a Kalman motion model.

    **Measured on the synthetic corpus** -- seed 42, 200 emitted frames at 120 ms,
    6 vehicles, ground-truth identity known per frame, a truth box counted as matched
    when it overlaps an emitted box by IoU >= 0.5. Two failure modes are counted
    separately because they cost different things: a *fragment* is one real vehicle
    spread across several track ids (a duplicate sighting, a broken journey), an *ID
    switch* is one track id covering several real vehicles (one vehicle's plate
    attributed to another -- the expensive one).

    Against a perfect oracle detector, so the numbers describe the tracker alone:

        bytetrack      recall 0.904    6 ids for 6 vehicles    0 fragments  0 switches
        iou baseline   recall 0.938    6 ids for 6 vehicles    0 fragments  0 switches

    With nothing to go wrong, a motion model earns nothing -- correctly. It is worth
    reporting that the naive baseline wins on recall here.

    Against a degraded oracle (15% miss rate, 4 px jitter, 0.1 false positives per
    frame, size-dependent confidence, 25% of vehicle-frames dipped to 0.3):

        full                  recall 0.588   6 ids   0 fragments  0 switches
        stage 2 disabled      recall 0.342   8 ids   2 fragments  0 switches
        motion gate disabled  recall 0.588   6 ids   0 fragments  0 switches
        iou baseline          recall 0.667   9 ids   5 fragments  2 switches

    Read those four rows in order, because each says something different:

    *Stage 2 is the whole tracker.* Turning it off costs 0.246 recall -- 42% of what
    the tracker was recovering -- and splits 6 vehicles across 8 identities. 41 of the
    matched frames come from low-confidence boxes a single-threshold tracker discards.
    This is the paper's claim, and it reproduces.

    *The motion gate contributes nothing measurable here, and is kept anyway.* Its
    ablation is identical to the full configuration, so on this fixture it is free but
    unproven. That is a statement about the fixture, not a defence of the gate: four
    well-separated directional lanes with no crossing trajectories do not contain the
    failure the gate exists to prevent. It is retained because a real junction does,
    and because at the 99% level it demonstrably costs nothing -- see
    ai/track/kalman.py for why 99 and not the conventional 95.

    *The IoU baseline has higher recall and is still the wrong tracker.* It matches
    more boxes while scattering 6 vehicles over 9 identities with 2 switches. The
    primary metric is correct-plate events per vehicle, and a switch pools two
    vehicles' plate evidence into one wrong answer -- a failure no amount of recall
    compensates for. Raw recall is the wrong summary statistic and this row is why.
    """

    def __init__(
        self,
        camera_id: str,
        stream_session_id: str,
        *,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        min_hits: int = DEFAULT_MIN_HITS,
        use_low_stage: bool = True,
        use_gating: bool = True,
        fuse_score: bool = True,
    ) -> None:
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError(
                f"need 0 <= low_threshold <= high_threshold <= 1, got "
                f"low={low_threshold} high={high_threshold}"
            )
        super().__init__(
            camera_id,
            stream_session_id,
            track_buffer=track_buffer,
            min_hits=min_hits,
        )
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        # Both flags exist for ablation. A claim that the second stage helps is
        # worth nothing unless it can be turned off and re-measured.
        self.use_low_stage = bool(use_low_stage)
        self.use_gating = bool(use_gating)
        self.fuse_score = bool(fuse_score)

        self._tracks: list[Track] = []
        self.matched_high = 0
        self.matched_low = 0
        self.matched_tentative = 0
        self.gate_vetoes = 0
        self.detections_dropped_low = 0

    # ------------------------------------------------------------------- update

    def _update(
        self,
        detections: Sequence[DetectorResult],
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        high, low = self._split(detections)

        # Every track, matched or not. An unmatched track that is not predicted
        # forward stays where it was last seen, so when the vehicle reappears two
        # metres down the road the boxes no longer overlap and it becomes a new
        # track -- the exact failure the lost buffer exists to prevent.
        for track in self._tracks:
            track.predict()

        pool = [t for t in self._tracks if t.state in (CONFIRMED, LOST)]
        tentative = [t for t in self._tracks if t.state not in (CONFIRMED, LOST)]

        # --- stage 1: high-confidence detections against established tracks
        matches, unmatched_tracks, unmatched_high = self._associate(
            pool, high, MATCH_THRESH_HIGH, fuse=self.fuse_score
        )
        for track_idx, det_idx in matches:
            self._apply(pool[track_idx], high[det_idx], frame_index, pts_ms)
        self.matched_high += len(matches)

        # --- stage 2: low-confidence detections against what stage 1 left over.
        # This is ByteTrack. See the module docstring.
        remaining = [pool[i] for i in unmatched_tracks]
        if self.use_low_stage and low and remaining:
            low_matches, still_unmatched, _ = self._associate(
                remaining, low, MATCH_THRESH_LOW, fuse=False
            )
            for track_idx, det_idx in low_matches:
                self._apply(remaining[track_idx], low[det_idx], frame_index, pts_ms)
            self.matched_low += len(low_matches)
            aged = [remaining[i] for i in still_unmatched]
        else:
            aged = remaining

        # --- stage 3: leftover high detections against tentative tracks.
        # Last, not first: a confirmed track has earned priority on a detection over
        # an unproven one, and reversing the order lets a spurious tentative box
        # steal a real vehicle's detection and break its track.
        #
        # gate=False, and this is not a shortcut. A tentative track has one or two
        # observations, so its velocity estimate is still converging from the zero it
        # was initiated at, and the Mahalanobis distance during that convergence
        # legitimately exceeds chi2(4, 0.95) for a perfectly correct pair. Measured on
        # a clean synthetic target moving 30 px/frame -- no detector noise at all --
        # the squared distance for the RIGHT detection runs 7.47, 10.44, 6.37, 3.27,
        # 1.79 over the first five frames. Only the second one matters: it is above
        # the 9.4877 threshold, so the gate vetoes the correct detection on frame 2,
        # the tentative track takes a miss, tentative tracks are dropped on their
        # first miss, and a fresh track starts on the next frame from the same
        # vehicle. That loop cost 32 track ids for 6 vehicles with a *perfect*
        # detector before this argument was passed. The motion model has nothing to
        # contribute until it has seen motion, and asking it anyway is worse than not
        # asking.
        leftover = [high[i] for i in unmatched_high]
        if tentative and leftover:
            tent_matches, unmatched_tent, unmatched_leftover = self._associate(
                tentative, leftover, MATCH_THRESH_TENTATIVE, fuse=False, gate=False
            )
            for track_idx, det_idx in tent_matches:
                self._apply(tentative[track_idx], leftover[det_idx], frame_index, pts_ms)
            self.matched_tentative += len(tent_matches)
            aged.extend(tentative[i] for i in unmatched_tent)
            starters = [leftover[i] for i in unmatched_leftover]
        else:
            aged.extend(tentative)
            starters = leftover

        for track in aged:
            track.mark_missed(max_age=self.track_buffer)

        for detection in starters:
            self._tracks.append(
                Track(
                    self._allocate_track_id(),
                    detection.bbox_xyxy,
                    detection.class_name,
                    detection.confidence,
                    frame_index,
                    pts_ms,
                )
            )

        removed = [t for t in self._tracks if t.is_removed]
        self.tracks_removed += len(removed)
        self.reacquisitions = sum(t.reacquisitions for t in self._tracks)
        self._tracks = [t for t in self._tracks if not t.is_removed]

        return [
            self._emit(
                bbox_xyxy=track.report_bbox_xyxy,
                class_name=track.class_name,
                confidence=track.confidence,
                track_id=track.track_id,
                frame_index=frame_index,
                pts_ms=pts_ms,
            )
            for track in self._tracks
            if track.is_active
        ]

    def _split(
        self, detections: Sequence[DetectorResult]
    ) -> tuple[list[DetectorResult], list[DetectorResult]]:
        high: list[DetectorResult] = []
        low: list[DetectorResult] = []
        for detection in detections:
            if detection.confidence >= self.high_threshold:
                high.append(detection)
            elif detection.confidence >= self.low_threshold:
                low.append(detection)
            else:
                self.detections_dropped_low += 1
        return high, low

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[DetectorResult],
        max_cost: float,
        *,
        fuse: bool,
        gate: bool = True,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = [t.bbox_xyxy for t in tracks]
        det_boxes = [d.bbox_xyxy for d in detections]
        cost = iou_cost(track_boxes, det_boxes)

        if fuse:
            cost = fuse_detection_score(cost, [d.confidence for d in detections])

        infeasible = None
        if self.use_gating and gate:
            infeasible = self._gate(tracks, detections)
            self.gate_vetoes += int(infeasible.sum())

        return solve(
            gate_cost(cost, max_cost=max_cost, infeasible=infeasible),
            max_cost=max_cost,
        )

    def _gate(
        self, tracks: Sequence[Track], detections: Sequence[DetectorResult]
    ) -> np.ndarray:
        """Boolean mask of pairs the motion model rules out.

        IoU cannot distinguish two vehicles in adjacent lanes whose boxes overlap;
        the filter can, because it knows where each one was going. This veto is what
        stops the plates being swapped between them, and a swap produces two wrong
        journeys rather than one missing vehicle.

        Position and height, not aspect ratio -- only_position=False would include
        it, and it is the one dimension of a detection box that is routinely wrong.
        A vehicle entering frame is clipped by the frame edge, so its measured
        aspect sweeps from 0.49 to 1.5 over three frames while its position is
        perfectly predictable. Measured contribution to the squared distance on such
        a pair: 9.6 of a 9.4877 budget from aspect alone, against 2.3 from position.
        Gating on it therefore rejects correct pairs at exactly the moment a track is
        least able to survive one.
        """
        measurements = np.array(
            [xyxy_to_cxcyah(d.bbox_xyxy) for d in detections], dtype=np.float64
        )
        mask = np.zeros((len(tracks), len(detections)), dtype=bool)
        for row, track in enumerate(tracks):
            mask[row] = (
                track.gating_distance(measurements, dims=GATING_DIMS) > GATING_THRESHOLD
            )
        return mask

    def _apply(
        self,
        track: Track,
        detection: DetectorResult,
        frame_index: int,
        pts_ms: int,
    ) -> None:
        track.update(
            detection.bbox_xyxy,
            detection.class_name,
            detection.confidence,
            frame_index,
            pts_ms,
            min_hits=self.min_hits,
        )

    def _reset(self) -> None:
        self._tracks = []

    # ----------------------------------------------------------------- metadata

    @property
    def tracker_name(self) -> str:
        return "bytetrack"

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_confirmed)

    def tracks(self) -> list[Track]:
        """Live track objects. For diagnostics and the benchmark, not the pipeline.

        The pipeline consumes TrackResult. Handing out mutable internals is fine for
        a metrics writer that reads them and wrong for a stage that might keep one
        across a reset.
        """
        return list(self._tracks)

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
                "matched_high": self.matched_high,
                "matched_low": self.matched_low,
                "matched_tentative": self.matched_tentative,
                "gate_vetoes": self.gate_vetoes,
                "dropped_below_low": self.detections_dropped_low,
                "low_stage_enabled": self.use_low_stage,
                "gating_enabled": self.use_gating,
                "assignment_solver": assignment_solver(),
                "linalg_solver": linalg_solver(),
            }
        )
        return base
