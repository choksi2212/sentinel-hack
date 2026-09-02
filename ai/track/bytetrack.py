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

# Which dimensions of (cx, cy, a, h) the motion gate judges, and the matching squared
# Mahalanobis threshold. The threshold is indexed BY the dimension count so the two
# cannot be mismatched: a 4-DOF threshold on a 3-DOF distance is a looser gate that
# looks entirely deliberate, and writing it as one expression makes that unreachable.
#
# Both halves were swept on the synthetic corpus -- 200 frames, seed 42, 6 vehicles with
# known identity, IoU >= 0.5 to count a truth box as matched. Perfect oracle and degraded
# oracle (15% miss, 4 px jitter, 0.1 FP/frame, size falloff, 25% dip). "veto" is the
# tracker's own gate_vetoes counter:
#
#                                    perfect                     degraded
#     gate dims / level      recall  ids frag  veto  |  recall  ids frag  veto
#     (cx, cy, h) 99%  SHIP   0.904    6    0   961  |   0.588    6    0   570
#     (cx, cy, h) 95%         0.904    6    0   961  |   0.588    6    0   570
#     (cx, cy, a, h) 95%      0.812   16   10  2201  |   0.442    9    3   828
#     (cx, cy, a, h) 99%      0.821   15    9  2045  |   0.442    9    3   820
#     (cx, cy) 95%            0.887    8    2  1213  |   0.508    8    2   731
#     (cx, cy) 99%            0.904    6    0   961  |   0.588    6    0   570
#     gate disabled           0.904    6    0     0  |   0.588    6    0     0
#
# Read the table for what it says rather than what a reader expects. **Including aspect
# ratio is the whole finding, and it is a finding about one direction only.** Aspect
# scatters 6 vehicles across 15 identities with a PERFECT detector -- _gate explains the
# mechanism -- and no other cell in the table moves a single identity.
#
# In particular, height earns nothing here. Rows 1 and 6 are identical on both fixtures
# down to the veto count, and that is not a coincidence of the summary statistics: over
# 2201 track/detection pairs across both fixtures, the number vetoed by (cx, cy, h) that
# (cx, cy) would have passed is **zero**. All 18 disagreements run the other way -- the
# two-dimensional gate is the *stricter* one, because adding a dimension raises the
# threshold (9.2103 -> 11.3449) faster than it adds distance on these trajectories. So the
# shipped gate is three dimensions of which two do the work.
#
# It is kept at three anyway, and the argument is the same one as for keeping the gate at
# all: what height catches is a box that holds its image position while changing size
# sharply, which is a vehicle emerging from behind an occluder or a detector snapping from
# a partial box to a full one. Four straight lanes with no crossing traffic contain none of
# that. An earlier version of this comment claimed dropping height "costs 2 fragments the
# other way" -- true, but only of row 5, which is at 95%; at the shipped level it costs
# nothing measurable. The 8 identities in row 5 are the cost of a *tighter* gate, not of a
# smaller one.
#
# The confidence level is immaterial wherever the dimensions are right: 95% and 99% are
# indistinguishable at three dims and near-identical at four. The one row where the level
# decides anything is row 5 versus row 6 -- two dimensions, where 95% fragments and 99%
# does not -- and neither is shipped. The level is kept at 99% for the reason in
# ai/track/kalman.py, which is an argument rather than a measurement, and that file now
# says so.
#
# The bottom row is the honest one and it is the same finding as the ByteTracker
# docstring's: correctly configured, this gate changes nothing on either fixture. It is
# retained because these four lanes contain no crossing trajectories and a real junction
# does. Anyone removing it should remove it on that argument, not on this table.
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
    failure the gate exists to prevent. It is retained because a real junction does, and
    because as configured it demonstrably costs nothing. What its configuration is *not*
    free in is the dimension count -- gating on aspect ratio would cost 10 fragments and
    16 identities for these same 6 vehicles with a perfect detector, which is the largest
    effect in this module and is tabulated at GATING_DIMS.

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
        # Reacquisitions earned by tracks that no longer exist. Needed because
        # BaseTracker.reacquisitions is a cumulative counter and the live tracks are not
        # a cumulative record: see the note where it is updated in _update.
        self._retired_reacquisitions = 0
        self.matched_high = 0
        self.matched_low = 0
        self.matched_tentative = 0
        self.gate_vetoes = 0
        self.detections_dropped_low = 0

    # -------------------------------------------------------------- provenance

    @property
    def model_name(self) -> str:
        """What this stage is, for the report's stage table.

        A tracker has no weights file, so the alternative to naming it here is the empty
        string -- and a provenance record that identifies three of four stages is not a
        provenance record. Included in the answer because they change the output: the two
        thresholds decide which detections may start a track at all, so two runs with the
        same footage, the same detector and different thresholds are not comparable, and
        a report that recorded them identically would invite exactly that comparison.
        """
        parts = [f"bytetrack@{self.high_threshold:g}/{self.low_threshold:g}"]
        # Only the ablation switches that are off, so the common case stays short and any
        # deviation is loud. A run with the second association stage disabled is a
        # different algorithm and its numbers must not be filed as ByteTrack's.
        for flag, label in (
            (self.use_low_stage, "no-low-stage"),
            (self.use_gating, "no-gating"),
            (self.fuse_score, "no-score-fusion"),
        ):
            if not flag:
                parts.append(label)
        return " ".join(parts)

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
        # gate=False, and it is the safe direction rather than a measured win. A tentative
        # track has one or two observations, so its velocity estimate is still converging
        # from the zero it was initiated at, and asking a motion model where a vehicle
        # will be before it has seen the vehicle move is asking a question it cannot
        # answer. Vetoing on the answer costs a whole track: tentative tracks are dropped
        # on their first miss, so one veto restarts the track from scratch.
        #
        # Measured, and the measurement does not support the strong version of that
        # story. A clean target moving 30 px/frame with no detector noise gives squared
        # distances to the CORRECT next box of 4.76, 6.80, 4.17, 2.13, 1.16, 0.69, 0.44
        # over the first seven frames -- every one of them inside the 11.3449 gate, so
        # nothing would have been vetoed. Turning the gate on for this stage changes
        # nothing on either fixture: perfect stays at recall 0.904 / 14 started / 6 ids,
        # degraded at 0.588 / 24 / 6, identical to the numbers in the class docstring.
        #
        # An earlier version of this comment cited 7.47, 10.44, 6.37, 3.27, 1.79 and a
        # veto on frame 2 against chi2(4, 0.95) = 9.4877, and claimed the loop cost 32
        # track ids for 6 vehicles. None of that reproduces, and 9.4877 is a threshold
        # this file has never shipped -- see GATING_THRESHOLD, which is the 3-DOF 99%
        # value. Those figures describe a gate configuration that was never committed.
        # What survives is the argument, and at 4 dimensions the same argument is
        # demonstrably right: the sweep above turns 6 vehicles into 16 identities. So this
        # stays off here, on the reasoning and not on a number.
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
                    min_hits=self.min_hits,
                )
            )

        removed = [t for t in self._tracks if t.is_removed]
        self.tracks_removed += len(removed)
        self._tracks = [t for t in self._tracks if not t.is_removed]

        # Retired tracks keep contributing. This was `sum(t.reacquisitions for t in
        # self._tracks)` over the live tracks only, which makes a counter that sits
        # beside tracks_started and tracks_removed in stats() -- both cumulative -- go
        # DOWN: a vehicle that came back from behind a bus and then left the junction
        # took its reacquisition with it, and a session change zeroed the total outright.
        # Measured: 1 while the track lived, 0 on the frame after it was removed.
        #
        # It matters because this number is the evidence that the lost buffer is doing
        # anything at all. A run reporting zero reacquisitions either never occluded a
        # vehicle or has a broken buffer, and the two look identical from the outside --
        # so a counter that quietly discards them is worse than no counter.
        self._retired_reacquisitions += sum(t.reacquisitions for t in removed)
        self.reacquisitions = self._retired_reacquisitions + sum(
            t.reacquisitions for t in self._tracks
        )

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
        perfectly predictable. Including it therefore rejects correct pairs at exactly
        the moment a track is least able to survive one.

        That is not a plausible mechanism offered without evidence -- it is the single
        largest effect measured anywhere in this module. Adding aspect to the gate, with
        everything else unchanged and a **perfect** detector, scatters 6 vehicles across
        16 track identities with 10 fragments and drops recall from 0.904 to 0.812. The
        full sweep is tabulated at GATING_DIMS. Nothing else in the tracker's
        configuration space does damage on that scale.
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
        # A session change drops every live track, so what they earned retires here
        # rather than vanishing. reset() does not zero the counters BaseTracker keeps --
        # sessions_seen counts up, tracks_started counts up -- and this one is no
        # different: "reacquisitions since this tracker was built" is the useful
        # quantity, and it has to survive the boundary that most often produces them.
        self._retired_reacquisitions += sum(t.reacquisitions for t in self._tracks)
        self.reacquisitions = self._retired_reacquisitions
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
