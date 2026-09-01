"""One tracked vehicle: its filter state, its lifecycle, and its class vote.

Split out from the tracker because the state machine is where the subtle bugs
live, and a class small enough to read in one screen is a class whose transitions
can be checked by eye.

The lifecycle:

    NEW ---> TENTATIVE ---> CONFIRMED ---> LOST ---> REMOVED
                  |                          |
                  +--> REMOVED               +--> CONFIRMED (re-found)

TENTATIVE is the one that earns its keep. A track is not reported until it has
been seen on several consecutive frames, so a single false-positive box -- a
headlight reflection on wet tarmac, of which the night frames have many -- never
becomes a vehicle event. The cost is that a genuine vehicle is reported a few
frames late, which is invisible at a 100 ms sampling interval.

LOST is the other one. A vehicle behind a bus keeps its identity for a buffer of
frames while the filter carries it forward, and re-acquires the same track_id when
it reappears. Without it, a two-second occlusion produces two vehicle events for
one car and a journey with an impossible gap.
"""

from typing import Optional, Sequence

import numpy as np

from ai.track.kalman import (
    MEASUREMENT_DIMS,
    KalmanBoxFilter,
    cxcyah_to_xyxy,
    shared_filter,
    xyxy_to_cxcyah,
)

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
LOST = "lost"
REMOVED = "removed"


class Track:
    """A single vehicle's identity across frames.

    Mutable by design. One instance per vehicle per session, updated in place --
    copying it every frame would be cleaner and would allocate a covariance matrix
    per track per frame for no benefit.
    """

    __slots__ = (
        "track_id",
        "state",
        "mean",
        "covariance",
        "class_votes",
        "confidence",
        "hits",
        "age",
        "time_since_update",
        "first_frame_index",
        "first_pts_ms",
        "last_frame_index",
        "last_pts_ms",
        "start_bbox",
        "last_bbox",
        "reacquisitions",
        "_filter",
    )

    def __init__(
        self,
        track_id: int,
        bbox_xyxy: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        frame_index: int,
        pts_ms: int,
        *,
        kalman: Optional[KalmanBoxFilter] = None,
    ) -> None:
        self.track_id = track_id
        self._filter = kalman or shared_filter()
        self.mean, self.covariance = self._filter.initiate(xyxy_to_cxcyah(bbox_xyxy))

        self.state = TENTATIVE
        # Class is decided by vote across the track's whole life, not taken from
        # the first frame. A vehicle's first detection is its smallest and worst,
        # and the detector's class on a 40 px box is close to a coin flip -- see the
        # auto-rickshaw note in ai/detect/rfdetr.py.
        self.class_votes: dict[str, float] = {class_name: float(confidence)}
        self.confidence = float(confidence)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.first_frame_index = frame_index
        self.first_pts_ms = pts_ms
        self.last_frame_index = frame_index
        self.last_pts_ms = pts_ms
        self.start_bbox = bbox_xyxy
        self.last_bbox = bbox_xyxy
        self.reacquisitions = 0

    # ------------------------------------------------------------------ geometry

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        """The filter's current estimate. This is the box association uses.

        Not the last observed box. On an occluded frame the estimate is where the
        vehicle is and the last observation is where it was, so matching against the
        observation loses a track the moment it goes behind a bus.

        For what to hand downstream, use report_bbox_xyxy instead -- the two answer
        different questions and the difference is measurable.
        """
        return cxcyah_to_xyxy(self.mean)

    @property
    def report_bbox_xyxy(self) -> tuple[int, int, int, int]:
        """The box to hand downstream: what was seen, or the estimate if nothing was.

        A plate crop is taken from this box, so the question it has to answer is
        "where in THIS frame is the vehicle", and on a frame where the detector
        actually fired, the detector's own box is the better answer than the
        filter's. The filter's estimate is a compromise between the observation and
        an extrapolation whose velocity is still converging, and early in a track
        that compromise sits between the two -- which is to say, on neither.

        Measured, perfect detector, 6 vehicles, 240 truth vehicle-frames, IoU of the
        emitted box against ground truth by track age:

            age 0-2     nothing emitted (min_hits=3, by design)
            age 3-4     0.348
            age 5-9     0.455
            age 10-19   0.681
            age 20+     0.883

        Every one of those numbers should be ~1.0 with a perfect detector, and the
        shortfall is entirely filter lag: 26 of the 51 unmatched vehicle-frames were
        vehicles the tracker was following correctly the whole time and reporting a
        box for that overlapped truth by less than half. A plate crop from an
        early-track box is therefore off by most of a vehicle -- and early track is
        exactly when a vehicle is far away and its plate is smallest, so it is the
        worst place to lose alignment.

        When the track is coasting through a miss there is no observation for this
        frame and the estimate is the only honest answer, which is the case the
        docstring on bbox_xyxy is about.
        """
        if self.time_since_update == 0:
            return self.last_bbox
        return cxcyah_to_xyxy(self.mean)

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.bbox_xyxy
        return x2 - x1

    @property
    def velocity_px_per_frame(self) -> tuple[float, float]:
        """Centre velocity from the filter. Used for direction, never for speed.

        Pixels per frame is not a speed. Converting it to km/h needs the camera's
        calibration and its ground plane, neither of which exists for a grid of
        80,000 heterogeneous cameras, so no speed is ever derived from this. The
        sign is still useful: it tells the emitter which way the vehicle was
        heading, which is a claim that survives having no calibration.
        """
        return float(self.mean[4]), float(self.mean[5])

    @property
    def class_name(self) -> str:
        """The majority class, weighted by the confidence of each vote.

        Ties go to the alphabetically first name so the result cannot depend on
        dictionary insertion order across runs.
        """
        return max(sorted(self.class_votes), key=lambda name: self.class_votes[name])

    @property
    def duration_ms(self) -> int:
        return self.last_pts_ms - self.first_pts_ms

    # ----------------------------------------------------------------- lifecycle

    def predict(self) -> None:
        """Advance the filter one frame. Every track, every frame."""
        self.mean, self.covariance = self._filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        frame_index: int,
        pts_ms: int,
        *,
        min_hits: int,
    ) -> None:
        """Correct with a matched detection and advance the state machine."""
        self.mean, self.covariance = self._filter.update(
            self.mean, self.covariance, xyxy_to_cxcyah(bbox_xyxy)
        )

        self.class_votes[class_name] = self.class_votes.get(class_name, 0.0) + float(
            confidence
        )
        self.confidence = float(confidence)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame_index = frame_index
        self.last_pts_ms = pts_ms
        self.last_bbox = bbox_xyxy

        if self.state == LOST:
            self.reacquisitions += 1
            self.state = CONFIRMED
        elif self.state == TENTATIVE and self.hits >= min_hits:
            self.state = CONFIRMED

    def mark_missed(self, *, max_age: int) -> None:
        """No detection matched this frame.

        A tentative track is deleted on the first miss. It has one or two hits and
        no evidence of being real, and keeping it alive is how a reflection becomes
        a confirmed vehicle. A confirmed track gets the full buffer.
        """
        if self.state == TENTATIVE:
            self.state = REMOVED
        elif self.time_since_update > max_age:
            self.state = REMOVED
        elif self.state == CONFIRMED:
            self.state = LOST

    def gating_distance(
        self, measurements: np.ndarray, *, dims: Sequence[int] = MEASUREMENT_DIMS
    ) -> np.ndarray:
        return self._filter.gating_distance(
            self.mean, self.covariance, measurements, dims=dims
        )

    # -------------------------------------------------------------- predicates

    @property
    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED

    @property
    def is_active(self) -> bool:
        """Reportable this frame: confirmed and matched on this frame.

        A LOST track is not reported. It exists so the identity survives, but
        emitting a sighting for a vehicle nobody can currently see would put a
        position the filter guessed into the database as an observation.
        """
        return self.state == CONFIRMED and self.time_since_update == 0

    @property
    def is_removed(self) -> bool:
        return self.state == REMOVED

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, {self.state}, class={self.class_name}, "
            f"hits={self.hits}, missed={self.time_since_update})"
        )
