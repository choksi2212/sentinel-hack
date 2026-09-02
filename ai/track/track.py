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

At min_hits <= 1 the TENTATIVE box is skipped and a track is born CONFIRMED, which
is the only reading of "require one hit" that is not off by one -- see __init__.
Nothing ships that way; it exists so that turning the requirement off to isolate a
bug turns it off rather than halving it.

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
        min_hits: int = 2,
        kalman: Optional[KalmanBoxFilter] = None,
    ) -> None:
        self.track_id = track_id
        self._filter = kalman or shared_filter()
        self.mean, self.covariance = self._filter.initiate(xyxy_to_cxcyah(bbox_xyxy))

        # A new track has one hit, so min_hits=1 is already satisfied and confirming
        # here is the only way to honour it. Promotion otherwise happens in update(),
        # which a track never passes through on the frame it was created -- so before
        # this line min_hits=1 and min_hits=2 produced identical output, and the
        # setting that reads as "report immediately, no persistence requirement"
        # quietly meant "wait one frame". No shipped config trips it (DEFAULT_MIN_HITS
        # is 3, and 3 is the number the night-glare argument above is about), which is
        # exactly why it went unnoticed: the only caller who would ever see it is
        # someone deliberately turning the requirement off to isolate a bug, and they
        # would have concluded the tracker was dropping their first frame.
        self.state = CONFIRMED if 1 >= int(min_hits) else TENTATIVE
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

        Measured, perfect detector, 6 vehicles, 240 truth vehicle-frames, IoU against
        ground truth by track age, for this property and for bbox_xyxy beside it:

            age        n    report_bbox_xyxy    bbox_xyxy
            0-2        -    nothing emitted (min_hits=3, by design)
            3-4       12               1.000        0.475
            5-9       30               1.000        0.465
            10-19     60               1.000        0.693
            20+      115               1.000        0.900

        The left column is 1.000 by construction rather than by luck, and saying so is
        the point: this property returns last_bbox whenever time_since_update == 0, and
        is_active -- the condition for being emitted at all -- is exactly that plus
        CONFIRMED. So every box the tracker emits for an active track is the detector's
        own box, unrounded, and against a perfect detector that is the truth box. All
        217 emitted boxes scored 1.0000, min and max.

        The right column is what the crop would be worth if this property did not
        exist. Filter lag is real and it is worst exactly where it hurts most: at age
        3-4 the estimate overlaps truth by 0.475, so a plate crop taken from it misses
        more than half the vehicle -- and early track is when a vehicle is furthest away
        and its plate smallest. The lag decays as the velocity estimate converges, but
        it never fully closes; even past age 20 the estimate is at 0.900.

        An earlier version of this docstring printed the right column's numbers under
        the left column's heading and blamed "filter lag" for degrading the emitted box.
        That inverted the finding. The lag is what this property routes around.

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
