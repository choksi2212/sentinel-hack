"""PTS validation. Owner's manual section 4.7.

    pts_ms decreases                  discontinuity -> new session
    jump forward > 5,000 ms           discontinuity -> new session
    0 / unavailable for N frames      synthetic monotonic clock, warn,
                                      mark session pts_unreliable
    identical across frames           stalled decoder -> force reconnect

The reason this module exists rather than a couple of inline comparisons: a
pts_unreliable session must never substantiate a latency claim, and that is only
enforceable if something tracks the flag. Once the source silently patches
around a broken clock, every temporal number downstream is fiction that looks
like measurement.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Contracts section 2.2.
TARGET_INTERVAL_MS = 100

# A forward jump larger than this is not a vehicle moving, it is a different
# scene. Sentinel streams loop with hard cuts, so this fires in normal
# operation rather than only in failure.
MAX_FORWARD_JUMP_MS = 5_000

# Consecutive frames reporting no usable PTS before we give up and run a
# synthetic clock. Half a second at 10 fps -- long enough not to trip on a
# single bad packet.
UNAVAILABLE_FRAME_THRESHOLD = 5

# Consecutive frames reporting the SAME non-zero PTS before we call the decoder
# stalled. One second of identical timestamps is not a slow stream, it is a
# stream that has stopped advancing while still handing back buffers.
STALL_FRAME_THRESHOLD = 10

# Step used by the synthetic clock when the source's own clock is unusable.
# 25 fps is the common case on these feeds. The value only has to be monotonic
# and roughly right, because any session using it is already barred from
# supporting a latency claim.
SYNTHETIC_STEP_MS = 40


class PtsAction(str, Enum):
    OK = "ok"
    SKIP = "skip"                    # unusable timestamp; drop the frame, keep the stream
    NEW_SESSION = "new_session"      # hard discontinuity in the source timeline
    FORCE_RECONNECT = "reconnect"    # decoder stalled; the transport is the problem


@dataclass(frozen=True)
class PtsVerdict:
    action: PtsAction
    pts_ms: int
    reason: Optional[str] = None
    pts_unreliable: bool = False

    @property
    def ok(self) -> bool:
        return self.action is PtsAction.OK


class PtsValidator:
    """One per session. Reset (or rebuild) whenever a session is minted.

    Holds the last accepted PTS and the counters for the two degenerate cases.
    """

    def __init__(
        self,
        *,
        max_forward_jump_ms: int = MAX_FORWARD_JUMP_MS,
        unavailable_threshold: int = UNAVAILABLE_FRAME_THRESHOLD,
        stall_threshold: int = STALL_FRAME_THRESHOLD,
        synthetic_step_ms: int = SYNTHETIC_STEP_MS,
    ) -> None:
        self.max_forward_jump_ms = max_forward_jump_ms
        self.unavailable_threshold = unavailable_threshold
        self.stall_threshold = stall_threshold
        self.synthetic_step_ms = synthetic_step_ms

        self.last_pts_ms: Optional[int] = None
        self.pts_unreliable = False

        self._unavailable_run = 0
        self._identical_run = 0
        self._synthetic_pts_ms = 0

        self.counts: dict[str, int] = {
            "accepted": 0,
            "unavailable": 0,
            "synthesized": 0,
            "backwards": 0,
            "forward_jump": 0,
            "stalled": 0,
        }

    def observe(self, raw_pts_ms: Optional[float]) -> PtsVerdict:
        """Validate one raw PTS reading and return what the source should do.

        raw_pts_ms is what the decoder reported. None, 0 and negatives all mean
        "no usable value" -- OpenCV returns 0.0 for streams that do not carry
        POS_MSEC, and returning a real 0 only ever happens on the very first
        frame, which the caller has already accounted for via last_pts_ms.
        """
        usable = raw_pts_ms is not None and raw_pts_ms > 0
        if self.last_pts_ms is None and raw_pts_ms is not None and raw_pts_ms == 0:
            usable = True  # genuine frame 0 of a file

        if not usable:
            return self._handle_unavailable()

        pts = int(raw_pts_ms)  # type: ignore[arg-type]
        self._unavailable_run = 0

        if self.last_pts_ms is None:
            return self._accept(pts)

        if pts == self.last_pts_ms:
            self._identical_run += 1
            if self._identical_run >= self.stall_threshold:
                self.counts["stalled"] += 1
                return self._verdict(
                    PtsAction.FORCE_RECONNECT,
                    pts,
                    f"decoder stalled: PTS {pts} unchanged for "
                    f"{self._identical_run} frames",
                )
            # Not yet a stall. Skip the frame rather than emit a duplicate
            # timestamp, which would make two frames indistinguishable in time.
            return self._verdict(PtsAction.SKIP, pts, "duplicate pts, frame skipped")

        self._identical_run = 0

        if pts < self.last_pts_ms:
            self.counts["backwards"] += 1
            return self._verdict(
                PtsAction.NEW_SESSION,
                pts,
                f"PTS went backwards: {self.last_pts_ms} -> {pts}",
            )

        delta = pts - self.last_pts_ms
        if delta > self.max_forward_jump_ms:
            self.counts["forward_jump"] += 1
            return self._verdict(
                PtsAction.NEW_SESSION,
                pts,
                f"PTS jumped forward {delta} ms (> {self.max_forward_jump_ms} ms)",
            )

        return self._accept(pts)

    def _handle_unavailable(self) -> PtsVerdict:
        self._unavailable_run += 1
        self.counts["unavailable"] += 1

        if self._unavailable_run < self.unavailable_threshold:
            # Below threshold: skip. A couple of bad readings is not worth
            # abandoning a working clock for. Skipping and not emitting matters
            # here -- the alternative is stamping the frame with the previous
            # frame's timestamp, which is a fabricated measurement.
            return self._verdict(
                PtsAction.SKIP,
                self.last_pts_ms or 0,
                f"pts unavailable ({self._unavailable_run}), frame skipped",
            )

        if not self.pts_unreliable:
            self.pts_unreliable = True
            self._synthetic_pts_ms = self.last_pts_ms or 0

        self._synthetic_pts_ms += self.synthetic_step_ms
        self.last_pts_ms = self._synthetic_pts_ms
        self.counts["synthesized"] += 1
        self.counts["accepted"] += 1
        return self._verdict(
            PtsAction.OK,
            self._synthetic_pts_ms,
            "synthetic monotonic clock; session is pts_unreliable",
        )

    def _accept(self, pts: int) -> PtsVerdict:
        self.last_pts_ms = pts
        self.counts["accepted"] += 1
        return self._verdict(PtsAction.OK, pts)

    def _verdict(
        self,
        action: PtsAction,
        pts_ms: int,
        reason: Optional[str] = None,
    ) -> PtsVerdict:
        """Stamp every verdict with the session's reliability, not just the accepting ones.

        The authoritative flag is this validator's, and ai/media/base.py reads it from there.
        But a rejection verdict reporting pts_unreliable=False while the session is running a
        synthetic clock is a field that lies, and the one failure this module exists to
        prevent is a timing number that looks measured. Stamping in one place means a verdict
        site added later cannot forget to.
        """
        return PtsVerdict(action, pts_ms, reason, pts_unreliable=self.pts_unreliable)

    def stats(self) -> dict[str, object]:
        return {
            "last_pts_ms": self.last_pts_ms,
            "pts_unreliable": self.pts_unreliable,
            **self.counts,
        }
