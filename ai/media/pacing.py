"""Replay pacing for the file-backed sources.

A clip read from disk decodes as fast as the CPU allows, which is what you want
for a benchmark and wrong for a demo -- an operator watching a 30x-speed replay
learns nothing about how the system behaves on real traffic.

Pacing is deliberately optional and off by default. Turning it on when measuring
throughput would make the measurement a report of the pacer's settings rather
than of the pipeline, so config/benchmark.yaml leaves it unset and
config/offline.yaml sets it only when demonstrating.

Note what is paced: emission is delayed to match the SOURCE timeline divided by
the speed factor. The pacer never touches pts_ms. A paced run and an unpaced run
of the same clip produce byte-identical events -- only slower.
"""

import time
from typing import Any, Callable, Optional


class ReplayPacer:
    """Delays emission so a recorded clip advances at a chosen speed.

    speed 1.0 is real time, 5.0 is five times faster, None disables pacing
    entirely. Never sleeps to catch up when the pipeline is already behind: if
    inference takes longer than the pacing interval, the pacer gets out of the
    way rather than compounding the lag.
    """

    def __init__(
        self,
        speed: Optional[float] = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if speed is not None and speed <= 0:
            raise ValueError(f"speed must be > 0 or None, got {speed}")
        self.speed = speed
        self._sleep = sleep
        self._monotonic = monotonic

        self._first_pts_ms: Optional[int] = None
        self._started_at: Optional[float] = None
        self.slept_seconds = 0.0
        self.behind_count = 0

    @property
    def enabled(self) -> bool:
        return self.speed is not None

    def wait_for(self, pts_ms: int) -> float:
        """Sleep until this frame is due. Returns the seconds actually slept."""
        if self.speed is None:
            return 0.0

        now = self._monotonic()
        if self._first_pts_ms is None or self._started_at is None:
            self._first_pts_ms = pts_ms
            self._started_at = now
            return 0.0

        source_elapsed = (pts_ms - self._first_pts_ms) / 1000.0 / self.speed
        real_elapsed = now - self._started_at
        deficit = source_elapsed - real_elapsed

        if deficit <= 0:
            self.behind_count += 1
            return 0.0

        self._sleep(deficit)
        self.slept_seconds += deficit
        return deficit

    def reset(self) -> None:
        """New session: the source timeline restarts, so the anchor must too."""
        self._first_pts_ms = None
        self._started_at = None

    def stats(self) -> dict[str, Any]:
        return {
            "speed": self.speed,
            "slept_seconds": round(self.slept_seconds, 3),
            "frames_behind_schedule": self.behind_count,
        }
