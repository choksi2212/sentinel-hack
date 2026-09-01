"""PTS-driven sampling and the depth-1 latest-frame buffer.

Contracts section 2.2 -- LOCKED:

    TARGET_INTERVAL_MS = 100        # ~10 inferences/sec/camera
    Emit a frame to AI only when  pts_ms - last_emitted_pts_ms >= TARGET_INTERVAL_MS
    Buffer depth = 1 (latest frame wins; drop the old one)

Freshness beats completeness. A live operator needs the vehicle passing now, not
a perfect analysis of the one that passed 40 seconds ago. An unbounded queue
turns a 200 ms processing deficit into 2 s of lag per second of runtime -- five
minutes in, the "live" view is ten minutes stale, and shortly after that the
process is OOM-killed. On demo day.
"""

import threading
from typing import Any, Optional

TARGET_INTERVAL_MS = 100


class PtsSampler:
    """Decides which frames reach the AI, on the source timeline.

    On the SOURCE timeline, not the wall clock. Sampling on arrival time gives a
    5x replay ten times fewer inferences per video second than a live stream,
    which makes offline accuracy measurements incomparable with live ones -- and
    quietly invalidates every benchmark taken during accelerated replay.
    """

    def __init__(self, target_interval_ms: int = TARGET_INTERVAL_MS) -> None:
        if target_interval_ms < 0:
            raise ValueError(f"target_interval_ms must be >= 0, got {target_interval_ms}")
        self.target_interval_ms = target_interval_ms
        self.last_emitted_pts_ms: Optional[int] = None
        self.emitted = 0
        self.skipped = 0

    def should_emit(self, pts_ms: int) -> bool:
        if self.last_emitted_pts_ms is None:
            return True
        return pts_ms - self.last_emitted_pts_ms >= self.target_interval_ms

    def note_emitted(self, pts_ms: int) -> None:
        self.last_emitted_pts_ms = pts_ms
        self.emitted += 1

    def note_skipped(self) -> None:
        self.skipped += 1

    def reset(self) -> None:
        """New session: the source timeline restarts, so the gate must too."""
        self.last_emitted_pts_ms = None

    def stats(self) -> dict[str, Any]:
        seen = self.emitted + self.skipped
        return {
            "emitted": self.emitted,
            "skipped": self.skipped,
            "decoded": seen,
            "emit_rate": round(self.emitted / seen, 4) if seen else None,
            "target_interval_ms": self.target_interval_ms,
        }


class LatestFrameBuffer:
    """Depth-1 buffer. Latest frame wins; the old one is dropped and counted.

    Used by the live sources, where a reader thread decodes as fast as the
    network delivers and the consumer takes whatever is newest. Not used by the
    file sources: reading a clip from disk has no real-time pressure, and
    dropping frames there would make an offline benchmark depend on how busy the
    laptop was, which is not a property of the model.

    The drop count is not incidental. It is how you tell a flaky camera from a
    GPU that cannot keep up, and it belongs in every performance claim made
    about this pipeline.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Optional[Any] = None
        self._available = threading.Event()
        self.dropped = 0
        self.accepted = 0
        self.consumed = 0

    def put(self, item: Any) -> None:
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.accepted += 1
            self._available.set()

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Take the newest item, or None on timeout."""
        if not self._available.wait(timeout=timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._available.clear()
            if item is not None:
                self.consumed += 1
            return item

    def clear(self) -> None:
        with self._lock:
            self._item = None
            self._available.clear()

    def wake(self) -> None:
        """Release a blocked get() without delivering an item, for shutdown."""
        self._available.set()

    @property
    def drop_rate(self) -> Optional[float]:
        return round(self.dropped / self.accepted, 4) if self.accepted else None

    def stats(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "consumed": self.consumed,
            "dropped": self.dropped,
            "drop_rate": self.drop_rate,
        }
