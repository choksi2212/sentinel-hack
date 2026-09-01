"""Reconnect backoff. Owner's manual section 4.5.

    BACKOFF_BASE_MS, BACKOFF_MAX_MS = 500, 30_000
    delay = min(BACKOFF_BASE_MS * (2 ** attempt), BACKOFF_MAX_MS)
    delay *= (0.5 + random.random())          # 50%-150% jitter

Jitter is not decoration. Thirty workers losing a shared upstream will otherwise
retry at identical instants and convert a brief outage into a self-inflicted
denial of service against the Sentinel grid -- from inside the hackathon, on
infrastructure the organisers are lending us.

Reset the attempt counter after sustained healthy reads, not on the first good
frame. A stream that hands back one frame and dies again is not recovered, and
resetting on that frame turns exponential backoff into a hot loop.
"""

import random
import time
from typing import Callable, Optional

BACKOFF_BASE_MS = 500
BACKOFF_MAX_MS = 30_000

# Sustained healthy reading required before the attempt counter resets.
HEALTHY_RESET_SECONDS = 30.0

# Attempt index above which the delay is capped anyway. Prevents 2 ** attempt
# from growing into a large int for no reason on a camera that is simply gone.
_MAX_SHIFT = 16


class ReconnectPolicy:
    """Exponential backoff with jitter, plus the healthy-reset rule.

    sleep is injectable so tests do not actually wait 30 seconds, and so the
    worker can substitute an interruptible sleep that responds to shutdown.
    """

    def __init__(
        self,
        *,
        base_ms: int = BACKOFF_BASE_MS,
        max_ms: int = BACKOFF_MAX_MS,
        healthy_reset_seconds: float = HEALTHY_RESET_SECONDS,
        rng: Optional[random.Random] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_ms = base_ms
        self.max_ms = max_ms
        self.healthy_reset_seconds = healthy_reset_seconds
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._monotonic = monotonic

        self.attempt = 0
        self.total_reconnects = 0
        self._healthy_since: Optional[float] = None

    def next_delay_ms(self) -> int:
        """Delay for the current attempt, with jitter applied. Does not sleep."""
        shift = min(self.attempt, _MAX_SHIFT)
        delay = min(self.base_ms * (2 ** shift), self.max_ms)
        delay *= 0.5 + self._rng.random()
        return int(delay)

    def wait(self) -> int:
        """Sleep for the next delay, then increment the attempt counter.

        Returns the delay actually waited, in milliseconds, so the caller can
        log it. A reconnect with no logged delay is indistinguishable from a hot
        loop when reading the output afterwards.
        """
        delay_ms = self.next_delay_ms()
        self.attempt += 1
        self.total_reconnects += 1
        self._sleep(delay_ms / 1000.0)
        return delay_ms

    def note_healthy_read(self) -> None:
        """Call on every successful frame.

        The counter resets only once reads have been healthy for
        healthy_reset_seconds continuously.
        """
        now = self._monotonic()
        if self._healthy_since is None:
            self._healthy_since = now
            return
        if self.attempt and now - self._healthy_since >= self.healthy_reset_seconds:
            self.attempt = 0

    def note_failure(self) -> None:
        """Call when a read fails, so the healthy window restarts."""
        self._healthy_since = None

    def stats(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "total_reconnects": self.total_reconnects,
            "next_delay_ms": self.next_delay_ms(),
        }
