"""Scene discontinuity detection -- the Sentinel-specific trap.

Owner's manual section 4.8. Sentinel streams loop with hard cuts, and a loop
boundary is visually indistinguishable from someone physically re-aiming the
camera: every tracked vehicle vanishes, unrelated ones appear instantly.

Skip this and ByteTrack will cheerfully associate a car leaving at the loop end
with a different car entering at the loop start, and the journey built from that
data shows a vehicle that was never there.

Two cheap independent signals are used. PTS discontinuity lives in ai/media/pts.py;
this module is the second one: global histogram delta between consecutive
EMITTED frames. Independent matters -- a re-aimed camera keeps perfect PTS, and
a stream that drops a chunk of time keeps a near-identical scene.

numpy only, no cv2, so it runs anywhere the contracts run.
"""

from typing import Optional

import numpy as np

# Correlation between consecutive emitted frames below which we call it a cut.
#
# Measured on the synthetic road scenes, 64 bins, 160 px long edge:
#     same scene, 100 ms apart, traffic moving      0.99+
#     different daytime road, different seed        0.72 - 0.95
#     unrelated scene (night, indoor)               0.00 - 0.01
#
# 0.70 sits just under the "different but similar scene" band, which makes this
# detector deliberately conservative: it fires on drastically different content
# and not on a camera re-aimed at a comparable-looking road. Loosening it to
# catch that case would put it inside the range that ordinary traffic and
# lighting drift can reach, and a session rotating spuriously destroys tracking
# far more reliably than a missed re-aim does.
#
# KNOWN LIMITATION, and the reason this is one of two signals rather than the
# only one: a Sentinel loop that restarts the SAME clip produces a near-identical
# histogram and is invisible here. PTS catches it, because the timeline jumps
# backwards. Neither signal is sufficient alone; that is the design, not a gap.
DISCONTINUITY_CORRELATION_THRESHOLD = 0.70

# Histogram bins. 64 is coarse enough that sensor noise and a passing truck do
# not move it, fine enough that a genuinely different scene does.
HISTOGRAM_BINS = 64

# Long edge of the downscaled frame used for the histogram. Full resolution
# buys nothing here and costs real time at 10 fps across 30 cameras.
DOWNSCALE_LONG_EDGE = 160


class DiscontinuityDetector:
    """One per session. Compares each emitted frame against the previous one.

    Deliberately fed only emitted frames, not every decoded frame: at the
    100 ms sampling interval, consecutive emitted frames are a fixed 100 ms
    apart on the source timeline, which makes the threshold meaningful. Fed
    every decoded frame, the interval would vary with decode speed and the
    threshold would mean nothing.
    """

    def __init__(
        self,
        *,
        threshold: float = DISCONTINUITY_CORRELATION_THRESHOLD,
        bins: int = HISTOGRAM_BINS,
        long_edge: int = DOWNSCALE_LONG_EDGE,
    ) -> None:
        self.threshold = threshold
        self.bins = bins
        self.long_edge = long_edge
        self._previous: Optional[np.ndarray] = None
        self.last_correlation: Optional[float] = None
        self.detections = 0
        self.comparisons = 0

    def check(self, frame_bgr: np.ndarray) -> tuple[bool, Optional[float]]:
        """Returns (is_discontinuity, correlation).

        correlation is None for the first frame of a session, where there is
        nothing to compare against and no discontinuity can be claimed.
        """
        histogram = self._histogram(frame_bgr)

        if self._previous is None:
            self._previous = histogram
            return False, None

        correlation = _correlate(self._previous, histogram)
        self._previous = histogram
        self.last_correlation = correlation
        self.comparisons += 1

        if correlation < self.threshold:
            self.detections += 1
            return True, correlation
        return False, correlation

    def reset(self) -> None:
        """Called when a session is minted. No cross-session comparison."""
        self._previous = None
        self.last_correlation = None

    def _histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = _to_gray_small(frame_bgr, self.long_edge)
        hist, _ = np.histogram(gray, bins=self.bins, range=(0.0, 256.0))
        total = hist.sum()
        if total == 0:
            return hist.astype(np.float64)
        return hist.astype(np.float64) / float(total)

    def stats(self) -> dict[str, object]:
        return {
            "comparisons": self.comparisons,
            "detections": self.detections,
            "last_correlation": (
                None if self.last_correlation is None else round(self.last_correlation, 4)
            ),
            "threshold": self.threshold,
        }


def _to_gray_small(frame_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    """Downscale by integer striding, then convert to luma.

    Striding rather than area-averaging: it is a few times faster, and for a
    histogram comparison the aliasing it introduces is irrelevant. Both frames
    get the same treatment, which is all the comparison requires.
    """
    arr = np.asarray(frame_bgr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = (
            0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
        )
    elif arr.ndim == 2:
        gray = arr.astype(np.float64)
    else:
        raise ValueError(f"expected HxW or HxWx3, got shape {arr.shape}")

    height, width = gray.shape[:2]
    step = max(1, int(max(height, width) // max(1, long_edge)))
    return gray[::step, ::step]


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two normalized histograms.

    Returns 1.0 when either histogram is constant and they are equal, 0.0 when
    one is constant and they differ. A constant histogram means a uniform frame
    -- a fully black or blown-out image -- where correlation is undefined; the
    equality fallback keeps a run of black frames from being reported as a
    scene change on every frame.
    """
    a_var, b_var = float(np.var(a)), float(np.var(b))
    if a_var == 0.0 or b_var == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
