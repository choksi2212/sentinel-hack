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
#
#     same scene, consecutive emitted frames         1.0000
#     same scene, 40 emitted frames apart            0.9983
#     same scene, 59 emitted frames apart            0.9933
#     different road scene (10 seed pairs)           0.30 - 0.93
#     unrelated content (uniform noise)              0.27
#     black frame vs blown-out frame                -0.02
#
# What those numbers do and do not license. The margin against SPURIOUS firing is large and
# that is the number that matters most: a whole scene's worth of traffic turning over moves
# the correlation from 1.0000 to 0.9933, which is still 0.29 clear of the threshold. A session
# rotating on ordinary traffic would destroy tracking on every camera at once, so this is the
# direction the margin has to be generous in, and it is.
#
# The other direction is weaker than it looks. Two visually comparable road scenes measured
# anywhere from 0.30 to 0.93 across ten pairs, and five of the ten fell below 0.70 -- so this
# threshold does NOT reliably distinguish "camera re-aimed at a similar-looking junction" from
# "camera re-aimed at something unrelated". It fires on about half of them. That is
# under-detection of re-aims rather than over-detection of cuts, which is the safe way round:
# a missed re-aim costs the ids in one scene, a spurious rotation costs every id everywhere.
# Anyone tempted to raise this to catch the other half should know they are trading against the
# 0.9933 figure above, and that ordinary lighting drift over a longer window has not been
# measured at all.
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
        # A Pearson correlation, so strictly inside (-1, 1). Checked here rather than at the
        # config layer because this class owns what the number means, and every route to it --
        # YAML, --set, a Python caller -- has to be covered by one check.
        #
        # Both ends are refused for the same reason: they disable the detector while looking
        # like they configure it. At or above 1.0 nothing correlates highly enough, so every
        # emitted frame rotates the session, every track is cut to one frame, and the run emits
        # nothing. At or below -1.0 nothing ever fires and a loop cut goes unnoticed. If the
        # detector is genuinely unwanted, `detect_discontinuity: false` says so honestly.
        #
        # The units trap is named in the message because it is the mistake that actually
        # happened: config/offline.yaml and config/live.yaml both carried 2000, a millisecond
        # figure meant for a PTS jump, and it was silently discarded on the way here.
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(
                f"discontinuity threshold must be a number, got {threshold!r}"
            )
        if not -1.0 < float(threshold) < 1.0:
            raise ValueError(
                f"discontinuity threshold {threshold!r} is out of range: it is a Pearson "
                "correlation between consecutive frame histograms and must be strictly "
                "inside (-1, 1). A value of 1.0 or more rotates the session on every frame "
                "and the run emits nothing; -1.0 or less never fires, so use "
                "detect_discontinuity=False instead. If this was a duration in milliseconds, "
                f"it belongs in ai/media/pts.py, not here (default "
                f"{DISCONTINUITY_CORRELATION_THRESHOLD})"
            )
        self.threshold = float(threshold)
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

    Correlation is undefined when either histogram has zero variance, so that case falls back
    to an equality test: 1.0 if the two are identical, 0.0 otherwise.

    Zero variance means every one of the 64 bins holds an exactly equal share -- a frame whose
    intensities are perfectly uniformly distributed. It is NOT the fully-black or blown-out
    frame that an earlier version of this comment claimed: a black frame puts every pixel in
    bin 0, which is the highest-variance histogram available (measured 0.0154), and even a
    clean 0-255 gradient only reaches 1.7e-05. So this branch is effectively unreachable from
    real footage and exists to keep the arithmetic total rather than to handle a known case.

    A run of black frames stays quiet for a different reason than that earlier comment gave:
    identical histograms correlate at exactly 1.0 through the ordinary path, no fallback
    involved. Black against blown-out measures -0.02 and fires, which is correct -- the camera
    has stopped showing the scene either way.
    """
    a_var, b_var = float(np.var(a)), float(np.var(b))
    if a_var == 0.0 or b_var == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
