"""Constant-velocity Kalman filter on (cx, cy, aspect, height).

The state space is SORT's, as adopted by ByteTrack: centre, aspect ratio and
height, each with a velocity. Not (x1, y1, x2, y2) -- the corners are strongly
correlated, so a diagonal covariance over them models a box that can shear, and
the filter spends its effort on a degree of freedom no vehicle has.

Height rather than area is deliberate and is ByteTrack's change from SORT. A
vehicle's apparent height is close to linear in its distance from the camera,
while area is quadratic, so the process noise scales sensibly with height and
badly with area. On an elevated junction camera -- which is every camera in this
grid -- a vehicle crosses most of the depth range in a single track, and the
difference shows up as lost tracks in the far half of the frame.

Noise parameters are the published DeepSORT/ByteTrack values, kept rather than
retuned. They are expressed as fractions of the box height, so they adapt to
scale on their own: a 30 px vehicle at the far end of the junction gets a
proportionally tighter gate than a 300 px bus in the foreground, which is the
behaviour wanted and the reason absolute pixel gates were not used.

numpy only. filterpy is not installed and is not needed for a four-dimensional
observation.
"""

from typing import Optional, Sequence

import numpy as np

# Fractions of box height. Position noise is eight times the velocity noise:
# a vehicle's position is observed directly and its velocity is inferred, so the
# filter should trust the measurement more than its own extrapolation.
STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160

# Aspect ratio is treated as near-constant. A car does not change shape; apparent
# aspect drifts only through perspective and box jitter, so its noise is a small
# constant instead of a fraction of height.
ASPECT_PROCESS_NOISE = 1e-2
ASPECT_VELOCITY_NOISE = 1e-5
ASPECT_MEASUREMENT_NOISE = 1e-1

# Indices into the measurement vector (cx, cy, a, h), for gating_distance.
#
# MEASUREMENT_DIMS is the whole observation and is what the filter updates on.
# The other two are gate subsets, and which one a caller picks is a real decision
# rather than a preference -- see gating_distance and ai/track/bytetrack.py.
MEASUREMENT_DIMS: tuple[int, ...] = (0, 1, 2, 3)
POSITION_DIMS: tuple[int, ...] = (0, 1)
POSITION_SIZE_DIMS: tuple[int, ...] = (0, 1, 3)


class KalmanBoxFilter:
    """Predict/update for one track's box. State is 8-dimensional.

    Stateless with respect to any particular track: the caller owns (mean, cov)
    and passes it in. That keeps the filter testable in isolation and means a
    track can be copied, serialised or rolled back without the filter knowing.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # Constant velocity: position += velocity * dt.
        self._motion = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = dt

        # Observe position only. Velocity is never measured, which is the whole
        # reason for the filter.
        self._update = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from one detection.

        Initial velocity is zero with a deliberately large variance. One box gives
        no information about motion, and pretending otherwise makes the first
        prediction confidently wrong -- which on a fast-moving motorcycle is
        enough to miss the second frame and start a second track.
        """
        mean = np.concatenate([measurement, np.zeros(4)])
        height = measurement[3]
        std = np.array(
            [
                2 * STD_WEIGHT_POSITION * height,
                2 * STD_WEIGHT_POSITION * height,
                ASPECT_PROCESS_NOISE,
                2 * STD_WEIGHT_POSITION * height,
                10 * STD_WEIGHT_VELOCITY * height,
                10 * STD_WEIGHT_VELOCITY * height,
                ASPECT_VELOCITY_NOISE,
                10 * STD_WEIGHT_VELOCITY * height,
            ]
        )
        return mean, np.diag(np.square(std))

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Called on every track every frame, matched or not.

        Predicting unmatched tracks is what lets a vehicle survive an occlusion:
        it keeps moving behind the bus, and when it reappears the predicted box is
        near where it actually is rather than where it was last seen.
        """
        height = mean[3]
        std_pos = np.array(
            [
                STD_WEIGHT_POSITION * height,
                STD_WEIGHT_POSITION * height,
                ASPECT_PROCESS_NOISE,
                STD_WEIGHT_POSITION * height,
            ]
        )
        std_vel = np.array(
            [
                STD_WEIGHT_VELOCITY * height,
                STD_WEIGHT_VELOCITY * height,
                ASPECT_VELOCITY_NOISE,
                STD_WEIGHT_VELOCITY * height,
            ]
        )
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """State space -> measurement space, with observation noise added."""
        height = mean[3]
        std = np.array(
            [
                STD_WEIGHT_POSITION * height,
                STD_WEIGHT_POSITION * height,
                ASPECT_MEASUREMENT_NOISE,
                STD_WEIGHT_POSITION * height,
            ]
        )
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update @ mean
        projected_cov = self._update @ covariance @ self._update.T
        return projected_mean, projected_cov + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with an observed box.

        Solved through a Cholesky factorisation rather than by inverting the
        projected covariance. Explicitly inverting a near-singular matrix -- which
        happens when a track has been stationary for a while and its uncertainty
        has collapsed -- produces a gain full of enormous values and the track
        teleports.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol, lower = _cho_factor(projected_cov)
        kalman_gain = _cho_solve(
            chol, lower, (covariance @ self._update.T).T
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        *,
        dims: Sequence[int] = MEASUREMENT_DIMS,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from this track to each measurement.

        Used to veto associations that IoU would happily accept. Two vehicles in
        adjacent lanes can overlap in IoU while being nowhere near each other in
        the filter's terms, and the veto is what stops the plates being swapped
        between them -- an ID switch that produces two wrong journeys, not one.

        `dims` selects which of (cx, cy, a, h) the distance is computed over, and
        the caller's threshold has to be the chi-squared value for len(dims) degrees
        of freedom -- not the 4-DOF value. Dropping a dimension and keeping the
        4-DOF threshold silently loosens the gate, which is the failure mode that
        looks like it works. Use CHI2_INV95[len(dims)]; ai/track/bytetrack.py derives
        GATING_THRESHOLD from GATING_DIMS for exactly this reason.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        index = list(dims)
        if index != list(MEASUREMENT_DIMS):
            projected_mean = projected_mean[index]
            projected_cov = projected_cov[np.ix_(index, index)]
            measurements = measurements[:, index]

        chol, lower = _cho_factor(projected_cov)
        delta = (measurements - projected_mean).T
        z = _solve_triangular(chol, delta, lower=lower)
        return np.sum(z * z, axis=0)


# 95% confidence interval of the chi-squared distribution, by degrees of freedom.
# Index 4 (a full four-dimensional observation) is the one used. Verbatim from the
# DeepSORT reference so that a comparison against published tracker numbers is
# comparing the same gate.
CHI2_INV95: dict[int, float] = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

# 99% of the same distribution, and the one the tracker actually gates on.
#
# The reason is arithmetic, not taste. A 95% gate rejects 5% of CORRECT pairs by
# definition -- that is what the confidence level means -- and in DeepSORT that is
# affordable because its matching cascade gives a track several chances per frame.
# Here a veto is unrecoverable: a tentative track is dropped on its first miss, and
# a confirmed track that loses one frame fragments. Measured on the synthetic
# fixture, of 170 track-detection pairs that stage 1 would have accepted on IoU,
# the 95% gate vetoed 11 -- 6.5%, almost exactly the 5% the level promises -- and
# those 11 cost 4 fragments and 0.037 recall while preventing zero ID switches.
#
# 99% keeps the gate doing the job it exists for. The pairs it is meant to stop are
# nowhere near the threshold: of 1217 vetoes in the same run, 1216 were pairs whose
# boxes overlapped by less than half, and those sit at squared distances in the
# hundreds. Moving the line from 7.81 to 11.35 does not let any of them through.
CHI2_INV99: dict[int, float] = {
    1: 6.6349,
    2: 9.2103,
    3: 11.3449,
    4: 13.2767,
    5: 15.0863,
    6: 16.8119,
    7: 18.4753,
    8: 20.0902,
    9: 21.6660,
}


def xyxy_to_cxcyah(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Corner box -> the filter's measurement vector.

    Guards against a zero height. A detector can return a degenerate box on a
    frame edge, and dividing by its height gives an infinite aspect that poisons
    the covariance for the rest of the track's life.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return np.array([x1 + width / 2, y1 + height / 2, width / height, height])


def cxcyah_to_xyxy(state: np.ndarray) -> tuple[int, int, int, int]:
    """The filter's state -> an integer corner box.

    Aspect is clamped to a plausible range before use. An unclamped filter can
    drift to a negative or absurd aspect during a long occlusion, and the box it
    then reports spans the frame and matches everything.
    """
    cx, cy, aspect, height = (float(v) for v in state[:4])
    height = max(1.0, height)
    aspect = min(max(aspect, 0.05), 20.0)
    width = aspect * height
    return (
        int(round(cx - width / 2)),
        int(round(cy - height / 2)),
        int(round(cx + width / 2)),
        int(round(cy + height / 2)),
    )


# ---------------------------------------------------------------- linear algebra
#
# scipy.linalg is used when present and a numpy path is kept for when it is not.
# The tracker is the one stage that must run everywhere -- it is how the pipeline
# is tested without weights -- so a hard scipy dependency here would make an
# import failure in a helper module break the whole test suite.

try:  # pragma: no cover - exercised by whichever branch the machine has
    from scipy.linalg import cho_factor as _scipy_cho_factor
    from scipy.linalg import cho_solve as _scipy_cho_solve
    from scipy.linalg import solve_triangular as _scipy_solve_triangular

    _HAVE_SCIPY_LINALG = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY_LINALG = False


def _cho_factor(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    if _HAVE_SCIPY_LINALG:
        return _scipy_cho_factor(matrix, lower=True, check_finite=False)
    # np.linalg.cholesky returns the lower factor directly. A tiny ridge is added
    # because a projected covariance can be numerically indefinite by a few ulp
    # after enough predict steps, and cholesky then raises on a matrix that is
    # positive definite in every sense that matters.
    ridge = 1e-9 * np.eye(matrix.shape[0])
    return np.linalg.cholesky(matrix + ridge), True


def _cho_solve(chol: np.ndarray, lower: bool, rhs: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY_LINALG:
        return _scipy_cho_solve((chol, lower), rhs, check_finite=False)
    intermediate = _solve_triangular(chol, rhs, lower=True)
    return _solve_triangular(chol.T, intermediate, lower=False)


def _solve_triangular(
    matrix: np.ndarray, rhs: np.ndarray, *, lower: bool
) -> np.ndarray:
    if _HAVE_SCIPY_LINALG:
        return _scipy_solve_triangular(
            matrix, rhs, lower=lower, check_finite=False, overwrite_b=False
        )
    return _numpy_solve_triangular(matrix, rhs, lower=lower)


def _numpy_solve_triangular(
    matrix: np.ndarray, rhs: np.ndarray, *, lower: bool
) -> np.ndarray:
    """Forward/back substitution. Only reached when scipy is absent."""
    n = matrix.shape[0]
    single_column = rhs.ndim == 1
    b = rhs.reshape(n, -1).astype(np.float64, copy=True)

    order = range(n) if lower else range(n - 1, -1, -1)
    for i in order:
        if lower:
            b[i] -= matrix[i, :i] @ b[:i]
        else:
            b[i] -= matrix[i, i + 1 :] @ b[i + 1 :]
        b[i] /= matrix[i, i]

    return b.ravel() if single_column else b


def solver_name() -> str:
    """Which linear-algebra path is live. Reported in tracker stats.

    Worth recording: the two paths are numerically equivalent but not bit-identical,
    so a track ID that differs between two machines is explained by this rather
    than by a bug.
    """
    return "scipy" if _HAVE_SCIPY_LINALG else "numpy"


_SHARED_FILTER: Optional[KalmanBoxFilter] = None


def shared_filter() -> KalmanBoxFilter:
    """One filter instance for every track.

    It holds no per-track state -- only the two constant matrices -- so sharing it
    saves rebuilding those per track without introducing any coupling.
    """
    global _SHARED_FILTER
    if _SHARED_FILTER is None:
        _SHARED_FILTER = KalmanBoxFilter()
    return _SHARED_FILTER
