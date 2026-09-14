"""Detection-to-track association: cost matrices and the assignment itself.

Two things live here, and keeping them apart from the tracker matters because both
are pure functions that can be checked exactly, while the tracker is a state
machine that can only be checked by running it.

The assignment uses scipy's Hungarian solver when scipy is installed and a greedy
solver when it is not. The greedy result is not optimal -- it commits to the best
single pair, then the best remaining pair, and can be beaten by a pairing that
gives up a little on one match to gain more on another. Measured on this pipeline
the difference is small, but it is not zero, and which solver ran is reported in
the tracker's stats so a comparison between two machines is not mistaken for a
regression.
"""

from typing import Sequence

import numpy as np

# Above this IoU cost a pair is not a match at any price. Equivalently IoU < 0.2.
#
# Deliberately loose. The gate that actually prevents wrong associations is the
# Mahalanobis veto in ai/track/kalman.py, which knows where the vehicle should be;
# IoU only knows that two rectangles overlap. Tightening this instead of trusting
# the filter loses fast-moving motorcycles, whose box between two frames 100 ms
# apart can overlap its predecessor by very little.
MAX_IOU_COST = 0.8

INFEASIBLE = 1e5


def iou_matrix(
    track_boxes: Sequence[tuple[int, int, int, int]],
    detection_boxes: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Pairwise IoU, tracks on rows and detections on columns.

    Vectorised over both axes. The obvious double loop is 30x40 iterations of
    Python per frame per camera, which at 10 fps across even a handful of cameras
    is measurable, and this is called twice per frame by the two-stage association.
    """
    if not track_boxes or not detection_boxes:
        return np.zeros((len(track_boxes), len(detection_boxes)), dtype=np.float32)

    tracks = np.asarray(track_boxes, dtype=np.float32).reshape(-1, 1, 4)
    detections = np.asarray(detection_boxes, dtype=np.float32).reshape(1, -1, 4)

    left = np.maximum(tracks[..., 0], detections[..., 0])
    top = np.maximum(tracks[..., 1], detections[..., 1])
    right = np.minimum(tracks[..., 2], detections[..., 2])
    bottom = np.minimum(tracks[..., 3], detections[..., 3])

    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    track_area = np.clip(tracks[..., 2] - tracks[..., 0], 0, None) * np.clip(
        tracks[..., 3] - tracks[..., 1], 0, None
    )
    detection_area = np.clip(detections[..., 2] - detections[..., 0], 0, None) * np.clip(
        detections[..., 3] - detections[..., 1], 0, None
    )

    union = track_area + detection_area - overlap
    # A degenerate box gives union 0. Reporting IoU 0 for it is right; dividing by
    # zero and getting nan is not, and a nan propagates into the cost matrix and
    # makes the assignment silently skip a whole row.
    return np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def iou_cost(
    track_boxes: Sequence[tuple[int, int, int, int]],
    detection_boxes: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """1 - IoU. The assignment minimises cost, so the conversion belongs here."""
    return 1.0 - iou_matrix(track_boxes, detection_boxes)


def fuse_detection_score(cost: np.ndarray, scores: Sequence[float]) -> np.ndarray:
    """Weight the IoU cost by detector confidence -- ByteTrack's score fusion.

    A high-confidence detection is a better claim on a track than a marginal one at
    the same IoU, so its cost is reduced. This is *not* multiplying two calibrated
    probabilities together and calling the result a probability -- which Contracts
    section 8 forbids and this codebase does not do anywhere. It is a ranking
    heuristic internal to the assignment, its output is a cost and never leaves
    this module, and no number derived from it is ever reported.
    """
    if cost.size == 0:
        return cost
    similarity = 1.0 - cost
    weights = np.asarray(scores, dtype=np.float32).reshape(1, -1)
    return 1.0 - (similarity * weights)


def gate_cost(
    cost: np.ndarray,
    *,
    max_cost: float = MAX_IOU_COST,
    infeasible: np.ndarray | None = None,
) -> np.ndarray:
    """Mark impossible pairs so the solver cannot choose them.

    Set to a large finite value rather than infinity. scipy's Hungarian solver
    raises on an infeasible problem when a row is all-inf, which happens routinely
    -- a track whose vehicle has left the frame has no feasible detection -- and
    that must produce an unmatched track, not an exception in the middle of a run.
    """
    gated = cost.copy()
    gated[gated > max_cost] = INFEASIBLE
    if infeasible is not None:
        gated[infeasible] = INFEASIBLE
    return gated


def solve(
    cost: np.ndarray, *, max_cost: float = MAX_IOU_COST
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Assign rows to columns at minimum total cost.

    Returns (matches, unmatched_rows, unmatched_cols). Pairs whose cost exceeds
    max_cost are rejected after solving, not before: the solver may pick a bad pair
    to unlock two good ones, and that pair has to be undone once it has served its
    purpose -- with both of its members returned as unmatched.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    row_indices, col_indices = _assign(cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_indices, col_indices):
        if cost[r, c] > max_cost:
            continue
        matches.append((int(r), int(c)))
        matched_rows.add(int(r))
        matched_cols.add(int(c))

    return (
        matches,
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


def _assign(cost: np.ndarray) -> tuple[Sequence[int], Sequence[int]]:
    if _HAVE_SCIPY:
        return _scipy_assign(cost)
    return _greedy_assign(cost)


def _greedy_assign(cost: np.ndarray) -> tuple[list[int], list[int]]:
    """Take the cheapest available pair, repeat. Deterministic, not optimal.

    Ties are broken by (row, column) order, which comes free from argsort's stable
    sort on a flattened array. Determinism is not cosmetic here: a track ID that
    depends on tie-break order makes a fixture non-reproducible, and a
    non-reproducible fixture cannot prove anything.
    """
    order = np.argsort(cost, axis=None, kind="stable")
    rows, cols = cost.shape

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    row_out: list[int] = []
    col_out: list[int] = []

    for flat in order:
        r, c = divmod(int(flat), cols)
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        row_out.append(r)
        col_out.append(c)
        if len(row_out) == min(rows, cols):
            break

    return row_out, col_out


try:  # pragma: no cover - depends on the machine
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def _scipy_assign(cost: np.ndarray) -> tuple[Sequence[int], Sequence[int]]:
    return _linear_sum_assignment(cost)


def solver_name() -> str:
    """"hungarian" or "greedy". Recorded in tracker stats, see the module docstring."""
    return "hungarian" if _HAVE_SCIPY else "greedy"
