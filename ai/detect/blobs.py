"""Connected-component labelling in numpy. No cv2, no scipy.

This exists so the detection stage has a working backend on a machine with
nothing but numpy installed -- which is the actual state of the development
environment today, and will be the state of any reviewer's machine that has not
run the full install. The pipeline must be runnable end to end against
SyntheticReplaySource before RF-DETR weights are downloaded, or nothing
downstream can be tested and D1 blocks two other people.

It is not a substitute for a detector on real footage and is not presented as
one. It finds contiguous regions in a boolean mask. On the synthetic scenes,
where vehicles are saturated blocks on a grey road, that is enough to produce
correct boxes and exercise every stage below it.

Two-pass union-find rather than iterative dilation: one pass over the mask
building provisional labels and a union table, one pass resolving them. Linear
in pixels, and on a 1280x720 mask it runs in single-digit milliseconds, which
keeps a 200-frame synthetic test under a couple of seconds.
"""

from typing import NamedTuple

import numpy as np


class Blob(NamedTuple):
    """One connected region. Coordinates are xyxy, exclusive on the far edge.

    xyxy matches BBox in ai/contracts/stages.py. Getting this wrong -- xywh here,
    xyxy there -- produces boxes that look plausible and crop the wrong pixels,
    which then reads as an OCR problem three stages later.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    label: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def fill_ratio(self) -> float:
        """Region area over bounding-box area.

        A vehicle-shaped blob fills most of its box. A blob tracing a lane
        marking across the frame fills almost none of it. Cheap and effective
        rejection for a stub detector.
        """
        box_area = self.width * self.height
        return 0.0 if box_area <= 0 else self.area / float(box_area)


class _UnionFind:
    """Path-compressing union-find over provisional labels."""

    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: list[int] = [0]  # index 0 is background, never used

    def make(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # compress
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Lower root wins, so labels stay roughly in raster order. Purely
            # cosmetic, but it makes debug output stable and readable.
            if root_a < root_b:
                self.parent[root_b] = root_a
            else:
                self.parent[root_a] = root_b


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected regions of a boolean mask.

    Returns (labels, count) where labels is int32, 0 is background and set pixels
    carry 1..count.

    4-connected, not 8: diagonal connectivity bridges two vehicles that touch at
    a single corner pixel into one box, and one merged box costs a whole vehicle
    event. Requiring an edge-adjacent pixel is the conservative choice.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")

    height, width = binary.shape
    provisional = np.zeros((height, width), dtype=np.int32)
    union = _UnionFind()

    # --- pass 1: provisional labels, recording equivalences -------------------
    for y in range(height):
        row = binary[y]
        if not row.any():
            continue

        current = provisional[y]
        above = provisional[y - 1] if y > 0 else None

        # np.flatnonzero over the row rather than iterating every pixel: on a
        # typical frame the mask is a few percent set, so this skips ~95% of the
        # work that a naive double loop would do.
        for x in np.flatnonzero(row):
            up = int(above[x]) if above is not None else 0
            left = int(current[x - 1]) if x > 0 else 0

            if up and left:
                current[x] = up
                union.union(up, left)
            elif up:
                current[x] = up
            elif left:
                current[x] = left
            else:
                current[x] = union.make()

    if len(union.parent) == 1:
        return provisional, 0

    # --- pass 2: resolve to dense final labels --------------------------------
    roots = np.array(
        [union.find(i) if i else 0 for i in range(len(union.parent))], dtype=np.int32
    )
    unique_roots = np.unique(roots[1:])
    remap = np.zeros(int(roots.max()) + 1, dtype=np.int32)
    remap[unique_roots] = np.arange(1, len(unique_roots) + 1, dtype=np.int32)

    labels = remap[roots[provisional]]
    return labels, int(len(unique_roots))


def blobs_from_mask(
    mask: np.ndarray,
    *,
    min_area: int = 1,
    min_width: int = 1,
    min_height: int = 1,
    min_fill_ratio: float = 0.0,
) -> list[Blob]:
    """Label a mask and return its regions as Blobs, largest first.

    Largest first because a stub detector's callers generally want the vehicles,
    and on the synthetic scenes the nearest vehicle is both the largest blob and
    the one whose plate is legible. Ordering by area makes truncating to a top-N
    meaningful rather than arbitrary.
    """
    labels, count = label_mask(mask)
    if count == 0:
        return []

    found: list[Blob] = []
    # One vectorised pass per axis instead of count passes over the frame. At 30
    # blobs on a 720p frame the naive version is ~30x slower for no benefit.
    flat = labels.ravel()
    ys, xs = np.divmod(np.flatnonzero(flat), labels.shape[1])
    ids = flat[flat != 0]

    order = np.argsort(ids, kind="stable")
    ids_sorted, ys, xs = ids[order], ys[order], xs[order]
    boundaries = np.searchsorted(ids_sorted, np.arange(1, count + 2))

    for label in range(1, count + 1):
        start, stop = int(boundaries[label - 1]), int(boundaries[label])
        if stop <= start:
            continue

        area = stop - start
        if area < min_area:
            continue

        segment_x, segment_y = xs[start:stop], ys[start:stop]
        x1, x2 = int(segment_x.min()), int(segment_x.max()) + 1
        y1, y2 = int(segment_y.min()), int(segment_y.max()) + 1

        if (x2 - x1) < min_width or (y2 - y1) < min_height:
            continue

        blob = Blob(x1=x1, y1=y1, x2=x2, y2=y2, area=area, label=label)
        if blob.fill_ratio < min_fill_ratio:
            continue
        found.append(blob)

    found.sort(key=lambda b: b.area, reverse=True)
    return found


def colourfulness(frame_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel channel spread: max(B,G,R) - min(B,G,R).

    The separator the synthetic scenes are built around. Road, sky, buildings and
    lane markings are grey or near-grey, so their channel spread is small.
    Vehicles are drawn in saturated colours, so theirs is large. A single
    subtraction distinguishes them with no model and no threshold tuning per
    scene.

    Returns uint8 so downstream thresholds read in familiar 0-255 units.
    """
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR, got shape {arr.shape}")

    as_int = arr.astype(np.int16)
    spread = as_int.max(axis=2) - as_int.min(axis=2)
    return spread.astype(np.uint8)


def darkness(frame_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel inverse luma, for finding dark regions on a light background.

    Used for plate localisation in the stub: the synthetic plates are light
    rectangles carrying dark glyphs, so the glyph run is a dark cluster inside a
    bright region. Real plates vary far more, which is why this is a stub and
    the real path is a trained detector.
    """
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR, got shape {arr.shape}")

    luma = (
        0.114 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.299 * arr[:, :, 2].astype(np.float32)
    )
    return (255.0 - luma).clip(0, 255).astype(np.uint8)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two xyxy boxes.

    Lives here rather than in the tracker because both the tracker and the
    detector's own suppression need it, and a second implementation of IoU is a
    second chance to get the exclusive-edge convention wrong.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    union_area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return 0.0 if union_area <= 0 else intersection / float(union_area)


def suppress_overlaps(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    *,
    iou_threshold: float = 0.55,
) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first.

    Every detector needs this and most bundle their own. The stub does not get
    one for free, and two boxes on one vehicle become two tracks, two plate
    reads and two sighting events -- which inflates the denominator of the
    primary metric and makes the demo show a phantom vehicle.
    """
    if len(boxes) != len(scores):
        raise ValueError(f"{len(boxes)} boxes but {len(scores)} scores")

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for index in order:
        if all(iou(boxes[index], boxes[k]) < iou_threshold for k in kept):
            kept.append(index)
    return kept
