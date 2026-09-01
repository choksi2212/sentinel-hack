"""Vehicle crops, plate boxes, and the coordinate frame each one lives in.

Plate detection runs on the vehicle crop rather than the full frame, which is
Technical Implementation C5 and is worth the two paragraphs it takes to say why.
A plate on an approaching car at a junction is 40x15 px in a 1920x1080 frame --
0.03% of the pixels. A detector given the whole frame spends its capacity on
2.07 million pixels of tarmac, shopfront and sky, and the plate is smaller than
its stride. Given a 180x140 crop instead, the same plate is 22% of the width, and
the road signage and shop hoardings that produce most plate false positives are
outside the crop entirely.

The cost is that every box comes back in crop coordinates and has to be mapped
home. Contracts section 3 requires plate_bbox_xyxy in FULL FRAME coordinates, and
the reason it says so in capitals is that a crop-local box is not obviously wrong:
it is a plausible box of a plausible size, so it passes every type check, lands in
the event, and produces a snapshot URI pointing at a region of road forty metres
from the vehicle plus a plate_width_px that happens to be right. Nothing downstream
can detect it. So the mapping lives in one function here, and the crop origin is
returned alongside the crop rather than recomputed by each backend.
"""

from typing import Optional, Sequence

import numpy as np

from ai.contracts.stages import BBox

# Padding added around the vehicle box before cropping, as a fraction of box size.
#
# Not zero, because the box being cropped is a tracker output. When a track is
# coasting through a missed detection the box is the Kalman estimate, which lags --
# measured at IoU 0.98 against truth while coasting, so about 2% of the box is in the
# wrong place, and a rear plate sits exactly at the trailing edge that lag moves off.
# Not large either: every padded pixel is a pixel of road that can hold a false
# positive, and the crop's whole value is that it excludes them.
CROP_PAD_FRACTION = 0.08

# Where on a vehicle a plate can be, as fractions of the vehicle box height.
#
# Plates are mounted low, and the top half of a vehicle box contains the windscreen,
# the roof, and on a truck a tailgate full of painted lettering that a plate detector
# finds extremely interesting. Restricting the search to the lower 65% is a prior, not
# a certainty -- it is applied as a *penalty region* rather than a hard crop, because
# a motorcycle box is mostly rider and its plate can land higher in the box than the
# geometry suggests.
PLATE_REGION_TOP_FRACTION = 0.35

# Indian plate aspect ratios (width / height), from the Central Motor Vehicles Rules
# dimensions, and the reason this is a range rather than a number.
#
#     cars, single row        500 x 120 mm    4.17
#     cars, two row           340 x 200 mm    1.70
#     motorcycles, two row    285 x 200 mm    1.43
#     motorcycles, single row 200 x 100 mm    2.00
#
# A 4:1 aspect filter is the obvious implementation and it rejects every two-row
# plate, which means it rejects most motorcycles. On Indian roads that is not an edge
# case -- it is a large fraction of the traffic, and a plate stage that silently
# cannot read motorcycles would show up as a vehicle-class accuracy gap that looks
# like a detector problem.
#
# Perspective widens the range further downward: a plate seen 60 degrees off-axis has
# its width compressed by cos(60) = 0.5, taking a 4.17 car plate to 2.08 and a 1.43
# motorcycle plate to 0.72. The bounds below cover that, which makes them loose enough
# that aspect alone rejects very little. That is the honest position: aspect is a
# sanity check against a box that is a shadow or a bumper edge, not a classifier.
PLATE_ASPECT_MIN = 0.70
PLATE_ASPECT_MAX = 6.00

# Below this a plate box is too small for the aspect check to mean anything -- a
# 6x3 box has an aspect quantised to 2.0 and carries no information.
MIN_PLATE_BOX_PX = 8


def crop_vehicle(
    frame_bgr: np.ndarray,
    bbox: BBox,
    *,
    pad_fraction: float = CROP_PAD_FRACTION,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract a padded vehicle crop and the frame coordinate of its top-left pixel.

    Returns (crop, (origin_x, origin_y)). The origin is the whole reason this
    returns a tuple: a backend that receives only the crop cannot map its findings
    back, and one that recomputes the origin from the bbox has to replicate the
    clamping below and will eventually do it differently.

    The crop is a view into the frame where possible, not a copy. Plate detection
    runs on every gated vehicle on every sampled frame, so at 10 fps with eight
    vehicles in view that is 80 crops a second per camera; copying each one is
    real memory traffic for no benefit, since no backend writes to its input.
    """
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    pad_x = int(round((x2 - x1) * pad_fraction))
    pad_y = int(round((y2 - y1) * pad_fraction))

    # Clamped to the frame, which is why the origin has to be returned: a vehicle
    # at the left edge gets origin_x = 0 rather than the negative value the padding
    # asked for, and the mapping back has to use what was actually taken.
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)

    if right <= left or bottom <= top:
        # Degenerate box: entirely outside the frame, or zero-area. Returning an
        # empty array rather than raising, because one bad tracker box on one frame
        # must not end the run -- the caller counts it and moves on.
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype), (max(0, left), max(0, top))

    return frame_bgr[top:bottom, left:right], (left, top)


def map_to_frame(
    local_bbox: BBox,
    origin: tuple[int, int],
    frame_shape: tuple[int, ...],
) -> BBox:
    """Crop-local box -> full-frame box, clamped to the frame.

    The clamp matters on a plate found at the edge of a crop that was itself taken
    at the edge of the frame: without it the box can extend past the frame, and a
    snapshot cropped to those coordinates comes back a different size than
    plate_width_px claims.
    """
    origin_x, origin_y = origin
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = local_bbox
    return (
        max(0, min(frame_w, int(x1) + origin_x)),
        max(0, min(frame_h, int(y1) + origin_y)),
        max(0, min(frame_w, int(x2) + origin_x)),
        max(0, min(frame_h, int(y2) + origin_y)),
    )


def aspect_ratio(bbox: BBox) -> float:
    """Width / height, guarded against a zero-height box."""
    x1, y1, x2, y2 = bbox
    height = max(1, y2 - y1)
    return (x2 - x1) / float(height)


def plausible_plate_box(
    bbox: BBox,
    *,
    aspect_min: float = PLATE_ASPECT_MIN,
    aspect_max: float = PLATE_ASPECT_MAX,
    min_size_px: int = MIN_PLATE_BOX_PX,
) -> bool:
    """Whether a box could be a plate on shape alone.

    Deliberately permissive -- see PLATE_ASPECT_MIN for why the range is wide. This
    rejects the box that is a shadow under a bumper or a strip of grille, and is not
    trying to do the detector's job.
    """
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width < min_size_px or height < min(min_size_px, 6):
        return False
    return aspect_min <= aspect_ratio(bbox) <= aspect_max


def region_prior(bbox: BBox, vehicle_bbox: BBox) -> float:
    """1.0 if the box sits where a plate belongs, tapering to 0.4 if it does not.

    A multiplier on detector confidence rather than a filter. The distinction is the
    point: a hard cut on vertical position throws away a correct motorcycle plate,
    whose box can sit high because most of a motorcycle box is rider. A soft penalty
    lets it survive while still ranking a windscreen-height box below a bumper-height
    one when both are found on the same vehicle.
    """
    v_top, v_bottom = vehicle_bbox[1], vehicle_bbox[3]
    v_height = max(1, v_bottom - v_top)
    centre_y = (bbox[1] + bbox[3]) / 2.0
    relative = (centre_y - v_top) / v_height

    if relative >= PLATE_REGION_TOP_FRACTION:
        return 1.0
    # Linear taper up to the top of the box. 0.4 rather than 0.0 so that a plate on a
    # vehicle whose box is badly bounded -- a tracker estimate mid-occlusion, say --
    # is demoted rather than deleted.
    shortfall = (PLATE_REGION_TOP_FRACTION - relative) / PLATE_REGION_TOP_FRACTION
    return 1.0 - 0.6 * max(0.0, min(1.0, shortfall))


def clip_to_crop(bbox: BBox, crop_shape: tuple[int, ...]) -> Optional[BBox]:
    """Clamp a box to the crop it was found in, or None if nothing survives.

    Backends that upscale a crop before inference can return boxes a pixel or two
    outside it through rounding. Silently keeping those gives a negative-width box
    after the mapping home, and a negative plate_width_px reaches the event.
    """
    crop_h, crop_w = crop_shape[0], crop_shape[1]
    x1 = max(0, min(crop_w, int(bbox[0])))
    y1 = max(0, min(crop_h, int(bbox[1])))
    x2 = max(0, min(crop_w, int(bbox[2])))
    y2 = max(0, min(crop_h, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def best_by_confidence(
    boxes: Sequence[BBox], confidences: Sequence[float]
) -> Optional[int]:
    """Index of the highest-confidence box, or None if there are none."""
    if not boxes:
        return None
    return max(range(len(boxes)), key=lambda i: confidences[i])
