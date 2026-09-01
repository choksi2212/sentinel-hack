"""Plate detection -- stage 6 of the 14. Runs on the vehicle crop, not the frame.

    build_plate_detector({"name": "rtdetr"})   real weights, Apache-2.0
    build_plate_detector({"name": "edge"})     numpy only, no download
    build_plate_detector({"name": "oracle"}, source=src)   ground truth

One PlateCandidate per vehicle per frame, in FULL FRAME coordinates. The stage
returns a dict keyed by track_id and a missing key means no plate was found, which is
a correct and common answer -- most vehicles at a junction are facing the wrong way.

Why the crop and not the frame is in geometry.py; why the aspect filter spans 0.7 to
6.0 rather than sitting at 4:1 is there too, and it is the detail most likely to be
got wrong by anyone extending this: a 4:1 filter rejects every two-row plate, which
is most motorcycles, which is a large share of Indian traffic.
"""

from ai.plate.base import (
    DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
    BasePlateDetector,
    PlateDetector,
)
from ai.plate.factory import (
    PLATE_DETECTOR_NAMES,
    SHIPPABLE_PLATE_DETECTORS,
    PlateConfigError,
    build_plate_detector,
    describe_plate_detector,
    normalize_plate_config,
    plate_detector_ships,
)
from ai.plate.geometry import (
    CROP_PAD_FRACTION,
    PLATE_ASPECT_MAX,
    PLATE_ASPECT_MIN,
    aspect_ratio,
    crop_vehicle,
    map_to_frame,
    plausible_plate_box,
    region_prior,
)

__all__ = [
    "CROP_PAD_FRACTION",
    "DEFAULT_PLATE_CONFIDENCE_THRESHOLD",
    "PLATE_ASPECT_MAX",
    "PLATE_ASPECT_MIN",
    "PLATE_DETECTOR_NAMES",
    "SHIPPABLE_PLATE_DETECTORS",
    "BasePlateDetector",
    "PlateConfigError",
    "PlateDetector",
    "aspect_ratio",
    "build_plate_detector",
    "crop_vehicle",
    "describe_plate_detector",
    "map_to_frame",
    "normalize_plate_config",
    "plate_detector_ships",
    "plausible_plate_box",
    "region_prior",
]
