"""Vehicle detection -- stage 3 of the fourteen in Contracts section 10.

One interface, four backends, one factory:

    VehicleDetector       the Protocol every backend satisfies
    BaseDetector          shared instrumentation: timing, counting, class filtering
    RFDETRDetector        Apache-2.0, real weights, the one that ships
    MotionBlobDetector    numpy background subtraction, no weights, no cv2
    OracleDetector        ground truth from the synthetic source
    ScriptedDetector      a fixed table, for tests
    build_detector        config block -> instance

Nothing downstream imports a concrete class. The tracker takes a list of
DetectorResult and cannot tell which backend produced it, which is what makes
swapping the detector a config change and lets the rest of the pipeline be tested
without a GPU or a 108 MB download.
"""

from ai.detect.base import (
    CLASSIFIED_VEHICLE_CLASSES,
    COCO_TO_VEHICLE_CLASS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    BaseDetector,
    VehicleDetector,
    is_shippable_class,
    map_class_name,
)
from ai.detect.blobs import Blob, blobs_from_mask, iou, label_mask, suppress_overlaps
from ai.detect.factory import (
    DETECTOR_NAMES,
    SHIPPABLE_DETECTORS,
    DetectorConfigError,
    build_detector,
    describe_detector,
    detector_ships,
    normalize_detector_config,
    resolve_allowed_classes,
)
from ai.detect.stub import (
    MotionBlobDetector,
    OracleDegradation,
    OracleDetector,
    ScriptedDetector,
)

# RFDETRDetector is deliberately absent from this list of eager imports. Importing
# it is cheap -- every heavy dependency is deferred into _load -- but keeping it
# out means `from ai.detect import ScriptedDetector` in a unit test cannot be
# broken by a change to the ONNX path. Import it explicitly:
#
#     from ai.detect.rfdetr import RFDETRDetector
#
# or, preferably, let build_detector do it.

__all__ = [
    "BaseDetector",
    "Blob",
    "CLASSIFIED_VEHICLE_CLASSES",
    "COCO_TO_VEHICLE_CLASS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DETECTOR_NAMES",
    "DetectorConfigError",
    "MotionBlobDetector",
    "OracleDegradation",
    "OracleDetector",
    "SHIPPABLE_DETECTORS",
    "ScriptedDetector",
    "VEHICLE_CLASSES",
    "VehicleDetector",
    "blobs_from_mask",
    "build_detector",
    "describe_detector",
    "detector_ships",
    "iou",
    "is_shippable_class",
    "label_mask",
    "map_class_name",
    "normalize_detector_config",
    "resolve_allowed_classes",
    "suppress_overlaps",
]
