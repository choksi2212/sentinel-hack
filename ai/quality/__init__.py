"""Image quality scoring and the vehicle quality gate.

The plate quality score has locked weights because it is used twice -- once to
rank crops and once as the temporal-fusion weight. One function, two uses, so
it has to be stable: changing it silently changes every fused plate.

Deliberately numpy-only, no cv2. That keeps the locked weights unit-testable in
CI on a machine with no OpenCV, and keeps a second image library off the hot
path.
"""

from ai.quality.gate import GateDecision, VehicleGate
from ai.quality.metrics import (
    EXPOSURE_CLIP_TOLERANCE,
    RESOLUTION_REFERENCE_PX,
    SHARPNESS_REFERENCE,
    exposure_norm,
    laplacian_variance,
    resolution_norm,
    sharpness_norm,
    to_gray,
)
from ai.quality.score import QUALITY_WEIGHTS, plate_quality, rank_crops
from ai.quality.taxonomy import (
    FAILURE_BUCKETS,
    FailureBucket,
    FailureTaxonomy,
    TaxonomyVerdict,
)

__all__ = [
    "EXPOSURE_CLIP_TOLERANCE",
    "FAILURE_BUCKETS",
    "QUALITY_WEIGHTS",
    "RESOLUTION_REFERENCE_PX",
    "SHARPNESS_REFERENCE",
    "FailureBucket",
    "FailureTaxonomy",
    "GateDecision",
    "TaxonomyVerdict",
    "VehicleGate",
    "exposure_norm",
    "laplacian_variance",
    "plate_quality",
    "rank_crops",
    "resolution_norm",
    "sharpness_norm",
    "to_gray",
]
