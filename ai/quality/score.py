"""The locked plate quality score. Canonical Contracts section 4.1.

    score = 0.30 * sharpness_norm        variance of Laplacian, normalized
          + 0.25 * resolution_norm       plate width vs 100 px reference
          + 0.25 * detector_confidence
          + 0.20 * exposure_norm         penalize clipped histograms

Used for crop ranking AND as the temporal-fusion weight. One function, two
uses, so it must be stable.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ai.quality.metrics import exposure_norm, resolution_norm, sharpness_norm

# Kept as data so a test can assert they still sum to 1.0 and still hold the
# canonical values. A silent reweighting here would change every fused plate in
# the project without changing any other line of code.
QUALITY_WEIGHTS: dict[str, float] = {
    "sharpness": 0.30,
    "resolution": 0.25,
    "detector_confidence": 0.25,
    "exposure": 0.20,
}


def plate_quality(
    crop_bgr: np.ndarray,
    *,
    plate_width_px: Optional[int] = None,
    detector_confidence: float = 0.0,
) -> float:
    """Returns 0.0..1.0.

    plate_width_px defaults to the crop's own width. Pass it explicitly when
    the crop has been upscaled for OCR, so that resolution is scored on the
    plate as it appeared in the frame rather than on the interpolation.
    """
    arr = np.asarray(crop_bgr)
    if arr.size == 0:
        return 0.0

    width = int(arr.shape[1]) if plate_width_px is None else int(plate_width_px)
    detector_confidence = _clamp(detector_confidence)

    score = (
        QUALITY_WEIGHTS["sharpness"] * sharpness_norm(arr)
        + QUALITY_WEIGHTS["resolution"] * resolution_norm(width)
        + QUALITY_WEIGHTS["detector_confidence"] * detector_confidence
        + QUALITY_WEIGHTS["exposure"] * exposure_norm(arr)
    )
    return _clamp(score)


def plate_quality_breakdown(
    crop_bgr: np.ndarray,
    *,
    plate_width_px: Optional[int] = None,
    detector_confidence: float = 0.0,
) -> dict[str, float]:
    """Same score, with the four components exposed.

    This is what goes in the failure taxonomy: a crop rejected at quality 0.31
    is actionable when you can see it was sharpness 0.9, resolution 0.28,
    exposure 0.1 -- a sharp plate in blinding glare, which is a camera problem,
    not a model problem.
    """
    arr = np.asarray(crop_bgr)
    if arr.size == 0:
        return {"sharpness": 0.0, "resolution": 0.0, "detector_confidence": 0.0,
                "exposure": 0.0, "score": 0.0}

    width = int(arr.shape[1]) if plate_width_px is None else int(plate_width_px)
    parts = {
        "sharpness": sharpness_norm(arr),
        "resolution": resolution_norm(width),
        "detector_confidence": _clamp(detector_confidence),
        "exposure": exposure_norm(arr),
    }
    parts["score"] = _clamp(sum(QUALITY_WEIGHTS[k] * v for k, v in parts.items()))
    return parts


def rank_crops(crops: Sequence[dict[str, Any]], top_k: int = 4) -> list[dict[str, Any]]:
    """Sort scored crops best first and keep the top K.

    Each dict needs a 'quality' key. Ties break on plate_width_px so the
    ordering is deterministic -- a benchmark that reorders between runs on
    equal scores is not reproducible, and reproducibility is the whole point
    of the frozen test set.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    ordered = sorted(
        crops,
        key=lambda c: (float(c.get("quality", 0.0)), int(c.get("plate_width_px", 0))),
        reverse=True,
    )
    return list(ordered[:top_k])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
