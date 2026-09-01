"""The three measured components of the plate quality score.

The fourth component, detector_confidence, comes from the plate detector and
needs no computation here.

The reference constants below are tuning knobs, not contract values -- but they
must not drift, because the score they feed is the temporal-fusion weight.
Changing one changes every fused plate in the project, so change it with a
benchmark, not a hunch.
"""

import numpy as np

# Variance of the Laplacian at which an image counts as fully sharp. Measured
# on 8-bit grayscale plate crops: a crisp crop lands in the high hundreds, a
# motion-blurred one in the low tens. Saturating at 500 keeps a very sharp crop
# from dominating the ranking on sharpness alone.
SHARPNESS_REFERENCE = 500.0

# Contracts section 4.1: "plate pixel width vs 100px reference". A 100 px plate
# scores 1.0; the 30 px plates that decide whether this works on real
# infrastructure score 0.3.
RESOLUTION_REFERENCE_PX = 100.0

# Pixels at or beyond these values carry no recoverable detail.
CLIP_LOW = 5
CLIP_HIGH = 250

# Fraction of clipped pixels at which exposure scores 0. A quarter of the crop
# blown out or crushed means the characters are gone.
EXPOSURE_CLIP_TOLERANCE = 0.25

# 4-neighbour Laplacian. Applied by slicing rather than convolution so that
# this module needs neither cv2 nor scipy.
_LAPLACIAN_MIN_SIDE = 3


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR or grayscale in, float32 grayscale out, values 0..255.

    Uses the ITU-R BT.601 luma weights in BGR channel order, matching what
    cv2.cvtColor(COLOR_BGR2GRAY) produces, so a crop scored here and a crop
    scored in a notebook with OpenCV agree.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        b, g, r = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32)
    raise ValueError(f"expected HxW or HxWx3 image, got shape {arr.shape}")


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian. Higher means sharper.

    Returns 0.0 for a crop too small to have an interior, which is the honest
    answer -- a 2 px tall crop has no measurable sharpness, and returning
    something nonzero would let it compete for a top-K slot.
    """
    gray = to_gray(image)
    if gray.shape[0] < _LAPLACIAN_MIN_SIDE or gray.shape[1] < _LAPLACIAN_MIN_SIDE:
        return 0.0

    interior = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]      # up
        + gray[2:, 1:-1]     # down
        + gray[1:-1, :-2]    # left
        + gray[1:-1, 2:]     # right
        - 4.0 * interior
    )
    return float(np.var(lap))


def sharpness_norm(image: np.ndarray) -> float:
    """Laplacian variance normalized to 0..1, saturating at SHARPNESS_REFERENCE."""
    return min(1.0, laplacian_variance(image) / SHARPNESS_REFERENCE)


def resolution_norm(plate_width_px: int) -> float:
    """Plate pixel width against the 100 px reference, capped at 1.0.

    Takes the width directly rather than measuring the array, because the crop
    handed to OCR may already have been upscaled. Scoring an upscaled crop as
    high-resolution would rank a 30 px plate stretched to 120 px above a real
    120 px plate, which is precisely backwards.
    """
    if plate_width_px <= 0:
        return 0.0
    return min(1.0, plate_width_px / RESOLUTION_REFERENCE_PX)


def exposure_norm(image: np.ndarray) -> float:
    """1.0 for a well-exposed crop, 0.0 when a quarter of it is clipped.

    Clipping is measured at both ends: a plate under a sodium lamp at night is
    crushed to black, and one in direct afternoon glare is blown to white. Both
    lose the characters, and both are common on the Sentinel feeds.
    """
    gray = to_gray(image)
    if gray.size == 0:
        return 0.0
    clipped = float(np.count_nonzero((gray <= CLIP_LOW) | (gray >= CLIP_HIGH)))
    fraction = clipped / gray.size
    return max(0.0, 1.0 - fraction / EXPOSURE_CLIP_TOLERANCE)


def contrast_norm(image: np.ndarray) -> float:
    """Diagnostic only -- standard deviation of luma, normalized.

    Not part of the locked score. Useful in the failure taxonomy for telling a
    flat grey crop apart from a sharp but badly exposed one.
    """
    gray = to_gray(image)
    if gray.size == 0:
        return 0.0
    return min(1.0, float(np.std(gray)) / 64.0)
