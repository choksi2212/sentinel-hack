"""Preprocessing variants for the OCR stage. Applied ONE AT A TIME, never composed.

Technical Implementation C7 lists six: raw, grayscale, contrast stretch, adaptive
threshold, 2x upscale, mild sharpening. Each produces an independent read and the
best-scoring one wins.

**Why one at a time, when composing them is obviously more powerful.** Three reasons,
in increasing order of how much they cost:

1. Six variants tried singly is six reads. Composed in every order and combination it
   is 1957, and at 0.7 ms a read on eight vehicles a frame that is no longer a
   preprocessing step, it is the frame budget.
2. Every one of these transforms is lossy. Adaptive threshold discards greyscale
   entirely; upscaling invents pixels between the real ones; sharpening amplifies
   whatever noise survived the previous step. Stacking two does not compose their
   benefits, it compounds their damage -- threshold-then-sharpen sharpens the
   quantisation artefacts of the threshold, not the plate.
3. The variant that won is diagnostic information, and composition destroys it. When
   the 40-60 px bucket reads badly and `variant_wins` says upscale_2x wins 80% of the
   reads there, that is a finding: the model is resolution-starved and the fix is a
   larger input size, not a better OCR model. "sharpen-then-threshold-then-upscale
   won" says nothing anyone can act on.

**Super-resolution is not here and is not a default.** It is the one preprocessing step
that changes what the image *says* rather than how clearly it says it: a generative
upscaler asked to sharpen a 3 px character produces a clean, confident, plausible,
different character -- and OCR confidence goes *up*, because the output really is
crisper. That is the exact shape of the worst failure this system can produce, a
fabricated plate that points at a real vehicle. upscale_2x below is bilinear
interpolation, which invents no structure: it blurs, and a blurred character reads as
low confidence, which is the correct signal.
"""

from typing import Callable, Optional

import numpy as np

# Order matters only for reproducibility of tie-breaks; the caller takes the maximum.
# raw first so that a backend which needs no help is not credited to a variant.
DEFAULT_VARIANTS: tuple[str, ...] = (
    "raw",
    "grayscale",
    "contrast_stretch",
    "adaptive_threshold",
    "upscale_2x",
    "sharpen",
)

# Window for the local mean in adaptive_threshold, as a fraction of crop height.
# A plate crop is roughly 4:1, so a window scaled to height covers about a quarter of
# the width -- two to three characters. Large enough to average over strokes and gaps
# rather than tracking individual characters, small enough to follow a lighting
# gradient across the plate, which is the whole reason to threshold adaptively rather
# than globally: a plate half in shadow has no single correct global threshold.
ADAPTIVE_WINDOW_FRACTION = 1.0

# Offset subtracted from the local mean. Positive, so a pixel must be meaningfully
# darker than its surroundings to become ink. Without it, flat regions split roughly
# half and half on sensor noise and the result is a field of speckle.
ADAPTIVE_OFFSET = 8

# Percentiles clipped by contrast_stretch. Not 0 and 100: a single hot pixel or one
# dark speck would then set the whole mapping, and one pixel deciding the contrast of
# the image is how a stretch makes things worse.
STRETCH_LOW_PERCENTILE = 2.0
STRETCH_HIGH_PERCENTILE = 98.0


def apply_variant(crop_bgr: np.ndarray, name: str) -> Optional[np.ndarray]:
    """Run one named variant. Unknown name is an error, not a silent pass-through.

    A typo in a config's variant list must not turn into "raw was tried twice", because
    that reads as a normal run producing slightly worse numbers.
    """
    fn = _VARIANTS.get(name)
    if fn is None:
        raise KeyError(
            f"unknown OCR preprocessing variant {name!r}; available: "
            f"{sorted(_VARIANTS)}"
        )
    if crop_bgr.size == 0:
        return None
    return fn(crop_bgr)


def variant_names() -> tuple[str, ...]:
    """Every implemented variant, for config validation."""
    return tuple(sorted(_VARIANTS))


# ---------------------------------------------------------------------- the variants


def raw(crop_bgr: np.ndarray) -> np.ndarray:
    """Unchanged, except copied.

    The copy is the point. Every other variant produces a new array anyway; this one
    would hand back a view into the frame, and a backend that writes to its input --
    or a caller that does -- would corrupt the frame for the snapshot stage and for
    every vehicle read after this one in the same frame. One wasted memcpy on a 60x20
    crop against a class of bug that appears as "the second vehicle in each frame reads
    worse than the first".
    """
    return crop_bgr.copy()


def grayscale(crop_bgr: np.ndarray) -> np.ndarray:
    """Luminance-weighted single channel, kept 3-channel for backend compatibility.

    Weights 0.114/0.587/0.299 on B/G/R -- the Rec.601 luma coefficients, which is what
    every OCR engine's own internal conversion uses. Doing it here rather than letting
    the engine do it is not redundant: it is a *variant*, and it differs from raw
    because an engine handed a 3-channel image may convert differently or not at all.

    Returned as 3-channel rather than 2-D so that a backend expecting colour does not
    silently fail on this one variant while succeeding on the others -- a failure mode
    that shows up only as this variant never appearing in variant_wins.
    """
    if crop_bgr.ndim == 2:
        grey = crop_bgr.astype(np.float32)
    else:
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        grey = crop_bgr[:, :, :3].astype(np.float32) @ weights
    out = np.clip(grey, 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


def contrast_stretch(crop_bgr: np.ndarray) -> np.ndarray:
    """Linear rescale so the 2nd and 98th percentiles land on 0 and 255.

    The variant that matters most at night. A plate under sodium light occupies maybe
    60 grey levels out of 256, and an OCR model trained on daylight plates is being
    asked to read a shape whose edges are 15 levels apart when it expects 200. This
    does not add information -- it puts the information that is there into the range
    the model's features were trained on.

    Percentiles rather than min and max: see STRETCH_LOW_PERCENTILE.
    """
    grey = grayscale(crop_bgr)[:, :, 0].astype(np.float32)
    low = float(np.percentile(grey, STRETCH_LOW_PERCENTILE))
    high = float(np.percentile(grey, STRETCH_HIGH_PERCENTILE))
    if high - low < 1.0:
        # A flat crop. Stretching it would amplify nothing into everything: dividing by
        # a sub-unit range turns sensor noise into full-scale black and white speckle,
        # and the result reads as confident nonsense rather than as unreadable.
        return grayscale(crop_bgr)
    scaled = np.clip((grey - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def adaptive_threshold(crop_bgr: np.ndarray) -> np.ndarray:
    """Binarise against a local mean. Pure numpy, via an integral image.

    cv2.adaptiveThreshold is the usual way to get this and OpenCV is not a dependency
    of this lane -- ai/media exists to run on a machine with no codecs, and an OCR
    preprocessing step that pulls in cv2 would undo that. The integral image makes the
    box filter O(1) per pixel regardless of window size, so this costs one cumsum in
    each direction and four adds per pixel.

    Binarisation is the variant most likely to win on a clean plate and most likely to
    destroy a marginal one, because it makes an irreversible decision per pixel with no
    confidence attached. That is exactly why it is one of six rather than the only step.
    """
    grey = grayscale(crop_bgr)[:, :, 0].astype(np.float32)
    height, width = grey.shape

    window = max(3, int(height * ADAPTIVE_WINDOW_FRACTION) | 1)  # odd, at least 3
    radius = window // 2

    # Integral image with a zero row and column, so a window sum is four lookups with
    # no boundary special-casing.
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = grey.cumsum(axis=0).cumsum(axis=1)

    ys = np.arange(height)
    xs = np.arange(width)
    y0 = np.clip(ys - radius, 0, height)
    y1 = np.clip(ys + radius + 1, 0, height)
    x0 = np.clip(xs - radius, 0, width)
    x1 = np.clip(xs + radius + 1, 0, width)

    total = (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )
    count = np.outer(y1 - y0, x1 - x0).astype(np.float64)
    local_mean = total / np.maximum(count, 1.0)

    # Ink is darker than its surroundings. Output keeps the ink-dark/background-light
    # polarity of a real plate rather than inverting, because every OCR engine and the
    # glyph templates in ai/media/glyphs.py both assume dark-on-light.
    ink = grey < (local_mean - ADAPTIVE_OFFSET)
    out = np.where(ink, 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


def upscale_2x(crop_bgr: np.ndarray) -> np.ndarray:
    """Bilinear 2x. Invents no structure -- see the module docstring on super-resolution.

    Worth its slot for one specific reason: a detection model has a minimum feature
    size below which a stroke simply does not activate anything, and doubling a 30 px
    plate to 60 px puts its strokes back above that size without changing what they
    are. The image contains no more information afterwards; the model can just act on
    the information it already had.

    Bilinear rather than nearest so that the interpolated pixels are visibly
    intermediate. Nearest-neighbour doubling produces crisp blocky edges that an OCR
    engine scores as high-confidence, which is precisely the false signal to avoid --
    the read should look as uncertain as it is.
    """
    height, width = crop_bgr.shape[:2]
    if height == 0 or width == 0:
        return crop_bgr.copy()
    return _resize_bilinear(crop_bgr, width * 2, height * 2)


def sharpen(crop_bgr: np.ndarray) -> np.ndarray:
    """Mild unsharp mask: original plus 0.6 of (original minus a 3x3 blur).

    "Mild" is doing real work in that sentence. A strong unsharp mask on a small plate
    produces ringing on either side of every stroke, and a bright halo next to a dark
    stroke is a new edge that was not in the scene -- which is how sharpening turns an
    8 into a B. 0.6 is chosen to be visibly below where ringing appears on a 30 px
    plate; the strength is not exposed in config for that reason.
    """
    grey = grayscale(crop_bgr)[:, :, 0].astype(np.float32)
    blurred = _box_blur_3x3(grey)
    out = np.clip(grey + 0.6 * (grey - blurred), 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------- internals


def _box_blur_3x3(grey: np.ndarray) -> np.ndarray:
    """3x3 mean with edge padding. Separable, so two passes of three adds."""
    padded = np.pad(grey, 1, mode="edge")
    # Sum of three consecutive rows, then of three consecutive columns.
    rows = padded[:-2, :] + padded[1:-1, :] + padded[2:, :]
    cols = rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]
    return cols / 9.0


def _resize_bilinear(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Bilinear resize without cv2 or PIL.

    Half-pixel centre convention -- source coordinate (dst + 0.5) * scale - 0.5 -- which
    is what cv2 and PIL both use. The naive dst * scale convention shifts the image by
    half a pixel times the scale factor, and on a 30 px plate doubled to 60 that is a
    visible offset that lands the whole read half a character to the left.
    """
    src_h, src_w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    flat = image.reshape(src_h, src_w, channels).astype(np.float32)

    y = (np.arange(new_height, dtype=np.float32) + 0.5) * (src_h / new_height) - 0.5
    x = (np.arange(new_width, dtype=np.float32) + 0.5) * (src_w / new_width) - 0.5
    y = np.clip(y, 0, src_h - 1)
    x = np.clip(x, 0, src_w - 1)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]

    top = flat[np.ix_(y0, x0)] * (1 - wx) + flat[np.ix_(y0, x1)] * wx
    bottom = flat[np.ix_(y1, x0)] * (1 - wx) + flat[np.ix_(y1, x1)] * wx
    out = top * (1 - wy) + bottom * wy

    result = np.clip(out, 0, 255).astype(np.uint8)
    return result if image.ndim == 3 else result[:, :, 0]


_VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": raw,
    "grayscale": grayscale,
    "contrast_stretch": contrast_stretch,
    "adaptive_threshold": adaptive_threshold,
    "upscale_2x": upscale_2x,
    "sharpen": sharpen,
}
