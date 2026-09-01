"""OCR -- stage 8 of the 14. Reads an already-located plate crop.

    build_ocr_engine({"name": "paddle"})              real weights, Apache-2.0
    build_ocr_engine({"name": "template"})            numpy only, no download
    build_ocr_engine({"name": "oracle"}, source=src)  ground truth

Two rules from Contracts section 12 govern everything in this package, and both are
easy to break by accident:

**`plate: null` is a valid correct answer.** A vehicle whose plate is turned away, or
26 px wide, or motion-blurred past legibility, has no readable plate, and reporting
none is the right output. Fabricating a plausible string for it is the worst failure
this pipeline can produce -- it puts a real registration number, belonging to a real
person, at a place and time it was never at. Every backend here therefore has an
explicit refusal path, and the refusal counters are in stats() rather than hidden.

**Confidence is never multiplied.** Detector confidence, quality score and OCR
confidence are three uncalibrated numbers on three different scales; their product is
not a probability of anything. They travel separately all the way to the event.

The variant loop is in base.py and the six variants are in preprocess.py, applied one
at a time and never composed -- the reason is measured and written there. What comes
out is an OCRRead carrying both `confidence` and `agreement`; the second exists because
max-of-N confidence is biased upward by construction and cannot be read as certainty
on its own.
"""

from ai.ocr.base import (
    MIN_OCR_PLATE_WIDTH_PX,
    PLATE_CROP_PAD_PX,
    BaseOCR,
    OCREngine,
    OCRRead,
)
from ai.ocr.factory import (
    OCR_ENGINE_NAMES,
    SHIPPABLE_OCR_ENGINES,
    OCRConfigError,
    build_ocr_engine,
    check_ocr_width_floor,
    default_variants_for,
    describe_ocr_engine,
    normalize_ocr_config,
    ocr_engine_ships,
)
from ai.ocr.preprocess import (
    DEFAULT_VARIANTS,
    apply_variant,
    variant_names,
)

__all__ = [
    "DEFAULT_VARIANTS",
    "MIN_OCR_PLATE_WIDTH_PX",
    "OCR_ENGINE_NAMES",
    "PLATE_CROP_PAD_PX",
    "SHIPPABLE_OCR_ENGINES",
    "BaseOCR",
    "OCRConfigError",
    "OCREngine",
    "OCRRead",
    "apply_variant",
    "build_ocr_engine",
    "check_ocr_width_floor",
    "default_variants_for",
    "describe_ocr_engine",
    "normalize_ocr_config",
    "ocr_engine_ships",
    "variant_names",
]
