"""Plate normalization and the soft grammar check.

Canonical Contracts section 4.2 and owner's manual section 5.9.
"""

import re

# COPIED FROM CANONICAL CONTRACTS -- DO NOT EDIT HERE (Contracts section 4.2).
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_plate(raw: str) -> str:
    """GJ 01 AB 1234 -> GJ01AB1234.  'gj-01-ab-1234' -> GJ01AB1234.

    Both raw and normalized are stored. raw is the audit trail; normalized is
    the search key. Normalization is NEVER applied to raw in place.
    """
    return _NON_ALNUM.sub("", raw.upper())
# END COPIED BLOCK.


# Soft grammar check. Owner's manual section 5.9.
#
#   two letters (state) + one or two digits (RTO) + up to three letters
#   (series) + four digits
#
# A miss DOWNGRADES CONFIDENCE. It never rewrites the string and never drops
# the observation. BH-series plates, older formats, and diplomatic and military
# plates all exist and all fail this pattern; a hard filter silently deletes
# real vehicles, which is a worse failure than a low-confidence record.
INDIAN_PLATE_GRAMMAR = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")


def grammar_ok(normalized: str) -> bool:
    """True when the normalized string matches the common Indian plate shape.

    Advisory only. Read the module docstring above before using this to filter
    anything.
    """
    return bool(normalized) and INDIAN_PLATE_GRAMMAR.match(normalized) is not None


def looks_like_partial(normalized: str) -> bool:
    """Heuristic: plausible plate fragment rather than noise.

    Used by the failure taxonomy to separate ocr_partial from ocr_wrong. Not
    used to gate emission -- a fragment is still emitted, with its own low
    confidence, because 'GJ01AB' plus a camera and a time is investigatively
    useful and inventing the missing digits is not.
    """
    if not normalized or len(normalized) < 4:
        return False
    if grammar_ok(normalized):
        return False
    has_letters = any(c.isalpha() for c in normalized)
    has_digits = any(c.isdigit() for c in normalized)
    return has_letters and has_digits and len(normalized) <= 10
