"""Normalization, grammar validation and match-state derivation.

Pure functions, no I/O, no models. Everything in here is locked by the
canonical contracts and reproduced in other people's code, so a change here is
a contract change and needs the section 7 protocol in the repository manual.
"""

from ai.normalize.matching import (
    CONFUSION_PAIRS,
    GRAMMAR_CONFIDENCE_PENALTY,
    derive_match_state,
    fuzzy_candidates,
    plate_distance,
)
from ai.normalize.plate import (
    INDIAN_PLATE_GRAMMAR,
    grammar_ok,
    normalize_plate,
)

__all__ = [
    "CONFUSION_PAIRS",
    "GRAMMAR_CONFIDENCE_PENALTY",
    "INDIAN_PLATE_GRAMMAR",
    "derive_match_state",
    "fuzzy_candidates",
    "grammar_ok",
    "normalize_plate",
    "plate_distance",
]
