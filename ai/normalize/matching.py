"""match_state derivation and candidate-only fuzzy matching.

Two rules govern this module, and both are safety rules rather than accuracy
rules:

1. match_state is DESCRIPTIVE, not an investigative conclusion. It never
   appears in the UI as "confirmed". Contracts section 3.3.

2. Fuzzy matching GENERATES CANDIDATES ONLY. It may never silently rewrite
   plate.normalized, never produce match_state "exact", and never independently
   raise a confirmed watchlist alert. Contracts section 4.6.
"""

from typing import Iterable, Optional

from ai.normalize.plate import grammar_ok

# Confidence multiplier applied when the fused plate fails the soft grammar
# check. Downgrade, never discard -- see ai/normalize/plate.py.
GRAMMAR_CONFIDENCE_PENALTY = 0.85

# Character pairs that OCR confuses on Indian plates. Substituting one of these
# costs less than an unrelated substitution, so 'GJ01A81234' scores as a near
# neighbour of 'GJ01AB1234' while 'GJ01AZ1234' does not.
# Owner's manual section 11.
CONFUSION_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"0", "O"}),
    frozenset({"1", "I", "L"}),
    frozenset({"8", "B"}),
    frozenset({"5", "S"}),
    frozenset({"2", "Z"}),
    frozenset({"6", "G"}),
)

_CONFUSABLE: dict[str, frozenset[str]] = {}
for _group in CONFUSION_PAIRS:
    for _ch in _group:
        _CONFUSABLE[_ch] = _group

# A confusion substitution costs less than a full one. The exact value is a
# tuning knob, not a contract: it only reorders a candidate list a human reads.
_CONFUSION_COST = 0.4
_SUBSTITUTION_COST = 1.0
_GAP_COST = 1.0


# COPIED FROM CANONICAL CONTRACTS -- DO NOT EDIT HERE (Contracts section 3.3).
def derive_match_state(
    evidence_count: int,
    fused_confidence: float,
    exact_watchlist_hit: bool,
) -> str:
    if exact_watchlist_hit:
        return "exact"
    if evidence_count >= 2 and fused_confidence >= 0.80:
        return "probable"
    return "low_confidence"
# END COPIED BLOCK.


def match_state_for(
    normalized: Optional[str],
    evidence_count: int,
    fused_confidence: float,
    exact_watchlist_hit: bool = False,
) -> str:
    """The full four-value derivation, including the unreadable case.

    The canonical function above covers three of the four states; a plate that
    was located but produced no usable text never reaches it. Handling that
    here keeps the copied block verbatim and still gives callers one function
    that cannot return an inconsistent pair.
    """
    if not normalized:
        # A located plate with no text. Never alerts. Contracts section 3.4.
        return "unreadable"
    return derive_match_state(evidence_count, fused_confidence, exact_watchlist_hit)


def plate_distance(a: str, b: str) -> float:
    """Weighted edit distance between two normalized plates.

    Levenshtein, except that substituting a known OCR confusion pair costs
    _CONFUSION_COST instead of 1.0. Lower is closer. Both inputs are assumed
    already normalized.

    Used only to rank a candidate list. Never used to decide equality --
    exact normalized equality is the only exact match there is.
    """
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    previous = [j * _GAP_COST for j in range(len(b) + 1)]
    for i, ch_a in enumerate(a, start=1):
        current = [i * _GAP_COST]
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                sub_cost = 0.0
            elif _CONFUSABLE.get(ch_a) is not None and _CONFUSABLE.get(ch_a) is _CONFUSABLE.get(ch_b):
                sub_cost = _CONFUSION_COST
            else:
                sub_cost = _SUBSTITUTION_COST
            current.append(
                min(
                    previous[j] + _GAP_COST,      # deletion
                    current[j - 1] + _GAP_COST,   # insertion
                    previous[j - 1] + sub_cost,   # substitution
                )
            )
        previous = current
    return previous[-1]


def fuzzy_candidates(
    query: str,
    known_plates: Iterable[str],
    *,
    max_distance: float = 2.0,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Ranked near neighbours of query. Candidates only.

    Returns dicts with plate, distance and match_state, where match_state is
    capped at 'probable' and is never 'exact'. The caller renders these as
    "candidate match, requires review" and nothing stronger.

    An exact hit is deliberately excluded from this list: exact belongs to
    exact search, and merging the two into one undifferentiated list is how a
    fuzzy guess ends up presented as a confirmed identification.
    """
    scored: list[tuple[float, str]] = []
    for plate in known_plates:
        if not plate or plate == query:
            continue
        distance = plate_distance(query, plate)
        if distance <= max_distance:
            scored.append((distance, plate))

    scored.sort(key=lambda pair: (pair[0], pair[1]))
    return [
        {
            "plate": plate,
            "distance": round(distance, 3),
            # Capped by construction. Contracts section 4.6.
            "match_state": "probable",
            "grammar_ok": grammar_ok(plate),
        }
        for distance, plate in scored[:limit]
    ]


def apply_grammar_penalty(confidence: float, normalized: Optional[str]) -> float:
    """Downgrade confidence when the plate fails the soft grammar check.

    Returns the confidence unchanged when the grammar passes or the plate is
    absent. Never touches the string itself.
    """
    if not normalized or grammar_ok(normalized):
        return confidence
    return confidence * GRAMMAR_CONFIDENCE_PENALTY
