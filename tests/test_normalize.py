"""Plate normalization, match_state derivation, and the fuzzy-matching safety rules.

Contracts 4.2, 3.3 and 4.6. The tests here divide into two kinds and the second kind
matters more.

The first kind checks that normalization does what it says: separators and case go away,
`GJ 01 AB 1234` and `gj-01-ab-1234` both become `GJ01AB1234`.

The second kind checks the things this module is forbidden to do. Normalization must not
touch `raw`. The grammar check must not filter. Fuzzy matching must not produce an exact
match, must not rewrite anything, and must not return the query itself as its own
neighbour. Each of those is a safety rule in a police system rather than an accuracy
choice, and each is one plausible-looking edit away from being broken by someone trying to
improve the hit rate.
"""

import pytest

from ai.normalize.matching import (
    CONFUSION_PAIRS,
    GRAMMAR_CONFIDENCE_PENALTY,
    apply_grammar_penalty,
    derive_match_state,
    fuzzy_candidates,
    match_state_for,
    plate_distance,
)
from ai.normalize.plate import grammar_ok, looks_like_partial, normalize_plate

# ------------------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("GJ 01 AB 1234", "GJ01AB1234"),   # the contract's own example
        ("gj-01-ab-1234", "GJ01AB1234"),   # the contract's second example
        ("GJ01AB1234", "GJ01AB1234"),      # already normalized, idempotent
        ("  GJ01AB1234  ", "GJ01AB1234"),  # leading and trailing whitespace
        ("GJ.01.AB.1234", "GJ01AB1234"),
        ("GJ/01/AB/1234", "GJ01AB1234"),
        ("GJ\t01\nAB 1234", "GJ01AB1234"),
        ("gj01ab1234", "GJ01AB1234"),
        ("MH12DE9812", "MH12DE9812"),
        ("", ""),
    ],
)
def test_normalize_plate(raw, expected):
    assert normalize_plate(raw) == expected


def test_normalize_is_idempotent():
    """Normalizing twice equals normalizing once, for every fixture-shaped input.

    The pipeline normalizes at fusion and the API normalizes the search query. If those
    were not the same function applied to the same effect, a plate stored one way and
    searched another would silently never match.
    """
    for raw in ("GJ 01 AB 1234", "gj-01-ab-1234", "MH 12 DE 9812", "GJ05CK4471", ""):
        once = normalize_plate(raw)
        assert normalize_plate(once) == once


def test_normalize_does_not_mutate_its_input():
    """Contracts 4.2: normalization is NEVER applied to raw in place.

    Python strings are immutable so this cannot fail as written, which is the point --
    the test exists to fail loudly if `raw` is ever made a mutable buffer or the function
    is ever handed an object it writes back to. `raw` is the audit trail. An operator
    asked in court what the camera actually read needs the unmodified string.
    """
    raw = "GJ 01 AB 1234"
    normalize_plate(raw)
    assert raw == "GJ 01 AB 1234"


def test_normalize_keeps_digits_that_look_like_separators():
    """A zero is not a separator. Stripping it would change the vehicle."""
    assert normalize_plate("GJ 00 AB 0000") == "GJ00AB0000"


def test_normalize_drops_unicode_lookalikes_rather_than_keeping_them():
    """An en dash and a full-width digit are both outside [A-Z0-9].

    Operators paste plate numbers out of PDFs and WhatsApp. The regex is an allowlist, so
    anything exotic is removed rather than smuggled into the search key -- which means a
    full-width digit is dropped, not silently converted. That is the safe direction: a
    shorter key fails to match, where a wrongly converted one matches the wrong vehicle.

    Written as escapes rather than literal characters so this file stays pure ASCII and the
    codepoint under test is stated outright rather than left to an editor round trip.
    """
    assert normalize_plate("GJ\u201301 AB 1234") == "GJ01AB1234"   # U+2013 EN DASH
    assert normalize_plate("GJ01AB123\uff14") == "GJ01AB123"      # U+FF14 FULLWIDTH FOUR


# ------------------------------------------------------------------------- soft grammar


@pytest.mark.parametrize(
    "plate",
    [
        "GJ01AB1234",   # two-digit RTO, two-letter series
        "GJ1AB1234",    # one-digit RTO
        "GJ01A1234",    # one-letter series
        "GJ01ABC1234",  # three-letter series
        "GJ011234",     # no series letters at all -- older format
        "MH12DE9812",
    ],
)
def test_grammar_accepts_the_common_shapes(plate):
    assert grammar_ok(plate)


@pytest.mark.parametrize(
    "plate",
    [
        "",
        "GJ01AB123",     # three trailing digits
        "GJ01AB12345",   # five trailing digits
        "G01AB1234",     # one state letter
        "GJ01ABCD1234",  # four series letters
        "1234AB01GJ",    # reversed
        "GJ 01 AB 1234", # not normalized -- grammar runs after normalization, never before
    ],
)
def test_grammar_rejects_these(plate):
    assert not grammar_ok(plate)


def test_grammar_failure_downgrades_confidence_and_leaves_the_string_alone():
    """The whole design of the grammar check, in one assertion.

    BH-series plates, older formats, diplomatic and military plates all fail this pattern
    and are all real vehicles. So a miss costs confidence and nothing else. A hard filter
    here would silently delete real vehicles from a police search, which is strictly worse
    than returning them with a lower score.
    """
    unusual = "22BH1234AA"  # a real BH-series format the pattern does not know
    assert not grammar_ok(unusual)

    penalized = apply_grammar_penalty(0.90, unusual)
    assert penalized == pytest.approx(0.90 * GRAMMAR_CONFIDENCE_PENALTY)
    assert penalized > 0.0, "a grammar miss must never zero the observation out"
    assert unusual == "22BH1234AA", "the string itself is never rewritten"


def test_grammar_pass_leaves_confidence_untouched():
    assert apply_grammar_penalty(0.93, "GJ01AB1234") == 0.93


def test_grammar_penalty_on_an_absent_plate_is_a_no_op():
    """No plate is not a grammar failure. There is nothing to be wrong about."""
    assert apply_grammar_penalty(0.5, None) == 0.5
    assert apply_grammar_penalty(0.5, "") == 0.5


def test_penalty_is_a_downgrade_not_a_rejection_threshold():
    """0.85 keeps a strong reading above the probable threshold. Deliberately.

    An unusual but confidently-read plate should still be `probable`. If the penalty were
    harsh enough to push 0.95 under 0.80, every BH-series plate in the state would be
    permanently low_confidence.
    """
    assert apply_grammar_penalty(0.95, "22BH1234AA") >= 0.80


@pytest.mark.parametrize("fragment", ["GJ01AB", "GJ01A", "J01AB12"])
def test_looks_like_partial_accepts_fragments(fragment):
    assert looks_like_partial(fragment)


@pytest.mark.parametrize("noise", ["", "AB", "###", "GJ01AB1234", "ABCDEFG", "1234567"])
def test_looks_like_partial_rejects_noise_and_complete_plates(noise):
    """A complete valid plate is not a partial, and neither is letters-only noise.

    This separates ocr_partial from ocr_wrong in the failure taxonomy. A fragment is still
    emitted with its own low confidence: `GJ01AB` plus a camera and a timestamp is
    investigatively useful, and inventing the missing digits is not.
    """
    assert not looks_like_partial(noise)


# ------------------------------------------------------------------- match_state (3.3)


@pytest.mark.parametrize(
    "evidence,confidence,watchlist,expected",
    [
        (1, 0.10, True, "exact"),        # a watchlist hit wins outright
        (4, 0.94, True, "exact"),
        (2, 0.80, False, "probable"),    # both thresholds exactly met
        (3, 0.85, False, "probable"),
        (2, 0.7999, False, "low_confidence"),  # confidence one hair short
        (1, 0.99, False, "low_confidence"),    # one observation, however confident
        (0, 0.99, False, "low_confidence"),
    ],
)
def test_derive_match_state(evidence, confidence, watchlist, expected):
    assert derive_match_state(evidence, confidence, watchlist) == expected


def test_probable_needs_both_conditions_not_either():
    """Contracts 3.3 joins the two with `and`. Flipping it to `or` is the likely bug.

    One very confident frame is exactly the single-frame OCR error the temporal fusion
    stage exists to suppress. If `or` crept in, that error would be labelled `probable` and
    shown to an operator as a corroborated reading.
    """
    assert derive_match_state(1, 0.99, False) == "low_confidence"
    assert derive_match_state(9, 0.10, False) == "low_confidence"
    assert derive_match_state(2, 0.80, False) == "probable"


def test_unreadable_is_reachable_only_through_match_state_for():
    """The canonical block covers three states; the fourth is the no-text case.

    Kept out of the copied block so that block stays verbatim against the contract, and
    provided here so no caller has to hand-assemble a state/plate pair that could be
    inconsistent.
    """
    assert match_state_for(None, 4, 0.99) == "unreadable"
    assert match_state_for("", 4, 0.99) == "unreadable"
    assert derive_match_state(4, 0.99, False) != "unreadable"


def test_a_plate_with_no_text_never_alerts_even_on_a_watchlist_hit():
    """Contracts 3.4. There is no text to have matched, so the flag is meaningless.

    Passing exact_watchlist_hit=True with no plate is caller error, and the safe response
    to caller error in an alerting path is to stay quiet.
    """
    assert match_state_for(None, 4, 0.99, exact_watchlist_hit=True) == "unreadable"


def test_match_state_for_agrees_with_the_canonical_block_whenever_a_plate_exists():
    for evidence in (0, 1, 2, 5):
        for confidence in (0.0, 0.5, 0.80, 0.99):
            for hit in (False, True):
                assert match_state_for("GJ01AB1234", evidence, confidence, hit) == (
                    derive_match_state(evidence, confidence, hit)
                )


# ------------------------------------------------------------- fuzzy matching (4.6)


def test_confusion_pairs_are_the_six_the_manual_lists():
    assert {frozenset(g) for g in CONFUSION_PAIRS} == {
        frozenset("0O"), frozenset("1IL"), frozenset("8B"),
        frozenset("5S"), frozenset("2Z"), frozenset("6G"),
    }


def test_identical_plates_are_distance_zero():
    assert plate_distance("GJ01AB1234", "GJ01AB1234") == 0.0


def test_a_confusion_substitution_is_cheaper_than_an_unrelated_one():
    """The one property that makes the candidate ranking useful.

    B->8 is a shape confusion a camera makes. B->Z is not. If both cost the same, the
    candidate list is ordered by nothing an operator cares about.
    """
    confusable = plate_distance("GJ01A81234", "GJ01AB1234")
    unrelated = plate_distance("GJ01AZ1234", "GJ01AB1234")
    assert 0 < confusable < unrelated


@pytest.mark.parametrize("a,b", [("0", "O"), ("1", "I"), ("I", "L"), ("8", "B"), ("5", "S"), ("2", "Z"), ("6", "G")])
def test_every_confusion_pair_is_cheap_in_both_directions(a, b):
    forward = plate_distance(f"GJ01{a}B1234", f"GJ01{b}B1234")
    backward = plate_distance(f"GJ01{b}B1234", f"GJ01{a}B1234")
    assert forward == backward, "distance must be symmetric"
    assert 0 < forward < 1.0


def test_distance_is_symmetric_and_handles_empty():
    assert plate_distance("GJ01AB1234", "") == 10.0
    assert plate_distance("", "GJ01AB1234") == 10.0
    assert plate_distance("", "") == 0.0


def test_distance_counts_length_changes():
    assert plate_distance("GJ01AB123", "GJ01AB1234") == 1.0


def test_fuzzy_candidates_never_returns_an_exact_match_state():
    """Contracts 4.6, the rule this whole module is constrained by.

    `exact` means normalized string equality. A fuzzy neighbour is a guess, and a guess
    labelled `exact` reaches an operator as a confirmed identification of a vehicle that
    was never seen.
    """
    known = ["GJ01AB1234", "GJ01AB1284", "GJ01A81234", "MH12DE9812"]
    for candidate in fuzzy_candidates("GJ01AB1234", known):
        assert candidate["match_state"] == "probable"
        assert candidate["match_state"] != "exact"


def test_fuzzy_candidates_excludes_the_query_itself():
    """The exact hit belongs to exact search, and only there.

    If the query came back inside the candidate list, a client rendering one merged list
    would show a real match and three guesses with nothing distinguishing them.
    """
    known = ["GJ01AB1234", "GJ01AB1284"]
    plates = [c["plate"] for c in fuzzy_candidates("GJ01AB1234", known)]
    assert "GJ01AB1234" not in plates
    assert plates == ["GJ01AB1284"]


def test_fuzzy_candidates_are_ranked_closest_first():
    known = ["GJ01AB1284", "GJ01A81234", "GJ99XY0000"]
    results = fuzzy_candidates("GJ01AB1234", known, max_distance=3.0)
    distances = [c["distance"] for c in results]
    assert distances == sorted(distances)
    assert results[0]["plate"] == "GJ01A81234", "the confusion-pair neighbour is closest"


def test_fuzzy_candidates_respects_max_distance_and_limit():
    known = [f"GJ01AB{n:04d}" for n in range(50)]
    assert fuzzy_candidates("GJ01AB1234", known, max_distance=0.0) == []
    assert len(fuzzy_candidates("GJ01AB1234", known, max_distance=9.0, limit=3)) == 3


def test_fuzzy_candidates_does_not_rewrite_the_query_or_the_corpus():
    """It reads. It does not write. Asserted because the tempting optimisation is to
    canonicalize the corpus in place on the way past, and the corpus is the sightings
    table."""
    known = ["GJ01AB1284", "gj01ab1284"]
    snapshot = list(known)
    query = "GJ01AB1234"
    fuzzy_candidates(query, known)
    assert known == snapshot
    assert query == "GJ01AB1234"


def test_fuzzy_candidates_skips_empty_corpus_entries():
    """An unreadable sighting stores no normalized plate. It is not a neighbour of
    everything."""
    assert fuzzy_candidates("GJ01AB1234", ["", None and "", "GJ01AB1284"], max_distance=2.0) == [
        {"plate": "GJ01AB1284", "distance": 1.0, "match_state": "probable", "grammar_ok": True}
    ]


def test_fuzzy_candidates_report_grammar_without_filtering_on_it():
    """A candidate that fails the grammar check is still returned, flagged.

    Same reasoning as the confidence penalty: the operator decides, and the unusual
    formats that fail this pattern are real plates.
    """
    results = fuzzy_candidates("GJ01AB1234", ["GJ01AB123"], max_distance=2.0)
    assert len(results) == 1
    assert results[0]["grammar_ok"] is False
