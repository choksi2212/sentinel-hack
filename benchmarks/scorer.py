#!/usr/bin/env python3
"""E2E correct-plate event rate. Per SPEC_BENCHMARK.md §2.

Only label_source in ("human", "synthetic_truth") rows are scored at all --
ocr_candidate rows are excluded from every reported number, full stop (an
OCR-produced label scored against an OCR-produced prediction would measure
the system against itself). Within scoreable rows:
  - eligible == False: fabrication_count += 1 if a prediction was emitted,
    excluded from n_eligible/n_correct entirely.
  - eligible == True: counted in n_eligible; correct iff normalised strings
    match exactly. No OCR-confusion fuzzing in the match itself (SPEC_BENCHMARK
    §2) -- 0/O, 1/I, 8/B confusions are tracked separately as a diagnostic.
"""
from collections import defaultdict

WIDTH_BUCKETS = [">100", "80-100", "60-80", "40-60", "30-40", "<30"]
SLICES = ["easy", "motion_blur", "night", "glare", "perspective", "tiny"]
_CONFUSABLE = {"0": "O", "O": "0", "1": "I", "I": "1", "8": "B", "B": "8"}


def normalize(text: str | None) -> str | None:
    """Uppercase, strip whitespace and hyphens. Nothing else (SPEC_BENCHMARK §2)."""
    if text is None:
        return None
    return text.upper().replace(" ", "").replace("-", "")


def fuzzy_equivalent(a: str | None, b: str | None) -> bool:
    """True if a and b differ only by OCR-confusable substitutions, same length."""
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    if a == b:
        return True
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if _CONFUSABLE.get(ca) != cb:
            return False
    return True


def _empty_bucket_counts():
    return {"n": 0, "correct": 0}


def score(rows: list[dict], predictions: dict[str, str | None]) -> dict:
    """rows: TRINETRA-HARD rows (any label_source). predictions: obs_id -> text|None."""
    n_eligible = 0
    n_correct = 0
    fabrication_count = 0
    fuzzy_count = 0
    by_width = {b: _empty_bucket_counts() for b in WIDTH_BUCKETS}
    by_slice_correct = defaultdict(int)
    by_slice_n = defaultdict(int)

    for row in rows:
        if row.get("label_source") not in ("human", "synthetic_truth"):
            continue  # ocr_candidate: excluded from every reported number
        pred = predictions.get(row["obs_id"])
        if not row["eligible"]:
            if pred is not None:
                fabrication_count += 1
            continue  # never enters n_eligible/n_correct
        n_eligible += 1
        gt_norm = normalize(row["plate_text"])
        pred_norm = normalize(pred)
        correct = pred_norm is not None and pred_norm == gt_norm
        if correct:
            n_correct += 1
        elif pred_norm is not None and fuzzy_equivalent(pred_norm, gt_norm):
            fuzzy_count += 1

        bucket = by_width[row["width_bucket"]]
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        by_slice_n[row["slice"]] += 1
        by_slice_correct[row["slice"]] += int(correct)

    by_width_out = {
        b: {"n": c["n"], "correct": c["correct"], "rate": (c["correct"] / c["n"]) if c["n"] else None}
        for b, c in by_width.items()
    }
    by_slice_out = {
        s: ((by_slice_correct[s] / by_slice_n[s]) if by_slice_n.get(s) else None)
        for s in SLICES
    }
    return {
        "n_eligible": n_eligible,
        "n_correct": n_correct,
        "e2e_correct_plate_event_rate": (n_correct / n_eligible) if n_eligible else None,
        "fabrication_count": fabrication_count,
        "by_plate_width": by_width_out,
        "by_slice": by_slice_out,
        "fuzzy_match_rate": (fuzzy_count / n_eligible) if n_eligible else None,
    }


def demo():
    """The six SPEC_BENCHMARK §5 fixtures, each isolated to prove the exact behaviour."""
    base = {
        "eligible": True, "label_source": "human",
        "width_bucket": "40-60", "slice": "easy",
    }

    # 1. exact match -> counted correct
    rows = [{**base, "obs_id": "a", "plate_text": "GJ01AB1234"}]
    r = score(rows, {"a": "GJ01AB1234"})
    assert r["n_correct"] == 1 and r["n_eligible"] == 1

    # 2. case/space differs -> counted correct after normalisation
    rows = [{**base, "obs_id": "b", "plate_text": "GJ01AB1234"}]
    r = score(rows, {"b": "gj 01-ab 1234"})
    assert r["n_correct"] == 1

    # 3. 0 vs O -> counted incorrect, fuzzy_match_rate increments
    rows = [{**base, "obs_id": "c", "plate_text": "GJ01AB1234"}]
    r = score(rows, {"c": "GJO1AB1234"})  # ground truth's leading '0' predicted as 'O'
    assert r["n_correct"] == 0
    assert r["fuzzy_match_rate"] == 1.0

    # 4. no prediction -> counted incorrect, no crash
    rows = [{**base, "obs_id": "d", "plate_text": "GJ01AB1234"}]
    r = score(rows, {})
    assert r["n_correct"] == 0 and r["n_eligible"] == 1

    # 5. eligible: false + prediction emitted -> fabrication_count increments, excluded from rate
    rows = [{**base, "obs_id": "e", "plate_text": None, "eligible": False}]
    r = score(rows, {"e": "ANYTHING"})
    assert r["fabrication_count"] == 1
    assert r["n_eligible"] == 0

    # 6. label_source: ocr_candidate -> excluded from every reported number
    rows = [{**base, "obs_id": "f", "plate_text": "GJ01AB1234", "label_source": "ocr_candidate"}]
    r = score(rows, {"f": "WRONG"})
    assert r["n_eligible"] == 0 and r["fabrication_count"] == 0 and r["n_correct"] == 0

    # synthetic_truth is scoreable exactly like human (Phase 4R)
    rows = [{**base, "obs_id": "g", "plate_text": "GJ01AB1234", "label_source": "synthetic_truth"}]
    r = score(rows, {"g": "GJ01AB1234"})
    assert r["n_correct"] == 1 and r["n_eligible"] == 1

    print("demo: all six SPEC_BENCHMARK §5 fixtures pass, plus synthetic_truth scoreable")


if __name__ == "__main__":
    demo()
