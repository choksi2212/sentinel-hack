#!/usr/bin/env python3
"""Canned predictor. Exists to validate the scorer/report/delta plumbing
before the real pipeline is ready (SPEC_BENCHMARK §5). Not a model.

Probabilistic, not threshold-based: a real model produces partial credit in
every bucket, so the stub does too, seeded per-row for full determinism
(same obs_id + fusion flag always yields the same prediction). Base hit rate
by width_bucket (fusion off), with an uplift when fusion is on that is
largest in the middle buckets and small at both ends (fusion helps most
where a single frame is borderline, least where it's already easy or where
even multiple frames rarely help):

  bucket    off    on-uplift
  >100      0.95   +0.03
  80-100    0.88   +0.05
  60-80     0.72   +0.15
  40-60     0.51   +0.20
  30-40     0.30   +0.12
  <30       0.12   +0.05

A miss returns a wrong-but-plausible plate string (one character corrupted)
85% of the time, and None (no detection) the other 15% -- both paths are
exercised, not just the "no prediction" one.
"""
import random

BASE_HIT_RATE = {
    ">100": 0.95, "80-100": 0.88, "60-80": 0.72,
    "40-60": 0.51, "30-40": 0.30, "<30": 0.12,
}
FUSION_UPLIFT = {
    ">100": 0.03, "80-100": 0.05, "60-80": 0.15,
    "40-60": 0.20, "30-40": 0.12, "<30": 0.05,
}
CORRUPT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WRONG_STRING_RATE = 0.85  # remaining misses return None (no detection)


def _corrupt(text: str, rng: random.Random) -> str:
    """One realistic single-character OCR-style error."""
    if not text:
        return text
    pos = rng.randrange(len(text))
    original = text[pos]
    replacement = rng.choice([c for c in CORRUPT_CHARS if c != original])
    return text[:pos] + replacement + text[pos + 1:]


def predict(row: dict, fusion_enabled: bool) -> str | None:
    ground_truth = row.get("plate_text")
    if ground_truth is None:
        # true text is genuinely unknown (e.g. indian_road) -- only the
        # fabrication path applies, never a text guess.
        return "FABRICATED0" if not row.get("eligible", True) else None

    rng = random.Random(f"{row['obs_id']}:{fusion_enabled}")
    width_bucket = row.get("width_bucket")
    hit_rate = BASE_HIT_RATE.get(width_bucket, 0.5)
    if fusion_enabled:
        hit_rate = min(hit_rate + FUSION_UPLIFT.get(width_bucket, 0.0), 0.99)

    if rng.random() < hit_rate:
        return ground_truth
    if rng.random() < WRONG_STRING_RATE:
        return _corrupt(ground_truth, rng)
    return None


def demo():
    easy = {"obs_id": "x1", "plate_text": "GJ01AB1234", "width_bucket": ">100", "eligible": True}
    ineligible = {"obs_id": "x3", "plate_text": None, "width_bucket": "<30", "eligible": False}

    # deterministic: same obs_id + fusion flag -> same prediction every call
    assert predict(easy, fusion_enabled=False) == predict(easy, fusion_enabled=False)
    assert predict(ineligible, fusion_enabled=False) == "FABRICATED0"

    # over many distinct rows, fusion-on hit rate should exceed fusion-off for a
    # borderline bucket (40-60), and no output should ever equal a *different*
    # ground truth than the row's own (no cross-row contamination)
    hits_off = hits_on = 0
    n = 500
    for i in range(n):
        row = {"obs_id": f"probe_{i}", "plate_text": "GJ01AB1234", "width_bucket": "40-60", "eligible": True}
        if predict(row, fusion_enabled=False) == "GJ01AB1234":
            hits_off += 1
        if predict(row, fusion_enabled=True) == "GJ01AB1234":
            hits_on += 1
    assert hits_on > hits_off, (hits_off, hits_on)
    assert 0.40 < hits_off / n < 0.62  # near the 0.51 target, seeded so this is stable
    print("demo: all assertions passed")


if __name__ == "__main__":
    demo()
