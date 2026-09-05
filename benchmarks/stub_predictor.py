#!/usr/bin/env python3
"""Canned predictor. Exists to validate the scorer/report/delta plumbing
before the real pipeline is ready (SPEC_BENCHMARK §5). Not a model.

fusion=False: predicts nothing for anything harder than "easy"/">100" width
(simulating a weaker no-fusion pipeline). fusion=True: also gets small/hard
buckets right most of the time. Both intentionally get a fixed few obs_ids
wrong so the delta table has real before/after movement to show, and both
always emit *something* (even for ineligible rows) so the fabrication-count
path is exercised, not just the happy path.
"""

_ALWAYS_WRONG = {"th_0001"}  # deliberately mis-predicted regardless of fusion


def predict(row: dict, fusion_enabled: bool) -> str | None:
    obs_id = row["obs_id"]
    ground_truth = row.get("plate_text")
    if obs_id in _ALWAYS_WRONG:
        return "ZZ99ZZ9999"
    if ground_truth is None:
        return "FABRICATED0" if not row.get("eligible", True) else None
    width_bucket = row.get("width_bucket")
    if fusion_enabled:
        return ground_truth  # fusion recovers everything else
    # no fusion: only the easy, wide buckets come out correct
    if width_bucket in (">100", "80-100"):
        return ground_truth
    return None


def demo():
    easy = {"obs_id": "x1", "plate_text": "GJ01AB1234", "width_bucket": ">100", "eligible": True}
    hard = {"obs_id": "x2", "plate_text": "GJ01AB1234", "width_bucket": "<30", "eligible": True}
    ineligible = {"obs_id": "x3", "plate_text": None, "width_bucket": "<30", "eligible": False}

    assert predict(easy, fusion_enabled=False) == "GJ01AB1234"
    assert predict(hard, fusion_enabled=False) is None
    assert predict(hard, fusion_enabled=True) == "GJ01AB1234"
    assert predict(ineligible, fusion_enabled=False) == "FABRICATED0"
    assert predict({**easy, "obs_id": "th_0001"}, fusion_enabled=True) == "ZZ99ZZ9999"
    print("demo: all assertions passed")


if __name__ == "__main__":
    demo()
