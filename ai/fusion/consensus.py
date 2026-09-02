"""Temporal OCR consensus. Canonical Contracts section 4.3.

Worked example, which is also the walkthrough slide:

    Frame  OCR text      conf   quality   weight
    1      GJ01AB1234    0.91   0.90      0.819
    2      GJ01AB1234    0.94   0.92      0.865
    3      GJ01AB1234    0.88   0.87      0.766
    4      GJ01A81234    0.63   0.55      0.347

GJ01AB1234 = 2.450 across 3 frames versus GJ01A81234 = 0.347 across 1, so the
fused answer is GJ01AB1234 with evidence_count 3 and confidence 0.876. A
single-frame system that happened to sample frame 4 would have been confidently
wrong -- and confidently wrong is worse than unreadable in a police system.
"""

from collections import defaultdict
from typing import Iterable, Optional, Sequence

from ai.contracts.stages import FusedPlate, PlateObservation
from ai.normalize.matching import apply_grammar_penalty
from ai.normalize.plate import grammar_ok, normalize_plate


# COPIED FROM CANONICAL CONTRACTS -- DO NOT EDIT HERE (Contracts section 4.3).
def fuse(observations):
    """observations: [{text, ocr_confidence, image_quality}, ...] for ONE TrackKey."""
    score, count = defaultdict(float), defaultdict(int)
    for o in observations:
        key = normalize_plate(o["text"])
        if not key:
            continue
        score[key] += o["ocr_confidence"] * o["image_quality"]
        count[key] += 1
    if not score:
        return None
    best = max(score, key=score.get)
    return {
        "normalized": best,
        "confidence": score[best] / sum(score.values()),   # share of total evidence
        "evidence_count": count[best],
    }
# END COPIED BLOCK.


def fuse_observations(
    observations: Sequence[PlateObservation],
    *,
    apply_grammar_downgrade: bool = True,
) -> Optional[FusedPlate]:
    """Typed wrapper around fuse() for one TrackKey.

    Returns None when nothing usable was observed -- either no observation produced any
    characters at all, or none of them carried any evidence weight. Both are the unreadable
    case, and both are correct outcomes: the caller emits plate: null rather than reaching
    for a guess.

    The confidence returned is a SHARE OF TOTAL EVIDENCE, not a probability.
    Do not multiply it by the detector confidence and present the product;
    the arithmetic is meaningless on uncalibrated scores and someone will ask.
    Contracts section 4.4.
    """
    if not observations:
        return None

    _assert_single_track(observations)

    # Drop observations carrying no evidence weight before fusing. Two reasons, and the
    # second one is why this is not merely defensive.
    #
    # The copied block divides by the total weight, so a track where every reading scored
    # exactly zero -- an OCR engine reporting no confidence at all, or a crop the quality
    # scorer rated worthless -- raises ZeroDivisionError and takes the frame down with it.
    #
    # More importantly, a zero-weight reading contributes nothing to the weighted vote and
    # still increments evidence_count. evidence_count is what promotes a plate to
    # "probable" (Contracts 3.3) and to the HIGH calibration band (4.4), so one real read
    # plus two worthless ones that happened to agree with it would be reported as
    # three-frame corroboration. Weightless agreement is not corroboration.
    weighted = [obs for obs in observations if obs.fusion_weight > 0.0]
    if not weighted:
        return None

    raw_result = fuse(
        [
            {
                "text": obs.plate_raw,
                "ocr_confidence": obs.ocr_confidence,
                "image_quality": obs.image_quality,
            }
            for obs in weighted
        ]
    )
    if raw_result is None:
        return None

    normalized = raw_result["normalized"]
    confidence = float(raw_result["confidence"])
    ok = grammar_ok(normalized)
    if apply_grammar_downgrade:
        confidence = apply_grammar_penalty(confidence, normalized)

    return FusedPlate(
        normalized=normalized,
        confidence=confidence,
        evidence_count=int(raw_result["evidence_count"]),
        best_observation=best_agreeing_observation(observations, normalized),
        grammar_ok=ok,
        total_observations=len(observations),
    )


def best_agreeing_observation(
    observations: Iterable[PlateObservation],
    normalized: str,
) -> Optional[PlateObservation]:
    """The highest-weight observation that agrees with the fused answer.

    Its crop is the snapshot we keep and its bbox is the one that goes on the
    wire. Taking the globally best observation instead would occasionally
    attach the image of the outlier reading to the consensus string -- an
    event whose evidence contradicts its own plate field, which is exactly the
    kind of detail that unravels a demo under questioning.
    """
    agreeing = [
        obs for obs in observations if normalize_plate(obs.plate_raw) == normalized
    ]
    if not agreeing:
        return None
    return max(agreeing, key=lambda obs: obs.fusion_weight)


def consensus_gain(observations: Sequence[PlateObservation]) -> dict[str, object]:
    """Diagnostic: what fusion changed versus trusting one frame.

    single_frame_pick is what a naive pipeline would have emitted -- the
    highest-weight individual read. changed is True when consensus disagreed
    with it. Aggregated over a benchmark run this is the before-versus-after
    number to ask Akshat for.
    """
    fused = fuse_observations(observations, apply_grammar_downgrade=False)
    if fused is None:
        return {
            "single_frame_pick": None,
            "fused": None,
            "changed": False,
            "observations": len(observations),
        }

    best_single = max(observations, key=lambda obs: obs.fusion_weight)
    single_pick = normalize_plate(best_single.plate_raw) or None

    return {
        "single_frame_pick": single_pick,
        "fused": fused.normalized,
        "changed": single_pick != fused.normalized,
        "evidence_count": fused.evidence_count,
        "observations": len(observations),
        "distinct_readings": len(
            {normalize_plate(o.plate_raw) for o in observations if normalize_plate(o.plate_raw)}
        ),
    }


def _assert_single_track(observations: Sequence[PlateObservation]) -> None:
    """Fusion is per-TrackKey. Mixing tracks would invent a vehicle.

    Cheap guard, but it catches the one mistake in this module that produces
    plausible-looking wrong output instead of an exception: passing the buffer
    for the whole camera instead of the buffer for one track.
    """
    keys = {obs.track_key for obs in observations}
    if len(keys) > 1:
        raise ValueError(
            f"fuse_observations received {len(keys)} distinct TrackKeys "
            f"({sorted(str(k) for k in keys)}); fusion is per-TrackKey only"
        )
