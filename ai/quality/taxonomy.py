"""The failure taxonomy -- the D3 deliverable that gates the A100.

Owner's manual section 6. After the baseline benchmark, every miss is
classified into exactly one of ten buckets, and the shape of that histogram is
the precondition for spending money:

    One dominant bucket is required before training anything. Two co-dominant
    buckets means the analysis is not finished, and a fine-tune launched on a
    guess is a guess with a credit card attached.

This module is the counting half of that deliverable. It does no classification
by itself -- a human (or the benchmark harness) decides which bucket a given
miss falls in, because the distinction between `ocr_wrong` and `ocr_partial`,
or between `track_broken` and `track_merged`, is a judgement about a specific
clip. What it does is hold the ten buckets as data, tally misses by bucket, and
turn the tally into a verdict that says one of three things: a single bucket
dominates, several co-dominate (stop and finish the analysis), or too few
misses have been classified to say either way.

The verdict carries one project-specific fact that the raw histogram does not:
whether the dominant bucket points at *this lane's* paid work. Only `ocr_wrong`
does -- it is the one miss a PaddleOCR recogniser fine-tune (config/training.yaml)
could fix. `plate_too_small` has no software fix at all; the rest are real
remedies that belong to other lanes (detector data, tracker tuning, dedup,
sampling) and would not be bought with the A100. So a dominant bucket is
necessary but not sufficient to justify the rental: it also has to be the
*right* bucket, and this module says which one it is.

Deliberately dependency-free -- no numpy, no cv2. It is arithmetic over a
Counter, and it has to run in CI on a machine with nothing installed, next to
the config gate it feeds.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Where a bucket's remedy lives. This is the fact the raw histogram lacks and
# the whole reason the verdict is more than argmax(counter).
# --------------------------------------------------------------------------

# The dominant bucket this lane's A100 rental is actually for. Exactly one
# bucket carries it, and the go/no-go in config/training.yaml is only honest
# when this is the bucket on top.
POINTS_AT_OCR_FINETUNE = "ocr_finetune"

# A legitimate finding with no software fix. If this dominates, the honest
# deliverable is a width-bucket report and a camera-placement recommendation --
# not a training run, and not this lane's problem to solve in code.
POINTS_AT_NO_SOFTWARE_FIX = "no_software_fix"

# A real remedy that exists, but not one the A100 buys: detector training,
# fusion weighting, tracker tuning, dedup, throughput. Named so a dominant
# bucket here reads as "fix this, elsewhere" rather than "train the recogniser".
POINTS_AT_OTHER_LANE = "other_lane"


@dataclass(frozen=True)
class FailureBucket:
    """One row of the manual's section-6 table, as data.

    `points_at` is the editorial addition: the table says what would help, and
    this collapses that column into the one distinction the training gate cares
    about -- is the remedy this lane's fine-tune, no software at all, or someone
    else's lane.
    """

    key: str
    symptom: str
    remedy: str
    points_at: str


# The ten buckets, in the manual's table order. Order is preserved because the
# report renders them in this order and a reordering would read as a change to
# the deliverable when nothing changed. A tuple, not a dict, so the order is
# part of the data rather than an implementation detail of iteration.
FAILURE_BUCKETS: tuple[FailureBucket, ...] = (
    FailureBucket(
        "vehicle_miss",
        "No vehicle detected",
        "Vehicle detector training / Indian road data",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "plate_miss",
        "Vehicle found, plate not",
        "Plate detector training on small plates",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "plate_too_small",
        "Plate < 30 px",
        "Nothing in software -- camera placement, or accept it",
        POINTS_AT_NO_SOFTWARE_FIX,
    ),
    FailureBucket(
        "ocr_wrong",
        "Plate found, text wrong",
        "OCR training / synthetic corpus",
        POINTS_AT_OCR_FINETUNE,
    ),
    FailureBucket(
        "ocr_partial",
        "Some characters correct",
        "Temporal consensus / more frames",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "fusion_wrong",
        "Best single frame right, consensus wrong",
        "Fusion weighting",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "track_broken",
        "One vehicle split across tracks",
        "Tracker tuning",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "track_merged",
        "Two vehicles in one track",
        "Discontinuity / session handling",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "duplicate",
        "One vehicle, several sightings",
        "Dedup window",
        POINTS_AT_OTHER_LANE,
    ),
    FailureBucket(
        "dropped_frame",
        "Vehicle never sampled",
        "Sampling interval / throughput",
        POINTS_AT_OTHER_LANE,
    ),
)

BUCKETS_BY_KEY: dict[str, FailureBucket] = {b.key: b for b in FAILURE_BUCKETS}


# The floor on classified misses below which no bucket may be called dominant.
# Naming a plurality off a handful of misses is exactly the guess the manual
# warns against; a benchmark of eleven clips produces enough misses to clear
# this, and a run that has not is not finished. Overridable for the same reason
# the gate thresholds are: a sweep needs to move it without editing the source.
DEFAULT_MIN_SAMPLE = 30

# How far ahead the leader must be to be called dominant rather than co-dominant.
# 1.25 means the top bucket carries at least a quarter more misses than the
# runner-up. A leader one miss ahead of the field is a coin toss, and a coin
# toss is not a mandate to spend the budget -- so anything short of a clear
# margin is reported as "not finished" rather than rounded up to a decision.
DEFAULT_MARGIN = 1.25

# Verdict states. Strings rather than an enum so to_dict() is JSON already and
# the report can print them without a lookup.
DOMINANT = "dominant"
CO_DOMINANT = "co_dominant"
INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class TaxonomyVerdict:
    """The answer the deliverable exists to produce.

    `status` is the load-bearing field. `dominant` is set only when status is
    DOMINANT; `co_dominant` lists the tied leaders only when status is
    CO_DOMINANT. `unlocks_ocr_finetune` is the single boolean the training gate
    reads -- true only when a bucket dominates *and* that bucket is `ocr_wrong`.
    """

    status: str
    total: int
    counts: dict[str, int]
    dominant: Optional[str] = None
    co_dominant: tuple[str, ...] = ()
    leader_share: Optional[float] = None
    points_at: Optional[str] = None
    unlocks_ocr_finetune: bool = False
    recommendation: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-ready. This is what the report embeds and what a log records."""
        return {
            "status": self.status,
            "total": self.total,
            "counts": dict(self.counts),
            "dominant": self.dominant,
            "co_dominant": list(self.co_dominant),
            "leader_share": self.leader_share,
            "points_at": self.points_at,
            "unlocks_ocr_finetune": self.unlocks_ocr_finetune,
            "recommendation": self.recommendation,
        }


@dataclass
class FailureTaxonomy:
    """Tally misses by bucket and turn the tally into a verdict.

    Modelled on VehicleGate: a Counter of reasons plus a summary. The parallel
    is deliberate -- both are "count what got rejected and why", and the count
    is the evidence, not a side effect.
    """

    counts: Counter = field(default_factory=Counter)

    def record(self, bucket: str, count: int = 1) -> None:
        """Add `count` misses to a bucket. Unknown bucket is a hard error.

        Rejecting an unknown key rather than tallying it is the point: a typo'd
        bucket that silently created an eleventh row would split the histogram
        and could turn a real dominant bucket into an apparent tie -- the exact
        "analysis not finished" false positive that stops a justified run.
        """
        if bucket not in BUCKETS_BY_KEY:
            raise ValueError(
                f"unknown failure bucket {bucket!r}; must be one of "
                f"{sorted(BUCKETS_BY_KEY)}"
            )
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count:
            self.counts[bucket] += count

    def record_many(self, buckets: "dict[str, int]") -> None:
        """Bulk form -- a whole benchmark's histogram at once."""
        for bucket, count in buckets.items():
            self.record(bucket, count)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def shares(self) -> dict[str, float]:
        """Fraction of all misses per bucket, only for buckets seen.

        Empty when nothing has been recorded -- a share of a total of zero is
        undefined, and reporting 0.0 for every bucket would read as ten equal
        findings rather than as no data.
        """
        total = self.total
        if not total:
            return {}
        return {bucket: count / total for bucket, count in self.counts.items()}

    def verdict(
        self,
        *,
        min_sample: int = DEFAULT_MIN_SAMPLE,
        margin: float = DEFAULT_MARGIN,
    ) -> TaxonomyVerdict:
        """Dominant, co-dominant, or insufficient -- and what to do about it."""
        total = self.total
        counts = dict(self.counts)

        if total < min_sample:
            return TaxonomyVerdict(
                status=INSUFFICIENT,
                total=total,
                counts=counts,
                recommendation=(
                    f"only {total} classified miss(es); below the floor of "
                    f"{min_sample}, no bucket can be called dominant. Classify "
                    f"more of the baseline benchmark's misses before deciding."
                ),
            )

        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        leader, leader_count = ranked[0]
        runner_count = ranked[1][1] if len(ranked) > 1 else 0
        leader_share = leader_count / total

        # Dominant iff the leader is clearly ahead of the runner-up. A tie
        # (runner_count == leader_count) can never clear the margin, so ties
        # fall through to co-dominant, which is the correct reading of a tie.
        clearly_ahead = leader_count >= margin * runner_count
        if not clearly_ahead:
            tied = tuple(
                bucket
                for bucket, count in ranked
                if count * margin >= leader_count
            )
            listed = ", ".join(tied)
            return TaxonomyVerdict(
                status=CO_DOMINANT,
                total=total,
                counts=counts,
                co_dominant=tied,
                leader_share=leader_share,
                recommendation=(
                    f"buckets {listed} are co-dominant (no clear leader at "
                    f"margin {margin}); the analysis is not finished -- do not "
                    f"train. Separate them before spending the budget."
                ),
            )

        bucket = BUCKETS_BY_KEY[leader]
        return TaxonomyVerdict(
            status=DOMINANT,
            total=total,
            counts=counts,
            dominant=leader,
            leader_share=leader_share,
            points_at=bucket.points_at,
            unlocks_ocr_finetune=(bucket.points_at == POINTS_AT_OCR_FINETUNE),
            recommendation=_recommendation_for(bucket),
        )


def _recommendation_for(bucket: FailureBucket) -> str:
    """The next action for a dominant bucket, in the manual's own terms."""
    if bucket.points_at == POINTS_AT_OCR_FINETUNE:
        return (
            f"{bucket.key} dominant: the OCR recogniser fine-tune is the "
            f"justified spend. Proceed to the config/training.yaml gate -- "
            f"which still requires a labelled dataset, a held-out split, and a "
            f"measured baseline before it opens."
        )
    if bucket.points_at == POINTS_AT_NO_SOFTWARE_FIX:
        return (
            f"{bucket.key} dominant: no software fix exists. The honest "
            f"deliverable is a width-bucket report and a camera-placement "
            f"recommendation, not a training run."
        )
    return (
        f"{bucket.key} dominant: the remedy is '{bucket.remedy}'. That is a "
        f"real fix but not this lane's A100 -- do not open the training gate on "
        f"the strength of it."
    )


__all__ = [
    "BUCKETS_BY_KEY",
    "CO_DOMINANT",
    "DEFAULT_MARGIN",
    "DEFAULT_MIN_SAMPLE",
    "DOMINANT",
    "FAILURE_BUCKETS",
    "INSUFFICIENT",
    "POINTS_AT_NO_SOFTWARE_FIX",
    "POINTS_AT_OCR_FINETUNE",
    "POINTS_AT_OTHER_LANE",
    "FailureBucket",
    "FailureTaxonomy",
    "TaxonomyVerdict",
]
