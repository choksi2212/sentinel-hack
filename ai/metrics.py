"""Measure the pipeline, and make the dishonest number hard to produce.

    counters = RunCounters(warmup_frames=2)
    counters.observe_frame(latency_ms=41.2)          # discarded while warming up
    counters.plate_located_no_read += 1              # a vehicle whose plate never read

    card = ScoreCard.score(ground_truth, events, link=link_by_track)
    print(card.format_table())                       # all six buckets, always
    report = BenchmarkReport.build(card, counters, task="e2e", run_id="rfdetr_s_001")
    report.write()                                   # refuses benchmark/reports/ if oracle

Two objects, deliberately separate, because they need different things and mixing them is
how the headline number gets inflated.

**RunCounters needs no ground truth.** Frames, latencies, VRAM, how many crops were offered
and how many produced no read. Available on every run including live, where there is no label
to compare against. All of it is diagnostic: it explains the primary number and never replaces
it.

**ScoreCard needs ground truth and is driven by it.** Contracts section 7.1 locks

    E2E correct-plate event rate = correct final plate events / eligible vehicle events

and the load-bearing word is *eligible*: a vehicle whose plate is human-readable in at least
one sampled frame of the clip. That is a property of the footage, not of what the pipeline
managed to emit. So ScoreCard.score() iterates the ground-truth vehicles and looks for a
matching event -- never the reverse. Iterating events instead computes "of the vehicles I
found, how many did I read", which is a different and much prettier number, and the two are
indistinguishable once they are printed as a percentage. A vehicle the detector missed
entirely still sits in the denominator.

**No number is ever reported as a single average.** Contracts section 7.2. A headline 92%
routinely decomposes into 98% above 80 px and 51% below 40 px, and the second number is the
one that decides whether this works on real CCTV. So the per-bucket table is the primary
output, format_table() always prints all six rows including the empty ones, and an empty
bucket reports `n=0` rather than being omitted -- a missing row reads as "nothing to see
there", which is the opposite of what it means.

**An oracle number cannot reach benchmark/reports/.** The oracle stages exist to isolate the
pipeline from the models, and their accuracy figures are meaningless as model claims -- an
oracle OCR at 92% is a statement about a config file. publishable() checks that every stage
ships and none is an oracle; write() sends anything else to runs/ with the headline field
null and the figure demoted to a note. The refusal is structural rather than a warning,
because the failure mode is somebody screenshotting a number two days later.
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- buckets

# Contracts section 7.2, verbatim keys. These strings go into the report JSON and Akshat's
# leaderboard reads them, so they are not cosmetic and must not be prettified.
BUCKET_KEYS: tuple[str, ...] = (">100", "80-100", "60-80", "40-60", "30-40", "<30")

# The boundary rule, which the table in the contract leaves ambiguous: it lists B2 as
# "80 - 100" and B3 as "60 - 80", so 80 appears in both. Resolved as lower-inclusive,
# upper-exclusive everywhere except the top bucket, which the contract writes as a strict
# "> 100" -- so a plate exactly 100 px wide is B2 and one exactly 80 px wide is B2 as well.
# Written down because an off-by-one here moves samples between adjacent buckets silently,
# and the buckets it moves them between are the small ones that decide the claim.
_BUCKET_BOUNDS: tuple[tuple[str, int, Optional[int]], ...] = (
    (">100", 101, None),
    ("80-100", 80, 101),
    ("60-80", 60, 80),
    ("40-60", 40, 60),
    ("30-40", 30, 40),
    ("<30", 0, 30),
)

# The legibility floor from the synthetic corpus, repeated here only as the boundary of the
# bottom bucket. ai/ocr owns the real threshold; this module owns reporting.
LEGIBLE_WIDTH_PX = 30

CONDITIONS: tuple[str, ...] = ("day", "night", "blur", "glare", "angle")

TASKS: tuple[str, ...] = (
    "vehicle_detection",
    "plate_detection",
    "ocr",
    "temporal_fusion",
    "e2e",
)

# Two frames, matching config/base.yaml. The first CUDA kernel launch of a run pays for
# context creation, autotuning and weight upload, and on this laptop that is a 4x outlier on
# frame zero -- large enough to move a p95 computed over a 300 frame clip.
DEFAULT_WARMUP_FRAMES = 2

REPORT_DIR = "benchmark/reports"
LEADERBOARD = "benchmark/TRINETRA_MODEL_LEADERBOARD.csv"
DIAGNOSTIC_DIR = "runs/diagnostics"


def width_bucket(width_px: Optional[int]) -> str:
    """Which reporting bucket a plate width falls in.

    A width of None or 0 lands in "<30": a plate that was never located has no width, and the
    honest place for it is the hardest bucket rather than excluded from the table. Excluding
    it would remove the misses from exactly the bucket where misses happen.
    """
    width = int(width_px or 0)
    for key, low, high in _BUCKET_BOUNDS:
        if width >= low and (high is None or width < high):
            return key
    return "<30"


def empty_buckets() -> dict[str, Any]:
    """The locked by_plate_width shape with every value null.

    null, not 0.0. An empty bucket means "no eligible vehicles of this size were in the clip";
    0.0 means "we got every one of them wrong". Printing the second when the first is true is
    a lie in the direction of looking worse, which is the harmless direction -- but it also
    hides that the benchmark has no coverage in that bucket, and that is not harmless.
    """
    return {key: None for key in BUCKET_KEYS}


# --------------------------------------------------------------------------- ground truth


@dataclass(frozen=True)
class GroundTruthVehicle:
    """One labelled vehicle from a benchmark clip. The denominator lives here.

    `readable` is the eligibility flag and it is the most consequential field in this module.
    It says a human could read the plate in at least one sampled frame. It is not "the plate
    was visible", not "a plate was present", and emphatically not "the pipeline read it".

    `plate_width_px` is the width in the frame where the plate was most readable, in *scene*
    pixels -- the same convention as PlateBlock.plate_width_px and the one ai/ocr/base.py
    documents, so an upscale_2x crop does not get scored two buckets up.
    """

    vehicle_uid: str
    readable: bool
    plate: Optional[str] = None
    plate_width_px: int = 0
    condition: str = "day"
    vehicle_type: str = "other"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.readable and not self.plate:
            # An eligible vehicle with no label cannot be scored: it would sit in the
            # denominator and be uncorrectable by construction, quietly capping the metric.
            raise ValueError(
                f"ground truth {self.vehicle_uid}: readable=True requires a plate string. "
                f"An eligible vehicle with no label can never be scored correct, so it "
                f"lowers the primary metric by existing -- which is not a measurement."
            )
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"ground truth {self.vehicle_uid}: condition {self.condition!r} not in "
                f"{list(CONDITIONS)}"
            )


# Outcomes, one per eligible vehicle. Ordered worst-to-best for reporting.
OUTCOME_FABRICATED = "fabricated"  # emitted a plate, wrong. The failure that matters.
OUTCOME_CONTRADICTED = "contradicted"  # two events, disagreeing, at least one wrong
OUTCOME_MISSED = "missed"  # no event at all for an eligible vehicle
OUTCOME_ABSTAINED = "abstained"  # event emitted, plate null. Honest, and not correct.
OUTCOME_CORRECT = "correct"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_FABRICATED,
    OUTCOME_CONTRADICTED,
    OUTCOME_MISSED,
    OUTCOME_ABSTAINED,
    OUTCOME_CORRECT,
)


@dataclass(frozen=True)
class ScoredVehicle:
    """The outcome for one eligible vehicle, with enough detail to argue about it."""

    vehicle_uid: str
    bucket: str
    condition: str
    outcome: str
    expected: Optional[str]
    predicted: Optional[str]
    event_count: int

    @property
    def correct(self) -> bool:
        return self.outcome == OUTCOME_CORRECT


@dataclass
class BucketTally:
    """Counts for one width bucket. The rate is None when there is nothing in it."""

    eligible: int = 0
    correct: int = 0
    fabricated: int = 0
    contradicted: int = 0
    missed: int = 0
    abstained: int = 0

    @property
    def rate(self) -> Optional[float]:
        if self.eligible == 0:
            return None
        return self.correct / self.eligible

    @property
    def fabrication_rate(self) -> Optional[float]:
        """Wrong plates as a share of eligible vehicles.

        Tracked as its own number because it is not the complement of the rate above and the
        difference is the whole honesty argument. Abstaining costs the primary metric exactly
        as much as fabricating, and the two are not equally bad: a null is a vehicle nobody
        can identify, a wrong string is a vehicle somebody else gets accused of being. If the
        two are only ever seen added together, tuning will happily trade one for the other.
        """
        if self.eligible == 0:
            return None
        return (self.fabricated + self.contradicted) / self.eligible

    def add(self, outcome: str) -> None:
        self.eligible += 1
        if outcome == OUTCOME_CORRECT:
            self.correct += 1
        elif outcome == OUTCOME_FABRICATED:
            self.fabricated += 1
        elif outcome == OUTCOME_CONTRADICTED:
            self.contradicted += 1
        elif outcome == OUTCOME_MISSED:
            self.missed += 1
        elif outcome == OUTCOME_ABSTAINED:
            self.abstained += 1
        else:  # pragma: no cover - guarded by OUTCOMES
            raise ValueError(f"unknown outcome {outcome!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "correct": self.correct,
            "rate": None if self.rate is None else round(self.rate, 4),
            "fabricated": self.fabricated,
            "contradicted": self.contradicted,
            "missed": self.missed,
            "abstained": self.abstained,
        }


def _normalize_for_comparison(text: Optional[str]) -> str:
    """Upper-case, strip everything that is not a letter or digit.

    Ground truth is written by hand and "GJ 01 AB 1234" versus "GJ01AB1234" is a
    transcription style, not a wrong read. ai/normalize owns the real rules for what the
    pipeline emits; this is only the comparison, and it is deliberately more forgiving than
    that -- being strict here would score a correct pipeline as wrong because of a space in a
    CSV.
    """
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


class ScoreCard:
    """The primary metric, per bucket, driven by ground truth.

    Built through score() rather than __init__ so there is one code path and it is the one
    that takes ground truth as the outer loop.
    """

    def __init__(self, *, task: str = "e2e") -> None:
        if task not in TASKS:
            raise ValueError(f"task {task!r} not in {list(TASKS)}")
        self.task = task
        self.buckets: dict[str, BucketTally] = {key: BucketTally() for key in BUCKET_KEYS}
        self.conditions: dict[str, BucketTally] = {c: BucketTally() for c in CONDITIONS}
        self.scored: list[ScoredVehicle] = []
        # Vehicles ground truth says were not readable. Not in the denominator, counted so
        # the clip can be described: "300 vehicles, 210 eligible" is a fact about the
        # footage that a reader needs in order to interpret the rate at all.
        self.ineligible: int = 0
        # Events that matched no ground-truth vehicle. Not part of the primary metric by its
        # definition, and still worth counting: a run that emits 400 events for 210 vehicles
        # is fragmenting tracks or double-counting, and the primary metric alone hides it.
        self.unmatched_events: int = 0

    # ------------------------------------------------------------------ construction

    @classmethod
    def score(
        cls,
        ground_truth: Iterable[GroundTruthVehicle],
        events: Sequence[Any],
        *,
        link: Callable[[Any], Optional[str]],
        task: str = "e2e",
    ) -> "ScoreCard":
        """Score events against ground truth.

        `link` maps an event to the vehicle_uid it belongs to, or None for an event that
        matches no labelled vehicle. Supplied by the caller because the join differs by
        corpus: the synthetic source knows its own vehicle ids, a real clip needs IoU against
        a labelled box, and burying either in here would make the other one impossible.

        Iteration is over ground_truth, not events. That direction is the metric's
        definition; reversing it silently changes the denominator to "vehicles we detected".
        """
        card = cls(task=task)

        by_vehicle: dict[str, list[Any]] = {}
        for event in events:
            uid = link(event)
            if uid is None:
                card.unmatched_events += 1
                continue
            by_vehicle.setdefault(uid, []).append(event)

        seen: set[str] = set()
        for truth in ground_truth:
            if truth.vehicle_uid in seen:
                raise ValueError(
                    f"duplicate ground truth vehicle_uid {truth.vehicle_uid!r}. Two rows for "
                    f"one vehicle would count it twice in the denominator."
                )
            seen.add(truth.vehicle_uid)

            if not truth.readable:
                card.ineligible += 1
                continue

            matched = by_vehicle.get(truth.vehicle_uid, [])
            outcome, predicted = _outcome_for(truth, matched)
            scored = ScoredVehicle(
                vehicle_uid=truth.vehicle_uid,
                bucket=width_bucket(truth.plate_width_px),
                condition=truth.condition,
                outcome=outcome,
                expected=truth.plate,
                predicted=predicted,
                event_count=len(matched),
            )
            card.scored.append(scored)
            card.buckets[scored.bucket].add(outcome)
            card.conditions[scored.condition].add(outcome)

        return card

    # ------------------------------------------------------------------ aggregates

    @property
    def eligible(self) -> int:
        return sum(t.eligible for t in self.buckets.values())

    @property
    def correct(self) -> int:
        return sum(t.correct for t in self.buckets.values())

    @property
    def rate(self) -> Optional[float]:
        """The locked primary metric. None when nothing was eligible.

        Present because Contracts section 7.3 puts it at the top level of the report, and it
        is never the only thing this class hands out: every method that renders or exports it
        renders the buckets alongside, because section 7.2 forbids the single average and a
        scalar with a nice accessor is how that rule gets broken by accident.
        """
        if self.eligible == 0:
            return None
        return self.correct / self.eligible

    @property
    def fabricated(self) -> int:
        return sum(t.fabricated + t.contradicted for t in self.buckets.values())

    def by_plate_width(self) -> dict[str, Optional[float]]:
        """The locked by_plate_width block: bucket key to rate or null."""
        out = empty_buckets()
        for key in BUCKET_KEYS:
            out[key] = (
                None if self.buckets[key].rate is None else round(self.buckets[key].rate, 4)
            )
        return out

    def by_condition(self) -> dict[str, Optional[float]]:
        return {
            c: (None if self.conditions[c].rate is None else round(self.conditions[c].rate, 4))
            for c in CONDITIONS
        }

    def coverage_gaps(self) -> list[str]:
        """Buckets with no eligible vehicles. A property of the corpus, not the models.

        Reported because a clean-looking table with two empty rows is a benchmark that did not
        test the hard cases, and that is indistinguishable from one that passed them unless
        somebody says so out loud.
        """
        return [key for key in BUCKET_KEYS if self.buckets[key].eligible == 0]

    def format_table(self) -> str:
        """All six buckets, every time, empty ones included."""
        lines = [
            f"  E2E correct-plate event rate, task={self.task}",
            f"  {'bucket':>8}  {'n':>5}  {'correct':>7}  {'rate':>7}  "
            f"{'wrong':>5}  {'missed':>6}  {'null':>5}",
        ]
        for key in BUCKET_KEYS:
            tally = self.buckets[key]
            rate = "     --" if tally.rate is None else f"{tally.rate * 100:6.1f}%"
            lines.append(
                f"  {key:>8}  {tally.eligible:5d}  {tally.correct:7d}  {rate}  "
                f"{tally.fabricated + tally.contradicted:5d}  {tally.missed:6d}  "
                f"{tally.abstained:5d}"
            )
        overall = "--" if self.rate is None else f"{self.rate * 100:.1f}%"
        lines.append(
            f"  {'ALL':>8}  {self.eligible:5d}  {self.correct:7d}  {overall:>7}  "
            f"{self.fabricated:5d}  "
            f"{sum(t.missed for t in self.buckets.values()):6d}  "
            f"{sum(t.abstained for t in self.buckets.values()):5d}"
        )
        lines.append(
            f"  ({self.ineligible} vehicle(s) not eligible -- plate unreadable in every "
            f"sampled frame; {self.unmatched_events} event(s) matched no labelled vehicle)"
        )
        gaps = self.coverage_gaps()
        if gaps:
            lines.append(
                f"  NO COVERAGE in bucket(s) {', '.join(gaps)} -- this corpus does not test "
                f"them, so the ALL row is not a claim about those widths."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "eligible": self.eligible,
            "correct": self.correct,
            "rate": None if self.rate is None else round(self.rate, 4),
            "ineligible": self.ineligible,
            "unmatched_events": self.unmatched_events,
            "by_plate_width": self.by_plate_width(),
            "by_condition": self.by_condition(),
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "coverage_gaps": self.coverage_gaps(),
        }


def _plate_string(event: Any) -> Optional[str]:
    """The normalized plate from an event, whatever shape the caller is holding.

    Three shapes accepted: a plain dict (a loaded fixture, a replayed spool file), an
    EventEnvelope (what the worker has in hand), and anything with a to_dict(). Anything else
    raises rather than returning None, and that choice is deliberate: an unrecognised shape
    silently read as "no plate" would score every vehicle `abstained` and produce a clean,
    plausible, entirely fictional 0% with no error anywhere.
    """
    if isinstance(event, Mapping):
        plate = event.get("plate")
    elif hasattr(event, "plate"):
        plate = event.plate
    elif hasattr(event, "to_dict"):
        plate = event.to_dict().get("plate")
    else:
        raise TypeError(
            f"cannot read a plate from {type(event).__name__}: expected a mapping, an "
            f"EventEnvelope, or an object with to_dict(). Guessing here would score every "
            f"vehicle as an abstention and report it as a measurement."
        )
    if plate is None:
        return None
    if isinstance(plate, Mapping):
        return plate.get("normalized") or plate.get("raw")
    return getattr(plate, "normalized", None) or getattr(plate, "raw", None)


def _outcome_for(
    truth: GroundTruthVehicle, events: Sequence[Any]
) -> tuple[str, Optional[str]]:
    """One eligible vehicle, zero or more events, one outcome.

    The interesting case is a fragmented track: one vehicle, two events, one right and one
    wrong. That is scored `contradicted` and counts against the metric, which is the strict
    reading and the correct one -- the system has published a wrong identification, and a
    search for the wrong plate returns this vehicle. Crediting the correct half would score
    fragmentation as free, and fragmentation is precisely what TrackKey and the session id
    exist to prevent.
    """
    if not events:
        return OUTCOME_MISSED, None

    expected = _normalize_for_comparison(truth.plate)
    reads: list[str] = []
    for event in events:
        value = _normalize_for_comparison(_plate_string(event))
        if value:
            reads.append(value)

    if not reads:
        # Every event carried plate: null. Honest, and not a correct answer -- Contracts
        # section 3.2 calls a null a valid event, which is a statement about the event, not
        # about this metric. It does not earn a point and it does not cost extra.
        return OUTCOME_ABSTAINED, None

    distinct = sorted(set(reads))
    if distinct == [expected]:
        return OUTCOME_CORRECT, distinct[0]
    if expected in distinct:
        return OUTCOME_CONTRADICTED, "|".join(distinct)
    return OUTCOME_FABRICATED, distinct[0] if len(distinct) == 1 else "|".join(distinct)


# --------------------------------------------------------------------------- run counters


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile. None for an empty sample.

    Hand-rolled rather than numpy so importing this module costs nothing -- it is imported by
    the worker's startup path and by every test that checks a counter.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def vram_peak_mb() -> Optional[float]:
    """Peak reserved CUDA memory in MB, or None when there is no GPU.

    Reserved rather than allocated. allocated() counts only what the caching allocator handed
    out and under-reports by the size of every cached block, which on this 12 GB laptop is the
    difference between a config that looks like it fits and one that does. The 12 GB is the
    real constraint on this project, so the number reported has to be the one the driver sees.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch absent is a normal offline state
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_reserved() / (1024 * 1024)
    except Exception:  # noqa: BLE001 - a driver problem is not this module's business
        return None


def reset_vram_peak() -> None:
    """Zero the peak counter, so a run measures itself and not its predecessor."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


@dataclass
class RunCounters:
    """Throughput, latency and pipeline counters. No ground truth, so available live.

    Warm-up applies to latency and FPS only, never to accuracy. Dropping a frame from a
    latency sample removes a measurement of this machine; dropping a vehicle from the
    accuracy denominator changes what is being measured, because the denominator is a
    property of the footage. Nothing here can touch a ScoreCard, which is why they are
    separate objects.
    """

    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    log_vram: bool = True

    frames_seen: int = 0
    frames_sampled: int = 0
    frames_discarded_warmup: int = 0
    frames_dropped_late: int = 0

    tracks_started: int = 0
    tracks_completed: int = 0
    sessions_started: int = 0

    crops_offered: int = 0
    crops_rejected_quality: int = 0
    ocr_attempts: int = 0

    # Tracks that produced at least one plate crop. The denominator for the counter below,
    # and it has to be its own field rather than reusing crops_offered: a track offers many
    # crops and only the top K are ever read, so crops_offered counts a population that
    # never reached OCR. Dividing a per-track count by it would report an OCR failure rate
    # of a few percent for a stage that failed on every single vehicle it was given.
    tracks_with_plate_crops: int = 0

    # The counter ai/emit/builder.py's _plate_block defers to this module. A plate was
    # located, a crop was cut and offered to OCR, and nothing readable came back -- so the
    # event has no plate block and there is nowhere in the event to record that a plate was
    # nonetheless found. It matters because "located but unread" and "never located" are
    # different failures with different fixes: the first is an OCR problem, the second a plate
    # detector problem, and without this counter they are the same null in the output.
    #
    # Counted per track, not per crop, because that is the granularity the thing it explains
    # has: one event, one plate block, present or absent.
    plate_located_no_read: int = 0

    events_built: int = 0
    events_with_plate: int = 0
    events_plate_null: int = 0

    stage_ms: dict[str, list[float]] = field(default_factory=dict)
    frame_ms: list[float] = field(default_factory=list)

    first_pts_ms: Optional[int] = None
    last_pts_ms: Optional[int] = None
    started_monotonic: Optional[float] = None
    ended_monotonic: Optional[float] = None
    # perf_counter at the first frame that was not discarded as warm-up. fps is measured from
    # here so that it describes the same frames the latency percentiles do -- see fps below.
    measure_started: Optional[float] = None
    vram_peak_mb_value: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.started_monotonic = time.perf_counter()
        if self.log_vram:
            reset_vram_peak()

    def stop(self) -> None:
        self.ended_monotonic = time.perf_counter()
        if self.log_vram:
            self.vram_peak_mb_value = vram_peak_mb()

    @property
    def warming_up(self) -> bool:
        return self.frames_sampled < self.warmup_frames

    def observe_frame(self, *, latency_ms: float, pts_ms: Optional[int] = None) -> bool:
        """Record one sampled frame. Returns True if it counted toward latency.

        The return value exists so a caller can log "frame 0 discarded (warm-up)" rather than
        silently producing a run whose frame count and latency sample size disagree by two,
        which reads as a bug in the sampler.
        """
        self.frames_sampled += 1
        if pts_ms is not None:
            if self.first_pts_ms is None:
                self.first_pts_ms = int(pts_ms)
            self.last_pts_ms = int(pts_ms)
        if self.frames_sampled <= self.warmup_frames:
            self.frames_discarded_warmup += 1
            return False
        if self.measure_started is None:
            self.measure_started = time.perf_counter()
        self.frame_ms.append(float(latency_ms))
        return True

    def observe_stage(self, stage: str, elapsed_ms: float) -> None:
        """Per-stage timing, subject to the same warm-up rule as the frame total."""
        if self.frames_sampled <= self.warmup_frames:
            return
        self.stage_ms.setdefault(stage, []).append(float(elapsed_ms))

    # ------------------------------------------------------------------ derived

    @property
    def wall_seconds(self) -> Optional[float]:
        """Elapsed wall time, from perf_counter.

        perf_counter rather than monotonic, and the difference is not academic on this
        machine: monotonic() on Windows is GetTickCount64 with roughly 15.6 ms granularity, so
        any interval shorter than one tick measures as exactly 0.0 -- which the guards below
        read as "never measured" and report as null. A real clip never runs that fast, but a
        test over six synthetic frames does, every time, so the one thing that could verify
        the fps path could not measure it. perf_counter is QueryPerformanceCounter and is
        monotonic as well, so nothing is given up by using it.
        """
        if self.started_monotonic is None:
            return None
        end = self.ended_monotonic if self.ended_monotonic is not None else time.perf_counter()
        return end - self.started_monotonic

    @property
    def measured_seconds(self) -> Optional[float]:
        """Wall time from the first counted frame to the end. The fps denominator."""
        if self.measure_started is None:
            return None
        end = self.ended_monotonic if self.ended_monotonic is not None else time.perf_counter()
        return max(0.0, end - self.measure_started)

    @property
    def stream_seconds(self) -> Optional[float]:
        """Span of stream time covered, from PTS. None if PTS was never seen.

        PTS, never frame count over nominal fps -- CAP_PROP_FPS lies on the Sentinel streams
        and ai/media exists in the shape it does because of that.
        """
        if self.first_pts_ms is None or self.last_pts_ms is None:
            return None
        return max(0.0, (self.last_pts_ms - self.first_pts_ms) / 1000.0)

    @property
    def fps(self) -> Optional[float]:
        """Counted frames per second of wall clock, measured from the first counted frame.

        From measured_seconds rather than wall_seconds, so this number and the latency
        percentiles describe the same set of frames. Dividing warm-up-excluded frames by
        warm-up-included time is a real skew and it points the wrong way: the two discarded
        frames are the slowest in the run, so their time lands in the denominator while their
        frames do not land in the numerator. On a 300-frame clip that is noise; on a 20-frame
        smoke run it halves the reported throughput, and two numbers in one report that
        disagree about which frames they cover are worse than either alone.

        Throughput, not a real-time claim. A file source read unthrottled reports a number far
        above the stream's own rate, which is a useful capacity figure and is not a latency
        measurement; real_time_factor below is the one that says whether a camera can be kept
        up with.
        """
        counted = len(self.frame_ms)
        elapsed = self.measured_seconds
        if not counted or not elapsed:
            return None
        return counted / elapsed

    @property
    def real_time_factor(self) -> Optional[float]:
        """Stream seconds processed per wall second. Below 1.0 means falling behind.

        Deliberately over the *whole* run, warm-up included, unlike fps. The question this
        answers is operational -- can this keep up with a camera -- and a camera does not wait
        for model load and CUDA autotuning. Every reconnect pays that cost again, so counting
        it is the honest reading rather than the flattering one.
        """
        stream = self.stream_seconds
        wall = self.wall_seconds
        if stream is None or not wall:
            return None
        return stream / wall

    @property
    def latency_p50_ms(self) -> Optional[float]:
        return _percentile(self.frame_ms, 0.50)

    @property
    def latency_p95_ms(self) -> Optional[float]:
        return _percentile(self.frame_ms, 0.95)

    @property
    def located_but_unread_rate(self) -> Optional[float]:
        """plate_located_no_read as a share of tracks that produced a plate crop.

        The ratio rather than the raw count, because the count alone cannot distinguish an OCR
        stage that fails on a tenth of what it is given from one that is being handed ten
        times as many vehicles.

        Against tracks_with_plate_crops, not crops_offered. Both counters describe plates
        that were located, but at different granularities: a vehicle in frame for three
        seconds offers thirty crops and has four of them read. Dividing the per-track
        numerator by the per-crop denominator would divide by roughly eight times too much,
        and a stage that failed to read every vehicle it saw would report a 12% shortfall.
        """
        if not self.tracks_with_plate_crops:
            return None
        return self.plate_located_no_read / self.tracks_with_plate_crops

    def stage_table(self) -> str:
        """Per-stage p50/p95. Empty stages are listed, not hidden."""
        if not self.stage_ms:
            return "  (no stage timings recorded)"
        width = max(len(name) for name in self.stage_ms)
        lines = [f"  {'stage':<{width}}  {'n':>5}  {'p50 ms':>8}  {'p95 ms':>8}"]
        for name in sorted(self.stage_ms):
            samples = self.stage_ms[name]
            p50 = _percentile(samples, 0.50)
            p95 = _percentile(samples, 0.95)
            lines.append(
                f"  {name:<{width}}  {len(samples):5d}  "
                f"{'--' if p50 is None else format(p50, '8.2f')}  "
                f"{'--' if p95 is None else format(p95, '8.2f')}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "frames_sampled": self.frames_sampled,
            "frames_discarded_warmup": self.frames_discarded_warmup,
            "frames_dropped_late": self.frames_dropped_late,
            "tracks_started": self.tracks_started,
            "tracks_completed": self.tracks_completed,
            "sessions_started": self.sessions_started,
            "crops_offered": self.crops_offered,
            "crops_rejected_quality": self.crops_rejected_quality,
            "ocr_attempts": self.ocr_attempts,
            "tracks_with_plate_crops": self.tracks_with_plate_crops,
            "plate_located_no_read": self.plate_located_no_read,
            "located_but_unread_rate": _round(self.located_but_unread_rate, 4),
            "events_built": self.events_built,
            "events_with_plate": self.events_with_plate,
            "events_plate_null": self.events_plate_null,
            "fps": _round(self.fps, 2),
            "real_time_factor": _round(self.real_time_factor, 3),
            "latency_p50_ms": _round(self.latency_p50_ms, 2),
            "latency_p95_ms": _round(self.latency_p95_ms, 2),
            "wall_seconds": _round(self.wall_seconds, 2),
            "measured_seconds": _round(self.measured_seconds, 2),
            "stream_seconds": _round(self.stream_seconds, 2),
            "vram_peak_mb": _round(self.vram_peak_mb_value, 1),
            "stages": {
                name: {
                    "n": len(samples),
                    "p50_ms": _round(_percentile(samples, 0.50), 2),
                    "p95_ms": _round(_percentile(samples, 0.95), 2),
                }
                for name, samples in sorted(self.stage_ms.items())
            },
        }


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------- provenance


def git_commit() -> Optional[str]:
    """Current commit, or None outside a repository.

    In the report because Contracts section 7.3 requires it: a benchmark number that cannot
    be tied to a commit is not reproducible, and every row on the leaderboard is a claim.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 - no git, or not a repo
        return None
    value = out.stdout.strip()
    return value or None


def git_is_dirty() -> Optional[bool]:
    """True when the working tree has uncommitted changes.

    Recorded because "measured at commit abc123" is false if the tree was dirty, and the
    difference is invisible in the report unless it is stated.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def sha256_file(path: str | os.PathLike[str]) -> Optional[str]:
    """Hash a weights file in chunks. None if it is not there."""
    target = Path(path)
    if not target.is_file():
        return None
    digest = sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def machine_description() -> str:
    """The GPU name if there is one, else the CPU. Goes in the report verbatim."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{name} {total:.0f}GB"
    except Exception:  # noqa: BLE001
        pass
    return f"{platform.processor() or platform.machine()} (CPU)"


def runtime_description() -> str:
    try:
        import torch

        cuda = torch.version.cuda or "no cuda"
        return f"torch {torch.__version__} + CUDA {cuda}"
    except Exception:  # noqa: BLE001
        return f"python {sys.version.split()[0]}, torch absent"


# --------------------------------------------------------------------------- report


@dataclass
class StageIdentity:
    """What one stage was, for the publishability check.

    Taken from the stage's own describe()/properties rather than from the config, because the
    config says which backend was asked for and this has to say which one ran. A config naming
    rfdetr that silently fell back to a stub is exactly the case worth catching.
    """

    name: str
    model_name: str = ""
    model_version: str = ""
    ships: bool = True
    is_oracle: bool = False
    weights_sha256: Optional[str] = None

    @classmethod
    def from_stage(cls, name: str, stage: Any) -> "StageIdentity":
        return cls(
            name=name,
            model_name=str(getattr(stage, "model_name", "") or ""),
            model_version=str(getattr(stage, "model_version", "") or ""),
            ships=bool(getattr(stage, "ships", True)),
            is_oracle=bool(getattr(stage, "is_oracle", False)),
            weights_sha256=getattr(stage, "weights_sha256", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": f"{self.model_name}@{self.model_version}".strip("@"),
            "ships": self.ships,
            "is_oracle": self.is_oracle,
            "weights_sha256": self.weights_sha256,
        }


@dataclass
class BenchmarkReport:
    """The locked report shape from Contracts section 7.3, plus the refusal to publish.

    to_dict() emits exactly the keys the contract lists, in that order, so a diff against the
    document is readable by eye and Akshat's leaderboard loader does not need a version check.
    Everything this module knows that the contract has no field for goes into `notes`, which
    is a list of strings and therefore the one place the shape allows.
    """

    run_id: str
    task: str
    scorecard: Optional[ScoreCard] = None
    counters: Optional[RunCounters] = None
    source_mode: str = "file"
    stages: dict[str, StageIdentity] = field(default_factory=dict)
    dataset_manifest_sha256: Optional[str] = None
    diagnostics_extra: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task {self.task!r} not in {list(TASKS)}")

    @classmethod
    def build(
        cls,
        scorecard: Optional[ScoreCard],
        counters: Optional[RunCounters],
        *,
        task: str,
        run_id: str,
        source_mode: str = "file",
        stages: Optional[Mapping[str, Any]] = None,
        dataset_manifest_sha256: Optional[str] = None,
        notes: Optional[Sequence[str]] = None,
    ) -> "BenchmarkReport":
        identities: dict[str, StageIdentity] = {}
        for label, stage in (stages or {}).items():
            identities[label] = (
                stage
                if isinstance(stage, StageIdentity)
                else StageIdentity.from_stage(label, stage)
            )
        return cls(
            run_id=run_id,
            task=task,
            scorecard=scorecard,
            counters=counters,
            source_mode=source_mode,
            stages=identities,
            dataset_manifest_sha256=dataset_manifest_sha256,
            notes=list(notes or []),
        )

    # ------------------------------------------------------------------ publishability

    def refusals(self) -> list[str]:
        """Why this run's accuracy figure is not a model claim. Empty means it is one."""
        reasons: list[str] = []
        for label, stage in sorted(self.stages.items()):
            if stage.is_oracle:
                reasons.append(
                    f"{label} is an oracle ({stage.model_name or stage.name}): it reads "
                    f"ground truth, so its accuracy describes this config file and not a "
                    f"model."
                )
            elif not stage.ships:
                reasons.append(
                    f"{label} ({stage.model_name or stage.name}) does not ship, so a number "
                    f"measured with it cannot be claimed for the submitted system."
                )
        if not self.stages:
            reasons.append(
                "no stage identities recorded, so there is nothing to attest that the "
                "models measured are the models that ship."
            )
        if self.scorecard is not None and self.scorecard.eligible == 0:
            reasons.append(
                "no eligible vehicles: the denominator is zero, so there is no rate."
            )
        if self.dataset_manifest_sha256 is None and self.scorecard is not None:
            reasons.append(
                "no dataset manifest hash, so the corpus this was measured on cannot be "
                "identified later."
            )
        return reasons

    @property
    def publishable(self) -> bool:
        return not self.refusals()

    # ------------------------------------------------------------------ serialisation

    def _diagnostics(self) -> dict[str, Any]:
        # The locked key set, all present, null when not measured. Extra keys are permitted
        # to be added by a task that measures more -- a detector run has a real map50 -- but
        # the locked ones never disappear, because a consumer reading them must not have to
        # test for presence.
        base: dict[str, Any] = {
            "precision": None,
            "recall": None,
            "map50": None,
            "small_plate_recall": None,
            "ocr_exact_accuracy": None,
            "cer": None,
            "fps": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "vram_peak_mb": None,
            "real_time_factor": None,
        }
        if self.counters is not None:
            base["fps"] = _round(self.counters.fps, 2)
            base["latency_p50_ms"] = _round(self.counters.latency_p50_ms, 2)
            base["latency_p95_ms"] = _round(self.counters.latency_p95_ms, 2)
            base["vram_peak_mb"] = _round(self.counters.vram_peak_mb_value, 1)
            base["real_time_factor"] = _round(self.counters.real_time_factor, 3)
        if self.scorecard is not None:
            # small_plate_recall is the two bottom buckets pooled, i.e. everything under
            # 40 px. Pooled rather than reported as B6 alone because the contract asks for one
            # number here and B6 on its own is often n=0 on a given clip, which would publish
            # null for the diagnostic that matters most.
            below_40 = BucketTally()
            for key in ("<30", "30-40"):
                tally = self.scorecard.buckets[key]
                below_40.eligible += tally.eligible
                below_40.correct += tally.correct
            base["small_plate_recall"] = _round(below_40.rate, 4)
        base.update(self.diagnostics_extra)
        return base

    def _notes(self) -> list[str]:
        notes = list(self.notes)
        composite = self._weights_note()
        if composite:
            notes.append(composite)
        dirty = git_is_dirty()
        if dirty:
            notes.append(
                "working tree was dirty at measurement time, so the commit below does not "
                "fully describe the code that produced this number."
            )
        for reason in self.refusals():
            notes.append(f"NOT PUBLISHABLE: {reason}")
        if not self.publishable and self.scorecard is not None:
            rate = self.scorecard.rate
            if rate is not None:
                notes.append(
                    f"diagnostic-only figure, deliberately not in the headline field: "
                    f"{rate:.4f} over {self.scorecard.eligible} eligible vehicle(s)."
                )
        if self.scorecard is not None:
            gaps = self.scorecard.coverage_gaps()
            if gaps:
                notes.append(
                    f"no eligible vehicles in bucket(s) {', '.join(gaps)}; the headline rate "
                    f"is not a claim about those widths."
                )
            if self.scorecard.unmatched_events:
                notes.append(
                    f"{self.scorecard.unmatched_events} event(s) matched no labelled "
                    f"vehicle and are outside the primary metric by its definition."
                )
        if self.counters is not None and self.counters.plate_located_no_read:
            notes.append(
                f"{self.counters.plate_located_no_read} vehicle(s) had a plate located but "
                f"never read, of {self.counters.tracks_with_plate_crops} that offered a crop "
                f"-- an OCR shortfall rather than a plate-detection one, and invisible in the "
                f"events themselves."
            )
        return notes

    # Which stage's weights the locked single `weights_sha256` field refers to, per task. A
    # single-stage task has one answer and it is verifiable: hash the file, compare.
    _TASK_STAGE = {
        "vehicle_detection": "detect",
        "plate_detection": "plate",
        "ocr": "ocr",
    }

    def _weights_sha256(self) -> Optional[str]:
        """The value for the locked single-hash field.

        For a single-stage task, that stage's own hash, so a reader can verify it against the
        checkpoint on disk. For e2e and temporal_fusion there are three checkpoints and one
        field, so the value is a composite -- a hash of the per-stage hashes -- and the
        components go into notes. Deterministic and reproducible either way; the alternative
        was picking one of the three by loop order, which is a coin flip dressed as
        provenance.
        """
        label = self._TASK_STAGE.get(self.task)
        if label is not None:
            stage = self.stages.get(label)
            return stage.weights_sha256 if stage is not None else None
        parts = [
            f"{name}={stage.weights_sha256}"
            for name, stage in sorted(self.stages.items())
            if stage.weights_sha256
        ]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0].split("=", 1)[1]
        return sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _weights_note(self) -> Optional[str]:
        """Spell out a composite hash, so nobody tries to match it against one file."""
        if self.task in self._TASK_STAGE:
            return None
        parts = [
            f"{name}={stage.weights_sha256}"
            for name, stage in sorted(self.stages.items())
            if stage.weights_sha256
        ]
        if len(parts) < 2:
            return None
        return (
            "weights_sha256 is a composite over multiple stages and will not match any single "
            "checkpoint; components are " + ", ".join(parts) + "."
        )

    def to_dict(self) -> dict[str, Any]:
        """The locked shape. Key order matches the contract document."""
        headline: Optional[float] = None
        if self.scorecard is not None and self.publishable:
            headline = _round(self.scorecard.rate, 4)

        return {
            "run_id": self.run_id,
            "task": self.task,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "git_commit": git_commit(),
            "weights_sha256": self._weights_sha256(),
            "machine": machine_description(),
            "runtime": runtime_description(),
            "source_mode": self.source_mode,
            "e2e_correct_plate_event_rate": headline,
            "by_plate_width": (
                empty_buckets()
                if self.scorecard is None or not self.publishable
                else self.scorecard.by_plate_width()
            ),
            "by_condition": (
                {c: None for c in CONDITIONS}
                if self.scorecard is None or not self.publishable
                else self.scorecard.by_condition()
            ),
            "diagnostics": self._diagnostics(),
            "notes": self._notes(),
        }

    def write(self, root: str | os.PathLike[str] = ".") -> Path:
        """Write the report. Publishable runs land in benchmark/reports/, others do not.

        The destination carries the meaning, which is the point. A warning in a log gets
        scrolled past and a `publishable: false` field gets skipped by eye; a file that is
        not in the directory the leaderboard is built from cannot be misread as a result two
        days later, and the full figure is still there in runs/diagnostics/ for whoever is
        debugging the pipeline today.
        """
        base = Path(root)
        if self.publishable:
            directory = base / REPORT_DIR
            payload = self.to_dict()
        else:
            directory = base / DIAGNOSTIC_DIR
            payload = self.to_dict()
            # The scorecard in full, since this file is explicitly not a claim and the
            # per-bucket detail is what makes it useful for debugging.
            payload["scorecard"] = (
                None if self.scorecard is None else self.scorecard.to_dict()
            )
            payload["counters"] = None if self.counters is None else self.counters.to_dict()
            payload["stages"] = {k: v.to_dict() for k, v in sorted(self.stages.items())}
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.task}_{_safe(self.run_id)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        return path

    def append_leaderboard(self, root: str | os.PathLike[str] = ".") -> Optional[Path]:
        """Append one row to the leaderboard. Returns None when it refused.

        Refuses for the same reasons write() redirects: the leaderboard is the file the
        submission claim is read off, and a row on it is a claim whether or not anybody meant
        it as one.
        """
        if not self.publishable or self.scorecard is None:
            return None
        base = Path(root)
        path = base / LEADERBOARD
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "run_id",
            "task",
            "git_commit",
            "source_mode",
            "eligible",
            "rate",
            *BUCKET_KEYS,
            "fps",
            "latency_p95_ms",
            "vram_peak_mb",
        ]
        widths = self.scorecard.by_plate_width()
        diagnostics = self._diagnostics()
        row = [
            self.run_id,
            self.task,
            (git_commit() or "")[:12],
            self.source_mode,
            self.scorecard.eligible,
            _round(self.scorecard.rate, 4),
            *[widths[key] for key in BUCKET_KEYS],
            diagnostics["fps"],
            diagnostics["latency_p95_ms"],
            diagnostics["vram_peak_mb"],
        ]
        exists = path.is_file()
        with path.open("a", encoding="utf-8", newline="") as handle:
            if not exists:
                handle.write(",".join(columns) + "\n")
            handle.write(",".join("" if v is None else str(v) for v in row) + "\n")
        return path

    def format_summary(self) -> str:
        """What the worker prints at the end of a run."""
        lines = [f"run {self.run_id}  task={self.task}  source_mode={self.source_mode}"]
        if self.scorecard is not None:
            lines.append(self.scorecard.format_table())
        if self.counters is not None:
            counters = self.counters
            lines.append(
                f"  frames {counters.frames_sampled} sampled "
                f"({counters.frames_discarded_warmup} discarded warming up), "
                f"events {counters.events_built} "
                f"({counters.events_plate_null} with plate null)"
            )
            fps = counters.fps
            rtf = counters.real_time_factor
            p95 = counters.latency_p95_ms
            vram = counters.vram_peak_mb_value
            lines.append(
                f"  fps {'--' if fps is None else format(fps, '.1f')}  "
                f"real-time x{'--' if rtf is None else format(rtf, '.2f')}  "
                f"p95 {'--' if p95 is None else format(p95, '.1f')} ms  "
                f"vram {'--' if vram is None else format(vram, '.0f')} MB"
            )
            if counters.plate_located_no_read:
                unread = counters.located_but_unread_rate
                lines.append(
                    f"  located but unread: {counters.plate_located_no_read} of "
                    f"{counters.tracks_with_plate_crops} vehicle(s) with a plate crop"
                    + ("" if unread is None else f" ({unread * 100:.1f}%)")
                )
            lines.append(counters.stage_table())
        for reason in self.refusals():
            lines.append(f"  NOT PUBLISHABLE: {reason}")
        return "\n".join(lines)


def _safe(text: str) -> str:
    """Filename-safe run id. Windows rejects most of what a run id might contain."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "run"


__all__ = [
    "BUCKET_KEYS",
    "CONDITIONS",
    "DEFAULT_WARMUP_FRAMES",
    "DIAGNOSTIC_DIR",
    "LEADERBOARD",
    "LEGIBLE_WIDTH_PX",
    "OUTCOMES",
    "OUTCOME_ABSTAINED",
    "OUTCOME_CONTRADICTED",
    "OUTCOME_CORRECT",
    "OUTCOME_FABRICATED",
    "OUTCOME_MISSED",
    "REPORT_DIR",
    "TASKS",
    "BenchmarkReport",
    "BucketTally",
    "GroundTruthVehicle",
    "RunCounters",
    "ScoreCard",
    "ScoredVehicle",
    "StageIdentity",
    "empty_buckets",
    "git_commit",
    "git_is_dirty",
    "machine_description",
    "reset_vram_peak",
    "runtime_description",
    "sha256_file",
    "vram_peak_mb",
    "width_bucket",
]
