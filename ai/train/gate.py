"""The training go/no-go, read from config and checked against disk.

Owner's manual section 7. The A100 is the only paid resource in the project --
RunPod A100 SXM 80GB at $1.64/hour -- so the decision to spend money is written
down as four flags in config/training.yaml rather than made at 2am on D6:

    local_benchmark_all_green      every one of the 11 benchmark rows is green
                                   on the local 12 GB card first
    labelled_dataset_exists        labelled Indian plates to learn from
    held_out_split_exists          a split that was never trained on
    baseline_measured_on_held_out  a before-number, so "we fine-tuned it" is a
                                   claim that can be checked

This module reads those flags and refuses to call the gate open while any is
false. That refusal is the feature: `scripts/train.py` cannot launch a paid run
until every flag is true, and there is deliberately no override to force it --
an override is how the gate becomes a formality.

Two things the flags alone do not cover, checked separately because they fail
in CI for legitimate reasons and must not be confused with a closed gate:

  * The dataset manifests. `labelled_dataset_exists` is a human attestation;
    `dataset_blockers` is the machine check of it -- do the manifest files
    actually sit on disk. They are gitignored (real registration plates) and
    not this lane's to produce, so their absence is a MANUAL STEP, not a bug.

  * The budget arithmetic. A ceiling that does not equal rate x hours is how a
    $6 experiment quietly becomes a $90 one; it is checked so the number in the
    file has to stay honest.

Dependency-free on purpose: reads an already-loaded AppConfig, imports nothing
heavy, and runs in CI beside scripts/validate_config.py.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.config import AppConfig

# The four flags, in the order the manual lists them, paired with the evidence
# each one attests to. Kept as data so a test can assert the set has not drifted
# from the config, and so the blocker messages are generated rather than
# hand-copied out of sync with the keys.
TRAINING_GATE_FLAGS: tuple[tuple[str, str], ...] = (
    (
        "local_benchmark_all_green",
        "every one of the 11 benchmark rows is green on the local 12 GB card",
    ),
    (
        "labelled_dataset_exists",
        "a labelled dataset of Indian plates exists to fine-tune on",
    ),
    (
        "held_out_split_exists",
        "a held-out split that was never trained on exists",
    ),
    (
        "baseline_measured_on_held_out",
        "a baseline number is measured on that held-out split",
    ),
)

_FLAG_KEYS: tuple[str, ...] = tuple(key for key, _ in TRAINING_GATE_FLAGS)

# Rate x hours must equal the ceiling to within this, or the ceiling is not a
# ceiling. A cent of slack absorbs float representation; more than that is a
# number that stopped matching its own arithmetic.
BUDGET_TOLERANCE_USD = 0.01


def gate_flags(config: "AppConfig") -> dict[str, bool]:
    """The four flags as booleans. A missing or non-bool value reads as false.

    False-on-missing is the safe default here: the gate protects a spend, so an
    absent flag must never be mistaken for permission. The blocker list below
    names it explicitly as unset so the reason is not lost in the coercion.
    """
    flags: dict[str, bool] = {}
    for key in _FLAG_KEYS:
        flags[key] = config.get(f"gate.{key}") is True
    return flags


def gate_blockers(config: "AppConfig") -> list[str]:
    """One message per gate flag that is not yet true, in flag order.

    Empty list means the four attestations are all in place -- necessary, not
    sufficient: the dataset still has to be on disk and the budget still has to
    add up before a real run may start. See launch_blockers.
    """
    blockers: list[str] = []
    for key, evidence in TRAINING_GATE_FLAGS:
        value = config.get(f"gate.{key}")
        if value is True:
            continue
        state = "unset" if value is None else repr(value)
        blockers.append(f"gate.{key} is {state}; need {evidence}")
    return blockers


def is_open(config: "AppConfig") -> bool:
    """True only when all four flags are true. This is the launch precondition."""
    return not gate_blockers(config)


def dataset_blockers(config: "AppConfig") -> list[str]:
    """Manifest files that the config names but that are not on disk.

    Separate from the gate because their absence is expected in CI and on any
    machine that has not done the MANUAL STEP of building the dataset -- the
    crops are real plates and data/ is gitignored. A blocker here is a "not
    ready yet", not a "misconfigured".
    """
    blockers: list[str] = []
    for key in ("train_manifest", "val_manifest"):
        raw = config.get(f"dataset.{key}")
        if not raw:
            blockers.append(f"dataset.{key} is not set")
            continue
        path = Path(str(raw))
        if not path.is_file():
            blockers.append(
                f"dataset.{key} does not exist: {path} "
                f"(MANUAL STEP -- gitignored, not this lane's to produce)"
            )
    return blockers


def budget_summary(config: "AppConfig") -> dict[str, object]:
    """The spend envelope, resolved. What a run log records before it starts."""
    return {
        "provider": config.get("budget.provider"),
        "instance": config.get("budget.instance"),
        "usd_per_hour": config.get("budget.usd_per_hour"),
        "max_hours": config.get("budget.max_hours"),
        "ceiling_usd": config.get("budget.ceiling_usd"),
    }


def budget_blockers(config: "AppConfig") -> list[str]:
    """Budget fields that are missing or that do not add up.

    The arithmetic check is the one with teeth: a ceiling that does not equal
    rate x max_hours is either a typo or a ceiling someone raised without
    changing the hours, and both are how an hourly instance overruns.
    """
    blockers: list[str] = []
    rate = config.get("budget.usd_per_hour")
    hours = config.get("budget.max_hours")
    ceiling = config.get("budget.ceiling_usd")

    for key, value in (
        ("usd_per_hour", rate),
        ("max_hours", hours),
        ("ceiling_usd", ceiling),
    ):
        if not isinstance(value, (int, float)):
            blockers.append(f"budget.{key} is missing or not a number")

    if blockers:
        return blockers

    expected = float(rate) * float(hours)
    if abs(expected - float(ceiling)) > BUDGET_TOLERANCE_USD:
        blockers.append(
            f"budget.ceiling_usd {ceiling} != usd_per_hour {rate} x max_hours "
            f"{hours} = {expected:.2f}; a ceiling that does not match the "
            f"arithmetic is not a ceiling"
        )
    return blockers


def launch_blockers(config: "AppConfig") -> list[str]:
    """Everything that must be clear before a real paid run may start.

    The gate flags, the dataset on disk, and the budget arithmetic, in that
    order -- gate first because it is the human decision, then the two machine
    checks. A non-empty list is a refusal to launch, and the list is the reason.
    """
    return gate_blockers(config) + dataset_blockers(config) + budget_blockers(config)


__all__ = [
    "BUDGET_TOLERANCE_USD",
    "TRAINING_GATE_FLAGS",
    "budget_blockers",
    "budget_summary",
    "dataset_blockers",
    "gate_blockers",
    "gate_flags",
    "is_open",
    "launch_blockers",
]
