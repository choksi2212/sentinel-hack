"""Read the training go/no-go, and refuse to launch a paid run until it passes.

    python scripts/train.py                     # show the gate for config/training.yaml
    python scripts/train.py --smoke             # run the model-free dry run (free, local)
    python scripts/train.py config/training.yaml --smoke --epochs 12

Exists because the A100 is the only paid resource in the project and the
decision to spend on it must be mechanical, not a judgement call made at 2am on
D6. This script is the mechanism: it reads the four attestations in
config/training.yaml, checks the dataset is on disk and the budget adds up, and
prints a refusal with the reasons if any of that is not true. There is no flag
to override the refusal -- an override is how a gate becomes a formality, and
the whole point is that it is not one.

What it deliberately does NOT do is run PaddleOCR. The real fine-tune needs the
framework, a labelled dataset that is not this lane's to produce, and the rented
instance; it is a MANUAL STEP by construction. When the gate is fully open this
script says so and hands off, rather than pretending to train.

`--smoke` is the one path that ignores the gate, because it spends nothing: it
runs the fine-tune loop's control logic (batching, the frozen-backbone phase,
checkpoint selection on val_exact_match, early stop, the ship-only-if-better
decision) on seeded synthetic numbers. It is how the harness that will wrap the
real run is tested before the instance is billed.

Exit codes:
  0  a smoke run completed
  1  a real run was refused -- the gate is closed (this is the state today)
  2  the config could not be read, or is not a training config
  3  the gate is open, but the real run is a MANUAL STEP this script will not do
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import ConfigError, load_config  # noqa: E402
from ai.train.gate import (  # noqa: E402
    budget_blockers,
    budget_summary,
    dataset_blockers,
    gate_blockers,
    gate_flags,
)
from ai.train.smoke import smoke_train  # noqa: E402

DEFAULT_CONFIG = "config/training.yaml"


def _print_gate(config) -> None:
    """The four attestations and their state. Printed on every path so the
    reader always sees what a real run would face, even under --smoke."""
    flags = gate_flags(config)
    print("  gate:")
    for key, value in flags.items():
        mark = "PASS" if value else "----"
        print(f"    [{mark}] gate.{key} = {value}")


def _print_budget(config) -> None:
    summary = budget_summary(config)
    print("  budget:")
    print(f"    provider     {summary['provider']}")
    print(f"    instance     {summary['instance']}")
    print(f"    rate         ${summary['usd_per_hour']}/hour")
    print(f"    max_hours    {summary['max_hours']}")
    print(f"    ceiling      ${summary['ceiling_usd']}")
    for message in budget_blockers(config):
        print(f"    WARNING  {message}")


def _run_smoke(config, *, epochs, baseline, as_json) -> int:
    """The model-free dry run. Bypasses the gate on purpose -- it spends nothing."""
    result = smoke_train(config, max_epochs=epochs, baseline=baseline)

    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("\n  smoke (model-free dry run; the gate is bypassed because this spends nothing):")
    print(f"    {result.batches_per_epoch} batches/epoch at batch_size {result.batch_size}")
    print(
        f"    epoch  val_exact_match  best   phase "
        f"(freeze first {result.freeze_backbone_epochs})"
    )
    for record in result.history:
        phase = "frozen" if record.frozen else "train"
        star = " *" if record.epoch == result.best_epoch else "  "
        print(
            f"    {record.epoch:>5}  {record.val_exact_match:>15.4f}  "
            f"{record.best_so_far:>5.4f}  {phase}{star}"
        )
    stop = "early stop" if result.stopped_early else f"ran all {result.max_epochs}"
    print(
        f"    selected epoch {result.best_epoch} on {result.select_on} "
        f"= {result.best_val:.4f} ({stop})"
    )
    verdict = "SHIP" if result.ship else "KEEP BASELINE"
    print(
        f"    ship_only_if_better: best {result.best_val:.4f} vs baseline "
        f"{result.baseline:.4f} -> {verdict}"
    )
    return 0


def _refuse_or_handoff(config) -> int:
    """The real-run path. Refuse while blocked; hand off to the MANUAL STEP when clear."""
    blockers = gate_blockers(config) + dataset_blockers(config) + budget_blockers(config)
    if blockers:
        print("\n  REFUSING TO LAUNCH -- the gate is not open:")
        for message in blockers:
            print(f"    - {message}")
        print(
            "\n  Flip the gate flags in the config only with the evidence to hand. "
            "There is deliberately no override."
        )
        return 1

    print("\n  Gate is open. The real fine-tune is a MANUAL STEP this script will not run:")
    print("    - it needs PaddleOCR and the rented A100, neither driven from here;")
    print("    - run the recogniser fine-tune from the config's train block, then")
    print("      report the before/after table per width bucket and ship only if better.")
    return 3


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"training config (default {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the model-free dry run instead of checking the gate for a real launch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="cap the smoke run's epochs (default: the config's train.epochs)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="baseline val_exact_match for the smoke ship decision",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the smoke result as JSON (implies --smoke)",
    )
    args = parser.parse_args(argv)

    path = Path(args.config)
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    try:
        config = load_config(str(path), validate=False)
    except ConfigError as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        return 2

    if config.kind != "training":
        print(
            f"ERROR  {path} has kind={config.kind!r}, not 'training'; this script "
            f"only drives the fine-tune lane",
            file=sys.stderr,
        )
        return 2

    print(f"\n{path}")
    print(f"  kind={config.kind}  run.name={config.get('run.name', '?')}")
    _print_budget(config)
    _print_gate(config)

    if args.smoke or args.json:
        return _run_smoke(config, epochs=args.epochs, baseline=args.baseline, as_json=args.json)
    return _refuse_or_handoff(config)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
