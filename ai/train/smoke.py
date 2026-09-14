"""A deterministic dry run of the fine-tune loop, with no model in it.

The real fine-tune (config/training.yaml, target `ocr_recognizer`) runs on a
rented A100 on D7, once and under time pressure, wrapped in PaddleOCR's own
training YAML. The failure mode that costs the most is not the model diverging;
it is the *harness* around it being wrong -- selecting the checkpoint on the
wrong metric, ignoring the early-stop, shipping a fine-tune that lost to the
baseline -- and discovering it after the instance is billed and destroyed.

So the control logic is separated from the model and exercised here, for free,
on synthetic numbers:

  * batching, including the last partial batch;
  * the frozen-backbone phase (the head warms up first, on its own);
  * checkpoint selection on `val_exact_match`, not training loss;
  * early stop after `early_stop_patience` epochs without improvement;
  * `ship_only_if_better`: a fine-tune that does not beat the baseline on the
    held-out split does not ship.

Every number here is drawn from a seeded generator, so the same config produces
the same run -- a smoke test that flickered would be worse than none. The curve
is a saturating exponential with a lower ceiling while the backbone is frozen;
it is not a claim about how the recogniser will actually learn, only a shape
with a plateau, so the plateau-driven paths (selection, early stop) are the
ones that get tested. `--smoke` in scripts/train.py runs exactly this and is
the one path that bypasses the go/no-go, because it spends nothing and touches
no dataset.
"""

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ai.config import AppConfig

# The synthetic learning curve. A lower ceiling while frozen so the frozen phase
# is visibly the warm-up it is, a higher one after, and a rate that plateaus
# inside thirty epochs so early stop has something to fire on. Constants, not
# magic numbers, because a test reads them back to check the phases are distinct.
FROZEN_CEILING = 0.55
UNFROZEN_CEILING = 0.86
CURVE_RATE = 0.5
CURVE_NOISE_SD = 0.004

# The synthetic dataset size, in crops. Only its interaction with batch_size
# matters here -- that the last partial batch is counted, not dropped -- so it
# is deliberately not a multiple of the config's batch size of 128.
SMOKE_SAMPLES = 500

# Stand-in baseline for the ship decision when none is passed. The real baseline
# is measured on the held-out split and does not exist yet; this only proves the
# comparison runs, it is not a number about the recogniser.
DEFAULT_BASELINE = 0.5


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    frozen: bool
    val_exact_match: float
    best_so_far: float


@dataclass(frozen=True)
class SmokeResult:
    """What the dry run proves. Every field maps to one control-logic property."""

    epochs_run: int
    max_epochs: int
    stopped_early: bool
    batch_size: int
    batches_per_epoch: int
    freeze_backbone_epochs: int
    select_on: str
    best_epoch: int
    best_val: float
    baseline: float
    ship: bool
    history: tuple[EpochRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs_run": self.epochs_run,
            "max_epochs": self.max_epochs,
            "stopped_early": self.stopped_early,
            "batch_size": self.batch_size,
            "batches_per_epoch": self.batches_per_epoch,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
            "select_on": self.select_on,
            "best_epoch": self.best_epoch,
            "best_val": self.best_val,
            "baseline": self.baseline,
            "ship": self.ship,
            "history": [
                {
                    "epoch": r.epoch,
                    "frozen": r.frozen,
                    "val_exact_match": r.val_exact_match,
                    "best_so_far": r.best_so_far,
                }
                for r in self.history
            ],
        }


def _batches_per_epoch(n_samples: int, batch_size: int) -> int:
    """Ceiling division -- the last partial batch is a batch, not a rounding error.

    Floor division here would silently drop up to batch_size-1 crops every
    epoch, which on a small plate corpus is a measurable slice of the data the
    fine-tune never sees.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return (n_samples + batch_size - 1) // batch_size


def _mean_curve(epoch: int, frozen_epochs: int) -> float:
    """The noise-free val_exact_match for a 1-based epoch."""
    ceiling = FROZEN_CEILING if epoch <= frozen_epochs else UNFROZEN_CEILING
    return ceiling * (1.0 - exp(-CURVE_RATE * epoch))


def smoke_train(
    config: "AppConfig",
    *,
    max_epochs: Optional[int] = None,
    seed: Optional[int] = None,
    baseline: Optional[float] = None,
    n_samples: int = SMOKE_SAMPLES,
) -> SmokeResult:
    """Run the fine-tune loop's control logic on synthetic numbers.

    Reads the hyperparameters from the config's `train` block so the dry run
    reflects the same knobs the real run would use; the overrides exist for
    tests and for a shorter smoke. Bypasses the go/no-go by construction -- it
    reads no gate flag and touches no dataset, because it spends nothing.
    """
    train = config.section("train")
    epochs = int(max_epochs if max_epochs is not None else train.get("epochs", 30))
    batch_size = int(train.get("batch_size", 128))
    freeze_epochs = int(train.get("freeze_backbone_epochs", 0))
    patience = int(train.get("early_stop_patience", 5))
    select_on = str(train.get("select_on", "val_exact_match"))
    if seed is None:
        seed = int(config.get("dataset.seed", 1337))
    if baseline is None:
        baseline = DEFAULT_BASELINE

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if patience < 1:
        raise ValueError(f"early_stop_patience must be >= 1, got {patience}")

    rng = np.random.default_rng(seed)
    batches = _batches_per_epoch(n_samples, batch_size)

    history: list[EpochRecord] = []
    best_val = -1.0
    best_epoch = 0
    since_improved = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        frozen = epoch <= freeze_epochs
        noisy = _mean_curve(epoch, freeze_epochs) + float(rng.normal(0.0, CURVE_NOISE_SD))
        val = float(min(1.0, max(0.0, noisy)))

        # Selection on val_exact_match: strictly-better only, so a later epoch
        # that merely ties the best does not steal the checkpoint. A plate with
        # one wrong character is a wrong plate, which is why the metric is exact
        # match and not CER -- recorded here so the dry run selects the way the
        # real run must.
        if val > best_val:
            best_val = val
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1

        history.append(EpochRecord(epoch, frozen, round(val, 6), round(best_val, 6)))

        # Early stop after `patience` epochs with no new best. Checked after the
        # record is appended so the epoch that triggered the stop is in the
        # history, not silently dropped.
        if since_improved >= patience:
            stopped_early = True
            break

    ship = best_val > float(baseline)  # ship_only_if_better, on the held-out metric
    return SmokeResult(
        epochs_run=len(history),
        max_epochs=epochs,
        stopped_early=stopped_early,
        batch_size=batch_size,
        batches_per_epoch=batches,
        freeze_backbone_epochs=freeze_epochs,
        select_on=select_on,
        best_epoch=best_epoch,
        best_val=round(best_val, 6),
        baseline=float(baseline),
        ship=ship,
        history=tuple(history),
    )


__all__ = [
    "CURVE_NOISE_SD",
    "CURVE_RATE",
    "DEFAULT_BASELINE",
    "FROZEN_CEILING",
    "SMOKE_SAMPLES",
    "UNFROZEN_CEILING",
    "EpochRecord",
    "SmokeResult",
    "smoke_train",
]
