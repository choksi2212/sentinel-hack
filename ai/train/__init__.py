"""The OCR-recogniser fine-tune lane: the go/no-go and a model-free dry run.

Two halves, both deliberately free of any ML framework so they run in CI and
next to config/validation:

  * `gate` -- reads config/training.yaml and refuses to call the paid run's
    gate open while any attestation is false, the dataset is absent, or the
    budget does not add up. There is no override.
  * `smoke` -- runs the fine-tune loop's control logic (batching, the frozen
    phase, checkpoint selection, early stop, the ship decision) on seeded
    synthetic numbers, so the harness that wraps the real A100 run is trusted
    before the instance is billed.

The real fine-tune itself is a MANUAL STEP: it needs PaddleOCR, a labelled
dataset that is not this lane's to produce, and the A100. This package is what
decides whether that step is allowed to happen and proves the code around it is
correct.
"""

from ai.train.gate import (
    TRAINING_GATE_FLAGS,
    budget_blockers,
    budget_summary,
    dataset_blockers,
    gate_blockers,
    gate_flags,
    is_open,
    launch_blockers,
)
from ai.train.smoke import EpochRecord, SmokeResult, smoke_train

__all__ = [
    "TRAINING_GATE_FLAGS",
    "EpochRecord",
    "SmokeResult",
    "budget_blockers",
    "budget_summary",
    "dataset_blockers",
    "gate_blockers",
    "gate_flags",
    "is_open",
    "launch_blockers",
    "smoke_train",
]
