"""The OCR-recogniser fine-tune lane: the go/no-go and the model-free dry run.

Owner's manual section 7. The A100 is the only paid resource, so the two things
tested here are the two things that stop money being spent on a guess:

  * the gate refuses to open while any attestation is false, the dataset is
    absent, or the budget does not add up -- and there is no override, so the
    refusal is the whole feature;

  * the smoke run reproduces the fine-tune loop's control logic exactly from a
    seed, so the harness that will wrap the real run on D7 -- checkpoint
    selection, early stop, the ship decision -- is proven correct before the
    instance is billed and destroyed.

The gate flags are attestations a human flips; the manifest and budget checks
are the machine verifying them. Both are pinned so neither can quietly turn into
a rubber stamp.
"""

import pytest

from ai.config import AppConfig
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
from ai.train.smoke import (
    FROZEN_CEILING,
    UNFROZEN_CEILING,
    SmokeResult,
    smoke_train,
)

FLAG_KEYS = ("local_benchmark_all_green", "labelled_dataset_exists",
             "held_out_split_exists", "baseline_measured_on_held_out")


def training_config(**overrides) -> AppConfig:
    """A structurally-complete training config, gate closed by default.

    Built as a dict rather than loaded from YAML so the gate tests need no
    PyYAML and no file on disk. `overrides` replaces whole top-level sections,
    which is all these tests need.
    """
    raw = {
        "kind": "training",
        "run": {"name": "training"},
        "gate": {key: False for key in FLAG_KEYS},
        "budget": {
            "provider": "runpod",
            "instance": "A100 SXM 80GB",
            "usd_per_hour": 1.64,
            "max_hours": 4,
            "ceiling_usd": 6.56,
        },
        "dataset": {
            "root": "data/plates/",
            "train_manifest": "data/plates/train.jsonl",
            "val_manifest": "data/plates/val.jsonl",
            "seed": 1337,
        },
        "train": {
            "epochs": 30,
            "batch_size": 128,
            "freeze_backbone_epochs": 3,
            "early_stop_patience": 5,
            "select_on": "val_exact_match",
        },
    }
    raw.update(overrides)
    return AppConfig(path=None, raw=raw)


def open_gate() -> dict:
    return {key: True for key in FLAG_KEYS}


# ------------------------------------------------------------------- gate flags


def test_the_four_flags_are_the_manuals_four_in_order():
    """The set is copied from the manual; a drift here is a drift in the go/no-go."""
    assert tuple(key for key, _ in TRAINING_GATE_FLAGS) == FLAG_KEYS


def test_a_fresh_config_has_every_flag_false_and_the_gate_shut():
    config = training_config()
    assert gate_flags(config) == {key: False for key in FLAG_KEYS}
    assert is_open(config) is False
    assert len(gate_blockers(config)) == 4


def test_flipping_all_four_opens_the_gate():
    config = training_config(gate=open_gate())
    assert gate_flags(config) == {key: True for key in FLAG_KEYS}
    assert gate_blockers(config) == []
    assert is_open(config) is True


def test_a_missing_flag_reads_as_unset_not_as_permission():
    """False-on-missing is the safe default for a spend gate; the blocker names
    it as unset so the reason is not lost in the coercion."""
    gate = open_gate()
    del gate["held_out_split_exists"]
    config = training_config(gate=gate)
    assert is_open(config) is False
    blockers = gate_blockers(config)
    assert any("held_out_split_exists is unset" in b for b in blockers)


def test_a_truthy_but_non_true_flag_does_not_open_the_gate():
    """`is True`, not truthiness. A stray 'yes' or 1 in the YAML must not be
    mistaken for the deliberate boolean the attestation requires."""
    for stand_in in ("yes", 1, "true"):
        config = training_config(gate={**open_gate(), "labelled_dataset_exists": stand_in})
        assert is_open(config) is False, f"{stand_in!r} should not open the gate"
        assert any("labelled_dataset_exists" in b for b in gate_blockers(config))


def test_each_blocker_names_the_evidence_it_is_waiting_on():
    config = training_config()
    blockers = gate_blockers(config)
    assert any("11 benchmark rows" in b for b in blockers)
    assert any("labelled dataset" in b for b in blockers)
    assert any("held-out split" in b for b in blockers)
    assert any("baseline" in b for b in blockers)


# --------------------------------------------------------------- dataset on disk


def test_the_dataset_manifests_are_a_blocker_until_they_exist():
    """Separate from the flags because their absence is expected in CI -- the
    crops are real plates and data/ is gitignored."""
    config = training_config()
    blockers = dataset_blockers(config)
    assert len(blockers) == 2
    assert all("MANUAL STEP" in b for b in blockers)


def test_manifests_on_disk_clear_the_dataset_blocker(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    config = training_config(dataset={
        "train_manifest": str(train),
        "val_manifest": str(val),
        "seed": 1337,
    })
    assert dataset_blockers(config) == []


def test_an_unset_manifest_key_is_reported_as_unset():
    config = training_config(dataset={"seed": 1337})
    blockers = dataset_blockers(config)
    assert any("train_manifest is not set" in b for b in blockers)
    assert any("val_manifest is not set" in b for b in blockers)


# ------------------------------------------------------------------- the budget


def test_the_budget_summary_resolves_the_spend_envelope():
    summary = budget_summary(training_config())
    assert summary["provider"] == "runpod"
    assert summary["usd_per_hour"] == 1.64
    assert summary["max_hours"] == 4
    assert summary["ceiling_usd"] == 6.56


def test_a_ceiling_that_matches_the_arithmetic_has_no_blocker():
    assert budget_blockers(training_config()) == []


def test_a_ceiling_that_does_not_match_rate_times_hours_is_a_blocker():
    """A ceiling raised without changing the hours is how a $6 run becomes a
    $90 one; the arithmetic is checked so the number stays honest."""
    config = training_config(budget={
        "provider": "runpod", "instance": "A100 SXM 80GB",
        "usd_per_hour": 1.64, "max_hours": 4, "ceiling_usd": 90.0,
    })
    blockers = budget_blockers(config)
    assert len(blockers) == 1
    assert "does not match" in blockers[0] or "!=" in blockers[0]


def test_a_missing_budget_number_is_a_blocker():
    config = training_config(budget={"provider": "runpod", "instance": "A100 SXM 80GB"})
    blockers = budget_blockers(config)
    assert any("usd_per_hour" in b for b in blockers)
    assert any("max_hours" in b for b in blockers)
    assert any("ceiling_usd" in b for b in blockers)


# ----------------------------------------------------------------- launch gate


def test_launch_is_blocked_until_gate_dataset_and_budget_are_all_clear(tmp_path):
    """A real run needs all three: the human decision, the dataset on disk, and
    a budget that adds up. Any one missing is a refusal."""
    # Closed gate, absent manifests: many blockers.
    assert launch_blockers(training_config()) != []

    # Open gate + real manifests + consistent budget: clear to launch.
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    ready = training_config(
        gate=open_gate(),
        dataset={"train_manifest": str(train), "val_manifest": str(val), "seed": 1337},
    )
    assert launch_blockers(ready) == []


# --------------------------------------------------------------- the smoke run


def test_the_smoke_run_is_deterministic():
    """A smoke test that flickered would be worse than none: the whole value is
    that the same config produces the same run."""
    config = training_config()
    first = smoke_train(config)
    second = smoke_train(config)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_the_smoke_run_reproduces_the_pinned_seed_1337_result():
    """The exact numbers for seed 1337 on the shipped hyperparameters. This is
    the harness nailed down: change the loop and this is what tells you."""
    result = smoke_train(training_config())
    assert result.batch_size == 128
    assert result.batches_per_epoch == 4          # ceil(500 / 128)
    assert result.freeze_backbone_epochs == 3
    assert result.select_on == "val_exact_match"
    assert result.stopped_early is True
    assert result.epochs_run == 21                # best at 16, patience 5
    assert result.best_epoch == 16
    assert result.best_val == pytest.approx(0.866963, abs=1e-6)
    assert result.ship is True                    # 0.867 > baseline 0.5


def test_the_last_partial_batch_is_counted_not_dropped():
    """Floor division would silently drop up to batch_size-1 crops an epoch."""
    config = training_config(train={**training_config().section("train"), "batch_size": 128})
    result = smoke_train(config, n_samples=500)
    assert result.batches_per_epoch == 4          # not 3
    exact = smoke_train(config, n_samples=512)
    assert exact.batches_per_epoch == 4           # a clean multiple


def test_the_frozen_phase_is_the_first_n_epochs_and_visibly_lower():
    """The backbone is frozen while the head warms up; the config's three frozen
    epochs must be the first three, and their accuracy must sit below the
    unfrozen phase or the phase distinction is not being modelled at all."""
    result = smoke_train(training_config())
    frozen = [r for r in result.history if r.frozen]
    unfrozen = [r for r in result.history if not r.frozen]
    assert [r.epoch for r in frozen] == [1, 2, 3]
    assert max(r.val_exact_match for r in frozen) < min(r.val_exact_match for r in unfrozen)
    # And the ceilings the two phases are drawn from are genuinely different.
    assert FROZEN_CEILING < UNFROZEN_CEILING


def test_the_selected_checkpoint_is_the_best_val_and_the_best_is_monotonic():
    """Selection is on val_exact_match, best-so-far, strictly better -- so the
    recorded best never decreases and the chosen epoch is the argmax."""
    result = smoke_train(training_config())
    vals = [r.val_exact_match for r in result.history]
    bests = [r.best_so_far for r in result.history]
    assert bests == sorted(bests)                 # non-decreasing
    assert result.best_val == max(vals)
    assert result.history[result.best_epoch - 1].val_exact_match == result.best_val


def test_early_stop_fires_after_patience_epochs_without_a_new_best():
    """Best at epoch 16, patience 5, so the run stops at 21 rather than 30 -- and
    the epoch that triggered the stop is in the history, not dropped."""
    result = smoke_train(training_config())
    assert result.stopped_early is True
    assert result.epochs_run == result.best_epoch + 5
    assert result.history[-1].epoch == result.epochs_run


def test_a_run_that_never_plateaus_uses_all_its_epochs():
    """Capped at three epochs -- inside the still-rising part of the curve -- the
    run cannot stall, so it must use every epoch and report stopped_early false."""
    result = smoke_train(training_config(), max_epochs=3)
    assert result.epochs_run == 3
    assert result.stopped_early is False


def test_ship_only_if_better_keeps_the_baseline_when_the_fine_tune_loses():
    """A fine-tune that does not beat the baseline on the held-out split does not
    ship; the smoke proves the comparison, on a baseline set above the curve."""
    beats = smoke_train(training_config(), baseline=0.5)
    assert beats.ship is True
    loses = smoke_train(training_config(), baseline=0.95)
    assert loses.ship is False


def test_different_seeds_give_different_runs():
    a = smoke_train(training_config(), seed=1337)
    b = smoke_train(training_config(), seed=7)
    assert a.history != b.history


def test_a_zero_epoch_or_zero_patience_config_is_refused():
    with pytest.raises(ValueError, match="epochs must be"):
        smoke_train(training_config(), max_epochs=0)
    bad_patience = training_config(train={**training_config().section("train"),
                                          "early_stop_patience": 0})
    with pytest.raises(ValueError, match="early_stop_patience"):
        smoke_train(bad_patience)


def test_a_zero_batch_size_is_refused():
    bad = training_config(train={**training_config().section("train"), "batch_size": 0})
    with pytest.raises(ValueError, match="batch_size"):
        smoke_train(bad)


def test_the_smoke_result_is_json_ready():
    payload = smoke_train(training_config()).to_dict()
    import json
    restored = json.loads(json.dumps(payload))
    assert restored["best_epoch"] == 16
    assert len(restored["history"]) == restored["epochs_run"]


# ----------------------------------------------- the shipped config, end to end


def test_the_shipped_training_config_has_the_gate_shut_today():
    """The real config/training.yaml, loaded and checked: every flag is false,
    the budget adds up, and the manifests are absent -- i.e. exactly the refusal
    the script prints today. Guarded on PyYAML like the other load tests."""
    pytest.importorskip("yaml")
    from ai.config import load_config

    config = load_config("config/training.yaml", validate=False)
    assert config.kind == "training"
    assert gate_flags(config) == {key: False for key in FLAG_KEYS}
    assert is_open(config) is False
    assert budget_blockers(config) == []          # 1.64 * 4 == 6.56
    assert dataset_blockers(config)                # manifests gitignored, absent


def test_the_shipped_config_smoke_runs_and_ships_against_the_default_baseline():
    pytest.importorskip("yaml")
    from ai.config import load_config

    config = load_config("config/training.yaml", validate=False)
    result = smoke_train(config)
    assert isinstance(result, SmokeResult)
    assert result.stopped_early is True
    assert result.ship is True
