"""Worker-level guards: delivery integrity, config strictness, lifecycle.

The worker is stage 1 + orchestration. Most of its logic is glue, but a
handful of decisions decide whether a run *silently loses data* or *lies
about success* -- those are the ones pinned here, hardest failure first.

Worst-first ordering:

  TIER 1  Delivery integrity. A run that spooled during an outage and then
          recovered must report success; a run still holding undelivered
          events (pending spool, drops, rejects, a full spool) must NOT.
          A watchlist that matches on un-normalised text silently misses
          every hit. A silent now() fallback for a bad replay anchor would
          corrupt replay determinism. These are the data-loss cliffs.

  TIER 2  Config strictness + lifecycle. A typo'd config key must fail loud,
          not be ignored. Ctrl-C once asks the run to wind down; twice is an
          abort. Source counters must survive a broken stats() call.

  TIER 3  CLI surface. --config is mandatory; a missing config file exits
          with the dedicated config code, never a stack trace.

Collaborators are constructed directly from dicts: AppConfig is a plain
dataclass and the sinks are import-light, so no YAML round-trip or network
is needed here (the loader and the sinks own their own suites).
"""

from __future__ import annotations

import argparse
import signal
from datetime import datetime, timezone

import pytest

from ai.config import AppConfig, ConfigError
from ai.emit.http_sink import (
    DEFAULT_INGEST_PATH,
    FileEventSink,
    HttpEventSink,
    NullEventSink,
)
from ai.metrics import RunCounters
import ai.worker as worker


# --------------------------------------------------------------------------
# Small fakes. A sink here is only ever asked for stats(); nothing opens a
# socket or touches disk.
# --------------------------------------------------------------------------

class StatsSink:
    """A sink that reports exactly the stats dict it is handed."""

    def __init__(self, stats):
        self._stats = stats

    def stats(self):
        return dict(self._stats)


class BrokenStatsSink:
    def stats(self):
        raise RuntimeError("stats backend unavailable")


class NonDictStatsSink:
    def stats(self):
        return ["not", "a", "mapping"]


def _cfg(raw):
    """An AppConfig built straight from a raw dict, no file behind it."""
    return AppConfig(path="test.yaml", raw=raw)


# ==========================================================================
# TIER 1 -- delivery integrity (the data-loss cliffs)
# ==========================================================================

class TestReportUndelivered:
    """`_report_undelivered` decides whether the run may claim success.

    It returns True when the run MUST NOT report success. The distinction
    that matters most: a spool that filled and then *drained* is a survived
    outage, not a failure -- `spooled` counts events that were spooled at
    some point, `pending_spool` counts events still stuck. Only the latter
    (plus hard losses) may fail the run.
    """

    def test_clean_run_does_not_fail_and_records_sink_stats(self):
        stats = {}
        fail = worker._report_undelivered(StatsSink({"accepted": 12}), stats)
        assert fail is False
        # The sink's own numbers are folded into the run stats so the
        # summary carries them even on the happy path.
        assert stats["sink"] == {"accepted": 12}

    def test_recovered_spool_does_not_fail_the_run(self):
        # Spooled during an outage, then fully replayed: pending is zero.
        # This is the load-bearing case -- a recovered run is a success.
        fail = worker._report_undelivered(
            StatsSink({"spooled": 40, "replayed": 40, "pending_spool": 0}), {}
        )
        assert fail is False

    def test_pending_spool_fails_the_run(self):
        # Events still on disk, never delivered -> silent loss if we shrug.
        assert worker._report_undelivered(StatsSink({"pending_spool": 1}), {}) is True

    def test_dropped_events_fail_the_run(self):
        assert worker._report_undelivered(StatsSink({"dropped": 1}), {}) is True

    def test_full_spool_fails_the_run(self):
        # The spool hit its ceiling and shed events -- hard loss.
        assert worker._report_undelivered(StatsSink({"spool_full_events": 1}), {}) is True

    def test_rejected_events_fail_the_run(self):
        # 422s are permanent server rejections; a run that produced any
        # must not be called clean.
        assert worker._report_undelivered(StatsSink({"rejected": 1}), {}) is True

    def test_broken_stats_call_does_not_fail_the_run(self):
        # A sink whose stats() raises can't prove loss; the run is not
        # failed on the strength of a broken probe (it warns instead).
        assert worker._report_undelivered(BrokenStatsSink(), {}) is False

    def test_non_dict_stats_does_not_fail_the_run(self):
        assert worker._report_undelivered(NonDictStatsSink(), {}) is False


class TestBuildSink:
    """`build_sink` wires emit config to a concrete sink.

    The ingest *path* is fixed by contract and never taken from config; a
    typo'd or empty URL must fail loudly rather than post nowhere.
    """

    def test_missing_emit_yields_null_sink(self):
        assert isinstance(worker.build_sink(_cfg({}), {}), NullEventSink)

    def test_off_none_null_all_yield_null_sink(self):
        for kind in ("off", "none", "null"):
            sink = worker.build_sink(_cfg({"emit": {"sink": kind}}), {})
            assert isinstance(sink, NullEventSink), kind

    def test_file_sink_selected(self):
        sink = worker.build_sink(
            _cfg({"emit": {"sink": "file", "path": "runs/events.jsonl"}}), {}
        )
        assert isinstance(sink, FileEventSink)

    def test_http_sink_appends_fixed_ingest_path(self):
        # The URL from config is the base; the contract path is appended and
        # is NOT configurable -- so a base with a trailing slash still lands
        # on exactly one ingest path.
        sink = worker.build_sink(
            _cfg({"emit": {"sink": "http", "url": "http://localhost:8000/"}}),
            {worker.INGEST_TOKEN_VAR: "synthetic-ingest-token"},
        )
        assert isinstance(sink, HttpEventSink)
        assert sink.url == "http://localhost:8000" + DEFAULT_INGEST_PATH

    def test_http_sink_empty_url_is_a_config_error(self):
        with pytest.raises(ConfigError):
            worker.build_sink(_cfg({"emit": {"sink": "http", "url": ""}}), {})

    def test_unknown_sink_kind_is_a_config_error(self):
        with pytest.raises(ConfigError):
            worker.build_sink(_cfg({"emit": {"sink": "carrier-pigeon"}}), {})


class TestLoadWatchlist:
    """A watchlist decides which plates raise an alert.

    Emit and dedup key on the *normalised* plate, so the watchlist must too.
    A list matched against raw human input ("GJ-01-AB-1234") would never fire
    against a normalised read ("GJ01AB1234") -- every hit silently missed.
    """

    def test_human_formatted_entry_matches_normalised_read(self, tmp_path):
        wl = tmp_path / "watch.txt"
        wl.write_text("GJ-01-AB-1234\n", encoding="utf-8")
        pred = worker.load_watchlist(str(wl))
        assert pred is not None
        assert pred("GJ01AB1234") is True
        assert pred("MH12ZZ9999") is False

    def test_lowercase_entry_is_normalised(self, tmp_path):
        wl = tmp_path / "watch.txt"
        wl.write_text("gj05mn6789\n", encoding="utf-8")
        pred = worker.load_watchlist(str(wl))
        assert pred("GJ05MN6789") is True

    def test_comments_and_blanks_are_skipped(self, tmp_path):
        wl = tmp_path / "watch.txt"
        wl.write_text("# a comment\n\nGJ01AB1234\n\n", encoding="utf-8")
        pred = worker.load_watchlist(str(wl))
        assert pred("GJ01AB1234") is True

    def test_unnormalisable_entry_does_not_become_match_all(self, tmp_path):
        # "!!" normalises to empty; it must be dropped, never stored as an
        # empty-string entry that a normalised read could accidentally equal.
        wl = tmp_path / "watch.txt"
        wl.write_text("!!\nGJ01AB1234\n", encoding="utf-8")
        pred = worker.load_watchlist(str(wl))
        assert pred("") is False
        assert pred("GJ01AB1234") is True

    def test_empty_or_all_skipped_watchlist_is_none(self, tmp_path):
        wl = tmp_path / "watch.txt"
        wl.write_text("# only comments\n!!\n", encoding="utf-8")
        assert worker.load_watchlist(str(wl)) is None

    def test_no_watchlist_path_is_none(self):
        assert worker.load_watchlist(None) is None


class TestReplayAnchor:
    """`_replay_anchor` reads run.replay_anchor for deterministic replay.

    A bad anchor must raise, never fall back to now() -- a silent now()
    would make a replay of the same footage stamp different observed_at
    values on every run, quietly destroying reproducibility.
    """

    def test_absent_anchor_is_none(self):
        assert worker._replay_anchor(_cfg({"run": {}})) is None

    def test_valid_iso_anchor_parses_to_aware_datetime(self):
        anchor = worker._replay_anchor(
            _cfg({"run": {"replay_anchor": "2026-09-01T10:00:00Z"}})
        )
        assert anchor == datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_unparseable_anchor_raises_rather_than_defaulting_to_now(self):
        with pytest.raises(ConfigError):
            worker._replay_anchor(_cfg({"run": {"replay_anchor": "yesterday"}}))

    def test_unknown_run_key_is_a_config_error(self):
        with pytest.raises(ConfigError):
            worker._replay_anchor(_cfg({"run": {"replay_anchr": "typo"}}))


# ==========================================================================
# TIER 2 -- config strictness + lifecycle
# ==========================================================================

class TestSubConfig:
    """`_sub_config` is the strict-key gate: a typo must not be swallowed."""

    def test_known_keys_pass_through(self):
        got = worker._sub_config(
            _cfg({"dedup": {"window_seconds": 5}}), "dedup", worker._DEDUP_KEYS
        )
        assert got == {"window_seconds": 5}

    def test_missing_section_is_empty_dict(self):
        assert worker._sub_config(_cfg({}), "dedup", worker._DEDUP_KEYS) == {}

    def test_unknown_key_is_a_config_error(self):
        with pytest.raises(ConfigError):
            worker._sub_config(
                _cfg({"quality": {"min_hieght_px": 40}}),  # deliberate typo
                "quality",
                worker._GATE_KEYS,
            )


class TestStopFlag:
    """First Ctrl-C requests an orderly stop; a second aborts hard.

    The handler is invoked directly rather than by raising a real signal, so
    the original SIGINT/SIGTERM handlers are saved and restored to leave the
    test runner's own signal handling untouched.
    """

    def test_first_signal_requests_stop_without_raising(self):
        orig_int = signal.getsignal(signal.SIGINT)
        orig_term = signal.getsignal(signal.SIGTERM)
        try:
            flag = worker.StopFlag()
            flag.install()
            assert flag.requested is False
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)  # must not raise
            assert flag.requested is True
            assert flag.reason  # a non-empty reason is recorded
        finally:
            signal.signal(signal.SIGINT, orig_int)
            signal.signal(signal.SIGTERM, orig_term)

    def test_second_signal_aborts_with_keyboard_interrupt(self):
        orig_int = signal.getsignal(signal.SIGINT)
        orig_term = signal.getsignal(signal.SIGTERM)
        try:
            flag = worker.StopFlag()
            flag.install()
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        finally:
            signal.signal(signal.SIGINT, orig_int)
            signal.signal(signal.SIGTERM, orig_term)


class TestApplySourceCounters:
    """Source telemetry feeds the run counters; a broken source is survivable."""

    def test_decoded_and_dropped_are_copied(self):
        class Source:
            def stats(self):
                return {"sampler": {"decoded": 100}, "buffer": {"dropped": 7}}

        counters = RunCounters()
        worker._apply_source_counters(counters, Source())
        assert counters.frames_seen == 100
        assert counters.frames_dropped_late == 7

    def test_broken_source_stats_leaves_counters_at_zero(self):
        class Source:
            def stats(self):
                raise RuntimeError("source went away")

        counters = RunCounters()
        worker._apply_source_counters(counters, Source())
        assert counters.frames_seen == 0
        assert counters.frames_dropped_late == 0


class TestDefaultRunId:
    """A run id is name_camera_timestamp; the name defaults to 'run'."""

    def test_uses_configured_run_name(self):
        rid = worker._default_run_id(_cfg({"run": {"name": "demo"}}), "cam04")
        assert rid.startswith("demo_cam04_")

    def test_defaults_name_when_absent(self):
        rid = worker._default_run_id(_cfg({}), "cam04")
        assert rid.startswith("run_cam04_")


# ==========================================================================
# TIER 3 -- CLI surface
# ==========================================================================

class TestArgParser:
    def test_config_is_required(self):
        parser = worker.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_set_overrides_accumulate(self):
        parser = worker.build_arg_parser()
        ns = parser.parse_args(
            ["--config", "c.yaml", "--set", "a.b=1", "--set", "c.d=2"]
        )
        assert ns.config == "c.yaml"
        assert ns.overrides == ["a.b=1", "c.d=2"]

    def test_defaults(self):
        parser = worker.build_arg_parser()
        ns = parser.parse_args(["--config", "c.yaml"])
        assert isinstance(parser, argparse.ArgumentParser)
        assert ns.overrides == []
        assert ns.camera is None
        assert ns.summary_root == "."
        assert ns.no_summary is False
        assert ns.env_file == ".env"


class TestMain:
    def test_missing_config_file_exits_with_config_code(self):
        code = worker.main(["--config", "does_not_exist_9f3a2b.yaml"])
        assert code == worker.EXIT_CONFIG
