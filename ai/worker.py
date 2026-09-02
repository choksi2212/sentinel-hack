"""The runnable AI worker: one process, one camera, one config file.

    python -m ai.worker --config config/offline.yaml --camera cam04

Stage 1 of the pipeline and everything around it. ai/pipeline.py owns stages 2 through 13
and deliberately knows nothing about where frames come from or where events go; this module
supplies both, which is what makes the source-independence invariant testable.

The invariant is about the source and nothing else: swapping `source.mode` between the five
modes -- live_rtsp, live_hls, file, frames, synthetic -- must build the identical pipeline
object with the identical stage configuration, and must not require a line of code to change.
If a stage has to change to make live work, the invariant is broken and the fix belongs in
ai/media, not here.

Worth being precise about, because config/offline.yaml and config/live.yaml differ in far more
than `source:` and that is not a violation. offline.yaml is the *synthetic* configuration: it
runs oracle stages, which read the generator's ground truth, so its detect/plate/ocr blocks
describe an instrument rather than a model. The two axes are independent -- which source, and
which stages -- and only the first one is what this invariant constrains. The pair that
actually tests it is one config with `source.*` overridden, which is what tests/ drives.

Six decisions worth stating, because each has a cheaper wrong answer:

**Events go to the sink one at a time, as they are produced.** Batching until the end of the
run is simpler and would halve the number of HTTP calls, but a demo would then show an empty
map for the length of the clip and a crash at minute four would lose four minutes of
sightings. HttpEventSink already batches internally where batching is safe -- a background
thread with a disk spool -- which is the level that can retry.

**frames_seen and frames_dropped_late are set here, from the source's own counters.** The
pipeline only ever sees frames the sampler emitted, so it cannot count the ones it never got.
Letting it try would make frames_seen equal frames_sampled and report a 100% emit rate for a
stage whose entire purpose is dropping nine frames in ten -- and it would report zero dropped
frames for a GPU that could not keep up with the stream, which is the single number that
distinguishes a flaky camera from an overloaded machine.

**Ctrl-C flushes.** At any moment every vehicle currently in frame has an open track buffer
and no event yet. Exiting on the signal loses all of them, which on a live grid is every
vehicle in the last three seconds and is silently a lower event count. The first signal asks
the loop to stop and lets the flush run; a second one is taken as "I meant now" and exits
without it.

**The run summary is written as a BenchmarkReport with no scorecard, so it lands in
runs/diagnostics/ and never in benchmark/reports/.** A worker run has no ground truth
attached -- nobody has said which vehicles were readable -- so it has a latency figure and an
event count but no accuracy. Reusing the locked report shape means one format to read;
leaving the scorecard empty means the refusal machinery in ai/metrics.py keeps it out of the
directory the leaderboard is built from.

**Credentials are read from the environment, never from argv.** A password on a command line
is visible in `ps` to every user on the machine and lands in the shell history file. The
config loader resolves ${VAR} references itself rather than through os.environ, so the live
stream password never becomes an environment variable of the ffmpeg subprocess either.

**A graceful stop exits 0.** Ctrl-C on a demo is how the operator ends the demo, not a
failure, and the events were delivered and the summary written. Exit 1 is a real error, 2 is
a bad config. CI runs the offline config and a nonzero code for a completed run is a false
alarm nobody investigates twice.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ai import PIPELINE_VERSION
from ai.config import AppConfig, ConfigError, load_config, load_env
from ai.contracts.event import EventEnvelope, ModelProvenance
from ai.contracts.timebase import parse_iso
from ai.detect import build_detector
from ai.dedup.key import SightingDeduper
from ai.emit.http_sink import FileEventSink, HttpEventSink, NullEventSink
from ai.emit.snapshot import build_snapshot_writer
from ai.fusion.accumulator import EvidenceAccumulator
from ai.logging_setup import get_logger, log_config, setup_logging
from ai.media import build_source
from ai.metrics import BenchmarkReport, RunCounters, StageIdentity
from ai.normalize.plate import normalize_plate
from ai.ocr import build_ocr_engine
from ai.pipeline import Pipeline
from ai.plate import build_plate_detector
from ai.quality.gate import VehicleGate
from ai.track import build_tracker

log = get_logger("worker")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2

# Keys the optional `quality:` and `dedup:` sections may set. Checked explicitly rather than
# splatted into the constructor, because **cfg on a typo raises TypeError naming a keyword
# argument, which reads as a bug in this file rather than a typo in a YAML file the operator
# just edited.
_GATE_KEYS = ("min_height_px", "min_confidence", "departing_height_px")
_DEDUP_KEYS = ("window_seconds",)
_FUSION_KEYS = ("top_k", "track_idle_ms", "max_track_duration_ms")

# Keys accepted in the `run:` section. Free-text provenance plus the replay anchor, which is
# the only one that changes what the pipeline computes.
_RUN_KEYS = ("name", "notes", "replay_anchor")

# The environment variable holding the ingest bearer token. Named here rather than inlined so
# .env.example and this file cannot drift apart silently.
INGEST_TOKEN_VAR = "TRINETRA_INGEST_TOKEN"


# --------------------------------------------------------------------------- argument parsing


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai.worker",
        description="Run the TRINETRA AI pipeline over one camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m ai.worker --config config/offline.yaml --camera cam04\n"
            "  python -m ai.worker --config config/live.yaml --camera cam07 "
            "--max-seconds 120\n"
            "  python -m ai.worker --config config/benchmark.yaml "
            "--set source.path=data/clip.mp4\n"
        ),
    )
    parser.add_argument("--config", required=True, help="path to a config/*.yaml")
    parser.add_argument(
        "--camera",
        default=None,
        help="camera id, overriding source.camera_id. One config drives thirty cameras.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="override one config value; repeatable. Never use this for a credential.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many sampled frames",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "stop after this much WALL CLOCK time. Not footage time: on a live source the "
            "two are the same, and on a file source --max-frames is the reproducible bound."
        ),
    )
    parser.add_argument(
        "--watchlist",
        default=None,
        help=(
            "file of plates to alert on, one per line, # for comments. Normalized on load, "
            "so GJ-01-AB-1234 and gj01ab1234 are the same entry."
        ),
    )
    parser.add_argument("--run-id", default=None, help="defaults to run name + camera + UTC")
    parser.add_argument(
        "--summary-root",
        default=".",
        help="root under which runs/diagnostics/ is written (default: cwd)",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="skip writing the run summary file"
    )
    parser.add_argument("--env-file", default=".env", help="path to .env (default: .env)")
    parser.add_argument("--log-file", default=None, help="also write logs to this file")
    return parser


# --------------------------------------------------------------------------- construction


def _sub_config(config: AppConfig, name: str, allowed: tuple[str, ...]) -> dict[str, Any]:
    """One optional section, with unknown keys refused by name."""
    block = config.section(name)
    unknown = sorted(k for k in block if k not in allowed)
    if unknown:
        raise ConfigError(
            f"{name}: unknown key(s) {unknown}. Accepted here: {list(allowed)}."
        )
    return block


def _replay_anchor(config: AppConfig) -> Optional[datetime]:
    """Read `run.replay_anchor` if the config pins one.

    Offline events derive observed_at from an anchor plus each frame's PTS (see
    Pipeline._observed_at). Left unpinned the anchor is the moment the run started, so two
    replays of the same clip produce different timestamps and cannot be diffed. Pinning it
    makes a rerun byte-identical, which is what turns "the benchmark is reproducible" into a
    claim someone can check rather than one they have to take on trust.

    An unparseable value raises. Falling back to now() would produce a run that looks pinned
    and is not, and the diff against the previous run would then show every event changed for
    a reason no one could find.
    """
    raw = _sub_config(config, "run", _RUN_KEYS).get("replay_anchor")
    if raw in (None, ""):
        return None
    try:
        return parse_iso(str(raw))
    except ValueError as exc:
        raise ConfigError(
            f"run.replay_anchor {raw!r} is not a timezone-aware ISO instant: {exc}"
        ) from exc


def build_sink(config: AppConfig, env: dict[str, str]) -> Any:
    """The event sink named by `emit.sink`: file, http, or null.

    The http URL is an origin only -- http://host:port -- and the path comes from
    DEFAULT_INGEST_PATH, which is the contract's own POST /api/v1/ingest/events. The path is
    not a per-machine setting: a config that could point the worker at a different endpoint
    is a config that can silently deliver events nowhere while reporting success, because a
    404 and a 201 are both "the server answered".
    """
    emit = config.section("emit")
    kind = emit.get("sink")
    kind = "null" if kind is None else str(kind).lower()

    if kind == "file":
        path = str(emit.get("path") or "runs/events.jsonl")
        return FileEventSink(path, append=bool(emit.get("append", False)))
    if kind == "http":
        url = str(emit.get("url") or "").strip()
        if not url:
            raise ConfigError(
                "emit.sink is http but emit.url is empty. It comes from "
                "TRINETRA_INGEST_URL, and an empty one is a run that produces nothing and "
                "blames the network."
            )
        token = env.get(INGEST_TOKEN_VAR) or None
        if not token:
            log.warning(
                "%s is not set: posting to ingest unauthenticated. Fine against a local "
                "backend, wrong against anything else.",
                INGEST_TOKEN_VAR,
            )
        return HttpEventSink(
            url,
            api_key=token,
            timeout_s=float(emit.get("timeout_s", 5.0)),
            max_attempts=int(emit.get("max_attempts", 4)),
            spool_dir=emit.get("spool_dir"),
        )
    if kind in ("null", "none", "off"):
        return NullEventSink()
    raise ConfigError(f"emit.sink {kind!r} is not one of file, http, null.")


def load_watchlist(path: Optional[str]) -> Optional[Callable[[str], bool]]:
    """A plate -> bool predicate from a text file, or None.

    The real watchlist lives in Postgres and is the backend's; this is the AI lane's half of
    the alert path, so the OCR and fusion stages can be demonstrated end to end -- plate read,
    plate matched, match_state upgraded to exact -- without a database running.

    Entries are normalized on load with the same normalizer the pipeline applies to a read,
    because otherwise a watchlist written the way a human writes a plate, GJ-01-AB-1234, never
    matches anything and the alert path looks broken rather than mis-keyed.
    """
    if not path:
        return None
    plates: set[str] = set()
    skipped: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.split("#", 1)[0].strip()
            if not raw:
                continue
            normalized = normalize_plate(raw)
            if normalized:
                plates.add(normalized)
            else:
                skipped.append(raw)
    log.info("watchlist: %d plate(s) from %s", len(plates), path)
    if skipped:
        log.warning(
            "watchlist: %d line(s) normalized to nothing and were dropped: %s",
            len(skipped),
            skipped[:5],
        )
    if not plates:
        return None
    return lambda normalized: normalized in plates


def build_pipeline(
    config: AppConfig,
    source: Any,
    *,
    camera_id: str,
    watchlist: Optional[Callable[[str], bool]] = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Assemble the pipeline from config. Returns it and the four stage objects.

    Every stage gets `source=source`. Only the oracle backends use it -- they read ground
    truth off the synthetic frame -- and the factories refuse to build an oracle without one,
    which is deliberate: an oracle that silently degraded to guessing would be a benchmark
    that measured nothing and said so nowhere.
    """
    source_mode = str(config.source_mode or "file")

    detector = build_detector(config.section("detect"), source=source)
    plate_detector = build_plate_detector(config.section("plate"), source=source)
    ocr = build_ocr_engine(config.section("ocr"), source=source)
    tracker = build_tracker(
        config.section("track"), camera_id, source.session_id, source=source
    )

    metrics_cfg = config.section("metrics")
    counters = RunCounters(
        warmup_frames=int(metrics_cfg.get("warmup_frames", 2)),
        log_vram=bool(metrics_cfg.get("log_vram", True)),
    )

    model = ModelProvenance(
        detector=str(getattr(detector, "model_name", "") or type(detector).__name__),
        plate_detector=str(
            getattr(plate_detector, "model_name", "") or type(plate_detector).__name__
        ),
        ocr=str(getattr(ocr, "model_name", "") or type(ocr).__name__),
        tracker=str(getattr(tracker, "model_name", "") or type(tracker).__name__),
        pipeline_version=PIPELINE_VERSION,
        detector_weights_sha256=getattr(detector, "weights_sha256", None),
        plate_detector_weights_sha256=getattr(plate_detector, "weights_sha256", None),
        ocr_weights_sha256=getattr(ocr, "weights_sha256", None),
    )

    pipeline = Pipeline(
        camera_id=camera_id,
        source_mode=source_mode,
        model=model,
        detector=detector,
        tracker=tracker,
        plate_detector=plate_detector,
        ocr=ocr,
        gate=VehicleGate(**_sub_config(config, "quality", _GATE_KEYS)),
        accumulator=EvidenceAccumulator(**_sub_config(config, "fusion", _FUSION_KEYS)),
        deduper=SightingDeduper(**_sub_config(config, "dedup", _DEDUP_KEYS)),
        snapshots=build_snapshot_writer(config.section("snapshot")),
        counters=counters,
        watchlist=watchlist,
        replay_anchor=_replay_anchor(config),
    )
    stages = {
        "detector": detector,
        "tracker": tracker,
        "plate_detector": plate_detector,
        "ocr": ocr,
    }
    return pipeline, stages


# --------------------------------------------------------------------------- the run


class StopFlag:
    """Set by SIGINT/SIGTERM. The loop checks it; the handler does no work.

    A handler that flushed the pipeline would run arbitrary Python inside a signal context,
    from whichever line the interpreter happened to be on -- including the middle of the
    accumulator's own bookkeeping -- and a partially updated buffer flushed from there emits
    events nobody can account for. Setting a bool is the only safe thing to do here.
    """

    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def install(self) -> None:
        def handle(signum: int, _frame: Any) -> None:
            name = signal.Signals(signum).name
            if self.requested:
                # Second signal: the operator has said it twice. Give up the flush rather
                # than appear hung, and say what it costs.
                log.error("%s again -- exiting without flushing open tracks", name)
                raise KeyboardInterrupt
            self.requested = True
            self.reason = name
            log.warning(
                "%s -- stopping after this frame, then flushing open tracks. "
                "Again to exit immediately.",
                name,
            )

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, handle)
            except (ValueError, OSError):
                # No handler off the main thread, and SIGTERM is not deliverable on every
                # platform. Not fatal: the run still stops at EOF or at --max-frames.
                log.debug("could not install handler for %s", sig)


def _apply_source_counters(counters: RunCounters, source: Any) -> None:
    """Copy the two counters only the source can know. See the module docstring."""
    try:
        stats = source.stats()
    except Exception:  # noqa: BLE001 - a broken stats() must not lose the run's totals
        log.warning("source.stats() failed; frames_seen left at 0", exc_info=True)
        return
    sampler = stats.get("sampler") or {}
    decoded = sampler.get("decoded")
    if isinstance(decoded, int):
        counters.frames_seen = decoded
    buffer = stats.get("buffer") or {}
    dropped = buffer.get("dropped")
    if isinstance(dropped, int):
        counters.frames_dropped_late = dropped


def run_once(args: argparse.Namespace) -> int:
    env = load_env(args.env_file)
    try:
        config = load_config(
            args.config,
            env=env,
            env_file=args.env_file,
            overrides=args.overrides,
        ).require_valid()
    except ConfigError as exc:
        log.error("config: %s", exc)
        return EXIT_CONFIG

    setup_logging(config.section("logging"), log_file=args.log_file, force=True)
    for warning in config.warnings():
        log.warning("config: %s", warning)
    log_config(log, config)

    source_cfg = config.source_config(args.camera)
    camera_id = str(source_cfg.get("camera_id") or "").strip()
    if not camera_id:
        log.error(
            "no camera: set source.camera_id in %s or pass --camera. Every event names the "
            "camera it came from and there is no sensible default.",
            args.config,
        )
        return EXIT_CONFIG

    source = build_source(source_cfg, camera_id=camera_id)
    watchlist = load_watchlist(args.watchlist)
    try:
        # Open before anything else is built. Two reasons, both learned the hard way:
        #
        #   * A bad RTSP URL or a --camera typo should cost three seconds of connect timeout,
        #     not forty seconds of loading three models onto a 12 GB card and then failing.
        #   * The tracker is keyed to (camera_id, stream_session_id) and a session id only
        #     exists once the source is connected, so build_pipeline cannot run before this.
        source.open()
        pipeline, stages = build_pipeline(
            config, source, camera_id=camera_id, watchlist=watchlist
        )
        sink = build_sink(config, env)
    except BaseException:
        # Including KeyboardInterrupt: a source opened and then abandoned holds a capture
        # handle and, on the live sources, a reader thread. Close it and re-raise unchanged so
        # main() can still tell a ConfigError from a real failure.
        try:
            source.close()
        except Exception:  # noqa: BLE001
            log.warning("close failed while aborting startup", exc_info=True)
        raise

    counters = pipeline.counters

    run_id = args.run_id or _default_run_id(config, camera_id)
    log.info(
        "run %s: camera %s, mode %s, session %s, %s -> %s",
        run_id,
        camera_id,
        config.source_mode,
        source.session_id,
        args.config,
        type(sink).__name__,
    )

    stop = StopFlag()
    stop.install()

    delivered = 0
    failed = 0
    exit_code = EXIT_OK
    flush_reason = "eof"
    measuring = False
    started_wall = time.perf_counter()

    def deliver(events: list[EventEnvelope]) -> None:
        nonlocal delivered, failed
        for event in events:
            try:
                event.validate()
            except Exception as exc:  # noqa: BLE001
                # An invalid event is a bug in the builder, not in the footage. Refuse it
                # here rather than let ingest return 422 for a batch and lose the rest.
                failed += 1
                log.error("refusing to send an invalid event: %s", exc)
                continue
            sink.send(event)
            delivered += 1

    try:
        pipeline.load()
        sink.open()
        counters.start()
        measuring = True
        for envelope in source:
            deliver(pipeline.process_frame(envelope))
            for change in source.drain_session_events():
                log.info(
                    "session %s -> %s (%s%s) at frame %d",
                    change.previous_session_id or "-",
                    change.new_session_id,
                    change.reason,
                    f": {change.detail}" if change.detail else "",
                    change.at_frame_index,
                )
            if stop.requested:
                flush_reason = "shutdown"
                break
            if args.max_frames is not None and counters.frames_sampled >= args.max_frames:
                log.info("--max-frames %d reached", args.max_frames)
                break
            if (
                args.max_seconds is not None
                and time.perf_counter() - started_wall >= args.max_seconds
            ):
                log.info("--max-seconds %.1f reached", args.max_seconds)
                break
        deliver(pipeline.flush(flush_reason))
    except KeyboardInterrupt:
        # The second signal, or a Ctrl-C outside the loop. Skip the flush -- that is what it
        # asked for -- and still report: a truncated run with honest numbers beats none. Not
        # folded into `except Exception`, which does not catch this and should not.
        open_tracks = pipeline.stats().get("open_tracks", 0)
        log.error("interrupted: %d open track(s) discarded", open_tracks)
        flush_reason = "shutdown"
    except Exception:
        log.exception("run failed")
        exit_code = EXIT_ERROR
    finally:
        if measuring:
            counters.stop()
        _apply_source_counters(counters, source)
        stats = _collect_stats(pipeline, source, sink)
        for closer in (source.close, pipeline.close):
            try:
                closer()
            except Exception:  # noqa: BLE001
                log.warning("close failed", exc_info=True)
        # The sink last and separately: it has undelivered events in a queue and a spool on
        # disk, and closing it is the only thing that drains them. A failure here is a real
        # failure -- events were built and then lost.
        try:
            if not sink.flush(timeout_s=10.0):
                log.error("sink did not drain within 10s; see its spool directory")
                exit_code = EXIT_ERROR
            sink.close(timeout_s=10.0)
        except Exception:
            log.exception("sink close failed: events may be undelivered")
            exit_code = EXIT_ERROR
        # After close, and these numbers replace the pre-close copy in `stats`. The source
        # loses information when it closes; the sink gains it, because close() is what pushes
        # the last queued events to the spool -- so the backlog is only final once it has run.
        if _report_undelivered(sink, stats):
            exit_code = EXIT_ERROR

    log.info("delivered %d event(s), refused %d", delivered, failed)

    # The stage table is not logged separately: format_summary() already carries it, and two
    # copies in one log is how somebody ends up comparing a run against itself.
    report = _build_report(run_id, config, counters, stages, stop, flush_reason)
    log.info("\n%s", report.format_summary())
    if not args.no_summary:
        path = report.write(args.summary_root)
        _write_run_stats(path, stats)
        log.info("summary: %s", path)
    return exit_code


def _default_run_id(config: AppConfig, camera_id: str) -> str:
    name = str(config.section("run").get("name") or "run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}_{camera_id}_{stamp}"


def _report_undelivered(sink: Any, stats: dict[str, Any]) -> bool:
    """Log whatever the sink could not deliver. True means the run must not report success.

    `flush()` returning True is not enough to call a run delivered. It reports that the
    in-memory queue drained, and during an outage every event drains by being written to the
    spool -- so a run that got nothing at all into the database leaves through that branch
    cleanly. The counters below are the ones that say where the events actually went.

    Worth an exit code rather than a log line, because a green exit on a run that put every
    sighting on disk instead of in Postgres is the one failure nobody notices until an
    operator searches for a vehicle that was never stored.
    """
    try:
        sink_stats = sink.stats()
    except Exception:  # noqa: BLE001
        log.warning("could not read sink stats: delivery is unverified", exc_info=True)
        return False
    if not isinstance(sink_stats, dict):
        return False

    stats["sink"] = sink_stats
    undelivered = False
    # pending_spool, not spooled. `spooled` counts spool writes, so an event spooled, replayed
    # and delivered leaves it at 1 for the rest of the run -- and failing a run over an outage
    # it recovered from teaches everybody to ignore the exit code.
    for key, message in (
        ("pending_spool", "left in the spool: undelivered, replayed on the next open"),
        ("dropped", "dropped with no spool configured: lost"),
        ("spool_full_events", "lost to a full spool"),
        # A rejection is a different failure from the other three: the endpoint parsed the
        # event and refused it. deliver() validates every event against the contract before
        # sending, so a 422 here means our reading of the contract and the backend's disagree,
        # which is worth failing a run over even though nothing was lost by accident.
        ("rejected", "rejected by the endpoint: the event did not match its schema"),
    ):
        count = sink_stats.get(key)
        if isinstance(count, int) and count > 0:
            log.error("%d event(s) %s", count, message)
            undelivered = True
    return undelivered


def _collect_stats(pipeline: Pipeline, source: Any, sink: Any) -> dict[str, Any]:
    """Every stage's own counters, gathered before anything is closed.

    Before, not after: a closed source has released its capture and some backends reset their
    counters on close, so the numbers gathered afterwards would be a quieter, wrong version
    of the run that just happened.
    """
    out: dict[str, Any] = {}
    for label, obj in (("pipeline", pipeline), ("source", source), ("sink", sink)):
        try:
            out[label] = obj.stats()
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": repr(exc)}
    return out


def _build_report(
    run_id: str,
    config: AppConfig,
    counters: RunCounters,
    stages: dict[str, Any],
    stop: StopFlag,
    flush_reason: str,
) -> BenchmarkReport:
    """The run summary, deliberately without a scorecard. See the module docstring."""
    notes = [
        "worker run, not a benchmark: no ground truth was scored, so this file carries "
        "throughput and event counts and no accuracy figure. scripts/ owns scoring.",
        f"config: {config.path}",
        f"run: {config.section('run').get('name', '')} -- "
        f"{config.section('run').get('notes', '')}".strip(" -"),
        f"ended: {flush_reason}",
    ]
    if stop.requested:
        notes.append(f"stopped early on {stop.reason}; totals cover the frames processed.")
    return BenchmarkReport.build(
        None,
        counters,
        task="e2e",
        run_id=run_id,
        source_mode=str(config.source_mode or "file"),
        stages={
            label: StageIdentity.from_stage(label, stage) for label, stage in stages.items()
        },
        notes=notes,
    )


def _write_run_stats(report_path: Any, stats: dict[str, Any]) -> None:
    """Per-stage counters beside the report, as <report>.stats.json.

    Not folded into the report: Contracts 7.3 locks that shape and every stage's internal
    counters are not in it. They are the first thing anybody wants when a run comes out wrong,
    so they go next to it rather than into a log that has already scrolled.
    """
    path = str(report_path) + ".stats.json"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, sort_keys=False, default=str)
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Bootstrap logging before the config is read, so a config error is not silent. Replaced
    # by the config's own level once it has loaded.
    setup_logging(level=os.environ.get("TRINETRA_LOG_LEVEL", "INFO"))
    try:
        return run_once(args)
    except ConfigError as exc:
        log.error("config: %s", exc)
        return EXIT_CONFIG
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        # Only reachable before the run's own handler is in place -- during config load or
        # while a model is loading. Nothing has been opened, so there is nothing to flush.
        log.error("interrupted before the run started")
        return EXIT_OK
    except Exception:
        # A failure during startup: no frames were processed, so there is no run to summarize
        # and nothing was written. Mid-run failures are handled inside run_once, which still
        # writes the summary for the frames that did go through.
        log.exception("could not start the run")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
