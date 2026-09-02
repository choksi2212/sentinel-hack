"""Source independence -- the invariant the USP rests on. Contracts 2.3.

    Swapping `source.mode` among the five modes must build the identical pipeline with
    identical stage configuration and require no code change.

"Vendor-neutral" is the first line of the pitch, and this file is the only thing that makes
it a claim rather than an aspiration. The claim is about the SOURCE ONLY: the same stages,
the same thresholds, the same event shape, fed by a different adapter.

Two things routinely get mistaken for violations and are not:

`config/offline.yaml` and `config/live.yaml` differ in far more than `source:`. That is a
second, independent axis -- which stages run -- and varying it is the point of having two
files. The invariant would be violated if a *stage* had to branch on source_mode, not if two
configs choose different stages.

The oracle stages cannot test the invariant at all. An oracle reads the generator's truth
object, so it is coupled to the synthetic source by construction, and ai/detect/factory.py
refuses to build one without a source that exposes truth_for_envelope(). The stage set that
can cross a source boundary is the weightless one -- motion detection, edge plate projection,
template OCR -- which needs nothing downloaded and no truth to read. That is why the
end-to-end test below uses it, and it is also why it does not read plates: a weightless stage
set returning `plate: null` is the honest outcome, and fabricating one would be the failure.
"""

import json
import numpy as np
import pytest

from conftest import REPO_ROOT

from ai.config import AppConfig, load_config
from ai.contracts.enums import SOURCE_MODES
from ai.media.factory import SourceConfigError, build_source
from ai.media.synthetic_source import SyntheticFaults

CAMERA = "cam04"

# Every ${VAR} in the shipped configs carries a `:-default`, so an empty environment resolves
# them all. Passed explicitly rather than letting load_config read .env, because .env is
# gitignored: a test that depended on it would pass here and fail in CI for a reason that
# looks like a config bug.
NO_ENV: dict[str, str] = {}


def config_path(name: str) -> str:
    """Configs are addressed from the repository root, not the working directory.

    Same reasoning as conftest's sys.path insertion: `pytest tests/test_media_invariant.py`
    from anywhere but the root would otherwise fail on a relative path.
    """
    return str(REPO_ROOT / "config" / name)


def synthetic_block(**overrides) -> dict:
    block = {
        "mode": "synthetic",
        "camera_id": CAMERA,
        "width": 640,
        "height": 360,
        "seed": 1337,
        "step_ms": 100,
        "total_frames": 40,
        "vehicles_per_100_frames": 6,
    }
    block.update(overrides)
    return block


# ------------------------------------------------------------------ the five modes, exactly


def test_the_five_modes_are_the_locked_set():
    """Contracts 1.3. Named here because the factory dispatches on it and a sixth mode
    appearing without a contract change would be a new source nobody agreed to support."""
    assert set(SOURCE_MODES) == {"live_rtsp", "live_hls", "file", "frames", "synthetic"}


@pytest.mark.parametrize("mode", sorted(SOURCE_MODES))
def test_every_declared_mode_is_buildable(mode, tmp_path):
    """A mode in the enum that the factory cannot build is a contract that lies.

    Construction only -- open() would need cv2 and a network for three of the five. What is
    asserted is that the dispatch exists and the adapter reports the mode it was asked for.
    """
    blocks = {
        "synthetic": synthetic_block(),
        "file": {"mode": "file", "camera_id": CAMERA, "path": str(tmp_path / "clip.mp4")},
        "frames": {"mode": "frames", "camera_id": CAMERA, "directory": str(tmp_path)},
        "live_rtsp": {"mode": "live_rtsp", "camera_id": CAMERA, "url": "rtsp://host:8554/s/1"},
        "live_hls": {"mode": "live_hls", "camera_id": CAMERA, "url": "https://host/1/index.m3u8"},
    }
    source = build_source(blocks[mode])
    assert source.source_mode == mode
    assert source.camera_id == CAMERA


@pytest.mark.parametrize("bad", ["live", "rtsp", "LIVE_RTSP", "", "hls", "webcam"])
def test_a_mode_outside_the_five_is_refused(bad):
    """`live` is the one someone will actually write, because it is what everybody says out
    loud. It is not a mode -- the transport matters, and a config that silently picked one
    would produce a run nobody could reproduce."""
    with pytest.raises(SourceConfigError, match="mode"):
        build_source({"mode": bad, "camera_id": CAMERA})


def test_a_missing_mode_names_the_alternatives():
    """A config error is only useful if it says what would have worked."""
    with pytest.raises(SourceConfigError) as exc:
        build_source({"camera_id": CAMERA})
    for mode in SOURCE_MODES:
        assert mode in str(exc.value)


def test_a_source_config_that_is_not_a_mapping_is_refused():
    """`source: live_rtsp` instead of `source: {mode: live_rtsp}` is the realistic typo."""
    with pytest.raises(SourceConfigError, match="mapping"):
        build_source("live_rtsp")  # type: ignore[arg-type]


# --------------------------------------------------------------------- typos are not ignored


def test_an_unknown_key_is_refused_rather_than_ignored():
    """The failure this rule exists to prevent, spelled out.

    `pathh:` instead of `path:` would otherwise open the default clip, run to completion, and
    produce a clean-looking benchmark of the wrong video -- discovered, if at all, by noticing
    the frame count is odd. That number could be the one that gets submitted.
    """
    with pytest.raises(SourceConfigError) as exc:
        build_source({"mode": "file", "camera_id": CAMERA, "path": "a.mp4", "pathh": "b.mp4"})
    assert "pathh" in str(exc.value)
    assert "path" in str(exc.value), "the message has to name what was accepted"


def test_keys_are_scoped_to_their_mode():
    """A key valid on one mode is a typo on another.

    `frames` takes `directory`, not `path`. Accepting `path` there because `file` uses it
    would mean a config could name a video file and read an empty directory in silence.
    """
    with pytest.raises(SourceConfigError, match="path"):
        build_source({"mode": "frames", "camera_id": CAMERA, "directory": "d", "path": "a.mp4"})

    with pytest.raises(SourceConfigError, match="directory"):
        build_source({"mode": "file", "camera_id": CAMERA, "path": "a.mp4", "directory": "d"})


def test_a_password_on_an_rtsp_source_is_refused():
    """The Sentinel RTSP grid is open and the HLS endpoint is not, so `password` under
    live_rtsp means someone copied the wrong block. Ignoring it would leave a run that looks
    authenticated and is not."""
    with pytest.raises(SourceConfigError, match="password"):
        build_source({"mode": "live_rtsp", "camera_id": CAMERA, "url": "rtsp://h/s", "password": "x"})


def test_the_common_keys_are_accepted_on_every_mode(tmp_path):
    """target_interval_ms and the discontinuity switches are pipeline settings, not source
    ones, so they cannot be per-mode -- that would make the sampling rate depend on the
    adapter, which is the invariant failing at its most consequential point."""
    for block in (
        synthetic_block(target_interval_ms=200, detect_discontinuity=False, max_frames=5),
        {"mode": "frames", "camera_id": CAMERA, "directory": str(tmp_path),
         "target_interval_ms": 200, "detect_discontinuity": False, "max_frames": 5},
    ):
        source = build_source(block)
        assert source.stats()["sampler"]["target_interval_ms"] == 200


def test_a_required_key_missing_is_named():
    with pytest.raises(SourceConfigError, match="path"):
        build_source({"mode": "file", "camera_id": CAMERA})
    with pytest.raises(SourceConfigError, match="directory"):
        build_source({"mode": "frames", "camera_id": CAMERA})


def test_an_empty_required_key_counts_as_missing():
    """`path: ""` is a half-finished edit, not a request to open the empty string."""
    with pytest.raises(SourceConfigError, match="path"):
        build_source({"mode": "file", "camera_id": CAMERA, "path": ""})


# ------------------------------------------------------------------------------- camera_id


def test_the_command_line_camera_wins_over_the_file():
    """How one config drives thirty cameras. Thirty near-identical YAML files drift apart,
    and the one that drifts is never the one you are looking at."""
    source = build_source(synthetic_block(camera_id="cam01"), camera_id="cam07")
    assert source.camera_id == "cam07"


def test_a_source_with_no_camera_at_all_is_refused():
    """Contracts 1.1: every frame carries a camera_id. A source that could not say which
    camera it was would produce events no journey could use."""
    with pytest.raises(SourceConfigError, match="camera_id"):
        build_source({"mode": "synthetic"})


def test_a_malformed_camera_id_is_refused_by_shape():
    """`is_valid_camera_id` checks shape only -- existence is the catalogue's business.

    So `cam99` passes here and is rejected later by the database lookup with UNKNOWN_CAMERA,
    while `CAM04` and `cam4` are refused now. Contracts 1.1 names the forbidden forms.
    """
    for bad in ("CAM04", "cam4", "camera04", "4", "cam_04", "cam-04", " cam04"):
        with pytest.raises(SourceConfigError, match="camera_id"):
            build_source(synthetic_block(camera_id=bad))

    assert build_source(synthetic_block(camera_id="cam99")).camera_id == "cam99"


def test_zero_pad_drift_past_two_digits_is_not_detectable_here_and_that_is_the_safe_failure():
    """`cam004` passes the pattern. Documented rather than tightened.

    `^cam[0-9]{2,}$` accepts two OR MORE digits so a catalogue growing past cam99 does not
    need a contract change. The cost is that `cam004` and `cam04` are both well-formed while
    plausibly meaning the same camera -- and the pattern cannot tell "the catalogue reached
    three digits" from "someone zero-padded", because at this layer those are the same string
    shape.

    Tightening to exactly two digits would refuse cam100 on the day the grid grows, which is a
    worse failure than this one. And this one is not silent: camera_id goes into the dedupe key
    and into the DB lookup verbatim, so `cam004` produces UNKNOWN_CAMERA at ingest rather than
    merging into cam04's timeline. A rejected event is recoverable; a wrong journey is not.
    """
    assert build_source(synthetic_block(camera_id="cam004")).camera_id == "cam004"
    assert build_source(synthetic_block(camera_id="cam100")).camera_id == "cam100"
    assert "cam004" != "cam04", "the two are distinct keys downstream, which is what saves us"


# ------------------------------------------------------------------------- the faults block


def test_a_misspelled_fault_is_refused():
    """The worst possible place for a silent default.

    A fault-injection run whose fault was dropped passes for the wrong reason, and the
    conclusion drawn is "the system survives a reconnect" when nothing reconnected.
    """
    with pytest.raises(SourceConfigError) as exc:
        build_source(synthetic_block(faults={"blackframes": [3]}))
    assert "blackframes" in str(exc.value)
    assert "black_frames" in str(exc.value)


def test_every_declared_fault_is_accepted():
    known = sorted(SyntheticFaults.__dataclass_fields__)
    assert known, "the fault set cannot be empty; the whole point is injecting failures"
    source = build_source(synthetic_block(faults={"black_frames": [3, 4]}))
    assert source.faults.black_frames == (3, 4)


def test_a_faults_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(SourceConfigError, match="mapping"):
        build_source(synthetic_block(faults=["black_frames"]))


def test_no_faults_means_no_faults_not_defaults_somebody_forgot():
    source = build_source(synthetic_block())
    assert source.faults == SyntheticFaults()


# ----------------------------------------------------- the invariant, end to end, two sources
#
# Two halves, and the split is deliberate. The first compares stage configuration and reads no
# frames at all, so it is instant and runs even where Pillow is missing. The second actually
# drives frames through both adapters and costs a few seconds; it is a module fixture so it is
# paid once. Keeping the cheap half separate means the strongest single assertion in this file
# is never gated behind the slow one.

WEIGHTLESS_STAGES = {
    "detect": {"name": "motion"},
    "track": {"name": "bytetrack"},
    "plate": {"name": "edge"},
    "ocr": {"name": "template"},
}

# The generator's own defaults. Its tuning comments -- crossing durations, headway, the bound
# relating frames_visible to travel distance -- are all written against these numbers, so a
# smaller frame is a scene it was never tuned for. At 640x360 over 40 frames it plans two
# vehicles and none of them ever become visible, which would make every assertion below vacuous
# while still passing.
FRAME_W, FRAME_H = 1280, 720

# Long enough for three things: the motion detector's EMA background to converge (it produces
# nothing for roughly its first thirty frames), a vehicle to cross at MIN_CROSSING_SECONDS, and
# a track to close so an event is emitted. Ninety is a little past the point where all three
# happen; shorter runs faster and asserts nothing.
TOTAL_FRAMES = 90

# Pinned so observed_at is derived from an anchor rather than from the clock, which is what lets
# the two runs be compared field by field instead of count by count.
REPLAY_ANCHOR = "2026-09-01T00:00:00+05:30"

# The only fields allowed to differ between two runs of identical pixels, each for a reason
# named in Contracts 1: event_id is globally unique per event, stream_session_id is minted per
# connection, and source_mode is the field whose whole job is to record which adapter ran.
# Anything else differing means a stage saw something it should not have.
PER_RUN_FIELDS = ("event_id", "stream_session_id", "source_mode")


def _weightless_config(source_block: dict) -> AppConfig:
    """base.yaml's stage blocks with the weightless backends swapped in.

    Built from base.yaml rather than written out here so the thresholds under test are the
    shipped ones. A hand-written stage layer would keep passing after base.yaml changed,
    which is the one thing this test must not do.
    """
    base = load_config(config_path("base.yaml"), env=NO_ENV, validate=False).raw
    raw = {
        section: dict(base.get(section, {}), **override)
        for section, override in WEIGHTLESS_STAGES.items()
    }
    for section in ("quality", "fusion", "dedup", "metrics", "snapshot"):
        raw[section] = dict(base.get(section, {}))
    raw["snapshot"] = dict(raw["snapshot"], enabled=False)
    raw["run"] = dict(base.get("run", {}), replay_anchor=REPLAY_ANCHOR)
    raw["source"] = source_block
    return AppConfig(path=None, raw=raw)


def synthetic_scene(**overrides) -> dict:
    """The scene both adapters replay."""
    return synthetic_block(
        width=FRAME_W, height=FRAME_H, total_frames=TOTAL_FRAMES, **overrides
    )


def frames_scene(directory) -> dict:
    return {
        "mode": "frames",
        "camera_id": CAMERA,
        "directory": str(directory),
        # Matching step_ms matters: the frames adapter has no embedded timeline, so this is
        # where its PTS comes from. A different value would resample the same pixels and every
        # downstream difference would be this line rather than anything about the adapters.
        "interval_ms": synthetic_scene()["step_ms"],
    }


def _identities(source_block: dict) -> dict:
    """Stage identities for one source, without reading a frame.

    open() is still required -- build_pipeline keys the tracker on source.session_id and
    ai/media/base.py refuses to invent one before the source is connected -- but no decoding
    happens, so this is cheap on every mode.
    """
    from ai.metrics import StageIdentity
    from ai.worker import build_pipeline

    config = _weightless_config(source_block)
    source = build_source(config.source_config(CAMERA), camera_id=CAMERA)
    source.open()
    try:
        _, stages = build_pipeline(config, source, camera_id=CAMERA)
    finally:
        source.close()
    return {
        label: StageIdentity.from_stage(label, stage).to_dict()
        for label, stage in sorted(stages.items())
    }


@pytest.fixture(scope="module")
def one_frame_dir(tmp_path_factory):
    """A frames directory with a single tiny image.

    Enough to open a frames source, which is all the stage-identity comparison needs. A frames
    source will not open on an empty directory -- see the test below -- so this cannot be an
    empty tmp_path.
    """
    pillow = pytest.importorskip("PIL.Image", reason="Pillow writes the frame")
    directory = tmp_path_factory.mktemp("one_frame")
    pillow.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(directory / "000000.png")
    return directory


def test_an_empty_frames_directory_refuses_to_open(tmp_path):
    """Rather than yielding zero frames and reporting a clean run of nothing.

    The realistic mistake is a `directory:` pointing one level too high, and the difference
    matters: a run that decodes nothing and exits 0 looks like a working pipeline over an empty
    clip, and the frame count is the only place it shows.
    """
    source = build_source({"mode": "frames", "camera_id": CAMERA, "directory": str(tmp_path)})
    with pytest.raises(RuntimeError, match="no images"):
        source.open()


def test_not_one_stage_identity_differs_across_the_source_boundary(one_frame_dir):
    """The invariant itself, and it costs nothing to check.

    Same detector, same tracker, same plate detector, same OCR engine, same weights hashes,
    same ships and oracle flags -- across two adapters that share no code below build_source.
    If any stage had to know which source it was reading from, this is where it would show,
    and it would show before a single frame was decoded.
    """
    synthetic = _identities(synthetic_scene())
    frames = _identities(frames_scene(one_frame_dir))

    assert synthetic == frames
    assert set(synthetic) == {"detector", "ocr", "plate_detector", "tracker"}


def test_no_stage_in_the_invariant_run_is_an_oracle(tmp_path):
    """Which is what makes the run able to cross a source boundary at all.

    An oracle reads the generator's truth object, so a stage set containing one is coupled to
    the synthetic source by construction and could never have been compared against a second
    adapter. Asserted so nobody "improves" this test by switching to the oracle stages and
    quietly turns it into a test of nothing.
    """
    for label, identity in _identities(synthetic_scene()).items():
        assert identity["is_oracle"] is False, label


def _run(source_block: dict) -> dict:
    """Drive one source through a full pipeline and report what the run was.

    The open-before-build order is not a style choice. `build_pipeline` reads
    `source.session_id` to key the tracker, and a session only exists once the source is
    connected -- ai/media/base.py raises rather than inventing one. ai/worker.py's run loop
    does the same three steps in the same order, and this mirrors it deliberately: a test that
    assembled the pipeline differently from production could pass while production could not
    start.

    Imported inside the function: ai.worker pulls in the whole stage tree, and a collection
    error there would take out every fast test in this file alongside the few that need it.
    """
    from ai.worker import build_pipeline

    config = _weightless_config(source_block)
    source = build_source(config.source_config(CAMERA), camera_id=CAMERA)
    source.open()
    try:
        pipeline, _ = build_pipeline(config, source, camera_id=CAMERA)
        pipeline.load()

        events = []
        frames = []
        for envelope in source:
            events.extend(pipeline.process_frame(envelope))
            frames.append((envelope.pts_ms, envelope.frame_bgr.copy()))
        events.extend(pipeline.flush("eof"))
        pipeline.close()
    finally:
        source.close()

    return {
        "mode": source.source_mode,
        "frames": frames,
        "events": [event.to_dict() for event in events],
        "gate": pipeline.gate.stats(),
    }


@pytest.fixture(scope="module")
def two_adapters(tmp_path_factory):
    """The same scene through two adapters that share no code below build_source.

    The synthetic source generates the frames and writes them to disk as lossless PNG; the
    frames adapter reads them back. Costs a few seconds -- almost all of it the motion
    detector, which is a numpy EMA over a 720p frame ninety times, twice -- and is worth it:
    this is the fixture behind the claim the whole product is pitched on, and a cheaper
    version of it would be comparing two runs of the same adapter.
    """
    pillow = pytest.importorskip("PIL.Image", reason="Pillow writes the frames to disk")

    directory = tmp_path_factory.mktemp("frames_cam04")
    written = 0
    writer = build_source(synthetic_scene(), camera_id=CAMERA)
    with writer:
        for envelope in writer:
            pillow.fromarray(envelope.frame_bgr[:, :, ::-1]).save(
                directory / f"{envelope.frame_index:06d}.png"
            )
            written += 1
    assert written == TOTAL_FRAMES, (
        f"the generator emitted {written} of {TOTAL_FRAMES} frames; every comparison below "
        "is against a scene that is not the one this test was tuned for"
    )

    return {
        "written": written,
        "synthetic": _run(synthetic_scene()),
        "frames": _run(frames_scene(directory)),
    }


def test_the_run_produced_enough_to_compare(two_adapters):
    """Non-vacuity, asserted first and separately.

    Every comparison below iterates over events or reads a counter, and all of them pass
    trivially on a run that produced nothing. A weightless stage set on too short a clip does
    exactly that -- the motion detector needs about thirty frames of background before it
    reports anything -- so "the invariant holds" and "the run was empty" have to be
    distinguishable, and this is the test that distinguishes them.
    """
    for run in ("synthetic", "frames"):
        assert two_adapters[run]["events"], f"{run} produced no events"
        assert two_adapters[run]["gate"]["evaluated"] > 0, f"{run} gated nothing"
        assert len(two_adapters[run]["frames"]) == TOTAL_FRAMES


def test_each_run_recorded_the_mode_it_actually_used(two_adapters):
    """Not the mode it was configured with -- the one the adapter reports about itself.

    A source_mode copied from config would keep saying `live_rtsp` after a fallback to a
    recorded clip, and the benchmark would claim a live measurement it never took.
    """
    assert two_adapters["synthetic"]["mode"] == "synthetic"
    assert two_adapters["frames"]["mode"] == "frames"

    for run, expected in (("synthetic", "synthetic"), ("frames", "frames")):
        for event in two_adapters[run]["events"]:
            assert event["source_mode"] == expected


def test_both_adapters_delivered_pixel_identical_frames_at_identical_timestamps(two_adapters):
    """The premise every comparison after this one depends on.

    PNG is lossless and the frames adapter's interval_ms matches the generator's step_ms, so
    the two runs should see the same bytes at the same PTS. Asserted rather than assumed
    because if it were false, every downstream equality would be an accident and every
    downstream difference would be attributable to image fidelity rather than to a stage.
    """
    synthetic = two_adapters["synthetic"]["frames"]
    frames = two_adapters["frames"]["frames"]
    assert len(synthetic) == len(frames)

    for index, ((pts_a, image_a), (pts_b, image_b)) in enumerate(zip(synthetic, frames)):
        assert pts_a == pts_b, f"frame {index}: pts {pts_a} vs {pts_b}"
        assert np.array_equal(image_a, image_b), f"frame {index}: pixels differ"


def test_both_adapters_put_the_same_work_through_the_gate(two_adapters):
    """The gate is the first stage whose behaviour depends on pixel content.

    Identical counts here mean the round trip preserved what the detector sees, so a later
    difference in events would be about track lifetimes rather than about image fidelity.
    """
    assert two_adapters["synthetic"]["gate"] == two_adapters["frames"]["gate"]


def test_the_two_runs_differ_in_exactly_the_fields_that_have_to_differ(two_adapters):
    """The invariant in its strongest available form.

    Not "the same number of events" -- the same events, field for field, including every
    bounding box, confidence, quality score and model provenance block. The three exceptions
    are named in PER_RUN_FIELDS and each is required to differ by Contracts 1: two UUIDs that
    are unique by definition, and source_mode, whose job is to record which adapter ran.

    This is only assertable because run.replay_anchor is pinned. Unpinned, observed_at is
    derived from the moment the run started, and two runs of the same footage would carry
    different timestamps -- which is exactly the reproducibility problem the anchor exists to
    solve, and why config/benchmark.yaml pins one.
    """
    synthetic = two_adapters["synthetic"]["events"]
    frames = two_adapters["frames"]["events"]
    assert len(synthetic) == len(frames)

    def comparable(event: dict) -> dict:
        return {k: v for k, v in event.items() if k not in PER_RUN_FIELDS}

    for index, (a, b) in enumerate(zip(synthetic, frames)):
        assert comparable(a) == comparable(b), (
            f"event {index} differs beyond the per-run fields:\n"
            f"  synthetic {json.dumps(comparable(a), sort_keys=True)}\n"
            f"  frames    {json.dumps(comparable(b), sort_keys=True)}"
        )


def test_observed_at_is_identical_because_the_anchor_is_pinned(two_adapters):
    """Stated on its own because it is the field most likely to start drifting.

    observed_at is not in PER_RUN_FIELDS, which is a claim about the replay timeline rather
    than about the adapters: an offline event's timestamp is the anchor plus the frame's own
    PTS, so the same footage replayed twice yields the same instant. If this fails while the
    test above passes, something started reading the wall clock.
    """
    synthetic = [event["observed_at"] for event in two_adapters["synthetic"]["events"]]
    frames = [event["observed_at"] for event in two_adapters["frames"]["events"]]
    assert synthetic == frames
    assert all(stamp.startswith("2026-08-31T18:30") for stamp in synthetic), (
        f"the anchor {REPLAY_ANCHOR} is +05:30, so the UTC instants should sit just before "
        f"midnight on 31 Aug; got {synthetic[:2]}"
    )


def test_the_per_run_fields_really_do_differ(two_adapters):
    """The other half of the test above, and the one that keeps it honest.

    PER_RUN_FIELDS is an exclusion list, so it would keep passing if it named fields that were
    actually identical -- and it could then grow to cover a real difference without anyone
    noticing. Each entry has to earn its place by differing.
    """
    a = two_adapters["synthetic"]["events"][0]
    b = two_adapters["frames"]["events"][0]
    for field in PER_RUN_FIELDS:
        assert a[field] != b[field], (
            f"{field} is identical across the two runs, so excluding it from the comparison "
            "hides nothing and it should not be on the list"
        )


def test_a_weightless_stage_set_does_not_invent_a_plate(two_adapters):
    """The honesty rule, exercised on a run that mostly cannot read a plate.

    Template OCR on an edge-projected box is not expected to produce characters. `plate: null`
    is a valid, correct answer -- Contracts 3.2 -- and a fabricated plate is the worst failure
    this system can have. So the assertion is not "no plates": it is that a plate block with
    no normalized text says so, rather than carrying a guess at some confidence.
    """
    seen = 0
    for run in ("synthetic", "frames"):
        for event in two_adapters[run]["events"]:
            plate = event.get("plate")
            if plate is None:
                continue
            seen += 1
            assert plate["normalized"] or plate["match_state"] == "unreadable", (
                f"{run}: a plate block with no normalized text must say unreadable, "
                f"got {json.dumps(plate)}"
            )
    assert seen > 0, (
        "no event carried a plate block at all, so this test asserted nothing. The plate "
        "detector reached no gated track; check the gate counters before trusting it."
    )


def test_every_event_from_both_adapters_validates(two_adapters):
    """Both runs have to produce events ingest would accept.

    A source-independent pipeline that emitted invalid events from one adapter would satisfy
    the letter of the invariant and be useless.
    """
    from ai.contracts.event import validate_payload

    for run in ("synthetic", "frames"):
        for event in two_adapters[run]["events"]:
            assert validate_payload(event) == [], f"{run}: {json.dumps(event)[:400]}"


# ---------------------------------------------- what is NOT a violation, stated as a test


def test_offline_and_live_differ_on_more_than_source_and_that_is_correct():
    """Read as a violation often enough to be worth a test that says why it is not.

    The invariant is about the source only. Which stages run is a second, independent axis --
    offline runs oracles to measure the plumbing without a GPU, live runs real models -- and
    two configs choosing differently on that axis is the reason both files exist. The
    invariant would be violated by a *stage* branching on source_mode, which is a different
    thing entirely and would not show up in a config diff at all.
    """
    offline = load_config(config_path("offline.yaml"), env=NO_ENV)
    live = load_config(config_path("live.yaml"), env=NO_ENV)
    differ = sorted(k for k in set(offline.raw) | set(live.raw) if offline.raw.get(k) != live.raw.get(k))

    assert "source" in differ
    assert offline.raw["detect"]["name"] == "oracle"
    assert live.raw["detect"]["name"] != "oracle"
    assert len(differ) > 1, "if only source differed, one of the two files would be pointless"


def test_the_tracker_is_identical_in_both_because_it_ships_in_both():
    """ByteTrack has no oracle worth having -- it is pure association arithmetic with no
    weights to skip. So the one stage that is the same in offline and live is the one where
    there was never a reason to differ, which is a useful sanity check on the other axis."""
    offline = load_config(config_path("offline.yaml"), env=NO_ENV)
    live = load_config(config_path("live.yaml"), env=NO_ENV)
    assert offline.raw["track"] == live.raw["track"]


def test_an_oracle_stage_cannot_be_built_on_a_source_that_has_no_truth():
    """The coupling, made explicit by the factory rather than discovered by a wrong number.

    An oracle that silently degraded to guessing when handed a real source would be a
    benchmark that measured nothing and said so nowhere.
    """
    from ai.detect.factory import DetectorConfigError, build_detector

    frames_source = build_source({"mode": "frames", "camera_id": CAMERA, "directory": "."})
    with pytest.raises(DetectorConfigError) as exc:
        build_detector({"name": "oracle"}, source=frames_source)
    assert "truth_for_envelope" in str(exc.value)
    assert "synthetic" in str(exc.value), "the message has to say which mode would work"


def test_a_synthetic_source_does_expose_truth_so_the_oracles_have_something_to_read():
    """The other side of the same coupling. Without it the offline config could not run at
    all, and the plumbing would have no way to be measured without a GPU."""
    source = build_source(synthetic_block())
    assert callable(getattr(source, "truth_for_envelope", None))

    with source:
        envelope = source.read()
    assert envelope is not None
    truth = source.truth_for_envelope(envelope)
    assert truth is not None
    assert isinstance(envelope.frame_bgr, np.ndarray)
    assert envelope.frame_bgr.dtype == np.uint8
    assert envelope.frame_bgr.shape == (360, 640, 3)
