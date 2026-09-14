"""What happens when the stream breaks. Owner's manual sections 4.5 and 4.8.

Two modules, one subject. A Sentinel stream fails in two ways and each has its own detector:

    the picture changes    ai/media/discontinuity.py -- histogram correlation between frames
    the socket dies        ai/media/backoff.py       -- exponential backoff with jitter

They are tested together because the thing that matters is downstream of both: a
stream_session_id boundary. ByteTrack restarts track numbering at 1 per session, so a break
that does not produce a boundary hands the tracker a frame where every vehicle teleported, and
it recovers by reassigning ids -- merging vehicles from before the break with vehicles after
it. The journey built from that data shows a vehicle that was never there, and no downstream
stage can tell, because the data is well-formed.

Both signals are deliberately weak on their own. A camera re-aimed by hand keeps perfect PTS,
and a stream that loses thirty seconds of time keeps a near-identical picture. Neither is
sufficient; that is the design, and the tests below assert the gaps as well as the coverage.
"""

import random
from typing import Optional

import numpy as np
import pytest

from ai.media.backoff import (
    BACKOFF_BASE_MS,
    BACKOFF_MAX_MS,
    HEALTHY_RESET_SECONDS,
    _MAX_SHIFT,
    ReconnectPolicy,
)
from ai.media.base import BaseMediaSource, SessionChange
from ai.media.discontinuity import (
    DISCONTINUITY_CORRELATION_THRESHOLD,
    DOWNSCALE_LONG_EDGE,
    HISTOGRAM_BINS,
    DiscontinuityDetector,
    _correlate,
    _to_gray_small,
)
from ai.media.factory import SourceConfigError, build_source

CAMERA = "cam04"


# --------------------------------------------------------------------------- frames

def road_frames(seed: int, count: int) -> list[np.ndarray]:
    """Frames from the synthetic generator at its tuned resolution.

    Real generated road scenes rather than hand-drawn rectangles, because the numbers this
    file pins are correlations between histograms of actual traffic. A synthetic block of
    colour would correlate at whatever the block happens to give and would pin nothing.
    """
    source = build_source(
        {
            "mode": "synthetic",
            "camera_id": CAMERA,
            "seed": seed,
            "width": 1280,
            "height": 720,
            "step_ms": 100,
            "total_frames": count,
            "vehicles_per_100_frames": 6,
            # Off. This function supplies frames TO a detector under test; leaving it on would
            # let the source's own detector rotate the session mid-collection and the frames
            # would come back from two different scenes.
            "detect_discontinuity": False,
        },
        camera_id=CAMERA,
    )
    source.open()
    try:
        return [envelope.frame_bgr.copy() for envelope in source]
    finally:
        source.close()


@pytest.fixture(scope="module")
def road() -> list[np.ndarray]:
    return road_frames(1337, 60)


def solid(value: int, size: tuple[int, int] = (720, 1280)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


# ================================================================= the picture changes

def test_the_first_frame_of_a_session_cannot_be_a_discontinuity(road):
    """None, not 1.0 and not 0.0.

    There is nothing to compare against, so any number here would be an invention -- and 0.0
    would be read as "completely different" and rotate the session on the first frame of every
    session, forever.
    """
    detector = DiscontinuityDetector()
    cut, correlation = detector.check(road[0])
    assert cut is False
    assert correlation is None
    assert detector.comparisons == 0, "the first frame is not a comparison"
    assert detector.last_correlation is None


def test_an_unchanged_scene_does_not_fire_and_the_margin_is_large(road):
    """The measurement the threshold most depends on.

    Spurious firing is the expensive failure: it rotates the session on a healthy stream,
    which discards tracker state on every camera at once. So what matters is not that ordinary
    traffic stays above 0.70 but by how much.
    """
    detector = DiscontinuityDetector()
    detector.check(road[0])
    correlations = [detector.check(frame)[1] for frame in road[1:]]

    assert detector.detections == 0, "ordinary traffic must never read as a scene cut"
    assert detector.comparisons == len(road) - 1
    worst = min(correlations)
    assert worst > DISCONTINUITY_CORRELATION_THRESHOLD
    assert worst > 0.98, (
        f"worst consecutive correlation on a healthy scene was {worst:.4f}. The threshold is "
        f"{DISCONTINUITY_CORRELATION_THRESHOLD}, so the margin has collapsed and a lighting "
        "change is now enough to rotate the session."
    )


def test_a_whole_scene_of_traffic_turning_over_still_does_not_fire(road):
    """Not adjacent frames -- the two ends of a minute, with entirely different vehicles.

    This is the strongest available statement that the threshold is safe: even the largest
    within-scene change the generator produces stays far above it.
    """
    detector = DiscontinuityDetector()
    detector.check(road[0])
    cut, correlation = detector.check(road[-1])
    assert cut is False
    assert correlation > 0.99, f"frame 0 vs frame {len(road) - 1} correlated {correlation:.4f}"


def test_unrelated_content_fires(road):
    detector = DiscontinuityDetector()
    detector.check(road[0])
    noise = np.random.default_rng(0).integers(0, 255, road[0].shape, dtype=np.uint8)
    cut, correlation = detector.check(noise)
    assert cut is True
    assert correlation < DISCONTINUITY_CORRELATION_THRESHOLD
    assert detector.detections == 1


def test_a_similar_looking_different_scene_is_only_caught_about_half_the_time():
    """The detector's real weakness, asserted rather than described.

    Ten pairs of different generated road scenes measured 0.30 to 0.93, and five fell below
    0.70. So this signal does not reliably distinguish "re-aimed at a comparable junction" from
    "re-aimed at something unrelated" -- it catches roughly half.

    That is the safe direction for the error to run. A missed re-aim costs the track ids in one
    scene; a spurious rotation costs every id on every camera. But it is a real limit on what
    the pipeline can claim, and it must not be discovered from a demo. If a future change makes
    every pair fire, that is not an improvement -- it means the threshold has climbed into the
    range ordinary traffic reaches, and the previous test is the one that will catch it.
    """
    seeds = (1337, 99, 7, 2026, 555)
    firsts = {seed: road_frames(seed, 1)[0] for seed in seeds}

    verdicts = []
    for index, left in enumerate(seeds):
        for right in seeds[index + 1:]:
            detector = DiscontinuityDetector()
            detector.check(firsts[left])
            verdicts.append(detector.check(firsts[right]))

    fired = [cut for cut, _ in verdicts]
    correlations = [correlation for _, correlation in verdicts]
    assert any(fired), "no pair of distinct scenes fired at all -- the detector is inert"
    assert not all(fired), (
        "every pair of distinct road scenes fired. Either the generator now produces wildly "
        "different scenes per seed, or the threshold has risen -- and if it has risen, "
        "test_an_unchanged_scene_does_not_fire_and_the_margin_is_large is the one to trust."
    )
    assert min(correlations) > 0.2 and max(correlations) < 0.99, (
        f"cross-scene correlations {[round(c, 3) for c in correlations]} left the measured "
        "0.30-0.93 band, so the numbers recorded in ai/media/discontinuity.py are now stale"
    )


def test_the_same_clip_restarting_is_invisible_here_and_that_is_why_pts_exists(road):
    """The Sentinel-specific trap, and the reason this is one of two signals.

    A Sentinel stream loops. When the loop restarts the SAME footage, the histogram is
    identical -- so this detector reports a perfectly healthy scene at the exact moment every
    vehicle in frame was replaced. Asserted, because it is the failure most likely to be
    mistaken for "discontinuity detection is handled".

    What catches it is ai/media/pts.py: the timeline jumps backwards, PtsValidator returns
    NEW_SESSION, and ai/media/base.py rotates. Removing either signal leaves a hole, and this
    is the half of the hole that is invisible from here.
    """
    detector = DiscontinuityDetector()
    detector.check(road[-1])
    cut, correlation = detector.check(road[0])

    assert cut is False, "a loop restart is not visible to a histogram, by construction"
    assert correlation > 0.99
    assert detector.detections == 0


def test_a_run_of_black_frames_stays_quiet(road):
    """A camera that has gone dark should report one cut, not one per frame.

    Note this works through ordinary correlation, not through the zero-variance fallback in
    _correlate: a black frame's histogram is the highest-variance one available, since every
    pixel lands in bin 0. The next test pins that, because an earlier comment in the module
    claimed the opposite and someone reading it would 'fix' this case in the wrong place.
    """
    detector = DiscontinuityDetector()
    detector.check(road[0])

    first_cut, _ = detector.check(solid(0))
    assert first_cut is True, "going dark is a genuine scene change"

    for _ in range(5):
        cut, correlation = detector.check(solid(0))
        assert cut is False
        assert correlation == pytest.approx(1.0)
    assert detector.detections == 1, "one cut for going dark, not one per dark frame"


def test_a_black_frame_is_not_a_constant_histogram():
    """The zero-variance branch of _correlate is unreachable from real footage.

    Measured: a fully black frame's histogram has variance 0.0154, and a clean 0-255 gradient
    reaches 1.7e-05. Zero requires all 64 bins exactly equal, which is a deliberately
    constructed array rather than a picture. The branch is kept because the arithmetic has to
    be total, and pinned here so its docstring stays honest about being unreachable.
    """
    detector = DiscontinuityDetector()
    assert float(np.var(detector._histogram(solid(0)))) > 0.0
    assert float(np.var(detector._histogram(solid(255)))) > 0.0

    gradient = np.dstack(
        [np.tile(np.repeat(np.arange(256, dtype=np.uint8), 5)[:1280], (720, 1))] * 3
    )
    assert float(np.var(detector._histogram(gradient))) > 0.0

    flat = np.full(HISTOGRAM_BINS, 1.0 / HISTOGRAM_BINS)
    assert float(np.var(flat)) == 0.0
    assert _correlate(flat, flat) == 1.0
    assert _correlate(flat, np.eye(HISTOGRAM_BINS)[0]) == 0.0


def test_black_against_blown_out_fires():
    """Both frames are uniform, and the answer is still a cut.

    -0.02 rather than the 0.0 the fallback would give, because neither histogram is constant.
    Either way the camera has stopped showing the scene, so the boundary is correct.
    """
    detector = DiscontinuityDetector()
    detector.check(solid(0))
    cut, correlation = detector.check(solid(255))
    assert cut is True
    assert correlation < 0.05


def test_reset_makes_the_next_frame_first_of_session_again(road):
    """Called on every session rotation. Without it, the first frame of a new session is
    compared against the last frame of the old one -- which is exactly the pair that just
    triggered the rotation, so every rotation would immediately trigger another."""
    detector = DiscontinuityDetector()
    detector.check(road[0])
    detector.check(road[1])
    assert detector.last_correlation is not None

    detector.reset()
    assert detector.last_correlation is None
    cut, correlation = detector.check(solid(0))
    assert (cut, correlation) == (False, None), "post-reset frame has nothing to compare to"

    assert detector.comparisons == 1, "reset clears history, not the lifetime counters"


def test_grayscale_input_is_accepted_and_a_fourth_channel_is_refused(road):
    """HxW for a mono camera, HxWx3 for the rest, and a hard refusal for HxWx4.

    A 4-channel array reaching here means someone passed RGBA, and the luma weights would
    silently read the alpha channel as a colour plane. Refusing names the caller's bug; a
    plausible-looking correlation would not.
    """
    detector = DiscontinuityDetector()
    gray = np.zeros((720, 1280), dtype=np.uint8)
    assert detector.check(gray) == (False, None)

    with pytest.raises(ValueError, match="expected HxW or HxWx3"):
        _to_gray_small(np.zeros((8, 8, 4), dtype=np.uint8), DOWNSCALE_LONG_EDGE)


def test_downscaling_strides_to_the_configured_long_edge(road):
    """Striding, not area averaging. Both frames get the same treatment, which is all a
    histogram comparison needs, and it costs a fraction of the time at 10 fps x 30 cameras."""
    small = _to_gray_small(road[0], DOWNSCALE_LONG_EDGE)
    assert max(small.shape) <= DOWNSCALE_LONG_EDGE
    assert small.shape == (90, 160), "1280x720 at step 8"

    # A long_edge larger than the frame must not upscale.
    assert _to_gray_small(np.zeros((4, 4), dtype=np.uint8), 1000).shape == (4, 4)


def test_stats_reports_what_a_run_log_needs(road):
    detector = DiscontinuityDetector()
    assert detector.stats() == {
        "comparisons": 0,
        "detections": 0,
        "last_correlation": None,
        "threshold": DISCONTINUITY_CORRELATION_THRESHOLD,
    }

    detector.check(road[0])
    detector.check(road[1])
    stats = detector.stats()
    assert stats["comparisons"] == 1
    assert stats["detections"] == 0
    assert isinstance(stats["last_correlation"], float)
    assert stats["last_correlation"] == round(detector.last_correlation, 4)


# ----------------------------------------------------- the threshold is a correlation

def test_a_millisecond_value_in_the_correlation_field_is_refused():
    """The regression test for a bug that shipped in two configs.

    config/offline.yaml and config/live.yaml both carried `discontinuity_threshold: 2000` -- a
    PTS-jump figure in milliseconds, in a field that is a Pearson correlation bounded by 1.0.
    It was harmless only by accident: ai/media/factory.py accepted the key and then dropped it
    on the floor, so the detector ran at its 0.70 default and nobody noticed for either reason.

    Had the key been plumbed without this check, every emitted frame would have rotated the
    session, every track would have been one frame long, and both configs would have produced
    zero events -- CI included.
    """
    with pytest.raises(ValueError, match="out of range"):
        DiscontinuityDetector(threshold=2000)


@pytest.mark.parametrize("bad", [1.0, 1.5, 2000, -1.0, -3.0, float("inf")])
def test_a_threshold_that_disables_the_detector_while_looking_configured_is_refused(bad):
    """Both ends, for the same reason.

    At or above 1.0 nothing correlates highly enough and every frame is a cut. At or below
    -1.0 nothing ever is. Both leave a config that reads as if discontinuity detection were
    tuned when it is off, and one of them is off in the direction that destroys every track.
    `detect_discontinuity: false` is how to say off.
    """
    with pytest.raises(ValueError):
        DiscontinuityDetector(threshold=bad)


@pytest.mark.parametrize("bad", ["0.7", None, True, False, [0.7]])
def test_a_non_number_threshold_is_refused(bad):
    """bool included deliberately. `True` is an int in Python and 1 > -1, so a bare isinstance
    check on (int, float) would accept `discontinuity_threshold: true` from YAML -- which is
    exactly the value someone types when they meant `detect_discontinuity: true`."""
    with pytest.raises(ValueError, match="must be a number"):
        DiscontinuityDetector(threshold=bad)


def test_a_threshold_of_zero_survives_the_trip_through_a_config():
    """0.0 is legal, and it used to be silently replaced.

    ai/media/base.py built the detector with `if discontinuity_threshold else {}`, so a
    configured 0.0 fell back to 0.70 -- a config that said one thing while the detector did
    another. Nothing sensible sets 0.0, which is exactly why it went unnoticed; the reason to
    fix it is that the same truthiness bug hides any falsy value someone does mean.

    Asserted through build_source rather than by constructing the detector directly, because
    the bug was in the wiring and not in the detector. A direct construction passes either way
    -- which is how the first version of this test managed to assert nothing at all.
    """
    source = build_source(
        {
            "mode": "synthetic",
            "camera_id": CAMERA,
            "total_frames": 4,
            "discontinuity_threshold": 0.0,
        }
    )
    assert source.stats()["discontinuity"]["threshold"] == 0.0, (
        "a configured 0.0 was replaced by the default, so ai/media/base.py is testing the "
        "threshold for truthiness again"
    )

    detector = DiscontinuityDetector(threshold=0.0)
    assert detector.threshold == 0.0
    detector.check(solid(0))
    cut, correlation = detector.check(solid(255))
    assert correlation < 0.05
    assert cut is True, "a negative correlation is still below zero"


def test_correlation_never_exceeds_one_which_is_why_a_larger_threshold_fires_always(road):
    """The arithmetic behind the range check, rather than a restatement of it.

    The rule is `correlation < threshold`, and Pearson correlation is bounded by 1.0. So any
    threshold above 1.0 makes the comparison unconditionally true -- not "too sensitive", but
    a session rotation on every single emitted frame.
    """
    detector = DiscontinuityDetector()
    detector.check(road[0])
    for frame in road[1:20]:
        _, correlation = detector.check(frame)
        assert -1.0 <= correlation <= 1.0

    for pair in ((solid(0), solid(255)), (road[0], road[1]), (solid(7), solid(7))):
        detector = DiscontinuityDetector()
        detector.check(pair[0])
        _, correlation = detector.check(pair[1])
        assert correlation <= 1.0


# ------------------------------------------------------- the threshold reaches the detector

@pytest.mark.parametrize("mode_block", [
    {"mode": "synthetic", "total_frames": 4},
    {"mode": "file", "path": "does-not-need-to-exist.mp4"},
    {"mode": "frames", "directory": "does-not-need-to-exist"},
    {"mode": "live_rtsp"},
    {"mode": "live_hls"},
])
def test_a_configured_threshold_actually_reaches_the_detector(mode_block):
    """The other half of the shipped bug, and the reason it went unnoticed for so long.

    ai/media/factory.py listed `discontinuity_threshold` in _COMMON_KEYS -- so the strict
    unknown-key check accepted it -- and then forwarded only three keys to the constructor. A
    correctly spelled key was accepted and discarded. That is precisely the failure the
    unknown-key strictness exists to prevent, arriving through the one door it does not watch.

    Asserted on all five modes, because the previous version worked for none of them and a
    per-mode fix would leave the same hole in four.
    """
    source = build_source({"camera_id": CAMERA, "discontinuity_threshold": 0.42, **mode_block})
    assert source.stats()["discontinuity"]["threshold"] == 0.42


def test_an_unset_threshold_uses_the_measured_default():
    source = build_source({"mode": "synthetic", "camera_id": CAMERA, "total_frames": 4})
    assert (
        source.stats()["discontinuity"]["threshold"] == DISCONTINUITY_CORRELATION_THRESHOLD
    )


def test_a_bad_threshold_from_a_config_is_a_config_error_not_a_value_error():
    """SourceConfigError, so scripts/validate_config.py labels it correctly.

    The validator catches SourceConfigError as "source: <message>" and everything else as
    "could not construct (...)", a label its own comment reserves for non-config failures like
    a missing clip. A malformed threshold is a malformed config, and describing it as a
    construction failure sends the reader looking for the wrong thing.
    """
    with pytest.raises(SourceConfigError, match="out of range"):
        build_source(
            {"mode": "synthetic", "camera_id": CAMERA, "discontinuity_threshold": 2000}
        )


def test_neither_shipped_config_sets_a_threshold():
    """Both carried 2000 until it was caught, and neither file's comment ever justified it.

    A config that restates a measured default silently stops tracking it, and this is the
    parameter where that went wrong once already. If a camera genuinely needs its own
    threshold, the finding belongs in the file that sets it -- with the measurement.
    """
    from ai.config import load_config
    from conftest import REPO_ROOT

    for name in ("base.yaml", "offline.yaml", "live.yaml", "benchmark.yaml"):
        raw = load_config(
            str(REPO_ROOT / "config" / name), env={}, validate=False
        ).raw
        source = raw.get("source", {})
        # base.yaml deliberately has no source block at all -- it would make one mode the
        # implicit default. Every other config must have a real one, checked so this test
        # cannot pass because load_config stopped returning source blocks.
        if name != "base.yaml":
            assert source, f"config/{name} has no source block to inspect"
        assert "discontinuity_threshold" not in source, (
            f"config/{name} sets discontinuity_threshold. If that is deliberate, the file has "
            "to say what was measured -- the last value in there was a millisecond figure."
        )


# ==================================================================== the socket dies

class FixedRandom(random.Random):
    """Pins the jitter draw so a delay is arithmetic rather than a range."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def random(self) -> float:  # noqa: D102
        return self.value


class FakeClock:
    """Injected monotonic. The healthy-reset rule is a 30 second rule and a test that waited
    for it would be a 30 second test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def policy(*, jitter: Optional[float] = None, seed: int = 0, **kwargs) -> ReconnectPolicy:
    slept: list[float] = []
    rng = FixedRandom(jitter) if jitter is not None else random.Random(seed)
    made = ReconnectPolicy(rng=rng, sleep=slept.append, **kwargs)
    made.slept = slept  # type: ignore[attr-defined]
    return made


def test_the_delay_doubles_from_the_base_and_then_stops_at_the_cap():
    """The formula from the owner's manual section 4.5, with jitter pinned to 1.0x."""
    backoff = policy(jitter=0.5)
    expected = [500, 1000, 2000, 4000, 8000, 16000, 30000, 30000]
    for attempt, want in enumerate(expected):
        backoff.attempt = attempt
        backoff._pending_delay = None
        assert backoff.next_delay_ms() == want, f"attempt {attempt}"


def test_jitter_spans_half_to_one_and_a_half_of_the_delay():
    for value, want in ((0.0, 250), (0.5, 500), (1.0, 750)):
        assert policy(jitter=value).next_delay_ms() == want


def test_the_cap_is_applied_before_jitter_so_a_wait_can_exceed_it():
    """BACKOFF_MAX_MS is not a maximum wait, and the name invites the opposite reading.

    The manual's formula caps first and jitters second, so a 30 s cap permits a 45 s wait.
    Pinned rather than corrected: the formula is specified, and someone who moves the cap
    inside the jitter to make the constant honest has changed the retry distribution of every
    worker against a document they did not read.
    """
    backoff = policy(jitter=1.0)
    backoff.attempt = 20
    assert backoff.next_delay_ms() == int(BACKOFF_MAX_MS * 1.5) == 45000

    floor = policy(jitter=0.0)
    floor.attempt = 20
    assert floor.next_delay_ms() == BACKOFF_MAX_MS // 2


def test_a_camera_that_is_simply_gone_does_not_build_a_huge_integer():
    """_MAX_SHIFT exists so 2 ** attempt cannot grow without bound on a camera that has been
    dead for a week. Without it, attempt would reach five figures and the shift would allocate
    a number with thousands of digits to compute a value the cap discards."""
    backoff = policy(jitter=1.0)
    backoff.attempt = 10 ** 6
    assert backoff.next_delay_ms() == int(BACKOFF_MAX_MS * 1.5)

    at_shift = policy(jitter=1.0)
    at_shift.attempt = _MAX_SHIFT
    assert at_shift.next_delay_ms() == int(BACKOFF_MAX_MS * 1.5)


def test_next_delay_is_the_delay_that_will_actually_be_waited():
    """A field named next_delay_ms has to be the delay that is coming.

    It draws from the rng, so before this was cached, two calls gave two answers: stats()
    logged 672 ms and wait() then slept 628. The log is the only record of whether a reconnect
    storm was backing off or hot-looping, and it cannot be checked against observed timing if
    the number in it was never used.
    """
    backoff = policy(seed=0)
    first = backoff.next_delay_ms()
    assert backoff.next_delay_ms() == first
    assert backoff.stats()["next_delay_ms"] == first

    waited = backoff.wait()
    assert waited == first
    assert backoff.slept == [first / 1000.0]

    assert backoff.next_delay_ms() != first or backoff.attempt == 1


def test_wait_sleeps_the_delay_it_reports_and_advances_both_counters():
    backoff = policy(jitter=0.5)
    assert backoff.wait() == 500
    assert backoff.wait() == 1000
    assert backoff.slept == [0.5, 1.0]
    assert backoff.attempt == 2
    assert backoff.total_reconnects == 2


def test_one_good_frame_does_not_reset_the_backoff():
    """The rule the module docstring calls out, and the one worth having a test for.

    A camera that hands back a single frame and dies again has not recovered. Resetting the
    attempt counter on that frame turns exponential backoff into a hot loop -- and thirty
    workers hot-looping against the organisers' grid is a self-inflicted outage on borrowed
    infrastructure.
    """
    clock = FakeClock()
    backoff = policy(jitter=0.5, monotonic=clock)
    backoff.wait()
    backoff.wait()
    assert backoff.attempt == 2

    backoff.note_healthy_read()
    assert backoff.attempt == 2, "the first healthy frame only starts the clock"

    clock.advance(HEALTHY_RESET_SECONDS - 0.01)
    backoff.note_healthy_read()
    assert backoff.attempt == 2, "still inside the healthy window"

    clock.advance(0.02)
    backoff.note_healthy_read()
    assert backoff.attempt == 0, "sustained healthy reading resets"
    assert backoff.total_reconnects == 2, "the lifetime count is not undone by recovering"


def test_a_failure_restarts_the_healthy_window_rather_than_extending_it():
    """Two nearly-complete healthy stretches must not add up to one complete one.

    A camera that works for 29 seconds, drops, and works for another 29 has not been healthy
    for 30 seconds -- it has failed twice. Summing the stretches would reset the counter on a
    stream that is visibly flapping, which is when backoff matters most.
    """
    clock = FakeClock()
    backoff = policy(jitter=0.5, monotonic=clock)
    backoff.wait()

    backoff.note_healthy_read()
    clock.advance(HEALTHY_RESET_SECONDS - 1.0)
    backoff.note_healthy_read()
    assert backoff.attempt == 1

    backoff.note_failure()
    backoff.note_healthy_read()
    clock.advance(HEALTHY_RESET_SECONDS - 1.0)
    backoff.note_healthy_read()
    assert backoff.attempt == 1, "58 seconds of healthy reading in two pieces is not 30 in one"

    clock.advance(2.0)
    backoff.note_healthy_read()
    assert backoff.attempt == 0


def test_recovering_discards_the_delay_drawn_during_the_outage():
    """A jitter sample belongs to the attempt it was drawn for.

    Carrying it across a recovery would hand the next outage a delay chosen during the
    previous one, which quietly correlates a worker's retries with its own history -- the
    opposite of what jitter is for.
    """
    clock = FakeClock()
    backoff = policy(seed=7, monotonic=clock)
    backoff.wait()
    stale = backoff.next_delay_ms()

    backoff.note_healthy_read()
    clock.advance(HEALTHY_RESET_SECONDS + 1)
    backoff.note_healthy_read()
    assert backoff.attempt == 0
    assert backoff._pending_delay is None, "the outage's sample must not survive recovery"

    fresh = backoff.next_delay_ms()
    assert fresh <= int(BACKOFF_BASE_MS * 1.5)
    assert stale != fresh or stale <= int(BACKOFF_BASE_MS * 1.5)


def test_thirty_workers_do_not_retry_in_unison():
    """What the jitter is actually for, stated as the thing that would go wrong without it.

    Thirty workers lose a shared upstream at the same instant. With a deterministic delay they
    all return at the same instant too, and a brief outage becomes a synchronised hammering of
    the Sentinel grid -- from inside the hackathon, on infrastructure the organisers lent us.
    """
    delays = [
        ReconnectPolicy(rng=random.Random(worker), sleep=lambda _: None).next_delay_ms()
        for worker in range(30)
    ]
    assert len(set(delays)) >= 25, f"only {len(set(delays))} distinct delays across 30 workers"
    assert min(delays) >= BACKOFF_BASE_MS // 2
    assert max(delays) <= int(BACKOFF_BASE_MS * 1.5)


def test_stats_reports_what_a_reconnect_log_needs():
    backoff = policy(jitter=0.5)
    assert backoff.stats() == {"attempt": 0, "total_reconnects": 0, "next_delay_ms": 500}
    backoff.wait()
    assert backoff.stats() == {"attempt": 1, "total_reconnects": 1, "next_delay_ms": 1000}


# ============================================== both signals, through a real source

class ScriptedSource(BaseMediaSource):
    """A source whose reads are a list, so a break can be placed exactly.

    Written here rather than in conftest because it exists to make one thing observable: what
    ai/media/base.py does between a failed read and the next envelope. A real adapter cannot
    be asked to fail on frame 3 without a network.

    Each script entry is (frame, pts_ms), the string "fail" to raise on read, or None for end
    of input.
    """

    source_mode = "file"

    def __init__(self, camera_id: str, script: list, *, live: bool = False, **kwargs) -> None:
        super().__init__(camera_id, **kwargs)
        self.script = list(script)
        self._live = live
        self.opens = 0
        self.closes = 0
        self._cursor = 0

    @property
    def supports_reconnect(self) -> bool:
        return self._live

    def _open_capture(self) -> None:
        self.opens += 1

    def _close_capture(self) -> None:
        self.closes += 1

    def _read_raw(self):
        if self._cursor >= len(self.script):
            return None
        item = self.script[self._cursor]
        self._cursor += 1
        if item == "fail":
            raise OSError("scripted read failure")
        if item is None:
            return None
        return item


def scripted(script: list, **kwargs) -> ScriptedSource:
    source = ScriptedSource(CAMERA, script, **kwargs)
    changes: list[SessionChange] = []
    source.add_session_listener(changes.append)
    source.changes = changes  # type: ignore[attr-defined]
    return source


def test_a_scene_cut_rotates_the_session_and_restarts_the_frame_index(road):
    """The whole point of detecting a cut: a session boundary the pipeline can act on.

    frame_index restarting at 0 matters as much as the new id. The pipeline keys evidence by
    (camera, session, track) and reports position within a session, so a boundary that kept
    counting would leave the two scenes sharing one coordinate space.
    """
    script = [(road[0], 0), (road[1], 100), (solid(0), 200), (solid(0), 300)]
    source = scripted(script)
    source.open()
    envelopes = list(source)
    source.close()

    sessions = [envelope.stream_session_id for envelope in envelopes]
    assert len(set(sessions)) == 2, f"expected one boundary, got sessions {sessions}"
    assert sessions[0] == sessions[1] and sessions[2] == sessions[3]

    indices = [envelope.frame_index for envelope in envelopes]
    assert indices == [0, 1, 0, 1]

    assert len(source.changes) == 2, "one for the open, one for the cut"
    cut = source.changes[-1]
    assert cut.reason == "discontinuity"
    assert "histogram correlation" in (cut.detail or "")
    assert cut.at_pts_ms == 200
    assert cut.at_frame_index == 0
    assert source.stats()["sessions_started"] == 2


def test_the_detector_is_reseeded_after_a_cut_so_a_second_cut_is_still_caught(road):
    """A rotation resets the detector's history. Without re-seeding it with the frame that
    caused the rotation, the next frame would have nothing to compare against -- and a second
    cut two frames later would be the one that went unnoticed, which on a looping stream is
    the common case rather than the exotic one."""
    script = [
        (road[0], 0),
        (solid(0), 100),      # cut one: road -> black
        (solid(0), 200),
        (road[1], 300),       # cut two: black -> road
        (road[2], 400),
    ]
    source = scripted(script)
    source.open()
    list(source)
    source.close()

    assert source.stats()["discontinuity"]["detections"] == 2
    reasons = [change.reason for change in source.changes]
    assert reasons == ["open", "discontinuity", "discontinuity"], reasons


def test_detection_off_means_the_detector_is_never_consulted(road):
    """`detect_discontinuity: false` has to be genuinely off, not "off but still counting".

    A frames-mode fixture of unrelated hard cases would otherwise rotate the session on every
    image, and the counters would report a stream in constant collapse.
    """
    script = [(road[0], 0), (solid(0), 100), (road[1], 200)]
    source = scripted(script, detect_discontinuity=False)
    source.open()
    envelopes = list(source)
    source.close()

    assert len({envelope.stream_session_id for envelope in envelopes}) == 1
    assert source.stats()["discontinuity"] == {
        "comparisons": 0,
        "detections": 0,
        "last_correlation": None,
        "threshold": DISCONTINUITY_CORRELATION_THRESHOLD,
    }


def test_a_read_failure_on_a_live_source_backs_off_reopens_and_mints_a_session(road):
    """The reconnect path end to end, with the ordering that matters.

    note_failure before the wait, so the healthy window is void before the delay is chosen.
    Close before reopen, because a half-open capture holds the socket. And a new session on the
    way back, because the stream moved on while we were away: everything the tracker knew is
    about vehicles that have left.
    """
    script = [(road[0], 0), "fail", (road[1], 100), (road[2], 200)]
    slept: list[float] = []
    backoff = ReconnectPolicy(rng=FixedRandom(0.5), sleep=slept.append)
    # max_frames is what stops this test, and the reason is the behaviour under test: a live
    # source treats running out of input as a failure and reconnects, so it never ends on its
    # own. Iterating one to exhaustion is an infinite loop by design.
    source = scripted(script, live=True, reconnect=backoff, max_frames=3)
    source.open()
    envelopes = list(source)
    source.close()

    assert slept == [0.5], "one failure, one backoff wait at the base delay"
    assert source.opens == 2, "reopened once"
    assert backoff.total_reconnects == 1

    sessions = [envelope.stream_session_id for envelope in envelopes]
    assert len(set(sessions)) == 2, "the frames after a reconnect are a new session"

    reconnect = source.changes[-1]
    assert reconnect.reason == "reconnect"
    assert "scripted read failure" in (reconnect.detail or "")
    assert "reconnected after 1 attempt(s)" in (reconnect.detail or "")
    assert source.stats()["reconnect"]["total_reconnects"] == 1


def test_a_read_failure_on_a_file_is_raised_rather_than_retried(road):
    """A file that fails mid-read is corrupt, not unreachable.

    Retrying it would loop forever on a benchmark run, and the caller needs the decoder error
    rather than a source that quietly returns fewer frames than the clip contains.
    """
    source = scripted([(road[0], 0), "fail", (road[1], 100)], live=False)
    source.open()
    assert source.read() is not None
    with pytest.raises(OSError, match="scripted read failure"):
        source.read()


def test_end_of_input_ends_a_file_and_reconnects_a_live_stream(road):
    """The same event, read two ways, and the reason _handle_exhausted is a hook.

    A file running out has finished. A live stream returning nothing has failed. Collapsing
    the two would either loop forever on a clip or treat a dead camera as a completed run.
    """
    finite = scripted([(road[0], 0), None, (road[1], 100)], live=False)
    finite.open()
    assert len(list(finite)) == 1, "reading stops at end of input"
    assert finite.stats()["sessions_started"] == 1

    live = scripted([(road[0], 0), None, (road[1], 100)], live=True, max_frames=2,
                    reconnect=ReconnectPolicy(rng=FixedRandom(0.5), sleep=lambda _: None))
    live.open()
    assert len(list(live)) == 2, "a live stream reconnects past the gap"
    assert live.stats()["sessions_started"] == 2


def test_a_healthy_read_is_noted_so_a_recovered_stream_stops_backing_off(road):
    """The wiring, not the rule: ai/media/base.py has to call note_healthy_read on every
    frame it accepts, or the policy never learns the stream came back and the next failure
    starts from wherever the last one left off."""
    clock = FakeClock()
    backoff = ReconnectPolicy(
        rng=FixedRandom(0.5), sleep=lambda _: None, monotonic=clock
    )
    source = scripted([(road[index], index * 100) for index in range(5)], reconnect=backoff)
    source.open()

    source.read()
    assert backoff._healthy_since == 0.0, "the first accepted frame starts the healthy window"
    clock.advance(HEALTHY_RESET_SECONDS + 1)
    backoff.attempt = 3
    source.read()
    assert backoff.attempt == 0, "sustained healthy reads reset through the source"


def test_a_reconnect_that_cannot_reopen_raises_rather_than_reporting_end_of_stream(road):
    """Giving up has to be loud.

    _attempt_reconnect returns False for an exhausted source, which the worker treats like end
    of file -- correct for a stream that ended, wrong for a camera that is refusing
    connections. A failure to reopen means the URL, the credentials or the camera is wrong, and
    a silent clean exit would report a successful run of zero events.
    """
    class DeadOnReopen(ScriptedSource):
        def _open_capture(self) -> None:
            super()._open_capture()
            if self.opens > 1:
                raise ConnectionRefusedError("camera refused the connection")

    source = DeadOnReopen(
        CAMERA,
        [(road[0], 0), "fail"],
        live=True,
        reconnect=ReconnectPolicy(rng=FixedRandom(0.5), sleep=lambda _: None),
    )
    source.open()
    source.read()
    with pytest.raises(RuntimeError, match="could not reconnect to cam04"):
        source.read()
    assert source._is_open is False, "a source that failed to reopen must not look open"
