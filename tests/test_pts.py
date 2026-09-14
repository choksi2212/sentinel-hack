"""The source timeline: PTS validation, sampling, the depth-1 buffer, replay pacing.

Everything temporal in this project rests on one decision recorded in Contracts 2.2 and
manual 4.7: time comes from the stream's own presentation timestamps, never from
CAP_PROP_FPS and never from the wall clock. Two consequences are worth stating before the
tests, because both are the kind of thing that produces measurements rather than errors.

Sampling on the source timeline is what makes an offline benchmark comparable with a live
run. Sample on arrival time instead and a 5x replay gets a fifth of the inferences per video
second, so every accuracy number measured during accelerated replay silently describes a
different sampling rate than the one that will run in production.

A session whose PTS had to be synthesized cannot substantiate a latency claim, and the flag
that records this is sticky on purpose. A validator that cleared it when the real clock came
back would leave a session that spent half its life on a made-up clock reporting timings as
though they were measured.
"""

import inspect
import threading
import time

import pytest

from ai.media.pacing import ReplayPacer
from ai.media.pts import (
    MAX_FORWARD_JUMP_MS,
    STALL_FRAME_THRESHOLD,
    SYNTHETIC_STEP_MS,
    TARGET_INTERVAL_MS,
    UNAVAILABLE_FRAME_THRESHOLD,
    PtsAction,
    PtsValidator,
    PtsVerdict,
)
from ai.media.sampler import TARGET_INTERVAL_MS as SAMPLER_TARGET_INTERVAL_MS
from ai.media.sampler import LatestFrameBuffer, PtsSampler


class FakeClock:
    """A monotonic clock that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ----------------------------------------------------------------------- locked constants


def test_the_sampling_interval_is_the_contract_value_in_both_modules():
    """Two modules declare it, and they have to agree.

    `pts.py` states it for documentation and `sampler.py` uses it. If they drifted apart,
    the module that reads as the spec would no longer be the one making the decision.
    """
    assert TARGET_INTERVAL_MS == 100
    assert SAMPLER_TARGET_INTERVAL_MS == TARGET_INTERVAL_MS


def test_the_pts_thresholds_are_the_manual_values():
    assert MAX_FORWARD_JUMP_MS == 5_000
    assert UNAVAILABLE_FRAME_THRESHOLD == 5
    assert STALL_FRAME_THRESHOLD == 10
    assert SYNTHETIC_STEP_MS == 40


# --------------------------------------------------------------------- PTS: the happy path


def test_the_first_frame_is_always_accepted():
    """There is nothing to compare it against, so no verdict but OK is defensible."""
    verdict = PtsValidator().observe(1234)
    assert verdict.action is PtsAction.OK
    assert verdict.pts_ms == 1234
    assert verdict.ok is True
    assert verdict.pts_unreliable is False


def test_a_monotonic_stream_is_accepted_frame_after_frame():
    validator = PtsValidator()
    for pts in range(0, 2001, 40):
        assert validator.observe(pts).ok, pts
    assert validator.counts["accepted"] == 51
    assert validator.last_pts_ms == 2000
    assert validator.pts_unreliable is False


def test_a_genuine_frame_zero_is_not_mistaken_for_a_missing_clock():
    """A file's first frame really is at PTS 0, and OpenCV also returns 0.0 for streams
    that carry no POS_MSEC at all. The two are only distinguishable by position: a real 0
    can only be the first frame, which is why the special case is guarded on last_pts_ms.
    """
    verdict = PtsValidator().observe(0)
    assert verdict.action is PtsAction.OK
    assert verdict.pts_ms == 0

    # Second frame reporting 0 is a missing clock, not a rewind to the start.
    validator = PtsValidator()
    validator.observe(0)
    assert validator.observe(0).action is PtsAction.SKIP


def test_a_fractional_pts_is_truncated_to_whole_milliseconds():
    """pts_ms is an integer on the wire, and the event schema constrains it.

    Truncation rather than rounding, matching int(): the value is a position in a stream,
    and consistently biasing it half a millisecond early is preferable to a value that
    depends on which side of .5 the decoder landed.
    """
    validator = PtsValidator()
    verdict = validator.observe(1234.87)
    assert verdict.pts_ms == 1234
    assert isinstance(verdict.pts_ms, int)


# -------------------------------------------------------------------- PTS: discontinuities


def test_pts_going_backwards_mints_a_new_session():
    """Manual 4.7. A Sentinel loop restart looks exactly like this.

    Without the rotation, ByteTrack associates a vehicle leaving at the loop end with an
    unrelated one entering at the loop start, and the journey built from that data shows a
    vehicle that was never there.
    """
    validator = PtsValidator()
    validator.observe(10_000)
    verdict = validator.observe(500)
    assert verdict.action is PtsAction.NEW_SESSION
    assert "10000" in verdict.reason and "500" in verdict.reason
    assert validator.counts["backwards"] == 1


def test_a_large_forward_jump_mints_a_new_session():
    """Not a vehicle moving. A gap that big is a different scene."""
    validator = PtsValidator()
    validator.observe(1_000)
    verdict = validator.observe(1_000 + MAX_FORWARD_JUMP_MS + 1)
    assert verdict.action is PtsAction.NEW_SESSION
    assert "5001 ms" in verdict.reason
    assert validator.counts["forward_jump"] == 1


def test_the_forward_jump_boundary_is_inclusive_of_the_limit():
    """Exactly 5,000 ms is tolerated; 5,001 is not.

    Stated because five seconds of dropped frames on a congested link is a real event on
    these feeds, and rotating the session on it would throw away every live track for no
    reason.
    """
    validator = PtsValidator()
    validator.observe(1_000)
    assert validator.observe(1_000 + MAX_FORWARD_JUMP_MS).ok

    other = PtsValidator()
    other.observe(1_000)
    assert other.observe(1_000 + MAX_FORWARD_JUMP_MS + 1).action is PtsAction.NEW_SESSION


def test_a_new_session_verdict_does_not_advance_the_clock_itself():
    """The validator reports; the source decides and rebuilds. Advancing last_pts_ms here
    would leave the old validator half-migrated into the new session's timeline if the
    source chose to handle the verdict differently."""
    validator = PtsValidator()
    validator.observe(10_000)
    validator.observe(500)
    assert validator.last_pts_ms == 10_000


# ---------------------------------------------------------------------- PTS: stalled decoder


def test_a_duplicate_timestamp_is_skipped_rather_than_emitted_twice():
    """Two frames with one timestamp are indistinguishable in time.

    Emitting both would put two sightings at the same instant on the same track, and the
    fusion evidence count would rise on a frame that carried no new information.
    """
    validator = PtsValidator()
    validator.observe(400)
    verdict = validator.observe(400)
    assert verdict.action is PtsAction.SKIP
    assert "duplicate pts" in verdict.reason


def test_ten_identical_timestamps_force_a_reconnect():
    """A stream still handing back buffers with a frozen clock is a transport problem, and
    no amount of skipping frames fixes it. Manual 4.7 calls for a reconnect."""
    validator = PtsValidator()
    validator.observe(400)
    for i in range(1, STALL_FRAME_THRESHOLD):
        assert validator.observe(400).action is PtsAction.SKIP, i
    verdict = validator.observe(400)
    assert verdict.action is PtsAction.FORCE_RECONNECT
    assert "stalled" in verdict.reason
    assert validator.counts["stalled"] == 1


def test_one_advancing_frame_clears_the_stall_counter():
    """A slow stream is not a stalled one. Without the reset, nine skips spread over a
    minute of a struggling link would eventually force a pointless reconnect."""
    validator = PtsValidator()
    validator.observe(400)
    for _ in range(STALL_FRAME_THRESHOLD - 1):
        validator.observe(400)
    assert validator.observe(440).ok
    for _ in range(STALL_FRAME_THRESHOLD - 1):
        assert validator.observe(440).action is PtsAction.SKIP
    assert validator.counts["stalled"] == 0


# ------------------------------------------------------------------- PTS: unavailable clock


def test_a_few_missing_readings_are_skipped_not_patched_over():
    """Below the threshold, skipping is the honest answer.

    The alternative -- stamping the frame with the previous frame's timestamp -- is a
    fabricated measurement that no downstream consumer could detect.
    """
    validator = PtsValidator()
    validator.observe(1_000)
    for i in range(1, UNAVAILABLE_FRAME_THRESHOLD):
        verdict = validator.observe(None)
        assert verdict.action is PtsAction.SKIP, i
        assert validator.pts_unreliable is False
    assert validator.last_pts_ms == 1_000, "a skipped frame must not move the clock"


def test_none_and_zero_and_negative_all_mean_no_usable_value():
    for raw in (None, 0, -1, -500.0):
        validator = PtsValidator()
        validator.observe(1_000)
        assert validator.observe(raw).action is PtsAction.SKIP, raw


def test_a_sustained_missing_clock_switches_to_a_synthetic_one():
    validator = PtsValidator()
    validator.observe(1_000)
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD - 1):
        validator.observe(None)

    verdict = validator.observe(None)
    assert verdict.action is PtsAction.OK
    assert verdict.pts_unreliable is True
    assert verdict.pts_ms == 1_000 + SYNTHETIC_STEP_MS
    assert "synthetic" in verdict.reason
    assert validator.counts["synthesized"] == 1


def test_the_synthetic_clock_is_monotonic():
    """It has one job. A non-monotonic replacement clock would trip the backwards rule and
    rotate the session on every frame."""
    validator = PtsValidator()
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD + 20):
        validator.observe(None)
    seen = [validator.observe(None).pts_ms for _ in range(10)]
    assert seen == sorted(seen)
    assert len(set(seen)) == 10


def test_the_unreliable_flag_is_sticky_once_set():
    """The point of the whole module.

    A session that ran a synthetic clock for any part of its life cannot substantiate a
    latency claim, even after the real clock recovers -- the recorded timings for the
    synthetic stretch are still fiction. Clearing the flag on recovery would hand a clean
    session to whoever reads the log.
    """
    validator = PtsValidator()
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD):
        validator.observe(None)
    assert validator.pts_unreliable is True
    assert validator.last_pts_ms == SYNTHETIC_STEP_MS

    recovered = validator.observe(SYNTHETIC_STEP_MS + 200)
    assert recovered.ok is True
    assert recovered.pts_unreliable is True, "the flag must survive recovery"
    assert validator.stats()["pts_unreliable"] is True


def test_a_rejection_verdict_reports_the_reliability_too():
    """Every verdict, not only the accepting ones.

    ai/media/base.py reads the flag off the validator, so nothing is misled today. But a
    NEW_SESSION verdict claiming pts_unreliable=False on a session running a synthetic clock
    is a field that lies, and the next reader to reach for the convenient one would record a
    clean session that was never clean.
    """
    validator = PtsValidator()
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD):
        validator.observe(None)

    rotated = validator.observe(SYNTHETIC_STEP_MS + MAX_FORWARD_JUMP_MS + 1)
    assert rotated.action is PtsAction.NEW_SESSION
    assert rotated.pts_unreliable is True

    skipped = validator.observe(None)
    assert skipped.pts_unreliable is True


def test_a_usable_reading_clears_the_unavailable_run():
    """So four bad readings, one good one, and four more bad ones do not add up to a
    synthetic clock."""
    validator = PtsValidator()
    validator.observe(1_000)
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD - 1):
        validator.observe(None)
    validator.observe(1_040)
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD - 1):
        assert validator.observe(None).action is PtsAction.SKIP
    assert validator.pts_unreliable is False


def test_a_real_clock_resuming_behind_the_synthetic_one_rotates_the_session():
    """Documented rather than special-cased, because the rotation is the safe outcome.

    The synthetic clock advances 40 ms per frame from wherever the real one stopped, so a
    stream that comes back reporting a lower timestamp looks like it went backwards. That
    mints a new session -- which is right: a stream that lost its clock and then contradicted
    the replacement is exactly the case where carrying tracks across the gap would associate
    two different vehicles.
    """
    validator = PtsValidator()
    validator.observe(1_000)
    for _ in range(UNAVAILABLE_FRAME_THRESHOLD):
        validator.observe(None)
    assert validator.last_pts_ms == 1_040

    assert validator.observe(1_020).action is PtsAction.NEW_SESSION


def test_stats_report_every_counter_the_session_log_needs():
    validator = PtsValidator()
    validator.observe(1_000)
    validator.observe(500)
    assert set(validator.stats()) == {
        "last_pts_ms",
        "pts_unreliable",
        "accepted",
        "unavailable",
        "synthesized",
        "backwards",
        "forward_jump",
        "stalled",
    }


def test_thresholds_are_injectable_so_a_test_can_reach_the_slow_paths():
    validator = PtsValidator(max_forward_jump_ms=100, unavailable_threshold=2, stall_threshold=2)
    validator.observe(1_000)
    assert validator.observe(1_200).action is PtsAction.NEW_SESSION

    other = PtsValidator(unavailable_threshold=1, synthetic_step_ms=7)
    other.observe(500)
    assert other.observe(None).pts_ms == 507


def test_a_verdict_is_immutable():
    """It crosses a thread boundary in the live sources. A mutable verdict is a race."""
    verdict = PtsVerdict(PtsAction.OK, 100)
    with pytest.raises(Exception):
        verdict.pts_ms = 200  # type: ignore[misc]


# ------------------------------------------------------------------------------- sampling


def test_the_first_frame_of_a_session_always_emits():
    assert PtsSampler().should_emit(0) is True
    assert PtsSampler().should_emit(999_999) is True


def test_the_interval_boundary_is_inclusive():
    """Exactly 100 ms since the last emission emits; 99 does not.

    The contract says `>=`, and at 25 fps the difference is the difference between emitting
    every fourth frame and every fifth -- a 25% change in inference load.
    """
    sampler = PtsSampler()
    sampler.note_emitted(1_000)
    assert sampler.should_emit(1_099) is False
    assert sampler.should_emit(1_100) is True


def test_should_emit_is_a_question_and_note_emitted_is_the_answer():
    """Asking twice must give the same answer.

    The caller asks, then does work, then records. If should_emit had a side effect the
    frame would be double-counted whenever the caller checked before logging.
    """
    sampler = PtsSampler()
    sampler.note_emitted(1_000)
    assert sampler.should_emit(1_100) is True
    assert sampler.should_emit(1_100) is True
    assert sampler.emitted == 1


def test_a_25fps_stream_emits_at_8_33_hz_not_10():
    """The honest number, and it belongs in a test rather than in someone's slide.

    Contracts 2.2 says "~10 inferences/sec/camera". With 40 ms frames and a `>= 100 ms`
    rule the achievable rate is every third frame -- 120 ms apart, 8.33 Hz. There is no
    subset of a 25 fps stream that is 100 ms apart, so the tilde in the contract is doing
    real work and "10 fps per camera" is not a claim this pipeline can make.
    """
    sampler = PtsSampler()
    emitted = []
    for pts in range(0, 1_000, 40):  # 25 fps, one second
        if sampler.should_emit(pts):
            sampler.note_emitted(pts)
            emitted.append(pts)

    assert emitted == [0, 120, 240, 360, 480, 600, 720, 840, 960]
    assert len(emitted) == 9
    gaps = {b - a for a, b in zip(emitted, emitted[1:])}
    assert gaps == {120}


def test_a_30fps_stream_also_undershoots_ten():
    """Same arithmetic, different feed. 33.33 ms frames give every fourth frame, 132 ms."""
    sampler = PtsSampler()
    emitted = []
    for pts in range(0, 1_000, 33):
        if sampler.should_emit(pts):
            sampler.note_emitted(pts)
            emitted.append(pts)

    assert emitted == [0, 132, 264, 396, 528, 660, 792, 924]
    assert {b - a for a, b in zip(emitted, emitted[1:])} == {132}


def test_sampling_reads_only_the_source_timeline():
    """Structural, and the reason offline and live numbers are comparable at all.

    `should_emit` takes a PTS and nothing else -- no clock, no default time source. If a
    wallclock fallback were ever added here, an accelerated replay would sample at a
    different rate than production and every benchmark taken with it would describe a
    pipeline that does not exist.
    """
    params = list(inspect.signature(PtsSampler.should_emit).parameters)
    assert params == ["self", "pts_ms"]


def test_a_reset_clears_the_gate_but_keeps_the_run_totals():
    """A new session restarts the timeline, so the first frame of it must emit.

    The counters are run totals and deliberately survive: they are the denominator of the
    emit rate for the whole worker, and zeroing them on every reconnect would make a flaky
    camera look like a clean one.
    """
    sampler = PtsSampler()
    sampler.note_emitted(5_000)
    sampler.note_skipped()
    sampler.reset()
    assert sampler.should_emit(0) is True
    assert sampler.emitted == 1
    assert sampler.skipped == 1


def test_a_zero_interval_emits_everything_and_a_negative_one_is_refused():
    """Zero is a legitimate benchmark setting -- score every decoded frame. Negative is
    always a typo, and it would emit every frame while looking like it was throttling."""
    every = PtsSampler(target_interval_ms=0)
    every.note_emitted(1_000)
    assert every.should_emit(1_000) is True

    with pytest.raises(ValueError, match="target_interval_ms"):
        PtsSampler(target_interval_ms=-1)


def test_sampler_stats_are_the_denominator_of_frames_seen():
    """`decoded` here is what the worker reports as frames_seen, so the balance matters."""
    sampler = PtsSampler()
    assert sampler.stats()["emit_rate"] is None, "no frames, no rate"

    for pts in range(0, 500, 40):
        if sampler.should_emit(pts):
            sampler.note_emitted(pts)
        else:
            sampler.note_skipped()

    stats = sampler.stats()
    assert stats["decoded"] == stats["emitted"] + stats["skipped"] == 13
    assert stats["emit_rate"] == round(stats["emitted"] / stats["decoded"], 4)
    assert stats["target_interval_ms"] == TARGET_INTERVAL_MS


# -------------------------------------------------------------------- the depth-1 buffer


def test_the_buffer_hands_back_what_was_put_in():
    buffer = LatestFrameBuffer()
    buffer.put("frame-1")
    assert buffer.get(timeout=0.1) == "frame-1"
    assert buffer.stats() == {"accepted": 1, "consumed": 1, "dropped": 0, "drop_rate": 0.0}


def test_the_latest_frame_wins_and_the_old_one_is_counted():
    """Freshness beats completeness, and the drop is counted rather than silent.

    The count is how you tell a flaky camera from a GPU that cannot keep up, and it belongs
    in every performance claim made about this pipeline.
    """
    buffer = LatestFrameBuffer()
    buffer.put("old")
    buffer.put("new")
    assert buffer.get(timeout=0.1) == "new"
    assert buffer.dropped == 1
    assert buffer.accepted == 2
    assert buffer.consumed == 1


def test_depth_is_one_no_matter_how_far_behind_the_consumer_is():
    """An unbounded queue turns a 200 ms deficit into 2 s of lag per second of runtime.

    Five minutes in, the "live" view is ten minutes stale; shortly after that the process is
    OOM-killed. Holding exactly one frame is what makes lag bounded by design rather than
    by hoping the GPU keeps up.
    """
    buffer = LatestFrameBuffer()
    for i in range(1_000):
        buffer.put(i)
    assert buffer.get(timeout=0.1) == 999
    assert buffer.dropped == 999
    assert buffer.get(timeout=0.01) is None, "one frame held, not a thousand"


def test_an_empty_buffer_times_out_rather_than_blocking_forever():
    buffer = LatestFrameBuffer()
    started = time.monotonic()
    assert buffer.get(timeout=0.05) is None
    assert time.monotonic() - started < 2.0


def test_clear_discards_without_counting_a_consumption():
    """Used on session rotation. A cleared frame belongs to the old timeline and must not
    appear in the new session's consumed count."""
    buffer = LatestFrameBuffer()
    buffer.put("stale")
    buffer.clear()
    assert buffer.get(timeout=0.01) is None
    assert buffer.consumed == 0
    assert buffer.accepted == 1


def test_wake_releases_a_blocked_consumer_for_shutdown():
    """Without it, a consumer blocked on an infinite get() never notices the shutdown flag
    and the worker hangs on exit -- which on demo day looks like a crash."""
    buffer = LatestFrameBuffer()
    result: list[object] = []

    def consume() -> None:
        result.append(buffer.get(timeout=5.0))

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.05)
    buffer.wake()
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "wake() did not release the blocked get()"
    assert result == [None]
    assert buffer.consumed == 0


def test_drop_rate_is_none_before_any_frame_arrives():
    """Not 0.0. A buffer that has seen nothing has no drop rate, and reporting 0% reads as
    a healthy stream in exactly the summary someone skims."""
    assert LatestFrameBuffer().drop_rate is None


def test_every_accepted_frame_is_consumed_dropped_or_still_held():
    """The balance invariant, exercised across a real thread boundary.

    A frame counted in none of the three columns is a frame nobody can account for, and a
    missing lock shows up here and almost nowhere else.
    """
    buffer = LatestFrameBuffer()
    stop = threading.Event()

    def produce() -> None:
        for i in range(2_000):
            buffer.put(i)
        stop.set()

    thread = threading.Thread(target=produce)
    thread.start()
    while not stop.is_set():
        buffer.get(timeout=0.01)
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    while buffer.get(timeout=0) is not None:
        pass

    assert buffer.accepted == 2_000
    assert buffer.accepted == buffer.consumed + buffer.dropped


# ----------------------------------------------------------------------------- replay pacing


def test_pacing_is_off_by_default():
    """Because pacing a throughput measurement makes it a report of the pacer's settings.

    config/benchmark.yaml leaves it unset for exactly that reason, and a default of 1.0
    would quietly cap every measured fps at real time.
    """
    pacer = ReplayPacer()
    assert pacer.enabled is False
    assert pacer.wait_for(0) == 0.0
    assert pacer.wait_for(10_000) == 0.0
    assert pacer.slept_seconds == 0.0


def test_a_nonpositive_speed_is_refused():
    """speed=0 would wait forever on the second frame. None is how you disable it."""
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="speed"):
            ReplayPacer(bad)


def test_the_first_frame_anchors_the_timeline_and_does_not_wait():
    clock = FakeClock()
    pacer = ReplayPacer(1.0, sleep=clock.sleep, monotonic=clock.monotonic)
    assert pacer.wait_for(5_000) == 0.0
    assert clock.slept == []


def test_real_time_pacing_waits_the_source_interval():
    clock = FakeClock()
    pacer = ReplayPacer(1.0, sleep=clock.sleep, monotonic=clock.monotonic)
    pacer.wait_for(0)
    assert pacer.wait_for(1_000) == pytest.approx(1.0)
    assert clock.slept == [pytest.approx(1.0)]


def test_a_speed_factor_divides_the_wait():
    """5x replay waits a fifth as long. The clip's timeline is unchanged; only the sleep is."""
    clock = FakeClock()
    pacer = ReplayPacer(5.0, sleep=clock.sleep, monotonic=clock.monotonic)
    pacer.wait_for(0)
    assert pacer.wait_for(1_000) == pytest.approx(0.2)


def test_a_pacer_already_behind_gets_out_of_the_way():
    """It never sleeps to catch up.

    If inference takes longer than the pacing interval, sleeping anyway would compound the
    lag every frame -- the pacer would be the reason the demo fell behind.
    """
    clock = FakeClock()
    pacer = ReplayPacer(1.0, sleep=clock.sleep, monotonic=clock.monotonic)
    pacer.wait_for(0)
    clock.advance(3.0)  # the pipeline spent three seconds on one frame
    assert pacer.wait_for(1_000) == 0.0
    assert clock.slept == []
    assert pacer.behind_count == 1


def test_pacing_never_alters_the_timestamps_it_paces():
    """A paced run and an unpaced run of the same clip stamp identical pts values.

    Structural rather than emergent -- `wait_for` returns seconds slept, so it has no channel
    to rewrite a timestamp -- and asserted anyway because it is the reason pacing is allowed
    to exist at all. A pacer that adjusted pts_ms would make a demo run and a benchmark run
    of the same file disagree about when everything happened, and only one of them could be
    the number in the report.
    """
    frames = list(range(0, 2_000, 40))
    clock = FakeClock()
    paced = ReplayPacer(2.0, sleep=clock.sleep, monotonic=clock.monotonic)
    unpaced = ReplayPacer()

    def replay(pacer: ReplayPacer) -> list[int]:
        stamped = []
        for pts in frames:
            pacer.wait_for(pts)
            stamped.append(pts)  # what the source puts on the envelope
        return stamped

    assert replay(paced) == replay(unpaced) == frames
    assert paced.slept_seconds > 0.0, "the paced run did wait; it just did not rewrite time"
    assert unpaced.slept_seconds == 0.0


def test_a_reset_re_anchors_on_the_next_frame():
    """A new session restarts the source timeline. Without the reset, the first frame of the
    new session would be measured against the old session's anchor and the pacer would
    either sleep for minutes or never sleep again."""
    clock = FakeClock()
    pacer = ReplayPacer(1.0, sleep=clock.sleep, monotonic=clock.monotonic)
    pacer.wait_for(0)
    pacer.wait_for(1_000)
    pacer.reset()
    assert pacer.wait_for(500_000) == 0.0


def test_pacer_stats_say_whether_the_run_was_paced():
    """A performance number from a paced run is not comparable with one from an unpaced run,
    so the report has to be able to tell them apart."""
    clock = FakeClock()
    pacer = ReplayPacer(1.0, sleep=clock.sleep, monotonic=clock.monotonic)
    pacer.wait_for(0)
    pacer.wait_for(2_000)
    assert pacer.stats() == {
        "speed": 1.0,
        "slept_seconds": 2.0,
        "frames_behind_schedule": 0,
    }
    assert ReplayPacer().stats()["speed"] is None
