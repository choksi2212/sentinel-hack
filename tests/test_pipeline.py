"""Tests for ai/pipeline.py -- the per-frame orchestrator, stages 3 to 13.

The pipeline decides what becomes a database row. Its four load-bearing decisions, and
the failure each one exists to prevent, are what these tests are weighted around, worst
first:

  1. It never fabricates a plate. A track whose plate is located but not read emits
     plate: null (or an "unreadable" block that keeps the raw attempt but refuses to
     normalize it) -- never a guess. A fabricated plate does not merely lose
     information; it points an investigation at a vehicle that was never there. This is
     Contracts section 12/3.2 and it is the single worst thing this file could do.

  2. The gate decides whether to *compute* a plate, not whether the vehicle *exists*.
     A gated-out vehicle is still emitted with plate: null. A count that quietly runs
     low because sightings were dropped is the second-worst failure -- invisible, unlike
     a crash.

  3. OCR is deferred to track finalization, not run per frame. The evidence for one
     event is complete only when the track goes idle, so the read happens once, over the
     top-K crops, at flush time.

  4. Events are returned, not sent. process_frame hands back a list; putting it on the
     wire is the worker's job. The pipeline holds no sink.

Plus the invariants that keep those decisions honest across a real run: a session change
finalizes the old session *before* touching state (the track-merge bug produces a vehicle
crossing a city in four seconds), a plate candidate for an ungated track raises rather
than attaching to a guessed vehicle, dedup suppresses a track finalized twice, and
observed_at is byte-identical across two offline replays of the same clip.

The collaborators are real -- VehicleGate, EvidenceAccumulator, SightingDeduper,
fuse_observations, build_event_with_evidence, plate_quality -- because their interaction
is exactly what is under test. Only the four model stages are scripted fakes: a real
detector would make the frame contents load-bearing, and the point here is the wiring, not
the models.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from ai.contracts.event import EventEnvelope, ModelProvenance
from ai.contracts.frame import FrameEnvelope
from ai.contracts.ids import new_session_id
from ai.contracts.stages import PlateCandidate, TrackResult
from ai.emit.snapshot import NullSnapshotWriter
from ai.ocr.base import OCRRead
from ai.pipeline import Pipeline
from ai.quality.gate import VehicleGate

# --------------------------------------------------------------------------- fixtures

MODEL = ModelProvenance(
    detector="fake-detector@1",
    plate_detector="fake-plate@1",
    ocr="fake-ocr@1",
    tracker="fake-tracker@1",
    pipeline_version="0.1.0-test",
)

# A fixed anchor so offline observed_at is a known constant. See ai/pipeline.py _observed_at:
# an offline replay stamps replay_anchor + the session's accumulated PTS offset, so pinning
# the anchor makes the whole run's timestamps reproducible to the millisecond.
ANCHOR = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
CAM = "cam04"

# One deterministic frame reused everywhere. The scripted tracker ignores the pixels, so the
# contents only need to be a valid HxWx3 uint8 BGR array -- seeded so the crop the OCR fake
# slices out is identical run to run, which is what makes crop quality reproducible.
FRAME = np.random.default_rng(0).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)

# Boxes chosen against the default VehicleGate. PASS_BOX clears the min height and sits below
# the departing-vehicle cutoff; SMALL_BOX is refused as vehicle_too_small; PLATE_BOX is a
# full-frame plate wide enough to clear the OCR width floor (80 px > 24).
PASS_BOX = (500, 300, 620, 400)    # w120 h100
SECOND_BOX = (100, 300, 220, 400)  # w120 h100, a second track with no plate
SMALL_BOX = (700, 300, 740, 336)   # w40 h36 -> vehicle_too_small
PLATE_BOX = (520, 360, 600, 384)   # w80 h24

READABLE = "GJ01AB1234"


class FakeDetector:
    """The scripted tracker ignores detections, so this need not return real boxes."""

    model_name = "fake-detector"

    def detect_envelope(self, envelope):
        return []


class FakeTracker:
    """Returns exactly the TrackResults scripted per frame_index; ignores detections.

    Driving tracks from a script rather than from box overlap is what lets a test say
    "track 1 is seen on frames 0 and 3 and nowhere else" without hand-tuning IoU.
    """

    tracker_name = "fake"

    def __init__(self, camera_id, script):
        self.camera_id = camera_id
        self.script = script
        self.session = None
        self.resets = 0

    def reset(self, *, stream_session_id=None):
        self.session = stream_session_id
        self.resets += 1

    def update(self, detections, *, frame_index, pts_ms):
        return [
            TrackResult(
                camera_id=self.camera_id,
                stream_session_id=self.session,
                track_id=tid,
                bbox_xyxy=bbox,
                class_name=cls,
                confidence=conf,
                frame_index=frame_index,
                pts_ms=pts_ms,
            )
            for (tid, bbox, cls, conf) in self.script.get(frame_index, ())
        ]


class FakePlateDetector:
    """Full-frame PlateCandidates per (frame_index -> {track_id: (bbox, conf)})."""

    model_name = "fake-plate"

    def __init__(self, script):
        self.script = script

    def detect_plates_envelope(self, envelope, tracks):
        wanted = self.script.get(envelope.frame_index, {})
        return {
            tid: PlateCandidate(plate_bbox_xyxy=bbox, detector_confidence=conf)
            for tid, (bbox, conf) in wanted.items()
        }


class FakeOCR:
    """A real-ish crop cut plus scripted reads keyed by the frame the crop came from.

    read_crop keys off frame_ref.frame_index because the pipeline reads a track's crops at
    finalization, by which point the frame is gone -- exactly the deferral FrameRef exists
    to support. A read entry of None (a missing key) means "no plate could be read", which
    is a valid answer, not an error.
    """

    model_name = "fake-ocr"

    def __init__(self, reads):
        self.reads = reads  # dict[frame_index -> (text, confidence)]

    def cut_crop(self, frame_bgr, candidate):
        x1, y1, x2, y2 = candidate.plate_bbox_xyxy
        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def read_crop(self, crop_bgr, candidate, *, frame_ref=None):
        entry = self.reads.get(frame_ref.frame_index if frame_ref else None)
        if entry is None:
            return None
        text, conf = entry
        return OCRRead(text=text, confidence=conf, variant="fake")


class RoguePlateDetector:
    """Returns a candidate for a track_id that was never gated -- must be rejected.

    The pipeline offers plate crops only for tracks it gated in this frame. A candidate
    keyed to an unknown track means the plate stage and the tracker disagree about what is
    in the frame, and attaching that plate to whatever track happened to be nearby is how a
    plate ends up on the wrong vehicle. The pipeline raises instead.
    """

    model_name = "rogue"

    def detect_plates_envelope(self, envelope, tracks):
        return {999: PlateCandidate(plate_bbox_xyxy=PLATE_BOX, detector_confidence=0.9)}


class RecordingSnapshotWriter:
    """A non-Null writer that records every call, so snapshot staging can be asserted.

    Must not subclass NullSnapshotWriter: _stage_snapshots early-returns on that type, so a
    Null writer would make every staging assertion vacuously pass.
    """

    def __init__(self):
        self.staged = []          # (track_key, rank)
        self.committed = []       # track_key
        self.dropped = []         # track_key
        self.dropped_sessions = []

    def stage_frame(self, track_key, frame_bgr, quality):
        self.staged.append((track_key, quality))
        return True

    def commit(self, track_key, *, event_id, observed_at=None, plate_crop_bgr=None):
        self.committed.append(track_key)
        return ("snap://%s" % event_id, "crop://%s" % event_id)

    def drop(self, track_key):
        self.dropped.append(track_key)

    def drop_session(self, session):
        self.dropped_sessions.append(session)

    def has_staged(self, track_key):
        return any(k == track_key for k, _ in self.staged)

    def clear(self):
        self.staged.clear()

    def stats(self):
        return {"enabled": True, "staged": len(self.staged), "committed": len(self.committed)}


class LoadCloseStage:
    """A stage that records load()/close() and can be told to raise from close()."""

    model_name = "loadclose"

    def __init__(self, *, raise_on_close=False):
        self.loaded = 0
        self.closed = 0
        self.raise_on_close = raise_on_close

    def load(self):
        self.loaded += 1

    def close(self):
        self.closed += 1
        if self.raise_on_close:
            raise RuntimeError("close blew up")

    def detect_envelope(self, envelope):
        return []


def frame(idx, pts, session, *, mode="file", wallclock=None):
    return FrameEnvelope(
        camera_id=CAM,
        stream_session_id=session,
        frame_index=idx,
        pts_ms=pts,
        wallclock_utc=wallclock,
        frame_bgr=FRAME,
        width=1280,
        height=720,
        source_mode=mode,
    )


def build(
    *,
    tracker_script,
    plate_script=None,
    reads=None,
    gate=None,
    mode="file",
    snapshots=None,
    watchlist=None,
    plate_detector=None,
):
    """A pipeline wired with real collaborators and scripted model stages."""
    return Pipeline(
        camera_id=CAM,
        source_mode=mode,
        model=MODEL,
        detector=FakeDetector(),
        tracker=FakeTracker(CAM, tracker_script),
        plate_detector=plate_detector or FakePlateDetector(plate_script or {}),
        ocr=FakeOCR(reads or {}),
        gate=gate,
        snapshots=snapshots,
        watchlist=watchlist,
        replay_anchor=ANCHOR,
    )


def one_track_with_plate(reads):
    """The common wiring: track 1 present on frame 0 with a plate box, read per `reads`."""
    return build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},
        reads=reads,
    )


# ============================================================================ TIER 1
# Fabrication and silent loss -- the two worst failure modes this file can produce.


def test_readable_plate_is_emitted_verbatim():
    """A plate that reads cleanly rides through to a valid event carrying that plate."""
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()

    opening = pipe.process_frame(frame(0, 0, session))
    assert opening == [], "a track still in frame must not emit yet"

    events = pipe.process_frame(frame(1, 1000, session))  # 1000 ms idle -> finalize
    assert len(events) == 1
    event = events[0]
    assert event.validate() == []
    assert event.plate is not None
    assert event.plate.normalized == READABLE
    assert event.plate.raw == READABLE
    assert event.camera_id == CAM
    assert event.stream_session_id == session
    assert event.source_mode == "file"


def test_unreadable_plate_is_never_fabricated():
    """OCR returns characters that normalize to nothing -> an 'unreadable' block.

    The raw attempt is preserved so an operator can see what was there, but normalized is
    null and match_state is 'unreadable'. This is the fabrication guard: a located plate
    that cannot be resolved is reported as unresolved, not guessed.
    """
    pipe = one_track_with_plate({0: ("!!", 0.6)})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    event = pipe.process_frame(frame(1, 1000, session))[0]

    assert event.validate() == []
    assert event.plate is not None, "an unreadable located plate is a block, not a drop"
    assert event.plate.match_state == "unreadable"
    assert event.plate.raw == "!!"
    assert not event.plate.normalized, "an unreadable plate must not normalize to a string"
    assert pipe.counters.plate_located_no_read == 1
    assert pipe.counters.events_plate_null == 1
    assert pipe.counters.events_with_plate == 0


def test_located_but_unread_plate_is_null_not_guessed():
    """A plate box is found but OCR returns None -> plate: null, counted as located-unread.

    Distinct from the unreadable case above: there OCR produced junk characters; here it
    produced nothing at all. Both refuse to invent a plate, and both are visible in
    plate_located_no_read -- the counter that keeps 'located but never read' from hiding
    inside a clean-looking plate: null rate.
    """
    pipe = one_track_with_plate({})  # plate box present, no read scripted
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    event = pipe.process_frame(frame(1, 1000, session))[0]

    assert event.validate() == []
    assert event.plate is None
    assert pipe.counters.plate_located_no_read == 1
    assert pipe.counters.events_plate_null == 1
    assert pipe.counters.tracks_with_plate_crops == 1


def test_gated_out_vehicle_still_emits_plate_null():
    """The gate decides plate compute, not event existence: a refused vehicle is still emitted.

    Dropping the whole event when the gate refuses would make the count quietly low --
    exactly the silent loss that is worse than a crash because nothing signals it.
    """
    pipe = build(tracker_script={0: [(1, SMALL_BOX, "car", 0.9)]})  # no plate script at all
    session = new_session_id()

    pipe.process_frame(frame(0, 0, session))
    assert "vehicle_too_small" in (pipe.last_outcome.gate_rejected or {})

    events = pipe.process_frame(frame(1, 1000, session))
    assert len(events) == 1
    assert events[0].plate is None
    assert events[0].validate() == []
    # No plate box was ever located, so this is not a "located but unread" -- it is a vehicle
    # the gate refused to spend a plate detector on.
    assert pipe.counters.plate_located_no_read == 0


def test_unreportable_track_is_dropped():
    """A sub-floor-confidence vehicle with no plate is not published at all.

    The floor keeps 0.20-confidence blobs out of the database. A track dropped here is a
    track that was never a vehicle worth a row -- the drop is counted, not silent.
    """
    pipe = build(tracker_script={0: [(1, PASS_BOX, "car", 0.20)]})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    events = pipe.process_frame(frame(1, 1000, session))

    assert events == []
    assert pipe.tracks_dropped_unreportable == 1
    assert pipe.counters.events_built == 0


def test_plate_evidence_exempts_a_subfloor_track():
    """A located plate is sufficient on its own: a sub-floor vehicle with a plate is reported.

    Whatever the detector scored it, a thing with a plate on it is a vehicle. The permissive
    gate is what lets the sub-floor vehicle reach plate detection in the first place; the
    exemption is what keeps it from being dropped afterwards.
    """
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.20)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},
        reads={0: ("GJ05MN6789", 0.9)},
        gate=VehicleGate(min_confidence=0.1),
    )
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    events = pipe.process_frame(frame(1, 1000, session))

    assert len(events) == 1
    assert events[0].plate is not None
    assert events[0].plate.normalized == "GJ05MN6789"
    assert pipe.tracks_dropped_unreportable == 0


def test_plate_candidate_for_unknown_track_raises():
    """A plate for a track the pipeline never gated is a bug, not a plate to attach.

    Silently attaching it would put a plate on whatever vehicle happened to be nearby. The
    pipeline raises so the disagreement between the plate stage and the tracker surfaces.
    """
    pipe = Pipeline(
        camera_id=CAM,
        source_mode="file",
        model=MODEL,
        detector=FakeDetector(),
        tracker=FakeTracker(CAM, {0: [(1, PASS_BOX, "car", 0.9)]}),
        plate_detector=RoguePlateDetector(),
        ocr=FakeOCR({}),
        replay_anchor=ANCHOR,
    )
    with pytest.raises(ValueError):
        pipe.process_frame(frame(0, 0, new_session_id()))


def test_camera_id_mismatch_raises():
    """A frame from another camera must never be processed on this pipeline.

    Two junctions share nothing; a frame routed to the wrong pipeline would merge their
    tracks. The camera_id is the scope of every track and event, so a mismatch is fatal.
    """
    pipe = build(tracker_script={})
    stray = FrameEnvelope(
        camera_id="cam99",
        stream_session_id=new_session_id(),
        frame_index=0,
        pts_ms=0,
        wallclock_utc=None,
        frame_bgr=FRAME,
        width=1280,
        height=720,
        source_mode="file",
    )
    with pytest.raises(ValueError):
        pipe.process_frame(stray)


def test_session_change_finalizes_old_session_before_clearing_state():
    """A new stream_session_id closes the old session first, so no track merges across it.

    The track-merge bug is a buffer from session A having session B's crops appended after a
    reconnect; the vehicle then appears to cross the city in the gap. Finalizing the old
    session before touching any state is what prevents it -- and the finalized event must
    belong to the old session, proving the boundary held.
    """
    first = new_session_id()
    second = new_session_id()
    pipe = one_track_with_plate({0: (READABLE, 0.9)})

    pipe.process_frame(frame(0, 0, first))
    # Same frame_index 0 in the new session so the tracker script fires again; the only thing
    # that changed is the session id.
    events = pipe.process_frame(frame(0, 0, second))

    assert len(events) == 1, "the old session's open track must be finalized on the change"
    assert events[0].stream_session_id == first
    assert pipe.tracker.session == second, "the tracker is reset to the new session"
    assert pipe.counters.sessions_started == 2


def test_confidences_are_not_collapsed_into_one_number():
    """image_quality and plate confidence are separate fields, never multiplied together.

    A single clean read scores plate confidence 1.0 (it is the whole evidence share) while
    image_quality is the crop's own quality, below 1.0. If the pipeline ever multiplied them
    into one 'probability', the two fields would collapse to the same value. They must not.
    """
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    event = pipe.process_frame(frame(1, 1000, session))[0]

    assert 0.0 <= event.image_quality <= 1.0
    assert 0.0 <= event.plate.confidence <= 1.0
    assert event.image_quality != event.plate.confidence
    assert event.plate.confidence == pytest.approx(1.0)
    assert event.image_quality < 1.0
    # A single observation is low_confidence however high the read scored -- one read is not
    # much evidence, and that judgement is about evidence count, not the crop.
    assert event.plate.match_state == "low_confidence"
    assert event.plate.evidence_count == 1


# ============================================================================ TIER 2
# Temporal fusion, dedup, and lifecycle -- correctness across a whole track's life.


def test_ocr_is_deferred_until_the_track_finalizes():
    """OCR runs when the track goes idle, not on the frame the plate appears.

    While the track is open the crop is cut and buffered but never read; the read happens
    once at finalization. Running OCR per frame would read the same plate ten times and
    throw nine away.
    """
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()

    pipe.process_frame(frame(0, 0, session))
    assert pipe.counters.crops_offered == 1
    assert pipe.counters.ocr_attempts == 0, "no read while the track is still open"

    pipe.process_frame(frame(1, 1000, session))
    assert pipe.counters.ocr_attempts == 1, "the read happens at finalization"


def test_top_k_crop_buffer_bounds_the_ocr_work():
    """Six plate frames on one track keep only the top-K crops, so OCR runs K times, not six.

    The accumulator holds the best four crops (DEFAULT_TOP_K); the other two are rejected on
    quality. OCR then reads the four kept crops, and fusion sees four agreeing observations.
    """
    session = new_session_id()
    pipe = build(
        tracker_script={i: [(1, PASS_BOX, "car", 0.9)] for i in range(6)},
        plate_script={i: {1: (PLATE_BOX, 0.9)} for i in range(6)},
        reads={i: (READABLE, 0.9) for i in range(6)},
    )
    for i in range(6):
        pipe.process_frame(frame(i, i * 100, session))  # 100 ms gaps keep the track open

    events = pipe.process_frame(frame(6, 1400, session))  # 900 ms gap -> finalize
    assert pipe.counters.crops_offered == 6
    assert pipe.counters.crops_rejected_quality == 2
    assert pipe.counters.ocr_attempts == 4
    assert events[0].plate.evidence_count == 4
    assert events[0].plate.normalized == READABLE
    assert events[0].plate.match_state == "probable"


def test_flush_finalizes_open_tracks_at_eof():
    """At end of clip every vehicle still in frame has an open buffer; flush emits them.

    Without flush the last few seconds of every clip are simply never reported. A second
    flush is empty because the buffers are already gone.
    """
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))

    flushed = pipe.flush("eof")
    assert len(flushed) == 1
    assert flushed[0].plate.normalized == READABLE
    assert pipe.flush("eof") == [], "nothing open the second time"


def test_flush_is_empty_when_nothing_is_open():
    pipe = build(tracker_script={})
    assert pipe.flush("shutdown") == []


def test_flush_rejects_an_unknown_reason():
    """A flush reason outside the locked set raises before touching any state.

    The reason lands in the run summary and in metrics; an arbitrary string there is a bug
    that should fail loudly, not a flush that quietly happens under a mislabelled cause.
    """
    pipe = build(tracker_script={})
    with pytest.raises(ValueError):
        pipe.flush("bogus")


def test_dedup_suppresses_a_track_finalized_twice():
    """The same TrackKey finalized twice within the window emits once.

    A vehicle stopped at a signal hits the max-duration cap and is finalized, then keeps
    being tracked and is finalized again. Same camera, session, track_id and plate -> same
    dedupe key; the second emission carries no better evidence, so it is suppressed.
    """
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9)], 3: [(1, PASS_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}, 3: {1: (PLATE_BOX, 0.9)}},
        reads={0: (READABLE, 0.9), 3: (READABLE, 0.9)},
    )
    pipe.process_frame(frame(0, 0, session))
    first = pipe.process_frame(frame(1, 1000, session))   # finalize #1 -> emit
    pipe.process_frame(frame(3, 1100, session))           # track 1 seen again
    second = pipe.process_frame(frame(4, 2100, session))  # finalize #2 -> same key

    assert len(first) == 1
    assert second == [], "the re-finalized sighting adds nothing and is suppressed"
    assert pipe.events_suppressed == 1


def test_observed_at_is_byte_identical_across_replays():
    """Two offline replays of the same clip produce the same observed_at, to the millisecond.

    Offline events are timestamped from replay_anchor plus the session's PTS, with no
    wallclock involved, so a pinned anchor makes the run reproducible. Anything else would
    make an offline regression test flaky on the clock.
    """
    session = new_session_id()

    def run():
        pipe = one_track_with_plate({0: (READABLE, 0.9)})
        pipe.process_frame(frame(0, 0, session))
        return pipe.process_frame(frame(1, 1000, session))[0].observed_at

    first = run()
    second = run()
    assert first == second
    assert first == "2026-09-01T10:00:00.000Z"


def test_live_wallclock_is_passed_through_unchanged():
    """In live mode the frame's own wallclock is the observed_at, verbatim."""
    wallclock = "2026-09-01T12:34:56.789Z"
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},
        reads={0: (READABLE, 0.9)},
        mode="live_rtsp",
    )
    pipe.process_frame(frame(0, 0, session, mode="live_rtsp", wallclock=wallclock))
    event = pipe.process_frame(frame(1, 1000, session, mode="live_rtsp", wallclock=wallclock))[0]

    assert event.observed_at == wallclock
    assert event.validate() == []


def test_live_without_wallclock_stamps_a_valid_timestamp():
    """A live frame with no wallclock falls back to arrival time -- still a valid tz-aware ISO.

    A live source that drops its wallclock must not crash the run or emit a naive timestamp
    that fails validation downstream; it stamps now() and the event stays valid.
    """
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},
        reads={0: (READABLE, 0.9)},
        mode="live_rtsp",
    )
    pipe.process_frame(frame(0, 0, session, mode="live_rtsp", wallclock=None))
    event = pipe.process_frame(frame(1, 1000, session, mode="live_rtsp", wallclock=None))[0]

    assert event.observed_at.endswith("Z")
    assert event.validate() == []


# ============================================================================ TIER 3
# Counters, snapshots, provenance, and structure -- the run summary must be true.


def test_lifecycle_counters_add_up():
    """The counters that make the run summary trustworthy track one readable track correctly."""
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    pipe.process_frame(frame(1, 1000, session))

    c = pipe.counters
    assert c.tracks_started == 1
    assert c.tracks_completed == 1
    assert c.events_built == 1
    assert c.events_with_plate == 1
    assert c.events_plate_null == 0
    assert c.crops_offered == 1
    assert pipe.events_emitted == 1


def test_snapshot_ranks_favour_plate_frames_over_fill():
    """Every track is staged; a plate frame outranks a plateless one by a whole point.

    A vehicle with a located plate stages at 1.0 + quality (above 1.0); a track with no plate
    this frame stages at its fill fraction (at or below 1.0). Both are staged -- even a
    plateless vehicle needs a photograph, or its plate: null event is unconfirmable noise.
    """
    writer = RecordingSnapshotWriter()
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9), (2, SECOND_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},  # only track 1 has a plate
        reads={0: (READABLE, 0.9)},
        snapshots=writer,
    )
    pipe.process_frame(frame(0, 0, session))

    ranks = {k.track_id: rank for k, rank in writer.staged}
    assert set(ranks) == {1, 2}, "both tracks staged, not only the gated-with-plate one"
    assert ranks[1] > 1.0, "the plate frame ranks above the fill-fraction ceiling"
    assert ranks[2] <= 1.0, "a plateless frame ranks by fill fraction"


def test_unreportable_track_drops_its_staged_snapshot():
    """A track that is dropped as unreportable also drops its staged frames.

    Leaving them staged would leak memory for a track that never becomes an event, and the
    drop is what NullSnapshotWriter makes a no-op -- so it has to be exercised with a real
    writer to be seen at all.
    """
    writer = RecordingSnapshotWriter()
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.20)]},  # sub-floor, no plate
        snapshots=writer,
    )
    pipe.process_frame(frame(0, 0, session))
    pipe.process_frame(frame(1, 1000, session))  # finalize -> dropped

    assert pipe.tracks_dropped_unreportable == 1
    assert len(writer.dropped) == 1
    assert writer.committed == [], "a dropped track is never committed"


def test_watchlist_hit_marks_the_event_exact():
    """A normalized plate on the watchlist is marked exact; a miss is not.

    'exact' is the state that fires an alert, so it must come only from a real watchlist hit,
    never from OCR confidence alone -- a single clean read off the watchlist is low_confidence.
    """
    session = new_session_id()
    pipe = build(
        tracker_script={0: [(1, PASS_BOX, "car", 0.9)]},
        plate_script={0: {1: (PLATE_BOX, 0.9)}},
        reads={0: (READABLE, 0.9)},
        watchlist=lambda plate: plate == READABLE,
    )
    pipe.process_frame(frame(0, 0, session))
    event = pipe.process_frame(frame(1, 1000, session))[0]
    assert event.plate.match_state == "exact"


def test_load_calls_each_stage_and_close_swallows_failures():
    """load() loads every model; close() releases them and never lets one failure mask another.

    A close() that raised would abort the shutdown of the stages after it and lose the error
    that actually killed the run, so a failing close is logged and swallowed.
    """
    detector = LoadCloseStage()
    plate = LoadCloseStage(raise_on_close=True)  # the one that misbehaves on close
    ocr = LoadCloseStage()
    pipe = Pipeline(
        camera_id=CAM,
        source_mode="file",
        model=MODEL,
        detector=detector,
        tracker=FakeTracker(CAM, {}),
        plate_detector=plate,
        ocr=ocr,
        replay_anchor=ANCHOR,
    )
    pipe.load()
    assert (detector.loaded, plate.loaded, ocr.loaded) == (1, 1, 1)

    pipe.close()  # must not raise despite plate.close() blowing up
    assert (detector.closed, plate.closed, ocr.closed) == (1, 1, 1)


def test_empty_camera_id_is_rejected():
    """Every track and event is scoped to a camera_id, so an empty one is a construction bug."""
    with pytest.raises(ValueError):
        build(tracker_script={}).__class__(
            camera_id="",
            source_mode="file",
            model=MODEL,
            detector=FakeDetector(),
            tracker=FakeTracker("", {}),
            plate_detector=FakePlateDetector({}),
            ocr=FakeOCR({}),
        )


def test_events_are_returned_not_sent():
    """process_frame hands events back; the pipeline holds no sink to put them on the wire.

    Emission is the worker's job. A pipeline built with no sink argument still produces fully
    formed events, and it exposes no send/post/emit/publish method that could leak one out a
    side channel.
    """
    pipe = one_track_with_plate({0: (READABLE, 0.9)})
    session = new_session_id()
    pipe.process_frame(frame(0, 0, session))
    events = pipe.process_frame(frame(1, 1000, session))

    assert all(isinstance(e, EventEnvelope) for e in events)
    assert pipe.events_emitted == len(events)
    for channel in ("send", "post", "emit", "publish", "sink", "http"):
        assert not hasattr(pipe, channel), f"the pipeline must not expose a {channel!r} channel"


def test_stats_reports_every_stage():
    """stats() gathers a line from every stage the pipeline holds, for the worker to print."""
    pipe = build(tracker_script={})
    stats = pipe.stats()
    for key in ("camera_id", "source_mode", "events_emitted", "gate", "dedup", "counters"):
        assert key in stats
    assert stats["camera_id"] == CAM
    assert isinstance(pipe.snapshots, NullSnapshotWriter)  # default writer, no snapshots configured
