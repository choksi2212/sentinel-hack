"""Emission -- stage 13 of the 14: a finished track becomes one event, and leaves.

This is the pipeline's narrowing point, so the tests are weighted the way the failures
are, worst first:

**Tier 1 -- fabrication and silent loss.** The two ways this stage can lie. It can attach a
real registration number to a place and time it was never at (a fabricated plate, worse
than a null because it is invisible downstream), and it can drop a sighting without saying
so (a vehicle count that is quietly a little low, the hardest error to notice because the
number is still a number). Most of this file is about the second one -- the sink's
accounting invariant, `build_event` raising rather than returning None, the four plate-block
outcomes -- because it is the one with no natural alarm.

**Tier 2 -- retry honesty.** 201 is a new sighting, 200 is a duplicate and NOT an error,
422 is permanent and must never be retried against a live backend, and a network error is
the dangerous case that only the stable event_id makes safe to retry at all. Getting the
classification wrong turns one bad event into an infinite loop, or a transient blip into
permanent data loss.

**Tier 3 -- the evidence stills and the storage arithmetic.** The USP is "we don't
centralize every video, we centralize intelligence"; a still per *event* is intelligence and
a still per *frame* is video with extra steps. The snapshot module's own docstring writes
down the bytes, so this file re-measures them rather than trusting the comment -- that group
is the one machine-dependent corner and is marked as such.

Nothing here needs Postgres or a real ingest server: the HTTP tests stand up a
`http.server` on a loopback port, and the ones that only classify a response drive
`_post_once` against a fake session so the retry matrix is exact rather than timing-
dependent. `requests` and `Pillow` are import-guarded so the bulk of the file still runs
without them.
"""

import glob
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from ai.contracts.enums import SOURCE_MODES
from ai.contracts.event import EventEnvelope, ModelProvenance
from ai.contracts.ids import TrackKey, new_event_id
from ai.contracts.stages import PlateCandidate, PlateObservation, TrackResult
from ai.emit import (
    EventBuildError,
    FileEventSink,
    HttpEventSink,
    NullEventSink,
    NullSnapshotWriter,
    SnapshotWriter,
    build_event,
    build_event_with_evidence,
    build_events,
    build_snapshot_writer,
    observations_from_buffer,
    safe_component,
    winning_crop,
)
from ai.emit.http_sink import (
    DEFAULT_INGEST_PATH,
    _backoff,
    _retry_after,
)
from ai.emit.snapshot import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SNAPSHOT_WIDTH,
    PLATE_CROP_QUALITY,
)
from ai.fusion.accumulator import CropBuffer, TrackCrop

try:  # HttpEventSink.open() imports requests lazily; the pure-logic tests do not need it.
    import requests as _requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_REQUESTS = False

try:  # The JPEG encode needs Pillow; without it _encode_jpeg returns None by design.
    from PIL import Image as _PILImage  # noqa: F401

    HAS_PIL = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PIL = False

requires_requests = pytest.mark.skipif(
    not HAS_REQUESTS, reason="requests not installed; HttpEventSink.open() needs it"
)
requires_pil = pytest.mark.skipif(
    not HAS_PIL, reason="Pillow not installed; JPEG encode is a no-op without it"
)

CAMERA = "cam04"
SESSION = "3a7f1e02-5c9b-4d18-8e63-2b4a9c7d1f05"
OBSERVED_AT = "2026-09-01T10:03:21.234Z"

MODEL = ModelProvenance(
    detector="rfdetr-small",
    plate_detector="rtdetrv2-justjuu",
    ocr="paddleocr-3.0",
    tracker="bytetrack",
    pipeline_version="0.1.0",
)


# --------------------------------------------------------------------------- builders
#
# Real objects, not mocks. build_event calls validate() before returning, so a helper that
# produced a subtly-invalid buffer would fail loudly here rather than passing a bad event
# downstream -- which is the property the whole module is defending, applied to its own
# fixtures.


def _track(track_id: int = 42, *, pts_ms: int = 100, session: str = SESSION) -> TrackResult:
    return TrackResult(
        camera_id=CAMERA,
        stream_session_id=session,
        track_id=track_id,
        bbox_xyxy=(10, 10, 210, 160),
        class_name="car",
        confidence=0.9,
        frame_index=pts_ms // 100,
        pts_ms=pts_ms,
    )


def _crop(
    frame_index: int, pts_ms: int, *, quality: float = 0.7, width: int = 62
) -> TrackCrop:
    return TrackCrop(
        quality=quality,
        crop_bgr=np.zeros((24, width, 3), dtype=np.uint8),
        candidate=PlateCandidate(
            plate_bbox_xyxy=(100, 200, 100 + width, 224), detector_confidence=0.8
        ),
        frame_index=frame_index,
        pts_ms=pts_ms,
        observed_at=OBSERVED_AT,
        vehicle_bbox_xyxy=(10, 10, 210, 160),
        vehicle_class="car",
        vehicle_confidence=0.9,
    )


def _obs(
    text: str,
    frame_index: int,
    pts_ms: int,
    *,
    ocr_confidence: float = 0.9,
    image_quality: float = 0.9,
    width: int = 62,
    track_id: int = 42,
    session: str = SESSION,
) -> PlateObservation:
    return PlateObservation(
        camera_id=CAMERA,
        stream_session_id=session,
        track_id=track_id,
        plate_bbox_xyxy=(100, 200, 100 + width, 224),
        plate_width_px=width,
        plate_raw=text,
        ocr_confidence=ocr_confidence,
        image_quality=image_quality,
        frame_index=frame_index,
        pts_ms=pts_ms,
        observed_at=OBSERVED_AT,
    )


def _readable_buffer(
    plate: str = "GJ01AB1234", *, n: int = 3, track_id: int = 42
) -> tuple[CropBuffer, list[PlateObservation]]:
    """A buffer that saw one vehicle over n frames and read the plate every time."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, track_id))
    observations = []
    for i in range(n):
        pts = 100 + i * 100
        buf.note_track(_track(track_id, pts_ms=pts), OBSERVED_AT)
        buf.offer(_crop(i, pts))
        observations.append(_obs(plate, i, pts))
    return buf, observations


def _event(plate: str = "GJ01AB1234", *, event_id=None, track_id: int = 42) -> EventEnvelope:
    buf, obs = _readable_buffer(plate, track_id=track_id)
    return build_event(
        buf, source_mode="file", model=MODEL, observations=obs, event_id=event_id
    )


# ============================================================================ TIER 1
# observations_from_buffer -- what becomes the audit trail.
# ============================================================================


def test_observations_from_buffer_keeps_a_garbage_read_and_drops_an_empty_one():
    """"!!" is an observation; "" is not. The filter is on empty text, not unnormalizable.

    A read of "!!" means OCR looked at the plate and returned characters that are not a
    registration number -- that is the "unreadable" case and the schema requires a non-empty
    raw to carry it. A read of "" means nothing came back, so there is nothing to audit.
    """
    buf, _ = _readable_buffer()
    rows = [
        (_crop(0, 100), "GJ01AB1234", 0.9),
        (_crop(1, 200), "", 0.5),
        (_crop(2, 300), "!!", 0.4),
    ]
    out = observations_from_buffer(buf, rows)
    assert [o.plate_raw for o in out] == ["GJ01AB1234", "!!"]


def test_observations_from_buffer_carries_the_crop_quality_not_the_frame_quality():
    """image_quality on the observation is the plate crop's, so a clean background cannot
    vouch for a blurry plate. See ai/emit/builder.py."""
    buf, _ = _readable_buffer()
    crop = _crop(0, 100, quality=0.33)
    (obs,) = observations_from_buffer(buf, [(crop, "GJ01AB1234", 0.9)])
    assert obs.image_quality == pytest.approx(0.33)
    assert obs.ocr_confidence == pytest.approx(0.9)


def test_observations_from_buffer_copies_the_track_identity_onto_every_row():
    """Every observation carries the buffer's TrackKey, because fusion runs after the track
    is gone and reconstructing which vehicle a read belonged to is the merge bug's doorway."""
    buf, _ = _readable_buffer(track_id=7)
    out = observations_from_buffer(buf, [(_crop(0, 100), "GJ01AB1234", 0.9)])
    assert out[0].track_key == TrackKey(CAMERA, SESSION, 7)


def test_observations_from_buffer_empty_reads_give_empty_list():
    assert observations_from_buffer(_readable_buffer()[0], []) == []


# ============================================================================ TIER 1
# build_event -- the join point where a bad event must die on this side of the wire.
# ============================================================================


def test_build_event_produces_a_valid_envelope():
    event = _event()
    assert event.validate() == []
    assert event.source_mode == "file"
    assert event.camera_id == CAMERA
    assert event.plate.normalized == "GJ01AB1234"


def test_build_event_mints_a_uuid_event_id_when_none_is_given():
    event = _event()
    # The idempotency key. If this is not a UUID the backend's CHECK rejects it with a 422.
    assert event.validate() == []
    assert len(event.event_id) == 36 and event.event_id.count("-") == 4


def test_build_event_keeps_the_event_id_it_is_handed():
    """Passed in so build_event_with_evidence can name the stills before the event exists,
    and so a retry mints nothing new. A second id would double-count the sighting."""
    given = new_event_id()
    assert _event(event_id=given).event_id == given


@pytest.mark.parametrize("mode", sorted(SOURCE_MODES))
def test_build_event_accepts_every_source_mode(mode):
    buf, obs = _readable_buffer()
    assert build_event(buf, source_mode=mode, model=MODEL, observations=obs).source_mode == mode


def test_build_event_rejects_a_source_mode_the_ingest_check_would_422():
    """"live" is the plausible-but-wrong one -- it is not in SOURCE_MODES (live_rtsp and
    live_hls are). Catching it here names the field in the process that built it."""
    buf, obs = _readable_buffer()
    with pytest.raises(EventBuildError, match="source_mode"):
        build_event(buf, source_mode="live", model=MODEL, observations=obs)


def test_build_event_raises_rather_than_returning_none_on_a_buffer_never_seen():
    """A track with no observed_at was never noted, so this buffer holds crops for a track
    that was never seen. Raising rather than returning None is the whole EventBuildError
    contract: a silently dropped track is a vehicle count that is quietly wrong."""
    empty = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 99))
    with pytest.raises(EventBuildError, match="observed_at"):
        build_event(empty, source_mode="file", model=MODEL)


def test_build_event_uses_the_last_observed_at_not_the_first():
    """A sighting is reported when the evidence was complete, not when the vehicle appeared.
    First-frame timestamps make a journey show vehicles arriving before they were read."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(pts_ms=100), "2026-09-01T10:00:00.000Z")
    buf.offer(_crop(0, 100))
    buf.note_track(_track(pts_ms=200), "2026-09-01T10:00:09.000Z")
    event = build_event(buf, source_mode="file", model=MODEL, observations=[_obs("GJ01AB1234", 0, 100)])
    assert event.observed_at == "2026-09-01T10:00:09.000Z"


def test_build_event_source_pts_is_the_last_pts():
    buf, obs = _readable_buffer(n=3)  # pts 100, 200, 300
    assert build_event(buf, source_mode="file", model=MODEL, observations=obs).source_pts_ms == 300


def test_build_event_validates_before_returning_and_names_the_field():
    """A degenerate vehicle box would be a 422 at ingest; build_event catches it here and
    the message names vehicle.bbox_xyxy rather than surfacing as an HTTP status two modules
    away in a retry loop."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    # note_track with a zero-area box: sets observed_at (so we get past that guard) but
    # leaves a degenerate best_vehicle_bbox that validate() must reject.
    buf.note_track(
        TrackResult(
            camera_id=CAMERA, stream_session_id=SESSION, track_id=42,
            bbox_xyxy=(50, 50, 50, 50), class_name="car", confidence=0.9,
            frame_index=1, pts_ms=100,
        ),
        OBSERVED_AT,
    )
    with pytest.raises(EventBuildError, match="bbox"):
        build_event(buf, source_mode="file", model=MODEL)


def test_build_event_passed_fused_wins_over_recomputing():
    """The worker computes the FusedPlate for logging before it decides to emit, so it
    passes it in. build_event must use it rather than re-running consensus."""
    from ai.contracts.stages import FusedPlate

    buf, obs = _readable_buffer()
    injected = FusedPlate(
        normalized="MH12DE9812", confidence=0.83, evidence_count=2,
        best_observation=obs[0], total_observations=len(obs),
    )
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs, fused=injected)
    assert event.plate.normalized == "MH12DE9812"


# ============================================================================ TIER 1
# _plate_block -- the four outcomes, and the line the schema draws between them.
# ============================================================================


def test_plate_block_no_crops_is_plate_null():
    """No plate was ever located. Nothing to report."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(), OBSERVED_AT)
    event = build_event(buf, source_mode="file", model=MODEL, observations=[])
    assert event.plate is None


def test_plate_block_located_but_no_read_is_plate_null():
    """A crop was kept but OCR returned nothing at all -- no raw to report, so plate: null.
    Distinct from 'unreadable', which requires characters to have come back."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(), OBSERVED_AT)
    buf.offer(_crop(0, 100))
    event = build_event(buf, source_mode="file", model=MODEL, observations=[])
    assert event.plate is None


def test_plate_block_located_with_garbage_is_unreadable_not_null():
    """OCR returned characters that are not a registration number. This is a measure of OCR
    performance and a different statement from 'no plate was visible'."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(), OBSERVED_AT)
    buf.offer(_crop(0, 100))
    event = build_event(buf, source_mode="file", model=MODEL, observations=[_obs("!!", 0, 100)])
    assert event.plate is not None
    assert event.plate.match_state == "unreadable"
    assert event.plate.raw == "!!"
    assert event.plate.normalized is None


def test_plate_block_unreadable_confidence_is_zero_not_the_discarded_read():
    """A confidence on a plate the event declares unreadable is a number about nothing.
    Ranking unreadable events by how nearly they were read is not a thing anyone should do."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(), OBSERVED_AT)
    buf.offer(_crop(0, 100))
    event = build_event(
        buf, source_mode="file", model=MODEL,
        observations=[_obs("!!", 0, 100, ocr_confidence=0.71)],
    )
    assert event.plate.confidence == 0.0


def test_plate_block_unreadable_evidence_count_is_the_number_of_failed_reads():
    """Not zero: the schema requires at least one, and one blurry attempt versus eleven
    consistent failures are different amounts of evidence for the same conclusion."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    for i in range(4):
        buf.note_track(_track(pts_ms=100 + i * 100), OBSERVED_AT)
        buf.offer(_crop(i, 100 + i * 100))
    obs = [_obs("!!", i, 100 + i * 100) for i in range(4)]
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs)
    assert event.plate.evidence_count == 4


def test_plate_block_unreadable_raw_is_the_highest_weight_attempt():
    """raw carries the best attempt so the failure is inspectable -- 'OCR returned GJ0 on a
    plate it could not resolve' is a diagnostic; 'it failed' is not."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    for i in range(2):
        buf.note_track(_track(pts_ms=100 + i * 100), OBSERVED_AT)
        buf.offer(_crop(i, 100 + i * 100))
    obs = [
        _obs("GJ0", 0, 100, ocr_confidence=0.9, image_quality=0.9),   # weight 0.81
        _obs("XX", 1, 200, ocr_confidence=0.2, image_quality=0.2),    # weight 0.04
    ]
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs)
    assert event.plate.raw == "GJ0"


def test_plate_block_read_keeps_raw_alongside_normalized():
    """raw is exactly what OCR returned; keeping it beside the normalized form is what lets
    a report separate OCR's accuracy from normalization's rescues and ruins."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    for i in range(3):
        pts = 100 + i * 100
        buf.note_track(_track(pts_ms=pts), OBSERVED_AT)
        buf.offer(_crop(i, pts))
    obs = [_obs("GJ 01 AB 1234", i, 100 + i * 100) for i in range(3)]
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs)
    assert event.plate.raw == "GJ 01 AB 1234"
    assert event.plate.normalized == "GJ01AB1234"


def test_plate_block_width_and_bbox_come_from_the_winning_observation():
    """Not the best crop. The width bucket has to describe the frame the answer came from,
    or a track whose best crop was 110 px but whose read came from a 60 px frame credits the
    >100 bucket for a read it did not produce."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    # Best-quality crop is a wide one that produced no winning read; the winning read is a
    # narrow, lower-quality frame.
    buf.note_track(_track(pts_ms=100), OBSERVED_AT)
    buf.offer(_crop(0, 100, quality=0.95, width=120))
    buf.note_track(_track(pts_ms=200), OBSERVED_AT)
    buf.offer(_crop(1, 200, quality=0.40, width=60))
    obs = [
        _obs("GJ01AB1234", 1, 200, width=60, ocr_confidence=0.9, image_quality=0.9),
        _obs("GJ01AB1234", 1, 200, width=60, ocr_confidence=0.8, image_quality=0.8),
    ]
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs)
    assert event.plate.plate_width_px == 60
    assert tuple(event.plate.bbox_xyxy) == (100, 200, 160, 224)


def test_plate_block_probable_requires_two_agreeing_reads_over_080():
    buf, obs = _readable_buffer(n=3)
    event = build_event(buf, source_mode="file", model=MODEL, observations=obs)
    assert event.plate.match_state == "probable"
    assert event.plate.evidence_count >= 2


def test_plate_block_exact_only_on_a_watchlist_hit():
    buf, obs = _readable_buffer(n=3)
    event = build_event(
        buf, source_mode="file", model=MODEL, observations=obs, exact_watchlist_hit=True
    )
    assert event.plate.match_state == "exact"


# ============================================================================ TIER 1
# winning_crop -- the operator must see the image the answer came from.
# ============================================================================


def test_winning_crop_matches_the_fused_frame():
    from ai.fusion.consensus import fuse_observations

    buf, obs = _readable_buffer(n=3)
    fused = fuse_observations(obs)
    crop = winning_crop(buf, fused)
    assert crop is not None
    assert crop.frame_index == fused.best_observation.frame_index


def test_winning_crop_falls_back_to_best_when_nothing_was_read():
    """No fused plate means no winning frame, so the sharpest view is the most useful thing
    to show. crops[0] is the highest-quality crop (offer() keeps them sorted)."""
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    buf.note_track(_track(), OBSERVED_AT)
    buf.offer(_crop(0, 100, quality=0.3))
    buf.offer(_crop(1, 200, quality=0.9))
    assert winning_crop(buf, None).frame_index == 1


def test_winning_crop_none_when_there_are_no_crops():
    buf = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 42))
    assert winning_crop(buf, None) is None


# ============================================================================ TIER 1
# build_event_with_evidence -- mint id -> commit stills -> build event, in that one order.
# ============================================================================


def test_build_event_with_evidence_names_stills_after_the_event_id():
    """The one sequence that works. A second id would orphan the JPEGs under the first and
    leave the event pointing at nothing."""
    buf, obs = _readable_buffer()
    recorded = {}

    class _Spy:
        def commit(self, track_key, *, event_id, observed_at=None, plate_crop_bgr=None):
            recorded["event_id"] = event_id
            recorded["had_crop"] = plate_crop_bgr is not None
            return f"file:///snap/{event_id}.jpg", f"file:///snap/{event_id}_plate.jpg"

    event = build_event_with_evidence(
        buf, source_mode="file", model=MODEL, snapshots=_Spy(), observations=obs
    )
    assert recorded["event_id"] == event.event_id
    assert recorded["had_crop"] is True
    assert event.evidence.snapshot_uri.endswith(f"{event.event_id}.jpg")
    assert event.evidence.plate_crop_uri.endswith(f"{event.event_id}_plate.jpg")


def test_build_event_with_evidence_null_writer_gives_a_valid_event_with_null_uris():
    buf, obs = _readable_buffer()
    event = build_event_with_evidence(
        buf, source_mode="file", model=MODEL, snapshots=NullSnapshotWriter(), observations=obs
    )
    assert event.validate() == []
    assert event.evidence.snapshot_uri is None
    assert event.evidence.plate_crop_uri is None


def test_build_event_with_evidence_survives_a_commit_that_returns_nulls():
    """A commit failure is a storage problem, not an event failure. Both URIs go null and
    the sighting is emitted regardless; dropping the vehicle over an unwritable JPEG would
    turn a missing photograph into a missing observation."""
    buf, obs = _readable_buffer()

    class _Broken:
        def commit(self, *a, **k):
            return None, None

    event = build_event_with_evidence(
        buf, source_mode="file", model=MODEL, snapshots=_Broken(), observations=obs
    )
    assert event.validate() == []
    assert event.evidence.snapshot_uri is None


# ============================================================================ TIER 1
# build_events -- batch, skipping none silently.
# ============================================================================


def test_build_events_builds_one_per_buffer():
    buffers = [_readable_buffer(track_id=t)[0] for t in (1, 2, 3)]

    def reads_for(buf):
        return [(_crop(0, 100), "GJ01AB1234", 0.9)]

    events = build_events(buffers, source_mode="file", model=MODEL, reads_for=reads_for)
    assert len(events) == 3
    assert {e.track_id for e in events} == {1, 2, 3}


def test_build_events_reraises_rather_than_skipping_a_bad_buffer():
    """A buffer that raises is re-raised, not swallowed. See EventBuildError."""
    good = _readable_buffer(track_id=1)[0]
    bad = CropBuffer(track_key=TrackKey(CAMERA, SESSION, 2))  # never noted -> no observed_at

    with pytest.raises(EventBuildError):
        build_events(
            [good, bad], source_mode="file", model=MODEL,
            reads_for=lambda b: [(_crop(0, 100), "GJ01AB1234", 0.9)],
        )


# ============================================================================ TIER 2
# The retry classification matrix -- exact, driven against a fake session, no threads.
# ============================================================================


class _Resp:
    """The parts of a requests.Response that _post_once reads."""

    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json body")  # exercises the unparseable-200 branch
        return self._body


class _Session:
    def __init__(self, response):
        self._response = response

    def post(self, url, json=None, timeout=None):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _classify(response):
    sink = HttpEventSink("http://ingest.invalid")
    sink._session = _Session(response)
    return sink._post_once({})


def test_201_is_accepted():
    assert _classify(_Resp(201)) == ("accepted", None)


def test_200_bare_is_a_duplicate():
    """200 is the duplicate response. NOT an error -- it is retry-safety working: a retry
    after an ambiguous failure lands as a duplicate instead of a second sighting."""
    assert _classify(_Resp(200)) == ("duplicate", None)


def test_200_with_status_accepted_body_is_accepted():
    """The one 200 that is not a duplicate. Some ingest builds answer a first-time write
    with 200 {"status":"accepted"} rather than 201, and treating that as a duplicate would
    undercount every such sighting."""
    assert _classify(_Resp(200, {"status": "accepted"})) == ("accepted", None)


def test_200_with_an_unparseable_body_is_still_a_duplicate():
    """The status is the contract; a body that fails to parse is not a reason to re-send an
    event the server already has."""
    assert _classify(_Resp(200, None)) == ("duplicate", None)


def test_200_with_some_other_body_is_a_duplicate():
    assert _classify(_Resp(200, {"status": "stored"})) == ("duplicate", None)


def test_422_is_rejected_permanently():
    """The payload is wrong; retrying sends the same wrong payload forever. Treating 422 as
    retryable is how a sink turns one bad event into an infinite loop against a live backend."""
    assert _classify(_Resp(422)) == ("rejected", None)


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses_retry(status):
    assert _classify(_Resp(status)) == ("retry", None)


def test_retryable_status_honours_retry_after():
    assert _classify(_Resp(503, headers={"Retry-After": "2"})) == ("retry", 2.0)


def test_retry_after_is_capped():
    """A proxy sending Retry-After: 3600 must not stall the sink for an hour with a queue
    filling behind it."""
    assert _classify(_Resp(503, headers={"Retry-After": "3600"})) == ("retry", 4.0)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_other_4xx_spools_rather_than_rejecting(status):
    """A wrong URL, missing auth or a proxy's 400 is not a validated rejection, so it must
    not land in rejected/ as if the payload were at fault. Spooling keeps the event and
    leaves the misconfiguration visible in the backlog."""
    assert _classify(_Resp(status)) == ("spool", None)


def test_every_exception_is_a_retry():
    """A sink that raises out of its own worker thread kills the thread and turns a transient
    network fault into permanent silent data loss."""
    assert _classify(ConnectionError("refused")) == ("retry", None)
    assert _classify(TimeoutError("slow")) == ("retry", None)


# ============================================================================ TIER 2
# Backoff and Retry-After -- pure functions, deterministic on purpose.
# ============================================================================


def test_backoff_is_exponential_and_capped():
    """No jitter, deliberately: one worker thread per camera and at most thirty cameras, so
    the thundering herd jitter protects against does not exist -- and a deterministic backoff
    makes a test assertion exact."""
    assert [_backoff(a) for a in range(1, 6)] == [0.5, 1.0, 2.0, 4.0, 4.0]


def test_retry_after_parses_caps_and_rejects_garbage():
    assert _retry_after(_Resp(503, headers={"Retry-After": "1.5"})) == 1.5
    assert _retry_after(_Resp(503, headers={"Retry-After": "9999"})) == 4.0
    assert _retry_after(_Resp(503, headers={"Retry-After": "soon"})) is None
    assert _retry_after(_Resp(503)) is None


# ============================================================================ TIER 2
# Spool mechanics -- no worker thread, no requests. Just the disk contract.
# ============================================================================


def _unopened_sink(spool_dir=None, **kw):
    """An HttpEventSink whose spool dirs exist but whose worker never started.

    _spool / _forget / replay_spool touch only disk and counters, so they are testable
    without requests or a thread. send() is not -- it guards on the worker.
    """
    return HttpEventSink("http://ingest.invalid", spool_dir=spool_dir, **kw)


def test_spool_filename_is_the_event_id_so_a_double_spool_overwrites(tmp_path):
    """The same one decision in builder.py that makes the POST retry safe makes the spool
    idempotent: spooling the same event twice leaves one file, not two."""
    sink = _unopened_sink(str(tmp_path))
    event = _event()
    sink._spool(event)
    sink._spool(event)
    files = glob.glob(str(tmp_path / "pending" / "*.json"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == f"{event.event_id}.json"
    assert sink.stats()["pending_spool"] == 1
    assert sink.spooled == 2  # writes, not distinct events -- the docstring is explicit


def test_spooled_file_is_a_whole_replayable_event(tmp_path):
    sink = _unopened_sink(str(tmp_path))
    event = _event()
    sink._spool(event)
    (path,) = glob.glob(str(tmp_path / "pending" / "*.json"))
    restored = EventEnvelope.from_json(open(path, encoding="utf-8").read())
    assert restored.event_id == event.event_id
    assert restored.validate() == []


def test_forget_removes_a_delivered_event_from_the_spool(tmp_path):
    sink = _unopened_sink(str(tmp_path))
    event = _event()
    sink._spool(event)
    assert sink.stats()["pending_spool"] == 1
    sink._forget(event.event_id)
    assert sink.stats()["pending_spool"] == 0
    assert glob.glob(str(tmp_path / "pending" / "*.json")) == []


def test_forget_of_an_absent_event_is_harmless(tmp_path):
    sink = _unopened_sink(str(tmp_path))
    sink._forget("does-not-exist")  # must not raise
    assert sink.stats()["pending_spool"] == 0


def test_no_spool_dir_means_undeliverables_are_counted_dropped(tmp_path):
    """dropped > 0 in stats() is the signal that a run's event count is incomplete. With no
    spool configured there is nowhere to keep the event, so it is dropped and never hidden."""
    sink = _unopened_sink(None)
    sink._spool(_event())
    assert sink.dropped == 1
    assert sink.spooled == 0


def test_spool_ceiling_drops_and_flags(tmp_path):
    """Past the ceiling a runaway cannot fill the disk; the overflow is counted as both
    dropped and spool_full so the two are distinguishable from an unconfigured spool."""
    sink = _unopened_sink(str(tmp_path), max_spool_files=2)
    for _ in range(5):
        sink._spool(_event())  # each a fresh event_id -> distinct file
    assert sink.stats()["pending_spool"] == 2
    assert sink.dropped == 3
    assert sink.spool_full_events == 3


def test_write_rejected_lands_under_rejected_not_pending(tmp_path):
    sink = _unopened_sink(str(tmp_path))
    event = _event()
    sink._write_rejected(event)
    assert glob.glob(str(tmp_path / "rejected" / "*.json")) == [
        str(tmp_path / "rejected" / f"{event.event_id}.json")
    ]
    assert sink.stats()["pending_spool"] == 0  # rejected/ is not the backlog


def test_replay_moves_an_unparseable_spool_file_to_rejected(tmp_path):
    """A spool file that will not parse is not recoverable by retrying, and leaving it in
    pending/ means every replay pass trips over it forever."""
    sink = _unopened_sink(str(tmp_path))
    bad = tmp_path / "pending" / "garbage.json"
    bad.write_text("{ not json", encoding="utf-8")
    sink.replay_spool()
    assert not bad.exists()
    assert (tmp_path / "rejected" / "garbage.json").exists()


def test_replay_reenqueues_parseable_files(tmp_path):
    """Re-enqueued rather than POSTed here, so replay goes through the same delivery path as
    everything else. The files stay put until delivery, which is what makes replay safe to
    interrupt."""
    sink = _unopened_sink(str(tmp_path))
    sink._spool(_event())
    sink._spool(_event())
    queued = sink.replay_spool()
    assert queued == 2
    assert sink.replayed == 2
    # Left in place -- a crash mid-replay must lose nothing.
    assert len(glob.glob(str(tmp_path / "pending" / "*.json"))) == 2


# ============================================================================ TIER 2
# send() guards -- the queue-full branch spills to disk in the caller.
# ============================================================================


def test_send_before_open_raises(tmp_path):
    sink = _unopened_sink(str(tmp_path))
    with pytest.raises(RuntimeError, match="open"):
        sink.send(_event())


def test_send_spills_to_disk_when_the_queue_is_full(tmp_path):
    """A full queue means the backend has been unreachable long enough to fill it, so the
    event goes straight to disk here rather than blocking the frame loop. Simulated by a
    queue that refuses every put -- the alternative is a race on a live worker."""

    class _FullQueue:
        def put_nowait(self, item):
            raise __import__("queue").Full

        def qsize(self):
            return 0

    sink = _unopened_sink(str(tmp_path))
    sink._thread = object()  # get past the open() guard without starting a worker
    sink._queue = _FullQueue()
    sink.send(_event())
    assert sink.submitted == 1  # counted on the way in, before the enqueue attempt
    assert sink.stats()["pending_spool"] == 1  # and it reached disk, not the void


# ============================================================================ TIER 2
# The accounting invariant -- nothing goes missing. Live loopback server.
# ============================================================================


class _Ingest:
    """A controllable stand-in for Mihir's ingest endpoint on a loopback port.

    `status` is an int, or a callable(nth_request:int) -> int for scripted sequences.
    Records every request so a test can assert what actually reached the wire.
    """

    def __init__(self, status=201, body=b"{}", retry_after=None):
        self.status = status
        self.body = body
        self.retry_after = retry_after
        self.received: list = []
        self.paths: list = []
        self.req_headers: list = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                outer.paths.append(self.path)
                outer.req_headers.append(dict(self.headers))
                try:
                    outer.received.append(json.loads(raw))
                except json.JSONDecodeError:
                    outer.received.append({"_unparseable": True})
                code = outer.status(len(outer.received)) if callable(outer.status) else outer.status
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                if outer.retry_after is not None:
                    self.send_header("Retry-After", str(outer.retry_after))
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *args):  # silence the default stderr logging
                pass

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _settle(sink, timeout=5.0):
    """Drive the sink to quiescence, where the accounting invariant is exact.

    The invariant reads short mid-flight -- send() bumps submitted before enqueuing, the
    worker dequeues before marking in_flight -- so a test must sample after the queue has
    drained and nothing is in flight, not at an arbitrary instant.
    """
    sink.flush(timeout_s=timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = sink.stats()
        if s["queue_depth"] == 0 and s["in_flight"] == 0:
            return s
        time.sleep(0.01)
    return sink.stats()


def _invariant(stats):
    return (
        stats["submitted"] + stats["replayed"]
        == stats["accepted"]
        + stats["duplicates"]
        + stats["rejected"]
        + stats["spooled"]
        + stats["dropped"]
        + stats["queue_depth"]
        + stats["in_flight"]
    )


@requires_requests
def test_accepted_run_delivers_everything_and_the_books_balance(tmp_path):
    with _Ingest(status=201) as ingest:
        sink = HttpEventSink(ingest.url, spool_dir=str(tmp_path))
        with sink:
            for _ in range(6):
                sink.send(_event())
            stats = _settle(sink)
    assert stats["accepted"] == 6
    assert stats["pending_spool"] == 0
    assert _invariant(stats)
    assert len(ingest.received) == 6
    assert all("event_id" in r for r in ingest.received)


@requires_requests
def test_posts_to_the_contract_path_not_a_guessed_one(tmp_path):
    with _Ingest(status=201) as ingest:
        with HttpEventSink(ingest.url, spool_dir=str(tmp_path)) as sink:
            sink.send(_event())
            _settle(sink)
    assert set(ingest.paths) == {DEFAULT_INGEST_PATH}


@requires_requests
def test_api_key_travels_as_a_header_never_in_the_payload(tmp_path):
    with _Ingest(status=201) as ingest:
        with HttpEventSink(ingest.url, api_key="scratch-token", spool_dir=str(tmp_path)) as sink:
            sink.send(_event())
            _settle(sink)
    assert any(h.get("X-API-Key") == "scratch-token" for h in ingest.req_headers)
    assert all("scratch-token" not in json.dumps(r) for r in ingest.received)


@requires_requests
def test_duplicate_run_counts_duplicates_and_delivers(tmp_path):
    """A backend that answers 200 for every write (the idempotency ledger already has these
    event_ids) is a delivered run, not a failure."""
    with _Ingest(status=200, body=b"{}") as ingest:
        with HttpEventSink(ingest.url, spool_dir=str(tmp_path)) as sink:
            for _ in range(4):
                sink.send(_event())
            stats = _settle(sink)
    assert stats["duplicates"] == 4
    assert stats["accepted"] == 0
    assert stats["delivered"] == 4
    assert _invariant(stats)


@requires_requests
def test_200_status_accepted_body_counts_as_accepted_end_to_end(tmp_path):
    """The same distinction as the unit test, proven through the worker thread."""
    with _Ingest(status=200, body=b'{"status":"accepted"}') as ingest:
        with HttpEventSink(ingest.url, spool_dir=str(tmp_path)) as sink:
            for _ in range(3):
                sink.send(_event())
            stats = _settle(sink)
    assert stats["accepted"] == 3
    assert stats["duplicates"] == 0
    assert _invariant(stats)


@requires_requests
def test_422_run_writes_to_rejected_and_never_retries(tmp_path):
    with _Ingest(status=422) as ingest:
        with HttpEventSink(ingest.url, spool_dir=str(tmp_path), max_attempts=4) as sink:
            sink.send(_event())
            stats = _settle(sink)
    assert stats["rejected"] == 1
    assert stats["pending_spool"] == 0  # rejected/, not pending/
    assert len(glob.glob(str(tmp_path / "rejected" / "*.json"))) == 1
    # One POST, not max_attempts of them: a rejection is permanent.
    assert len(ingest.received) == 1
    assert _invariant(stats)


@requires_requests
def test_a_dead_backend_spools_rather_than_losing_events(tmp_path):
    """Nothing listening on the port. Every event ends on disk, keyed by event_id, and the
    books still balance."""
    sink = HttpEventSink(
        "http://127.0.0.1:9", spool_dir=str(tmp_path), max_attempts=1, timeout_s=0.3
    )
    sink.open()
    ids = set()
    for _ in range(5):
        event = _event()
        ids.add(event.event_id)
        sink.send(event)
    stats = _settle(sink)
    sink.close()
    files = glob.glob(str(tmp_path / "pending" / "*.json"))
    assert stats["dropped"] == 0
    assert stats["spooled"] == 5
    assert {os.path.basename(f)[:-5] for f in files} == ids
    assert _invariant(stats)


@requires_requests
def test_close_spools_the_undrained_remainder_never_discards(tmp_path):
    """The timeout bounds shutdown, not delivery. Whatever is still queued when close()
    returns is on disk, so a slow backend costs no events."""
    sink = HttpEventSink(
        "http://127.0.0.1:9", spool_dir=str(tmp_path), max_attempts=1, timeout_s=0.3
    )
    sink.open()
    for _ in range(8):
        sink.send(_event())
    sink.close(timeout_s=3.0)
    stats = sink.stats()
    assert stats["pending_spool"] == 8
    assert stats["dropped"] == 0
    assert _invariant(stats)


@requires_requests
def test_a_later_run_replays_what_the_outage_stranded(tmp_path):
    """The spool survives the process. A first run against a dead backend strands events on
    disk; a second run against a live one replays them before anything new."""
    # First run: strand five events.
    dead = HttpEventSink(
        "http://127.0.0.1:9", spool_dir=str(tmp_path), max_attempts=1, timeout_s=0.3
    )
    dead.open()
    stranded = set()
    for _ in range(5):
        event = _event()
        stranded.add(event.event_id)
        dead.send(event)
    _settle(dead)
    dead.close()
    assert len(glob.glob(str(tmp_path / "pending" / "*.json"))) == 5

    # Second run: same spool dir, live backend, replay_on_start does the rest.
    with _Ingest(status=201) as ingest:
        live = HttpEventSink(ingest.url, spool_dir=str(tmp_path))
        with live:
            _settle(live)
        stats = live.stats()
    delivered = {r["event_id"] for r in ingest.received}
    assert stranded <= delivered
    assert glob.glob(str(tmp_path / "pending" / "*.json")) == []
    assert stats["replayed"] == 5
    assert _invariant(stats)


@requires_requests
def test_a_transient_503_then_201_recovers_within_the_retry_budget(tmp_path):
    """The first attempt fails retryably, the second succeeds. The event is delivered, not
    spooled, and the retry counter records the one retry."""
    seq = {"n": 0}

    def status(_count):
        seq["n"] += 1
        return 503 if seq["n"] == 1 else 201

    with _Ingest(status=status) as ingest:
        with HttpEventSink(ingest.url, spool_dir=str(tmp_path), max_attempts=4) as sink:
            sink.send(_event())
            stats = _settle(sink, timeout=8.0)
    assert stats["accepted"] == 1
    assert stats["retries"] >= 1
    assert stats["pending_spool"] == 0
    assert len(ingest.received) == 2  # the 503 and the 201
    assert _invariant(stats)


# ============================================================================ TIER 3
# FileEventSink and NullEventSink -- the sinks every measurement actually uses.
# ============================================================================


def test_file_sink_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    with FileEventSink(str(path)) as sink:
        sink.send(_event())
        sink.send(_event())
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(EventEnvelope.from_json(line).validate() == [] for line in lines)


def test_file_sink_stats_report_written_not_sent(tmp_path):
    with FileEventSink(str(tmp_path / "e.jsonl")) as sink:
        sink.send(_event())
        assert sink.stats() == {"path": str(tmp_path / "e.jsonl"), "written": 1}


def test_file_sink_send_before_open_raises(tmp_path):
    sink = FileEventSink(str(tmp_path / "e.jsonl"))
    with pytest.raises(RuntimeError, match="open"):
        sink.send(_event())


def test_file_sink_truncates_by_default_and_appends_on_request(tmp_path):
    path = tmp_path / "e.jsonl"
    with FileEventSink(str(path)) as s:
        s.send(_event())
    with FileEventSink(str(path)) as s:  # default: truncate
        s.send(_event())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with FileEventSink(str(path), append=True) as s:
        s.send(_event())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_file_sink_partial_file_is_parseable_up_to_the_last_newline(tmp_path):
    """One object per line, no wrapping array, so a run can be tailed while in progress."""
    path = tmp_path / "e.jsonl"
    sink = FileEventSink(str(path))
    sink.open()
    sink.send(_event())
    sink.flush()  # fsync, so a killed process leaves a complete line
    body = path.read_text(encoding="utf-8")
    assert body.endswith("\n")
    assert EventEnvelope.from_json(body.splitlines()[0]).validate() == []
    sink.close()


def test_null_sink_counts_and_discards():
    """For measuring the pipeline without a JSONL write or an HTTP round trip in the number."""
    sink = NullEventSink()
    with sink:
        sink.send(_event())
        sink.send(_event())
    assert sink.stats() == {"written": 2}


# ============================================================================ TIER 3
# safe_component -- ids become filenames, so a hostile id must not escape the root.
# ============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("cam04", "cam04"),
        ("../../etc/passwd", "_.._etc_passwd"),  # one component, no separators survive
        ("a/b\\c", "a_b_c"),
        ("..", "unknown"),  # collapses to empty, falls through to the fallback
        ("", "unknown"),
        (" .hidden", "hidden"),  # no leading dot: no hidden files, no traversal games
    ],
)
def test_safe_component_produces_one_harmless_path_segment(value, expected):
    assert safe_component(value) == expected


def test_safe_component_never_contains_a_separator():
    for hostile in ("../../../root", "a/b/c/d", "x\\y\\z", "..\\..\\win"):
        out = safe_component(hostile)
        assert "/" not in out and "\\" not in out
        assert not out.startswith(".")


# ============================================================================ TIER 3
# SnapshotWriter -- stage the best frame, write both stills once the event exists.
# ============================================================================


@requires_pil
def test_stage_frame_keeps_the_best_quality_frame(tmp_path):
    writer = SnapshotWriter(root=str(tmp_path))
    key = TrackKey(CAMERA, SESSION, 42)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    assert writer.stage_frame(key, frame, 0.5) is True
    assert writer.stage_frame(key, frame, 0.4) is False  # worse, not staged
    assert writer.stage_frame(key, frame, 0.9) is True   # better, replaces
    assert writer.has_staged(key)
    assert writer.frames_encoded == 2


@requires_pil
def test_commit_writes_both_stills_named_after_the_event(tmp_path):
    writer = SnapshotWriter(root=str(tmp_path))
    key = TrackKey(CAMERA, SESSION, 42)
    writer.stage_frame(key, np.zeros((40, 60, 3), np.uint8), 0.8)
    snap_uri, crop_uri = writer.commit(
        key, event_id="abc123", observed_at=OBSERVED_AT,
        plate_crop_bgr=np.zeros((24, 62, 3), np.uint8),
    )
    assert snap_uri is not None and snap_uri.endswith("abc123.jpg")
    assert crop_uri is not None and crop_uri.endswith("abc123_plate.jpg")
    # Partitioned <root>/<camera>/<YYYY-MM-DD>/, date sliced off the timestamp.
    assert (tmp_path / CAMERA / "2026-09-01" / "abc123.jpg").is_file()
    assert (tmp_path / CAMERA / "2026-09-01" / "abc123_plate.jpg").is_file()


@requires_pil
def test_commit_is_idempotent_because_the_name_is_the_event_id(tmp_path):
    """A re-run or retry overwrites rather than accumulating a second copy -- the same
    property that makes the POST retry and the disk spool idempotent."""
    writer = SnapshotWriter(root=str(tmp_path))
    key = TrackKey(CAMERA, SESSION, 42)
    for _ in range(2):
        writer.stage_frame(key, np.zeros((40, 60, 3), np.uint8), 0.8)
        writer.commit(key, event_id="dup", observed_at=OBSERVED_AT,
                      plate_crop_bgr=np.zeros((24, 62, 3), np.uint8))
    day = tmp_path / CAMERA / "2026-09-01"
    assert len(glob.glob(str(day / "dup*.jpg"))) == 2  # snapshot + plate, not four


@requires_pil
def test_commit_consumes_the_staged_frame(tmp_path):
    writer = SnapshotWriter(root=str(tmp_path))
    key = TrackKey(CAMERA, SESSION, 42)
    writer.stage_frame(key, np.zeros((40, 60, 3), np.uint8), 0.8)
    writer.commit(key, event_id="once", observed_at=OBSERVED_AT)
    assert not writer.has_staged(key)  # popped, so a second commit stages nothing


@requires_pil
def test_commit_without_a_staged_frame_still_writes_the_plate_crop(tmp_path):
    """Either URI may be None independently. No snapshot staged, but a plate crop was
    located -- the useful half is the one that costs a kilobyte."""
    writer = SnapshotWriter(root=str(tmp_path))
    key = TrackKey(CAMERA, SESSION, 42)
    snap_uri, crop_uri = writer.commit(
        key, event_id="platonly", observed_at=OBSERVED_AT,
        plate_crop_bgr=np.zeros((24, 62, 3), np.uint8),
    )
    assert snap_uri is None
    assert crop_uri is not None


def test_drop_session_forgets_every_frame_of_that_session():
    """Driven from the same place as EvidenceAccumulator.take_session and for the same
    reason: state keyed on a dead track is how the cross-session merge bug comes back -- here
    it would attach one vehicle's photograph to another's event."""
    writer = SnapshotWriter(root="unused", write_snapshot=True)
    a = TrackKey(CAMERA, SESSION, 1)
    b = TrackKey(CAMERA, "other-session-uuid", 2)
    # Stage via the internal dict to avoid needing PIL for a pure bookkeeping test.
    writer._staged[a] = (0.5, b"x")
    writer._staged[b] = (0.5, b"y")
    writer.drop_session(SESSION)
    assert not writer.has_staged(a)
    assert writer.has_staged(b)


def test_stage_frame_is_a_noop_when_snapshots_are_disabled():
    writer = SnapshotWriter(root="unused", write_snapshot=False)
    assert writer.stage_frame(TrackKey(CAMERA, SESSION, 42), np.zeros((40, 60, 3), np.uint8), 0.9) is False


def test_disabled_writer_commit_returns_two_nulls():
    writer = SnapshotWriter(root="unused", enabled=False)
    assert writer.commit(TrackKey(CAMERA, SESSION, 42), event_id="x") == (None, None)


# ============================================================================ TIER 3
# _encode_jpeg -- the fallbacks, and the aliasing trap the fast path must not fall into.
# ============================================================================


@requires_pil
def test_encode_downscales_the_snapshot_but_never_the_plate_crop(tmp_path):
    """The snapshot is for human confirmation and 1280 is wide enough; the plate crop is the
    evidence OCR was given and resampling it would invent detail in the one image whose value
    is that it shows exactly what was read."""
    writer = SnapshotWriter(root=str(tmp_path))
    big = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    snap = _PILImage.open(io.BytesIO(writer._encode_jpeg(big, max_width=DEFAULT_MAX_SNAPSHOT_WIDTH, quality=DEFAULT_JPEG_QUALITY)))
    assert snap.size == (1280, 720)

    plate = np.random.default_rng(1).integers(0, 255, (40, 120, 3), dtype=np.uint8)
    crop = _PILImage.open(io.BytesIO(writer._encode_jpeg(plate, max_width=0, quality=PLATE_CROP_QUALITY)))
    assert crop.size == (120, 40)  # unchanged


@requires_pil
def test_encode_rejects_non_image_arrays(tmp_path):
    writer = SnapshotWriter(root=str(tmp_path))
    assert writer._encode_jpeg(np.zeros((8, 8), np.uint8), max_width=0, quality=85) is None       # 2-D
    assert writer._encode_jpeg(np.zeros((0, 8, 3), np.uint8), max_width=0, quality=85) is None      # empty
    assert writer.encode_failures == 2


@requires_pil
def test_encode_clips_a_float_frame_rather_than_failing(tmp_path):
    writer = SnapshotWriter(root=str(tmp_path))
    assert writer._encode_jpeg(np.full((8, 8, 3), 300.0), max_width=0, quality=85) is not None


@requires_pil
def test_encode_does_not_alias_the_callers_frame(tmp_path):
    """The decoder reuses frame buffers; an encoded image sharing memory with the next frame
    would silently stage the wrong photograph. The fast path's rawmode conversion forces PIL
    to copy -- this asserts the copy really happened."""
    writer = SnapshotWriter(root=str(tmp_path))
    frame = np.full((8, 8, 3), 10, dtype=np.uint8)  # contiguous -> fast path
    first = writer._encode_jpeg(frame, max_width=0, quality=85)
    frame[:] = 200  # mutate the caller's buffer after encoding
    second = writer._encode_jpeg(frame, max_width=0, quality=85)
    assert first != second  # the first encode did not see the mutation


@requires_pil
def test_encode_handles_a_noncontiguous_frame_via_the_fallback(tmp_path):
    """A non-contiguous array cannot take the frombuffer fast path; the fromarray fallback
    must still produce a valid JPEG rather than a sheared one."""
    writer = SnapshotWriter(root=str(tmp_path))
    noncontig = np.random.default_rng(2).integers(0, 255, (16, 16, 3), dtype=np.uint8)[::2]
    assert not noncontig.flags["C_CONTIGUOUS"]
    out = writer._encode_jpeg(noncontig, max_width=0, quality=85)
    assert out is not None
    assert _PILImage.open(io.BytesIO(out)).size == (16, 8)


# ============================================================================ TIER 3
# NullSnapshotWriter and the factory.
# ============================================================================


def test_null_snapshot_writer_stages_and_writes_nothing():
    writer = NullSnapshotWriter()
    key = TrackKey(CAMERA, SESSION, 42)
    assert writer.stage_frame(key, np.zeros((4, 4, 3), np.uint8), 0.9) is False
    assert writer.has_staged(key) is False
    assert writer.commit(key, event_id="x", plate_crop_bgr=np.zeros((4, 4, 3), np.uint8)) == (None, None)
    assert writer.stats()["enabled"] is False


@pytest.mark.parametrize("config", [None, {}, {"enabled": False}])
def test_build_snapshot_writer_gives_the_null_writer_for_absent_or_off(config):
    assert isinstance(build_snapshot_writer(config), NullSnapshotWriter)


def test_build_snapshot_writer_builds_a_real_writer_when_enabled(tmp_path):
    writer = build_snapshot_writer({"enabled": True, "root": str(tmp_path), "max_width": 800})
    assert isinstance(writer, SnapshotWriter)
    assert writer.max_width == 800
    assert writer.enabled is True


# ============================================================================ TIER 3
# The storage arithmetic -- the USP, in bytes. Machine-dependent; marked as such.
# ============================================================================


@requires_pil
def test_the_snapshot_costs_far_more_than_the_plate_crop(tmp_path):
    """"We centralize intelligence, not video." The plate crop is the evidence an operator
    decides on and it is the cheap half; the snapshot is the expensive half. The exact bytes
    depend on the Pillow build and the frame content, so this asserts the *relationship* the
    design rests on, not a pinned number -- the pinned numbers live in the module docstring,
    which is re-measured by the next test with a wide tolerance.

    A realistic traffic still (structured, not white noise) so the JPEG sizes are in the
    right ballpark rather than at the incompressible extreme.
    """
    writer = SnapshotWriter(root=str(tmp_path))
    rng = np.random.default_rng(42)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # A few broad blocks and gradients -- compressible the way a real scene is.
    frame[:540] = (90, 90, 100)
    frame[540:] = (40, 40, 45)
    frame[:, :, 0] = np.linspace(0, 255, 1920, dtype=np.uint8)[None, :]
    frame += rng.integers(0, 12, frame.shape, dtype=np.uint8)

    snap = writer._encode_jpeg(frame, max_width=DEFAULT_MAX_SNAPSHOT_WIDTH, quality=DEFAULT_JPEG_QUALITY)
    plate = writer._encode_jpeg(np.zeros((24, 62, 3), np.uint8), max_width=0, quality=PLATE_CROP_QUALITY)
    assert len(snap) > 10 * len(plate)


@requires_pil
def test_a_1080p_snapshot_stays_well_under_a_100_kilobyte_budget(tmp_path):
    """The per-event storage claim that turns 30 cameras into ~150 GB/day rather than the
    terabytes a per-frame store would cost. A wide bound: the assertion is 'a downscaled q85
    still is tens of KB, not hundreds', which is what makes the arithmetic hold."""
    writer = SnapshotWriter(root=str(tmp_path))
    rng = np.random.default_rng(7)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:540] = (100, 110, 120)
    frame[540:] = (30, 35, 40)
    frame += rng.integers(0, 20, frame.shape, dtype=np.uint8)
    size = len(writer._encode_jpeg(frame, max_width=DEFAULT_MAX_SNAPSHOT_WIDTH, quality=DEFAULT_JPEG_QUALITY))
    assert 10_000 < size < 150_000


# ============================================================================
# Package surface.
# ============================================================================


def test_emit_exports_the_documented_names():
    import ai.emit as emit

    for name in (
        "build_event", "build_event_with_evidence", "build_events",
        "observations_from_buffer", "winning_crop", "EventBuildError",
        "HttpEventSink", "FileEventSink", "NullEventSink",
        "SnapshotWriter", "NullSnapshotWriter", "build_snapshot_writer", "safe_component",
    ):
        assert hasattr(emit, name), name
