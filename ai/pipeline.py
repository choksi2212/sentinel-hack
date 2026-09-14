"""The fourteen stages, wired once. One FrameEnvelope in, zero or more EventEnvelopes out.

    pipeline = Pipeline(
        camera_id="cam04", source_mode="file", model=provenance,
        detector=detector, tracker=tracker, gate=VehicleGate(),
        plate_detector=plates, ocr=ocr, snapshots=snapshots, counters=counters,
    )
    pipeline.load()
    for envelope in source:
        for event in pipeline.process_frame(envelope):
            sink.send(event)
    for event in pipeline.flush("eof"):
        sink.send(event)

Every stage is injected. Nothing here constructs a model, opens a stream, or writes to the
network, and that is the whole design: this file is the only place that knows the order of
the stages, and it can be run end to end against stubs with no GPU, no camera and no
database. The worker owns the source, the sink and the process; this owns the sequence.

Four decisions are load-bearing enough to state before the code.

**The gate decides whether to spend plate compute, not whether the vehicle exists.**
ai/quality/gate.py rejects a vehicle "before spending plate-detection compute on it", so
note_track runs for every track on every sampled frame and only gate-passing tracks reach
the plate detector. A vehicle that never passes the gate still accumulates a buffer, still
gets a snapshot, and still emits an event with plate: null -- which is correct, because it
was really there and really could not be read. Skipping note_track for gated vehicles would
silently delete them from the vehicle count, and a count that is quietly low is the hardest
kind of wrong to notice.

**OCR is deferred to the top-K crops at track flush, not run per frame.** A vehicle in
frame for three seconds at the 100 ms sampling interval offers thirty plate crops; twenty-six
of them contribute nothing but latency. So the crop is cut when it is found, carried in the
buffer, and read when the track finishes -- four reads per vehicle instead of thirty. The
cost of that choice is that the frame is gone by read time, which is why ai/ocr/base.py has
cut_crop/read_crop and a FrameRef: a backend that needs frame identity has to be handed it
back, and a deferred oracle read with no FrameRef resolves against the wrong frame and
returns nothing at all.

**A track that was never confidently a vehicle and never yielded a plate is not emitted.**
ByteTrack will happily follow a 0.06-confidence blob for five frames -- a shadow, a bin bag,
half a hoarding -- and every one of those becomes a row in a police database that says a
vehicle was at a place at a time. The gate already refuses to spend plate compute below 0.35
confidence; MIN_REPORTABLE_VEHICLE_CONFIDENCE applies the same floor to the *claim*. A track
that produced a plate crop is exempt: something with a readable plate on it is a vehicle
regardless of what the detector scored it.

**Events are returned, not sent.** process_frame hands back a list. The retry, the spool and
the 201-versus-200 belong to ai/emit/http_sink.py and the worker, and keeping them out of
here is what makes a full-pipeline test a function call instead of a fixture with a socket.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ai.contracts.enums import is_live
from ai.contracts.event import EventEnvelope, ModelProvenance
from ai.contracts.frame import FrameEnvelope
from ai.contracts.ids import TrackKey
from ai.contracts.stages import FusedPlate, PlateCandidate, PlateObservation, TrackResult
from ai.contracts.timebase import iso_from_datetime, utc_now_iso
from ai.dedup.key import SightingDeduper, dedupe_key_for
from ai.emit.builder import build_event_with_evidence, observations_from_buffer
from ai.emit.snapshot import NullSnapshotWriter
from ai.fusion.accumulator import CropBuffer, EvidenceAccumulator, TrackCrop
from ai.fusion.consensus import fuse_observations
from ai.logging_setup import get_logger
from ai.metrics import RunCounters
from ai.ocr.base import FrameRef
from ai.quality.gate import MIN_DETECTOR_CONFIDENCE, VehicleGate
from ai.quality.score import plate_quality

log = get_logger("pipeline")

# Below this peak detector confidence, a track with no plate evidence is dropped rather
# than emitted. Deliberately the same value as the gate's own confidence floor: two
# different numbers here would mean a band of tracks the gate refuses to spend compute on
# but the pipeline still reports as vehicles, which is the worst of both -- an unverifiable
# claim made from evidence nobody looked at.
#
# Under ByteTrack's default thresholds this never fires, and saying so is more useful than
# implying otherwise: only a detection at or above high_threshold (0.5) can start a track,
# so every track ByteTrack produces already peaked above 0.5. The rule is a guard against a
# *configuration*, not against the tracker -- config/*.yaml can set high_threshold to 0.2,
# ai/track/stub.py's IOUTracker has no such threshold at all, and either would otherwise put
# 0.1-confidence blobs into a police database as vehicle sightings. It is checked here rather
# than trusted to the tracker because the claim is made here.
MIN_REPORTABLE_VEHICLE_CONFIDENCE = MIN_DETECTOR_CONFIDENCE

# How a frame is ranked as a track's snapshot. Two populations on one scale, which needs
# saying: a frame where a plate was found is ranked 1.0 + plate_quality, and a frame with no
# plate is ranked by how much of the frame the vehicle fills. Any plate frame therefore beats
# any plate-less frame, and within each group the better one wins. Mixing the two raw scores
# on one scale would let a big blurry vehicle outrank the one frame whose plate was legible,
# and the snapshot would show a vehicle whose plate the event claims to have read from a
# frame where it was never located.
_PLATE_FRAME_RANK_BASE = 1.0

# Reasons a flush can happen, for the log line and for the counter. Not an enum in
# ai/contracts/enums.py: EndReason there is the *source's* vocabulary and it is on the wire,
# whereas these are this module's internal bookkeeping and adding to a locked enum to
# describe an internal event is how a locked enum stops being locked.
FLUSH_REASONS = ("eof", "session_end", "shutdown", "manual")


@dataclass
class FrameOutcome:
    """What one frame produced. Returned alongside the events for the worker's log line.

    Exists so the worker can say "frame 412: 6 tracks, 4 gated, 2 plates, 1 event" without
    reaching into five stages' stats() and subtracting. A per-frame line that has to be
    reconstructed from cumulative counters is a line nobody writes.
    """

    frame_index: int
    pts_ms: int
    tracks: int = 0
    gate_passed: int = 0
    gate_rejected: dict[str, int] = field(default_factory=dict)
    plates_found: int = 0
    crops_kept: int = 0
    tracks_finished: int = 0
    events: int = 0
    events_suppressed: int = 0
    tracks_dropped_unreportable: int = 0
    latency_ms: float = 0.0

    def summary(self) -> str:
        parts = [
            f"frame {self.frame_index} pts={self.pts_ms}ms",
            f"{self.tracks} track(s)",
            f"{self.gate_passed} gated",
            f"{self.plates_found} plate(s)",
        ]
        if self.tracks_finished:
            parts.append(f"{self.tracks_finished} finished")
        if self.events:
            parts.append(f"{self.events} event(s)")
        if self.events_suppressed:
            parts.append(f"{self.events_suppressed} deduped")
        if self.tracks_dropped_unreportable:
            parts.append(f"{self.tracks_dropped_unreportable} unreportable")
        parts.append(f"{self.latency_ms:.1f}ms")
        return ", ".join(parts)


class Pipeline:
    """Stages 2 through 13, in order, for one camera.

    One instance per camera. The stages it holds are stateful in ways that are scoped to a
    camera -- the tracker's id counter, the gate's per-track history, the accumulator's
    buffers -- so sharing one across cameras would merge two junctions into one vehicle
    stream. Thirty cameras means thirty of these, which is also how the worker parallelizes.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        source_mode: str,
        model: ModelProvenance,
        detector: Any,
        tracker: Any,
        plate_detector: Any,
        ocr: Any,
        gate: Optional[VehicleGate] = None,
        accumulator: Optional[EvidenceAccumulator] = None,
        deduper: Optional[SightingDeduper] = None,
        snapshots: Any = None,
        counters: Optional[RunCounters] = None,
        watchlist: Optional[Callable[[str], bool]] = None,
        min_reportable_confidence: float = MIN_REPORTABLE_VEHICLE_CONFIDENCE,
        replay_anchor: Optional[datetime] = None,
    ) -> None:
        if not camera_id:
            raise ValueError("Pipeline needs a camera_id; every track and event is scoped to one")
        self.camera_id = camera_id
        self.source_mode = source_mode
        self.model = model

        self.detector = detector
        self.tracker = tracker
        self.plate_detector = plate_detector
        self.ocr = ocr
        self.gate = gate if gate is not None else VehicleGate()
        self.accumulator = accumulator if accumulator is not None else EvidenceAccumulator()
        self.deduper = deduper if deduper is not None else SightingDeduper()
        self.snapshots = snapshots if snapshots is not None else NullSnapshotWriter()
        self.counters = counters if counters is not None else RunCounters()
        self.watchlist = watchlist
        self.min_reportable_confidence = float(min_reportable_confidence)

        # Peak detector confidence per track, for the reportability rule. Kept here rather
        # than on CropBuffer because the buffer already records the confidence of the
        # *largest* box -- note_track's snapshot tie-break -- and that is a different number:
        # a vehicle scored 0.9 at ten metres and 0.2 filling the frame reports 0.2, which
        # would fail a floor it comfortably cleared.
        self._peak_confidence: dict[TrackKey, float] = {}

        self._session_id: Optional[str] = None
        self._last_pts_ms: int = 0
        self._frames = 0

        # Per-stage milliseconds for the frame currently being processed, summed across
        # however many times a stage runs within it -- one detect call, but four OCR reads.
        # Cleared at the top of process_frame and handed to the counters at the bottom, only
        # if observe_frame says the frame counted. Submitting as we go would seem simpler and
        # would be subtly wrong: observe_stage applies the warm-up rule against
        # frames_sampled, which this frame has not been added to yet, so frame 3's stages
        # would be discarded while frame 3's total latency was kept and the stage table's n
        # would sit one below the frame count for no visible reason.
        #
        # An instance attribute rather than a parameter threaded through five methods. Its
        # lifetime is one frame and one pipeline drives one camera on one thread, which is the
        # only reason that is safe.
        self._frame_ms_by_stage: dict[str, float] = {}

        # Offline runs have no wallclock, and an event without observed_at is a 422. See
        # _observed_at for why this is an anchored replay timeline rather than time.now().
        self._replay_base = (replay_anchor or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self._replay_session_offset_ms = 0

        self.events_emitted = 0
        self.events_suppressed = 0
        self.tracks_dropped_unreportable = 0
        self.flushes: dict[str, int] = {}
        self.last_outcome: Optional[FrameOutcome] = None

    # ------------------------------------------------------------------------ lifecycle

    def _mark(self, stage: str, started: float) -> float:
        """Add this stage's elapsed time to the current frame and return a fresh mark.

        Returns the new perf_counter reading so stages can be chained without a second call
        per boundary: `t = self._mark("detect", t)`. perf_counter rather than monotonic --
        monotonic on Windows is GetTickCount64 at roughly 15.6 ms granularity, so every stage
        that takes under a tick would measure as exactly 0.0 and the table would report a
        pipeline with no cost at all.
        """
        now = time.perf_counter()
        stage_ms = (now - started) * 1000.0
        self._frame_ms_by_stage[stage] = self._frame_ms_by_stage.get(stage, 0.0) + stage_ms
        return now

    def load(self) -> None:
        """Load every model. Raises whatever the backend raises, unwrapped.

        Sequential rather than concurrent: three models loading at once on a 12 GB card is
        how a run dies with a CUDA OOM whose traceback names whichever one lost the race,
        rather than the one that was too big.
        """
        for name, stage in (
            ("detector", self.detector),
            ("plate_detector", self.plate_detector),
            ("ocr", self.ocr),
        ):
            loader = getattr(stage, "load", None)
            if callable(loader):
                loader()
                log.info("loaded %s: %s", name, getattr(stage, "model_name", type(stage).__name__))

    def close(self) -> None:
        """Release the models. Does not flush -- call flush("shutdown") first if you mean to.

        Deliberately not flushing: closing is a resource operation and flushing produces
        events, and a close() that quietly emitted a dozen sightings would put them on the
        wire after the worker had already reported its totals.
        """
        for stage in (self.detector, self.plate_detector, self.ocr):
            closer = getattr(stage, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - a failed close must not mask the real error
                    log.warning("close failed for %s", type(stage).__name__, exc_info=True)

    # --------------------------------------------------------------------------- per frame

    def process_frame(self, envelope: FrameEnvelope) -> list[EventEnvelope]:
        """Run one frame through stages 3 to 13. Returns the events it completed.

        The events returned are usually not about the vehicle in this frame -- they are for
        tracks that went idle by this frame's PTS. That lag is the temporal fusion working:
        an event is emitted when the evidence is complete, not when the vehicle is first
        seen.
        """
        started = time.perf_counter()
        outcome = FrameOutcome(frame_index=envelope.frame_index, pts_ms=envelope.pts_ms)
        events: list[EventEnvelope] = []
        self._frame_ms_by_stage.clear()

        if envelope.camera_id != self.camera_id:
            raise ValueError(
                f"pipeline for {self.camera_id} was handed a frame from "
                f"{envelope.camera_id}. One pipeline per camera: sharing one merges two "
                f"junctions into a single vehicle stream and the tracker cannot tell."
            )

        events.extend(self._handle_session(envelope))

        self._frames += 1
        self._last_pts_ms = envelope.pts_ms
        observed_at = self._observed_at(envelope)

        mark = time.perf_counter()
        detections = self.detector.detect_envelope(envelope)
        mark = self._mark("detect", mark)
        tracks: list[TrackResult] = self.tracker.update(
            detections, frame_index=envelope.frame_index, pts_ms=envelope.pts_ms
        )
        mark = self._mark("track", mark)
        outcome.tracks = len(tracks)

        gated: list[TrackResult] = []
        for track in tracks:
            buffer = self.accumulator.note_track(track, observed_at)
            if buffer.frames_seen == 1:
                self.counters.tracks_started += 1
            key = track.track_key
            self._peak_confidence[key] = max(
                self._peak_confidence.get(key, 0.0), float(track.confidence)
            )

            decision = self.gate.check(track, envelope.width, envelope.height)
            if decision.passed:
                gated.append(track)
            else:
                reason = decision.reason or "unknown"
                outcome.gate_rejected[reason] = outcome.gate_rejected.get(reason, 0) + 1
        outcome.gate_passed = len(gated)
        mark = self._mark("gate", mark)

        candidates: dict[int, PlateCandidate] = {}
        if gated:
            candidates = self.plate_detector.detect_plates_envelope(envelope, gated)
        outcome.plates_found = len(candidates)
        mark = self._mark("plate_detect", mark)

        by_id = {track.track_id: track for track in gated}
        plate_qualities: dict[int, float] = {}
        for track_id, candidate in candidates.items():
            track = by_id.get(track_id)
            if track is None:
                # A plate box for a track that was not offered. Not survivable as a silent
                # skip: the candidate carries no identity of its own, so attaching it to
                # anything would be a guess about which vehicle it belongs to.
                raise ValueError(
                    f"plate detector returned a candidate for track_id {track_id}, which "
                    f"was not among the {len(gated)} track(s) it was given"
                )
            quality = self._offer_crop(envelope, track, candidate, observed_at)
            if quality is not None:
                plate_qualities[track_id] = quality
                outcome.crops_kept += 1
        mark = self._mark("crop_quality", mark)

        self._stage_snapshots(envelope, tracks, plate_qualities)
        mark = self._mark("snapshot", mark)

        finished = self.accumulator.take_finished(envelope.pts_ms)
        outcome.tracks_finished = len(finished)
        events.extend(self._finalize(finished, outcome))
        # No umbrella mark around _finalize: it records ocr, fuse and emit itself, and a
        # parent stage covering the same milliseconds would double-count them and break the
        # one property that makes the table readable -- that the columns add up.

        outcome.latency_ms = (time.perf_counter() - started) * 1000.0
        # observe_frame increments frames_sampled. frames_seen is deliberately NOT touched
        # here: it counts frames the source *decoded*, and this stage only ever sees the ones
        # the sampler emitted. The worker sets it from the source's own sampler stats at the
        # end of the run. Incrementing it per frame here would make the two counters equal
        # and quietly report an emit rate of 100% for a stage designed to drop nine frames in
        # ten.
        counted = self.counters.observe_frame(
            latency_ms=outcome.latency_ms, pts_ms=envelope.pts_ms
        )
        if counted:
            # Gated on the same return value the frame total is gated on, so a stage sample
            # and a frame sample always describe the same set of frames. The stage columns
            # then sum to roughly the frame p50, which is what makes the table worth reading:
            # it says which stage to spend the next hour on.
            for stage, stage_ms in self._frame_ms_by_stage.items():
                self.counters.observe_stage(stage, stage_ms)
        self.last_outcome = outcome
        return events

    def flush(self, reason: str = "manual") -> list[EventEnvelope]:
        """Finalize every open track. Call at EOF and at shutdown.

        Not optional and not a cleanup nicety: at the end of a clip every vehicle still in
        frame has an open buffer, and without this they are simply never reported. On a
        30-second demo clip that is every vehicle in the last three seconds.
        """
        if reason not in FLUSH_REASONS:
            raise ValueError(f"flush reason {reason!r} is not one of {list(FLUSH_REASONS)}")
        self.flushes[reason] = self.flushes.get(reason, 0) + 1
        buffers = self.accumulator.take_all()
        if not buffers:
            return []
        log.info("flush(%s): finalizing %d open track(s)", reason, len(buffers))
        outcome = FrameOutcome(frame_index=-1, pts_ms=self._last_pts_ms)
        events = self._finalize(buffers, outcome)
        # _finalize marked ocr/fuse/emit into _frame_ms_by_stage, and nothing submits them:
        # this is not a frame, and folding a burst of thirty tracks finalized at once into the
        # per-frame percentiles would put a p95 in the run summary that no frame ever took.
        # process_frame clears the dict, so they cannot leak into the next one either.
        self._frame_ms_by_stage.clear()
        return events

    # -------------------------------------------------------------------------- internals

    def _handle_session(self, envelope: FrameEnvelope) -> list[EventEnvelope]:
        """Detect a session change and close out the old one. Contracts section 1.2.

        Everything keyed on a session is dropped here, and the order matters: the old
        session's buffers are finalized *first*, because those vehicles really did pass the
        camera and their evidence is complete. Only then is the state cleared, so nothing
        from the new session can be appended to a buffer belonging to the old one -- which is
        the track-merge bug, and it produces a vehicle crossing Ahmedabad in four seconds.
        """
        session = envelope.stream_session_id
        if session == self._session_id:
            return []

        events: list[EventEnvelope] = []
        previous = self._session_id
        if previous is not None:
            log.info("session change on %s: %s -> %s", self.camera_id, previous, session)
            buffers = self.accumulator.take_session(previous)
            if buffers:
                outcome = FrameOutcome(frame_index=envelope.frame_index, pts_ms=self._last_pts_ms)
                events.extend(self._finalize(buffers, outcome))
                self.flushes["session_end"] = self.flushes.get("session_end", 0) + 1
            self.gate.flush_session(previous)
            self.snapshots.drop_session(previous)
            for key in [k for k in self._peak_confidence if k.stream_session_id == previous]:
                del self._peak_confidence[key]
            # The replay clock advances past the old session so that offline observed_at
            # stays monotonic when PTS restarts at 0 after a reconnect. Without this a
            # reconnect rewinds every subsequent event's timestamp, and a journey
            # reconstructed from them has the vehicle arriving before it left.
            self._replay_session_offset_ms += self._last_pts_ms + 1

        self.tracker.reset(stream_session_id=session)
        self._session_id = session
        self._last_pts_ms = 0
        self.counters.sessions_started += 1
        return events

    def _observed_at(self, envelope: FrameEnvelope) -> str:
        """When the system saw this frame. Contracts section 2.1.

        Live modes carry a real wallclock and it is used unchanged.

        Offline modes carry None -- ai/contracts/timebase.py refuses to stamp arrival time
        onto recorded footage -- but observed_at is required on every event, so one has to be
        derived. It is an anchor plus the frame's own PTS, not time.now(), and the difference
        is not cosmetic: a five-minute clip replayed in twenty seconds would give every event
        a timestamp inside one twenty-second span, so the deduper's ten-second window would
        cover the whole clip and suppress the second sighting of a vehicle that genuinely
        passed twice. Anchoring on PTS means the window means ten seconds *of footage*, which
        is what it means on a live camera.

        The anchor is a constructor argument so a benchmark can pin it and get byte-identical
        events across two runs of the same clip.
        """
        if envelope.wallclock_utc:
            return envelope.wallclock_utc
        if is_live(self.source_mode):
            # A live source with no wallclock is a bug in the adapter, not a case to paper
            # over: the timestamp is the one field a live event cannot be reconstructed
            # without, so it is stamped now and said out loud.
            log.warning(
                "live frame %d on %s carried no wallclock_utc; stamping arrival time",
                envelope.frame_index,
                self.camera_id,
            )
            return utc_now_iso()
        offset = self._replay_session_offset_ms + max(0, envelope.pts_ms)
        return iso_from_datetime(self._replay_base + timedelta(milliseconds=offset))

    def _offer_crop(
        self,
        envelope: FrameEnvelope,
        track: TrackResult,
        candidate: PlateCandidate,
        observed_at: str,
    ) -> Optional[float]:
        """Cut, score and buffer one plate crop. Returns its quality, or None if not kept.

        The quality is returned rather than kept private because the snapshot ranking needs
        the same number -- see _stage_snapshots. Scoring it twice would not merely cost a
        Laplacian variance per frame; the second call would have to re-cut the crop, and a
        second cut with different padding is a second, quietly different score.

        The crop is copied. cut_crop returns a view, and a view holds its whole base frame
        alive -- four views across eight open tracks would pin eight 1920x1080 frames, 48 MB
        that no profiler attributes to this line and that looks like a decoder leak.
        """
        crop = self.ocr.cut_crop(envelope.frame_bgr, candidate)
        if crop is None or crop.size == 0:
            return None

        quality = plate_quality(
            crop,
            # The plate's width in the SCENE, not the crop's own width. The crop is padded
            # and may later be upscaled; scoring resolution on the crop would credit
            # interpolation with detail it cannot contain, and it is also the number the
            # width buckets in ai/metrics.py mean.
            plate_width_px=candidate.plate_width_px,
            detector_confidence=candidate.detector_confidence,
        )

        self.counters.crops_offered += 1
        kept = self.accumulator.offer_crop(
            track.track_key,
            TrackCrop(
                quality=quality,
                crop_bgr=np.ascontiguousarray(crop),
                candidate=candidate,
                frame_index=envelope.frame_index,
                pts_ms=envelope.pts_ms,
                observed_at=observed_at,
                vehicle_bbox_xyxy=track.bbox_xyxy,
                vehicle_class=track.class_name,
                vehicle_confidence=track.confidence,
            ),
        )
        if not kept:
            self.counters.crops_rejected_quality += 1
        # The quality is returned whether or not the buffer kept the crop. Rejection means
        # only that four better crops of this vehicle already exist; this frame still had a
        # plate in it, which is what the snapshot ranking is asking about.
        return quality

    def _stage_snapshots(
        self,
        envelope: FrameEnvelope,
        tracks: Sequence[TrackResult],
        plate_qualities: dict[int, float],
    ) -> None:
        """Offer this frame as each track's evidence still, ranked. See _PLATE_FRAME_RANK_BASE.

        Every track, not only the gated ones. A vehicle the gate refused still emits an event
        with plate: null, and an event with no photograph is not evidence of anything -- an
        operator cannot confirm or dismiss it, which makes it noise in a queue.
        """
        if isinstance(self.snapshots, NullSnapshotWriter):
            return
        frame_area = max(1, envelope.width * envelope.height)
        for track in tracks:
            quality = plate_qualities.get(track.track_id)
            if quality is not None:
                rank = _PLATE_FRAME_RANK_BASE + quality
            else:
                rank = min(1.0, (max(0, track.width) * max(0, track.height)) / frame_area)
            self.snapshots.stage_frame(track.track_key, envelope.frame_bgr, rank)

    def _finalize(
        self, buffers: Iterable[CropBuffer], outcome: FrameOutcome
    ) -> list[EventEnvelope]:
        """Stages 8 to 13 for each finished track: OCR, fuse, normalize, dedup, emit."""
        events: list[EventEnvelope] = []
        for buffer in buffers:
            self.counters.tracks_completed += 1
            key = buffer.track_key
            peak = self._peak_confidence.pop(key, 0.0)

            if not self._reportable(buffer, peak):
                self.tracks_dropped_unreportable += 1
                outcome.tracks_dropped_unreportable += 1
                self.snapshots.drop(key)
                log.debug(
                    "dropped %s: peak confidence %.2f < %.2f and no plate evidence",
                    key,
                    peak,
                    self.min_reportable_confidence,
                )
                continue

            mark = time.perf_counter()
            observations = self._read_track(buffer)
            mark = self._mark("ocr", mark)
            fused = fuse_observations(observations) if observations else None
            mark = self._mark("fuse", mark)

            if buffer.has_plate_evidence:
                self.counters.tracks_with_plate_crops += 1
                if fused is None or not fused.normalized:
                    # Located, not read. Invisible in the event -- see _plate_block in
                    # ai/emit/builder.py -- so it is counted here or nowhere.
                    self.counters.plate_located_no_read += 1

            normalized = fused.normalized if fused is not None else None
            if not self._should_emit(buffer, fused, normalized):
                self.events_suppressed += 1
                outcome.events_suppressed += 1
                self.snapshots.drop(key)
                self._mark("emit", mark)
                continue

            hit = bool(self.watchlist(normalized)) if (self.watchlist and normalized) else False
            event = build_event_with_evidence(
                buffer,
                source_mode=self.source_mode,
                model=self.model,
                snapshots=self.snapshots,
                observations=observations,
                fused=fused,
                exact_watchlist_hit=hit,
            )
            events.append(event)
            self.events_emitted += 1
            outcome.events += 1
            self.counters.events_built += 1
            if event.plate is not None and event.plate.normalized:
                self.counters.events_with_plate += 1
            else:
                self.counters.events_plate_null += 1
            # Dedup, the watchlist lookup and the JPEG encode behind build_event_with_evidence
            # all land here. The encode is the expensive one -- around 10 ms a still -- and it
            # is worth seeing separately from OCR, because "the pipeline is slow" has been the
            # snapshot writer twice as often as it has been a model.
            self._mark("emit", mark)
        return events

    def _reportable(self, buffer: CropBuffer, peak_confidence: float) -> bool:
        """Whether this track may be published as a vehicle sighting.

        A plate crop is sufficient on its own: whatever the detector scored it, a thing with
        a located plate on it is a vehicle. Without one, the peak detector confidence has to
        clear the same floor the gate uses -- see MIN_REPORTABLE_VEHICLE_CONFIDENCE.

        The plate exemption only actually fires when the gate is configured more permissively
        than this floor, and that is worth knowing rather than discovering. With both at 0.35
        a sub-floor track is refused plate compute by the gate, so it can never acquire the
        evidence that would exempt it, and the first branch is unreachable. It becomes
        reachable under `VehicleGate(min_confidence=0.1)` with this floor left at 0.35 --
        "look at everything, report only what you can stand behind" -- which is a reasonable
        thing for a config to say and the reason the branch is written this way round.
        """
        if buffer.has_plate_evidence:
            return True
        return peak_confidence >= self.min_reportable_confidence

    def _read_track(self, buffer: CropBuffer) -> list[PlateObservation]:
        """Stage 8, deferred: OCR the track's kept crops, best first.

        Every kept crop, not just the single best. Reading only the best one would give
        temporal fusion a single observation to fuse, which makes stages 7 and 9 decorative
        -- and the worked example in ai/fusion/consensus.py is precisely a case where the
        highest-quality frame is not the one that reads correctly.
        """
        reads: list[tuple[TrackCrop, str, float]] = []
        for crop in buffer.crops:
            self.counters.ocr_attempts += 1
            read = self.ocr.read_crop(
                crop.crop_bgr,
                crop.candidate,
                frame_ref=FrameRef(
                    camera_id=buffer.track_key.camera_id,
                    stream_session_id=buffer.track_key.stream_session_id,
                    frame_index=crop.frame_index,
                    pts_ms=crop.pts_ms,
                ),
            )
            if read is None or not read.text:
                continue
            reads.append((crop, read.text, read.confidence))
        return observations_from_buffer(buffer, reads)

    def _should_emit(
        self,
        buffer: CropBuffer,
        fused: Optional[FusedPlate],
        normalized: Optional[str],
    ) -> bool:
        """Stage 12. False means a better version of this sighting already went out.

        The evidence weight handed to the deduper is the winning observation's own
        ocr_confidence x image_quality -- the same product fuse() weighted by, so "keep the
        best evidence" means the same thing on both sides of the boundary. Not the fused
        confidence, which is a share of total evidence: a track with one mediocre read scores
        a share of 1.0 and would outrank a track with four excellent agreeing reads.
        """
        observed_at = buffer.last_observed_at or buffer.first_observed_at
        if not observed_at:
            return True
        weight = 0.0
        if fused is not None and fused.best_observation is not None:
            weight = fused.best_observation.fusion_weight
        return self.deduper.should_emit(
            dedupe_key_for(buffer.track_key, normalized), observed_at, weight
        )

    # ------------------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        """Everything the worker prints at the end, from every stage it holds."""
        return {
            "camera_id": self.camera_id,
            "source_mode": self.source_mode,
            "session_id": self._session_id,
            "frames_processed": self._frames,
            "events_emitted": self.events_emitted,
            "events_suppressed": self.events_suppressed,
            "tracks_dropped_unreportable": self.tracks_dropped_unreportable,
            "open_tracks": len(self.accumulator),
            "flushes": dict(self.flushes),
            "detector": _stage_stats(self.detector),
            "tracker": _stage_stats(self.tracker),
            "gate": self.gate.stats(),
            "plate": _stage_stats(self.plate_detector),
            "ocr": _stage_stats(self.ocr),
            "accumulator": self.accumulator.stats(),
            "dedup": self.deduper.stats(),
            "snapshots": self.snapshots.stats(),
            "counters": self.counters.to_dict(),
        }


def _stage_stats(stage: Any) -> dict[str, Any]:
    """A stage's stats(), or its class name when it has none.

    Returning something rather than raising, because stats() is what gets called on the way
    out of a run that is already failing, and a KeyError inside a summary is how the actual
    error gets lost.
    """
    getter = getattr(stage, "stats", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:  # noqa: BLE001 - a broken counter must not eat the run's summary
            log.warning("stats() failed for %s", type(stage).__name__, exc_info=True)
    return {"stage": type(stage).__name__}
