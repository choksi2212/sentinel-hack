"""The MediaSource protocol and the shared source machinery.

Five adapters converge on one FrameEnvelope. Everything that is the same for all
five -- session identity, frame indexing, PTS validation, sampling,
discontinuity detection, reconnect -- lives here, so that an adapter is only
responsible for getting pixels and a timestamp out of one specific kind of
input.

That split is what makes the offline-to-live swap a configuration change. If an
adapter starts making decisions about sessions or sampling, the adapters drift
apart and the invariant quietly stops holding.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Deque, Iterator, Optional, Protocol

import numpy as np

from ai.contracts.frame import FrameEnvelope
from ai.contracts.ids import new_session_id, require_camera_id
from ai.contracts.timebase import utc_now_iso, wallclock_for_source
from ai.media.backoff import ReconnectPolicy
from ai.media.discontinuity import DiscontinuityDetector
from ai.media.pacing import ReplayPacer
from ai.media.pts import PtsAction, PtsValidator
from ai.media.sampler import TARGET_INTERVAL_MS, PtsSampler


# COPIED FROM CANONICAL CONTRACTS -- DO NOT EDIT HERE (Contracts section 2.3).
class MediaSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[FrameEnvelope]: ...   # None = end of stream
    def close(self) -> None: ...
    @property
    def session_id(self) -> str: ...
# END COPIED BLOCK.


@dataclass(frozen=True)
class SessionChange:
    """A session boundary the pipeline must react to.

    The pipeline's obligation on receiving one of these is to flush tracker
    state, evidence buffers and every in-flight fusion accumulator for the
    previous session. Contracts section 1.2. This type exists so that obligation
    is delivered as data rather than remembered as a convention.

    camera_id is here because these events get fanned in. One source knows which
    camera it is and does not need telling, but the moment a process pulls from N
    sources into one queue -- which is the whole point of ai/track/registry.py --
    an event that cannot say which camera it belongs to can only be routed by
    whatever happened to be next to it in the loop. That is a bug waiting for the
    second camera, and the field costs nothing.
    """

    camera_id: str
    previous_session_id: Optional[str]
    new_session_id: str
    reason: str            # one of EndReason
    detail: Optional[str]
    at_pts_ms: Optional[int]
    at_frame_index: int
    at_wallclock: str


class BaseMediaSource(ABC):
    """Shared implementation of MediaSource.

    Subclasses implement three methods and declare one class attribute:

        source_mode      one of the five values in Contracts section 2
        _open_capture()  acquire the underlying input
        _read_raw()      return (frame_bgr, raw_pts_ms) or None at end of input
        _close_capture() release it

    and may override supports_reconnect when the input is a live stream that
    should be retried rather than treated as finished.
    """

    source_mode: ClassVar[str] = "file"

    def __init__(
        self,
        camera_id: str,
        *,
        target_interval_ms: int = TARGET_INTERVAL_MS,
        detect_discontinuity: bool = True,
        discontinuity_threshold: Optional[float] = None,
        reconnect: Optional[ReconnectPolicy] = None,
        max_frames: Optional[int] = None,
        pacer: Optional[ReplayPacer] = None,
    ) -> None:
        self.camera_id = require_camera_id(camera_id)
        self.target_interval_ms = target_interval_ms
        self.detect_discontinuity = detect_discontinuity
        self.max_frames = max_frames
        self.pacer = pacer

        self._sampler = PtsSampler(target_interval_ms)
        self._pts = PtsValidator()
        self._discontinuity = DiscontinuityDetector(
            # `is not None`, not truthiness. A threshold of 0.0 is a real (if unwise) value and
            # truthiness silently replaced it with the 0.70 default -- a config that said one
            # thing while the detector did another, which is the failure this whole module
            # exists to make impossible for streams.
            **(
                {}
                if discontinuity_threshold is None
                else {"threshold": discontinuity_threshold}
            )
        )
        self._reconnect = reconnect or ReconnectPolicy()

        self._session_id: Optional[str] = None
        self._frame_index = 0
        self._is_open = False
        self._emitted_total = 0

        self.session_events: Deque[SessionChange] = deque(maxlen=256)
        self._session_listeners: list[Callable[[SessionChange], None]] = []
        self.sessions_started = 0

    # ---------------------------------------------------------------- protocol

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError(
                f"{type(self).__name__}.session_id read before open(); "
                "a session only exists once the source is connected"
            )
        return self._session_id

    def open(self) -> None:
        if self._is_open:
            return
        self._open_capture()
        self._is_open = True
        self._start_session(reason="open", detail=f"{type(self).__name__} connected")

    def read(self) -> Optional[FrameEnvelope]:
        """Return the next frame that passes sampling, or None at end of stream.

        Skipping happens inside this call: a caller that asks for a frame gets a
        frame it should process, never a frame it has to decide about. That is
        what keeps sampling policy out of the pipeline.
        """
        if not self._is_open:
            raise RuntimeError(f"{type(self).__name__}.read() before open()")

        while True:
            if self.max_frames is not None and self._emitted_total >= self.max_frames:
                return None

            try:
                raw = self._read_raw()
            except Exception as exc:  # decoder and transport errors both land here
                if not self.supports_reconnect:
                    raise
                if not self._attempt_reconnect(f"read failed: {exc}"):
                    return None
                continue

            if raw is None:
                if not self._handle_exhausted("end of input"):
                    return None
                continue

            frame_bgr, raw_pts_ms = raw
            self._reconnect.note_healthy_read()

            verdict = self._pts.observe(raw_pts_ms)

            if verdict.action is PtsAction.SKIP:
                # Unusable timestamp. Dropping the frame is the point: emitting
                # it would stamp it with a duplicate or a stale PTS, and every
                # temporal number derived from it afterwards would be wrong in a
                # way nothing downstream could detect.
                self._sampler.note_skipped()
                continue

            if verdict.action is PtsAction.FORCE_RECONNECT:
                if not self.supports_reconnect:
                    # A stalled file is a corrupt file, not a network problem.
                    return None
                if not self._attempt_reconnect(verdict.reason or "decoder stalled"):
                    return None
                continue

            if verdict.action is PtsAction.NEW_SESSION:
                self._rotate_session(
                    reason="discontinuity",
                    detail=verdict.reason,
                    at_pts_ms=verdict.pts_ms,
                    seed_pts_ms=verdict.pts_ms,
                )
                # Fall through: this frame is the first of the new scene, and it
                # belongs to the new session rather than being thrown away.

            if not self._sampler.should_emit(verdict.pts_ms):
                self._sampler.note_skipped()
                continue

            if self.detect_discontinuity:
                cut, correlation = self._discontinuity.check(frame_bgr)
                if cut:
                    self._rotate_session(
                        reason="discontinuity",
                        detail=(
                            f"histogram correlation {correlation:.3f} below "
                            f"{self._discontinuity.threshold:.2f}"
                        ),
                        at_pts_ms=verdict.pts_ms,
                        seed_pts_ms=verdict.pts_ms,
                    )
                    # Re-seed the detector with this frame. The rotation cleared
                    # its history, and without seeding the next frame would have
                    # nothing to compare against -- a second cut arriving two
                    # frames later would go unnoticed.
                    self._discontinuity.check(frame_bgr)

            return self._build_envelope(frame_bgr, verdict.pts_ms)

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            self._close_capture()
        finally:
            self._is_open = False

    # ------------------------------------------------------------- subclass API

    @property
    def supports_reconnect(self) -> bool:
        """True for live streams. False means end of input is genuinely the end.

        A file that runs out has not failed, and retrying it forever would turn
        a finished benchmark run into an infinite loop.
        """
        return False

    @abstractmethod
    def _open_capture(self) -> None:
        ...

    @abstractmethod
    def _read_raw(self) -> Optional[tuple[np.ndarray, Optional[float]]]:
        """Return (frame_bgr, raw_pts_ms), or None at end of input.

        raw_pts_ms is whatever the decoder reports, unmassaged -- None and 0 are
        both fine and are handled by PtsValidator. Do not synthesize a timestamp
        here; a source that invents its own clock defeats the detection of the
        very failure the validator exists to catch.
        """

    @abstractmethod
    def _close_capture(self) -> None:
        ...

    def _handle_exhausted(self, detail: str) -> bool:
        """The input ran out. Return True if reading should continue.

        A live stream reconnects. A file ends. A looping file restarts under a
        new session -- which is why this is a hook and not an inline branch: a
        replay restart is a scene cut in every way that matters to the tracker,
        and the only safe way to express that is a session boundary.
        """
        if self.supports_reconnect:
            return self._attempt_reconnect(detail)
        return False

    # ------------------------------------------------------------------ session

    def add_session_listener(self, listener: Callable[[SessionChange], None]) -> None:
        """Register a callback fired the moment a session starts.

        Not what the pipeline uses, and the distinction is worth stating because a
        push callback looks like the obvious way to wire this up. VehiclePipeline
        compares envelope.stream_session_id against the session it is holding and
        flushes on the difference -- see ai/pipeline.py's _handle_session -- so the
        boundary is discovered from the frame itself rather than from an event that
        has to arrive first. A listener can fire inside the same read() that returns
        frame 0 of the new session, so ordering is not guaranteed; the envelope is
        never early or late because it *is* the frame.

        Use this for logging and metrics, where being a frame off does not matter.
        """
        self._session_listeners.append(listener)

    def drain_session_events(self) -> list[SessionChange]:
        """Take the queued session changes. For callers that poll instead."""
        events = list(self.session_events)
        self.session_events.clear()
        return events

    def _start_session(
        self,
        *,
        reason: str,
        detail: Optional[str],
        at_pts_ms: Optional[int] = None,
        seed_pts_ms: Optional[int] = None,
    ) -> None:
        previous = self._session_id
        self._session_id = new_session_id()
        self._frame_index = 0
        self.sessions_started += 1

        self._sampler.reset()
        self._pts = PtsValidator()
        self._discontinuity.reset()
        if self.pacer is not None:
            self.pacer.reset()

        if seed_pts_ms is not None:
            # The frame that triggered this rotation is about to be emitted as
            # frame 0 of the new session, so the new validator has to know its
            # timestamp. Otherwise it treats the *next* frame as first-of-session
            # and a duplicate or slightly-earlier timestamp slips through
            # unchecked.
            self._pts.last_pts_ms = seed_pts_ms

        change = SessionChange(
            camera_id=self.camera_id,
            previous_session_id=previous,
            new_session_id=self._session_id,
            reason=reason,
            detail=detail,
            at_pts_ms=at_pts_ms,
            at_frame_index=0,
            at_wallclock=utc_now_iso(),
        )
        self.session_events.append(change)
        for listener in self._session_listeners:
            listener(change)

    def _rotate_session(
        self,
        *,
        reason: str,
        detail: Optional[str],
        at_pts_ms: Optional[int] = None,
        seed_pts_ms: Optional[int] = None,
    ) -> None:
        """End the current session and mint a new one.

        Everything session-scoped resets: frame index, PTS validator, sampler
        gate, discontinuity history. Listeners flush tracker state and evidence
        buffers. Skip any one of those and a vehicle from before the boundary
        can be matched to a vehicle after it.
        """
        self._start_session(
            reason=reason,
            detail=detail,
            at_pts_ms=at_pts_ms,
            seed_pts_ms=seed_pts_ms,
        )

    def _attempt_reconnect(self, detail: str) -> bool:
        """Backoff, reopen, mint a new session. False means give up.

        Returning False rather than raising lets a worker treat an exhausted
        live source the same way it treats end of file, which is what keeps the
        offline and live paths identical downstream.
        """
        self._reconnect.note_failure()
        try:
            self._close_capture()
        except Exception:
            # Already broken; the reason we are here. Nothing to salvage, and
            # letting this mask the original failure would be worse.
            pass

        self._reconnect.wait()

        try:
            self._open_capture()
        except Exception as exc:
            self._is_open = False
            raise RuntimeError(
                f"{type(self).__name__} could not reconnect to {self.camera_id}: {exc}"
            ) from exc

        self._is_open = True
        self._start_session(
            reason="reconnect",
            detail=f"{detail}; reconnected after {self._reconnect.attempt} attempt(s)",
        )
        return True

    # ----------------------------------------------------------------- envelope

    def _build_envelope(self, frame_bgr: np.ndarray, pts_ms: int) -> FrameEnvelope:
        if self.pacer is not None:
            self.pacer.wait_for(pts_ms)

        arr = np.ascontiguousarray(frame_bgr)
        height, width = arr.shape[0], arr.shape[1]

        envelope = FrameEnvelope(
            camera_id=self.camera_id,
            stream_session_id=self.session_id,
            frame_index=self._frame_index,
            pts_ms=pts_ms,
            wallclock_utc=wallclock_for_source(self.source_mode),
            frame_bgr=arr,
            width=width,
            height=height,
            source_mode=self.source_mode,  # type: ignore[arg-type]
        )

        self._frame_index += 1
        self._emitted_total += 1
        self._sampler.note_emitted(pts_ms)
        return envelope

    # -------------------------------------------------------------------- misc

    @property
    def pts_unreliable(self) -> bool:
        """True once this session fell back to a synthetic clock.

        A pts_unreliable session must never substantiate a latency claim. The
        benchmark writer reads this and refuses to record timing diagnostics for
        such a run rather than publishing numbers derived from a clock we made up.
        """
        return self._pts.pts_unreliable

    def stats(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source_mode": self.source_mode,
            "session_id": self._session_id,
            "sessions_started": self.sessions_started,
            "frames_emitted": self._emitted_total,
            "pts_unreliable": self.pts_unreliable,
            "sampler": self._sampler.stats(),
            "pts": self._pts.stats(),
            "discontinuity": self._discontinuity.stats(),
            "reconnect": self._reconnect.stats(),
            "pacer": None if self.pacer is None else self.pacer.stats(),
        }

    def __enter__(self) -> "BaseMediaSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FrameEnvelope]:
        """Frames until the stream ends, so the worker loop reads as a for loop.

        Not part of the MediaSource Protocol -- read() is, and it stays the
        primitive. This is sugar over `while (env := src.read()) is not None`, and
        it deliberately does not open the source: forgetting `with` would
        otherwise raise deep inside a decoder rather than at the point of the
        mistake.

        A live source never terminates on its own, which is the same behaviour the
        while loop has. Bound it with max_frames or an external stop flag, not by
        hoping the iterator ends.
        """
        while True:
            envelope = self.read()
            if envelope is None:
                return
            yield envelope

    def __repr__(self) -> str:
        session = self._session_id[:8] if self._session_id else "none"
        return (
            f"{type(self).__name__}(camera_id={self.camera_id!r}, "
            f"mode={self.source_mode!r}, session={session})"
        )
