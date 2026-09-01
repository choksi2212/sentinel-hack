"""TrackerRegistry -- one tracker per camera, flushed on every session change.

This module is the reason the TrackKey story actually holds in running code rather
than only in the dataclass. Two jobs:

  1. Hold one tracker per camera, so a worker process serving several cameras
     cannot leak state between them.
  2. Listen for SessionChange from the media layer and reset the affected tracker
     before the first frame of the new session reaches it.

Job 2 has an ordering requirement that is easy to get wrong and expensive when it
is. The media source emits SessionChange when it detects the break, and the worker
must drain those events and call on_session_change() **before** passing the frame
to the tracker. Drain afterwards and the first frame of the new session is
associated against the old session's tracks: a vehicle from before the reconnect
matches a different vehicle after it, and the two get one track id and one plate.

The correct order is asserted rather than documented, which is what
expect_session() is for. Passing a frame whose session id does not match the
tracker's raises instead of silently mixing sessions -- a loud failure in a test run
is worth an unbounded amount of quiet wrongness in a demo.
"""

from typing import Any, Callable, Iterator, Optional

from ai.contracts.stages import DetectorResult, TrackResult


class SessionMismatchError(RuntimeError):
    """A frame arrived from a session the tracker was not told about.

    Its own type because it means one specific mistake -- session events were not
    drained before the frame was processed -- and the fix is always the same. See
    the module docstring.
    """


class TrackerRegistry:
    """Per-camera trackers with session-boundary flushing.

    The factory is a callable rather than a class plus kwargs so a caller can bind
    whatever a particular tracker needs -- the oracle needs the media source, the
    others do not -- without this module knowing which trackers exist.
    """

    def __init__(
        self,
        factory: Callable[[str, str], Any],
        *,
        strict_sessions: bool = True,
    ) -> None:
        self._factory = factory
        self._trackers: dict[str, Any] = {}
        self._sessions: dict[str, str] = {}
        self.strict_sessions = bool(strict_sessions)

        self.session_changes_handled = 0
        self.trackers_created = 0
        self.frames_routed = 0

    # ------------------------------------------------------------------ tracking

    def tracker_for(self, camera_id: str, stream_session_id: str) -> Any:
        """The tracker for this camera, creating or re-sessioning it as needed.

        A session id that differs from the one on record resets the tracker. That is
        the safety net, not the mechanism -- on_session_change() is the mechanism,
        because it fires at the boundary and this fires on first sight of a frame
        that is already too late to associate correctly. With strict_sessions on,
        reaching this path raises instead.
        """
        existing = self._trackers.get(camera_id)
        if existing is None:
            tracker = self._factory(camera_id, stream_session_id)
            self._trackers[camera_id] = tracker
            self._sessions[camera_id] = stream_session_id
            self.trackers_created += 1
            return tracker

        if self._sessions.get(camera_id) != stream_session_id:
            if self.strict_sessions:
                raise SessionMismatchError(
                    f"frame for {camera_id} carries session "
                    f"{stream_session_id[:8]} but the tracker is on "
                    f"{self._sessions.get(camera_id, 'none')[:8]}. Session events "
                    "must be drained and on_session_change() called before the "
                    "frame is tracked, or the first frame of the new session is "
                    "associated against the previous session's tracks."
                )
            existing.reset(stream_session_id=stream_session_id)
            self._sessions[camera_id] = stream_session_id
            self.session_changes_handled += 1

        return existing

    def update(
        self,
        camera_id: str,
        stream_session_id: str,
        detections: Any,
        *,
        frame_index: int,
        pts_ms: int,
    ) -> list[TrackResult]:
        """Route one frame's detections to the right tracker."""
        tracker = self.tracker_for(camera_id, stream_session_id)
        self.frames_routed += 1
        return tracker.update(detections, frame_index=frame_index, pts_ms=pts_ms)

    # ------------------------------------------------------------------ sessions

    def on_session_change(self, change: Any) -> None:
        """Handle one SessionChange from the media layer.

        Accepts anything carrying camera_id and stream_session_id, so the registry
        does not import the media package. The AI stages are meant to be usable
        without a media source at all -- that is what makes them testable from a
        fixture -- and a type-only import would break it.
        """
        camera_id = getattr(change, "camera_id", None)
        session_id = getattr(change, "stream_session_id", None) or getattr(
            change, "session_id", None
        )
        if not camera_id or not session_id:
            raise ValueError(
                "session change needs camera_id and stream_session_id, got "
                f"{change!r}"
            )

        tracker = self._trackers.get(camera_id)
        if tracker is None:
            # Nothing to flush; record the session so the first frame does not look
            # like a mismatch.
            self._sessions[camera_id] = session_id
            return

        tracker.reset(stream_session_id=session_id)
        self._sessions[camera_id] = session_id
        self.session_changes_handled += 1

    def drain(self, source: Any) -> int:
        """Pull pending session events off a media source and apply them.

        Call this immediately after read() and before the frame is tracked. Returns
        how many were applied, which the worker logs -- a reconnect that nobody
        noticed is a reconnect that gets blamed on the model later.
        """
        events = source.drain_session_events()
        for event in events:
            self.on_session_change(event)
        return len(events)

    def attach(self, source: Any) -> None:
        """Subscribe to a source's session events instead of polling it.

        The push path is convenient and is not a substitute for drain(): a listener
        fires when the source notices the break, which may be inside the same read()
        that returns the first frame of the new session. Ordering is only guaranteed
        by draining explicitly before tracking. Use this for logging, use drain() for
        correctness.
        """
        source.add_session_listener(self.on_session_change)

    def expect_session(self, camera_id: str, stream_session_id: str) -> None:
        """Assert the tracker is on the session this frame belongs to.

        Cheap, and it converts the pipeline's most expensive silent bug into an
        exception at the exact line where the ordering was got wrong.
        """
        current = self._sessions.get(camera_id)
        if current is not None and current != stream_session_id:
            raise SessionMismatchError(
                f"{camera_id} frame is from session {stream_session_id[:8]}, "
                f"tracker is on {current[:8]}"
            )

    # -------------------------------------------------------------- housekeeping

    def reset_all(self) -> None:
        for camera_id, tracker in self._trackers.items():
            tracker.reset(stream_session_id=self._sessions.get(camera_id))

    def forget(self, camera_id: str) -> None:
        """Drop a camera entirely. For a worker that stops serving it."""
        self._trackers.pop(camera_id, None)
        self._sessions.pop(camera_id, None)

    def cameras(self) -> list[str]:
        return sorted(self._trackers)

    def __len__(self) -> int:
        return len(self._trackers)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(sorted(self._trackers.items()))

    def stats(self) -> dict[str, Any]:
        return {
            "cameras": len(self._trackers),
            "trackers_created": self.trackers_created,
            "session_changes_handled": self.session_changes_handled,
            "frames_routed": self.frames_routed,
            "strict_sessions": self.strict_sessions,
            "per_camera": {
                camera_id: tracker.stats()
                for camera_id, tracker in sorted(self._trackers.items())
            },
        }

    def __repr__(self) -> str:
        return (
            f"TrackerRegistry(cameras={len(self._trackers)}, "
            f"session_changes={self.session_changes_handled})"
        )
