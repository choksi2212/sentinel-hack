"""Emission -- stage 13 of the 14. One finished track becomes one event, and leaves.

    build_event(buffer, source_mode=..., model=..., observations=obs)  -> EventEnvelope
    HttpEventSink("http://127.0.0.1:8000")                            -> POST, retry, spool
    FileEventSink("out/events.jsonl")                                  -> the offline sink
    SnapshotWriter("artifacts/snapshots")                              -> the two stills

This is the narrowing point of the whole pipeline. Everything upstream is per-frame;
everything downstream -- ingest, search, GIS, watchlist -- is one row per vehicle sighting.
Three properties are established here and depended on everywhere after:

**`plate: null` is a correct event.** A vehicle passed and could not be identified. That is
information: it counts the vehicle, it proves the camera was working, and it is true. The
alternative attaches a real registration number to a place and time it was never at, and
unlike a null that error is invisible -- it looks exactly like a correct event.

**event_id is minted once, in builder.py.** It is the backend's idempotency key, so it must
survive every retry unchanged. That single decision is what makes three separate things
safe at once: the POST retry (a duplicate lands as 200, not a second sighting), the disk
spool (the filename is the event_id, so spooling twice overwrites), and the snapshot
(same name, so a re-run replaces rather than accumulates).

**Nothing is dropped quietly.** build_event raises rather than returning None; the sink's
counters balance continuously, including what is mid-retry and what is sitting on disk; the
snapshot writer returns None instead of a URI to a file it failed to write. A pipeline that
loses events silently reports a vehicle count that is merely a bit low, which is the
hardest kind of error to notice -- the number is still a number.
"""

from ai.emit.builder import (
    EventBuildError,
    build_event,
    build_event_with_evidence,
    build_events,
    observations_from_buffer,
    winning_crop,
)
from ai.emit.http_sink import (
    DEFAULT_INGEST_PATH,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_SPOOL_FILES,
    DEFAULT_QUEUE_MAXSIZE,
    DEFAULT_TIMEOUT_S,
    FileEventSink,
    HttpEventSink,
    NullEventSink,
)
from ai.emit.snapshot import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SNAPSHOT_WIDTH,
    DEFAULT_SNAPSHOT_ROOT,
    NullSnapshotWriter,
    SnapshotWriter,
    build_snapshot_writer,
    safe_component,
)

__all__ = [
    "DEFAULT_INGEST_PATH",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_SNAPSHOT_WIDTH",
    "DEFAULT_MAX_SPOOL_FILES",
    "DEFAULT_QUEUE_MAXSIZE",
    "DEFAULT_SNAPSHOT_ROOT",
    "DEFAULT_TIMEOUT_S",
    "EventBuildError",
    "FileEventSink",
    "HttpEventSink",
    "NullEventSink",
    "NullSnapshotWriter",
    "SnapshotWriter",
    "build_event",
    "build_event_with_evidence",
    "build_events",
    "build_snapshot_writer",
    "observations_from_buffer",
    "safe_component",
    "winning_crop",
]
