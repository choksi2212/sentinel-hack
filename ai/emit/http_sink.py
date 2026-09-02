"""POST events to the backend without ever blocking the frame loop. Stage 13's outbound half.

Two requirements pull in opposite directions and both are non-negotiable.

**The frame loop must not wait for HTTP.** At a 100 ms sampling interval, one 5-second
socket timeout is fifty dropped frames. A synchronous POST inside the pipeline makes
throughput a function of the backend's latency, and the first time Postgres pauses for a
checkpoint the AI stage's measured FPS collapses for reasons that have nothing to do with
the AI stage.

**No event may be lost silently.** A sink that swallows failures reports a vehicle count
that is quietly too low, and a count that is quietly wrong is worse than an error --
nobody investigates a number that looks fine.

So: send() enqueues and returns, a single worker thread does the POSTing, and anything the
worker cannot deliver goes to a bounded spool on disk keyed by event_id. stats() reports
the backlog, because a sink claiming it sent 400 events while 30 sit unsent on disk is
lying in the direction that flatters it.

**Retry policy, from the ingest contract (Mihir's manual section 5.1):**

    201 accepted    delivered, new sighting
    200 duplicate   delivered. NOT an error -- this is retry-safety working. The
                    event_id is minted once in builder.py and never changes, so a
                    retry after an ambiguous failure lands as a duplicate instead of
                    a second sighting. If this ever returns 201 on a retry, the
                    idempotency ledger is broken and every count is inflated.
    422 rejected    PERMANENT. The payload is wrong; retrying sends the same wrong
                    payload forever, so it goes straight to the rejected spool where
                    it can be inspected. Treating 422 as retryable is how a sink turns
                    one bad event into an infinite loop against a live backend.
    503 unavailable RETRYABLE. Postgres is down and will come back.
    network error   RETRYABLE, and the dangerous case: a timeout after the server
                    committed looks identical to one before. Only the stable event_id
                    makes that safe to retry at all.
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ai.contracts.event import EventEnvelope

DEFAULT_INGEST_PATH = "/api/v1/ingest/events"

# Socket timeout per attempt. Short on purpose: the retry loop is the mechanism for
# surviving a slow backend, not a long timeout. A 30-second timeout with 4 retries means
# a two-minute stall holding one worker thread and a queue filling behind it.
DEFAULT_TIMEOUT_S = 5.0

# Attempts per event before it goes to the spool. Four attempts at the backoff below
# spans roughly 7 seconds, which covers a restart but not an outage -- an outage should
# put events on disk, not hold them in memory where a crash loses them.
DEFAULT_MAX_ATTEMPTS = 4

# Exponential, capped. Capped because unbounded backoff on a bounded queue just moves the
# failure from "retrying" to "queue full", with a longer delay before anyone notices.
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 4.0

# In-memory queue depth. Bounded so a backend outage cannot grow the worker's memory
# without limit; past this, events spill to disk in send() itself.
DEFAULT_QUEUE_MAXSIZE = 512

# Spool ceiling. At roughly 1 KB per event this is about 50 MB, which is far more than a
# demo can produce and small enough that a runaway cannot fill the disk.
DEFAULT_MAX_SPOOL_FILES = 50_000

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpEventSink:
    """Delivers EventEnvelopes to the ingest endpoint, spooling what it cannot send.

    Not thread-safe for close(); is thread-safe for send(), which is the direction that
    matters since the pipeline may run several camera workers in one process.
    """

    def __init__(
        self,
        base_url: str,
        *,
        path: str = DEFAULT_INGEST_PATH,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        spool_dir: Optional[str] = None,
        max_spool_files: int = DEFAULT_MAX_SPOOL_FILES,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        replay_on_start: bool = True,
    ) -> None:
        self.url = base_url.rstrip("/") + path
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.max_spool_files = int(max_spool_files)

        self._spool_dir = Path(spool_dir) if spool_dir else None
        self._pending_dir: Optional[Path] = None
        self._rejected_dir: Optional[Path] = None
        if self._spool_dir is not None:
            self._pending_dir = self._spool_dir / "pending"
            self._rejected_dir = self._spool_dir / "rejected"
            self._pending_dir.mkdir(parents=True, exist_ok=True)
            self._rejected_dir.mkdir(parents=True, exist_ok=True)

        self._queue: "queue.Queue[Optional[EventEnvelope]]" = queue.Queue(
            maxsize=max(1, int(queue_maxsize))
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._session: Any = None
        self._lock = threading.Lock()

        self.accepted = 0
        self.duplicates = 0
        self.rejected = 0
        self.spooled = 0
        self.replayed = 0
        self.dropped = 0
        self.submitted = 0
        self.attempts = 0
        self.retries = 0
        self.spool_full_events = 0
        # Events the worker has dequeued and not yet resolved. Exists so the balance in
        # stats() closes at every instant rather than only after close(): an event part
        # way through its retry loop is in no other counter, so without this a reader
        # sampling mid-run sees a total that is short by however many are in flight and
        # concludes events were lost. They were not -- they are mid-retry.
        self.in_flight = 0
        self._pending_count = 0
        self._replay_on_start = bool(replay_on_start)

    # ---------------------------------------------------------------- lifecycle

    def open(self) -> None:
        if self._thread is not None:
            return
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment dependent
            # -------------------------------------------------------------------
            # MANUAL STEP REQUIRED -- not a code defect, an environment gap.
            #
            #     pip install requests
            #
            # Listed in ai/README.md for Mihir's requirements.txt. Until then use
            # FileEventSink, which writes the same envelopes as JSONL and is what the
            # offline config uses anyway -- the whole pipeline is measurable without a
            # running backend.
            # -------------------------------------------------------------------
            raise RuntimeError(
                "requests is not installed, so HttpEventSink cannot POST. Run: pip "
                "install requests -- or set the sink to 'file', which needs no network "
                "and no backend. See ai/README.md."
            ) from exc

        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # The ingest contract does not specify auth, so this is optional and off by
            # default. Supported because adding a header later is a config change and
            # retrofitting one into a running sink is not.
            headers["X-API-Key"] = self.api_key
        self._session.headers.update(headers)

        self._stop.clear()
        self._rescan_pending()
        self._thread = threading.Thread(
            target=self._run, name="event-sink", daemon=True
        )
        self._thread.start()

        if self._replay_on_start:
            # Events spooled by a previous run go out before anything new. Ordering is
            # not guaranteed across the boundary and does not need to be: the backend
            # keys on event_id and dedupe_key, not on arrival order.
            self.replay_spool()

    def close(self, *, timeout_s: float = 10.0) -> None:
        """Drain, then spool whatever did not make it. Never discards.

        The timeout bounds shutdown, not delivery. Anything still queued when it expires
        goes to disk, so a slow backend delays the process exiting by at most timeout_s
        and costs no events.
        """
        if self._thread is None:
            return
        try:
            # Bounded, because a full queue means the worker is stuck on a slow backend
            # and an unbounded put here would hang shutdown on the same outage the spool
            # exists to survive. If the sentinel does not fit, _stop ends the worker
            # instead and the loop below spools whatever is left.
            self._queue.put(None, timeout=max(0.0, min(1.0, timeout_s)))
        except queue.Full:
            self._stop.set()
        self._thread.join(timeout=max(0.0, timeout_s))
        self._stop.set()
        # A sentinel that the worker never reached means the queue still holds events.
        # Spool them rather than letting the daemon thread die with them in memory.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._spool(item)
        self._thread = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "HttpEventSink":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------- sending

    def send(self, envelope: EventEnvelope) -> None:
        """Enqueue for delivery. Returns immediately and never raises on backend trouble.

        A full queue means the backend has been unreachable long enough to fill it, so the
        event goes straight to disk here rather than blocking the caller. Blocking would
        propagate the outage into the frame loop, which is the one thing this class exists
        to prevent.
        """
        if self._thread is None:
            raise RuntimeError("HttpEventSink.open() was not called")
        with self._lock:
            self.submitted += 1
        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            self._spool(envelope)

    def flush(self, *, timeout_s: float = 10.0) -> bool:
        """Block until the in-memory queue is empty. True if it emptied in time.

        For tests and for the end of an offline run. Not for the frame loop.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self._queue.empty():
                return True
            time.sleep(0.02)
        return self._queue.empty()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            self._deliver(item)

    def _deliver(self, envelope: EventEnvelope) -> None:
        with self._lock:
            self.in_flight += 1
        try:
            self._deliver_inner(envelope)
        finally:
            with self._lock:
                self.in_flight = max(0, self.in_flight - 1)

    def _deliver_inner(self, envelope: EventEnvelope) -> None:
        payload = envelope.to_dict()
        for attempt in range(1, self.max_attempts + 1):
            self.attempts += 1
            outcome, retry_after = self._post_once(payload)
            if outcome == "accepted":
                with self._lock:
                    self.accepted += 1
                self._forget(envelope.event_id)
                return
            if outcome == "duplicate":
                with self._lock:
                    self.duplicates += 1
                self._forget(envelope.event_id)
                return
            if outcome == "rejected":
                # Permanent. Kept on disk under rejected/ rather than dropped: a 422 is
                # a bug in this lane's output and the payload is the evidence for fixing
                # it. Counting it and discarding it would leave "12 rejected" with
                # nothing to look at.
                with self._lock:
                    self.rejected += 1
                self._write_rejected(envelope)
                self._forget(envelope.event_id)
                return
            if outcome == "spool":
                # Unretryable but not a validation failure -- a wrong URL, missing auth,
                # a proxy in the way. Retrying four times changes nothing except how long
                # it takes to find out, so go to the spool now and let the backlog make
                # the misconfiguration visible.
                break
            if attempt < self.max_attempts:
                self.retries += 1
                time.sleep(retry_after if retry_after is not None else _backoff(attempt))
        self._spool(envelope)

    def _post_once(self, payload: dict[str, Any]) -> tuple[str, Optional[float]]:
        """One POST. Returns (outcome, retry_after_seconds).

        Outcomes: accepted, duplicate, rejected, retry. Every exception path returns
        "retry" -- a sink that raises out of its own worker thread kills the thread and
        turns a transient network fault into permanent silent data loss.
        """
        try:
            response = self._session.post(
                self.url, json=payload, timeout=self.timeout_s
            )
        except Exception:  # noqa: BLE001 - see docstring
            return "retry", None

        status = response.status_code
        if status == 201:
            return "accepted", None
        if status == 200:
            # 200 is the duplicate response per the contract. Read the body when it is
            # there, but do not depend on it: the status is the contract and a body that
            # fails to parse is not a reason to re-send an event the server already has.
            try:
                if response.json().get("status") == "accepted":
                    return "accepted", None
            except Exception:  # noqa: BLE001
                pass
            return "duplicate", None
        if status == 422:
            return "rejected", None
        if status in _RETRYABLE_STATUS:
            return "retry", _retry_after(response)
        # Anything else -- 401, 404, a proxy's 400 -- is not something a retry fixes, but
        # it is also not a validated rejection, so it must not land in rejected/ as if
        # the payload were at fault. Spooling it keeps the event and leaves the wrong
        # URL or missing auth visible in the backlog.
        return "spool", None

    # -------------------------------------------------------------------- spool

    def _spool(self, envelope: EventEnvelope) -> None:
        """Persist an undelivered event, named by event_id.

        The filename is the event_id, so spooling the same event twice overwrites rather
        than duplicating -- the same property that makes the POST retry safe makes the
        spool idempotent, from the same one decision in builder.py.
        """
        if self._pending_dir is None:
            # No spool configured. Counted as dropped and never hidden: stats() showing
            # dropped > 0 is the signal that a run's event count is incomplete.
            with self._lock:
                self.dropped += 1
            return
        with self._lock:
            if self._pending_count >= self.max_spool_files:
                self.dropped += 1
                self.spool_full_events += 1
                return
        path = self._pending_dir / f"{envelope.event_id}.json"
        existed = path.exists()
        _write_atomic(path, envelope.to_json())
        with self._lock:
            self.spooled += 1
            if not existed:
                self._pending_count += 1

    def _write_rejected(self, envelope: EventEnvelope) -> None:
        if self._rejected_dir is None:
            return
        _write_atomic(
            self._rejected_dir / f"{envelope.event_id}.json", envelope.to_json()
        )

    def _forget(self, event_id: str) -> None:
        """Remove a delivered event from the spool, if it was there."""
        if self._pending_dir is None:
            return
        try:
            (self._pending_dir / f"{event_id}.json").unlink()
        except FileNotFoundError:
            return
        except OSError:  # pragma: no cover - filesystem dependent
            return
        with self._lock:
            self._pending_count = max(0, self._pending_count - 1)

    def _count_pending(self) -> int:
        """The cached backlog size.

        Cached rather than globbed. Globbing per spool is O(files), and during an outage
        every event spools -- so the ceiling check would be O(files^2) across the outage,
        making the sink slowest at exactly the moment it is under load. The count is
        seeded once by _rescan_pending at open() and maintained by _spool/_forget.
        """
        return self._pending_count

    def _rescan_pending(self) -> int:
        """Recount the spool from disk. Called at open(), not in the hot path."""
        if self._pending_dir is None:
            self._pending_count = 0
            return 0
        try:
            count = sum(1 for _ in self._pending_dir.glob("*.json"))
        except OSError:  # pragma: no cover - filesystem dependent
            count = 0
        with self._lock:
            self._pending_count = count
        return count

    def replay_spool(self, *, limit: Optional[int] = None) -> int:
        """Re-enqueue spooled events. Returns how many were queued.

        Re-enqueued rather than POSTed here, so replay goes through the same delivery
        path as everything else. Files are left in place until delivery succeeds, which
        is what makes replay safe to interrupt: a crash mid-replay loses nothing.
        """
        if self._pending_dir is None:
            return 0
        queued = 0
        for path in sorted(self._pending_dir.glob("*.json")):
            if limit is not None and queued >= limit:
                break
            try:
                envelope = EventEnvelope.from_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, TypeError):
                # A spool file that will not parse is not recoverable by retrying, and
                # leaving it in pending/ means every replay pass trips over it forever.
                # Moved aside so the backlog can drain and the file still exists to look
                # at.
                if self._rejected_dir is not None:
                    try:
                        path.rename(self._rejected_dir / path.name)
                    except OSError:  # pragma: no cover
                        continue
                    with self._lock:
                        self._pending_count = max(0, self._pending_count - 1)
                continue
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                break
            queued += 1
            with self._lock:
                self.replayed += 1
        return queued

    # -------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        """Delivery counters, including what has NOT been delivered.

        pending_spool and dropped are here because they are the two numbers that make the
        others honest. A report of accepted=400 alongside pending_spool=30 describes a
        run that produced 430 events; accepted=400 alone describes one that produced 400.

        **The accounting closes, and this is the invariant to assert on:**

            submitted + replayed
                == accepted + duplicates + rejected + spooled + dropped
                   + queue_depth + in_flight

        Left side is every event handed to the sink -- by a caller, or by replay off disk.
        Right side is every place one can be. A spooled event that later replays is
        counted once on each side, which is why the identity holds through an outage and
        its recovery rather than only at the ends. queue_depth and in_flight are the two
        terms a reader forgets, and forgetting them is what makes a mid-run sample look
        like data loss: an event two seconds into a retry is in no other counter.

        **It is exact at quiescence, not at literally every instant**, and the difference
        matters to anyone writing an assertion. There are two hand-off windows where an
        event is momentarily in no term on the right: send() increments `submitted` before
        it enqueues, and the worker dequeues before _deliver increments `in_flight`. A
        sample taken inside one of those reads SHORT by however many events are in them.
        It cannot read long -- no path increments a right-hand term before the matching
        left-hand one -- so the only skew possible is the one that looks like loss, never
        the one that looks like success. Sample after flush() for an exact equality.

        Two things this deliberately does not do. `spooled` counts spool *writes*, not
        distinct undelivered events -- an event spooled, replayed and delivered leaves
        spooled=1 forever, which is a true statement about what happened rather than about
        the current backlog. `pending_spool` is the backlog. And no counter here is a
        substitute for close(): the invariant holding does not mean everything was
        delivered, only that nothing went missing.
        """
        delivered = self.accepted + self.duplicates
        return {
            "url": self.url,
            "submitted": self.submitted,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "delivered": delivered,
            "rejected": self.rejected,
            "spooled": self.spooled,
            "replayed": self.replayed,
            "dropped": self.dropped,
            "pending_spool": self._count_pending(),
            "queue_depth": self._queue.qsize(),
            "in_flight": self.in_flight,
            "post_attempts": self.attempts,
            "retries": self.retries,
            "spool_full_events": self.spool_full_events,
        }


class FileEventSink:
    """Append events to a JSONL file. The offline sink, and the fixture generator.

    Not a fallback for HttpEventSink so much as the primary sink for every measurement
    that does not involve the backend -- which is all of them on this lane. A benchmark
    that needs Postgres running to produce a number is a benchmark that stops working the
    first time someone else is mid-migration.

    One JSON object per line, no wrapping array, so a run can be tailed while it is in
    progress and a partial file is still parseable up to its last newline.
    """

    def __init__(self, path: str, *, append: bool = False) -> None:
        self.path = Path(path)
        self._append = append
        self._handle: Any = None
        self._lock = threading.Lock()
        self.written = 0

    def open(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "a" if self._append else "w", encoding="utf-8", newline="\n"
        )

    def send(self, envelope: EventEnvelope) -> None:
        if self._handle is None:
            raise RuntimeError("FileEventSink.open() was not called")
        line = envelope.to_json()
        with self._lock:
            self._handle.write(line + "\n")
            self.written += 1

    def flush(self, *, timeout_s: float = 0.0) -> bool:
        if self._handle is not None:
            with self._lock:
                self._handle.flush()
                # fsync so that a killed process leaves a complete file. Fault injection
                # on D5 kills the worker mid-run and then asserts on the events it wrote;
                # without this the assertion depends on the OS page cache.
                os.fsync(self._handle.fileno())
        return True

    def close(self, *, timeout_s: float = 0.0) -> None:
        if self._handle is None:
            return
        self.flush()
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "FileEventSink":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        return {"path": str(self.path), "written": self.written}


class NullEventSink:
    """Counts and discards. For measuring the pipeline without the sink in the number.

    Exists because "how fast is the AI pipeline" and "how fast is the AI pipeline plus a
    JSONL write plus an HTTP round trip" are different questions, and the benchmark that
    goes in the submission has to be able to answer the first one.
    """

    def __init__(self) -> None:
        self.written = 0

    def open(self) -> None:
        return None

    def send(self, envelope: EventEnvelope) -> None:
        self.written += 1

    def flush(self, *, timeout_s: float = 0.0) -> bool:
        return True

    def close(self, *, timeout_s: float = 0.0) -> None:
        return None

    def __enter__(self) -> "NullEventSink":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"written": self.written}


def _backoff(attempt: int) -> float:
    """Exponential with a cap. No jitter, deliberately.

    Jitter matters when many clients retry against one server at once. Here there is one
    worker thread per camera and at most thirty cameras, so the thundering herd this would
    protect against does not exist -- and a deterministic backoff makes the retry timing
    in a test assertion exact rather than approximate.
    """
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** (attempt - 1)))


def _retry_after(response: Any) -> Optional[float]:
    """Honour Retry-After when the server sends it, capped.

    Capped because the header is a request, not a command, and a proxy sending
    Retry-After: 3600 would otherwise stall the sink for an hour with a queue filling
    behind it.
    """
    raw = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    if not raw:
        return None
    try:
        return min(_BACKOFF_CAP_S, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a killed process leaves no partial JSON.

    A half-written spool file is worse than a missing one: it parses as far as the
    truncation and then throws, and replay_spool would move a perfectly good event to
    rejected/ because the write was interrupted rather than because the event was bad.
    """
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:  # pragma: no cover - filesystem dependent
        try:
            tmp.unlink()
        except OSError:
            pass
