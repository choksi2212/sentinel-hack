"""Shared machinery for the two live adapters. Owner's manual section 4.4.

RTSP and HLS differ in exactly three things: the URL shape, how credentials are
supplied, and which FFmpeg options are set before the capture is created.
Everything else -- the reader thread, the depth-1 handoff, the bounded join on
shutdown, the error capture -- is identical, and lives here.

Written as one class on purpose. Two copies of a threaded reader diverge within a
day: one grows a timeout fix, the other doesn't, and then the live demo behaves
differently depending on which URL was in the config file.

The depth-1 handoff is Contracts section 2.2. A reader thread decodes as fast as
the network delivers; the pipeline takes whatever is newest and the rest is
dropped and counted. A queue instead of a buffer converts a 200 ms per-frame
deficit into unbounded lag: five minutes in, the "live" view is ten minutes
stale, and shortly after that the process is OOM-killed.

The drop count is not diagnostics noise. It is how you distinguish a flaky camera
from a GPU that cannot keep up, and it belongs in every performance claim made
about this pipeline.
"""

import os
import threading
from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from ai.media.backoff import ReconnectPolicy
from ai.media.base import BaseMediaSource
from ai.media.sampler import TARGET_INTERVAL_MS, LatestFrameBuffer

# No frame at all for this long means the stream is gone, not slow.
DEFAULT_READ_TIMEOUT_SECONDS = 10.0

# Decoder-side buffer. 1 keeps FFmpeg from accumulating its own backlog behind
# our depth-1 buffer, which would reintroduce the lag we just designed out.
CAPTURE_BUFFERSIZE = 1

# Seconds to wait for the reader thread on shutdown. A blocked FFmpeg read can
# outlive the request to stop, and hanging shutdown forever is worse than leaking
# a daemon thread that dies with the process.
READER_JOIN_TIMEOUT_SECONDS = 5.0


class ThreadedLiveSource(BaseMediaSource):
    """A live network stream read on a background thread into a depth-1 buffer.

    Subclasses provide:
        source_mode        "live_rtsp" or "live_hls"
        _stream_url()      the URL to open
        capture_options    FFmpeg options string, or None
    """

    def __init__(
        self,
        camera_id: str,
        *,
        read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS,
        target_interval_ms: int = TARGET_INTERVAL_MS,
        detect_discontinuity: bool = True,
        reconnect: Optional[ReconnectPolicy] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__(
            camera_id,
            target_interval_ms=target_interval_ms,
            detect_discontinuity=detect_discontinuity,
            reconnect=reconnect,
            max_frames=max_frames,
        )
        self.read_timeout_seconds = read_timeout_seconds

        self.buffer = LatestFrameBuffer()
        self._cap: Any = None
        self._cv2: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._reader_error: Optional[BaseException] = None
        self._reader_finished = threading.Event()
        self._decoded = 0
        self._read_timeouts = 0

    @property
    def supports_reconnect(self) -> bool:
        return True

    # ------------------------------------------------------------- subclass API

    @abstractmethod
    def _stream_url(self) -> str:
        """The URL to open. May include credentials read from the environment."""

    @property
    def capture_options(self) -> Optional[str]:
        """Value for OPENCV_FFMPEG_CAPTURE_OPTIONS, or None to leave it alone."""
        return None

    # ------------------------------------------------------------------ capture

    def _open_capture(self) -> None:
        import cv2  # lazy: the contracts package must import without OpenCV

        self._cv2 = cv2
        url = self._stream_url()

        options = self.capture_options
        if options:
            # Must be set before the capture is constructed -- FFmpeg reads it
            # when the context is created, so setting it afterwards silently does
            # nothing and the transport falls back with no warning anywhere.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"could not open {self.source_mode} stream for {self.camera_id} "
                f"({redact_url(url)})"
            )

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFERSIZE)
        except Exception:
            # Not honoured by every backend build. Our own depth-1 buffer is the
            # real guarantee; this is belt and braces.
            pass

        self._cap = cap
        self._reader_error = None
        self._stop.clear()
        self._reader_finished.clear()
        self.buffer.clear()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"{self.source_mode}-reader-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        """Decode continuously, keeping only the newest frame.

        Runs at network speed, independent of how long inference takes. Failures
        are captured rather than raised: a thread that dies with a traceback into
        a daemon's stderr is a stream that stops with no explanation.
        """
        try:
            while not self._stop.is_set():
                cap, cv2 = self._cap, self._cv2
                if cap is None or cv2 is None:
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                self._decoded += 1
                self.buffer.put((frame, cap.get(cv2.CAP_PROP_POS_MSEC)))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the read path
            self._reader_error = exc
        finally:
            self._reader_finished.set()
            self.buffer.wake()

    def _read_raw(self) -> Optional[tuple[np.ndarray, Optional[float]]]:
        item = self.buffer.get(timeout=self.read_timeout_seconds)

        if self._reader_error is not None:
            error, self._reader_error = self._reader_error, None
            raise RuntimeError(
                f"{self.source_mode} reader failed for {self.camera_id}: {error}"
            )

        if item is None:
            if not self._reader_finished.is_set():
                self._read_timeouts += 1
            return None  # base decides: reconnect for a live source

        frame, pts_ms = item
        return frame, pts_ms

    def _close_capture(self) -> None:
        self._stop.set()
        self.buffer.wake()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=READER_JOIN_TIMEOUT_SECONDS)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.buffer.clear()

    # --------------------------------------------------------------------- misc

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "url": redact_url(self._stream_url()),
                "capture_options": self.capture_options,
                "frames_decoded": self._decoded,
                "read_timeouts": self._read_timeouts,
                "buffer": self.buffer.stats(),
            }
        )
        return base


def redact_url(url: str) -> str:
    """Strip userinfo from a URL before it reaches a log or a stats payload.

    Stats end up in log files, in benchmark JSON and occasionally on a slide. A
    stream password that reaches any of those is a credential leak, and the
    Sentinel HLS credentials are the organisers', not ours.
    """
    if "@" not in url:
        return url
    scheme, _, remainder = url.partition("://")
    if not remainder:
        return url
    _, _, host_and_path = remainder.rpartition("@")
    return f"{scheme}://***@{host_and_path}" if scheme else f"***@{host_and_path}"
