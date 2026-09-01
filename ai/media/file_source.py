"""VideoFileSource -- recorded clip playback. Owner's manual section 4.1.

Built first, and not as a fallback. The Sentinel grid is live-only with no
seeking, which means you cannot replay the moment a bug happened; you get one
shot at observing it and then the traffic has moved on. Debugging a tracker
against a stream you cannot rewind is not debugging, it is guessing.

So: every accuracy number, every regression test, every fixture and every
bug fix is reproduced against a file first. The RTSP adapter exists to prove
source independence and to run the live demo. This adapter is where the work
happens.

PTS comes from CAP_PROP_POS_MSEC, per Contracts section 2.1. Never CAP_PROP_FPS,
never frame_index/fps. A variable-frame-rate clip -- which is most CCTV export --
makes the computed version drift steadily against the real timeline, and the
drift is invisible until you compare two cameras and the vehicle appears to
arrive before it left.
"""

import os
from typing import Any, Optional

import numpy as np

from ai.media.base import BaseMediaSource
from ai.media.pacing import ReplayPacer
from ai.media.sampler import TARGET_INTERVAL_MS


class VideoFileSource(BaseMediaSource):
    """Reads a video file and emits FrameEnvelopes indistinguishable from live.

    Indistinguishable is the requirement, not a nicety. Everything downstream
    reads source_mode for labelling and nothing else; if a stage can tell a file
    from a stream by any other means, the offline-to-live swap has stopped being
    a configuration change and the invariant is gone.
    """

    source_mode = "file"

    def __init__(
        self,
        camera_id: str,
        path: str,
        *,
        loop: bool = False,
        max_loops: Optional[int] = None,
        speed: Optional[float] = None,
        target_interval_ms: int = TARGET_INTERVAL_MS,
        detect_discontinuity: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__(
            camera_id,
            target_interval_ms=target_interval_ms,
            detect_discontinuity=detect_discontinuity,
            max_frames=max_frames,
            pacer=ReplayPacer(speed) if speed is not None else None,
        )
        self.path = path
        self.loop = loop
        self.max_loops = max_loops
        self.loops_played = 0
        self._cap: Any = None
        self._decoded = 0

    # ------------------------------------------------------------------ capture

    def _open_capture(self) -> None:
        import cv2  # lazy: contracts and CI import this package without OpenCV

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"video file not found: {self.path}")

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"OpenCV could not open {self.path}; the container or codec is "
                "unsupported by this build"
            )
        self._cap = cap
        self._cv2 = cv2

    def _read_raw(self) -> Optional[tuple[np.ndarray, Optional[float]]]:
        if self._cap is None:
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        self._decoded += 1
        pts_ms = self._cap.get(self._cv2.CAP_PROP_POS_MSEC)
        return frame, pts_ms

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -------------------------------------------------------------------- loops

    def _handle_exhausted(self, detail: str) -> bool:
        """End of clip. Restart under a new session if looping is enabled.

        The new session is the whole point. A loop boundary looks exactly like a
        hard scene cut -- every vehicle disappears and a different set appears
        one frame later -- and treating it as continuous is how you get a
        journey showing a car that teleported across the city.
        """
        if not self.loop:
            return False
        if self.max_loops is not None and self.loops_played + 1 >= self.max_loops:
            return False

        self.loops_played += 1
        self._close_capture()
        self._open_capture()
        self._start_session(
            reason="discontinuity",
            detail=f"replay restart (loop {self.loops_played})",
        )
        return True

    # --------------------------------------------------------------------- misc

    @property
    def frame_count(self) -> Optional[int]:
        """Total frames the container claims, or None.

        Advisory only. Container metadata is frequently wrong on CCTV exports,
        so this is used for progress reporting and never for timing.
        """
        if self._cap is None:
            return None
        count = int(self._cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "path": self.path,
                "frames_decoded": self._decoded,
                "loops_played": self.loops_played,
                "frame_count_reported": self.frame_count,
            }
        )
        return base
