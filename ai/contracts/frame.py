"""FrameEnvelope -- the source boundary.

Every media source converges here. Downstream code must never call
cv2.VideoCapture or FFmpeg directly; if it does, the source-independence
invariant is gone and the offline-to-live swap stops being a config change.

Canonical Contracts section 2.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

SourceMode = Literal["live_rtsp", "live_hls", "file", "frames", "synthetic"]


# COPIED FROM CANONICAL CONTRACTS -- DO NOT EDIT HERE (Contracts section 2).
@dataclass(frozen=True)
class FrameEnvelope:
    camera_id: str                # "cam04" -- section 1.1
    stream_session_id: str        # UUID str, minted at connect
    frame_index: int              # monotonic within session, starts at 0
    pts_ms: int                   # SOURCE timeline position, milliseconds
    wallclock_utc: Optional[str]  # ISO-8601 Z; None for pure file replay
    frame_bgr: np.ndarray         # HxWx3 uint8, BGR (OpenCV order)
    width: int
    height: int
    source_mode: SourceMode
# END COPIED BLOCK.


def validate_frame(frame: FrameEnvelope) -> list[str]:
    """Return a list of contract violations; empty means valid.

    Validation lives outside the dataclass on purpose. The dataclass is a
    verbatim copy of the canonical contract and adding a __post_init__ to it
    would make the copy inexact. This runs at adapter boundaries and in tests,
    not on the hot path.
    """
    from ai.contracts.enums import SOURCE_MODES
    from ai.contracts.ids import is_valid_camera_id

    errors: list[str] = []

    if not is_valid_camera_id(frame.camera_id):
        errors.append(f"camera_id {frame.camera_id!r} is not a Sentinel catalogue ID")
    if not frame.stream_session_id:
        errors.append("stream_session_id is empty -- see Contracts section 1.2")
    if frame.frame_index < 0:
        errors.append(f"frame_index {frame.frame_index} is negative")
    if frame.pts_ms < 0:
        errors.append(f"pts_ms {frame.pts_ms} is negative")
    if frame.source_mode not in SOURCE_MODES:
        errors.append(f"source_mode {frame.source_mode!r} not in {SOURCE_MODES}")

    arr = frame.frame_bgr
    if not isinstance(arr, np.ndarray):
        errors.append(f"frame_bgr is {type(arr).__name__}, expected numpy.ndarray")
    else:
        if arr.ndim != 3 or arr.shape[2] != 3:
            errors.append(f"frame_bgr shape {arr.shape} is not HxWx3")
        if arr.dtype != np.uint8:
            errors.append(f"frame_bgr dtype {arr.dtype} is not uint8")
        if arr.ndim == 3 and (arr.shape[1], arr.shape[0]) != (frame.width, frame.height):
            errors.append(
                f"width/height {frame.width}x{frame.height} disagrees with "
                f"frame_bgr {arr.shape[1]}x{arr.shape[0]}"
            )

    return errors
