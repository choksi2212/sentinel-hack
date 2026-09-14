"""ai.media -- the source-independence boundary.

Five adapters, one FrameEnvelope. Contracts section 2.3:

    file        VideoFileSource        recorded clip, the primary dev path
    frames      FrameSequenceSource    a directory of images
    live_rtsp   SentinelRTSPSource     the live AI path, TCP transport
    live_hls    SentinelHLSSource      the organisers' HLS feed
    synthetic   SyntheticReplaySource  deterministic frames with known answers

No business layer below this package may know which one it is reading from. The
single acceptance test for that is build_source: if switching config/offline.yaml
for config/live.yaml requires no other change anywhere, source independence holds.

Importing this module imports no OpenCV. Every adapter defers `import cv2` to the
moment it actually opens something, so the contract tests, the fusion tests and
the whole synthetic path run on a machine with numpy and nothing else.
"""

from ai.media.backoff import BACKOFF_BASE_MS, BACKOFF_MAX_MS, ReconnectPolicy
from ai.media.base import BaseMediaSource, MediaSource, SessionChange
from ai.media.discontinuity import (
    DISCONTINUITY_CORRELATION_THRESHOLD,
    DiscontinuityDetector,
)
from ai.media.factory import SourceConfigError, build_source
from ai.media.file_source import VideoFileSource
from ai.media.frames_source import FrameSequenceSource
from ai.media.hls_source import SentinelHLSSource, sentinel_hls_url
from ai.media.live_base import ThreadedLiveSource, redact_url
from ai.media.pacing import ReplayPacer
from ai.media.pts import (
    MAX_FORWARD_JUMP_MS,
    PtsAction,
    PtsValidator,
    PtsVerdict,
)
from ai.media.rtsp_source import (
    FFMPEG_CAPTURE_OPTIONS,
    SentinelRTSPSource,
    sentinel_rtsp_url,
)
from ai.media.sampler import TARGET_INTERVAL_MS, LatestFrameBuffer, PtsSampler
from ai.media.synthetic_source import (
    DEFAULT_SEED,
    FrameTruth,
    SyntheticFaults,
    SyntheticReplaySource,
    VehicleTruth,
)

__all__ = [
    "BACKOFF_BASE_MS",
    "BACKOFF_MAX_MS",
    "BaseMediaSource",
    "DEFAULT_SEED",
    "DISCONTINUITY_CORRELATION_THRESHOLD",
    "DiscontinuityDetector",
    "FFMPEG_CAPTURE_OPTIONS",
    "FrameSequenceSource",
    "FrameTruth",
    "LatestFrameBuffer",
    "MAX_FORWARD_JUMP_MS",
    "MediaSource",
    "PtsAction",
    "PtsSampler",
    "PtsValidator",
    "PtsVerdict",
    "ReconnectPolicy",
    "ReplayPacer",
    "SentinelHLSSource",
    "SentinelRTSPSource",
    "SessionChange",
    "SourceConfigError",
    "SyntheticFaults",
    "SyntheticReplaySource",
    "TARGET_INTERVAL_MS",
    "ThreadedLiveSource",
    "VehicleTruth",
    "VideoFileSource",
    "build_source",
    "redact_url",
    "sentinel_hls_url",
    "sentinel_rtsp_url",
]
