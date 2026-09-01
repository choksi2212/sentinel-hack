"""SentinelRTSPSource -- the live path for the AI pipeline. Manual section 4.2.

One thing in this file is non-negotiable:

    OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"

Over UDP, FFmpeg silently accepts packet loss and hands back frames with torn
macroblocks across them. Those frames decode, display, and look merely "a bit
glitchy" to a human -- and they destroy plate OCR, because the tear lands on a
region 40 pixels wide. The failure presents as a model that cannot read plates,
which sends you debugging the wrong subsystem for a day. TCP costs a little
latency and removes the entire class of problem.

RTSP is the AI path specifically. HLS is segmented, which adds seconds of
buffering latency for no benefit to a detector -- fine for a browser, wrong here.

Everything else lives in ai/media/live_base.py, shared with the HLS adapter.
"""

from typing import Optional

from ai.media.live_base import ThreadedLiveSource

# Sentinel grid RTSP endpoint. Contracts section 1.1.
SENTINEL_RTSP_HOST = "103.250.160.189"
SENTINEL_RTSP_PORT = 8554
SENTINEL_RTSP_TEMPLATE = "rtsp://{host}:{port}/stream/{camera_id}"

# LOCKED. Anything else is a bug, not a preference.
FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"


def sentinel_rtsp_url(camera_id: str) -> str:
    return SENTINEL_RTSP_TEMPLATE.format(
        host=SENTINEL_RTSP_HOST, port=SENTINEL_RTSP_PORT, camera_id=camera_id
    )


class SentinelRTSPSource(ThreadedLiveSource):
    """Live RTSP over TCP, depth-1 buffered, with reconnect and session rotation."""

    source_mode = "live_rtsp"

    def __init__(
        self,
        camera_id: str,
        *,
        url: Optional[str] = None,
        transport_options: str = FFMPEG_CAPTURE_OPTIONS,
        **kwargs: object,
    ) -> None:
        super().__init__(camera_id, **kwargs)  # type: ignore[arg-type]
        self.url = url or sentinel_rtsp_url(self.camera_id)
        self.transport_options = transport_options

    def _stream_url(self) -> str:
        return self.url

    @property
    def capture_options(self) -> Optional[str]:
        return self.transport_options
