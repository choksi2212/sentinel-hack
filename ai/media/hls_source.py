"""SentinelHLSSource -- the password-protected HLS path. Manual section 4.3.

    https://cctv.corp8.cloud/<camera_id>/index.m3u8

Present because Contracts section 2.3 mandates five adapters and because the
organisers' catalogue is served over HLS, so the ability to read it is part of
proving vendor-neutrality. It is deliberately NOT the AI path: HLS is segmented,
so a detector waits on a whole segment before it sees the first frame of it,
adding seconds of latency that buy nothing. RTSP for inference, HLS for browsers
and for the demonstration that the source really is swappable.

**Credentials come from the environment. Never from a file in this repository.**

    SENTINEL_HLS_USERNAME
    SENTINEL_HLS_PASSWORD

They are the organisers' credentials, lent to us for the duration of a hackathon
on infrastructure we do not own. A committed password is not a style problem: it
is in the git history permanently, on a repository that gets submitted and read
by judges. Everything logged out of this adapter runs through redact_url first,
because stats payloads reach log files, benchmark JSON and occasionally a slide.
"""

import os
from typing import Optional
from urllib.parse import quote

from ai.media.live_base import ThreadedLiveSource

# Sentinel grid HLS endpoint. Contracts section 1.1.
SENTINEL_HLS_TEMPLATE = "https://cctv.corp8.cloud/{camera_id}/index.m3u8"

USERNAME_ENV = "SENTINEL_HLS_USERNAME"
PASSWORD_ENV = "SENTINEL_HLS_PASSWORD"


def sentinel_hls_url(
    camera_id: str,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Build the HLS URL, embedding credentials only if both are present.

    quote() with an empty safe set on both parts: a password containing @ or /
    would otherwise silently reshape the URL into a request for a different host,
    and the resulting error says nothing about the real cause.
    """
    url = SENTINEL_HLS_TEMPLATE.format(camera_id=camera_id)
    if not (username and password):
        return url

    scheme, _, remainder = url.partition("://")
    user = quote(username, safe="")
    secret = quote(password, safe="")
    return f"{scheme}://{user}:{secret}@{remainder}"


class SentinelHLSSource(ThreadedLiveSource):
    """Live HLS, depth-1 buffered, with reconnect and session rotation."""

    source_mode = "live_hls"

    def __init__(
        self,
        camera_id: str,
        *,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(camera_id, **kwargs)  # type: ignore[arg-type]
        self._explicit_url = url
        self.username = username or os.environ.get(USERNAME_ENV)
        self.password = password or os.environ.get(PASSWORD_ENV)

    def _stream_url(self) -> str:
        if self._explicit_url:
            return self._explicit_url
        return sentinel_hls_url(
            self.camera_id, username=self.username, password=self.password
        )

    @property
    def has_credentials(self) -> bool:
        """False means the request will go out unauthenticated and get a 401.

        Worth checking before a demo. An unauthenticated HLS attempt fails in a
        way that looks like a network problem, and the fix is two environment
        variables rather than anything to do with the network.
        """
        return bool(self.username and self.password)
