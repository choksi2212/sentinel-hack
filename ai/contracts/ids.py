"""The identity model. Read this before anything else in the package.

Five different IDs exist and confusing them is the most likely silent-corruption
bug in the project. Canonical Contracts section 1.

    external_camera_id  global, provider-owned   'cam04'   permanent
    cameras.id          internal DB              UUID      permanent
    stream_session_id   one runtime connection   UUID      dies on reconnect
    track_id            one camera + session     INTEGER   REUSED across sessions
    event_id            global within TRINETRA   UUID      permanent

The only one of these that is dangerous is track_id, because it looks globally
meaningful and is not.
"""

import re
import uuid
from typing import NamedTuple

# Canonical Contracts section 1.1 -- LOCKED.
#
#   On the wire:  camera_id = the Sentinel catalogue ID, verbatim, lowercase.
#   FORBIDDEN:    CAM_001  CAM-001  Cam04  CAMERA_4  cam4
#
# Two or more digits, so cam4 is rejected (zero-pad drift) while a catalogue
# that grows past cam99 does not require a contract change. The catalogue is
# the authority for which IDs exist; this pattern only rules on their shape.
CAMERA_ID_PATTERN = re.compile(r"^cam[0-9]{2,}$")


def is_valid_camera_id(camera_id: object) -> bool:
    """Shape check only. Existence is the catalogue's business, not ours."""
    return isinstance(camera_id, str) and CAMERA_ID_PATTERN.match(camera_id) is not None


def require_camera_id(camera_id: object) -> str:
    """Validate at the adapter boundary so a bad ID cannot reach the wire.

    Raises ValueError naming the offending value. A malformed camera_id that
    slips through produces fixtures, DB seeds and UI screenshots that all have
    to be redone on the day live access arrives.
    """
    if not is_valid_camera_id(camera_id):
        raise ValueError(
            f"invalid camera_id {camera_id!r}: expected the Sentinel catalogue ID "
            f"verbatim and lowercase, e.g. 'cam04' (pattern {CAMERA_ID_PATTERN.pattern})"
        )
    return str(camera_id)


def new_session_id() -> str:
    """Mint a stream_session_id.

    Called on initial connect, on any reconnect after transport or decoder
    failure, on detected hard scene discontinuity, and on replay restart.
    Never reused. Canonical Contracts section 1.2.
    """
    return str(uuid.uuid4())


def new_event_id() -> str:
    """Mint an event_id. This is Mihir's idempotency key.

    Generated once per event and kept stable across POST retries -- that
    stability is the whole reason retry-safe ingestion works.
    """
    return str(uuid.uuid4())


class TrackKey(NamedTuple):
    """The invariant.

        TrackKey = (camera_id, stream_session_id, track_id)

    ByteTrack restarts numbering at 1 on every new session. Two vehicles seen
    on cam04 before and after a reconnect will both be track_id 42. Any dict,
    cache or DB row keyed on (camera_id, track_id) merges them into one vehicle
    and produces a journey showing a car crossing Ahmedabad in four seconds.

    Nobody notices for two days, and then every number in the project is
    suspect. This type exists so that the correct key is the easy key.
    """

    camera_id: str
    stream_session_id: str
    track_id: int

    def __str__(self) -> str:
        # Short session prefix keeps log lines readable while still
        # distinguishing sessions.
        return f"{self.camera_id}/{self.stream_session_id[:8]}/{self.track_id}"
