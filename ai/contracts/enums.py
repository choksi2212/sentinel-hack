"""Locked enumerations.

Every value here has a CHECK constraint behind it in Mihir's schema. Emitting a
value that is not in one of these tuples returns 422 from ingest, so the tuples
exist to be asserted against in tests rather than remembered.

Canonical Contracts sections 2, 3.1, 3.3, 5.2.
"""

from typing import Literal, get_args

# Canonical Contracts section 3.1. Backend rejects unknown majors.
SCHEMA_VERSION = "1.1"

SourceMode = Literal["live_rtsp", "live_hls", "file", "frames", "synthetic"]
VehicleType = Literal["car", "motorcycle", "bus", "truck", "auto_rickshaw", "other"]
MatchState = Literal["exact", "probable", "low_confidence", "unreadable"]
EndReason = Literal["eof", "reconnect", "discontinuity", "shutdown", "error"]

SOURCE_MODES: tuple[str, ...] = get_args(SourceMode)
VEHICLE_TYPES: tuple[str, ...] = get_args(VehicleType)
MATCH_STATES: tuple[str, ...] = get_args(MatchState)
END_REASONS: tuple[str, ...] = get_args(EndReason)

# The two source modes that read from the Sentinel grid in real time. Used by
# the LIVE/REPLAY badge and by anything that must refuse to make a latency
# claim from replayed footage.
LIVE_SOURCE_MODES: frozenset[str] = frozenset({"live_rtsp", "live_hls"})

# match_state values that are allowed to raise an alert. alerts.match_state has
# a CHECK on exactly this pair; low_confidence and unreadable never alert.
# Canonical Contracts sections 3.4 and 5.7.
ALERTABLE_MATCH_STATES: frozenset[str] = frozenset({"exact", "probable"})


def is_live(source_mode: str) -> bool:
    """True only for the two real-time grid modes.

    The UI badge and every performance claim depend on this being the single
    place the question is answered. A replayed clip at 5x acceleration is not
    live no matter how fast it runs.
    """
    return source_mode in LIVE_SOURCE_MODES
