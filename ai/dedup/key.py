"""Sighting dedup. Canonical Contracts section 4.7.

    dedupe_key = sha256(f"{camera_id}|{stream_session_id}|{track_id}|{normalized_plate or ''}")
    DEDUP_WINDOW_SECONDS = 10

Mihir enforces this in the database with a UNIQUE constraint, which is where it
actually matters. The AI side computes the same key for two reasons: so the
value on the wire is already correct, and so a track that gets finalized twice
(the max-duration cap does exactly that to a vehicle stopped at a signal) does
not turn into two POSTs for one vehicle.

Without dedup, one row per frame per vehicle lands in the database and search
results become unusable within minutes.
"""

import hashlib
from collections import OrderedDict
from typing import Optional

from ai.contracts.ids import TrackKey
from ai.contracts.timebase import parse_iso

DEDUP_WINDOW_SECONDS = 10

# Bounded so a long run cannot grow the cache without limit. At 10 fps across
# 30 cameras this holds several minutes of history, far more than the 10 second
# window needs.
_MAX_CACHE_ENTRIES = 20_000


def dedupe_key(
    camera_id: str,
    stream_session_id: str,
    track_id: int,
    normalized_plate: Optional[str],
) -> str:
    """The locked key. Note the empty string for an unreadable plate.

    Using the plate-less form for unreadable vehicles means two different
    unidentified vehicles on the same camera still get distinct keys, because
    track_id differs. Dropping track_id from the key would collapse every
    unreadable vehicle on a camera into one row.
    """
    payload = f"{camera_id}|{stream_session_id}|{track_id}|{normalized_plate or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedupe_key_for(track_key: TrackKey, normalized_plate: Optional[str]) -> str:
    return dedupe_key(
        track_key.camera_id,
        track_key.stream_session_id,
        track_key.track_id,
        normalized_plate,
    )


class SightingDeduper:
    """Suppresses repeat emissions of the same sighting within the window.

    Keeps the best evidence rather than the first: on a repeat, if the new
    observation scores higher on ocr_confidence x image_quality, the caller is
    told to re-emit so the better snapshot wins. Contracts section 4.7 requires
    exactly that behaviour on the database side, and matching it here keeps the
    two from disagreeing about which evidence is canonical.
    """

    def __init__(self, window_seconds: int = DEDUP_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        # key -> (last_observed_at_iso, best_weight)
        self._seen: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.suppressed = 0
        self.upgraded = 0

    def should_emit(
        self,
        key: str,
        observed_at: str,
        evidence_weight: float = 0.0,
    ) -> bool:
        """True when this sighting should go on the wire.

        False means a better or equal version of the same sighting was emitted
        within the window and this one adds nothing.
        """
        previous = self._seen.get(key)
        if previous is None:
            self._remember(key, observed_at, evidence_weight)
            return True

        last_seen, best_weight = previous
        try:
            elapsed = (parse_iso(observed_at) - parse_iso(last_seen)).total_seconds()
        except (ValueError, TypeError):
            # An unparseable timestamp is a contract violation that validation
            # will catch and report properly. Failing open here means the event
            # still reaches ingest, which rejects it with a specific field --
            # far more useful than silently dropping it in the deduper.
            self._remember(key, observed_at, evidence_weight)
            return True

        if elapsed > self.window_seconds:
            self._remember(key, observed_at, evidence_weight)
            return True

        if evidence_weight > best_weight:
            self.upgraded += 1
            self._remember(key, observed_at, evidence_weight)
            return True

        self.suppressed += 1
        return False

    def _remember(self, key: str, observed_at: str, weight: float) -> None:
        previous = self._seen.pop(key, None)
        best = weight if previous is None else max(previous[1], weight)
        self._seen[key] = (observed_at, best)
        while len(self._seen) > _MAX_CACHE_ENTRIES:
            self._seen.popitem(last=False)

    def stats(self) -> dict[str, object]:
        return {
            "tracked_keys": len(self._seen),
            "suppressed": self.suppressed,
            "upgraded": self.upgraded,
            "window_seconds": self.window_seconds,
        }
