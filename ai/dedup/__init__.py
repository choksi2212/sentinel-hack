"""Sighting dedup. Canonical Contracts section 4.7."""

from ai.dedup.key import (
    DEDUP_WINDOW_SECONDS,
    SightingDeduper,
    dedupe_key,
    dedupe_key_for,
)

__all__ = [
    "DEDUP_WINDOW_SECONDS",
    "SightingDeduper",
    "dedupe_key",
    "dedupe_key_for",
]
