"""Temporal fusion -- the highest accuracy-per-effort stage in the pipeline.

Single-frame OCR on a 60 px plate is close to a coin flip. Three frames voting
is a decision. This costs zero GPU and requires zero training, and the
before-versus-after-consensus delta is the strongest technical claim the
project has.
"""

from ai.fusion.accumulator import CropBuffer, EvidenceAccumulator, TrackCrop
from ai.fusion.consensus import fuse, fuse_observations

__all__ = [
    "CropBuffer",
    "EvidenceAccumulator",
    "TrackCrop",
    "fuse",
    "fuse_observations",
]
