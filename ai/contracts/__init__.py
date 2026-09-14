"""Versioned contracts.

Normative source: docs/TRINETRA_Canonical_Contracts.md.

Nothing in this package may import cv2, torch, paddleocr or any other heavy
dependency. Mihir's ingest tests and CI's contract job import from here on a
machine with nothing but numpy installed, and that has to keep working.
"""

from ai.contracts.enums import (
    END_REASONS,
    MATCH_STATES,
    SCHEMA_VERSION,
    SOURCE_MODES,
    VEHICLE_TYPES,
    EndReason,
    MatchState,
    SourceMode,
    VehicleType,
)
from ai.contracts.event import (
    EvidenceBlock,
    EventEnvelope,
    ModelProvenance,
    PlateBlock,
    VehicleBlock,
)
from ai.contracts.frame import FrameEnvelope
from ai.contracts.ids import (
    CAMERA_ID_PATTERN,
    TrackKey,
    is_valid_camera_id,
    new_event_id,
    new_session_id,
    require_camera_id,
)
from ai.contracts.stages import (
    DetectorResult,
    FusedPlate,
    PlateCandidate,
    PlateObservation,
    TrackResult,
)

__all__ = [
    "CAMERA_ID_PATTERN",
    "END_REASONS",
    "MATCH_STATES",
    "SCHEMA_VERSION",
    "SOURCE_MODES",
    "VEHICLE_TYPES",
    "DetectorResult",
    "EndReason",
    "EventEnvelope",
    "EvidenceBlock",
    "FrameEnvelope",
    "FusedPlate",
    "MatchState",
    "ModelProvenance",
    "PlateBlock",
    "PlateCandidate",
    "PlateObservation",
    "SourceMode",
    "TrackKey",
    "TrackResult",
    "VehicleBlock",
    "VehicleType",
    "is_valid_camera_id",
    "new_event_id",
    "new_session_id",
    "require_camera_id",
]
