"""EventEnvelope v1.1 -- the CV-to-backend boundary.

This is the only thing Mihir consumes from the AI side, and the only artefact
Parth's UI ultimately renders. Canonical Contracts section 3.

The dataclasses here exist so that a required field cannot be forgotten, and
validate() exists so that a malformed event is caught on this side of the wire
rather than as a 422 during a demo.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ai.contracts.enums import (
    MATCH_STATES,
    SCHEMA_VERSION,
    SOURCE_MODES,
    VEHICLE_TYPES,
)
from ai.contracts.ids import is_valid_camera_id
from ai.contracts.timebase import is_timezone_aware_iso

BBox = tuple[int, int, int, int]

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _is_uuid(value: object) -> bool:
    if not isinstance(value, str) or not _UUID_RE.match(value):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _in_unit_interval(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _bbox_list(bbox: Optional[BBox]) -> Optional[list[int]]:
    return None if bbox is None else [int(v) for v in bbox]


def _bbox_tuple(raw: object) -> Optional[BBox]:
    if raw is None:
        return None
    values = [int(v) for v in raw]  # type: ignore[union-attr]
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(values)}")
    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True)
class VehicleBlock:
    type: str
    confidence: float
    bbox_xyxy: BBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": _bbox_list(self.bbox_xyxy),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VehicleBlock":
        return cls(
            type=raw["type"],
            confidence=float(raw["confidence"]),
            bbox_xyxy=_bbox_tuple(raw["bbox_xyxy"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PlateBlock:
    """The plate block. The whole object is null when no plate was read.

    plate: null is a valid, expected, correct event. It says a vehicle passed
    and could not be identified, which is real information. Fabricating a guess
    to avoid a null is the worst failure mode available to this system, and
    unlike a null it is invisible downstream. Contracts section 3.2.
    """

    raw: str                       # exactly what OCR returned, unmodified
    normalized: Optional[str]      # may be null if normalization yields ""
    confidence: float
    match_state: str
    plate_width_px: int            # required -- width buckets depend on it
    evidence_count: int
    bbox_xyxy: Optional[BBox] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "confidence": round(float(self.confidence), 4),
            "match_state": self.match_state,
            "plate_width_px": int(self.plate_width_px),
            "evidence_count": int(self.evidence_count),
            "bbox_xyxy": _bbox_list(self.bbox_xyxy),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlateBlock":
        return cls(
            raw=raw["raw"],
            normalized=raw.get("normalized"),
            confidence=float(raw["confidence"]),
            match_state=raw["match_state"],
            plate_width_px=int(raw["plate_width_px"]),
            evidence_count=int(raw["evidence_count"]),
            bbox_xyxy=_bbox_tuple(raw.get("bbox_xyxy")),
        )


@dataclass(frozen=True)
class EvidenceBlock:
    """Snapshot URIs.

    Treated by the backend as untrusted metadata: never dereferenced, never
    used to build a filesystem path. Contracts section 3.1.
    """

    snapshot_uri: Optional[str] = None
    plate_crop_uri: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_uri": self.snapshot_uri, "plate_crop_uri": self.plate_crop_uri}

    @classmethod
    def from_dict(cls, raw: Optional[dict[str, Any]]) -> "EvidenceBlock":
        raw = raw or {}
        return cls(
            snapshot_uri=raw.get("snapshot_uri"),
            plate_crop_uri=raw.get("plate_crop_uri"),
        )


@dataclass(frozen=True)
class ModelProvenance:
    """Which models produced this event.

    Required. Without it no benchmark number is citeable -- a report that says
    "we hit 74%" and cannot say with which weights is not evidence.
    """

    detector: str
    plate_detector: str
    ocr: str
    tracker: str
    pipeline_version: str
    detector_weights_sha256: Optional[str] = None
    plate_detector_weights_sha256: Optional[str] = None
    ocr_weights_sha256: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "detector": self.detector,
            "detector_weights_sha256": self.detector_weights_sha256,
            "plate_detector": self.plate_detector,
            "ocr": self.ocr,
            "tracker": self.tracker,
            "pipeline_version": self.pipeline_version,
        }
        # Only emitted when known, so the common shape stays exactly the
        # canonical example and does not grow noise for the stub backends.
        if self.plate_detector_weights_sha256:
            out["plate_detector_weights_sha256"] = self.plate_detector_weights_sha256
        if self.ocr_weights_sha256:
            out["ocr_weights_sha256"] = self.ocr_weights_sha256
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelProvenance":
        return cls(
            detector=raw["detector"],
            plate_detector=raw["plate_detector"],
            ocr=raw["ocr"],
            tracker=raw["tracker"],
            pipeline_version=raw["pipeline_version"],
            detector_weights_sha256=raw.get("detector_weights_sha256"),
            plate_detector_weights_sha256=raw.get("plate_detector_weights_sha256"),
            ocr_weights_sha256=raw.get("ocr_weights_sha256"),
        )

    @property
    def is_citeable(self) -> bool:
        """True when every shipped model reports a weights hash.

        A benchmark run whose provenance is not citeable may be recorded as a
        diagnostic but must not appear in a submission claim.
        """
        return bool(self.detector_weights_sha256 and self.plate_detector_weights_sha256)


@dataclass(frozen=True)
class EventEnvelope:
    """One vehicle sighting, ready to POST.

    Field order in to_dict() matches the canonical example so that a diff
    against the contract document is readable by eye.
    """

    event_id: str
    camera_id: str
    stream_session_id: str
    track_id: int
    observed_at: str
    source_pts_ms: int
    source_mode: str
    vehicle: VehicleBlock
    image_quality: float
    model: ModelProvenance
    plate: Optional[PlateBlock] = None
    evidence: EvidenceBlock = field(default_factory=EvidenceBlock)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "stream_session_id": self.stream_session_id,
            "track_id": int(self.track_id),
            "observed_at": self.observed_at,
            "source_pts_ms": int(self.source_pts_ms),
            "source_mode": self.source_mode,
            "vehicle": self.vehicle.to_dict(),
            "plate": None if self.plate is None else self.plate.to_dict(),
            "image_quality": round(float(self.image_quality), 4),
            "evidence": self.evidence.to_dict(),
            "model": self.model.to_dict(),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EventEnvelope":
        plate_raw = raw.get("plate")
        return cls(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            event_id=raw["event_id"],
            camera_id=raw["camera_id"],
            stream_session_id=raw["stream_session_id"],
            track_id=int(raw["track_id"]),
            observed_at=raw["observed_at"],
            source_pts_ms=int(raw["source_pts_ms"]),
            source_mode=raw["source_mode"],
            vehicle=VehicleBlock.from_dict(raw["vehicle"]),
            plate=None if plate_raw is None else PlateBlock.from_dict(plate_raw),
            image_quality=float(raw["image_quality"]),
            evidence=EvidenceBlock.from_dict(raw.get("evidence")),
            model=ModelProvenance.from_dict(raw["model"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "EventEnvelope":
        return cls.from_dict(json.loads(text))

    def validate(self) -> list[str]:
        """Contract violations, empty when valid. Never raises."""
        return validate_payload(self.to_dict())


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Validate a raw event dict against Contracts section 3.1.

    Ordered to mirror Mihir's ingest validation (section 6.1) so that anything
    this accepts, ingest accepts. Each message names the specific failing
    field, because a generic "invalid event" tells you nothing at 2am.
    """
    errors: list[str] = []

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {version!r}"
        )

    if not _is_uuid(payload.get("event_id")):
        errors.append(f"event_id: not a UUID ({payload.get('event_id')!r})")

    if not is_valid_camera_id(payload.get("camera_id")):
        errors.append(
            f"camera_id: {payload.get('camera_id')!r} is not a Sentinel catalogue ID "
            "(expected e.g. 'cam04')"
        )

    if not _is_uuid(payload.get("stream_session_id")):
        errors.append(
            "stream_session_id: not a UUID -- never omit this, see Contracts section 1.2"
        )

    track_id = payload.get("track_id")
    if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
        errors.append(f"track_id: expected integer >= 0, got {track_id!r}")

    if not is_timezone_aware_iso(payload.get("observed_at")):
        errors.append("observed_at: must be timezone-aware ISO-8601")

    pts = payload.get("source_pts_ms")
    if not isinstance(pts, int) or isinstance(pts, bool) or pts < 0:
        errors.append(f"source_pts_ms: expected integer >= 0, got {pts!r}")

    if payload.get("source_mode") not in SOURCE_MODES:
        errors.append(
            f"source_mode: {payload.get('source_mode')!r} not in {list(SOURCE_MODES)}"
        )

    vehicle = payload.get("vehicle")
    if not isinstance(vehicle, dict):
        errors.append("vehicle: missing or not an object")
    else:
        if vehicle.get("type") not in VEHICLE_TYPES:
            errors.append(
                f"vehicle.type: {vehicle.get('type')!r} not in {list(VEHICLE_TYPES)}"
            )
        if not _in_unit_interval(vehicle.get("confidence")):
            errors.append(
                f"vehicle.confidence: expected 0.0..1.0, got {vehicle.get('confidence')!r}"
            )
        errors.extend(_bbox_errors("vehicle.bbox_xyxy", vehicle.get("bbox_xyxy"), required=True))

    if not _in_unit_interval(payload.get("image_quality")):
        errors.append(
            f"image_quality: expected 0.0..1.0, got {payload.get('image_quality')!r}"
        )

    if "plate" not in payload:
        errors.append("plate: key missing -- emit null explicitly when unreadable")
    else:
        errors.extend(_plate_errors(payload.get("plate")))

    model = payload.get("model")
    if not isinstance(model, dict):
        errors.append("model: missing or not an object -- provenance is required")
    else:
        for key in ("detector", "plate_detector", "ocr", "tracker", "pipeline_version"):
            if not model.get(key):
                errors.append(f"model.{key}: missing")

    evidence = payload.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        errors.append("evidence: must be an object or absent")

    return errors


def _plate_errors(plate: object) -> list[str]:
    if plate is None:
        return []  # plate: null is valid and correct -- Contracts section 3.2

    if not isinstance(plate, dict):
        return [f"plate: expected object or null, got {type(plate).__name__}"]

    errors: list[str] = []

    if not isinstance(plate.get("raw"), str) or not plate.get("raw"):
        errors.append("plate.raw: required when plate is present, and never modified")

    normalized = plate.get("normalized")
    if normalized is not None and not isinstance(normalized, str):
        errors.append(f"plate.normalized: expected string or null, got {normalized!r}")

    if not _in_unit_interval(plate.get("confidence")):
        errors.append(
            f"plate.confidence: expected 0.0..1.0, got {plate.get('confidence')!r}"
        )

    match_state = plate.get("match_state")
    if match_state not in MATCH_STATES:
        errors.append(
            f"plate.match_state: {match_state!r} not in {list(MATCH_STATES)} -- "
            "Mihir's CHECK constraint rejects anything else with a 422"
        )

    width = plate.get("plate_width_px")
    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        errors.append(
            f"plate.plate_width_px: required integer >= 0, got {width!r} -- "
            "width-bucket reporting is impossible without it"
        )

    count = plate.get("evidence_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        errors.append(f"plate.evidence_count: expected integer >= 1, got {count!r}")

    errors.extend(_bbox_errors("plate.bbox_xyxy", plate.get("bbox_xyxy"), required=False))

    # Cross-field consistency. Neither of these is spelled out as a row in the
    # field table, but both follow from section 3.3: unreadable means the plate
    # was located and no usable text came out, so a normalized string and an
    # unreadable state cannot both be true.
    if normalized is None and match_state != "unreadable":
        errors.append(
            f"plate.match_state: normalized is null so match_state must be "
            f"'unreadable', got {match_state!r}"
        )
    if match_state == "unreadable" and normalized:
        errors.append(
            f"plate.normalized: match_state is 'unreadable' but normalized is "
            f"{normalized!r} -- one of the two is wrong"
        )

    return errors


def _bbox_errors(label: str, bbox: object, *, required: bool) -> list[str]:
    if bbox is None:
        return [f"{label}: required"] if required else []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [f"{label}: expected [x1, y1, x2, y2]"]
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
        return [f"{label}: all four values must be integers"]
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return [f"{label}: degenerate box {list(bbox)} -- xyxy order, x2>x1 and y2>y1"]
    return []
