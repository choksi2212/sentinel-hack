"""Internal stage contracts.

These are Manas's alone -- nobody outside ai/ consumes them. But every one
carries stream_session_id, because a stage contract without a session is where
the track-merge bug gets reintroduced six days from now by someone in a hurry.

Canonical Contracts section 3.3 (owner's manual section 3.3).
"""

from dataclasses import dataclass
from typing import Optional

from ai.contracts.ids import TrackKey

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 -- always xyxy, never xywh


@dataclass(frozen=True)
class DetectorResult:
    """One vehicle box out of the vehicle detector, before tracking.

    Frame-local: it has no identity yet. That is what makes it distinct from
    TrackResult and why it deliberately carries no session.
    """

    bbox_xyxy: BBox
    class_name: str            # car|motorcycle|bus|truck|auto_rickshaw|other
    confidence: float

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class TrackResult:
    camera_id: str
    stream_session_id: str     # never omit
    track_id: int
    bbox_xyxy: BBox
    class_name: str
    confidence: float
    frame_index: int
    pts_ms: int

    @property
    def track_key(self) -> TrackKey:
        return TrackKey(self.camera_id, self.stream_session_id, self.track_id)

    @property
    def width(self) -> int:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> int:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


@dataclass(frozen=True)
class PlateCandidate:
    """A plate box found inside a vehicle crop, before OCR.

    plate_bbox_xyxy is in FULL FRAME coordinates, already mapped back from the
    crop. Keeping crop-local coordinates here was tempting and would have put
    an offset bug into every snapshot URI and every width measurement.
    """

    plate_bbox_xyxy: BBox
    detector_confidence: float

    @property
    def plate_width_px(self) -> int:
        return self.plate_bbox_xyxy[2] - self.plate_bbox_xyxy[0]

    @property
    def plate_height_px(self) -> int:
        return self.plate_bbox_xyxy[3] - self.plate_bbox_xyxy[1]


@dataclass(frozen=True)
class PlateObservation:
    """One OCR read of one plate in one frame.

    This is the unit temporal fusion consumes and the unit that lands in
    plate_observations -- the audit trail that proves the consensus.
    """

    camera_id: str
    stream_session_id: str     # never omit
    track_id: int
    plate_bbox_xyxy: BBox
    plate_width_px: int
    plate_raw: str
    ocr_confidence: float
    image_quality: float
    frame_index: int
    pts_ms: int
    observed_at: str

    @property
    def track_key(self) -> TrackKey:
        return TrackKey(self.camera_id, self.stream_session_id, self.track_id)

    @property
    def fusion_weight(self) -> float:
        """ocr_confidence * image_quality -- the weight used by fuse().

        Also the tie-break for "keep the best evidence" on dedup update.
        """
        return self.ocr_confidence * self.image_quality


@dataclass(frozen=True)
class FusedPlate:
    """Output of temporal consensus over all observations for one TrackKey.

    confidence is a share of total evidence, not a probability. Do not multiply
    it by anything and present the product. Canonical Contracts section 4.4.
    """

    normalized: str
    confidence: float
    evidence_count: int
    best_observation: Optional[PlateObservation] = None
    grammar_ok: bool = True
    total_observations: int = 0

    @property
    def calibration_band(self) -> str:
        """HIGH / MEDIUM / LOW per Canonical Contracts section 4.4.

        Reported as an evidence band, never as a percentage.
        """
        if self.evidence_count >= 3 and self.confidence >= 0.85:
            return "HIGH"
        if self.evidence_count >= 2:
            return "MEDIUM"
        return "LOW"
