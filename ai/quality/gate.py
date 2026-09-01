"""The vehicle quality gate -- where the GPU budget is won.

Rejects vehicles cheaply, before spending plate-detection compute on them.
Owner's manual section 5.4:

    Vehicle bbox height < 60 px            plate cannot exceed ~20 px
    Bbox touches frame edge on >=2 sides   plate likely cut off
    Bbox shrinking and already small       vehicle departing, better frames taken
    Detector confidence < 0.35             probably not a vehicle

Every rejection is counted by reason. That count is the evidence for the claim
that the gate is worth having, and it is also the first place to look when
recall collapses -- a gate tuned too tight looks exactly like a bad detector.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ai.contracts.ids import TrackKey
from ai.contracts.stages import TrackResult

MIN_VEHICLE_HEIGHT_PX = 60
MIN_DETECTOR_CONFIDENCE = 0.35
EDGE_MARGIN_PX = 2
MAX_EDGE_CONTACTS = 1  # 2 or more is a reject

# "Already small" for the departing-vehicle rule. A vehicle above this height is
# still worth a plate attempt even while shrinking.
DEPARTING_HEIGHT_PX = 110

# Area must have dropped by at least this fraction to count as departing rather
# than as detector jitter.
DEPARTING_SHRINK_RATIO = 0.85


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: Optional[str] = None
    detail: Optional[str] = None

    def __bool__(self) -> bool:
        return self.passed


_PASS = GateDecision(passed=True)


class VehicleGate:
    """Stateful gate. Holds one previous bbox area per TrackKey.

    Stateful because the departing rule needs history, and keyed on TrackKey
    for the usual reason: keyed on (camera_id, track_id) it would compare the
    area of one car before a reconnect against a different car after it, and
    reject or admit on the basis of a comparison between two vehicles.
    """

    def __init__(
        self,
        *,
        min_height_px: int = MIN_VEHICLE_HEIGHT_PX,
        min_confidence: float = MIN_DETECTOR_CONFIDENCE,
        departing_height_px: int = DEPARTING_HEIGHT_PX,
    ) -> None:
        self.min_height_px = min_height_px
        self.min_confidence = min_confidence
        self.departing_height_px = departing_height_px
        self._last_area: dict[TrackKey, int] = {}
        self.rejections: Counter[str] = Counter()
        self.passes = 0

    def check(
        self,
        track: TrackResult,
        frame_width: int,
        frame_height: int,
    ) -> GateDecision:
        """Evaluate one tracked vehicle. Cheapest tests first."""
        decision = self._evaluate(track, frame_width, frame_height)

        # Update history after evaluating, so this frame is compared against
        # the previous frame rather than against itself.
        self._last_area[track.track_key] = max(0, track.width) * max(0, track.height)

        if decision.passed:
            self.passes += 1
        else:
            self.rejections[decision.reason or "unknown"] += 1
        return decision

    def _evaluate(
        self,
        track: TrackResult,
        frame_width: int,
        frame_height: int,
    ) -> GateDecision:
        if track.confidence < self.min_confidence:
            return GateDecision(
                False,
                "low_detector_confidence",
                f"{track.confidence:.2f} < {self.min_confidence:.2f}",
            )

        height = track.height
        if height < self.min_height_px:
            return GateDecision(
                False,
                "vehicle_too_small",
                f"height {height}px < {self.min_height_px}px, so the plate cannot "
                f"exceed roughly {height // 3}px",
            )

        contacts = self._edge_contacts(track, frame_width, frame_height)
        if contacts > MAX_EDGE_CONTACTS:
            return GateDecision(
                False,
                "edge_clipped",
                f"bbox touches {contacts} frame edges; plate likely cut off",
            )

        previous_area = self._last_area.get(track.track_key)
        if previous_area:
            area = max(0, track.width) * max(0, track.height)
            shrinking = area < previous_area * DEPARTING_SHRINK_RATIO
            if shrinking and height < self.departing_height_px:
                return GateDecision(
                    False,
                    "departing",
                    f"area {area} down from {previous_area} at height {height}px; "
                    "better frames already captured",
                )

        return _PASS

    @staticmethod
    def _edge_contacts(track: TrackResult, frame_width: int, frame_height: int) -> int:
        x1, y1, x2, y2 = track.bbox_xyxy
        return sum(
            (
                x1 <= EDGE_MARGIN_PX,
                y1 <= EDGE_MARGIN_PX,
                x2 >= frame_width - EDGE_MARGIN_PX,
                y2 >= frame_height - EDGE_MARGIN_PX,
            )
        )

    def flush_session(self, stream_session_id: str) -> None:
        """Drop history for one session. Called whenever a session ends."""
        for key in [k for k in self._last_area if k.stream_session_id == stream_session_id]:
            del self._last_area[key]

    def reset(self) -> None:
        self._last_area.clear()

    def stats(self) -> dict[str, object]:
        total = self.passes + sum(self.rejections.values())
        return {
            "evaluated": total,
            "passed": self.passes,
            "rejected": sum(self.rejections.values()),
            "pass_rate": round(self.passes / total, 4) if total else None,
            "by_reason": dict(self.rejections),
        }
