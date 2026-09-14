"""Multi-object tracking -- stage 4 of the fourteen in Contracts section 10.

    VehicleTracker      the Protocol every tracker satisfies
    BaseTracker         session identity, ID allocation, TrackResult construction
    ByteTracker         MIT, two-stage association + Kalman, the one that ships
    IOUTracker          greedy IoU, no motion model, numpy only
    OracleTracker       identity from ground truth, for isolating error sources
    ScriptedTracker     a fixed table, for downstream tests
    TrackerRegistry     one tracker per camera, flushed on session change
    build_tracker       config block -> instance
    build_registry      config block -> registry, which is what the worker uses

The one thing to carry away from this package: **TrackKey is three parts**, and
the third is the stream session. A tracker restarts its numbering at 1 after a
reconnect, so a two-part key silently merges an unrelated vehicle into an existing
track and fusion then emits one plate belonging to neither. TrackerRegistry is what
makes the session boundary reach the tracker in time; ai/track/base.py explains the
failure in full.
"""

from ai.track.assignment import (
    MAX_IOU_COST,
    fuse_detection_score,
    gate_cost,
    iou_cost,
    iou_matrix,
    solve,
)
from ai.track.base import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MIN_HITS,
    DEFAULT_TRACK_BUFFER,
    BaseTracker,
    VehicleTracker,
)
from ai.track.bytetrack import GATING_THRESHOLD, ByteTracker
from ai.track.factory import (
    SHIPPABLE_TRACKERS,
    TRACKER_NAMES,
    TrackerConfigError,
    build_registry,
    build_tracker,
    describe_tracker,
    normalize_tracker_config,
    tracker_factory,
    tracker_ships,
)
from ai.track.kalman import KalmanBoxFilter, cxcyah_to_xyxy, xyxy_to_cxcyah
from ai.track.registry import SessionMismatchError, TrackerRegistry
from ai.track.stub import IOUTracker, OracleTracker, ScriptedTracker
from ai.track.track import CONFIRMED, LOST, REMOVED, TENTATIVE, Track

__all__ = [
    "BaseTracker",
    "ByteTracker",
    "CONFIRMED",
    "DEFAULT_HIGH_THRESHOLD",
    "DEFAULT_LOW_THRESHOLD",
    "DEFAULT_MIN_HITS",
    "DEFAULT_TRACK_BUFFER",
    "GATING_THRESHOLD",
    "IOUTracker",
    "KalmanBoxFilter",
    "LOST",
    "MAX_IOU_COST",
    "OracleTracker",
    "REMOVED",
    "SHIPPABLE_TRACKERS",
    "ScriptedTracker",
    "SessionMismatchError",
    "TENTATIVE",
    "TRACKER_NAMES",
    "Track",
    "TrackerConfigError",
    "TrackerRegistry",
    "VehicleTracker",
    "build_registry",
    "build_tracker",
    "cxcyah_to_xyxy",
    "describe_tracker",
    "fuse_detection_score",
    "gate_cost",
    "iou_cost",
    "iou_matrix",
    "normalize_tracker_config",
    "solve",
    "tracker_factory",
    "tracker_ships",
    "xyxy_to_cxcyah",
]
