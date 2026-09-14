"""build_tracker -- the one place a tracker name becomes an instance.

Same rejection discipline as the media and detector factories. A config with
`track_bufer:` would otherwise run on the default buffer and produce a benchmark
row that silently describes different settings than the one next to it.

The publication split matches ai/detect/factory.py:

    ships       bytetrack     MIT, the locked choice, goes in the submission
    never ships iou           real technique, not the one claimed
                oracle        reads ground truth
                scripted      fixed table, for tests
"""

from typing import Any, Callable, Mapping, Optional

from ai.track.base import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MIN_HITS,
    DEFAULT_TRACK_BUFFER,
    BaseTracker,
)
from ai.track.registry import TrackerRegistry

TRACKER_NAMES: tuple[str, ...] = ("bytetrack", "iou", "oracle", "scripted")

SHIPPABLE_TRACKERS: frozenset[str] = frozenset({"bytetrack"})

_COMMON_KEYS = frozenset({"name", "track_buffer", "min_hits"})

_TRACKER_KEYS: dict[str, frozenset[str]] = {
    "bytetrack": frozenset(
        {
            "high_threshold",
            "low_threshold",
            "use_low_stage",
            "use_gating",
            "fuse_score",
        }
    ),
    "iou": frozenset({"min_iou", "confidence_threshold"}),
    "oracle": frozenset({"min_iou"}),
    "scripted": frozenset({"script"}),
}


class TrackerConfigError(ValueError):
    """A tracker config that cannot produce a working tracker."""


def build_tracker(
    config: Mapping[str, Any],
    camera_id: str,
    stream_session_id: str,
    *,
    source: Any = None,
) -> BaseTracker:
    """Construct the tracker described by a config block.

    camera_id and stream_session_id are positional and required. Not optional with
    a default: a tracker without a session id produces a two-part TrackKey and
    merges unrelated vehicles across a reconnect, and making that reachable by
    forgetting an argument is not a risk worth the convenience.
    """
    name = _validated_name(config)
    allowed = _COMMON_KEYS | _TRACKER_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise TrackerConfigError(
            f"unknown key(s) {unknown} for tracker {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )

    shared: dict[str, Any] = {}
    if "track_buffer" in config:
        shared["track_buffer"] = int(config["track_buffer"])
    if "min_hits" in config:
        shared["min_hits"] = int(config["min_hits"])

    if name == "bytetrack":
        from ai.track.bytetrack import ByteTracker

        kwargs = dict(shared)
        for key, cast in (
            ("high_threshold", float),
            ("low_threshold", float),
            ("use_low_stage", bool),
            ("use_gating", bool),
            ("fuse_score", bool),
        ):
            if key in config:
                kwargs[key] = cast(config[key])
        return ByteTracker(camera_id, stream_session_id, **kwargs)

    if name == "iou":
        from ai.track.stub import IOUTracker

        kwargs = dict(shared)
        for key in ("min_iou", "confidence_threshold"):
            if key in config:
                kwargs[key] = float(config[key])
        return IOUTracker(camera_id, stream_session_id, **kwargs)

    if name == "oracle":
        from ai.track.stub import OracleTracker

        if source is None:
            raise TrackerConfigError(
                "tracker 'oracle' needs the media source to read ground truth from, "
                "and none was passed. It only works with source mode 'synthetic'."
            )
        kwargs = dict(shared)
        if "min_iou" in config:
            kwargs["min_iou"] = float(config["min_iou"])
        return OracleTracker(camera_id, stream_session_id, source, **kwargs)

    from ai.track.stub import ScriptedTracker

    script = config.get("script")
    if script is None:
        raise TrackerConfigError("tracker 'scripted' requires a 'script' mapping")
    if not isinstance(script, Mapping):
        raise TrackerConfigError(
            f"'script' must be a mapping of frame index to rows, got "
            f"{type(script).__name__}"
        )
    return ScriptedTracker(
        camera_id,
        stream_session_id,
        {int(k): tuple(v or ()) for k, v in script.items()},
    )


def build_registry(
    config: Mapping[str, Any],
    *,
    source: Any = None,
    strict_sessions: bool = True,
) -> TrackerRegistry:
    """A registry whose factory builds this config's tracker for any camera.

    This is what the worker uses. It closes over the config once, so adding a
    camera mid-run cannot pick up different settings than the cameras already
    running -- a drift that would make a multi-camera benchmark incomparable with
    itself.
    """
    validated = normalize_tracker_config(config)

    def factory(camera_id: str, stream_session_id: str) -> BaseTracker:
        return build_tracker(
            validated, camera_id, stream_session_id, source=source
        )

    return TrackerRegistry(factory, strict_sessions=strict_sessions)


def _validated_name(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise TrackerConfigError(
            f"tracker config must be a mapping, got {type(config).__name__}"
        )
    name = config.get("name")
    if name is None:
        raise TrackerConfigError(
            f"tracker config has no 'name'; expected one of {list(TRACKER_NAMES)}"
        )
    if name not in _TRACKER_KEYS:
        raise TrackerConfigError(
            f"unknown tracker {name!r}; expected one of {list(TRACKER_NAMES)}"
        )
    return str(name)


def tracker_ships(name: str) -> bool:
    """Whether a benchmark using this tracker may be published."""
    return name in SHIPPABLE_TRACKERS


def normalize_tracker_config(
    config: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate a tracker block and return it as a plain dict.

    for_publication=True refuses a non-shipping tracker, so a benchmark run fails
    in its first second rather than after producing numbers that describe a
    tracker the submission does not use.
    """
    name = _validated_name(config)
    allowed = _COMMON_KEYS | _TRACKER_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise TrackerConfigError(
            f"unknown key(s) {unknown} for tracker {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )

    if name == "bytetrack":
        low = float(config.get("low_threshold", DEFAULT_LOW_THRESHOLD))
        high = float(config.get("high_threshold", DEFAULT_HIGH_THRESHOLD))
        if not 0.0 <= low <= high <= 1.0:
            raise TrackerConfigError(
                f"need 0 <= low_threshold <= high_threshold <= 1, got "
                f"low={low} high={high}"
            )
        # An ablation config is legal but must never be mistaken for the real thing.
        if for_publication and not config.get("use_low_stage", True):
            raise TrackerConfigError(
                "use_low_stage=False disables ByteTrack's two-stage association, "
                "which is the whole method. The result is a SORT variant and must "
                "not be published as ByteTrack."
            )
    if for_publication and not tracker_ships(name):
        raise TrackerConfigError(
            f"tracker {name!r} must not appear in a published benchmark: it either "
            "reads ground truth, replays a fixture, or is not the method the "
            f"submission claims. Publishable trackers are {sorted(SHIPPABLE_TRACKERS)}."
        )
    return dict(config)


def describe_tracker(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise a tracker config without constructing one."""
    name = str(config.get("name"))
    return {
        "name": name,
        "ships": tracker_ships(name),
        "track_buffer": config.get("track_buffer", DEFAULT_TRACK_BUFFER),
        "min_hits": config.get("min_hits", DEFAULT_MIN_HITS),
        "two_stage": bool(config.get("use_low_stage", True))
        if name == "bytetrack"
        else False,
    }


def tracker_factory(
    config: Mapping[str, Any], *, source: Any = None
) -> Callable[[str, str], BaseTracker]:
    """A bound (camera_id, session_id) -> tracker callable.

    For a caller that wants its own registry, or none.
    """
    validated = normalize_tracker_config(config)

    def factory(camera_id: str, stream_session_id: str) -> BaseTracker:
        return build_tracker(validated, camera_id, stream_session_id, source=source)

    return factory
