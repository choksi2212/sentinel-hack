"""build_source -- the one place a source_mode string becomes an adapter.

This function is the acceptance test for source independence. If swapping
config/offline.yaml for config/live.yaml changes nothing but this call, the
invariant in Contracts section 2.3 holds. If any other module has to branch on
source_mode to work, it does not.

Unknown keys are rejected rather than ignored. A config with `pathh:` instead of
`path:` would otherwise open the default clip, run to completion, and produce a
clean-looking benchmark of the wrong video -- discovered, if at all, by noticing
the frame count is odd. Failing on the typo costs a second; ignoring it costs an
afternoon and can cost a submitted number.
"""

from typing import Any, Mapping, Optional

from ai.contracts.enums import SOURCE_MODES
from ai.contracts.ids import require_camera_id
from ai.media.base import BaseMediaSource
from ai.media.file_source import VideoFileSource
from ai.media.frames_source import FrameSequenceSource
from ai.media.hls_source import SentinelHLSSource
from ai.media.rtsp_source import SentinelRTSPSource
from ai.media.synthetic_source import SyntheticFaults, SyntheticReplaySource

# Accepted on every mode.
_COMMON_KEYS = frozenset(
    {
        "mode",
        "camera_id",
        "target_interval_ms",
        "detect_discontinuity",
        "discontinuity_threshold",
        "max_frames",
    }
)

_MODE_KEYS: dict[str, frozenset[str]] = {
    "file": frozenset({"path", "loop", "max_loops", "speed"}),
    "frames": frozenset({"directory", "interval_ms", "loop", "max_loops", "speed"}),
    "live_rtsp": frozenset({"url", "transport_options", "read_timeout_seconds"}),
    "live_hls": frozenset({"url", "username", "password", "read_timeout_seconds"}),
    "synthetic": frozenset(
        {
            "seed",
            "width",
            "height",
            "total_frames",
            "step_ms",
            "plates",
            "vehicles_per_100_frames",
            "faults",
            "speed",
        }
    ),
}

_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "file": ("path",),
    "frames": ("directory",),
    "live_rtsp": (),
    "live_hls": (),
    "synthetic": (),
}


class SourceConfigError(ValueError):
    """A source config that cannot produce a working adapter.

    Its own type so the worker can report a configuration problem as a
    configuration problem, rather than as a crash that looks like a bug in the
    pipeline.
    """


def build_source(
    config: Mapping[str, Any],
    *,
    camera_id: Optional[str] = None,
) -> BaseMediaSource:
    """Construct the adapter described by a source config block.

    camera_id overrides config["camera_id"] when given, which is how one config
    file drives thirty cameras: `--camera cam07` on the command line rather than
    thirty near-identical YAML files that drift apart.
    """
    if not isinstance(config, Mapping):
        raise SourceConfigError(
            f"source config must be a mapping, got {type(config).__name__}"
        )

    mode = config.get("mode")
    if mode is None:
        raise SourceConfigError(
            f"source config has no 'mode'; expected one of {sorted(SOURCE_MODES)}"
        )
    if mode not in _MODE_KEYS:
        raise SourceConfigError(
            f"unknown source mode {mode!r}; expected one of {sorted(SOURCE_MODES)}"
        )

    resolved_camera = camera_id or config.get("camera_id")
    if not resolved_camera:
        raise SourceConfigError(
            f"source config for mode {mode!r} has no 'camera_id' and none was "
            "passed; every frame must carry one (Contracts section 1.1)"
        )
    try:
        resolved_camera = require_camera_id(str(resolved_camera))
    except ValueError as exc:
        raise SourceConfigError(str(exc)) from exc

    allowed = _COMMON_KEYS | _MODE_KEYS[mode]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SourceConfigError(
            f"unknown key(s) {unknown} for source mode {mode!r}; "
            f"accepted keys are {sorted(allowed)}"
        )

    missing = [key for key in _REQUIRED_KEYS[mode] if not config.get(key)]
    if missing:
        raise SourceConfigError(
            f"source mode {mode!r} requires {missing}, which "
            f"{'is' if len(missing) == 1 else 'are'} missing or empty"
        )

    shared = {
        key: config[key]
        for key in (
            "target_interval_ms",
            "detect_discontinuity",
            "discontinuity_threshold",
            "max_frames",
        )
        if key in config
    }

    # ValueError from a constructor is a malformed config value, not a bug: every source
    # constructor validates its own numbers and that is the only way they can refuse one.
    # Relabelled so a caller that catches SourceConfigError sees it -- scripts/validate_config.py
    # reports anything else as "could not construct", a label reserved for non-config failures
    # like a missing clip, which would describe a bad threshold as the wrong kind of problem.
    try:
        return _construct(mode, resolved_camera, config, shared)
    except ValueError as exc:
        raise SourceConfigError(f"source mode {mode!r}: {exc}") from exc


def _construct(
    mode: str,
    resolved_camera: str,
    config: Mapping[str, Any],
    shared: dict[str, Any],
) -> BaseMediaSource:
    """Per-mode construction. Split out so one try/except covers every mode."""
    if mode == "file":
        return VideoFileSource(
            resolved_camera,
            str(config["path"]),
            loop=bool(config.get("loop", False)),
            max_loops=config.get("max_loops"),
            speed=config.get("speed"),
            **shared,
        )

    if mode == "frames":
        return FrameSequenceSource(
            resolved_camera,
            str(config["directory"]),
            interval_ms=int(config.get("interval_ms", 40)),
            loop=bool(config.get("loop", False)),
            max_loops=config.get("max_loops"),
            speed=config.get("speed"),
            **shared,
        )

    if mode == "live_rtsp":
        return SentinelRTSPSource(
            resolved_camera,
            url=config.get("url"),
            **_live_kwargs(config),
            **shared,
        )

    if mode == "live_hls":
        return SentinelHLSSource(
            resolved_camera,
            url=config.get("url"),
            username=config.get("username"),
            password=config.get("password"),
            **_live_kwargs(config, transport=False),
            **shared,
        )

    return SyntheticReplaySource(
        resolved_camera,
        faults=_build_faults(config.get("faults")),
        **{
            key: config[key]
            for key in (
                "seed",
                "width",
                "height",
                "total_frames",
                "step_ms",
                "vehicles_per_100_frames",
                "speed",
            )
            if key in config
        },
        **(
            {"plates": tuple(str(p) for p in config["plates"])}
            if config.get("plates")
            else {}
        ),
        **shared,
    )


def _live_kwargs(config: Mapping[str, Any], *, transport: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "read_timeout_seconds" in config:
        kwargs["read_timeout_seconds"] = float(config["read_timeout_seconds"])
    if transport and "transport_options" in config:
        kwargs["transport_options"] = str(config["transport_options"])
    return kwargs


def _build_faults(raw: Any) -> SyntheticFaults:
    """Turn the faults block into SyntheticFaults, rejecting unknown fields.

    Silently dropping a misspelled fault would make a fault-injection run pass
    for the wrong reason -- the worst possible outcome for a test whose job is to
    prove the system survives a failure.
    """
    if raw is None:
        return SyntheticFaults()
    if not isinstance(raw, Mapping):
        raise SourceConfigError(
            f"'faults' must be a mapping, got {type(raw).__name__}"
        )

    known = set(SyntheticFaults.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SourceConfigError(
            f"unknown fault(s) {unknown}; accepted faults are {sorted(known)}"
        )

    values = dict(raw)
    if "black_frames" in values and values["black_frames"] is not None:
        values["black_frames"] = tuple(int(f) for f in values["black_frames"])
    return SyntheticFaults(**values)
