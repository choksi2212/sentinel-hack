"""build_detector -- the one place a detector name becomes a model.

Same contract as ai/media/factory.py, for the same reason: unknown keys are an
error, not a shrug. A benchmark row produced by a config with `confidence_thresold:`
is a benchmark row for the default threshold, and nothing in the output says so.

The backends divide into two groups and the split is load-bearing:

  ships       rfdetr        Apache-2.0, real weights, goes in the submission
  never ships oracle        reads ground truth; measures the pipeline, not the model
              motion        numpy fallback; proves the pipeline runs with no weights
              scripted      fixed responses; for tests

ai/metrics.py refuses to publish an accuracy number from a detector whose ships
property is False. Both halves of that check matter -- this factory is where the
flag gets attached, and nothing downstream can recover it if it is wrong here.
"""

from typing import Any, Mapping, Optional

from ai.detect.base import BaseDetector, DEFAULT_CONFIDENCE_THRESHOLD

DETECTOR_NAMES: tuple[str, ...] = ("rfdetr", "motion", "oracle", "scripted")

# Detectors that may appear in a published benchmark. Kept as a literal set rather
# than derived from the ships property, so that adding a backend forces a
# deliberate decision here instead of inheriting a default.
SHIPPABLE_DETECTORS: frozenset[str] = frozenset({"rfdetr"})

_COMMON_KEYS = frozenset({"name", "confidence_threshold", "allowed_classes"})

_DETECTOR_KEYS: dict[str, frozenset[str]] = {
    "rfdetr": frozenset(
        {
            "variant",
            "device",
            "precision",
            "tile",
            "tile_grid",
            "tile_overlap",
            "include_full_frame",
            "merge_iou",
            "max_detections",
            "coco_classes_are_advisory",
            "local_weights",
            "backend",
            "cache_dir",
        }
    ),
    "motion": frozenset(
        {
            "diff_threshold",
            "background_alpha",
            "background_frames",
            "min_area",
            "min_width",
            "min_height",
            "min_fill_ratio",
            "max_detections",
        }
    ),
    "oracle": frozenset(
        {
            "miss_rate",
            "jitter_px",
            "false_positives_per_frame",
            "confidence",
            "false_positive_confidence",
            "seed",
        }
    ),
    "scripted": frozenset({"script"}),
}


class DetectorConfigError(ValueError):
    """A detector config that cannot produce a working model.

    Distinct type so the worker reports a bad config as a bad config rather than
    as a crash somewhere inside onnxruntime.
    """


def build_detector(
    config: Mapping[str, Any],
    *,
    source: Any = None,
) -> BaseDetector:
    """Construct the detector described by a config block.

    source is required only by the oracle backend, which reads ground truth from
    a SyntheticReplaySource. Passing it for the others is harmless and ignored --
    the worker does not know or care which backend needs it.
    """
    if not isinstance(config, Mapping):
        raise DetectorConfigError(
            f"detector config must be a mapping, got {type(config).__name__}"
        )

    name = config.get("name")
    if name is None:
        raise DetectorConfigError(
            f"detector config has no 'name'; expected one of {list(DETECTOR_NAMES)}"
        )
    if name not in _DETECTOR_KEYS:
        raise DetectorConfigError(
            f"unknown detector {name!r}; expected one of {list(DETECTOR_NAMES)}"
        )

    allowed = _COMMON_KEYS | _DETECTOR_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise DetectorConfigError(
            f"unknown key(s) {unknown} for detector {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )

    shared: dict[str, Any] = {
        "confidence_threshold": float(
            config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
    }
    classes = config.get("allowed_classes")
    if classes:
        shared["allowed_classes"] = frozenset(str(c) for c in classes)

    if name == "rfdetr":
        return _build_rfdetr(config, shared)
    if name == "motion":
        return _build_motion(config, shared)
    if name == "oracle":
        return _build_oracle(config, shared, source)
    return _build_scripted(config, shared)


def _build_rfdetr(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseDetector:
    # Imported here, not at module scope. The import pulls in onnxruntime lazily
    # via RFDETRDetector._load, but even the module-level `import json, os` is
    # avoidable for a test run that only wants the scripted backend.
    from ai.detect.rfdetr import RFDETRDetector

    kwargs: dict[str, Any] = dict(shared)
    for key in (
        "variant",
        "device",
        "precision",
        "include_full_frame",
        "local_weights",
        "backend",
        "cache_dir",
    ):
        if key in config:
            kwargs[key] = config[key]
    if "tile" in config:
        kwargs["tile"] = bool(config["tile"])
    if "coco_classes_are_advisory" in config:
        kwargs["coco_classes_are_advisory"] = bool(config["coco_classes_are_advisory"])
    if "tile_overlap" in config:
        kwargs["tile_overlap"] = float(config["tile_overlap"])
    if "merge_iou" in config:
        kwargs["merge_iou"] = float(config["merge_iou"])
    if "max_detections" in config:
        kwargs["max_detections"] = int(config["max_detections"])
    if "tile_grid" in config:
        kwargs["tile_grid"] = _tile_grid(config["tile_grid"])

    return RFDETRDetector(**kwargs)


def _tile_grid(raw: Any) -> tuple[int, int]:
    """Parse [cols, rows]. YAML gives a list; the detector wants a 2-tuple."""
    try:
        cols, rows = (int(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise DetectorConfigError(
            f"tile_grid must be two integers [cols, rows], got {raw!r}"
        ) from exc
    if cols < 1 or rows < 1:
        raise DetectorConfigError(f"tile_grid values must be >= 1, got {raw!r}")
    return cols, rows


def _build_motion(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseDetector:
    from ai.detect.stub import MotionBlobDetector

    # The config key is background_alpha because a bare `alpha:` in a YAML file
    # says nothing about what it decays. The constructor parameter is alpha, so
    # the rename happens here and nowhere else.
    kwargs: dict[str, Any] = dict(shared)
    for key, param, cast in (
        ("diff_threshold", "diff_threshold", int),
        ("background_alpha", "alpha", float),
        ("background_frames", "background_frames", int),
        ("min_area", "min_area", int),
        ("min_width", "min_width", int),
        ("min_height", "min_height", int),
        ("min_fill_ratio", "min_fill_ratio", float),
        ("max_detections", "max_detections", int),
    ):
        if key in config:
            kwargs[param] = cast(config[key])
    return MotionBlobDetector(**kwargs)


def _build_oracle(
    config: Mapping[str, Any], shared: dict[str, Any], source: Any
) -> BaseDetector:
    from ai.detect.stub import OracleDegradation, OracleDetector

    if source is None:
        raise DetectorConfigError(
            "detector 'oracle' needs the media source to read ground truth from, "
            "and none was passed. It only works with source mode 'synthetic'."
        )
    if not hasattr(source, "truth_for_envelope"):
        raise DetectorConfigError(
            f"detector 'oracle' requires a source exposing truth_for_envelope(); "
            f"{type(source).__name__} does not. Use source mode 'synthetic'."
        )

    fields = set(OracleDegradation.__dataclass_fields__)
    degradation = OracleDegradation(
        **{key: config[key] for key in fields if key in config}
    )
    return OracleDetector(source, degradation=degradation, **shared)


def _build_scripted(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseDetector:
    from ai.contracts.stages import DetectorResult
    from ai.detect.stub import ScriptedDetector

    raw = config.get("script")
    if raw is None:
        raise DetectorConfigError("detector 'scripted' requires a 'script' mapping")
    if not isinstance(raw, Mapping):
        raise DetectorConfigError(
            f"'script' must be a mapping of frame index to detections, got "
            f"{type(raw).__name__}"
        )

    script: dict[int, list[DetectorResult]] = {}
    for key, rows in raw.items():
        detections: list[DetectorResult] = []
        for row in rows or ():
            if isinstance(row, DetectorResult):
                detections.append(row)
                continue
            if not isinstance(row, Mapping):
                raise DetectorConfigError(
                    f"script[{key!r}] entries must be mappings, got "
                    f"{type(row).__name__}"
                )
            try:
                detections.append(
                    DetectorResult(
                        bbox_xyxy=tuple(int(v) for v in row["bbox_xyxy"]),
                        class_name=str(row["class_name"]),
                        confidence=float(row.get("confidence", 1.0)),
                    )
                )
            except KeyError as exc:
                raise DetectorConfigError(
                    f"script[{key!r}] entry is missing {exc.args[0]!r}; each needs "
                    "bbox_xyxy and class_name"
                ) from exc
        script[int(key)] = detections

    # ScriptedDetector's own default threshold is 0.0 -- a script is an exact
    # instruction and filtering it would be surprising. Only override when the
    # config asked.
    kwargs = dict(shared)
    if "confidence_threshold" not in config:
        kwargs["confidence_threshold"] = 0.0
    return ScriptedDetector(script, **kwargs)


def detector_ships(name: str) -> bool:
    """Whether a benchmark using this detector may be published.

    Used by ai/metrics.py before writing a result row, and by
    scripts/validate_config.py so the mistake is caught at config-load time
    rather than after a forty-minute run.
    """
    return name in SHIPPABLE_DETECTORS


def describe_detector(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise a detector config without constructing it.

    Config validation must not download a 108 MB checkpoint or touch the GPU, so
    scripts/validate_config.py reports on the block statically.
    """
    name = config.get("name")
    return {
        "name": name,
        "ships": detector_ships(str(name)),
        "confidence_threshold": config.get(
            "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        ),
        "tiled": bool(config.get("tile", True)) if name == "rfdetr" else False,
    }


def normalize_detector_config(
    config: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate a detector block and return it as a plain dict.

    for_publication=True additionally refuses a non-shipping backend. That check
    lives here rather than only in ai/metrics.py so a benchmark run fails in the
    first second instead of after producing numbers that have to be thrown away.
    """
    if not isinstance(config, Mapping):
        raise DetectorConfigError(
            f"detector config must be a mapping, got {type(config).__name__}"
        )
    name = str(config.get("name", ""))
    if name not in _DETECTOR_KEYS:
        raise DetectorConfigError(
            f"unknown detector {name!r}; expected one of {list(DETECTOR_NAMES)}"
        )
    allowed = _COMMON_KEYS | _DETECTOR_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise DetectorConfigError(
            f"unknown key(s) {unknown} for detector {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )
    if for_publication and not detector_ships(name):
        raise DetectorConfigError(
            f"detector {name!r} must not appear in a published benchmark: it either "
            "reads ground truth or carries no real weights. Publishable backends "
            f"are {sorted(SHIPPABLE_DETECTORS)}."
        )
    return dict(config)


def resolve_allowed_classes(raw: Any) -> Optional[frozenset[str]]:
    """Turn an allowed_classes list into the frozenset BaseDetector expects.

    Rejects names outside the contract vocabulary. A filter set containing
    "van" silently matches nothing, and the run reports zero vans -- which reads
    as a detector failure rather than a config typo.
    """
    if not raw:
        return None
    from ai.detect.base import VEHICLE_CLASSES

    names = frozenset(str(c) for c in raw)
    unknown = sorted(names - VEHICLE_CLASSES)
    if unknown:
        raise DetectorConfigError(
            f"allowed_classes contains {unknown}, which are not vehicle types in "
            f"ai/contracts/enums.py. Valid: {sorted(VEHICLE_CLASSES)}"
        )
    return names
