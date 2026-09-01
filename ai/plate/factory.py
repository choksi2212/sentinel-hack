"""build_plate_detector -- the one place a plate backend name becomes a model.

Same contract as ai/detect/factory.py and ai/media/factory.py: unknown keys are an
error. A config with `confidence_thresold:` runs at the default threshold and nothing
in the output says so, which turns a typo into a benchmark row that is quietly about
a different configuration than the one it claims.

    ships       rtdetr    justjuu RT-DETRv2, Apache-2.0, real weights
    never ships oracle    reads ground truth; measures the stages after this one
                edge      numpy projection profile; proves the pipeline runs bare
                scripted  fixed answers; for tests
"""

from typing import Any, Mapping, Optional

from ai.plate.base import BasePlateDetector, DEFAULT_PLATE_CONFIDENCE_THRESHOLD

PLATE_DETECTOR_NAMES: tuple[str, ...] = ("rtdetr", "edge", "oracle", "scripted")

# Backends whose numbers may appear in a published claim. A literal set rather than a
# derivation from the ships property, so adding a backend forces a decision here.
SHIPPABLE_PLATE_DETECTORS: frozenset[str] = frozenset({"rtdetr"})

_COMMON_KEYS = frozenset(
    {
        "name",
        "confidence_threshold",
        "pad_fraction",
        "apply_region_prior",
        "min_vehicle_width_px",
    }
)

_PLATE_KEYS: dict[str, frozenset[str]] = {
    "rtdetr": frozenset(
        {
            "repo_id",
            "device",
            "precision",
            "local_weights",
            "cache_dir",
            "hf_token",
            "batch_crops",
            "max_batch",
        }
    ),
    "edge": frozenset(
        {
            "gradient_threshold",
            "row_min_fill",
            "col_quantile",
            "search_lower_fraction",
        }
    ),
    "oracle": frozenset({"miss_rate", "confidence", "require_legible"}),
    "scripted": frozenset({"script"}),
}


class PlateConfigError(ValueError):
    """A plate config that cannot produce a working detector.

    Distinct type so the worker reports a bad config as a bad config rather than as a
    crash inside transformers.
    """


def build_plate_detector(
    config: Mapping[str, Any],
    *,
    source: Any = None,
) -> BasePlateDetector:
    """Construct the plate detector described by a config block.

    source is required only by the oracle backend, which reads ground truth from a
    SyntheticReplaySource. Passing it otherwise is harmless and ignored.
    """
    if not isinstance(config, Mapping):
        raise PlateConfigError(
            f"plate config must be a mapping, got {type(config).__name__}"
        )

    name = config.get("name")
    if name is None:
        raise PlateConfigError(
            f"plate config has no 'name'; expected one of {list(PLATE_DETECTOR_NAMES)}"
        )
    if name not in _PLATE_KEYS:
        raise PlateConfigError(
            f"unknown plate detector {name!r}; expected one of "
            f"{list(PLATE_DETECTOR_NAMES)}"
        )

    allowed = _COMMON_KEYS | _PLATE_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise PlateConfigError(
            f"unknown key(s) {unknown} for plate detector {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )

    shared: dict[str, Any] = {}
    if "confidence_threshold" in config:
        shared["confidence_threshold"] = float(config["confidence_threshold"])
    if "pad_fraction" in config:
        shared["pad_fraction"] = float(config["pad_fraction"])
    if "apply_region_prior" in config:
        shared["apply_region_prior"] = bool(config["apply_region_prior"])
    if "min_vehicle_width_px" in config:
        shared["min_vehicle_width_px"] = int(config["min_vehicle_width_px"])

    if name == "rtdetr":
        return _build_rtdetr(config, shared)
    if name == "edge":
        return _build_edge(config, shared)
    if name == "oracle":
        return _build_oracle(config, shared, source)
    return _build_scripted(config, shared)


def _build_rtdetr(
    config: Mapping[str, Any], shared: dict[str, Any]
) -> BasePlateDetector:
    # Imported here so that a run using the edge backend never imports torch. That is
    # not a micro-optimisation: torch import is several seconds and allocates CUDA
    # context, and the no-weights path exists precisely to be runnable where neither
    # is available.
    from ai.plate.rtdetr import RTDETRPlateDetector

    kwargs: dict[str, Any] = dict(shared)
    for key in ("repo_id", "device", "precision", "local_weights", "cache_dir"):
        if key in config:
            kwargs[key] = config[key]
    if "batch_crops" in config:
        kwargs["batch_crops"] = bool(config["batch_crops"])
    if "max_batch" in config:
        kwargs["max_batch"] = int(config["max_batch"])

    # The token is read from the environment when the config does not carry one, and
    # the config should not carry one. A checkpoint reference in version control is
    # fine; a credential in version control is not, which is why .env is gitignored
    # and .env.example holds the key names only.
    kwargs["hf_token"] = config.get("hf_token") or _env_token()
    return RTDETRPlateDetector(**kwargs)


def _env_token() -> Optional[str]:
    import os

    for key in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _build_edge(config: Mapping[str, Any], shared: dict[str, Any]) -> BasePlateDetector:
    from ai.plate.stub import EdgePlateDetector

    kwargs: dict[str, Any] = dict(shared)
    for key, cast in (
        ("gradient_threshold", int),
        ("row_min_fill", float),
        ("col_quantile", float),
        ("search_lower_fraction", float),
    ):
        if key in config:
            kwargs[key] = cast(config[key])
    return EdgePlateDetector(**kwargs)


def _build_oracle(
    config: Mapping[str, Any], shared: dict[str, Any], source: Any
) -> BasePlateDetector:
    from ai.plate.stub import OraclePlateDetector

    if source is None:
        raise PlateConfigError(
            "plate detector 'oracle' needs the media source to read ground truth "
            "from, and none was passed. It only works with source mode 'synthetic'."
        )
    if not hasattr(source, "truth_for_envelope"):
        raise PlateConfigError(
            f"plate detector 'oracle' requires a source exposing "
            f"truth_for_envelope(); {type(source).__name__} does not. Use source "
            f"mode 'synthetic'."
        )

    kwargs: dict[str, Any] = dict(shared)
    for key, cast in (
        ("miss_rate", float),
        ("confidence", float),
        ("require_legible", bool),
    ):
        if key in config:
            kwargs[key] = cast(config[key])
    return OraclePlateDetector(source, **kwargs)


def _build_scripted(
    config: Mapping[str, Any], shared: dict[str, Any]
) -> BasePlateDetector:
    from ai.plate.stub import ScriptedPlateDetector

    raw = config.get("script")
    if raw is None:
        raise PlateConfigError("plate detector 'scripted' requires a 'script' mapping")
    if not isinstance(raw, Mapping):
        raise PlateConfigError(
            f"'script' must be a mapping of frame index to boxes, got "
            f"{type(raw).__name__}"
        )

    script: dict[int, list[tuple[tuple[int, int, int, int], float]]] = {}
    for key, rows in raw.items():
        entries: list[tuple[tuple[int, int, int, int], float]] = []
        for row in rows or ():
            if isinstance(row, Mapping):
                box = row.get("bbox_xyxy")
                confidence = float(row.get("confidence", 1.0))
            else:
                try:
                    box, confidence = row[0], float(row[1])
                except (TypeError, IndexError, ValueError) as exc:
                    raise PlateConfigError(
                        f"script[{key!r}] entries must be (bbox_xyxy, confidence) "
                        f"pairs or mappings, got {row!r}"
                    ) from exc
            if box is None:
                raise PlateConfigError(
                    f"script[{key!r}] entry is missing bbox_xyxy"
                )
            entries.append((tuple(int(v) for v in box), confidence))
        script[int(key)] = entries

    kwargs = dict(shared)
    # A script is an exact instruction; filtering it at the default 0.25 would drop
    # entries a test deliberately wrote as low-confidence to exercise that path.
    if "confidence_threshold" not in config:
        kwargs["confidence_threshold"] = 0.0
    return ScriptedPlateDetector(script, **kwargs)


def plate_detector_ships(name: str) -> bool:
    """Whether a benchmark using this backend may be published."""
    return name in SHIPPABLE_PLATE_DETECTORS


def describe_plate_detector(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise a plate config without constructing it.

    Config validation must not download 171 MB of weights, so
    scripts/validate_config.py reports on the block statically.
    """
    name = config.get("name")
    return {
        "name": name,
        "ships": plate_detector_ships(str(name)),
        "confidence_threshold": config.get(
            "confidence_threshold", DEFAULT_PLATE_CONFIDENCE_THRESHOLD
        ),
        "batched": bool(config.get("batch_crops", True)) if name == "rtdetr" else False,
    }


def normalize_plate_config(
    config: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate a plate block and return it as a plain dict.

    for_publication=True refuses a non-shipping backend, so a benchmark run fails in
    the first second rather than after producing numbers that have to be discarded.
    """
    if not isinstance(config, Mapping):
        raise PlateConfigError(
            f"plate config must be a mapping, got {type(config).__name__}"
        )
    name = str(config.get("name", ""))
    if name not in _PLATE_KEYS:
        raise PlateConfigError(
            f"unknown plate detector {name!r}; expected one of "
            f"{list(PLATE_DETECTOR_NAMES)}"
        )
    allowed = _COMMON_KEYS | _PLATE_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise PlateConfigError(
            f"unknown key(s) {unknown} for plate detector {name!r}; "
            f"accepted keys are {sorted(allowed)}"
        )
    if for_publication and not plate_detector_ships(name):
        raise PlateConfigError(
            f"plate detector {name!r} must not appear in a published benchmark: it "
            f"either reads ground truth or carries no trained weights. Publishable "
            f"backends are {sorted(SHIPPABLE_PLATE_DETECTORS)}."
        )
    return dict(config)
