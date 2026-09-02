"""build_ocr_engine -- the one place an OCR backend name becomes a recogniser.

Same contract as ai/plate/factory.py and ai/detect/factory.py: unknown keys are an
error, not a default. A config with `drop_scor:` runs at the default and nothing in the
output says so, which turns a typo into a benchmark row that is quietly about a
different configuration than the one it claims.

    ships       paddle    PaddleOCR recogniser, Apache-2.0, real weights
    never ships oracle    reads ground truth; isolates the stages after this one
                template  5x7 glyph matching; circular on the synthetic corpus
                scripted  fixed answers; for tests

Note which reason disqualifies which. `oracle` never ships because it reads the answer
key. `template` never ships because the fonts it matches against are the fonts the
fixtures are drawn with -- a licensing-clean, genuinely classical matcher that cannot
be honestly measured on the only corpus available to it. Both are excluded from
publication; only one of them is excluded for cheating.
"""

from typing import Any, Mapping, Optional, Sequence

from ai.ocr.base import BaseOCR, MIN_OCR_PLATE_WIDTH_PX, PLATE_CROP_PAD_PX

OCR_ENGINE_NAMES: tuple[str, ...] = ("paddle", "template", "oracle", "scripted")

# Backends whose numbers may appear in a published claim. A literal set rather than a
# derivation from the ships property, so adding a backend forces a decision here.
SHIPPABLE_OCR_ENGINES: frozenset[str] = frozenset({"paddle"})

_COMMON_KEYS = frozenset(
    {
        "name",
        "min_plate_width_px",
        "pad_px",
        "variants",
    }
)

_OCR_KEYS: dict[str, frozenset[str]] = {
    "paddle": frozenset(
        {
            "lang",
            "use_gpu",
            "drop_score",
            "model_dir",
            "use_angle_cls",
        }
    ),
    "template": frozenset({"ink_level", "min_chars", "max_chars", "min_score"}),
    "oracle": frozenset(
        {
            "char_error_rate",
            "error_full_width_px",
            "confidence",
            "min_confidence",
            "require_legible",
        }
    ),
    "scripted": frozenset({"script"}),
}


class OCRConfigError(ValueError):
    """An OCR config that cannot produce a working engine.

    Distinct type so the worker reports a bad config as a bad config rather than as a
    crash inside paddle's C++ bindings, where the message will not mention the config.
    """


def build_ocr_engine(
    config: Mapping[str, Any],
    *,
    source: Any = None,
) -> BaseOCR:
    """Construct the OCR engine described by a config block.

    source is required only by the oracle backend, which reads ground truth from a
    SyntheticReplaySource. Passing it otherwise is harmless and ignored.
    """
    if not isinstance(config, Mapping):
        raise OCRConfigError(
            f"ocr config must be a mapping, got {type(config).__name__}"
        )

    name = config.get("name")
    if name is None:
        raise OCRConfigError(
            f"ocr config has no 'name'; expected one of {list(OCR_ENGINE_NAMES)}"
        )
    if name not in _OCR_KEYS:
        raise OCRConfigError(
            f"unknown ocr engine {name!r}; expected one of {list(OCR_ENGINE_NAMES)}"
        )

    allowed = _COMMON_KEYS | _OCR_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise OCRConfigError(
            f"unknown key(s) {unknown} for ocr engine {name!r}; accepted keys are "
            f"{sorted(allowed)}"
        )

    shared: dict[str, Any] = {}
    if "min_plate_width_px" in config:
        shared["min_plate_width_px"] = int(config["min_plate_width_px"])
    if "pad_px" in config:
        shared["pad_px"] = int(config["pad_px"])
    if "variants" in config:
        shared["variants"] = _parse_variants(config["variants"])

    if name == "paddle":
        return _build_paddle(config, shared)
    if name == "template":
        return _build_template(config, shared)
    if name == "oracle":
        return _build_oracle(config, shared, source)
    return _build_scripted(config, shared)


def _parse_variants(raw: Any) -> tuple[str, ...]:
    """Validate the variant list against the ones preprocess.py actually implements.

    Checked here rather than at first use, because an unknown variant name raises
    KeyError inside the per-plate loop -- meaning a misspelled variant in a config
    would surface as a crash a thousand frames into a benchmark run, not as a config
    error in the first second.
    """
    from ai.ocr.preprocess import variant_names

    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise OCRConfigError(
            f"'variants' must be a list of variant names, got {type(raw).__name__}"
        )
    names = tuple(str(v) for v in raw)
    if not names:
        raise OCRConfigError(
            "'variants' is empty; an engine with no variants reads nothing. Omit the "
            "key to get the default set."
        )
    known = variant_names()
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise OCRConfigError(
            f"unknown preprocessing variant(s) {unknown}; implemented variants are "
            f"{sorted(known)}"
        )
    return names


def _build_paddle(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseOCR:
    # Imported here so that a run on the template backend never imports paddle. Not a
    # micro-optimisation: importing paddlepaddle loads a CUDA context and costs several
    # seconds, and the no-weights path exists precisely to be runnable where neither is
    # available. It is also why an environment without paddleocr can still run the full
    # 14 stages -- the ImportError only fires if this branch is taken.
    from ai.ocr.paddle import PaddleOCR

    kwargs: dict[str, Any] = dict(shared)
    if "lang" in config:
        kwargs["lang"] = str(config["lang"])
    if "use_gpu" in config:
        kwargs["use_gpu"] = bool(config["use_gpu"])
    if "drop_score" in config:
        kwargs["drop_score"] = float(config["drop_score"])
    if "model_dir" in config:
        kwargs["model_dir"] = str(config["model_dir"])
    if "use_angle_cls" in config:
        kwargs["use_angle_cls"] = bool(config["use_angle_cls"])
    return PaddleOCR(**kwargs)


def _build_template(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseOCR:
    from ai.ocr.stub import TemplateOCR

    kwargs: dict[str, Any] = dict(shared)
    for key, cast in (
        ("ink_level", float),
        ("min_chars", int),
        ("max_chars", int),
        ("min_score", float),
    ):
        if key in config:
            kwargs[key] = cast(config[key])

    low = int(kwargs.get("min_chars", 4))
    high = int(kwargs.get("max_chars", 12))
    if low > high:
        raise OCRConfigError(
            f"template ocr has min_chars={low} above max_chars={high}, which admits no "
            f"length hypothesis at all and would refuse every plate"
        )
    return TemplateOCR(**kwargs)


def _build_oracle(
    config: Mapping[str, Any], shared: dict[str, Any], source: Any
) -> BaseOCR:
    from ai.ocr.stub import OracleOCR

    if source is None:
        raise OCRConfigError(
            "ocr engine 'oracle' needs the media source to read ground truth from, "
            "and none was passed. It only works with source mode 'synthetic'."
        )
    if not hasattr(source, "truth_for_envelope"):
        raise OCRConfigError(
            f"ocr engine 'oracle' requires a source exposing truth_for_envelope(); "
            f"{type(source).__name__} does not. Use source mode 'synthetic'."
        )

    kwargs: dict[str, Any] = dict(shared)
    for key, cast in (
        ("char_error_rate", float),
        ("error_full_width_px", int),
        ("confidence", float),
        ("min_confidence", float),
        ("require_legible", bool),
    ):
        if key in config:
            kwargs[key] = cast(config[key])

    # An oracle with errors is a fusion rig; an oracle with a single variant is a
    # cheaper one. Not forced, because a test may want the variant loop exercised, but
    # the default set costs six identical reads of the same ground truth per plate.
    return OracleOCR(source, **kwargs)


def _build_scripted(config: Mapping[str, Any], shared: dict[str, Any]) -> BaseOCR:
    from ai.ocr.stub import ScriptedOCR

    raw = config.get("script")
    if raw is None:
        raise OCRConfigError("ocr engine 'scripted' requires a 'script' mapping")
    if not isinstance(raw, Mapping):
        raise OCRConfigError(
            f"'script' must be a mapping of frame index to reads, got "
            f"{type(raw).__name__}"
        )

    script: dict[int, list[tuple[str, float]]] = {}
    for key, rows in raw.items():
        entries: list[tuple[str, float]] = []
        for row in rows or ():
            if isinstance(row, Mapping):
                text = row.get("text")
                confidence = float(row.get("confidence", 1.0))
            elif isinstance(row, str):
                # A bare string is a read at full confidence. Convenient in YAML, where
                # the common case is "frame 7 reads GJ01AB1234" with no interest in the
                # score.
                text, confidence = row, 1.0
            else:
                try:
                    text, confidence = str(row[0]), float(row[1])
                except (TypeError, IndexError, ValueError) as exc:
                    raise OCRConfigError(
                        f"script[{key!r}] entries must be (text, confidence) pairs, "
                        f"mappings, or bare strings, got {row!r}"
                    ) from exc
            if text is None:
                raise OCRConfigError(f"script[{key!r}] entry is missing text")
            entries.append((str(text), confidence))
        script[int(key)] = entries

    return ScriptedOCR(script, **shared)


def ocr_engine_ships(name: str) -> bool:
    """Whether a benchmark using this backend may be published."""
    return name in SHIPPABLE_OCR_ENGINES


def default_variants_for(name: str) -> tuple[str, ...]:
    """The variant list a backend runs when its config does not name one.

    Not simply DEFAULT_VARIANTS, because TemplateOCR narrows to one -- measured, see its
    docstring. Kept here as well as in the backend's own __init__ so that
    describe_ocr_engine can report the true cost of a config without constructing the
    engine, which is the whole point of it being static. The alternative, letting
    describe fall back to DEFAULT_VARIANTS, made it report six reads per plate for a
    config that performs one: a validator's cost estimate off by 6x, in the number the
    stage is actually budgeted on.
    """
    from ai.ocr.preprocess import DEFAULT_VARIANTS

    if name == "template":
        from ai.ocr.stub import TEMPLATE_DEFAULT_VARIANTS

        return TEMPLATE_DEFAULT_VARIANTS
    return tuple(DEFAULT_VARIANTS)


def describe_ocr_engine(config: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise an OCR config without constructing it.

    Config validation must not import paddle or download recogniser weights, so
    scripts/validate_config.py reports on the block statically.
    """
    name = config.get("name")
    variants = config.get("variants") or default_variants_for(str(name))
    return {
        "name": name,
        "ships": ocr_engine_ships(str(name)),
        "min_plate_width_px": config.get("min_plate_width_px", MIN_OCR_PLATE_WIDTH_PX),
        "pad_px": config.get("pad_px", PLATE_CROP_PAD_PX),
        "variants": list(variants),
        # Reads per plate, which is the number that actually sets this stage's cost:
        # the variant loop runs the backend once per variant on every plate.
        "reads_per_plate": len(variants),
    }


def normalize_ocr_config(
    config: Mapping[str, Any],
    *,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate an OCR block and return it as a plain dict.

    for_publication=True refuses a non-shipping backend, so a benchmark run fails in
    the first second rather than after producing an end-to-end correct-plate rate that
    has to be thrown away. That matters more here than at any other stage: the primary
    metric is a plate-string rate, so an unshippable OCR backend does not merely taint
    one diagnostic, it invalidates the headline number.
    """
    if not isinstance(config, Mapping):
        raise OCRConfigError(
            f"ocr config must be a mapping, got {type(config).__name__}"
        )
    name = str(config.get("name", ""))
    if name not in _OCR_KEYS:
        raise OCRConfigError(
            f"unknown ocr engine {name!r}; expected one of {list(OCR_ENGINE_NAMES)}"
        )
    allowed = _COMMON_KEYS | _OCR_KEYS[name]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise OCRConfigError(
            f"unknown key(s) {unknown} for ocr engine {name!r}; accepted keys are "
            f"{sorted(allowed)}"
        )
    if "variants" in config:
        _parse_variants(config["variants"])
    if for_publication and not ocr_engine_ships(name):
        raise OCRConfigError(
            f"ocr engine {name!r} must not appear in a published benchmark: it either "
            f"reads ground truth or matches against the same font the fixtures are "
            f"drawn with. Publishable backends are {sorted(SHIPPABLE_OCR_ENGINES)}."
        )
    return dict(config)


def check_ocr_width_floor(
    ocr_config: Mapping[str, Any],
    *,
    synthetic: bool = False,
) -> Optional[str]:
    """Warn when the OCR width floor sits below the point ground truth calls legible.

    The invariant: **on the synthetic corpus, OCR must not be asked to read a plate
    that truth has already marked illegible.** ai/media/synthetic_source.py sets
    plate_legible False below MIN_LEGIBLE_PLATE_WIDTH_PX (30 px) -- meaning the
    renderer did not draw legible characters there, so any string returned for such a
    plate is a fabrication by construction, not a hard read. ai/ocr/base.py's floor is
    MIN_OCR_PLATE_WIDTH_PX (24 px), which is lower. Plates in the 24-29 px band are
    therefore handed to the engine by default.

    This is not hypothetical, and it is what set TemplateOCR's min_score. Rendering the
    corpus's seven plate strings standalone and reading them with that floor removed gives
    seven fabricated strings at every width from 24 px up: 0.216-0.244 at 24 px, which the
    floor catches, but 0.419-0.522 at 26 px and 0.409-0.505 at 28 px, where five and then
    six of the seven survive it. None correct. That is precisely the failure Contracts
    section 12 names as the worst this pipeline can produce.

    The floors are deliberately different rather than reconciled: 24 px is the point
    below which cropping is pointless for *any* backend, and 30 px is a property of
    this particular renderer, which a trained recogniser on real frames may well beat.
    Hard-coding 30 into ai/ocr would bake a fixture's limitation into the shipping
    stage. So the gap stays and this warning marks it.

    Returned as a string rather than raised, because reading the 24-29 band is a
    legitimate experiment -- measuring how often a backend fabricates is exactly how
    you find out whether its refusal threshold is set right. Called by
    scripts/validate_config.py, which prints it.
    """
    from ai.media.synthetic_source import MIN_LEGIBLE_PLATE_WIDTH_PX

    if not synthetic:
        return None
    floor = int(ocr_config.get("min_plate_width_px", MIN_OCR_PLATE_WIDTH_PX))
    if floor >= MIN_LEGIBLE_PLATE_WIDTH_PX:
        return None
    return (
        f"ocr min_plate_width_px={floor} is below the synthetic corpus's legibility "
        f"floor of {MIN_LEGIBLE_PLATE_WIDTH_PX} px, so plates in the "
        f"[{floor}, {MIN_LEGIBLE_PLATE_WIDTH_PX}) band are handed to the engine after "
        f"truth has already marked them illegible. Any string read there is fabricated "
        f"rather than recovered, and the engines do not reliably refuse it: TemplateOCR "
        f"returns a string for every plate in that band and its score floor rejects only "
        f"the narrowest of them. Set min_plate_width_px to "
        f"{MIN_LEGIBLE_PLATE_WIDTH_PX} unless measuring the fabrication rate is the "
        f"point, and read the width buckets rather than the average either way."
    )
