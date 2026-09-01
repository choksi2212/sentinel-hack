"""Load and validate a run's configuration. One file drives thirty cameras.

    cfg = load_config("config/offline.yaml")
    src = build_source(cfg.source_config(camera_id="cam04"))
    print(cfg.redacted())            # safe to log
    for w in cfg.warnings(): ...     # the misconfigurations that produce no error

The acceptance test for the whole source-independence design is that swapping offline for
live is *a configuration change only* -- so this file is load-bearing for the claim, and
the interesting parts are the three places it refuses to be helpful.

**Secrets come from the environment, never from YAML.** A password in config/live.yaml is
a password in the git history forever, and rotating it after submission does not remove it
from the clone a judge already made. YAML references a secret as `${SENTINEL_RTSP_PASSWORD}`
and this module resolves it.

**An unresolved `${VAR}` is an error, not a literal.** This is the one that would have cost
an evening. If `${SENTINEL_RTSP_PASSWORD}` is unset and the placeholder survives into the
config, the RTSP URL carries the eleven characters `${SENTINEL_` etc. as its password, the
server returns 401, and the failure is indistinguishable from a wrong password -- so
somebody rotates a credential that was never wrong. Write `${VAR:-}` to mean "optional,
empty if unset"; the explicit default is how you say you meant it.

**`.env` is not loaded into os.environ.** python-dotenv is installed here and does exactly
that, and it is the wrong shape for this pipeline: ai/media shells out to ffmpeg, and a
subprocess inherits os.environ. Loading the database password into the environment of every
ffmpeg the worker spawns puts it in that process's `/proc/<pid>/environ` and in any crash
report the process writes. So .env is parsed into a dict this module owns, os.environ is
consulted as a fallback but never written, and the values reach exactly the two places that
need them.
"""

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

DEFAULT_ENV_FILE = ".env"
DEFAULT_CONFIG_DIR = "config"

# What gets masked in redacted(). Substring match on the key, case-insensitive, because the
# alternative -- an allowlist of known-secret keys -- fails silently the first time someone
# adds a field. A false positive here costs a masked value in a log line; a false negative
# costs a credential in one.
_SECRET_KEY = re.compile(
    r"password|passwd|secret|token|api_?key|credential|bearer|dsn", re.IGNORECASE
)
REDACTED = "***"

# ${VAR} and ${VAR:-default}. Deliberately not $VAR: a bare dollar appears in real values
# (a regex, a currency figure) and making it magic means a config that breaks depending on
# its content. The braces are the opt-in.
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """The configuration cannot be loaded or is internally inconsistent.

    Raised, never warned. The distinction this file draws throughout: an error means the
    run cannot proceed correctly, a warning means it can proceed and will produce a worse
    number than the operator expects. Both are reported; only one stops the run.
    """


# ------------------------------------------------------------------------ .env


def parse_env_text(text: str) -> dict[str, str]:
    """Parse .env content. KEY=VALUE, # comments, optional quotes, nothing clever.

    No interpolation inside .env itself and no `export` prefix handling, because both are
    shell features and this is not a shell. A value is taken literally after the first `=`,
    which is what makes a password containing `=` or `#` work -- and passwords contain both.
    Quotes are stripped only when they wrap the whole value, so `pa"ss` survives intact.
    """
    values: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ConfigError(f".env line {line_no} is not KEY=VALUE: {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ConfigError(f".env line {line_no} has an invalid key {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(
    path: Optional[str] = DEFAULT_ENV_FILE, *, include_os_environ: bool = True
) -> dict[str, str]:
    """Read .env into a dict. os.environ wins, and os.environ is never written.

    os.environ wins because that is how CI and a container pass overrides, and a committed
    .env.example that someone filled in locally must not silently beat the value the
    deployment set. A missing .env is not an error: every value it holds is either optional
    or has a `${VAR:-}` default, and the offline configuration needs none of them.
    """
    values: dict[str, str] = {}
    if path:
        env_path = Path(path)
        if env_path.is_file():
            values.update(parse_env_text(env_path.read_text(encoding="utf-8")))
    if include_os_environ:
        values.update({k: v for k, v in os.environ.items()})
    return values


# ----------------------------------------------------------------- resolution


def _resolve_scalar(value: str, env: Mapping[str, str], *, where: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        name, default = match.group(1), match.group(2)
        if name in env and env[name] != "":
            return env[name]
        if default is not None:
            return default
        raise ConfigError(
            f"{where}: {name} is not set and ${{{name}}} has no default. Either set it in "
            f".env (see .env.example) or write ${{{name}:-}} to mean it is optional. "
            f"Leaving the placeholder unresolved would send the literal text "
            f"'${{{name}}}' as the value, and a server rejecting that looks exactly like "
            f"a wrong credential."
        )

    return _PLACEHOLDER.sub(replace, value)


def _coerce(text: str) -> Any:
    """Turn a fully-substituted string back into a bool/int/float where it clearly is one.

    Needed because `port: ${TRINETRA_DB_PORT}` resolves to the string "5432" and a caller
    doing `int(port)` works while `port + 1` does not. Only applied to values that *were*
    interpolated -- a YAML string that always was a string stays one, so `camera_id: "01"`
    does not become the integer 1 and lose its leading zero.
    """
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def resolve(node: Any, env: Mapping[str, str], *, where: str = "config") -> Any:
    """Walk the tree substituting ${VAR}. Returns a new tree; the input is untouched."""
    if isinstance(node, Mapping):
        return {k: resolve(v, env, where=f"{where}.{k}") for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [resolve(v, env, where=f"{where}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, str) and "${" in node:
        substituted = _resolve_scalar(node, env, where=where)
        # A value that is *entirely* one placeholder takes the coerced type; one embedded
        # in a larger string stays a string, because "postgres://...:5432/db" is text.
        if _PLACEHOLDER.fullmatch(node):
            return _coerce(substituted)
        return substituted
    return node


def deep_merge(base: Any, overlay: Any) -> Any:
    """Recursive merge for mappings. Lists and scalars in the overlay REPLACE.

    Replacement rather than concatenation for lists, because the alternative is worse in a
    specific way. `variants: [upscale_2x]` in base plus `variants: [gray]` in an overlay
    would concatenate to two variants, silently doubling the cost of the OCR stage -- the
    one number ai/ocr/factory.py calls the number that sets the stage's cost. An overlay
    that wants both says both.
    """
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return overlay


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        # -------------------------------------------------------------------------
        # MANUAL STEP REQUIRED -- not a code defect, an environment gap.
        #
        #     pip install pyyaml
        #
        # Listed in ai/README.md.
        # -------------------------------------------------------------------------
        raise ConfigError("pyyaml is not installed. Run: pip install pyyaml") from exc
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        # safe_load, not load: a config file is data. full_load would let a YAML tag
        # construct arbitrary Python objects, and config/ is the kind of path that ends up
        # taking a filename from a command line.
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - yaml raises several unrelated types
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level, got {type(loaded).__name__}")
    return loaded


def _load_layers(path: Path, *, seen: Optional[list[Path]] = None) -> dict[str, Any]:
    """Resolve `extends:` chains into one mapping, base first.

    `extends` is relative to the file that declares it, so config/offline.yaml says
    `extends: base.yaml` and moving the directory does not break it. Cycles are detected and
    named rather than recursing until the stack dies -- the traceback from that tells you
    nothing about which two files point at each other.
    """
    seen = list(seen or [])
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in seen + [resolved])
        raise ConfigError(f"circular extends: {chain}")
    seen.append(resolved)

    raw = _read_yaml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    if not isinstance(parent_ref, str):
        raise ConfigError(f"{path.name}: extends must be a filename, got {parent_ref!r}")
    parent = _load_layers(path.parent / parent_ref, seen=seen)
    return deep_merge(parent, raw)


def apply_overrides(tree: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides from the command line.

    For one-off runs -- `--set ocr.name=oracle` to check a stage against ground truth
    without editing a file that is under review. The value goes through the same coercion
    as an interpolated one, so `--set detect.confidence_threshold=0.1` is a float.

    Refuses to create a new leaf under a key that does not exist, because `--set
    ocr.min_score=0.6` when the field is called min_template_score would otherwise be
    accepted, do nothing, and leave a run labelled with a setting it did not use.

    Deep-copies first. A shallow copy shares its nested dicts with the caller's tree, so
    setting `detect.confidence_threshold` would reach through and change the input -- which
    matters for a caller that loads one config and applies several override sets to compare
    them, and would silently make the second comparison inherit the first.
    """
    result = deepcopy(dict(tree))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not key=value")
        dotted, _, raw_value = item.partition("=")
        parts = [p for p in dotted.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"override {item!r} has an empty key")
        node: Any = result
        for depth, part in enumerate(parts[:-1]):
            if not isinstance(node, dict) or part not in node:
                path_so_far = ".".join(parts[: depth + 1])
                raise ConfigError(
                    f"override {dotted!r}: no section {path_so_far!r} in this config"
                )
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(
                f"override {dotted!r}: no such key. Overrides set existing values; a typo "
                f"that created a new one would be accepted and ignored, and the run would "
                f"be labelled with a setting it never used."
            )
        node[parts[-1]] = _coerce(raw_value)
    return result


# -------------------------------------------------------------------- redaction


def redact(node: Any, *, _key: str = "") -> Any:
    """Copy of the tree with anything that looks like a credential masked.

    Applied by key name rather than by value, so a password that happens to look like a
    port number is still masked. Nested under a secret-looking key counts too: a whole
    `credentials:` block is masked rather than only its leaves, because the safe default
    for a structure nobody anticipated is to hide it.
    """
    if _SECRET_KEY.search(_key or ""):
        if isinstance(node, Mapping):
            return {k: REDACTED for k in node}
        if isinstance(node, (list, tuple)):
            return [REDACTED for _ in node]
        return REDACTED if node not in (None, "") else node
    if isinstance(node, Mapping):
        return {k: redact(v, _key=str(k)) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [redact(v, _key=_key) for v in node]
    return node


# ------------------------------------------------------------------- AppConfig


@dataclass
class AppConfig:
    """One resolved run configuration.

    `raw` holds resolved values including secrets and must not be logged. Use redacted().
    """

    path: Optional[str]
    raw: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- accessors

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """A top-level block as a dict. Missing means empty, not an error.

        Empty rather than raising because every factory in ai/ already validates its own
        block and produces a better message than this file could -- build_source names the
        accepted keys for the mode it was given. Duplicating that here would mean two
        places to update when a key is added, and the one that drifts is this one.
        """
        value = self.raw.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name!r} must be a mapping, got {type(value).__name__}")
        return dict(value)

    def source_config(self, camera_id: Optional[str] = None) -> dict[str, Any]:
        """The `source:` block, with camera_id applied.

        camera_id from the command line wins over the file, which is how `--camera cam07`
        drives thirty cameras from one config instead of thirty files that drift apart.
        """
        block = self.section("source")
        if camera_id:
            block["camera_id"] = camera_id
        return block

    @property
    def source_mode(self) -> Optional[str]:
        mode = self.get("source.mode")
        return str(mode) if mode is not None else None

    @property
    def is_synthetic(self) -> bool:
        return self.source_mode == "synthetic"

    @property
    def kind(self) -> str:
        """"pipeline" (the default) or "training". They validate differently.

        A field rather than a filename check, because "config/training.yaml is special"
        is a rule that lives nowhere and breaks the moment somebody copies the file. A
        training run has no source adapter and no inference stages, so requiring them
        would make the correct config fail; declaring the kind is how it says so.
        """
        return str(self.raw.get("kind", "pipeline"))

    def redacted(self) -> dict[str, Any]:
        """The whole config with credentials masked. This is the loggable form."""
        return redact(self.raw)

    # --------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Errors that mean the run cannot proceed. Empty list is a valid config.

        Structural only. Every stage's own factory does the deep validation of its block,
        and calling them here would mean importing torch to check a YAML file -- which is
        why scripts/validate_config.py exists as a separate step that may be slow.
        """
        errors: list[str] = []
        if self.kind not in ("pipeline", "training"):
            errors.append(f"kind {self.kind!r} must be 'pipeline' or 'training'")
        elif self.kind == "pipeline":
            errors.extend(self._validate_pipeline())
        for leftover in _find_placeholders(self.raw):
            # Belt and braces: resolve() raises on an unresolved placeholder, so reaching
            # here means a config was built by hand rather than loaded. Still checked,
            # because the cost of missing it is a literal '${...}' sent as a password.
            errors.append(f"unresolved placeholder at {leftover}")
        return errors

    def _validate_pipeline(self) -> list[str]:
        errors: list[str] = []
        mode = self.source_mode
        if not mode:
            errors.append("source.mode is required; without it no adapter can be built")
        else:
            from ai.contracts.enums import SOURCE_MODES

            if mode not in SOURCE_MODES:
                errors.append(
                    f"source.mode {mode!r} is not one of {sorted(SOURCE_MODES)}; the event "
                    f"schema has a CHECK constraint on this value and ingest returns 422"
                )
        for required in ("detect", "track", "plate", "ocr"):
            if required not in self.raw:
                errors.append(
                    f"missing required section {required!r}. The four model stages must be "
                    f"named explicitly: defaulting one means a run is labelled with a "
                    f"model choice nobody made, and model provenance travels in every "
                    f"event. Infrastructure sections (emit, snapshot) do default."
                )
        return errors

    def warnings(self) -> list[str]:
        """Misconfigurations that run fine and quietly produce a worse number.

        These are the ones worth having a function for. An error stops the run and gets
        fixed; a warning of this kind shows up as a benchmark figure nobody can explain,
        two days later, with the cause three files away. Both known instances are measured
        and documented at their source.
        """
        found: list[str] = []
        from ai.ocr.factory import check_ocr_width_floor
        from ai.track.base import DEFAULT_LOW_THRESHOLD, check_detector_threshold

        detect = self.section("detect")
        track = self.section("track")
        if "confidence_threshold" in detect:
            low = float(track.get("low_threshold", DEFAULT_LOW_THRESHOLD))
            message = check_detector_threshold(
                float(detect["confidence_threshold"]), low
            )
            if message:
                found.append(message)
        ocr_warning = check_ocr_width_floor(self.section("ocr"), synthetic=self.is_synthetic)
        if ocr_warning:
            found.append(ocr_warning)
        return found

    def require_valid(self) -> "AppConfig":
        errors = self.validate()
        if errors:
            where = self.path or "<config>"
            raise ConfigError(f"{where} is not usable:\n  - " + "\n  - ".join(errors))
        return self


def _find_placeholders(node: Any, where: str = "config") -> list[str]:
    if isinstance(node, Mapping):
        out: list[str] = []
        for key, value in node.items():
            out.extend(_find_placeholders(value, f"{where}.{key}"))
        return out
    if isinstance(node, (list, tuple)):
        out = []
        for index, value in enumerate(node):
            out.extend(_find_placeholders(value, f"{where}[{index}]"))
        return out
    if isinstance(node, str) and _PLACEHOLDER.search(node):
        return [where]
    return []


def load_config(
    path: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    env_file: Optional[str] = DEFAULT_ENV_FILE,
    overrides: Sequence[str] = (),
    validate: bool = True,
) -> AppConfig:
    """Load a config file, resolve its extends chain and its ${VAR} references.

    Order matters and is: extends chain merged base-first, then command-line overrides,
    then interpolation. Overrides before interpolation so that `--set
    source.password='${OTHER_VAR}'` resolves rather than arriving as a literal; and
    interpolation last so a placeholder introduced by any layer is caught by the one check.
    """
    config_path = Path(path)
    merged = _load_layers(config_path)
    if overrides:
        merged = apply_overrides(merged, overrides)
    environment = dict(env) if env is not None else load_env(env_file)
    resolved = resolve(merged, environment, where=config_path.name)
    config = AppConfig(path=str(config_path), raw=resolved)
    if validate:
        config.require_valid()
    return config
