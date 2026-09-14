"""Load every config and check it, without loading a model.

    python scripts/validate_config.py                    # all of config/
    python scripts/validate_config.py config/live.yaml   # one
    python scripts/validate_config.py --deep             # also build the stages

Exists because the alternative is finding out at frame 1 of a live run. Two levels:

**Shallow (default).** Structure, ${VAR} resolution, and every stage factory's own key check
via its normalize_* function. Imports nothing heavy, so it runs in CI where there is no GPU
and no model weights, and it catches the whole class of error that is a typo -- a key the
factory does not accept is rejected here rather than silently ignored for a week.

**Deep (--deep).** Actually constructs the source and the four stages. Downloads weights and
needs a GPU, so it is not what CI runs; it is what you run before a demo, because a config
can be perfectly valid and still name a checkpoint that does not exist.

**Warnings are printed and do not fail the run**, and that distinction is the point of this
script. The two known ones -- a detector threshold above the tracker's floor, an OCR width
floor below the corpus's legibility floor -- produce a pipeline that works and quietly
returns a worse number. Nothing crashes, so nothing tells you, and two days later there is a
benchmark figure with no explanation. Neither is an error because both are legitimate
experiments; both are printed every single time so choosing one is deliberate.

Exit codes: 0 clean, 1 errors, 2 could not read a file. Warnings alone still exit 0 -- a
warning that fails a build gets suppressed, and a suppressed warning is worse than none.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import ConfigError, load_config  # noqa: E402

DEFAULT_CONFIG_DIR = "config"


def _stage_checks(config) -> tuple[list[str], list[str]]:
    """Run each stage factory's own normalize_* on its block. Errors, warnings.

    Delegated rather than reimplemented. Each factory already knows the keys its backends
    accept and produces a better message than this script could -- build_source names the
    accepted keys for the mode it was given. A second copy here would be the one that drifts
    when a key is added.
    """
    errors: list[str] = []
    from ai.detect.factory import normalize_detector_config
    from ai.media.factory import SourceConfigError
    from ai.ocr.factory import normalize_ocr_config
    from ai.plate.factory import normalize_plate_config
    from ai.track.factory import normalize_tracker_config

    checks = [
        ("detect", normalize_detector_config),
        ("track", normalize_tracker_config),
        ("plate", normalize_plate_config),
        ("ocr", normalize_ocr_config),
    ]
    for section, normalize in checks:
        block = config.section(section)
        if not block:
            continue
        try:
            normalize(block)
        except (ValueError, SourceConfigError) as exc:
            errors.append(f"{section}: {exc}")

    # The source block is validated by building it, because build_source is where the
    # mode-specific key check lives and there is no normalize-only entry point. Cheap for
    # every mode except the live ones, which open a socket -- so those are left to --deep.
    source = config.section("source")
    mode = source.get("mode")
    if source and mode not in ("live_rtsp", "live_hls"):
        from ai.media.factory import build_source

        try:
            build_source(source, camera_id=source.get("camera_id") or "cam01")
        except SourceConfigError as exc:
            errors.append(f"source: {exc}")
        except Exception as exc:  # noqa: BLE001
            # A synthetic or file source that fails for a non-config reason -- a missing
            # clip, most often. Reported as an error because the run would fail too, but
            # labelled so it is not mistaken for a malformed config.
            errors.append(f"source: could not construct ({type(exc).__name__}: {exc})")

    return errors, list(config.warnings())


def _missing_media(config) -> list[str]:
    """Warn when a file/frames config names footage that is not on disk.

    A warning rather than an error, and this cut both ways. VideoFileSource does not open its
    path at construction, so the config validates cleanly and the run then fails at frame
    zero -- which is the exact "valid config, missing artifact" case --deep exists for, and
    it costs nothing to catch here. But benchmark clips are gitignored (large, and they
    contain real registration plates), so in CI the file is legitimately absent and an error
    would fail every build.
    """
    source = config.section("source")
    mode = source.get("mode")
    key = {"file": "path", "frames": "directory"}.get(str(mode))
    if not key or not source.get(key):
        return []
    target = Path(str(source[key]))
    if target.exists():
        return []
    return [
        f"source.{key} does not exist: {target}. The config is structurally fine and this "
        f"is expected in CI -- benchmark footage is gitignored -- but a run started now "
        f"would fail at frame zero."
    ]


def _deep_checks(config) -> list[str]:
    """Construct the real stages. Downloads weights, needs a GPU."""
    errors: list[str] = []
    from ai.detect.factory import build_detector
    from ai.media.factory import build_source
    from ai.ocr.factory import build_ocr_engine
    from ai.plate.factory import build_plate_detector
    from ai.track.factory import build_tracker

    builders = [
        ("detect", build_detector, config.section("detect")),
        ("track", build_tracker, config.section("track")),
        ("plate", build_plate_detector, config.section("plate")),
        ("ocr", build_ocr_engine, config.section("ocr")),
    ]
    for label, build, block in builders:
        if not block:
            continue
        try:
            build(block)
        except Exception as exc:  # noqa: BLE001 - any failure here is a real failure
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    source = config.section("source")
    if source:
        try:
            build_source(source, camera_id=source.get("camera_id") or "cam01")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"source: {type(exc).__name__}: {exc}")
    return errors


def validate_file(path: Path, *, deep: bool, show_config: bool) -> tuple[int, int]:
    """Check one file. Returns (error count, warning count)."""
    print(f"\n{path}")
    try:
        # validate=False so structural errors are collected and printed alongside the stage
        # ones instead of raising on the first. A report listing four problems is one round
        # trip; a report listing the first of four is four.
        config = load_config(str(path), validate=False)
    except ConfigError as exc:
        print(f"  ERROR  {exc}")
        return 1, 0

    errors = list(config.validate())
    warnings: list[str] = []
    if not errors:
        stage_errors, warnings = _stage_checks(config)
        errors.extend(stage_errors)
        warnings.extend(_missing_media(config))
        if deep and not errors:
            errors.extend(_deep_checks(config))

    kind = config.kind
    label = config.get("run.name", "?")
    mode = config.source_mode or "-"
    print(f"  kind={kind}  run.name={label}  source.mode={mode}")

    for message in errors:
        print(f"  ERROR    {message}")
    for message in warnings:
        print(f"  WARNING  {message}")
    if not errors and not warnings:
        print("  OK")
    elif not errors:
        print("  OK (with warnings)")

    if show_config:
        # redacted(), never raw. This is the one place in the codebase that prints a whole
        # config, so it is the one place a live stream password could reach a terminal and
        # from there a pasted screenshot.
        import json

        print("  --- resolved (secrets redacted) ---")
        for line in json.dumps(config.redacted(), indent=2, default=str).splitlines():
            print(f"  {line}")

    return len(errors), len(warnings)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="config files; default is all of config/")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also build the stages (downloads weights, needs a GPU)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print each resolved config with secrets redacted",
    )
    args = parser.parse_args(argv)

    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        # base.yaml is excluded: it is a defaults layer with no source block, so validating
        # it as a runnable config would report a missing source every time and train
        # everyone to ignore this script's output.
        directory = Path(DEFAULT_CONFIG_DIR)
        if not directory.is_dir():
            print(f"no {DEFAULT_CONFIG_DIR}/ directory here", file=sys.stderr)
            return 2
        paths = sorted(p for p in directory.glob("*.yaml") if p.name != "base.yaml")

    missing = [p for p in paths if not p.is_file()]
    if missing:
        for path in missing:
            print(f"not found: {path}", file=sys.stderr)
        return 2

    total_errors = 0
    total_warnings = 0
    for path in paths:
        errors, warnings = validate_file(path, deep=args.deep, show_config=args.show)
        total_errors += errors
        total_warnings += warnings

    print(
        f"\n{len(paths)} config(s): {total_errors} error(s), {total_warnings} warning(s)"
    )
    if total_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
