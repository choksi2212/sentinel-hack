"""One YAML file drives thirty cameras, so its loader is load-bearing for the whole
source-independence claim -- swapping offline for live is meant to be a configuration change
and nothing else. These tests are weighted toward the three places ai/config.py deliberately
refuses to be helpful, because those are the ones whose failure is silent and expensive:

**Tier 1 -- the credential-safety refusals.** An unresolved ``${VAR}`` must raise, never
survive as the literal eleven characters ``${SENTINEL_`` sent as a password -- a server
rejecting that is indistinguishable from a wrong credential, and someone rotates a password
that was never wrong. Secrets come from the environment, never from YAML that lives in git
forever. And ``.env`` is parsed into a dict this module owns, never written into
``os.environ``, because ai/media shells out to ffmpeg and a subprocess inherits the
environment -- the database password must not end up in every ffmpeg's ``/proc/<pid>/environ``.
Redaction is the fourth: a credential in a log line is a leak whether or not the run
succeeded.

**Tier 2 -- resolution, layering and override correctness.** ``deep_merge`` replacing lists
rather than concatenating them (concatenation silently doubles the OCR stage's variant cost);
``apply_overrides`` refusing to create a new leaf (a typo'd ``--set`` would be accepted,
do nothing, and label the run with a setting it never used); the extends chain, and its
cycle detection.

**Tier 3 -- validation, warnings and accessors.** The structural validation, and the two
warnings that run fine and quietly produce a worse number -- most sharply the shipped default
where the detector's 0.35 threshold sits above the tracker's 0.1 low band and silently
deletes ByteTrack's second stage.

Only the extends-chain and load_config tests need PyYAML; it is import-guarded so the rest of
the module runs without it.
"""

import os

import pytest

from ai.config import (
    REDACTED,
    AppConfig,
    ConfigError,
    _coerce,
    _SECRET_KEY,
    apply_overrides,
    deep_merge,
    load_config,
    load_env,
    parse_env_text,
    redact,
    resolve,
)

try:
    import yaml  # noqa: F401

    HAS_YAML = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_YAML = False

requires_yaml = pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")


def _pipeline_raw(**over):
    """A minimally-valid pipeline config as a plain dict (no YAML, no file)."""
    raw = {
        "source": {"mode": "file", "camera_id": "cam04"},
        "detect": {},
        "track": {},
        "plate": {},
        "ocr": {},
    }
    raw.update(over)
    return raw


# ============================================================================ TIER 1
# The one that would have cost an evening: an unresolved ${VAR} is an error.
# ============================================================================


def test_unset_placeholder_without_a_default_raises_rather_than_leaking_the_literal():
    """Not '${VAR}' sent as the value. The whole point: a literal placeholder reaching an
    RTSP URL as its password produces a 401 that looks exactly like a wrong credential."""
    with pytest.raises(ConfigError) as exc:
        resolve({"source": {"password": "${SENTINEL_RTSP_PASSWORD}"}}, {})
    # The message has to teach the fix, because the person hitting it is mid-incident.
    assert "SENTINEL_RTSP_PASSWORD" in str(exc.value)
    assert ":-" in str(exc.value)  # points at the ${VAR:-} escape hatch


def test_empty_default_means_optional_and_resolves_to_empty_string():
    """${VAR:-} is how you say 'optional, empty if unset' -- the explicit default is the
    opt-in that distinguishes 'I meant this to be blank' from 'I forgot to set it'."""
    assert resolve({"p": "${NOT_SET:-}"}, {}) == {"p": ""}


def test_default_is_used_when_the_var_is_unset():
    assert resolve({"p": "${PORT:-5432}"}, {}) == {"p": 5432}  # and coerced, see below


def test_a_set_var_beats_its_default():
    assert resolve({"p": "${PORT:-5432}"}, {"PORT": "6000"}) == {"p": 6000}


def test_an_empty_env_value_falls_through_to_the_default():
    """env[name] == '' is treated as unset. An exported-but-blank variable is a common CI
    accident, and honouring it over a sensible default is how a blank password ships."""
    assert resolve({"p": "${PORT:-5432}"}, {"PORT": ""}) == {"p": 5432}


def test_a_bare_dollar_is_not_magic():
    """Only ${...} interpolates. A bare $ appears in regexes and currency and must survive,
    or a config breaks depending on the content of its values."""
    assert resolve({"p": "$5 or a regex like a$"}, {}) == {"p": "$5 or a regex like a$"}


# ============================================================================ TIER 1
# Secrets come from the environment; the resolved tree is new, the input untouched.
# ============================================================================


def test_secret_is_pulled_from_env_not_written_in_yaml():
    tree = {"source": {"url": "rtsp://user:${PW}@cam/1"}}
    out = resolve(tree, {"PW": "s3cret"})
    assert out["source"]["url"] == "rtsp://user:s3cret@cam/1"
    # The secret never leaks back into the input tree; it still holds the placeholder.
    assert "s3cret" not in str(tree)
    assert tree["source"]["url"] == "rtsp://user:${PW}@cam/1"


def test_resolve_returns_a_new_tree_and_does_not_mutate_the_input():
    tree = {"a": {"b": "${X}"}}
    resolve(tree, {"X": "1"})
    assert tree == {"a": {"b": "${X}"}}


def test_embedded_placeholder_stays_a_string_but_a_whole_placeholder_takes_a_type():
    """"postgres://...:5432/db" is text even though 5432 looks numeric; ${PORT} alone is a
    port. The line is drawn at 'is the entire value one placeholder'."""
    out = resolve({"dsn": "db://h:${PORT}/x", "port": "${PORT}"}, {"PORT": "5432"})
    assert out["dsn"] == "db://h:5432/x"  # embedded -> string
    assert out["port"] == 5432  # whole -> coerced int


def test_a_non_interpolated_string_is_never_coerced():
    """camera_id: "01" is a literal string and must keep its leading zero; only values that
    were interpolated get type coercion."""
    assert resolve({"camera_id": "01"}, {}) == {"camera_id": "01"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("true", True), ("yes", True), ("on", True),
        ("false", False), ("no", False), ("off", False),
        ("null", None), ("none", None), ("~", None),
        ("42", 42), ("-3", -3), ("1.5", 1.5),
        ("01", 1),  # coerced values do lose a leading zero -- see the test above for why
        ("cam04", "cam04"), ("", ""),
    ],
)
def test_coerce_recognises_scalars_and_leaves_text_alone(text, expected):
    assert _coerce(text) == expected


# ============================================================================ TIER 1
# .env is parsed into a dict this module owns and is never written to os.environ.
# ============================================================================


def test_env_file_values_never_reach_os_environ(tmp_path, monkeypatch):
    """The load-bearing security property: a subprocess (ffmpeg) inherits os.environ, so a
    password written there lands in that process's /proc/<pid>/environ and any crash dump.
    load_env returns a dict; os.environ must be untouched."""
    key = "TRINETRA_DB_PASSWORD"
    monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}=synthetic-db-secret\n", encoding="utf-8")

    values = load_env(str(env_file))
    assert values[key] == "synthetic-db-secret"  # reached the dict this module owns
    assert key not in os.environ  # and never the process environment


def test_os_environ_wins_over_the_env_file(tmp_path, monkeypatch):
    """That is how a container or CI passes an override; a committed .env someone filled in
    locally must not silently beat the deployment's value."""
    key = "TRINETRA_OVERRIDE_PROBE"
    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}=from_file\n", encoding="utf-8")
    monkeypatch.setenv(key, "from_os")
    assert load_env(str(env_file))[key] == "from_os"


def test_missing_env_file_is_not_an_error(tmp_path):
    """Every value .env holds is optional or has a default, and the offline config needs
    none of them, so a missing file loads to (at most) the process environment."""
    values = load_env(str(tmp_path / "nope.env"))
    assert isinstance(values, dict)


def test_env_file_can_be_read_without_the_process_environment():
    assert load_env(path=None, include_os_environ=False) == {}


# ---- parse_env_text: passwords contain = and # and quotes, so nothing clever ----


def test_parse_env_takes_the_value_literally_after_the_first_equals():
    """A password with an '=' in it is common and must survive; partition on the first =."""
    assert parse_env_text("DSN=k=v=w")["DSN"] == "k=v=w"


def test_parse_env_does_not_treat_a_hash_inside_a_value_as_a_comment():
    """Only a whole-line comment is a comment. An inline # is a legal password character."""
    assert parse_env_text("PW=a#b#c")["PW"] == "a#b#c"


def test_parse_env_strips_only_quotes_that_wrap_the_whole_value():
    env = parse_env_text('A="quoted value"\nB=pa"ss\n')
    assert env["A"] == "quoted value"  # wrapping quotes removed
    assert env["B"] == 'pa"ss'  # an interior quote is part of the password


def test_parse_env_handles_export_prefix_comments_blanks_and_trim():
    env = parse_env_text("\n# a comment\nexport FOO = bar \n\n")
    assert env == {"FOO": "bar"}


def test_parse_env_rejects_a_line_that_is_not_key_value():
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        parse_env_text("this is not an assignment")


def test_parse_env_rejects_an_invalid_key():
    with pytest.raises(ConfigError, match="invalid key"):
        parse_env_text("1LEADING_DIGIT=x")


# ============================================================================ TIER 1
# Redaction -- the loggable form. Masked by key name, not by value.
# ============================================================================


@pytest.mark.parametrize(
    "key", ["password", "passwd", "db_secret", "api_token", "apikey", "api_key",
            "aws_credential", "bearer", "TRINETRA_DB_DSN"],
)
def test_secret_key_pattern_matches_the_credential_names(key):
    assert _SECRET_KEY.search(key)


@pytest.mark.parametrize("key", ["host", "port", "camera_id", "name", "mode", "width"])
def test_secret_key_pattern_leaves_ordinary_names_alone(key):
    assert not _SECRET_KEY.search(key)


def test_redact_masks_by_key_even_when_the_value_looks_innocent():
    """A password that happens to be numeric ('5432') is still a password."""
    assert redact({"db": {"password": "5432", "host": "h"}}) == {
        "db": {"password": REDACTED, "host": "h"}
    }


def test_redact_masks_a_whole_block_under_a_secret_looking_key():
    """The safe default for a structure nobody anticipated is to hide all of it, not to
    walk into it looking for leaves that happen to match."""
    assert redact({"credentials": {"user": "a", "pass": "b"}}) == {
        "credentials": {"user": REDACTED, "pass": REDACTED}
    }
    assert redact({"tokens": [1, 2, 3]}) == {"tokens": [REDACTED, REDACTED, REDACTED]}


def test_redact_leaves_an_empty_or_absent_secret_as_is():
    """Nothing to hide, and masking '' to '***' would falsely suggest a secret is set."""
    assert redact({"password": "", "token": None}) == {"password": "", "token": None}


def test_appconfig_redacted_is_the_loggable_form_and_raw_keeps_the_secret():
    cfg = AppConfig(path=None, raw={"source": {"password": "FAKE-FEED-PASS", "mode": "live_rtsp"}})
    assert cfg.redacted()["source"]["password"] == REDACTED
    assert cfg.redacted()["source"]["mode"] == "live_rtsp"  # non-secret survives
    assert cfg.raw["source"]["password"] == "FAKE-FEED-PASS"  # raw is unmasked, do not log


# ============================================================================ TIER 2
# deep_merge -- lists REPLACE. Concatenation silently doubles the OCR stage's cost.
# ============================================================================


def test_deep_merge_replaces_a_list_rather_than_concatenating_it():
    """[upscale_2x] + [gray] must be [gray], not two variants. Concatenation would double
    the OCR stage's per-crop cost -- the one number ocr/factory.py calls the stage's cost."""
    merged = deep_merge({"ocr": {"variants": ["upscale_2x"]}}, {"ocr": {"variants": ["gray"]}})
    assert merged["ocr"]["variants"] == ["gray"]


def test_deep_merge_recurses_into_mappings_and_keeps_untouched_keys():
    merged = deep_merge(
        {"a": {"x": 1, "y": 2}, "keep": True},
        {"a": {"y": 9, "z": 3}},
    )
    assert merged == {"a": {"x": 1, "y": 9, "z": 3}, "keep": True}


def test_deep_merge_overlay_scalar_replaces_base_mapping():
    assert deep_merge({"a": {"deep": 1}}, {"a": 5}) == {"a": 5}


# ============================================================================ TIER 2
# apply_overrides -- for one-off runs, and it refuses to invent a key.
# ============================================================================


def test_override_sets_an_existing_value_with_coercion():
    out = apply_overrides({"detect": {"confidence_threshold": 0.25}}, ["detect.confidence_threshold=0.1"])
    assert out["detect"]["confidence_threshold"] == 0.1
    assert isinstance(out["detect"]["confidence_threshold"], float)


def test_override_refuses_to_create_a_new_leaf():
    """--set ocr.min_score=0.6 when the field is min_template_score would otherwise be
    accepted, do nothing, and leave the run labelled with a setting it did not use."""
    with pytest.raises(ConfigError, match="no such key"):
        apply_overrides({"ocr": {"min_template_score": 0.6}}, ["ocr.min_score=0.9"])


def test_override_refuses_a_missing_section():
    with pytest.raises(ConfigError, match="no section"):
        apply_overrides({"detect": {}}, ["nosuch.key=1"])


@pytest.mark.parametrize("bad", ["noequals", "=emptykey"])
def test_override_rejects_malformed_items(bad):
    with pytest.raises(ConfigError):
        apply_overrides({"a": {"b": 1}}, [bad])


def test_override_deep_copies_so_the_input_is_untouched():
    """A caller that loads one config and applies several override sets to compare them must
    not have the second comparison inherit the first."""
    tree = {"detect": {"confidence_threshold": 0.25}}
    apply_overrides(tree, ["detect.confidence_threshold=0.9"])
    assert tree["detect"]["confidence_threshold"] == 0.25


# ============================================================================ TIER 2
# extends chains -- resolved base-first, relative to the declaring file, cycles named.
# ============================================================================


@requires_yaml
def test_extends_merges_base_first_and_the_child_wins(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "source:\n  mode: file\ndetect: {}\ntrack: {}\nplate: {}\nocr: {}\n"
        "emit:\n  enabled: false\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nemit:\n  enabled: true\n", encoding="utf-8"
    )
    cfg = load_config(str(tmp_path / "child.yaml"))
    assert cfg.get("source.mode") == "file"  # inherited
    assert cfg.get("emit.enabled") is True  # overridden


@requires_yaml
def test_extends_cycle_is_detected_and_names_the_chain(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(str(tmp_path / "a.yaml"), validate=False)


@requires_yaml
def test_extends_must_be_a_filename(tmp_path):
    (tmp_path / "c.yaml").write_text("extends: [not, a, name]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="extends"):
        load_config(str(tmp_path / "c.yaml"), validate=False)


@requires_yaml
def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "ghost.yaml"))


@requires_yaml
def test_top_level_must_be_a_mapping(tmp_path):
    (tmp_path / "list.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(str(tmp_path / "list.yaml"), validate=False)


# ============================================================================ TIER 2 / 3
# validate() -- structural only. Empty list is a valid config.
# ============================================================================


def test_a_minimal_pipeline_config_is_valid():
    assert AppConfig(path=None, raw=_pipeline_raw()).validate() == []


def test_source_mode_live_is_rejected_because_ingest_has_a_check_on_it():
    """'live' is the plausible wrong value -- the schema knows live_rtsp and live_hls, and a
    bare 'live' would 422 at ingest. Catch it here where the field can be named."""
    errors = AppConfig(path=None, raw=_pipeline_raw(source={"mode": "live"})).validate()
    assert any("live" in e for e in errors)


def test_missing_model_stage_sections_are_each_flagged():
    """The four model stages must be named explicitly: defaulting one labels a run with a
    model choice nobody made, and provenance travels in every event."""
    raw = {"source": {"mode": "file"}}  # no detect/track/plate/ocr
    errors = AppConfig(path=None, raw=raw).validate()
    for stage in ("detect", "track", "plate", "ocr"):
        assert any(stage in e for e in errors)


def test_missing_source_mode_is_an_error():
    errors = AppConfig(path=None, raw={"detect": {}, "track": {}, "plate": {}, "ocr": {}}).validate()
    assert any("source.mode" in e for e in errors)


def test_training_kind_skips_the_pipeline_requirements():
    """A training run has no source adapter and no inference stages; requiring them would
    make the correct config fail."""
    assert AppConfig(path=None, raw={"kind": "training"}).validate() == []


def test_an_unknown_kind_is_rejected():
    assert any("kind" in e for e in AppConfig(path=None, raw={"kind": "sideways"}).validate())


def test_a_hand_built_config_with_a_leftover_placeholder_is_flagged():
    """resolve() raises on load, so reaching validate() with a '${...}' means the config was
    built by hand -- still caught, because the cost of missing it is a literal sent as a value."""
    raw = _pipeline_raw(source={"mode": "file", "url": "${STILL_HERE}"})
    assert any("placeholder" in e for e in AppConfig(path=None, raw=raw).validate())


def test_require_valid_raises_with_every_error_listed():
    with pytest.raises(ConfigError) as exc:
        AppConfig(path="x.yaml", raw={"source": {"mode": "live"}}).require_valid()
    assert "x.yaml" in str(exc.value)


# ============================================================================ TIER 3
# warnings() -- runs fine, produces a worse number. The shipped-default trap.
# ============================================================================


def test_warnings_catches_the_shipped_default_where_the_detector_starves_the_tracker():
    """detector 0.35 above tracker low 0.1 discards every box in [0.1, 0.35) before the
    tracker sees it, deleting ByteTrack's second association stage -- no error, no obviously
    wrong number, just a matched_low counter at zero. This is the default config's bug."""
    cfg = AppConfig(path=None, raw={
        "detect": {"confidence_threshold": 0.35},
        "track": {"low_threshold": 0.1},
    })
    assert any("low_threshold" in w for w in cfg.warnings())


def test_warnings_is_quiet_when_the_detector_sits_at_or_below_the_low_band():
    cfg = AppConfig(path=None, raw={
        "detect": {"confidence_threshold": 0.1},
        "track": {"low_threshold": 0.1},
    })
    assert all("low_threshold" not in w for w in cfg.warnings())


def test_warnings_returns_a_list_even_with_nothing_configured():
    assert AppConfig(path=None, raw={}).warnings() == []


# ============================================================================ TIER 3
# Accessors.
# ============================================================================


def test_get_walks_a_dotted_path_and_returns_the_default_on_a_miss():
    cfg = AppConfig(path=None, raw={"a": {"b": {"c": 7}}})
    assert cfg.get("a.b.c") == 7
    assert cfg.get("a.b.x", "fallback") == "fallback"
    assert cfg.get("nope") is None


def test_section_missing_is_empty_and_a_non_mapping_section_raises():
    cfg = AppConfig(path=None, raw={"source": "not-a-map"})
    assert AppConfig(path=None, raw={}).section("source") == {}
    with pytest.raises(ConfigError, match="mapping"):
        cfg.section("source")


def test_source_config_lets_the_command_line_camera_win():
    """--camera cam07 drives thirty cameras from one config instead of thirty files that
    drift apart."""
    cfg = AppConfig(path=None, raw={"source": {"mode": "file", "camera_id": "cam01"}})
    assert cfg.source_config("cam07")["camera_id"] == "cam07"
    assert cfg.source_config()["camera_id"] == "cam01"  # falls back to the file


def test_source_mode_is_synthetic_and_kind_defaults():
    cfg = AppConfig(path=None, raw={"source": {"mode": "synthetic"}})
    assert cfg.source_mode == "synthetic"
    assert cfg.is_synthetic is True
    assert cfg.kind == "pipeline"


# ============================================================================ TIER 3
# load_config -- extends, then overrides, then interpolation, in that order.
# ============================================================================


@requires_yaml
def test_load_config_applies_overrides_before_interpolation(tmp_path):
    """An override that introduces a ${VAR} must still resolve, so overrides run first and
    interpolation last -- one check for a placeholder introduced by any layer."""
    (tmp_path / "c.yaml").write_text(
        "source:\n  mode: file\n  note: placeholder\ndetect: {}\ntrack: {}\nplate: {}\nocr: {}\n",
        encoding="utf-8",
    )
    cfg = load_config(
        str(tmp_path / "c.yaml"), env={"NOTE": "resolved"}, overrides=["source.note=${NOTE}"]
    )
    assert cfg.get("source.note") == "resolved"


@requires_yaml
def test_load_config_validate_toggle(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "source:\n  mode: file\ndetect: {}\ntrack: {}\nplate: {}\nocr: {}\n", encoding="utf-8"
    )
    # validate=False lets a knowingly-bad override through for inspection.
    bad = load_config(str(tmp_path / "c.yaml"), env={}, overrides=["source.mode=live"], validate=False)
    assert bad.get("source.mode") == "live"
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "c.yaml"), env={}, overrides=["source.mode=live"])


@requires_yaml
def test_load_config_raises_on_an_unresolved_secret_at_load_time(tmp_path):
    """The end-to-end version of the tier-1 unit test: a real file with an unset ${VAR}
    fails at load, not at connect."""
    (tmp_path / "c.yaml").write_text(
        "source:\n  mode: live_rtsp\n  password: ${SENTINEL_RTSP_PASSWORD}\n"
        "detect: {}\ntrack: {}\nplate: {}\nocr: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="SENTINEL_RTSP_PASSWORD"):
        load_config(str(tmp_path / "c.yaml"), env={})
