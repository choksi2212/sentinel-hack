"""Make it structurally hard to log a credential -- the log is the file someone pastes into
a chat window when a camera stops responding, and the Sentinel HLS credentials belong to the
organisers. This module's whole reason to exist is that "remember to redact at the call site"
is a rule that holds until the one exception (a traceback, a dict repr, an f-string added
while debugging at 1am), so redaction lives in the log path instead. These tests are weighted
almost entirely toward the tier-1 property: a secret, in any of the shapes it actually
arrives in, must not reach a handler's output.

Every credential in this file is SYNTHETIC and credential-*shaped*, never a real one. Putting
the actual Postgres or Sentinel or HuggingFace value in a test as a string literal would
write it into git history forever -- which is precisely the failure ai/config.py exists to
prevent, reintroduced in the test suite. The redaction patterns match on structure and the
word next to the secret, not on any specific value, so a realistic fake exercises them
exactly as well.

The one architectural fact these tests exist to defend: redaction is a Formatter, not a
Filter. A Filter runs *before* formatting, so it never sees the line that gets written -- a
`%(url)s` in the format string and a traceback appended by exc_info both go to disk unexamined
while the filter reports success. That is not hypothetical; it is how a real token reached a
real line. So the formatter tests below feed exactly those two shapes.
"""

import io
import logging
import sys

import pytest

from ai.config import AppConfig
from ai.logging_setup import (
    DEFAULT_LEVEL,
    REDACTED,
    RedactingFilter,
    RedactingFormatter,
    get_logger,
    log_config,
    redact_text,
    setup_logging,
)

# Synthetic, credential-shaped. None of these is a real secret. `SECRET` is the sentinel we
# assert never survives; the others carry it inside a specific structure.
SECRET = "fakepw123456789"
DSN = f"postgresql://trinetra:{SECRET}@localhost:5432/trinetra"
HLS = f"https://sentinel.example/live.m3u8?token={SECRET}&camera=7"
RTSP = f"rtsp://operator:{SECRET}@cam.local:554/stream1"


def _record(msg, *, name="ai.test", level=logging.INFO, args=None, exc_info=None, **extra):
    rec = logging.LogRecord(name, level, "test.py", 1, msg, args, exc_info)
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


@pytest.fixture
def clean_ai_logger():
    """Save and restore the process-wide 'ai' logger, so setup_logging tests do not leak
    handlers into the rest of the suite (the very bug -- duplicate handlers -- one of them
    is about)."""
    logger = logging.getLogger("ai")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers = []
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.handlers, logger.level, logger.propagate = saved


# ============================================================================ TIER 1
# redact_text -- every shape a secret arrives in. Weighted heaviest because a miss here is
# a credential on disk.
# ============================================================================


def test_url_userinfo_password_is_removed_and_the_host_is_kept():
    """The line exists to say which camera stopped responding, so the host must survive; the
    password is the part that ends up in the log the instant a stream fails to open."""
    out = redact_text(f"open failed {RTSP}")
    assert SECRET not in out
    assert "cam.local:554" in out  # host and port kept -- the diagnostic value
    assert REDACTED in out


def test_signed_url_query_secret_is_removed_and_other_params_are_kept():
    """The HLS / object-store shape. Neither the userinfo pass nor the key=value pass would
    catch a ?token= on its own -- one looks only at userinfo, the other skips URLs."""
    out = redact_text(f"playlist {HLS}")
    assert SECRET not in out
    assert "camera=7" in out  # a non-secret query parameter is not collateral


def test_a_dsn_value_keeps_its_host_and_database():
    """dsn is a secret-named key whose value is a whole connection string. Blanket-masking it
    to *** is safe and useless -- it throws away the one thing the line was written to say.
    The userinfo pass removes the credential; the rest is kept."""
    out = redact_text(f"db dsn: {DSN}")
    assert SECRET not in out
    assert "localhost:5432/trinetra" in out


def test_key_value_secret_is_masked():
    assert SECRET not in redact_text(f"password={SECRET} host=h")
    assert "host=h" in redact_text(f"password={SECRET} host=h")


def test_json_dict_secret_is_masked_including_the_quote_before_the_colon():
    """{"password": "x"} is the single most likely way a config reaches a log line, and it
    only matches because the pattern allows a quote on the key side of the separator."""
    assert SECRET not in redact_text(f'config {{"password": "{SECRET}", "host": "h"}}')


def test_bearer_token_separated_by_whitespace_is_masked():
    """The secret here is separated from its keyword by a space, not = or :, so the key=value
    pass walks straight past it -- which it did, on a real token, until the harness said so."""
    out = redact_text(f"Authorization: Bearer {SECRET}")
    assert SECRET not in out
    assert "Bearer ***" in out


def test_the_bearer_floor_leaves_short_prose_alone():
    """A 12-character floor keeps 'token expired' and 'bearer required' from being eaten; a
    real credential shorter than 12 chars is rarer than that prose."""
    assert redact_text("the token expired") == "the token expired"
    assert redact_text("a bearer is required") == "a bearer is required"


def test_an_already_redacted_query_value_is_not_re_eaten():
    """After the query pass leaves token=***, the key=value pass must not swallow the &x=1
    that follows -- losing a query parameter is worse than the mask it already has."""
    out = redact_text(f"url https://s.example/x.m3u8?token={SECRET}&x=1")
    assert "&x=1" in out
    assert SECRET not in out


def test_multiple_secrets_on_one_line_are_all_removed():
    line = f"{RTSP} and {HLS} and password={SECRET}"
    out = redact_text(line)
    assert out.count(SECRET) == 0


def test_empty_text_is_returned_unchanged():
    assert redact_text("") == ""


def test_the_documented_limit_a_bare_word_secret_passes_through():
    """This is a net, not a guarantee: it recognises secrets by the word next to them. A
    password logged as a bare token with no key and no URL cannot be caught by anything
    operating on formatted text, which is why config.py still redacts by key at the source.
    Pinned so the limit is a known boundary, not a surprise."""
    assert redact_text(f"the value is {SECRET}") == f"the value is {SECRET}"


# ============================================================================ TIER 1
# Realistic composite lines -- the "test against real credentials" discipline, run against
# realistic fakes rather than the real thing.
# ============================================================================


@pytest.mark.parametrize("line", [
    f"psycopg2.OperationalError: could not connect to {DSN}",
    f"ffmpeg: failed to open input {RTSP}",
    f"401 Unauthorized fetching {HLS}",
    f'startup config {{"source": {{"password": "{SECRET}", "mode": "live_rtsp"}}}}',
])
def test_no_secret_survives_a_realistic_log_line(line):
    assert SECRET not in redact_text(line)


# ============================================================================ TIER 1
# RedactingFormatter -- a Formatter, not a Filter. This is the class's whole point.
# ============================================================================


def test_formatter_redacts_a_url_substituted_from_the_format_string():
    """A Filter cannot do this: it never sees %(url)s resolved. The formatter redacts the
    line after substitution, which is the last thing before a handler writes it."""
    formatter = RedactingFormatter("%(message)s url=%(url)s")
    out = formatter.format(_record("stream open", url=RTSP))
    assert SECRET not in out


def test_formatter_redacts_a_traceback_appended_by_exc_info():
    """The other thing a Filter misses: a library's exception text that helpfully included the
    URL it failed to open, appended to the record by exc_info."""
    try:
        raise RuntimeError(f"connection to {RTSP} refused")
    except RuntimeError:
        record = _record("upstream error", level=logging.ERROR, exc_info=sys.exc_info())
    out = RedactingFormatter("%(message)s").format(record)
    assert SECRET not in out
    assert "RuntimeError" in out  # the exception type survives; only the credential goes


def test_formatter_falls_back_rather_than_raising_on_a_bad_record():
    """A bad format string must not kill a worker: logging that raises from an error path
    loses the very thing it was reporting. The fallback names the record and is itself
    redacted."""
    record = _record("x %s y %s", args=("only-one-arg",))  # too few args for the format
    out = RedactingFormatter("%(message)s").format(record)
    assert "unformattable log record" in out


def test_formatter_fallback_is_also_redacted():
    record = _record("connecting %s %s", args=(RTSP,))  # too few args -> fallback path
    out = RedactingFormatter("%(message)s").format(record)
    assert "unformattable log record" in out
    assert SECRET not in out


# ============================================================================ TIER 2
# RedactingFilter -- belt to the braces, for a handler someone else attached.
# ============================================================================


def test_filter_masks_a_secret_named_field_by_key():
    """A foreign handler (caplog, a library's) has its own formatter and would render
    extra={"password": ...} verbatim. Masking on the record means it is gone first."""
    record = _record("hi", password=SECRET, host="h")
    assert RedactingFilter().filter(record) is True
    assert record.password == REDACTED
    assert record.host == "h"


def test_filter_redacts_a_url_valued_field_even_when_the_key_is_innocent():
    record = _record("hi", url=RTSP)
    RedactingFilter().filter(record)
    assert SECRET not in record.url


def test_filter_leaves_the_message_and_args_to_the_formatter():
    """An earlier version cleared args after folding them into msg, which made two handlers
    disagree. The filter must not touch msg/args."""
    record = _record("open %s", args=("plain",), password=SECRET)
    RedactingFilter().filter(record)
    assert record.msg == "open %s"
    assert record.args == ("plain",)


# ============================================================================ TIER 2
# setup_logging -- idempotent, on the 'ai' logger, redacting by default.
# ============================================================================


def test_setup_logging_is_idempotent(clean_ai_logger):
    """The thirty-handlers bug: a worker that reconfigures per camera prints every line
    thirty times, which reads as an event-loop bug and is not one."""
    first = setup_logging({}, stream=io.StringIO())
    count = len(first.handlers)
    second = setup_logging({}, stream=io.StringIO())
    assert second is first
    assert len(second.handlers) == count


def test_force_replaces_the_handlers(clean_ai_logger):
    setup_logging({}, stream=io.StringIO())
    setup_logging({}, stream=io.StringIO(), force=True)
    assert len(clean_ai_logger.handlers) == 1  # replaced, not appended


def test_setup_logging_configures_ai_not_root(clean_ai_logger):
    """Root belongs to whatever imports this -- pytest, a notebook, a FastAPI host. A library
    reconfiguring root is how a host loses its own log format to an import."""
    logger = setup_logging({}, stream=io.StringIO())
    assert logger.name == "ai"
    assert logger.propagate is False
    assert logging.getLogger().handlers is not logger.handlers


def test_level_override_beats_the_config_block(clean_ai_logger):
    logger = setup_logging({"level": "WARNING"}, level="DEBUG", stream=io.StringIO())
    assert logger.level == logging.DEBUG


def test_default_level_when_unspecified(clean_ai_logger):
    logger = setup_logging({}, stream=io.StringIO())
    assert logger.level == getattr(logging, DEFAULT_LEVEL)


def test_redaction_is_on_by_default(clean_ai_logger):
    logger = setup_logging({}, stream=io.StringIO())
    handler = logger.handlers[0]
    assert isinstance(handler.formatter, RedactingFormatter)
    assert any(isinstance(f, RedactingFilter) for f in handler.filters)


def test_redact_secrets_false_is_the_only_way_to_an_unredacted_handler(clean_ai_logger):
    """And config/live.yaml's comment says never to write it. There is no other config value
    that produces an unredacted handler by accident."""
    logger = setup_logging({"redact_secrets": False}, stream=io.StringIO())
    handler = logger.handlers[0]
    assert not isinstance(handler.formatter, RedactingFormatter)
    assert not any(isinstance(f, RedactingFilter) for f in handler.filters)


def test_end_to_end_a_secret_in_message_and_extra_never_reaches_the_stream(clean_ai_logger):
    buffer = io.StringIO()
    logger = setup_logging({}, stream=buffer, force=True)
    logger.info("stream open", extra={"url": RTSP, "password": SECRET})
    logger.info("db %s", DSN)
    output = buffer.getvalue()
    assert SECRET not in output
    assert "localhost:5432/trinetra" in output  # the diagnostic host survives


def test_log_file_creates_the_directory_and_delays_the_file(clean_ai_logger, tmp_path):
    """delay=True so a run that never logs leaves no empty file, and a permission problem
    surfaces on first write with a message rather than at import with a stray traceback."""
    path = tmp_path / "logs" / "run.log"
    logger = setup_logging({}, stream=io.StringIO(), log_file=str(path), force=True)
    assert path.parent.is_dir()
    assert not path.exists()  # delayed
    logger.warning("first line")
    assert path.exists()
    assert SECRET not in path.read_text(encoding="utf-8")


# ============================================================================ TIER 3
# get_logger and log_config.
# ============================================================================


def test_get_logger_namespaces_under_ai():
    """getLogger('worker') would attach to root and bypass the redacting handlers -- the one
    mistake here with no symptom until a password is already in a file."""
    assert get_logger("worker").name == "ai.worker"
    assert get_logger("ai.media").name == "ai.media"  # already-namespaced is left alone


def test_log_config_redacts_an_appconfig(clean_ai_logger):
    buffer = io.StringIO()
    logger = setup_logging({}, stream=buffer, force=True)
    cfg = AppConfig(path="live.yaml", raw={"source": {"password": SECRET, "mode": "live_rtsp"}})
    log_config(logger, cfg)
    output = buffer.getvalue()
    assert SECRET not in output
    assert "live.yaml" in output  # the path is logged so the run is reproducible


def test_log_config_on_a_raw_dict_is_caught_by_the_filter(clean_ai_logger):
    """log_config would happily format a plain dict with a password in it; the RedactingFilter
    is the second line that stops it reaching the file."""
    buffer = io.StringIO()
    logger = setup_logging({}, stream=buffer, force=True)
    log_config(logger, {"source": {"password": SECRET}})
    assert SECRET not in buffer.getvalue()
