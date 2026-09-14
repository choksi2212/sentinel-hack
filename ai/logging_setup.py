"""Configure logging once, and make it structurally hard to log a credential.

    setup_logging(config)                      # from a config's `logging:` block
    log = logging.getLogger("ai.worker")
    log.info("stream open", extra={"url": url})  # url is redacted on the way out

Two reasons this is a module rather than a `basicConfig` call at the top of the worker.

**Redaction happens in the log path, not at every call site.** ai/media/live_base.py has
redact_url and calls it carefully, but "remember to redact" is a rule that holds until the
one exception -- a traceback, a repr of a config dict, an f-string somebody added while
debugging at 1am. Redacting in the formatter catches all of those, because the formatter is
the last thing to touch a line before a handler writes it. The rule to rely on is that, not
anybody's discipline.

The formatter is where it happens rather than a filter, and that was not the first design. A
filter runs *before* formatting: it sees record.msg and record.args, but not the line that
gets written. So a format string containing `%(url)s`, and the traceback appended from
exc_info, both went out unexamined while the filter reported success -- caught by testing
against the real credentials rather than against a placeholder.

The specific thing being protected: the Sentinel HLS credentials belong to the organisers,
and the log is the file somebody pastes into a chat window when a camera stops responding.

**Handlers are added once.** A worker that reconfigures logging per camera ends up with
thirty handlers on the root logger and prints every line thirty times, which reads as an
event loop bug and is not one.
"""

import logging
import os
import re
import sys
from typing import Any, Mapping, Optional

DEFAULT_LEVEL = "INFO"

# Console format. The camera id is in the message rather than a column because thirty workers
# write to thirty files, and a line copied out of one needs to say which camera it came from.
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

# The vocabulary, in one place. Written once and shared by every pattern below, because three
# copies of an alternation is three chances to add "passphrase" to two of them.
_SECRET_WORDS = (
    r"password|passwd|passphrase|pwd|\bpw\b|secret|token|api_?key|credential|bearer|dsn"
)

# Same key names ai/config.py masks, and the same reasoning: substring, case-insensitive,
# because an allowlist of known-secret names fails silently the first time a field is added.
_SECRET_KEY = re.compile(_SECRET_WORDS, re.IGNORECASE)

# userinfo in a URL: scheme://user:pass@host. The password is the part worth removing and it
# is the part that ends up in a log the moment a stream fails to open.
_URL_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@")

# A credential in a query string: ...?token=abc&x=1. Signed-URL style, which is how an HLS
# playlist or an object-store link carries its authorisation, and neither of the two patterns
# below would see it -- one only looks at userinfo, the other skips URLs on purpose.
_URL_QUERY_SECRET = re.compile(
    rf"(?i)(?P<lead>[?&][\w.\-]*(?:{_SECRET_WORDS})[\w.\-]*=)(?P<value>[^&\s]+)"
)

# key=value and "key": "value" inside an already-formatted message. Several shapes, because
# one comes from an f-string somebody wrote, one from a dict repr, one from a %(field)s the
# formatter substituted, and one from a library's exception text.
#
# The optional quotes on BOTH sides of the separator are not decoration. Without the one
# before it, `{"password": "x"}` does not match: the regex reaches the closing quote of the
# key where it expects the colon, and a JSON dict is the single most likely way a config
# reaches a log line.
#
# Concatenated rather than an f-string: the value class contains a literal `}`, which an
# f-string reads as its own syntax.
_KV_SECRET = re.compile(
    r"""(?ix)
    (?P<key>[\w.]*(?:"""
    + _SECRET_WORDS
    + r""")[\w.]*)
    (?P<close>['"]?)
    (?P<sep>\s*[=:]\s*)
    (?P<quote>['"]?)
    (?P<value>[^\s,;'"})\]]+)
    (?P=quote)
    """
)

# `Authorization: Bearer <token>` and `Bearer <token>`. Separate because the secret here is
# separated from its keyword by whitespace rather than by = or :, so the pattern above walks
# straight past it -- which it did, on a real token, until the harness said so.
#
# The 12-character floor keeps it from eating prose: "token expired", "bearer required". A
# 12-character English word following one of those two words is rare, and a real credential
# shorter than 12 characters is rarer -- the shortest this project handles is 14.
_BEARER = re.compile(r"(?i)\b(bearer|token)\s+(?P<value>[A-Za-z0-9._\-]{12,})")

REDACTED = "***"


def _kv_replacement(match: "re.Match[str]") -> str:
    """Mask a key=value hit, unless an earlier pass already handled that value.

    Two exceptions, both the same shape: this pass runs last, and it must not undo the more
    precise work of the two before it.

    **A URL value.** `dsn` is a secret-named key whose value is normally an entire connection
    string, so the blunt rule turned

        db dsn: postgresql://trinetra:s3cr3t@localhost:5432/trinetra

    into `db dsn: ***` -- safe, and useless. The line exists to say which database refused the
    connection, and that is exactly what got thrown away. The userinfo pass has already
    removed the credential; what is left is worth keeping.

    **A value that is already redacted.** After the query-string pass,
    `...m3u8?token=***&x=1` still matches this pattern, with `***&x=1` as the value -- so the
    blunt rule replaced the lot and ate `&x=1`. `&` is deliberately not a value terminator
    here: a password containing one would otherwise be half-redacted, and losing a query
    parameter is the cheaper of those two failures.
    """
    value = match.group("value")
    if "://" in value or value.startswith(REDACTED):
        return match.group(0)
    return (
        f"{match.group('key')}{match.group('close')}{match.group('sep')}"
        f"{match.group('quote')}{REDACTED}{match.group('quote')}"
    )


def redact_text(text: str) -> str:
    """Remove userinfo and key=value secrets from an already-formatted line.

    Operates on the formatted string rather than on the arguments, which is the only place
    that catches all of the ways a secret arrives: an f-string, a dict repr, a %(field)s the
    formatter substituted, a `str(exception)` from a library that helpfully included the URL
    it failed to open.

    Order matters. The two URL passes run first so that by the time the key=value pass sees a
    URL-valued secret, the credential inside it is already gone and the host can be kept.

    This is a net, not a guarantee, and the limit is worth being precise about: it recognises
    secrets by the *word next to them*. A password logged as a bare word with no key name and
    no URL around it passes through, and nothing operating on formatted text could catch
    that. So ai/config.py still redacts by key at the source and ai/media still calls
    redact_url; this is the layer that catches what those two miss.
    """
    if not text:
        return text
    text = _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}{REDACTED}@", text)
    text = _URL_QUERY_SECRET.sub(lambda m: f"{m.group('lead')}{REDACTED}", text)
    text = _KV_SECRET.sub(_kv_replacement, text)
    return _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)


class RedactingFormatter(logging.Formatter):
    """Format the record, then redact the result.

    A Formatter rather than a Filter, and that distinction is the whole point of this class.
    A filter runs *before* formatting, so it sees record.msg and record.args but not the
    line that actually gets written -- which means a format string containing `%(url)s`, or a
    traceback appended by exc_info, goes to disk unexamined. That is not hypothetical: it is
    how a real token reached a real line while the filter reported success.

    Redacting last means every byte a handler writes has been through redact_text: the
    message, the substituted extra fields, the exception text and the stack trace.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            formatted = super().format(record)
        except Exception:  # noqa: BLE001 - a bad format string must not kill the process
            # Logging swallowing an exception is bad; logging *raising* one from inside a
            # worker's error path is worse, because the thing being reported is lost.
            return redact_text(
                f"<unformattable log record: msg={record.msg!r} args={record.args!r}>"
            )
        return redact_text(formatted)


class RedactingFilter(logging.Filter):
    """Mask secret-named fields on the record itself, by key.

    Belt to RedactingFormatter's braces, and it covers a different case: a handler somebody
    else attaches -- pytest's caplog, a library that adds its own -- has its own formatter and
    would render `extra={"password": ...}` verbatim. Masking on the record means the value is
    already gone by the time any formatter sees it.

    Does not touch record.msg or record.args. An earlier version did, and clearing args after
    folding them into msg was a subtle way to make two handlers disagree; the formatter makes
    it unnecessary.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in list(getattr(record, "__dict__", {}).items()):
            if key in ("msg", "args", "exc_info", "exc_text", "stack_info"):
                continue
            if _SECRET_KEY.search(key):
                record.__dict__[key] = REDACTED
            elif isinstance(value, str) and "@" in value and "://" in value:
                record.__dict__[key] = redact_text(value)
        return True


def setup_logging(
    config: Optional[Mapping[str, Any]] = None,
    *,
    level: Optional[str] = None,
    stream: Any = None,
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the `ai` logger. Idempotent unless force=True.

    Takes the `logging:` block from a config. level= overrides it, which is how a `-v` flag
    works without rewriting a file.

    Configures the "ai" logger rather than root, and turns propagate off. Root belongs to
    whatever imports this -- pytest, a notebook, Mihir's FastAPI app if the worker ever runs
    in-process -- and reconfiguring root from a library is how a host application loses its
    own log format to an import.

    redact_secrets defaults to True and there is no config value that produces an
    unredacted handler by accident: only an explicit `redact_secrets: false`, which
    config/live.yaml's comment says never to write.
    """
    block = dict(config or {})
    resolved_level = str(level or block.get("level") or DEFAULT_LEVEL).upper()
    redact = bool(block.get("redact_secrets", True))

    logger = logging.getLogger("ai")
    if logger.handlers and not force:
        logger.setLevel(resolved_level)
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(resolved_level)
    logger.propagate = False

    fmt = block.get("format", DEFAULT_FORMAT)
    datefmt = block.get("datefmt", DEFAULT_DATEFMT)

    def make_formatter() -> logging.Formatter:
        # One instance per handler. Sharing one works today, but a handler that later wants a
        # different format would silently change the other's.
        return RedactingFormatter(fmt, datefmt) if redact else logging.Formatter(fmt, datefmt)

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(stream or sys.stderr)
    handlers.append(console)

    target = log_file or block.get("file")
    if target:
        directory = os.path.dirname(str(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # delay=True so a run that never logs does not leave an empty file, and so a
        # permission problem surfaces on the first write with a message rather than at import
        # time with a traceback nobody can place.
        handlers.append(logging.FileHandler(str(target), encoding="utf-8", delay=True))

    for handler in handlers:
        handler.setFormatter(make_formatter())
        if redact:
            handler.addFilter(RedactingFilter())
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """A logger under the `ai` namespace, so it inherits the redacting handlers.

    `get_logger("worker")` gives `ai.worker`. Calling logging.getLogger("worker") directly
    would attach to root instead and bypass the filter -- which is the one mistake in this
    module that has no visible symptom until a password is already in a file.
    """
    return logging.getLogger(name if name.startswith("ai") else f"ai.{name}")


def log_config(logger: logging.Logger, config: Any) -> None:
    """Log a config at startup, redacted, so a run is reproducible from its own log.

    Takes an AppConfig and calls redacted(). Passing a raw dict is possible and is why the
    RedactingFilter exists as a second line: this function would happily format a plain dict
    containing a password, and the filter is what stops it reaching the file.
    """
    import json

    payload = config.redacted() if hasattr(config, "redacted") else config
    path = getattr(config, "path", None)
    logger.info("config %s", path or "<inline>")
    for line in json.dumps(payload, indent=2, default=str, sort_keys=True).splitlines():
        logger.info("  %s", line)
