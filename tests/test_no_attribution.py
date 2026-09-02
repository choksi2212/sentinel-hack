"""Repo-wide guard: the tracked tree carries no authorship or tool-provenance
markers, and no live access token.

Two independent promises are pinned here, worst first:

  1. Not one tracked file names a model vendor or product, carries the
     co-authorship git trailer, or wears a "made by a tool" provenance
     phrase. This is the whole-repo version of the per-package guards that
     live beside the code they cover (e.g. the tracking suite has its own);
     this one leaves nothing out.

  2. No file carries the shape of a HuggingFace access token, and -- when a
     denylist of exact secret strings is supplied out of band -- no file
     carries one of those literals either.

Design notes that matter:

* The marker list below is assembled from split string fragments, never
  written whole. So this guard file is itself clean under the same blunt
  substring scan it enforces -- it does not have to exempt itself, and the
  tree contains zero whole occurrences of any listed word. Every per-package
  guard follows the same rule, so nothing needs an exemption.

* Enumeration is `git ls-files`, i.e. exactly what a clone or a diff would
  carry -- ignored scratch, caches and local-only files are out of scope by
  construction, and there is no hand-maintained skip list to drift.

* A scan that silently walks nothing would pass every promise vacuously, so
  a floor on the file count is asserted first. A guard that can quietly turn
  into a no-op is worse than none.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from conftest import REPO_ROOT


# --------------------------------------------------------------------------
# Markers, assembled from fragments so no whole word is ever written to disk.
# Compared case-insensitively as plain substrings against file text.
# --------------------------------------------------------------------------

_MARKERS = (
    "cla" "ude",
    "anthro" "pic",
    "chat" "gpt",
    "open" "ai",
    "copi" "lot",
    "co-auth" "ored-by",
    "ai-" "generated",
    "generated " "with",
    "as an " "ai",
)

# The shape of a HuggingFace access token: the hf_ prefix and a long run of
# token characters. Written as a character class, so this pattern is not
# itself a token and does not trip the scan above.
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{30,}")

# Floor on tracked files. The tree is well over a hundred; 50 still catches
# an enumeration that silently returned nothing.
_MIN_TRACKED_FILES = 50


# --------------------------------------------------------------------------
# Enumeration + text loading, cached across the tests in this module.
# --------------------------------------------------------------------------

_loaded: "list[tuple[str, str]] | None" = None


def _tracked_paths() -> "list[str]":
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=True,
    )
    return [chunk.decode("utf-8", "surrogateescape")
            for chunk in result.stdout.split(b"\0") if chunk]


def _tracked_text() -> "list[tuple[str, str]]":
    """(relpath, text) for every tracked file that decodes as UTF-8.

    Binary blobs (fixture images and the like) do not carry text markers and
    are skipped on a decode failure rather than guessed at.
    """
    global _loaded
    if _loaded is None:
        items: "list[tuple[str, str]]" = []
        for rel in _tracked_paths():
            try:
                text = (REPO_ROOT / rel).read_bytes().decode("utf-8")
            except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
                continue
            items.append((rel, text))
        _loaded = items
    return _loaded


# ==========================================================================
# 0 -- the scan must actually cover the tree (no silent no-op)
# ==========================================================================

def test_the_scan_covers_the_whole_tracked_tree():
    tracked = _tracked_text()
    assert len(tracked) >= _MIN_TRACKED_FILES, (
        f"only {len(tracked)} tracked text files enumerated; the scan may be "
        f"walking nothing and passing every other check vacuously"
    )


# ==========================================================================
# 1 -- no authorship or tool-provenance markers, anywhere
# ==========================================================================

def test_no_authorship_or_tool_provenance_markers_in_any_tracked_file():
    hits: "list[str]" = []
    for rel, text in _tracked_text():
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for marker in _MARKERS:
                if marker in lowered:
                    hits.append(f"{rel}:{lineno}: {marker!r}")
    assert not hits, "authorship/provenance markers found:\n" + "\n".join(hits[:50])


# ==========================================================================
# 2 -- no live access-token shape, and no denylisted secret literals
# ==========================================================================

def test_no_huggingface_access_token_shape_in_any_tracked_file():
    hits: "list[str]" = []
    for rel, text in _tracked_text():
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _HF_TOKEN.search(line):
                hits.append(f"{rel}:{lineno}")
    assert not hits, "access-token shape found:\n" + "\n".join(hits[:50])


def _denylist_from_env() -> "list[str]":
    """Exact secret strings to hunt for, supplied out of band.

    The real values are never committed; export TRINETRA_SECRET_DENYLIST
    (newline- or comma-separated) locally to turn this into an exact-literal
    sweep. Unset -> the check is skipped rather than run against nothing.
    """
    raw = os.environ.get("TRINETRA_SECRET_DENYLIST", "")
    parts: "list[str]" = []
    for line in raw.splitlines():
        parts.extend(piece.strip() for piece in line.split(","))
    return [p for p in parts if p]


def test_no_denylisted_secret_literals_in_any_tracked_file():
    denylist = _denylist_from_env()
    if not denylist:
        pytest.skip(
            "set TRINETRA_SECRET_DENYLIST (comma/newline separated) to sweep "
            "the tree for exact secret literals"
        )
    hits: "list[str]" = []
    for rel, text in _tracked_text():
        lowered_text = text.lower()
        for secret in denylist:
            if secret.lower() in lowered_text:
                hits.append(f"{rel}: matched a denylisted literal")
    assert not hits, "denylisted secret literal found:\n" + "\n".join(hits[:50])
