"""Shared pytest fixtures and the one path fix that makes `pytest` work from the repo root.

There is no packaging file in this repository -- no pyproject.toml, no setup.py -- because
the AI lane does not own one (see docs/REPOSITORY.md section 5.1). So `import ai` only works
if the repository root is on sys.path, and pytest does not put it there: with the default
prepend import mode it inserts the *test* file's rootdir, which is tests/, not the parent.
Running `pytest` from the root happens to work when Python adds '' to sys.path, and stops
working the moment anyone runs `pytest tests/test_contracts.py` from another directory or a
CI job sets PYTHONSAFEPATH. Four lines here is cheaper than that bug.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# The twelve filenames Canonical Contracts section 9 locks, in the order the document
# lists them. Hard-coded rather than globbed: a glob would silently pass if a fixture were
# deleted, and the whole point of section 9 is that these exact names exist.
CONTRACT_FIXTURES = (
    "ai_event_high_confidence.json",
    "ai_event_low_confidence.json",
    "ai_event_unreadable.json",
    "ai_event_duplicate.json",
    "ai_event_bad_timestamp.json",
    "ai_event_unknown_camera.json",
    "camera_reconnect.json",
    "scene_discontinuity.json",
    "journey_four_cameras.json",
    "journey_implausible.json",
    "watchlist_match.json",
    "search_response.json",
)


def load_fixture(name: str) -> dict:
    """Read one fixture by filename. Raises if it is missing, rather than skipping.

    A skipped fixture test is indistinguishable from a passing one in CI output, and these
    files are a deliverable another lane is waiting on.
    """
    path = FIXTURE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} is missing from {FIXTURE_DIR}. Canonical Contracts section 9 locks "
            "these twelve filenames; they are not optional and not regenerable from a run."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_events(payload: dict) -> list[dict]:
    """Every EventEnvelope in a fixture, whatever wrapper shape it uses.

    Six of the twelve are a bare envelope; four wrap a list under `events`; one puts a
    single envelope under `matching_event`; search_response.json has none. Callers that
    want to validate "every event in the directory" should not each re-derive that.
    """
    if "events" in payload:
        return list(payload["events"])
    if "matching_event" in payload:
        return [payload["matching_event"]]
    if "schema_version" in payload:
        return [payload]
    return []


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def expectations() -> dict:
    """The shared expected-outcome table. See tests/fixtures/expectations.json."""
    return json.loads((FIXTURE_DIR / "expectations.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixtures() -> dict[str, dict]:
    """All twelve, keyed by filename."""
    return {name: load_fixture(name) for name in CONTRACT_FIXTURES}
