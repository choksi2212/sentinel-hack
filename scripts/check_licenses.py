#!/usr/bin/env python3
"""Fail if a manifest in datasets/manifests/ references a dataset with no
verified row in datasets/LICENSES.md (CLAUDE.md SS5: no row -> excluded)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSES = ROOT / "datasets" / "LICENSES.md"
MANIFESTS = ROOT / "datasets" / "manifests"


def verified_names(licenses_text: str) -> set[str]:
    """Names listed in the 'Verified' table (first '|'-delimited column)."""
    names = set()
    in_verified = False
    for line in licenses_text.splitlines():
        if line.startswith("## Verified"):
            in_verified = True
            continue
        if line.startswith("## ") and in_verified:
            break
        if in_verified and line.startswith("|"):
            cell = line.split("|")[1].strip()
            if cell and cell != "name" and not set(cell) <= {"-"}:
                names.add(cell)
    return names


def check(manifests_dir: Path, licenses_text: str) -> list[str]:
    """Return list of manifest dataset names with no verified license row."""
    allowed = verified_names(licenses_text)
    if not manifests_dir.is_dir():
        return []
    violations = []
    for manifest in sorted(manifests_dir.glob("*.sha256")):
        name = manifest.stem
        if name not in allowed:
            violations.append(name)
    return violations


def main() -> int:
    if not LICENSES.exists():
        print(f"BLOCKED: {LICENSES} not found", file=sys.stderr)
        return 1
    violations = check(MANIFESTS, LICENSES.read_text(encoding="utf-8"))
    if violations:
        print("BLOCKED: manifests with no verified LICENSES.md row:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("OK: every manifest has a verified license row.")
    return 0


def demo():
    """Self-check: assert-based, no framework."""
    good_licenses = (
        "## Verified\n"
        "| name | source |\n"
        "|---|---|\n"
        "| indian_road | x |\n"
        "| justjuu_plates | x |\n"
        "## Flagged\n"
        "| kedarsai_plates | no evidence |\n"
    )
    assert verified_names(good_licenses) == {"indian_road", "justjuu_plates"}

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        manifests_dir = Path(td)
        (manifests_dir / "indian_road.sha256").write_text("a b c\n")
        (manifests_dir / "kedarsai_plates.sha256").write_text("a b c\n")
        violations = check(manifests_dir, good_licenses)
        assert violations == ["kedarsai_plates"], violations

        (manifests_dir / "kedarsai_plates.sha256").unlink()
        assert check(manifests_dir, good_licenses) == []
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
