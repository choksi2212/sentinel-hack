#!/usr/bin/env python3
"""Assert no clip_id appears in two splits. Exits non-zero on violation.

CLAUDE.md SS5: indian_road splits by CLIP ID, never by frame. This checks any
markdown doc shaped like datasets/trinetra-hard/CLIP_RESERVATION.md: one or
more '## SPLIT_NAME' sections each followed by a '| clip_id | ... |' table.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "datasets" / "trinetra-hard" / "CLIP_RESERVATION.md"

HEADING_RE = re.compile(r"^## ([A-Z_][A-Z0-9_]*)")


def parse_splits(text: str) -> dict[str, set[str]]:
    """Return {split_name: {clip_id, ...}} from '## NAME' + '| clip_id | ... |' tables."""
    splits: dict[str, set[str]] = {}
    current = None
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            current = m.group(1)
            splits.setdefault(current, set())
            continue
        if current and line.startswith("|"):
            cell = line.split("|")[1].strip()
            if cell and cell != "clip_id" and not set(cell) <= {"-"}:
                splits[current].add(cell)
    return splits


def find_leaks(splits: dict[str, set[str]]) -> dict[str, list[str]]:
    """Return {clip_id: [split names it appears in]} for clip_ids in >1 split."""
    membership: dict[str, list[str]] = {}
    for split_name, clip_ids in splits.items():
        for cid in clip_ids:
            membership.setdefault(cid, []).append(split_name)
    return {cid: names for cid, names in membership.items() if len(names) > 1}


def main(argv: list[str]) -> int:
    target = Path(argv[0]) if argv else DEFAULT_TARGET
    if not target.exists():
        print(f"BLOCKED: {target} not found", file=sys.stderr)
        return 1
    splits = parse_splits(target.read_text(encoding="utf-8"))
    leaks = find_leaks(splits)
    if leaks:
        print(f"LEAKAGE in {target}:", file=sys.stderr)
        for cid, names in sorted(leaks.items()):
            print(f"  - {cid} in splits: {', '.join(names)}", file=sys.stderr)
        return 1
    total = sum(len(v) for v in splits.values())
    print(f"OK: no leakage across {len(splits)} splits, {total} clip_ids checked.")
    return 0


def demo():
    """Self-check: assert-based, no framework."""
    clean = (
        "## RESERVED\n"
        "| clip_id | frames |\n"
        "|---|---|\n"
        "| aaa | 60 |\n"
        "| bbb | 60 |\n"
        "## TRAIN_SAFE\n"
        "| clip_id | frames |\n"
        "|---|---|\n"
        "| ccc | 60 |\n"
    )
    splits = parse_splits(clean)
    assert splits == {"RESERVED": {"aaa", "bbb"}, "TRAIN_SAFE": {"ccc"}}
    assert find_leaks(splits) == {}

    broken = clean.replace("| ccc | 60 |", "| aaa | 60 |")  # aaa now in both splits
    broken_splits = parse_splits(broken)
    leaks = find_leaks(broken_splits)
    assert list(leaks.keys()) == ["aaa"], leaks
    assert set(leaks["aaa"]) == {"RESERVED", "TRAIN_SAFE"}
    print("demo: all assertions passed (clean fixture clean, broken fixture caught)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main([a for a in sys.argv[1:] if a != "--demo"]))
