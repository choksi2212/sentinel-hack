#!/usr/bin/env python3
"""Walk a dataset dir under datasets/raw/<name> and emit
datasets/manifests/<name>.sha256 with one 'sha256  size  relpath' line per file.

Usage: py -3.11 scripts/freeze_manifest.py <name> [<name> ...]
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
MANIFESTS = ROOT / "datasets" / "manifests"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(dataset_dir: Path) -> list[str]:
    """Return sorted 'sha256  size  relpath' lines for every file under dataset_dir."""
    lines = []
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(dataset_dir).as_posix()
            lines.append(f"{sha256_of(path)}  {path.stat().st_size}  {rel}")
    return sorted(lines, key=lambda l: l.split("  ", 2)[2])


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: freeze_manifest.py <dataset_name> [...]", file=sys.stderr)
        return 1
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for name in argv:
        dataset_dir = RAW / name
        if not dataset_dir.is_dir():
            print(f"BLOCKED: {dataset_dir} not found", file=sys.stderr)
            return 1
        lines = freeze(dataset_dir)
        out = MANIFESTS / f"{name}.sha256"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{name}: {len(lines)} files -> {out}")
    return 0


def demo():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "toy"
        (d / "sub").mkdir(parents=True)
        (d / "a.txt").write_bytes(b"hello")
        (d / "sub" / "b.txt").write_bytes(b"world")
        lines = freeze(d)
        assert len(lines) == 2
        assert lines[0].endswith("a.txt")
        assert lines[1].endswith("sub/b.txt")
        h = hashlib.sha256(b"hello").hexdigest()
        assert lines[0].startswith(f"{h}  5  ")
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main(sys.argv[1:]))
