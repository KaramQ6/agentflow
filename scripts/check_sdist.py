"""Fail if the source distribution contains anything it should not.

0.6.0 was yanked because its sdist bundled a local `.bridgespace/` agent
workspace whose contents included a live PyPI token. The wheel was clean —
only the sdist carried it, and nothing in the release pipeline looked. This is
that missing gate.

The allowlist is fail-closed on purpose: a new top-level entry breaks the build
until someone decides it belongs in a published archive. That is the whole
point — the last leak arrived as a directory nobody had thought about.

Usage:
    python -m build --sdist
    python scripts/check_sdist.py [dist_dir]
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

ALLOWED_TOP_LEVEL = {
    # Packaging metadata
    "PKG-INFO",
    "pyproject.toml",
    "uv.lock",
    # Documentation set
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "SECURITY.md",
    "PUBLIC_API.md",
    "LEVEL_UP.md",
    "CODE_OF_CONDUCT.md",
    "mkdocs.yml",
    "docs",
    # Source and everything needed to verify it
    "src",
    "tests",
    "examples",
    "benchmarks",
    "scripts",
    # Repo config
    ".gitattributes",
    ".gitignore",
    ".github",
}


def check_archive(archive: Path) -> list[str]:
    """Return the top-level entries in *archive* that are not allowlisted."""
    with tarfile.open(archive) as tar:
        names = tar.getnames()

    unexpected = set()
    for name in names:
        parts = name.split("/")
        if len(parts) < 2 or not parts[1]:
            continue  # the version-stamped root directory itself
        if parts[1] not in ALLOWED_TOP_LEVEL:
            unexpected.add(parts[1])
    return sorted(unexpected)


def main(argv: list[str]) -> int:
    dist_dir = Path(argv[1]) if len(argv) > 1 else Path("dist")
    archives = sorted(dist_dir.glob("*.tar.gz"))

    if not archives:
        print(f"error: no sdist found in {dist_dir}/ — run `python -m build --sdist` first")
        return 1

    failed = False
    for archive in archives:
        unexpected = check_archive(archive)
        if unexpected:
            failed = True
            print(f"error: {archive.name} contains unexpected top-level entries:")
            for entry in unexpected:
                print(f"  - {entry}")
            print(
                "\nIf this belongs in a published archive, add it to "
                "ALLOWED_TOP_LEVEL in scripts/check_sdist.py. If it does not, "
                "add it to .gitignore — it will otherwise ship to PyPI."
            )
        else:
            print(f"ok: {archive.name} contains only allowlisted entries")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
