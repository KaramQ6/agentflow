"""The sdist allowlist must actually reject a stray directory.

0.6.0 shipped a local agent workspace containing a live PyPI token inside its
source distribution. scripts/check_sdist.py is the gate added afterwards, so it
needs a test proving it fails on exactly that shape of archive.
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_CHECKER = Path(__file__).resolve().parent.parent / "scripts" / "check_sdist.py"
_spec = importlib.util.spec_from_file_location("check_sdist", _CHECKER)
assert _spec is not None and _spec.loader is not None
check_sdist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_sdist)


def _make_sdist(tmp_path: Path, entries: list[str]) -> Path:
    archive = tmp_path / "agentflowkit-9.9.9.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for entry in entries:
            info = tarfile.TarInfo(f"agentflowkit-9.9.9/{entry}")
            payload = b"x"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return archive


def test_a_clean_archive_passes(tmp_path):
    archive = _make_sdist(
        tmp_path, ["pyproject.toml", "README.md", "src/agentflow/__init__.py"]
    )
    assert check_sdist.check_archive(archive) == []


def test_the_leaked_workspace_is_rejected(tmp_path):
    """The exact shape that got 0.6.0 yanked."""
    archive = _make_sdist(
        tmp_path,
        ["pyproject.toml", ".bridgespace/swarms/board.md", ".bridgespace/token.txt"],
    )
    assert check_sdist.check_archive(archive) == [".bridgespace"]


def test_unknown_entries_are_reported_sorted_and_deduplicated(tmp_path):
    archive = _make_sdist(
        tmp_path, ["zeta_notes.md", ".venv/pyvenv.cfg", ".venv/lib/x.py", "alpha.tmp"]
    )
    assert check_sdist.check_archive(archive) == [".venv", "alpha.tmp", "zeta_notes.md"]


def test_main_exits_non_zero_on_a_dirty_archive(tmp_path, capsys):
    _make_sdist(tmp_path, ["pyproject.toml", ".bridgespace/token.txt"])

    exit_code = check_sdist.main(["check_sdist.py", str(tmp_path)])

    assert exit_code == 1
    assert ".bridgespace" in capsys.readouterr().out


def test_main_exits_zero_on_a_clean_archive(tmp_path):
    _make_sdist(tmp_path, ["pyproject.toml", "src/agentflow/__init__.py"])
    assert check_sdist.main(["check_sdist.py", str(tmp_path)]) == 0


def test_main_fails_loudly_when_there_is_no_sdist(tmp_path, capsys):
    """A missing archive must not silently pass as 'nothing wrong found'."""
    exit_code = check_sdist.main(["check_sdist.py", str(tmp_path)])

    assert exit_code == 1
    assert "no sdist found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "required",
    ["src", "pyproject.toml", "README.md", "LICENSE", "tests", "scripts"],
)
def test_allowlist_covers_what_the_repo_actually_ships(required):
    assert required in check_sdist.ALLOWED_TOP_LEVEL
