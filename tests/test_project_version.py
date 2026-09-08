"""Project software-version contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

import planning
from planning.version import SOFTWARE_VERSION, software_version


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_XML = ROOT / "package.xml"


def _manifest_version() -> str:
    node = ElementTree.parse(str(PACKAGE_XML)).getroot().find("version")
    assert node is not None and node.text
    return node.text.strip()


@pytest.mark.unit
def test_python_and_setup_versions_come_from_package_manifest():
    expected = _manifest_version()
    assert software_version(PACKAGE_XML) == expected
    assert SOFTWARE_VERSION == expected
    assert planning.__version__ == expected

    setup_result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert setup_result.returncode == 0, setup_result.stderr
    assert setup_result.stdout.strip() == expected


@pytest.mark.unit
def test_version_reader_rejects_missing_or_empty_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest missing"):
        software_version(tmp_path / "missing.xml")

    empty = tmp_path / "package.xml"
    empty.write_text("<package><name>planning</name></package>", encoding="utf-8")
    with pytest.raises(ValueError, match="has no version"):
        software_version(empty)
