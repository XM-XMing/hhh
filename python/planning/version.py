"""Resolve the project software version from the canonical ROS manifest.

``package.xml`` is the single authored version source for this catkin package.
Source checkouts can reach it relative to this module, while installed/devel
spaces expose it below a prefix in ``CMAKE_PREFIX_PATH``.  Distribution
metadata is retained as a final installed-package fallback because
``setup.py`` is itself generated from the same manifest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree


def _manifest_candidates() -> Iterable[Path]:
    """Yield package manifests in deterministic precedence order."""

    # Source checkout: planning/python/planning/version.py -> planning/package.xml.
    yield Path(__file__).resolve().parents[2] / "package.xml"
    for raw_prefix in os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep):
        prefix = raw_prefix.strip()
        if prefix:
            yield Path(prefix).expanduser() / "share" / "planning" / "package.xml"


def _version_from_manifest(path: Path) -> str:
    root = ElementTree.parse(str(path)).getroot()
    node = root.find("version")
    value = "" if node is None or node.text is None else node.text.strip()
    if not value:
        raise ValueError("planning package manifest has no version: {}".format(path))
    return value


def software_version(package_xml: Optional[Path] = None) -> str:
    """Return the version authored in ``package.xml``.

    ``package_xml`` exists for focused contract tests and tooling. Normal
    callers should omit it so source, devel and install layouts are resolved
    automatically.
    """

    if package_xml is not None:
        resolved = Path(package_xml).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                "planning package manifest missing: {}".format(resolved)
            )
        return _version_from_manifest(resolved)

    seen = set()
    for candidate in _manifest_candidates():
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return _version_from_manifest(resolved)

    # Wheels and conventional setuptools installs carry distribution metadata
    # whose version was produced by catkin_pkg from this package.xml.
    try:
        try:
            from importlib.metadata import version
        except ImportError:  # pragma: no cover - Python < 3.8 compatibility
            from importlib_metadata import version
        return str(version("planning"))
    except Exception as exc:
        raise RuntimeError(
            "cannot resolve planning software version from package.xml or "
            "installed distribution metadata"
        ) from exc


SOFTWARE_VERSION = software_version()


__all__ = ["SOFTWARE_VERSION", "software_version"]
