"""Stable package and workspace path resolution for source and catkin runs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

def planning_package_root() -> Path:
    """Return the planning package root without depending on the caller's cwd."""
    try:
        import rospkg  # type: ignore

        root = Path(rospkg.RosPack().get_path("planning")).resolve()
        if (root / "package.xml").is_file():
            return root
    except Exception:
        pass

    module_path = Path(__file__).resolve()
    candidates = (module_path.parents[2], Path.cwd(), Path.cwd() / "planning")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "package.xml").is_file():
            return candidate
    raise RuntimeError("cannot locate the planning package root")

def planning_workspace_root() -> Path:
    """Return the catkin workspace containing the planning package."""
    package_root = planning_package_root()
    if package_root.parent.name == "src":
        return package_root.parent.parent
    for parent in package_root.parents:
        if (parent / "src" / "planning" / "package.xml").is_file():
            return parent
    return package_root

def resolve_package_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve() if path.is_absolute() else (planning_package_root() / path).resolve()
