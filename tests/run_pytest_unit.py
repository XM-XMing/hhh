#!/usr/bin/env python3
"""Run CPU-only pytest contracts and emit catkin-compatible JUnit XML."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_pytest_unit.py JUNIT_XML")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "python"))
    result = Path(sys.argv[1]).expanduser().resolve()
    result.parent.mkdir(parents=True, exist_ok=True)

    import pytest

    return int(
        pytest.main(
            [
                "-q",
                str(project_root / "tests"),
                "-m",
                "unit",
                "--junitxml={}".format(result),
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
