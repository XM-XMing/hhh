#!/usr/bin/env python3
"""Canonical flag-style entry point for parallel collection.

The worker default is resolved by ``planning.teacher.parallel_collection``
from ``config/pre_bc.yaml::collection.workers``.
"""

from __future__ import annotations

from planning.teacher.parallel_collection import main

def cli_main(argv=None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli_main())
