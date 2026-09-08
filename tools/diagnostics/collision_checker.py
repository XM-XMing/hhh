#!/usr/bin/env python3
"""Diagnostic-only wrapper for :mod:`planning.safety.collision_checker`."""

from planning.safety.collision_checker import main

if __name__ == "__main__":
    raise SystemExit(main())
