#!/usr/bin/env python3
"""Diagnostic-only wrapper for :mod:`planning.diagnostics.base_system`."""

from planning.diagnostics.base_system import main

if __name__ == "__main__":
    raise SystemExit(main())
