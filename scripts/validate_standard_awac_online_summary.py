#!/usr/bin/env python3
"""Validate one canonical Standard AWAC online summary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planning.awac.online_runtime import validate_standard_online_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_standard_awac_online_summary.py",
        description="Validate a Standard AWAC online summary against its typed contract.",
    )
    parser.add_argument("--summary", required=True, help="summary.json to validate")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.summary).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Standard AWAC summary does not exist: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_standard_online_summary(payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "summary": str(path),
                "schema_id": validated["schema_id"],
                "online_env_steps": int(validated["online_env_steps"]),
                "online_env_steps_budget": int(
                    validated["online_env_steps_budget"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
