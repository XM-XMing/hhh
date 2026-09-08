#!/usr/bin/env python3
"""Validate the frozen runtime, map, and MPL identities before collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from planning.runtime.identity import validate_pre_collection_runtime


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the formal Pre-Collection runtime identity."
    )
    parser.add_argument("--unity-bin", required=True)
    parser.add_argument("--unity-data", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--voxel-cache", required=True)
    parser.add_argument("--mpl-npz", required=True)
    parser.add_argument("--mpl-json", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--expected-observation-contract", required=True)
    parser.add_argument("--manifest", default="")
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = validate_pre_collection_runtime(
            player=Path(args.unity_bin),
            unity_data=Path(args.unity_data),
            bridge=Path(args.bridge),
            point_cloud=Path(args.point_cloud),
            voxel_cache=Path(args.voxel_cache),
            mpl_npz=Path(args.mpl_npz),
            mpl_json=Path(args.mpl_json),
            max_steps=int(args.max_steps),
            expected_observation_contract=args.expected_observation_contract,
            manifest_path=Path(args.manifest) if str(args.manifest).strip() else None,
        )
    except Exception as error:
        print("RUNTIME_VALIDATION=FAIL")
        print("ERROR={}".format(error), file=sys.stderr)
        return 1
    print("RUNTIME_VALIDATION=PASS")
    for key in (
        "runtime_manifest",
        "runtime_manifest_sha256",
        "unity_player_sha256",
        "assembly_sha256",
        "bridge_sha256",
        "point_cloud_sha256",
        "voxel_cache_sha256",
        "mpl_npz_sha256",
        "mpl_json_sha256",
        "mpl_contract_sha256",
        "task_contract_sha256",
        "max_steps",
        "observation_contract",
    ):
        print("{}={}".format(key.upper(), result[key]))
    print("NATIVE_GATE=PASS")
    print("MPL_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
