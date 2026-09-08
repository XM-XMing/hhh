#!/usr/bin/env python3
"""Build the geometry-only AWAC primitive neighborhood artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from planning.awac.primitive_neighborhood import (
    build_primitive_neighborhood_from_mpl,
    write_primitive_neighborhood_artifact,
)
from planning.common.hashing import file_sha256


DEFAULT_SOURCE = Path("data/motion_primitives/motion_primitives_105.npz")
DEFAULT_OUTPUT = Path("data/motion_primitives/awac_primitive_neighborhood_v1.npz")
DEFAULT_METADATA = Path("data/motion_primitives/awac_primitive_neighborhood_v1.json")
DEFAULT_MPL_CONTRACT_SHA256 = "f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--mpl-contract-sha256", default=DEFAULT_MPL_CONTRACT_SHA256)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("MPL source does not exist: {}".format(source))
    neighborhood = build_primitive_neighborhood_from_mpl(
        source,
        metadata={
            "source_npz_keys": [
                "pos_ref",
                "vel_ref",
                "acc_ref",
                "yaw_ref",
                "cmd_seq",
                "t_ref",
                "t_cmd",
                "lateral_end",
                "vertical_end",
                "terminal_heading_rad",
                "horizontal_index",
                "vertical_index",
            ],
            "trajectory_field": "pos_ref",
            "action_index_semantics": "not_used_for_distance",
        },
    )
    artifact_sha = write_primitive_neighborhood_artifact(
        neighborhood,
        npz_path=Path(args.output),
        metadata_path=Path(args.metadata),
        source_path=str(source),
        source_sha256=file_sha256(source),
        mpl_contract_sha256=str(args.mpl_contract_sha256),
    )
    print("PRIMITIVE_COUNT={}".format(neighborhood.action_count))
    print("ARTIFACT_SHA256={}".format(artifact_sha))
    print("NEAREST_0={}".format(neighborhood.top_k(0, 5).tolist()))
    print(
        "FARTHEST_0={}".format(
            neighborhood.distance_matrix[0].argsort()[::-1][:5].tolist()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
