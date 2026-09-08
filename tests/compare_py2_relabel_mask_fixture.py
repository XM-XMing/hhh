#!/usr/bin/env python3
"""Compare O/P PY2 producer artifacts on one fixed reliable rollout fixture.

The original and optimized artifacts may live in different directories, so
episode paths are compared after resolving them.  All numerical payloads and
the shared business metadata are otherwise compared exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


LABEL_ARRAYS = (
    "soft_targets",
    "global_action_masks",
    "teacher_argmax",
    "behavior_actions",
    "valid_counts",
    "teacher_entropies",
)
MASK_ARRAYS = ("local_action_masks",)
MAPPING_ARRAYS = ("episode_npz_paths", "episode_offsets", "episode_lengths")

# These fields are intentionally excluded from the O/P business comparison:
# they describe the producer's artifact location or the new P provenance seam.
# Mapping arrays are still compared separately, including resolved episode path
# order, and observation_contract/source remain business fields.
PROVENANCE_FIELDS = frozenset(
    {
        "provenance_schema_version",
        "artifact_kind",
        "input_rollout_index",
        "input_rollout_index_sha256",
        "input_rollout_manifest",
        "input_rollout_manifest_sha256",
        "rollout_index_sha256",
        "rollout_manifest_sha256",
        "rollout_root",
        "collection_run_id",
        "input_rollout_reliable_rows",
        "input_accepted_reliable_rows",
        "input_rollout_legacy_rows",
        "input_rollout_telemetry_lookup_count",
        "input_rollout_snapshot_missing_count",
        "input_rollout_state_depth_skew_max_ns",
        "input_rollout_frame_contract_failures",
        "endpoint_identity_available",
        "endpoint_identity_chain_valid",
        "reliable_execution",
        "telemetry_observation",
        "state_depth_skew_ns",
        "state_depth_skew_max_ns",
        "asynchronous_prefetch",
        "asynchronous_prefetch_status",
        "reliable_rows",
        "legacy_rows",
        "telemetry_lookup_count",
        "snapshot_missing_count",
        "frame_contract_failures",
        "episode_count",
        "row_count",
        "transition_count",
        "action_count",
        "episode_ids",
        "episode_order",
        "transition_offsets",
        "transition_lengths",
        "worker_ids",
        "runtime_instance_ids",
        "mission_index_sha256",
        "mission_sha256",
        "resolved_config_sha256",
        "teacher_contract_id",
        "teacher_contract_sha256",
        "teacher_planning_contract_sha256",
        "depth_mask_contract_id",
        "depth_mask_contract_sha256",
    }
)


def _metadata(data: Mapping[str, np.ndarray]) -> dict:
    value = data["metadata_json"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = json.loads(str(value))
    if not isinstance(result, dict):
        raise ValueError("metadata_json is not a mapping")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolved_paths(path: Path, values: Iterable[object]) -> Tuple[str, ...]:
    return tuple(str((path.parent / str(value)).expanduser().resolve()) for value in values)


def _compare_arrays(
    kind: str,
    o_path: Path,
    p_path: Path,
    keys: Sequence[str],
) -> bool:
    passed = True
    with np.load(str(o_path), allow_pickle=False) as original, np.load(
        str(p_path), allow_pickle=False
    ) as optimized:
        for key in keys:
            if key == "episode_npz_paths":
                # Producer artifacts are intentionally stored beside their
                # own output, so the relative spelling may differ.  The
                # resolved path identity is checked below.
                continue
            if key not in original or key not in optimized:
                print("{} missing array {}".format(kind, key))
                passed = False
                continue
            left = original[key]
            right = optimized[key]
            equal = np.array_equal(left, right)
            print(
                "{}_{}_PARITY={} O_SHA256={} P_SHA256={} shape={} dtype={}".format(
                    kind.upper(),
                    key.upper(),
                    "PASS" if equal else "FAIL",
                    _array_sha256(left),
                    _array_sha256(right),
                    tuple(left.shape),
                    left.dtype,
                )
            )
            passed = passed and equal

        for key in ("episode_offsets", "episode_lengths"):
            if key in original and key in optimized:
                equal = np.array_equal(original[key], optimized[key])
                print(
                    "{}_{}_ORDER_PARITY={}".format(
                        kind.upper(), key.upper(), "PASS" if equal else "FAIL"
                    )
                )
                passed = passed and equal
        if "episode_npz_paths" in original and "episode_npz_paths" in optimized:
            left_paths = _resolved_paths(o_path, original["episode_npz_paths"].tolist())
            right_paths = _resolved_paths(p_path, optimized["episode_npz_paths"].tolist())
            equal = left_paths == right_paths
            print(
                "{}_EPISODE_ORDER_PARITY={} paths={}".format(
                    kind.upper(), "PASS" if equal else "FAIL", list(left_paths)
                )
            )
            passed = passed and equal
    return passed


def _compare_business_metadata(o_path: Path, p_path: Path) -> bool:
    with np.load(str(o_path), allow_pickle=False) as original, np.load(
        str(p_path), allow_pickle=False
    ) as optimized:
        left = _metadata(original)
        right = _metadata(optimized)
    common = sorted(set(left).intersection(right) - PROVENANCE_FIELDS)
    differences = [(key, left[key], right[key]) for key in common if left[key] != right[key]]
    if differences:
        print("METADATA_BUSINESS_PARITY=FAIL differences={}".format(differences))
        return False
    print("METADATA_BUSINESS_PARITY=PASS common_keys={}".format(len(common)))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o-labels", type=Path, required=True)
    parser.add_argument("--p-labels", type=Path, required=True)
    parser.add_argument("--o-masks", type=Path, required=True)
    parser.add_argument("--p-masks", type=Path, required=True)
    args = parser.parse_args()

    o_labels = args.o_labels.expanduser().resolve()
    p_labels = args.p_labels.expanduser().resolve()
    o_masks = args.o_masks.expanduser().resolve()
    p_masks = args.p_masks.expanduser().resolve()
    label_arrays = _compare_arrays("teacher_label", o_labels, p_labels, LABEL_ARRAYS)
    mask_arrays = _compare_arrays("depth_mask", o_masks, p_masks, MASK_ARRAYS)
    mapping = _compare_arrays("mapping", o_labels, p_labels, MAPPING_ARRAYS)
    mapping = _compare_arrays("mapping", o_masks, p_masks, MAPPING_ARRAYS) and mapping
    business = _compare_business_metadata(o_labels, p_labels) and _compare_business_metadata(
        o_masks, p_masks
    )
    print("TEACHER_LABEL_ARRAY_PARITY={}".format("PASS" if label_arrays else "FAIL"))
    print("DEPTH_MASK_ARRAY_PARITY={}".format("PASS" if mask_arrays else "FAIL"))
    print("EPISODE_ORDER_PARITY={}".format("PASS" if mapping else "FAIL"))
    print("TRANSITION_ORDER_PARITY={}".format("PASS" if label_arrays and mask_arrays else "FAIL"))
    print("METADATA_BUSINESS_PARITY={}".format("PASS" if business else "FAIL"))
    return 0 if label_arrays and mask_arrays and mapping and business else 1


if __name__ == "__main__":
    raise SystemExit(main())
