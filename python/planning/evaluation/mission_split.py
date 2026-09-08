"""Mission-level data isolation contract for AWAC training and evaluation.

The split is built once from a complete mission index plus independently
chosen development and final-holdout indexes.  AWAC training then consumes the
generated train index and this manifest instead of trying to infer isolation
from unrelated file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from planning.contracts.collection import canonical_sha256
from planning.data.rollout import read_csv, write_csv_atomic
from planning.mission.spec import mission_id_from_row
from planning.common import file_sha256, write_json_atomic


AWAC_MISSION_SPLIT_CONTRACT_ID = "awac_train_dev_final_disjoint"
AWAC_FINAL_HOLDOUT_SELECTION_CONTRACT_ID = "awac_final_holdout_selection"
AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION_CONTRACT_ID = (
    "awac_evaluation_holdout_pair_selection"
)
AWAC_DEV_SET_CONTRACT_ID = "awac_dev_set_phase0_v1"
AWAC_CALIBRATION_SPLIT_CONTRACT_ID = "awac_calibration_episode_split_v1"
_ARTIFACT_KEYS = ("source", "train", "dev", "final_holdout")
_INTERSECTION_KEYS = ("train_dev", "train_final_holdout", "dev_final_holdout")


def _ordered_mission_sha256(mission_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(value) for value in mission_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _relative_artifact_path(path: Path, manifest_path: Path) -> str:
    return os.path.relpath(
        str(Path(path).expanduser().resolve()),
        str(Path(manifest_path).expanduser().resolve().parent),
    )


def _mission_ids(rows: Iterable[Dict], *, label: str) -> List[str]:
    mission_ids: List[str] = []
    seen: Set[str] = set()
    duplicates: List[str] = []
    for row in rows:
        try:
            mission_id = str(mission_id_from_row(row)).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("{} index contains a row without a valid mission identity".format(label)) from exc
        if not mission_id:
            raise ValueError("{} index contains an empty mission_id".format(label))
        if mission_id in seen:
            duplicates.append(mission_id)
        seen.add(mission_id)
        mission_ids.append(mission_id)
    if duplicates:
        preview = ",".join(sorted(set(duplicates))[:5])
        raise ValueError("{} index contains duplicate mission_id values: {}".format(label, preview))
    if not mission_ids:
        raise ValueError("{} index is empty".format(label))
    return mission_ids


def _read_index(path: Path, *, label: str) -> Tuple[List[Dict], List[str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} index does not exist: {}".format(label, resolved))
    rows = read_csv(resolved)
    return rows, _mission_ids(rows, label=label)


def _read_exclusion_index(path: Path, *, label: str) -> Tuple[List[Dict], List[str]]:
    """Read an index used only as a mission exclusion set.

    BC rollout indexes can legitimately contain more than one episode for a
    mission, so unlike source/dev/final artifacts this helper permits repeated
    identities while still rejecting missing identities and empty indexes.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} index does not exist: {}".format(label, resolved))
    rows = read_csv(resolved)
    mission_ids: List[str] = []
    for row in rows:
        try:
            mission_id = str(mission_id_from_row(row)).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "{} index contains a row without a valid mission identity".format(label)
            ) from exc
        if not mission_id:
            raise ValueError("{} index contains an empty mission_id".format(label))
        mission_ids.append(mission_id)
    if not mission_ids:
        raise ValueError("{} index is empty".format(label))
    return rows, mission_ids


def _artifact_metadata(
    path: Path,
    manifest_path: Path,
    rows: Sequence[Dict],
    mission_ids: Sequence[str],
) -> Dict:
    return {
        "path": _relative_artifact_path(path, manifest_path),
        "file_sha256": file_sha256(Path(path).expanduser().resolve()),
        "row_count": len(rows),
        "mission_count": len(mission_ids),
        "ordered_mission_ids_sha256": _ordered_mission_sha256(mission_ids),
    }


def build_awac_final_holdout(
    source_index_path: Path,
    bc_train_index_path: Path,
    dev_index_path: Path,
    final_holdout_index_path: Path,
    selection_path: Path,
    *,
    count: int,
    seed: int,
    overwrite: bool = False,
) -> Dict:
    """Select a deterministic final holdout unseen by BC and AWAC development.

    Candidate membership is ranked by SHA256 of ``seed`` and ``mission_id`` so
    it is independent of Python or NumPy RNG implementations.  Selected rows
    are written in their original source order, making the evaluation order
    stable and directly auditable.
    """

    if int(count) <= 0:
        raise ValueError("final holdout count must be positive")
    if int(seed) < 0:
        raise ValueError("final holdout seed must be non-negative")

    source_path = Path(source_index_path).expanduser().resolve()
    bc_path = Path(bc_train_index_path).expanduser().resolve()
    dev_path = Path(dev_index_path).expanduser().resolve()
    final_path = Path(final_holdout_index_path).expanduser().resolve()
    selection = Path(selection_path).expanduser().resolve()
    input_paths = {source_path, bc_path, dev_path}
    if final_path == selection or final_path in input_paths or selection in input_paths:
        raise ValueError("final holdout outputs must not overwrite an input or each other")
    for output in (final_path, selection):
        if output.exists() and not bool(overwrite):
            raise FileExistsError("output exists; pass --overwrite: {}".format(output))

    source_rows, source_order = _read_index(source_path, label="source")
    bc_rows, bc_order = _read_exclusion_index(bc_path, label="BC train")
    dev_rows, dev_order = _read_index(dev_path, label="dev")
    source_ids = set(source_order)
    bc_ids = set(bc_order)
    dev_ids = set(dev_order)
    bc_missing = sorted(bc_ids.difference(source_ids))
    dev_missing = sorted(dev_ids.difference(source_ids))
    if bc_missing or dev_missing:
        details = {}
        if bc_missing:
            details["bc_train_not_in_source"] = bc_missing[:5]
        if dev_missing:
            details["dev_not_in_source"] = dev_missing[:5]
        raise ValueError("exclusion indexes are not subsets of source: {}".format(details))

    excluded_ids = bc_ids.union(dev_ids)
    candidate_order = [
        mission_id for mission_id in source_order if mission_id not in excluded_ids
    ]
    candidate_count = len(candidate_order)
    if int(count) > candidate_count:
        raise ValueError(
            "requested {} final missions but only {} candidates remain after BC/dev exclusion".format(
                int(count), candidate_count
            )
        )

    def selection_rank(mission_id: str) -> str:
        token = "{}\0{}".format(int(seed), mission_id).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    selected_ids = set(
        sorted(candidate_order, key=lambda mission_id: (selection_rank(mission_id), mission_id))[
            : int(count)
        ]
    )
    final_rows = [
        row for row, mission_id in zip(source_rows, source_order)
        if mission_id in selected_ids
    ]
    final_order = [
        mission_id for mission_id in source_order if mission_id in selected_ids
    ]
    if len(final_order) != int(count):
        raise AssertionError("deterministic final holdout selection returned the wrong count")
    if selected_ids.intersection(excluded_ids):
        raise AssertionError("final holdout selection contains a BC train or dev mission")

    write_csv_atomic(final_path, final_rows, fieldnames=list(source_rows[0].keys()))
    persisted_rows, persisted_order = _read_index(final_path, label="final holdout")
    if persisted_order != final_order:
        raise RuntimeError("persisted final holdout changed mission ordering")

    def input_metadata(path: Path, rows: Sequence[Dict], order: Sequence[str]) -> Dict:
        return {
            "path": _relative_artifact_path(path, selection),
            "file_sha256": file_sha256(path),
            "row_count": len(rows),
            "unique_mission_count": len(set(order)),
            "ordered_mission_ids_sha256": _ordered_mission_sha256(order),
        }

    payload = {
        "contract_id": AWAC_FINAL_HOLDOUT_SELECTION_CONTRACT_ID,
        "source": input_metadata(source_path, source_rows, source_order),
        "bc_train": input_metadata(bc_path, bc_rows, bc_order),
        "dev": input_metadata(dev_path, dev_rows, dev_order),
        "final_holdout": _artifact_metadata(
            final_path, selection, persisted_rows, persisted_order
        ),
        "seed": int(seed),
        "requested_count": int(count),
        "candidate_count": candidate_count,
        "candidate_ordered_mission_ids_sha256": _ordered_mission_sha256(candidate_order),
        "selection_method": "sha256_seed_mission_id_then_source_order",
        "excluded_unique_mission_counts": {
            "bc_train": len(bc_ids),
            "dev": len(dev_ids),
            "union": len(excluded_ids),
        },
        "intersections": {
            "final_bc_train": len(selected_ids.intersection(bc_ids)),
            "final_dev": len(selected_ids.intersection(dev_ids)),
        },
    }
    payload["selection_sha256"] = canonical_sha256(payload)
    write_json_atomic(selection, payload)
    return payload


def build_awac_evaluation_holdouts(
    source_index_path: Path,
    bc_train_index_path: Path,
    dev_index_path: Path,
    final_holdout_index_path: Path,
    selection_path: Path,
    *,
    dev_count: int,
    final_count: int,
    seed: int,
    additional_bc_train_index_paths: Sequence[Path] = (),
    overwrite: bool = False,
) -> Dict:
    """Select a deterministic BC-disjoint dev/final pair in one contract.

    Candidates are ranked once using ``seed`` and ``mission_id``.  The first
    ``dev_count`` missions become development missions and the next
    ``final_count`` become the untouched final holdout.  CSV rows retain source
    order.  The selection JSON is written last and acts as the commit record for
    the two individually atomic CSV writes.
    """

    if int(dev_count) <= 0 or int(final_count) <= 0:
        raise ValueError("dev and final holdout counts must be positive")
    if int(seed) < 0:
        raise ValueError("evaluation holdout seed must be non-negative")

    source_path = Path(source_index_path).expanduser().resolve()
    bc_paths = tuple(
        Path(path).expanduser().resolve()
        for path in (bc_train_index_path, *additional_bc_train_index_paths)
    )
    if len(set(bc_paths)) != len(bc_paths):
        raise ValueError("BC train exclusion indexes must be distinct")
    bc_path = bc_paths[0]
    dev_path = Path(dev_index_path).expanduser().resolve()
    final_path = Path(final_holdout_index_path).expanduser().resolve()
    selection = Path(selection_path).expanduser().resolve()
    inputs = {source_path, *bc_paths}
    outputs = {dev_path, final_path, selection}
    if len(outputs) != 3 or outputs.intersection(inputs):
        raise ValueError("evaluation holdout outputs must not overwrite inputs or each other")
    for output in outputs:
        if output.exists() and not bool(overwrite):
            raise FileExistsError("output exists; pass --overwrite: {}".format(output))

    source_rows, source_order = _read_index(source_path, label="source")
    bc_artifacts = [
        (path, *_read_exclusion_index(path, label="BC train {}".format(index)))
        for index, path in enumerate(bc_paths)
    ]
    bc_rows = [row for _, rows, _ in bc_artifacts for row in rows]
    bc_order = [mission_id for _, _, order in bc_artifacts for mission_id in order]
    source_ids = set(source_order)
    bc_ids = set(bc_order)
    bc_missing = sorted(bc_ids.difference(source_ids))
    if bc_missing:
        raise ValueError(
            "BC train index is not a subset of source: {}".format(bc_missing[:5])
        )

    candidate_order = [
        mission_id for mission_id in source_order if mission_id not in bc_ids
    ]
    requested_count = int(dev_count) + int(final_count)
    if requested_count > len(candidate_order):
        raise ValueError(
            "requested {} evaluation missions but only {} candidates remain after BC exclusion".format(
                requested_count, len(candidate_order)
            )
        )

    def selection_rank(mission_id: str) -> str:
        token = "{}\0{}".format(int(seed), mission_id).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    ranked = sorted(
        candidate_order,
        key=lambda mission_id: (selection_rank(mission_id), mission_id),
    )
    dev_ids = set(ranked[: int(dev_count)])
    final_ids = set(ranked[int(dev_count) : requested_count])
    if dev_ids.intersection(final_ids) or dev_ids.intersection(bc_ids) or final_ids.intersection(bc_ids):
        raise AssertionError("evaluation holdout selection is not mission-disjoint")

    dev_rows = [
        row for row, mission_id in zip(source_rows, source_order) if mission_id in dev_ids
    ]
    final_rows = [
        row for row, mission_id in zip(source_rows, source_order) if mission_id in final_ids
    ]
    write_csv_atomic(dev_path, dev_rows, fieldnames=list(source_rows[0].keys()))
    write_csv_atomic(final_path, final_rows, fieldnames=list(source_rows[0].keys()))
    persisted_dev_rows, persisted_dev_order = _read_index(dev_path, label="dev")
    persisted_final_rows, persisted_final_order = _read_index(
        final_path, label="final holdout"
    )
    if len(persisted_dev_order) != int(dev_count) or len(persisted_final_order) != int(final_count):
        raise RuntimeError("persisted evaluation holdout counts changed")

    def input_metadata(path: Path, rows: Sequence[Dict], order: Sequence[str]) -> Dict:
        return {
            "path": _relative_artifact_path(path, selection),
            "file_sha256": file_sha256(path),
            "row_count": len(rows),
            "unique_mission_count": len(set(order)),
            "ordered_mission_ids_sha256": _ordered_mission_sha256(order),
        }

    payload = {
        "contract_id": AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION_CONTRACT_ID,
        "source": input_metadata(source_path, source_rows, source_order),
        "bc_train": input_metadata(
            bc_artifacts[0][0], bc_artifacts[0][1], bc_artifacts[0][2]
        ),
        "additional_bc_train": [
            input_metadata(path, rows, order)
            for path, rows, order in bc_artifacts[1:]
        ],
        "dev": _artifact_metadata(
            dev_path, selection, persisted_dev_rows, persisted_dev_order
        ),
        "final_holdout": _artifact_metadata(
            final_path, selection, persisted_final_rows, persisted_final_order
        ),
        "seed": int(seed),
        "requested_counts": {
            "dev": int(dev_count),
            "final_holdout": int(final_count),
        },
        "candidate_count": len(candidate_order),
        "candidate_ordered_mission_ids_sha256": _ordered_mission_sha256(candidate_order),
        "selection_method": "sha256_seed_mission_id_rank_then_source_order",
        "excluded_unique_mission_counts": {
            "bc_train_union": len(bc_ids),
            "per_index": [len(set(order)) for _, _, order in bc_artifacts],
        },
        "intersections": {
            "dev_bc_train": len(dev_ids.intersection(bc_ids)),
            "final_bc_train": len(final_ids.intersection(bc_ids)),
            "dev_final_holdout": len(dev_ids.intersection(final_ids)),
        },
    }
    payload["selection_sha256"] = canonical_sha256(payload)
    write_json_atomic(selection, payload)
    return payload


def build_awac_dev_set(
    source_index_path: Path,
    bc_train_index_paths: Sequence[Path],
    final_test_index_paths: Sequence[Path],
    dev_index_path: Path,
    manifest_path: Path,
    *,
    count: int,
    seed: int,
    overwrite: bool = False,
) -> Dict:
    """Build the independent Phase-0 AWAC development mission set.

    The candidate pool is the source mission order minus every mission seen by
    the BC train artifacts and the frozen final-test artifacts.  Selection is
    deterministic from ``seed`` and mission identity; persisted rows retain
    source order so a rerun cannot depend on filesystem ordering.
    """

    if int(count) <= 0:
        raise ValueError("AWAC dev-set count must be positive")
    if int(seed) < 0:
        raise ValueError("AWAC dev-set seed must be non-negative")

    source_path = Path(source_index_path).expanduser().resolve()
    bc_paths = tuple(
        Path(path).expanduser().resolve() for path in bc_train_index_paths
    )
    final_paths = tuple(
        Path(path).expanduser().resolve() for path in final_test_index_paths
    )
    dev_path = Path(dev_index_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not bc_paths or not final_paths:
        raise ValueError("AWAC dev-set exclusions must include BC and final indexes")
    input_paths = {source_path, *bc_paths, *final_paths}
    if dev_path in input_paths or manifest in input_paths or dev_path == manifest:
        raise ValueError("AWAC dev-set outputs must not overwrite inputs or each other")
    if dev_path.exists() and not bool(overwrite):
        raise FileExistsError("output exists; pass --overwrite: {}".format(dev_path))
    if manifest.exists() and not bool(overwrite):
        raise FileExistsError("output exists; pass --overwrite: {}".format(manifest))

    source_rows, source_order = _read_index(source_path, label="source")
    bc_artifacts = [
        (path, *_read_exclusion_index(path, label="BC train {}".format(index)))
        for index, path in enumerate(bc_paths)
    ]
    final_artifacts = [
        (path, *_read_index(path, label="final test {}".format(index)))
        for index, path in enumerate(final_paths)
    ]
    source_ids = set(source_order)
    bc_ids = {mission_id for _, _, order in bc_artifacts for mission_id in order}
    final_ids = {
        mission_id for _, _, order in final_artifacts for mission_id in order
    }
    missing_bc = sorted(bc_ids.difference(source_ids))
    # Final-test indexes may come from a separately generated seed and need not
    # be subsets of this candidate source.  Their identities still participate
    # in the exclusion set, so any accidental overlap is rejected below.
    if missing_bc:
        details = {}
        details["bc_train_not_in_source"] = missing_bc[:5]
        raise ValueError("AWAC dev exclusions are not subsets of source: {}".format(details))

    excluded_ids = bc_ids.union(final_ids)
    candidate_order = [
        mission_id for mission_id in source_order if mission_id not in excluded_ids
    ]
    if int(count) > len(candidate_order):
        raise ValueError(
            "requested {} AWAC dev missions but only {} candidates remain after exclusions".format(
                int(count), len(candidate_order)
            )
        )

    def selection_rank(mission_id: str) -> str:
        token = "{}\0{}".format(int(seed), mission_id).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    selected_ids = set(
        sorted(candidate_order, key=lambda value: (selection_rank(value), value))[
            : int(count)
        ]
    )
    dev_rows = [
        row for row, mission_id in zip(source_rows, source_order)
        if mission_id in selected_ids
    ]
    dev_order = [mission_id for mission_id in source_order if mission_id in selected_ids]
    if len(dev_order) != int(count):
        raise AssertionError("AWAC dev-set selection returned the wrong count")
    if selected_ids.intersection(excluded_ids):
        raise AssertionError("AWAC dev-set selection overlaps an exclusion set")
    write_csv_atomic(dev_path, dev_rows, fieldnames=list(source_rows[0].keys()))
    persisted_rows, persisted_order = _read_index(dev_path, label="dev")
    if persisted_order != dev_order:
        raise RuntimeError("persisted AWAC dev index changed mission ordering")

    def metadata(path: Path, rows: Sequence[Dict], order: Sequence[str]) -> Dict:
        return {
            "path": _relative_artifact_path(path, manifest),
            "file_sha256": file_sha256(path),
            "row_count": len(rows),
            "mission_count": len(order),
            "ordered_mission_ids_sha256": _ordered_mission_sha256(order),
        }

    payload = {
        "contract_id": AWAC_DEV_SET_CONTRACT_ID,
        "role": "DEV",
        "seed": int(seed),
        "requested_count": int(count),
        "selection_method": "sha256_seed_mission_id_then_source_order",
        "source": metadata(source_path, source_rows, source_order),
        "dev": metadata(dev_path, persisted_rows, persisted_order),
        "bc_train": [
            metadata(path, rows, order) for path, rows, order in bc_artifacts
        ],
        "final_test": [
            metadata(path, rows, order) for path, rows, order in final_artifacts
        ],
        "candidate_count": len(candidate_order),
        "candidate_ordered_mission_ids_sha256": _ordered_mission_sha256(candidate_order),
        "excluded_unique_mission_counts": {
            "bc_train": len(bc_ids),
            "final_test": len(final_ids),
            "union": len(excluded_ids),
        },
        "intersections": {
            "dev_bc_train": len(selected_ids.intersection(bc_ids)),
            "dev_final_test": len(selected_ids.intersection(final_ids)),
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json_atomic(manifest, payload)
    return payload


def validate_awac_dev_set(
    manifest_path: Path,
    *,
    verify_referenced_indexes: bool = True,
) -> Dict:
    """Validate the independent Phase-0 dev artifact and exclusions."""

    manifest = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest)
    if str(payload.get("contract_id", "")) != AWAC_DEV_SET_CONTRACT_ID:
        raise ValueError("AWAC dev-set contract mismatch")
    if str(payload.get("role", "")) != "DEV":
        raise ValueError("AWAC dev-set role must be DEV")
    stored_hash = str(payload.get("manifest_sha256", ""))
    hash_payload = dict(payload)
    hash_payload.pop("manifest_sha256", None)
    if stored_hash != canonical_sha256(hash_payload):
        raise ValueError("AWAC dev-set canonical hash mismatch")
    for key in ("source", "dev"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError("AWAC dev-set is missing {} metadata".format(key))
    for key in ("bc_train", "final_test"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise ValueError("AWAC dev-set is missing {} exclusions".format(key))
    for key in ("source", "dev"):
        metadata = payload[key]
        for required in (
            "path",
            "file_sha256",
            "row_count",
            "mission_count",
            "ordered_mission_ids_sha256",
        ):
            if required not in metadata:
                raise ValueError("AWAC dev-set {} metadata lacks {}".format(key, required))
        if Path(str(metadata["path"])).is_absolute():
            raise ValueError("AWAC dev-set artifact paths must be relative")
    for key in ("bc_train", "final_test"):
        for index, metadata in enumerate(payload[key]):
            if not isinstance(metadata, Mapping):
                raise ValueError("AWAC dev-set {} exclusion {} is invalid".format(key, index))
            for required in (
                "path",
                "file_sha256",
                "row_count",
                "mission_count",
                "ordered_mission_ids_sha256",
            ):
                if required not in metadata:
                    raise ValueError(
                        "AWAC dev-set {} exclusion {} lacks {}".format(
                            key, index, required
                        )
                    )
            if Path(str(metadata["path"])).is_absolute():
                raise ValueError("AWAC dev-set artifact paths must be relative")

    intersections = payload.get("intersections")
    if not isinstance(intersections, Mapping):
        raise ValueError("AWAC dev-set intersections are missing")
    if {
        name: int(intersections.get(name, -1))
        for name in ("dev_bc_train", "dev_final_test")
    } != {"dev_bc_train": 0, "dev_final_test": 0}:
        raise ValueError("AWAC dev-set intersection evidence is non-zero")
    if not bool(verify_referenced_indexes):
        return payload

    def resolved(metadata: Mapping) -> Path:
        return (manifest.parent / str(metadata["path"])).resolve()

    artifact_orders: Dict[str, List[str]] = {}
    source_path = resolved(payload["source"])
    dev_path = resolved(payload["dev"])
    source_rows, source_order = _read_index(source_path, label="source")
    dev_rows, dev_order = _read_index(dev_path, label="dev")
    artifact_orders["source"] = source_order
    artifact_orders["dev"] = dev_order
    for key, rows, order, metadata in (
        ("source", source_rows, source_order, payload["source"]),
        ("dev", dev_rows, dev_order, payload["dev"]),
    ):
        if file_sha256(resolved(metadata)) != str(metadata["file_sha256"]):
            raise ValueError("AWAC dev-set {} index hash differs from manifest".format(key))
        if len(rows) != int(metadata["row_count"]) or len(order) != int(metadata["mission_count"]):
            raise ValueError("AWAC dev-set {} count differs from manifest".format(key))
        if _ordered_mission_sha256(order) != str(metadata["ordered_mission_ids_sha256"]):
            raise ValueError("AWAC dev-set {} ordering hash differs from manifest".format(key))

    exclusion_ids = {"bc_train": set(), "final_test": set()}
    for key in ("bc_train", "final_test"):
        for index, metadata in enumerate(payload[key]):
            path = resolved(metadata)
            rows, order = _read_exclusion_index(
                path, label="{} exclusion {}".format(key, index)
            )
            if file_sha256(path) != str(metadata["file_sha256"]):
                raise ValueError(
                    "AWAC dev-set {} exclusion {} hash differs from manifest".format(
                        key, index
                    )
                )
            if len(rows) != int(metadata["row_count"]) or len(order) != int(metadata["mission_count"]):
                raise ValueError(
                    "AWAC dev-set {} exclusion {} count differs from manifest".format(
                        key, index
                    )
                )
            exclusion_ids[key].update(order)
    source_ids = set(source_order)
    dev_ids = set(dev_order)
    if not dev_ids.issubset(source_ids):
        raise ValueError("AWAC dev-set index is not a source subset")
    for key, ids in exclusion_ids.items():
        if key == "final_test":
            continue
        if not ids.issubset(source_ids):
            raise ValueError("AWAC dev-set {} exclusion is not a source subset".format(key))
    if dev_ids.intersection(exclusion_ids["bc_train"]):
        raise ValueError("AWAC dev-set overlaps BC train")
    if dev_ids.intersection(exclusion_ids["final_test"]):
        raise ValueError("AWAC dev-set overlaps final test")
    expected_order = [
        mission_id for mission_id in source_order
        if mission_id in dev_ids
    ]
    if expected_order != dev_order:
        raise ValueError("AWAC dev-set ordering is not source-stable")
    return payload


def build_awac_calibration_split(
    source_index_path: Path,
    exclusion_index_paths: Sequence[Path],
    output_path: Path,
    *,
    seed: int,
    holdout_fraction: float = 0.10,
    overwrite: bool = False,
) -> Dict:
    """Create a deterministic episode-disjoint calibration train/holdout.

    Exclusions include every mission that belongs to BC training, the frozen
    final test, or the independent AWAC development set.  The split stores
    explicit identities, preventing transition-level random leakage on resume.
    """

    if int(seed) < 0:
        raise ValueError("calibration split seed must be non-negative")
    fraction = float(holdout_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration holdout fraction must be in (0,1)")
    source_path = Path(source_index_path).expanduser().resolve()
    exclusion_paths = tuple(
        Path(path).expanduser().resolve() for path in exclusion_index_paths
    )
    output = Path(output_path).expanduser().resolve()
    if not exclusion_paths:
        raise ValueError("calibration split requires at least one exclusion index")
    if output in {source_path, *exclusion_paths}:
        raise ValueError("calibration split output must not overwrite an input")
    if output.exists() and not bool(overwrite):
        raise FileExistsError("output exists; pass --overwrite: {}".format(output))

    source_rows, source_order = _read_index(source_path, label="calibration source")
    exclusion_artifacts = [
        (path, *_read_exclusion_index(path, label="calibration exclusion {}".format(index)))
        for index, path in enumerate(exclusion_paths)
    ]
    source_ids = set(source_order)
    excluded_ids = {
        mission_id
        for _, _, order in exclusion_artifacts
        for mission_id in order
    }
    eligible_order = [
        mission_id for mission_id in source_order if mission_id not in excluded_ids
    ]
    if len(eligible_order) < 2:
        raise ValueError("calibration source has fewer than two eligible missions")
    holdout_count = min(
        len(eligible_order) - 1,
        max(1, int(round(len(eligible_order) * fraction))),
    )

    def selection_rank(mission_id: str) -> str:
        token = "{}\0{}".format(int(seed), mission_id).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    holdout_ids = set(
        sorted(eligible_order, key=lambda value: (selection_rank(value), value))[
            :holdout_count
        ]
    )
    train_ids = set(eligible_order).difference(holdout_ids)
    train_order = [mission_id for mission_id in source_order if mission_id in train_ids]
    holdout_order = [
        mission_id for mission_id in source_order if mission_id in holdout_ids
    ]
    if not train_order or not holdout_order:
        raise AssertionError("calibration split produced an empty partition")

    def metadata(path: Path, rows: Sequence[Dict], order: Sequence[str]) -> Dict:
        return {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "row_count": len(rows),
            "mission_count": len(order),
            "ordered_mission_ids_sha256": _ordered_mission_sha256(order),
        }

    payload = {
        "contract_id": AWAC_CALIBRATION_SPLIT_CONTRACT_ID,
        "role": "CALIBRATION_SPLIT",
        "seed": int(seed),
        "holdout_fraction": fraction,
        "selection_method": "sha256_seed_mission_id_then_source_order",
        "source": metadata(source_path, source_rows, source_order),
        "exclusions": [
            metadata(path, rows, order)
            for path, rows, order in exclusion_artifacts
        ],
        "eligible_mission_count": len(eligible_order),
        "eligible_ordered_mission_ids_sha256": _ordered_mission_sha256(eligible_order),
        "train_episode_ids": train_order,
        "holdout_episode_ids": holdout_order,
        "train_mission_count": len(train_order),
        "holdout_mission_count": len(holdout_order),
        "intersections": {
            "train_holdout": len(train_ids.intersection(holdout_ids)),
            "train_excluded": len(train_ids.intersection(excluded_ids)),
            "holdout_excluded": len(holdout_ids.intersection(excluded_ids)),
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json_atomic(output, payload)
    return payload


def validate_awac_calibration_split(
    manifest_path: Path,
    *,
    verify_referenced_indexes: bool = True,
) -> Dict:
    """Validate a persisted calibration split and its episode partition."""

    manifest = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest)
    if str(payload.get("contract_id", "")) != AWAC_CALIBRATION_SPLIT_CONTRACT_ID:
        raise ValueError("AWAC calibration split contract mismatch")
    if str(payload.get("role", "")) != "CALIBRATION_SPLIT":
        raise ValueError("AWAC calibration split role mismatch")
    stored_hash = str(payload.get("manifest_sha256", ""))
    hash_payload = dict(payload)
    hash_payload.pop("manifest_sha256", None)
    if stored_hash != canonical_sha256(hash_payload):
        raise ValueError("AWAC calibration split canonical hash mismatch")
    train_ids = payload.get("train_episode_ids")
    holdout_ids = payload.get("holdout_episode_ids")
    if not isinstance(train_ids, list) or not isinstance(holdout_ids, list):
        raise ValueError("AWAC calibration split is missing explicit episode IDs")
    train_ids = [str(value) for value in train_ids]
    holdout_ids = [str(value) for value in holdout_ids]
    if not train_ids or not holdout_ids:
        raise ValueError("AWAC calibration split has an empty partition")
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError("AWAC calibration split contains duplicate episode IDs")
    if set(train_ids).intersection(holdout_ids):
        raise ValueError("AWAC calibration train/holdout overlap")
    if int(payload.get("train_mission_count", -1)) != len(train_ids):
        raise ValueError("AWAC calibration train count mismatch")
    if int(payload.get("holdout_mission_count", -1)) != len(holdout_ids):
        raise ValueError("AWAC calibration holdout count mismatch")
    intersections = payload.get("intersections")
    if not isinstance(intersections, Mapping) or any(
        int(intersections.get(key, -1)) != 0
        for key in ("train_holdout", "train_excluded", "holdout_excluded")
    ):
        raise ValueError("AWAC calibration split intersection evidence is non-zero")
    if not verify_referenced_indexes:
        return payload

    source = payload.get("source")
    exclusions = payload.get("exclusions")
    if not isinstance(source, Mapping) or not isinstance(exclusions, list) or not exclusions:
        raise ValueError("AWAC calibration split source metadata is incomplete")
    source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    source_rows, source_order = _read_index(source_path, label="calibration source")
    if file_sha256(source_path) != str(source.get("file_sha256", "")):
        raise ValueError("AWAC calibration source hash differs from manifest")
    if len(source_rows) != int(source.get("row_count", -1)):
        raise ValueError("AWAC calibration source row count differs from manifest")
    excluded_ids = set()
    for index, metadata in enumerate(exclusions):
        if not isinstance(metadata, Mapping):
            raise ValueError("AWAC calibration exclusion {} is invalid".format(index))
        path = Path(str(metadata.get("path", ""))).expanduser().resolve()
        rows, order = _read_exclusion_index(
            path, label="calibration exclusion {}".format(index)
        )
        if file_sha256(path) != str(metadata.get("file_sha256", "")):
            raise ValueError("AWAC calibration exclusion hash differs from manifest")
        if len(rows) != int(metadata.get("row_count", -1)):
            raise ValueError("AWAC calibration exclusion count differs from manifest")
        excluded_ids.update(order)
    source_ids = set(source_order)
    train_set = set(train_ids)
    holdout_set = set(holdout_ids)
    if not train_set.union(holdout_set).issubset(source_ids):
        raise ValueError("AWAC calibration IDs are not source identities")
    if train_set.intersection(excluded_ids) or holdout_set.intersection(excluded_ids):
        raise ValueError("AWAC calibration split includes an excluded identity")
    eligible_order = [
        mission_id for mission_id in source_order if mission_id not in excluded_ids
    ]
    expected_train_order = [mission_id for mission_id in eligible_order if mission_id not in holdout_set]
    expected_holdout_order = [mission_id for mission_id in eligible_order if mission_id in holdout_set]
    if train_ids != expected_train_order or holdout_ids != expected_holdout_order:
        raise ValueError("AWAC calibration split is not source-stable")
    if _ordered_mission_sha256(eligible_order) != str(
        payload.get("eligible_ordered_mission_ids_sha256", "")
    ):
        raise ValueError("AWAC calibration eligible identity hash differs from manifest")
    return payload


def _intersection_counts(
    train_ids: Set[str], dev_ids: Set[str], final_ids: Set[str]
) -> Dict[str, int]:
    return {
        "train_dev": len(train_ids.intersection(dev_ids)),
        "train_final_holdout": len(train_ids.intersection(final_ids)),
        "dev_final_holdout": len(dev_ids.intersection(final_ids)),
    }


def _require_zero_intersections(intersections: Mapping[str, int]) -> None:
    nonzero = {
        key: int(intersections.get(key, -1))
        for key in _INTERSECTION_KEYS
        if int(intersections.get(key, -1)) != 0
    }
    if nonzero:
        raise ValueError("AWAC mission splits overlap: {}".format(nonzero))


def build_awac_mission_split(
    source_index_path: Path,
    dev_index_path: Path,
    final_holdout_index_path: Path,
    train_index_path: Path,
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> Dict:
    """Build and persist a train index disjoint from dev and final holdout.

    Source row ordering and fields are preserved.  Development and final
    indexes must each be a mission-level subset of the source index, and every
    source mission is assigned to exactly one of train, dev, or final holdout.
    """

    source_path = Path(source_index_path).expanduser().resolve()
    dev_path = Path(dev_index_path).expanduser().resolve()
    final_path = Path(final_holdout_index_path).expanduser().resolve()
    train_path = Path(train_index_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()

    input_paths = {source_path, dev_path, final_path}
    if train_path == manifest or train_path in input_paths or manifest in input_paths:
        raise ValueError("AWAC split outputs must not overwrite an input or each other")
    for output in (train_path, manifest):
        if output.exists() and not bool(overwrite):
            raise FileExistsError("output exists; pass --overwrite: {}".format(output))

    source_rows, source_order = _read_index(source_path, label="source")
    dev_rows, dev_order = _read_index(dev_path, label="dev")
    final_rows, final_order = _read_index(final_path, label="final holdout")
    source_ids = set(source_order)
    dev_ids = set(dev_order)
    final_ids = set(final_order)

    dev_missing = sorted(dev_ids.difference(source_ids))
    final_missing = sorted(final_ids.difference(source_ids))
    if dev_missing or final_missing:
        details = {}
        if dev_missing:
            details["dev_not_in_source"] = dev_missing[:5]
        if final_missing:
            details["final_holdout_not_in_source"] = final_missing[:5]
        raise ValueError("holdout indexes are not subsets of source: {}".format(details))

    preliminary_intersections = _intersection_counts(
        source_ids.difference(dev_ids).difference(final_ids), dev_ids, final_ids
    )
    _require_zero_intersections(preliminary_intersections)

    excluded_ids = dev_ids.union(final_ids)
    train_rows = [
        row for row, mission_id in zip(source_rows, source_order)
        if mission_id not in excluded_ids
    ]
    train_order = [
        mission_id for mission_id in source_order if mission_id not in excluded_ids
    ]
    if not train_rows:
        raise ValueError("AWAC train split is empty after excluding dev and final holdout")

    train_ids = set(train_order)
    intersections = _intersection_counts(train_ids, dev_ids, final_ids)
    _require_zero_intersections(intersections)
    assigned_ids = train_ids.union(dev_ids).union(final_ids)
    if assigned_ids != source_ids:
        raise AssertionError("AWAC split did not partition the complete source mission set")

    write_csv_atomic(train_path, train_rows, fieldnames=list(source_rows[0].keys()))
    persisted_train_rows, persisted_train_order = _read_index(train_path, label="train")
    if persisted_train_order != train_order:
        raise RuntimeError("persisted AWAC train index changed mission ordering")

    payload = {
        "contract_id": AWAC_MISSION_SPLIT_CONTRACT_ID,
        "source": _artifact_metadata(source_path, manifest, source_rows, source_order),
        "train": _artifact_metadata(
            train_path, manifest, persisted_train_rows, persisted_train_order
        ),
        "dev": _artifact_metadata(dev_path, manifest, dev_rows, dev_order),
        "final_holdout": _artifact_metadata(
            final_path, manifest, final_rows, final_order
        ),
        "intersections": intersections,
        "partition": {
            "source_mission_count": len(source_ids),
            "assigned_mission_count": len(assigned_ids),
            "unassigned_mission_count": len(source_ids.difference(assigned_ids)),
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json_atomic(manifest, payload)
    return payload


def _load_manifest(manifest_path: Path) -> Dict:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("AWAC split manifest does not exist: {}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("cannot read AWAC split manifest: {}".format(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("AWAC split manifest must contain a JSON object")
    return payload


def validate_awac_mission_split(
    manifest_path: Path,
    *,
    train_index_path: Optional[Path] = None,
    verify_referenced_indexes: bool = True,
) -> Dict:
    """Validate a persisted split and return its canonical manifest.

    When ``verify_referenced_indexes`` is true (the formal/default mode), every
    referenced CSV is rehashed and its mission identities are recomputed before
    the disjoint partition is accepted.  A trainer may also pass its explicit
    train index; it must resolve to the manifest's train artifact.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest)
    if str(payload.get("contract_id", "")) != AWAC_MISSION_SPLIT_CONTRACT_ID:
        raise ValueError("AWAC mission split contract mismatch")
    stored_manifest_sha256 = str(payload.get("manifest_sha256", ""))
    hash_payload = dict(payload)
    hash_payload.pop("manifest_sha256", None)
    if stored_manifest_sha256 != canonical_sha256(hash_payload):
        raise ValueError("AWAC mission split canonical hash mismatch")
    for key in _ARTIFACT_KEYS:
        metadata = payload.get(key)
        if not isinstance(metadata, dict):
            raise ValueError("AWAC mission split is missing {} metadata".format(key))
        for required in (
            "path",
            "file_sha256",
            "row_count",
            "mission_count",
            "ordered_mission_ids_sha256",
        ):
            if required not in metadata:
                raise ValueError("AWAC mission split {} metadata lacks {}".format(key, required))
        if Path(str(metadata["path"])).is_absolute():
            raise ValueError("AWAC mission split artifact paths must be relative")

    intersections = payload.get("intersections", {})
    if not isinstance(intersections, dict):
        raise ValueError("AWAC mission split intersections must be an object")
    _require_zero_intersections(intersections)

    resolved_paths = {
        key: (manifest.parent / str(payload[key]["path"])).resolve()
        for key in _ARTIFACT_KEYS
    }
    if train_index_path is not None:
        explicit_train = Path(train_index_path).expanduser().resolve()
        if explicit_train != resolved_paths["train"]:
            raise ValueError(
                "--train-index does not match the train artifact referenced by the AWAC split manifest"
            )

    if not bool(verify_referenced_indexes):
        train_path = resolved_paths["train"]
        if not train_path.is_file():
            raise FileNotFoundError("AWAC train index does not exist: {}".format(train_path))
        if file_sha256(train_path) != str(payload["train"]["file_sha256"]):
            raise ValueError("AWAC train index hash differs from split manifest")
        return payload

    artifact_orders: Dict[str, List[str]] = {}
    for key in _ARTIFACT_KEYS:
        rows, mission_order = _read_index(resolved_paths[key], label=key)
        metadata = payload[key]
        if file_sha256(resolved_paths[key]) != str(metadata["file_sha256"]):
            raise ValueError("{} index hash differs from AWAC split manifest".format(key))
        if len(rows) != int(metadata["row_count"]):
            raise ValueError("{} row count differs from AWAC split manifest".format(key))
        if len(mission_order) != int(metadata["mission_count"]):
            raise ValueError("{} mission count differs from AWAC split manifest".format(key))
        if _ordered_mission_sha256(mission_order) != str(
            metadata["ordered_mission_ids_sha256"]
        ):
            raise ValueError("{} ordered mission digest differs from AWAC split manifest".format(key))
        artifact_orders[key] = mission_order

    source_ids = set(artifact_orders["source"])
    train_ids = set(artifact_orders["train"])
    dev_ids = set(artifact_orders["dev"])
    final_ids = set(artifact_orders["final_holdout"])
    actual_intersections = _intersection_counts(train_ids, dev_ids, final_ids)
    _require_zero_intersections(actual_intersections)
    if actual_intersections != {
        key: int(intersections[key]) for key in _INTERSECTION_KEYS
    }:
        raise ValueError("AWAC split intersection evidence differs from manifest")
    if not dev_ids.issubset(source_ids) or not final_ids.issubset(source_ids):
        raise ValueError("AWAC dev/final holdout indexes are not subsets of source")
    expected_train_order = [
        mission_id for mission_id in artifact_orders["source"]
        if mission_id not in dev_ids and mission_id not in final_ids
    ]
    if artifact_orders["train"] != expected_train_order:
        raise ValueError("AWAC train index is not the ordered source-minus-holdouts partition")
    if train_ids.union(dev_ids).union(final_ids) != source_ids:
        raise ValueError("AWAC split does not partition every source mission")

    partition = payload.get("partition", {})
    expected_partition = {
        "source_mission_count": len(source_ids),
        "assigned_mission_count": len(train_ids.union(dev_ids).union(final_ids)),
        "unassigned_mission_count": len(
            source_ids.difference(train_ids.union(dev_ids).union(final_ids))
        ),
    }
    if not isinstance(partition, dict) or {
        key: int(partition.get(key, -1)) for key in expected_partition
    } != expected_partition:
        raise ValueError("AWAC split partition evidence differs from manifest")
    return payload


__all__ = [
    "AWAC_CALIBRATION_SPLIT_CONTRACT_ID",
    "AWAC_DEV_SET_CONTRACT_ID",
    "AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION_CONTRACT_ID",
    "AWAC_FINAL_HOLDOUT_SELECTION_CONTRACT_ID",
    "AWAC_MISSION_SPLIT_CONTRACT_ID",
    "build_awac_calibration_split",
    "build_awac_dev_set",
    "build_awac_evaluation_holdouts",
    "build_awac_final_holdout",
    "build_awac_mission_split",
    "validate_awac_calibration_split",
    "validate_awac_dev_set",
    "validate_awac_mission_split",
]
