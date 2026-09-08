from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from planning.evaluation.mission_split import (
    AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION_CONTRACT_ID,
    AWAC_FINAL_HOLDOUT_SELECTION_CONTRACT_ID,
    AWAC_MISSION_SPLIT_CONTRACT_ID,
    build_awac_evaluation_holdouts,
    build_awac_final_holdout,
    build_awac_mission_split,
    validate_awac_mission_split,
)


def _write_index(path: Path, mission_ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "mission_id", "marker"])
        writer.writeheader()
        for episode_id, mission_id in enumerate(mission_ids):
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "mission_id": mission_id,
                    "marker": "source-{}".format(mission_id),
                }
            )


@pytest.mark.unit
def test_build_and_validate_disjoint_awac_mission_split(tmp_path: Path) -> None:
    source = tmp_path / "input" / "missions.csv"
    dev = tmp_path / "eval" / "dev.csv"
    final = tmp_path / "eval" / "final.csv"
    train = tmp_path / "run" / "train.csv"
    manifest_path = tmp_path / "run" / "split.json"
    _write_index(source, ["m0", "m1", "m2", "m3", "m4", "m5"])
    _write_index(dev, ["m1", "m4"])
    _write_index(final, ["m2"])

    manifest = build_awac_mission_split(source, dev, final, train, manifest_path)
    validated = validate_awac_mission_split(
        manifest_path, train_index_path=train, verify_referenced_indexes=True
    )

    assert manifest["contract_id"] == AWAC_MISSION_SPLIT_CONTRACT_ID
    assert validated["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["intersections"] == {
        "train_dev": 0,
        "train_final_holdout": 0,
        "dev_final_holdout": 0,
    }
    assert manifest["train"]["mission_count"] == 3
    assert not Path(manifest["source"]["path"]).is_absolute()
    with train.open("r", newline="", encoding="utf-8") as handle:
        assert [row["mission_id"] for row in csv.DictReader(handle)] == ["m0", "m3", "m5"]


@pytest.mark.unit
def test_split_rejects_dev_final_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    dev = tmp_path / "dev.csv"
    final = tmp_path / "final.csv"
    _write_index(source, ["m0", "m1", "m2"])
    _write_index(dev, ["m1"])
    _write_index(final, ["m1"])

    with pytest.raises(ValueError, match="overlap"):
        build_awac_mission_split(
            source,
            dev,
            final,
            tmp_path / "train.csv",
            tmp_path / "manifest.json",
        )


@pytest.mark.unit
def test_split_rejects_holdout_outside_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    dev = tmp_path / "dev.csv"
    final = tmp_path / "final.csv"
    _write_index(source, ["m0", "m1", "m2"])
    _write_index(dev, ["outside"])
    _write_index(final, ["m2"])

    with pytest.raises(ValueError, match="not subsets"):
        build_awac_mission_split(
            source,
            dev,
            final,
            tmp_path / "train.csv",
            tmp_path / "manifest.json",
        )


@pytest.mark.unit
def test_validator_rejects_modified_train_index_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    dev = tmp_path / "dev.csv"
    final = tmp_path / "final.csv"
    train = tmp_path / "train.csv"
    manifest_path = tmp_path / "manifest.json"
    _write_index(source, ["m0", "m1", "m2", "m3"])
    _write_index(dev, ["m1"])
    _write_index(final, ["m2"])
    build_awac_mission_split(source, dev, final, train, manifest_path)

    _write_index(train, ["m0", "m2", "m3"])
    with pytest.raises(ValueError, match="hash differs"):
        validate_awac_mission_split(manifest_path, train_index_path=train)

    # Restore a valid split, then prove the canonical manifest hash is checked.
    build_awac_mission_split(source, dev, final, train, manifest_path, overwrite=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["partition"]["unassigned_mission_count"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical hash"):
        validate_awac_mission_split(manifest_path, train_index_path=train)


@pytest.mark.unit
def test_split_never_overwrites_an_input_index(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    dev = tmp_path / "dev.csv"
    final = tmp_path / "final.csv"
    _write_index(source, ["m0", "m1", "m2"])
    _write_index(dev, ["m1"])
    _write_index(final, ["m2"])

    with pytest.raises(ValueError, match="must not overwrite"):
        build_awac_mission_split(
            source,
            dev,
            final,
            source,
            tmp_path / "manifest.json",
            overwrite=True,
        )


@pytest.mark.unit
def test_final_holdout_is_deterministic_and_excludes_bc_and_dev(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    bc_train = tmp_path / "bc_train.csv"
    dev = tmp_path / "dev.csv"
    final_a = tmp_path / "a" / "final.csv"
    selection_a = tmp_path / "a" / "selection.json"
    final_b = tmp_path / "b" / "final.csv"
    selection_b = tmp_path / "b" / "selection.json"
    _write_index(source, ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"])
    _write_index(bc_train, ["m0", "m2", "m4"])
    _write_index(dev, ["m5"])

    first = build_awac_final_holdout(
        source,
        bc_train,
        dev,
        final_a,
        selection_a,
        count=2,
        seed=20260802,
    )
    second = build_awac_final_holdout(
        source,
        bc_train,
        dev,
        final_b,
        selection_b,
        count=2,
        seed=20260802,
    )

    with final_a.open("r", newline="", encoding="utf-8") as handle:
        first_ids = [row["mission_id"] for row in csv.DictReader(handle)]
    with final_b.open("r", newline="", encoding="utf-8") as handle:
        second_ids = [row["mission_id"] for row in csv.DictReader(handle)]
    assert first_ids == second_ids
    assert not set(first_ids).intersection({"m0", "m2", "m4", "m5"})
    assert first["contract_id"] == AWAC_FINAL_HOLDOUT_SELECTION_CONTRACT_ID
    assert first["candidate_count"] == 4
    assert first["intersections"] == {"final_bc_train": 0, "final_dev": 0}
    assert first["final_holdout"]["ordered_mission_ids_sha256"] == second[
        "final_holdout"
    ]["ordered_mission_ids_sha256"]
    assert not Path(first["source"]["path"]).is_absolute()


@pytest.mark.unit
def test_final_holdout_rejects_insufficient_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    bc_train = tmp_path / "bc_train.csv"
    dev = tmp_path / "dev.csv"
    _write_index(source, ["m0", "m1", "m2"])
    _write_index(bc_train, ["m0", "m1"])
    _write_index(dev, ["m2"])

    with pytest.raises(ValueError, match="only 0 candidates"):
        build_awac_final_holdout(
            source,
            bc_train,
            dev,
            tmp_path / "final.csv",
            tmp_path / "selection.json",
            count=1,
            seed=7,
        )


@pytest.mark.unit
def test_evaluation_holdout_pair_is_deterministic_and_bc_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    bc_train = tmp_path / "bc_train.csv"
    older_bc_train = tmp_path / "older_bc_train.csv"
    _write_index(source, ["m{}".format(index) for index in range(12)])
    _write_index(bc_train, ["m0", "m2", "m4", "m6"])
    _write_index(older_bc_train, ["m1", "m3"])

    first = build_awac_evaluation_holdouts(
        source,
        bc_train,
        tmp_path / "a" / "dev.csv",
        tmp_path / "a" / "final.csv",
        tmp_path / "a" / "selection.json",
        dev_count=2,
        final_count=3,
        seed=55,
        additional_bc_train_index_paths=(older_bc_train,),
    )
    second = build_awac_evaluation_holdouts(
        source,
        bc_train,
        tmp_path / "b" / "dev.csv",
        tmp_path / "b" / "final.csv",
        tmp_path / "b" / "selection.json",
        dev_count=2,
        final_count=3,
        seed=55,
        additional_bc_train_index_paths=(older_bc_train,),
    )

    assert first["contract_id"] == AWAC_EVALUATION_HOLDOUT_PAIR_SELECTION_CONTRACT_ID
    assert first["intersections"] == {
        "dev_bc_train": 0,
        "final_bc_train": 0,
        "dev_final_holdout": 0,
    }
    assert len(first["additional_bc_train"]) == 1
    assert first["excluded_unique_mission_counts"] == {
        "bc_train_union": 6,
        "per_index": [4, 2],
    }
    assert first["dev"]["ordered_mission_ids_sha256"] == second["dev"][
        "ordered_mission_ids_sha256"
    ]
    assert first["final_holdout"]["ordered_mission_ids_sha256"] == second[
        "final_holdout"
    ]["ordered_mission_ids_sha256"]
    assert first["selection_sha256"] == second["selection_sha256"]
