"""Audited development-set checkpoint selection for AWAC.

This module deliberately accepts only the development artifact named by a
validated AWAC split manifest.  The final holdout is therefore unavailable to
the repeated train/select loop.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from planning.contracts.policy_runtime import (
    MIN_FORMAL_HOLDOUT_EPISODES,
    POLICY_RUNTIME_CONTRACT_ID,
    POLICY_UNITY_EVALUATION_CONTRACT_ID,
    POLICY_UNITY_OUTCOME_CONTRACT_ID,
    checkpoint_depth_mask_numeric_config,
    holdout_score,
)
from planning.evaluation.mission_split import validate_awac_mission_split
from planning.common import file_sha256, write_json_atomic
from planning.data.rollout import read_csv
from planning.mission.spec import mission_id_from_row


AWAC_DEV_SELECTION_CONTRACT_ID = "awac_dev_checkpoint_selection"
_OUTCOMES = (
    "success",
    "collision",
    "dead_end",
    "timeout",
    "far",
    "altitude_violation",
)


def _load_json(path: Path, *, label: str) -> Dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(label, resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("cannot read {}: {}".format(label, resolved)) from exc
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(label))
    return payload


def _require_file(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(label, resolved))
    return resolved


def _manifest_contains_path(value: Any, target: Path) -> bool:
    """Return whether a manifest value recursively names ``target``."""

    target = Path(target).expanduser().resolve()
    if isinstance(value, Mapping):
        return any(_manifest_contains_path(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_manifest_contains_path(item, target) for item in value)
    if isinstance(value, str):
        try:
            return Path(value).expanduser().resolve() == target
        except (OSError, RuntimeError, ValueError):
            return False
    return False


def validate_training_mission_source(
    source_index_path: Path,
    *,
    final_test_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate an index is safe to use as an AWAC training/calibration source.

    The final holdout is an immutable evaluation artifact.  A role-bearing
    final-test manifest is rejected before the source index is opened, which
    also prevents a missing or malformed holdout path from being silently
    treated as a training input.
    """

    source = Path(source_index_path).expanduser().resolve()
    if final_test_manifest is not None:
        manifest_path = Path(final_test_manifest).expanduser().resolve()
        manifest = _load_json(manifest_path, label="final-test manifest")
        role = str(manifest.get("role", "")).strip().upper()
        if role in {"FINAL_TEST", "FINAL_TEST_ONLY"} or bool(
            manifest.get("final_test_only", False)
        ):
            raise ValueError(
                "FINAL_TEST_ONLY artifact cannot be used as a training mission source"
            )
        if _manifest_contains_path(manifest, source):
            raise ValueError(
                "FINAL_TEST_ONLY mission index cannot be used as a training source"
            )

    source = _require_file(source, label="training mission source")
    rows = read_csv(source)
    if not rows:
        raise ValueError("training mission source is empty: {}".format(source))
    mission_ids = []
    seen = set()
    for row in rows:
        try:
            mission_id = str(mission_id_from_row(row)).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("training mission source has an invalid mission identity") from exc
        if not mission_id:
            raise ValueError("training mission source has an empty mission identity")
        if mission_id in seen:
            raise ValueError("training mission source has duplicate mission identity: {}".format(mission_id))
        seen.add(mission_id)
        mission_ids.append(mission_id)
    return {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "row_count": len(rows),
        "mission_ids": mission_ids,
        "final_test_excluded": True,
    }


def _validate_outcomes(summary: Mapping, *, label: str) -> int:
    try:
        episodes = int(summary["episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} has no valid episode count".format(label)) from exc
    if episodes < MIN_FORMAL_HOLDOUT_EPISODES:
        raise ValueError(
            "{} requires at least {} episodes".format(
                label, MIN_FORMAL_HOLDOUT_EPISODES
            )
        )
    if summary.get("quality_gate_applicable") is not True:
        raise ValueError("{} quality gate is not applicable".format(label))
    if summary.get("outcome_partition_valid") is not True:
        raise ValueError("{} outcome partition is invalid".format(label))
    if str(summary.get("policy_unity_outcome_contract_id", "")) != (
        POLICY_UNITY_OUTCOME_CONTRACT_ID
    ):
        raise ValueError("{} outcome contract mismatch".format(label))
    if int(summary.get("terminal_outcome_total", -1)) != episodes:
        raise ValueError("{} terminal outcome count does not equal episodes".format(label))
    if int(summary.get("unclassified_count", -1)) != 0:
        raise ValueError("{} contains unclassified outcomes".format(label))
    if int(summary.get("ambiguous_outcome_count", -1)) != 0:
        raise ValueError("{} contains ambiguous outcomes".format(label))

    total = 0
    for outcome in _OUTCOMES:
        count_key = "{}_count".format(outcome)
        rate_key = "{}_rate".format(outcome)
        try:
            count = int(summary[count_key])
            rate = float(summary[rate_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "{} has incomplete {} outcome evidence".format(label, outcome)
            ) from exc
        if count < 0 or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("{} has invalid {} outcome values".format(label, outcome))
        expected_rate = float(count) / float(episodes)
        if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("{} {} rate disagrees with its count".format(label, outcome))
        total += count
    if total != episodes:
        raise ValueError("{} outcome counts do not partition episodes".format(label))
    return episodes


def _runtime_evidence(summary: Mapping, *, label: str) -> Dict:
    if str(summary.get("evaluation_contract_id", "")) != (
        POLICY_UNITY_EVALUATION_CONTRACT_ID
    ):
        raise ValueError("{} evaluation contract mismatch".format(label))
    if str(summary.get("policy_runtime_contract_id", "")) != (
        POLICY_RUNTIME_CONTRACT_ID
    ):
        raise ValueError("{} runtime contract mismatch".format(label))
    if summary.get("runtime_contract_override") is not False:
        raise ValueError("{} used a runtime contract override".format(label))
    if summary.get("depth_mask_runtime_override") is not False:
        raise ValueError("{} used a depth-mask runtime override".format(label))
    policy_action_mode = str(summary.get("policy_action_mode", ""))
    try:
        policy_temperature = float(summary["policy_temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} lacks a valid policy temperature".format(label)) from exc
    if policy_action_mode != "deterministic_argmax" or not math.isclose(
        policy_temperature, 0.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "{} must use deterministic_argmax with policy_temperature=0".format(
                label
            )
        )
    nested = summary.get("depth_mask_config")
    if not isinstance(nested, Mapping):
        raise ValueError("{} lacks resolved depth-mask configuration".format(label))
    for key in (
        "collision_radius_m",
        "slack_m",
        "sample_stride",
        "max_patch_radius_px",
    ):
        if key not in nested:
            raise ValueError("{} depth-mask configuration lacks {}".format(label, key))
    depth_config = checkpoint_depth_mask_numeric_config(summary)
    evidence = {
        "policy_runtime_contract_id": str(summary["policy_runtime_contract_id"]),
        "safety_mask": str(summary.get("safety_mask", "")),
        "execution_mode": str(summary.get("execution_mode", "")),
        "policy_action_mode": policy_action_mode,
        "policy_temperature": policy_temperature,
        "local_depth_collision_mask": bool(summary.get("local_depth_collision_mask", False)),
        "global_collision_mask": bool(summary.get("global_collision_mask", False)),
        "depth_mask_config": depth_config,
    }
    if evidence["safety_mask"] != "depth":
        raise ValueError("{} did not use the deployable depth safety mask".format(label))
    if not evidence["local_depth_collision_mask"] or evidence["global_collision_mask"]:
        raise ValueError("{} depth/global collision-mask evidence is invalid".format(label))
    return evidence


def _validate_summary(
    summary: Mapping,
    *,
    checkpoint_path: Path,
    dev_index_sha256: str,
    final_index_sha256: str,
    label: str,
) -> Tuple[int, Dict]:
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if str(summary.get("checkpoint_sha256", "")) != checkpoint_sha256:
        raise ValueError("{} checkpoint SHA256 does not match its file".format(label))
    mission_sha256 = str(summary.get("mission_index_sha256", ""))
    if mission_sha256 == str(final_index_sha256):
        raise ValueError("{} illegally uses the final holdout index".format(label))
    if mission_sha256 != str(dev_index_sha256):
        raise ValueError("{} does not use the split manifest dev index".format(label))
    episodes = _validate_outcomes(summary, label=label)
    runtime = _runtime_evidence(summary, label=label)
    return episodes, runtime


def _atomic_copy(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(os.path.abspath(str(Path(destination).expanduser())))
    if destination.is_symlink():
        raise ValueError("selected checkpoint output must not be a symbolic link")
    if source == destination or (
        destination.exists() and os.path.samefile(str(source), str(destination))
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            file_descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        shutil.copystat(str(source), str(temporary))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_path(path: Path, decision_path: Path) -> str:
    return os.path.relpath(
        str(Path(path).expanduser().resolve()),
        str(Path(os.path.abspath(str(Path(decision_path).expanduser()))).parent),
    )


def select_awac_dev_checkpoint(
    *,
    split_manifest_path: Path,
    candidate_checkpoint_path: Path,
    candidate_summary_path: Path,
    incumbent_checkpoint_path: Path,
    incumbent_summary_path: Path,
    out_checkpoint_path: Path,
    out_decision_path: Path,
) -> Dict:
    """Select a candidate using only the manifest's independent dev split."""

    split_manifest_path = _require_file(split_manifest_path, label="split manifest")
    candidate_checkpoint_path = _require_file(
        candidate_checkpoint_path, label="candidate checkpoint"
    )
    incumbent_checkpoint_path = _require_file(
        incumbent_checkpoint_path, label="incumbent checkpoint"
    )
    candidate_summary = _load_json(candidate_summary_path, label="candidate summary")
    incumbent_summary = _load_json(incumbent_summary_path, label="incumbent summary")
    manifest = validate_awac_mission_split(
        split_manifest_path,
        verify_referenced_indexes=True,
    )
    dev_sha256 = str(manifest["dev"]["file_sha256"])
    final_sha256 = str(manifest["final_holdout"]["file_sha256"])
    candidate_episodes, candidate_runtime = _validate_summary(
        candidate_summary,
        checkpoint_path=candidate_checkpoint_path,
        dev_index_sha256=dev_sha256,
        final_index_sha256=final_sha256,
        label="candidate summary",
    )
    incumbent_episodes, incumbent_runtime = _validate_summary(
        incumbent_summary,
        checkpoint_path=incumbent_checkpoint_path,
        dev_index_sha256=dev_sha256,
        final_index_sha256=final_sha256,
        label="incumbent summary",
    )
    if candidate_episodes != incumbent_episodes:
        raise ValueError("candidate and incumbent used different dev episode counts")
    if candidate_summary.get("episode_filter_ids", []) != incumbent_summary.get(
        "episode_filter_ids", []
    ):
        raise ValueError("candidate and incumbent used different dev episode filters")
    if candidate_runtime != incumbent_runtime:
        raise ValueError("candidate and incumbent runtime/depth contracts differ")

    candidate_score = tuple(float(value) for value in holdout_score(candidate_summary))
    incumbent_score = tuple(float(value) for value in holdout_score(incumbent_summary))
    collision_non_regression = float(candidate_summary["collision_rate"]) <= float(
        incumbent_summary["collision_rate"]
    )
    score_strict_improvement = candidate_score > incumbent_score
    candidate_selected = bool(collision_non_regression and score_strict_improvement)
    selected_source = (
        candidate_checkpoint_path if candidate_selected else incumbent_checkpoint_path
    )
    out_checkpoint_path = Path(
        os.path.abspath(str(Path(out_checkpoint_path).expanduser()))
    )
    out_decision_path = Path(
        os.path.abspath(str(Path(out_decision_path).expanduser()))
    )
    if out_checkpoint_path == out_decision_path:
        raise ValueError("selected checkpoint and decision outputs must differ")
    _atomic_copy(selected_source, out_checkpoint_path)
    selected_sha256 = file_sha256(out_checkpoint_path)
    expected_selected_sha256 = file_sha256(selected_source)
    if selected_sha256 != expected_selected_sha256:
        raise RuntimeError("atomic selected checkpoint copy failed SHA256 verification")

    decision = {
        "contract_id": AWAC_DEV_SELECTION_CONTRACT_ID,
        "split_manifest_sha256": str(manifest["manifest_sha256"]),
        "dev_index_sha256": dev_sha256,
        "final_holdout_index_sha256": final_sha256,
        "episodes": candidate_episodes,
        "runtime": candidate_runtime,
        "candidate": {
            "checkpoint": _relative_path(candidate_checkpoint_path, out_decision_path),
            "checkpoint_sha256": file_sha256(candidate_checkpoint_path),
            "summary": _relative_path(candidate_summary_path, out_decision_path),
            "score": list(candidate_score),
            "collision_rate": float(candidate_summary["collision_rate"]),
        },
        "incumbent": {
            "checkpoint": _relative_path(incumbent_checkpoint_path, out_decision_path),
            "checkpoint_sha256": file_sha256(incumbent_checkpoint_path),
            "summary": _relative_path(incumbent_summary_path, out_decision_path),
            "score": list(incumbent_score),
            "collision_rate": float(incumbent_summary["collision_rate"]),
        },
        "gates": {
            "collision_non_regression": collision_non_regression,
            "score_strict_improvement": score_strict_improvement,
        },
        "candidate_selected": candidate_selected,
        "selected_source": "candidate" if candidate_selected else "incumbent",
        "selected_checkpoint": _relative_path(out_checkpoint_path, out_decision_path),
        "selected_checkpoint_sha256": selected_sha256,
    }
    write_json_atomic(out_decision_path, decision)
    return decision


__all__ = [
    "AWAC_DEV_SELECTION_CONTRACT_ID",
    "select_awac_dev_checkpoint",
    "validate_training_mission_source",
]
