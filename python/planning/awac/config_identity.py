"""Field-level identity checks for Calibration and Standard AWAC configs."""

from __future__ import annotations

from typing import Any, Dict, Mapping


EXPECTED_PHASE_CHANGE = "EXPECTED_PHASE_CHANGE"
EXPECTED_ACTOR_ENABLE_CHANGE = "EXPECTED_ACTOR_ENABLE_CHANGE"
INVOCATION_ONLY = "INVOCATION_ONLY"
IDENTITY = "IDENTITY"
TRAINING_SEMANTIC = "TRAINING_SEMANTIC"
UNKNOWN = "UNKNOWN"

_INVOCATION_PATHS = {
    "bc_checkpoint_path",
    "train_index",
    "calibration_split_path",
    "device",
    "env_workers",
    "cpu_threads",
    "reliable_execution_enabled",
    "fresh_online_calibration",
}
_KNOWN_IDENTITY_SUFFIXES = (
    "_contract_id",
    "_contract_sha256",
    "_sha256",
    "schema_version",
    "software_version",
)
_KNOWN_SEMANTIC_PREFIXES = (
    "training_contract.",
    "optimization_config.",
)
_KNOWN_TOP_LEVEL_FIELDS = {
    "phase",
    "actor_update_enabled",
    "training_contract",
    "optimization_config",
    "phase1_training_contract",
    "phase1_training_contract_sha256",
}
_TRAINING_CONTRACT_FIELDS = {
    "schema_version",
    "contract_id",
    "algorithm",
    "phase",
    "feature_contract_id",
    "policy_input_contract",
    "policy_input_contract_sha256",
    "task_contract_id",
    "task_contract_sha256",
    "max_primitive_steps",
    "observation_contract",
    "observation_source",
    "reward_contract_id",
    "reward_contract_sha256",
    "reward_scale",
    "reward_scale_owner",
    "gamma",
    "tau",
    "replay_contract_id",
    "replay_contract_sha256",
    "replay_fields",
    "behavior_sources",
    "actor_initialization",
    "critic_initialization",
    "safety_mask_contract",
    "terminal_contract",
    "actor_update_enabled",
    "privileged_policy_inputs",
    "calibration",
}


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}}
        result: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            child = str(key) if not prefix else "{}.{}".format(prefix, key)
            result.update(_flatten(value[key], child))
        return result
    return {prefix: value}


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _classification(path: str) -> str:
    if path in {"phase", "training_contract.phase"}:
        return EXPECTED_PHASE_CHANGE
    if path in {"actor_update_enabled", "training_contract.actor_update_enabled"}:
        return EXPECTED_ACTOR_ENABLE_CHANGE
    if path.startswith("phase1_training_contract"):
        return EXPECTED_ACTOR_ENABLE_CHANGE
    if path in _INVOCATION_PATHS:
        return INVOCATION_ONLY
    leaf = path.rsplit(".", 1)[-1]
    if leaf.endswith(_KNOWN_IDENTITY_SUFFIXES):
        return IDENTITY
    if path.startswith(_KNOWN_SEMANTIC_PREFIXES):
        return TRAINING_SEMANTIC
    if path in _KNOWN_TOP_LEVEL_FIELDS:
        return TRAINING_SEMANTIC
    return UNKNOWN


def project_awac_training_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Project either CLI or checkpoint config to comparable AWAC semantics.

    Calibration checkpoints historically store the training contract fields at
    the top level of ``resolved_training_config`` while the trainer's current
    CLI resolver wraps them under ``training_contract``.  This projection
    removes that representation difference without changing any value.
    """

    if not isinstance(config, Mapping):
        raise TypeError("AWAC resolved config must be a mapping")
    if isinstance(config.get("training_contract"), Mapping):
        result: Dict[str, Any] = {
            "training_contract": dict(config["training_contract"]),
            "optimization_config": dict(config.get("optimization_config", {})),
        }
    else:
        contract = {
            name: config[name]
            for name in _TRAINING_CONTRACT_FIELDS
            if name in config
        }
        result = {
            "training_contract": contract,
            "optimization_config": dict(config.get("optimization_config", {})),
        }
    for name in ("phase1_training_contract", "phase1_training_contract_sha256"):
        if name in config:
            result[name] = config[name]
    return result


def compare_awac_config_identity(
    calibration_config: Mapping[str, Any],
    handoff_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare two resolved semantic configs and classify every changed field."""

    if not isinstance(calibration_config, Mapping) or not isinstance(
        handoff_config, Mapping
    ):
        raise TypeError("AWAC config identity inputs must be mappings")
    old = _flatten(calibration_config)
    new = _flatten(handoff_config)
    differences = []
    for path in sorted(set(old) | set(new)):
        left = old.get(path, "<missing>")
        right = new.get(path, "<missing>")
        if _values_equal(left, right):
            continue
        differences.append(
            {
                "path": path,
                "calibration": left,
                "handoff": right,
                "classification": _classification(path),
            }
        )
    counts = {
        EXPECTED_PHASE_CHANGE: 0,
        EXPECTED_ACTOR_ENABLE_CHANGE: 0,
        INVOCATION_ONLY: 0,
        IDENTITY: 0,
        TRAINING_SEMANTIC: 0,
        UNKNOWN: 0,
    }
    for difference in differences:
        counts[difference["classification"]] += 1
    return {
        "status": (
            "PASS"
            if counts[TRAINING_SEMANTIC] == 0 and counts[UNKNOWN] == 0
            else "FAIL"
        ),
        "differences": differences,
        "config_diff_count": len(differences),
        "expected_phase_diff_count": counts[EXPECTED_PHASE_CHANGE],
        "expected_actor_enable_diff_count": counts[EXPECTED_ACTOR_ENABLE_CHANGE],
        "expected_invocation_diff_count": counts[INVOCATION_ONLY],
        "identity_diff_count": counts[IDENTITY],
        "training_semantic_unexpected_diff_count": counts[TRAINING_SEMANTIC],
        "unknown_diff_count": counts[UNKNOWN],
    }


__all__ = [
    "EXPECTED_ACTOR_ENABLE_CHANGE",
    "EXPECTED_PHASE_CHANGE",
    "IDENTITY",
    "INVOCATION_ONLY",
    "TRAINING_SEMANTIC",
    "UNKNOWN",
    "compare_awac_config_identity",
    "project_awac_training_config",
]
