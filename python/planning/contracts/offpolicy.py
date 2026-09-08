"""AWAC checkpoint identity and evaluation validation contracts.

The module keeps the historical generic function names used by the evaluator,
but there is exactly one formal off-policy algorithm: discrete AWAC. Unknown
or foreign checkpoint identities fail closed and are never interpreted as BC.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from planning.contracts.collection import canonical_sha256
from planning.awac.contract import (
    AWAC_ALGORITHM_ID,
    AWAC_CHECKPOINT_CONTRACT_ID,
    AWAC_MODEL_TYPE,
    AWAC_TRAINING_CONFIG_CONTRACT_ID,
)
from planning.sac.contract import (
    SAC_ALGORITHM_ID,
    SAC_CHECKPOINT_CONTRACT_ID,
    SAC_MODEL_TYPE,
    SAC_TRAINING_CONFIG_CONTRACT_ID,
)
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.reward import REWARD_CONTRACT_ID
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256
from planning.version import SOFTWARE_VERSION


@dataclass(frozen=True)
class OffPolicyAlgorithmSpec:
    """Immutable identity required to serialize the formal learner."""

    display_name: str
    algorithm_id: str
    checkpoint_contract_field: str
    checkpoint_contract_id: str
    training_config_contract_id: str
    model_type: str


AWAC_OFFPOLICY_SPEC = OffPolicyAlgorithmSpec(
    display_name="AWAC",
    algorithm_id=AWAC_ALGORITHM_ID,
    checkpoint_contract_field="awac_checkpoint_contract_id",
    checkpoint_contract_id=AWAC_CHECKPOINT_CONTRACT_ID,
    training_config_contract_id=AWAC_TRAINING_CONFIG_CONTRACT_ID,
    model_type=AWAC_MODEL_TYPE,
)

SAC_OFFPOLICY_SPEC = OffPolicyAlgorithmSpec(
    display_name="BC-initialized discrete SAC",
    algorithm_id=SAC_ALGORITHM_ID,
    checkpoint_contract_field="sac_checkpoint_contract_id",
    checkpoint_contract_id=SAC_CHECKPOINT_CONTRACT_ID,
    training_config_contract_id=SAC_TRAINING_CONFIG_CONTRACT_ID,
    model_type=SAC_MODEL_TYPE,
)

_OFFPOLICY_SPECS: Dict[str, OffPolicyAlgorithmSpec] = {
    AWAC_ALGORITHM_ID: AWAC_OFFPOLICY_SPEC,
    SAC_ALGORITHM_ID: SAC_OFFPOLICY_SPEC,
}

# This is an identity manifest, not dynamic source discovery. It binds shared
# policy/runtime contracts and the formal AWAC implementations.
OFFPOLICY_SHARED_SOURCE_FILES: Tuple[str, ...] = (
    "config/motion_primitives.yaml",
    "python/planning/bc/model.py",
    "python/planning/contracts/feature.py",
    "python/planning/contracts/offpolicy.py",
    "python/planning/contracts/policy_runtime.py",
    "python/planning/contracts/reward.py",
    "python/planning/contracts/task.py",
    "python/planning/mission/spec.py",
    "python/planning/primitives/library.py",
    "python/planning/runtime/unity_env.py",
    "python/planning/version.py",
    "python/planning/awac/contract.py",
    "python/planning/awac/interaction.py",
    "python/planning/awac/learner.py",
    "python/planning/awac/model.py",
    "python/planning/awac/optimization.py",
    "python/planning/awac/replay.py",
    "python/planning/runtime/parallel_env.py",
)

SAC_OFFPOLICY_SOURCE_FILES: Tuple[str, ...] = (
    "config/motion_primitives.yaml",
    "python/planning/bc/model.py",
    "python/planning/contracts/feature.py",
    "python/planning/contracts/offpolicy.py",
    "python/planning/contracts/policy_runtime.py",
    "python/planning/contracts/reward.py",
    "python/planning/contracts/task.py",
    "python/planning/mission/spec.py",
    "python/planning/primitives/library.py",
    "python/planning/runtime/parallel_env.py",
    "python/planning/runtime/unity_env.py",
    "python/planning/runtime/managed_runtime.py",
    "python/planning/awac/calibration_runtime.py",
    "python/planning/version.py",
    "python/planning/sac/contract.py",
    "python/planning/sac/network.py",
    "python/planning/sac/replay.py",
    "python/planning/sac/checkpoint.py",
    "python/planning/sac/runtime.py",
)


def get_offpolicy_algorithm_spec(algorithm_id: str) -> OffPolicyAlgorithmSpec:
    key = str(algorithm_id)
    try:
        return _OFFPOLICY_SPECS[key]
    except KeyError as exc:
        raise ValueError("unsupported formal off-policy algorithm_id: {!r}".format(key)) from exc


def offpolicy_source_files(algorithm_id: str) -> Tuple[str, ...]:
    spec = get_offpolicy_algorithm_spec(algorithm_id)
    if spec.algorithm_id == SAC_ALGORITHM_ID:
        return tuple(sorted(set(SAC_OFFPOLICY_SOURCE_FILES)))
    return tuple(sorted(set(OFFPOLICY_SHARED_SOURCE_FILES)))


def _source_sha256(package_root: Path, relative_files: Tuple[str, ...]) -> str:
    root = Path(package_root).expanduser().resolve()
    digest = hashlib.sha256()
    for relative in sorted(str(value) for value in relative_files):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("off-policy source-contract file missing: {}".format(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def algorithm_source_sha256(package_root: Path, algorithm_id: str) -> str:
    return _source_sha256(package_root, offpolicy_source_files(algorithm_id))


def is_offpolicy_checkpoint(checkpoint: Mapping) -> bool:
    if not isinstance(checkpoint, Mapping):
        return False
    return bool(
        "actor_state_dict" in checkpoint
        or "algorithm_id" in checkpoint
        or "awac_checkpoint_contract_id" in checkpoint
        or str(checkpoint.get("model_type", "")) == AWAC_MODEL_TYPE
    )


def validate_policy_checkpoint_algorithm(
    checkpoint: Mapping, state_key: str
) -> Optional[OffPolicyAlgorithmSpec]:
    """Validate an AWAC checkpoint; return ``None`` for an ordinary BC model."""
    model_type = str(checkpoint.get("model_type", ""))
    if not bool(
        str(state_key) == "actor_state_dict"
        or is_offpolicy_checkpoint(checkpoint)
        or model_type == AWAC_MODEL_TYPE
    ):
        return None
    return validate_offpolicy_checkpoint_contract(checkpoint)


def validate_offpolicy_checkpoint_contract(
    checkpoint: Mapping,
    *,
    expected_algorithm_id: Optional[str] = None,
) -> OffPolicyAlgorithmSpec:
    """Validate the complete formal AWAC identity and V2 task contract."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("off-policy checkpoint must be a mapping")
    actual_algorithm_id = str(checkpoint.get("algorithm_id", ""))
    if expected_algorithm_id is not None:
        expected_spec = get_offpolicy_algorithm_spec(expected_algorithm_id)
        if actual_algorithm_id != expected_spec.algorithm_id:
            raise ValueError("{} algorithm contract mismatch".format(expected_spec.display_name))
        spec = expected_spec
    else:
        spec = get_offpolicy_algorithm_spec(actual_algorithm_id)
    if checkpoint.get(spec.checkpoint_contract_field) != spec.checkpoint_contract_id:
        raise ValueError("{} checkpoint contract mismatch".format(spec.display_name))
    if checkpoint.get("model_type") not in (None, spec.model_type):
        raise ValueError("{} model type mismatch".format(spec.display_name))
    expected = {
        "software_version": SOFTWARE_VERSION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "reward_contract_id": REWARD_CONTRACT_ID,
        "training_config_contract_id": spec.training_config_contract_id,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError("{} {} mismatch".format(spec.display_name, field))
    if checkpoint.get("privileged_runtime_inputs", []):
        raise ValueError("{} checkpoint requires privileged runtime inputs".format(spec.display_name))
    resolved_config = checkpoint.get("resolved_training_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("{} resolved training configuration is missing".format(spec.display_name))
    if resolved_config.get("contract_id") not in (None, spec.training_config_contract_id):
        raise ValueError("{} resolved training-config contract mismatch".format(spec.display_name))
    if resolved_config.get("algorithm_id") not in (None, spec.algorithm_id):
        raise ValueError("{} resolved algorithm contract mismatch".format(spec.display_name))
    resolved_hash = str(checkpoint.get("resolved_training_config_sha256", ""))
    if len(resolved_hash) != 64 or canonical_sha256(resolved_config) != resolved_hash:
        raise ValueError("{} resolved training-config hash mismatch".format(spec.display_name))
    for field in ("source_code_sha256", "split_manifest_sha256"):
        if len(str(checkpoint.get(field, ""))) != 64:
            raise ValueError("{} {} is missing".format(spec.display_name, field))
    if int(checkpoint.get("vec_dim", -1)) != POLICY_VECTOR_DIM:
        raise ValueError("{} vector dimension mismatch".format(spec.display_name))
    if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
        raise ValueError("{} action count mismatch".format(spec.display_name))
    return spec


def validate_awac_checkpoint_contract(checkpoint: Mapping) -> None:
    validate_offpolicy_checkpoint_contract(checkpoint, expected_algorithm_id=AWAC_ALGORITHM_ID)


__all__ = [
    "AWAC_ALGORITHM_ID",
    "AWAC_CHECKPOINT_CONTRACT_ID",
    "AWAC_MODEL_TYPE",
    "AWAC_OFFPOLICY_SPEC",
    "AWAC_TRAINING_CONFIG_CONTRACT_ID",
    "OFFPOLICY_SHARED_SOURCE_FILES",
    "SAC_OFFPOLICY_SOURCE_FILES",
    "SAC_OFFPOLICY_SPEC",
    "OffPolicyAlgorithmSpec",
    "algorithm_source_sha256",
    "get_offpolicy_algorithm_spec",
    "is_offpolicy_checkpoint",
    "offpolicy_source_files",
    "validate_awac_checkpoint_contract",
    "validate_offpolicy_checkpoint_contract",
    "validate_policy_checkpoint_algorithm",
]
