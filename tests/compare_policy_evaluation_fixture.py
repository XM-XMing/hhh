"""Compare O/P evaluator policy inputs and actions on one fixed fixture.

This is intentionally a no-Unity, no-training smoke: it validates checkpoint
loading, tensor construction, raw/masked logits, and deterministic action
selection. Episode outcomes and returns therefore remain NOT_APPLICABLE.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from planning.bc.model import build_model, require_torch
from planning.contracts.feature import (
    CONTINUOUS_DIM,
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.mission.spec import TASK_CONTRACT_ID, task_contract_sha256
from planning.primitives.generator import generate_library, save_library
from planning.primitives.library import MotionPrimitiveLibrary, load_motion_primitive_config


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()
SUMMARY_BUSINESS_FIELDS = (
    "evaluation_contract_id",
    "episodes",
    "max_steps",
    "success_rate",
    "collision_rate",
    "dead_end_rate",
    "timeout_rate",
    "far_rate",
    "episode_return_mean",
    "episode_return_median",
    "final_distance_mean",
    "policy_runtime_contract_id",
    "safety_mask",
    "execution_mode",
    "reliable_v4",
    "policy_action_mode",
    "policy_temperature",
    "quality_gate_applicable",
    "task_contract_id",
    "task_contract_sha256",
    "checkpoint_sha256",
    "mission_index_sha256",
    "algorithm_id",
    "model_type",
    "rollout_csv",
)


RUNNER = r'''
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np


def _install_ros_import_stubs():
    """Allow evaluator policy helpers to load without starting ROS."""
    rospy = types.ModuleType("rospy")
    rospy.core = types.SimpleNamespace(is_initialized=lambda: False)
    sys.modules.setdefault("rospy", rospy)
    for package, names in (
        ("geometry_msgs.msg", ("PoseStamped", "Twist")),
        ("sensor_msgs.msg", ("CameraInfo", "Image")),
        ("std_msgs.msg", ("Bool",)),
        ("planning.msg", ("PrimitiveExecution", "XMState")),
    ):
        module = types.ModuleType(package)
        for name in names:
            setattr(module, name, type(name, (), {}))
        sys.modules.setdefault(package, module)
        parent_name = package.split(".")[0]
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, "msg", module)


source_root = Path(sys.argv[1]).resolve()
checkpoint_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source_root / "python"))
import planning
_install_ros_import_stubs()
module_name = (
    "planning.cli.evaluate_policy_unity"
    if sys.argv[3] == "O"
    else "planning.evaluation.policy_evaluator"
)
module = __import__(module_name, fromlist=["*"])
torch, nn, _, _, _ = module.require_torch()
try:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
provenance_validated = False
if sys.argv[3] == "P":
    from planning.evaluation.observation_provenance import (
        resolve_evaluation_observation_provenance,
    )
    resolve_evaluation_observation_provenance(
        checkpoint,
        runtime_observation_contract="reliable_exact_endpoint_snapshot",
        runtime_observation_source="reliable_exact_endpoint_snapshot",
        reliable_execution_enabled=True,
        telemetry_observation_enabled=False,
        telemetry_fallback_enabled=False,
        state_depth_exact_endpoint_binding=True,
    )
    provenance_validated = True
normalizer = module.VectorNormalizer.from_checkpoint(checkpoint)
model = module.build_model(nn, depth_channels=int(checkpoint["depth_history_frames"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

obs = {
    "state": {
        "position": np.asarray([1.0, 2.0, 1.6], dtype=np.float32),
        "velocity": np.asarray([0.1, -0.2, 0.05], dtype=np.float32),
        "yaw": 0.3,
        "z_to_min": 0.6,
        "z_to_max": 3.4,
    },
    "goal": {
        "relative": np.asarray([12.0, -2.0, 0.1], dtype=np.float32),
        "direction_xy": np.asarray([0.98, -0.17], dtype=np.float32),
        "direction_body_xy": np.asarray([0.91, -0.41], dtype=np.float32),
        "distance_xy": 12.165525,
        "distance_xy_norm40": 0.30413812,
        "dz": 0.1,
    },
}
depth = np.linspace(0.2, 9.5, num=90 * 160, dtype=np.float32).reshape(90, 160)
obs["depth"] = depth
previous_action = -1
mask = np.ones((105,), dtype=np.bool_)
mask[[1, 7, 99]] = False
depth_tensor, vector_tensor = module.observation_tensors(
    obs, previous_action, normalizer, torch, torch.device("cpu"), [depth]
)
mask_tensor = torch.from_numpy(mask.reshape(1, -1))
with torch.no_grad():
    raw_logits = model(depth_tensor, vector_tensor)
    masked_logits = module.mask_logits(raw_logits, mask_tensor)
raw = raw_logits[0].cpu().numpy().astype(np.float32)
masked = masked_logits[0].cpu().numpy().astype(np.float32)
action_result = module.choose_action(
    model,
    obs,
    previous_action,
    mask,
    normalizer,
    torch,
    torch.device("cpu"),
    0.0,
    [depth],
    return_diagnostics=True,
    return_full_logits=True,
)


def fp(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_fp(value):
    digest = hashlib.sha256()
    for key in sorted(value):
        array = value[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalizer_fp(checkpoint):
    digest = hashlib.sha256()
    for name in ("feature_mean", "feature_std"):
        array = np.ascontiguousarray(np.asarray(checkpoint[name], dtype=np.float32))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


print(json.dumps({
    "label": sys.argv[3],
    "checkpoint_load": True,
    "provenance_validated": provenance_validated,
    "model_state_sha256": state_fp(checkpoint["model_state_dict"]),
    "normalizer_sha256": normalizer_fp(checkpoint),
    "depth_tensor_sha256": fp(depth_tensor.cpu().numpy()),
    "vector_tensor_sha256": fp(vector_tensor.cpu().numpy()),
    "raw_logits_sha256": fp(raw),
    "masked_logits_sha256": fp(masked),
    "raw_action": int(torch.argmax(raw_logits, dim=1).item()),
    "mask_sha256": fp(mask),
    "masked_action": int(torch.argmax(masked_logits, dim=1).item()),
    "chosen_action": int(action_result[0]),
    "confidence": float(action_result[1]),
    "top5": action_result[2],
}, sort_keys=True))
'''


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_business_schema_parity() -> bool:
    sources = (
        O_ROOT / "python/planning/cli/evaluate_policy_unity.py",
        P_ROOT / "python/planning/evaluation/policy_evaluator.py",
    )
    return all(
        all('"{}"'.format(field) in path.read_text(encoding="utf-8") for field in SUMMARY_BUSINESS_FIELDS)
        for path in sources
    )


def _make_checkpoint(path: Path) -> None:
    torch, nn, _, _, _ = require_torch()
    motion_config = deepcopy(load_motion_primitive_config())
    motion_config["paths"] = {
        "motion_primitives_npz": str(path.parent / "motion_primitives_105.npz"),
        "metadata_json": str(path.parent / "motion_primitives_105.json"),
    }
    npz_path, metadata_path = save_library(
        generate_library(motion_config), motion_config
    )
    os.environ["PLANNING_MOTION_PRIMITIVES_NPZ"] = str(npz_path)
    os.environ["PLANNING_MOTION_PRIMITIVES_JSON"] = str(metadata_path)
    torch.manual_seed(17)
    model = build_model(nn, depth_channels=1)
    checkpoint = {
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
        "feature_mean": np.zeros((CONTINUOUS_DIM,), dtype=np.float32),
        "feature_std": np.ones((CONTINUOUS_DIM,), dtype=np.float32),
        "model_type": "bc_fixture",
        "model_state_dict": model.state_dict(),
        "mpl_contract_sha256": MotionPrimitiveLibrary().contract_sha256,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    }
    torch.save(checkpoint, str(path))


def _run(label: str, root: Path, checkpoint: Path) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        "/home/xm/XM/xm_ws/devel/lib/python3/dist-packages:"
        "/opt/ros/noetic/lib/python3/dist-packages"
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(root), str(checkpoint), label],
        cwd=str(root),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} evaluator fixture failed ({}):\n{}\n{}".format(
                label, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="xm-m03-eval-") as temporary:
        root = Path(temporary)
        checkpoint = root / "fixed_reliable_exact_checkpoint.pt"
        _make_checkpoint(checkpoint)
        p_result = _run("P", P_ROOT, checkpoint)
        o_result = _run("O", O_ROOT, checkpoint)

        keys = (
            "model_state_sha256",
            "normalizer_sha256",
            "depth_tensor_sha256",
            "vector_tensor_sha256",
            "raw_logits_sha256",
            "masked_logits_sha256",
            "mask_sha256",
            "raw_action",
            "masked_action",
            "chosen_action",
            "top5",
        )
        parity = {key: p_result[key] == o_result[key] for key in keys}
        all_policy_parity = all(parity.values())
        summary_business_pass = _summary_business_schema_parity()
        print("FIXTURE_CHECKPOINT_SHA256={}".format(_file_sha256(checkpoint)))
        print("P_CHECKPOINT_EVALUATOR_LOAD={}".format(
            "PASS" if p_result["checkpoint_load"] and p_result["provenance_validated"] else "FAIL"
        ))
        print("LOGIT_PARITY={}".format(
            "PASS" if parity["raw_logits_sha256"] and parity["masked_logits_sha256"] else "FAIL"
        ))
        print("RAW_ACTION_PARITY={}".format(
            "PASS" if parity["raw_action"] else "FAIL"
        ))
        print("MASK_PARITY={}".format(
            "PASS" if parity["mask_sha256"] else "FAIL"
        ))
        print("MASKED_ACTION_PARITY={}".format(
            "PASS" if parity["masked_action"] and parity["chosen_action"] else "FAIL"
        ))
        print("OUTCOME_PARITY=NOT_APPLICABLE")
        print("RETURN_PARITY=NOT_APPLICABLE")
        print("EVALUATION_SUMMARY_BUSINESS_PARITY={}".format(
            "PASS" if summary_business_pass else "FAIL"
        ))
        print("POLICY_FIXTURE_PARITY={}".format("PASS" if all_policy_parity else "FAIL"))
        print(json.dumps({"O": o_result, "P": p_result, "parity": parity}, sort_keys=True))
        return 0 if all_policy_parity and summary_business_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
