"""Compare one deterministic CPU BC training step between O and P."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from planning.contracts.feature import NUM_ACTIONS
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.data.bc_mmap import build_bc_mmap_dataset
from planning.data.rollout import (
    TeacherLabelStore,
    load_rollout_episode,
    read_csv,
    save_rollout_episode,
)
from planning.mission.spec import TASK_CONTRACT_ID, task_contract_sha256
from planning.primitives.generator import generate_library, save_library
from planning.primitives.library import MotionPrimitiveLibrary, load_motion_primitive_config
from planning.safety.depth_mask import DepthActionMaskStore

from test_bc_mmap_observation_provenance import _make_fixture


O_ROOT = Path("/home/xm/XM/xm_ws/src/planning").resolve()
P_ROOT = Path("/home/xm/XM/src").resolve()
TRAIN_ARGS = (
    "--epochs", "1",
    "--batch-size", "2",
    "--val-ratio", "0.25",
    "--seed", "17",
    "--device", "cpu",
    "--num-workers", "0",
    "--prefetch-factor", "1",
    "--cpu-threads", "1",
    "--depth-history-frames", "1",
    "--deployment-safety-mask", "depth",
    "--deployment-execution-mode", "continuous",
    "--loss-mask", "height",
    "--no-amp",
    "--disable-tensorboard",
    "--observation-contract", EXACT_ENDPOINT_OBSERVATION_CONTRACT,
)


RUNNER = r'''
import importlib
import json
from pathlib import Path
import sys
import torch

module_name = sys.argv[1]
optimizer_path = Path(sys.argv[2])
initial_path = Path(sys.argv[3])
order_path = Path(sys.argv[4])
module_args = sys.argv[5:]
module = importlib.import_module(module_name)
torch_module, nn, _, _, _ = module.require_torch()
module.seed_everything(17, torch=torch_module)
initial_model = module.build_model(nn, depth_channels=1)
torch_module.save(initial_model.state_dict(), str(initial_path))

real_adamw = torch_module.optim.AdamW
optimizer_instances = []

class CapturedAdamW(real_adamw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        optimizer_instances.append(self)

torch_module.optim.AdamW = CapturedAdamW
order_records = []
original_make_dataset = module.make_torch_dataset

def recording_make_dataset(base, *args, **kwargs):
    dataset = original_make_dataset(base, *args, **kwargs)
    record = {"phase": "train" if not order_records else "val", "indices": []}
    order_records.append(record)

    class RecordingDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            record["indices"].append(int(index))
            return dataset[index]

    return RecordingDataset()

module.make_torch_dataset = recording_make_dataset
sys.argv = [module_name] + module_args
return_code = module.main()
if not optimizer_instances:
    raise RuntimeError("BC trainer did not create AdamW")
torch_module.save(optimizer_instances[-1].state_dict(), str(optimizer_path))
order_path.write_text(json.dumps(order_records, sort_keys=True), encoding="utf-8")
raise SystemExit(return_code)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_sha256(state) -> str:
    digest = hashlib.sha256()
    def update(value):
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence")
            for item in value:
                update(item)
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))

    update(state)
    return digest.hexdigest()


def _normalizer_sha256(checkpoint) -> str:
    digest = hashlib.sha256()
    for name in ("feature_mean", "feature_std"):
        value = np.asarray(checkpoint[name], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _same_nested(left, right) -> bool:
    if hasattr(left, "detach") and hasattr(right, "detach"):
        return bool(torch_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            _same_nested(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_nested(a, b) for a, b in zip(left, right)
        )
    return left == right


def torch_equal(left, right) -> bool:
    return bool(left.shape == right.shape and left.dtype == right.dtype and left.equal(right))


def _prepare_fixture(root: Path):
    motion_config = deepcopy(load_motion_primitive_config())
    motion_config["paths"] = {
        "motion_primitives_npz": str(root / "motion_primitives_105.npz"),
        "metadata_json": str(root / "motion_primitives_105.json"),
    }
    library = generate_library(motion_config)
    npz_path, metadata_path = save_library(library, motion_config)
    os.environ["PLANNING_MOTION_PRIMITIVES_NPZ"] = str(npz_path)
    os.environ["PLANNING_MOTION_PRIMITIVES_JSON"] = str(metadata_path)
    mpl_hash = MotionPrimitiveLibrary().contract_sha256

    fixture = root / "fixture"
    index_path, labels_path, masks_path, _ = _make_fixture(
        fixture,
        rollout_contracts=(
            EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        ),
    )

    for episode_path in sorted((fixture / "episodes").glob("episode_*.npz")):
        episode = load_rollout_episode(episode_path, validate=True)
        metadata = dict(episode.pop("metadata"))
        metadata["mpl_contract_sha256"] = mpl_hash
        save_rollout_episode(episode_path, episode, metadata)

    with np.load(str(labels_path), allow_pickle=False) as data:
        labels_payload = {key: data[key] for key in data.files if key != "metadata_json"}
        labels_metadata = json.loads(data["metadata_json"].item())
    labels_metadata["mpl_contract_sha256"] = mpl_hash
    labels_payload["metadata_json"] = np.asarray(json.dumps(labels_metadata, sort_keys=True))
    np.savez(str(labels_path), **labels_payload)

    with index_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys()) + [
        "mission_id", "task_contract_id", "task_contract_sha256",
        "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z",
    ]
    for row in rows:
        episode_id = int(row["episode_id"])
        row.update(
            {
                "mission_id": "fixture-mission-{}".format(episode_id),
                "task_contract_id": TASK_CONTRACT_ID,
                "task_contract_sha256": task_contract_sha256(),
                "start_x": "0.0", "start_y": "0.0", "start_z": "1.5",
                "goal_x": "40.0", "goal_y": "0.0", "goal_z": "1.5",
            }
        )
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows = read_csv(index_path)
    cache_dir = root / "cache"
    build_bc_mmap_dataset(
        rows,
        index_path,
        TeacherLabelStore(labels_path),
        cache_dir,
        DepthActionMaskStore(masks_path),
        expected_observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    )
    return index_path, labels_path, masks_path, cache_dir, npz_path, metadata_path


def _run_training(root: Path, label: str, index_path: Path, labels_path: Path, masks_path: Path, cache_dir: Path, npz_path: Path, metadata_path: Path):
    source_root = O_ROOT if label == "O" else P_ROOT
    module_name = "planning.cli.train_soft_bc" if label == "O" else "planning.bc.trainer"
    output = root / ("out_" + label.lower())
    optimizer_path = root / ("optimizer_" + label.lower() + ".pt")
    initial_path = root / ("initial_" + label.lower() + ".pt")
    order_path = root / ("order_" + label.lower() + ".json")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root / "python")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PLANNING_MOTION_PRIMITIVES_NPZ"] = str(npz_path)
    environment["PLANNING_MOTION_PRIMITIVES_JSON"] = str(metadata_path)
    command = [
        sys.executable, "-c", RUNNER, module_name, str(optimizer_path),
        str(initial_path), str(order_path),
        "--index", str(index_path), "--labels", str(labels_path),
        "--depth-action-masks", str(masks_path), "--dataset-cache", str(cache_dir),
        "--out-dir", str(output),
    ] + list(TRAIN_ARGS)
    result = subprocess.run(
        command,
        cwd=str(source_root),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} BC training failed ({}):\n{}\n{}".format(
                label, result.returncode, result.stdout, result.stderr
            )
        )
    return output, optimizer_path, initial_path, order_path


def _load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def main() -> int:
    import torch

    with tempfile.TemporaryDirectory(prefix="xm-m02-bc-training-") as temporary:
        root = Path(temporary)
        fixture = _prepare_fixture(root)
        index_path, labels_path, masks_path, cache_dir, npz_path, metadata_path = fixture
        o_run = _run_training(root, "O", *fixture)
        p_run = _run_training(root, "P", *fixture)
        o_out, o_optimizer, o_initial, o_order = o_run
        p_out, p_optimizer, p_initial, p_order = p_run

        o_checkpoint = _load_checkpoint(o_out / "checkpoint_best.pt")
        p_checkpoint = _load_checkpoint(p_out / "checkpoint_best.pt")
        o_summary = json.loads((o_out / "summary.json").read_text(encoding="utf-8"))
        p_summary = json.loads((p_out / "summary.json").read_text(encoding="utf-8"))
        o_initial_state = _load_checkpoint(o_initial)
        p_initial_state = _load_checkpoint(p_initial)
        o_optimizer_state = _load_checkpoint(o_optimizer)
        p_optimizer_state = _load_checkpoint(p_optimizer)
        o_order_value = json.loads(o_order.read_text(encoding="utf-8"))
        p_order_value = json.loads(p_order.read_text(encoding="utf-8"))

        initial_pass = _tensor_state_sha256(o_initial_state) == _tensor_state_sha256(p_initial_state)
        model_pass = _tensor_state_sha256(o_checkpoint["model_state_dict"]) == _tensor_state_sha256(p_checkpoint["model_state_dict"])
        normalizer_pass = (
            np.array_equal(np.asarray(o_checkpoint["feature_mean"]), np.asarray(p_checkpoint["feature_mean"]))
            and np.array_equal(np.asarray(o_checkpoint["feature_std"]), np.asarray(p_checkpoint["feature_std"]))
            and _normalizer_sha256(o_checkpoint) == _normalizer_sha256(p_checkpoint)
        )
        optimizer_pass = _same_nested(o_optimizer_state, p_optimizer_state)
        order_pass = o_order_value == p_order_value

        metric_fields = (
            "epoch", "train_loss", "train_action_loss", "train_ce", "train_kl",
            "train_teacher_top1", "train_teacher_top5", "train_behavior_top1",
            "train_behavior_top5", "train_samples", "train_ce_samples",
            "train_kl_samples", "train_invalid_teacher", "train_invalid_behavior",
            "train_dropped", "val_loss", "val_action_loss", "val_ce", "val_kl",
            "val_teacher_top1", "val_teacher_top5", "val_behavior_top1",
            "val_behavior_top5", "val_samples", "val_ce_samples", "val_kl_samples",
            "val_invalid_teacher", "val_invalid_behavior", "val_dropped",
        )
        with (o_out / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
            o_metrics = list(csv.DictReader(handle))
        with (p_out / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
            p_metrics = list(csv.DictReader(handle))
        metric_pass = len(o_metrics) == len(p_metrics)
        if metric_pass:
            for o_row, p_row in zip(o_metrics, p_metrics):
                for field in metric_fields:
                    if field == "epoch":
                        metric_pass = metric_pass and int(o_row[field]) == int(p_row[field])
                    else:
                        metric_pass = metric_pass and float(o_row[field]) == float(p_row[field])

        common_checkpoint_fields = (
            "feature_contract_id", "policy_input_contract_sha256", "task_contract_id",
            "task_contract_sha256", "teacher_label_contract_id", "mpl_contract_sha256",
            "vec_dim", "num_actions", "depth_history_frames", "initial_prev_action",
            "observation_contract", "observation_source",
        )
        common_summary_fields = common_checkpoint_fields + (
            "train_episodes", "val_episodes", "train_transitions", "val_transitions",
            "best_soft_epoch", "best_hard_epoch",
        )
        checkpoint_business_pass = all(
            o_checkpoint.get(field) == p_checkpoint.get(field)
            for field in common_checkpoint_fields
        )
        summary_business_pass = all(
            o_summary.get(field) == p_summary.get(field)
            for field in common_summary_fields
        )
        provenance_fields = (
            "observation_contract", "observation_source", "reliable_rows", "legacy_rows",
            "dataset_manifest_sha256", "dataset_manifest_contract_id",
            "source_index_sha256", "source_labels_sha256", "source_depth_masks_sha256",
            "normalizer_sha256", "resolved_training_config_sha256",
        )
        checkpoint_provenance_pass = all(field in p_checkpoint for field in provenance_fields)
        resolved_config = json.loads(
            (p_out / "resolved_training_config.json").read_text(encoding="utf-8")
        )
        checkpoint_provenance_pass = checkpoint_provenance_pass and (
            p_checkpoint["resolved_training_config"] == resolved_config
            and p_checkpoint["resolved_training_config_sha256"] == p_summary["resolved_training_config_sha256"]
        )
        summary_provenance_pass = all(field in p_summary for field in provenance_fields)
        summary_provenance_pass = summary_provenance_pass and (
            p_summary["checkpoint_sha256"] == _sha256(p_out / "checkpoint_best.pt")
            and p_summary["checkpoint_last_sha256"] == _sha256(p_out / "checkpoint_last.pt")
            and p_summary["observation_contract"] == EXACT_ENDPOINT_OBSERVATION_CONTRACT
            and p_summary["reliable_rows"] == 8
            and p_summary["legacy_rows"] == 0
        )
        summary_provenance_pass = summary_provenance_pass and summary_business_pass

        reload_pass = False
        from planning.bc.model import VectorNormalizer, build_model
        from planning.data.bc_mmap import MappedSoftRolloutDataset
        from planning.contracts.feature import POLICY_VECTOR_DIM
        dataset = MappedSoftRolloutDataset(cache_dir, read_csv(index_path), index_path, labels_path, masks_path)
        normalizer = VectorNormalizer.from_checkpoint(p_checkpoint)
        dataset.apply_normalizer(normalizer)
        _, history, vector = dataset.sample(0, 1)
        depth = torch.from_numpy(np.asarray(dataset.depths[history], dtype=np.float32)).unsqueeze(0)
        vector_tensor = torch.from_numpy(vector).reshape(1, POLICY_VECTOR_DIM)
        _, nn, _, _, _ = __import__("planning.bc.model", fromlist=["require_torch"]).require_torch()
        before = build_model(nn, depth_channels=int(p_checkpoint["depth_history_frames"]))
        before.load_state_dict(p_checkpoint["model_state_dict"])
        before.eval()
        with torch.no_grad():
            logits_before = before(depth, vector_tensor)
        after = build_model(nn, depth_channels=int(p_checkpoint["depth_history_frames"]))
        after.load_state_dict(p_checkpoint["model_state_dict"])
        after.eval()
        with torch.no_grad():
            logits_after = after(depth, vector_tensor)
        reload_pass = torch.equal(logits_before, logits_after)

        print("MODEL_TENSOR_SHA_PARITY={}".format("PASS" if model_pass and initial_pass else "FAIL"))
        print("NORMALIZER_PARITY={}".format("PASS" if normalizer_pass else "FAIL"))
        print("OPTIMIZER_STATE_PARITY={}".format("PASS" if optimizer_pass else "FAIL"))
        print("TRAINING_METRIC_PARITY={}".format("PASS" if metric_pass and order_pass else "FAIL"))
        print("CHECKPOINT_PROVENANCE_PARITY={}".format("PASS" if checkpoint_provenance_pass and checkpoint_business_pass else "FAIL"))
        print("SUMMARY_PROVENANCE_PARITY={}".format("PASS" if summary_provenance_pass else "FAIL"))
        print("CHECKPOINT_RELOAD_LOGIT_PARITY={}".format("PASS" if reload_pass else "FAIL"))
        print("TRAIN_ROW_INDICES_PARITY={}".format("PASS" if o_order_value[0] == p_order_value[0] else "FAIL"))
        print("VALIDATION_ROW_INDICES_PARITY={}".format("PASS" if o_order_value[1] == p_order_value[1] else "FAIL"))
        print("BATCH_ORDER_PARITY={}".format("PASS" if order_pass else "FAIL"))
        print("INITIAL_MODEL_TENSOR_SHA256={}".format(_tensor_state_sha256(p_initial_state)))
        print("FINAL_MODEL_TENSOR_SHA256={}".format(_tensor_state_sha256(p_checkpoint["model_state_dict"])))
        print("NORMALIZER_SHA256={}".format(_normalizer_sha256(p_checkpoint)))
        print("OPTIMIZER_STATE_SHA256={}".format(_tensor_state_sha256(p_optimizer_state)))
        print("P_CHECKPOINT_SHA256={}".format(_sha256(p_out / "checkpoint_best.pt")))
        print("P_SUMMARY_SHA256={}".format(_sha256(p_out / "summary.json")))
        print(json.dumps({"checkpoint_business_pass": checkpoint_business_pass, "summary_business_pass": summary_business_pass}, sort_keys=True))
        return 0 if all((model_pass, initial_pass, normalizer_pass, optimizer_pass, metric_pass, order_pass, checkpoint_provenance_pass, summary_provenance_pass, reload_pass)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
