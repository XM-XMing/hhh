#!/usr/bin/env python3
"""Reproduce the V3 first sentinel crossing and validate V4 backtracking offline.

This command consumes only the immutable V3/V2 evidence chain.  It does not
start ROS, Unity, Bridge, or any environment worker.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from planning.common.hashing import file_sha256
from planning.sac.checkpoint import load_checkpoint
from planning.sac.contract import SAC_KL_BACKTRACKING_FACTORS
from planning.sac.network import DiscreteSACAgent
from planning.sac.replay import SACReplayBuffer


EXPECTED_BC_SHA256 = "ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2"
EXPECTED_PRE = 0.9461058378219604
EXPECTED_POST = 1.0449196100234985


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _restore_agent(agent: Any, payload: Mapping[str, Any]) -> None:
    for name in ("actor", "bc_reference", "critic1", "critic2", "target_critic1", "target_critic2"):
        getattr(agent, name).load_state_dict(payload[name + "_state_dict"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    for name in (
        "actor_optimizer_step_count", "critic_optimizer_step_count", "actor_update_count",
        "critic_update_count", "actor_proposal_count", "actor_rejection_count",
        "actor_recovery_update_count", "nan_count", "invalid_action_count",
    ):
        if name in payload:
            setattr(agent, name, int(payload[name]))


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    v3_dir = args.v3_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    audit = json.loads((v3_dir / "audit_summary.json").read_text(encoding="utf-8"))
    crossing = json.loads((v3_dir / "first_crossing.json").read_text(encoding="utf-8"))
    v3_identity = json.loads((v3_dir / "input_identity.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not audit.get("exact_replay_after_fix"):
        raise RuntimeError("V3 audit is not a passing exact-replay source")
    sentinel = crossing.get("sentinel", {})
    if int(sentinel.get("actor_step", -1)) != 143:
        raise RuntimeError("V3 first sentinel actor step is not 143")
    if abs(float(sentinel.get("kl_before", 0.0)) - EXPECTED_PRE) > 1e-6:
        raise RuntimeError("V3 pre-crossing KL does not match the registered value")
    if abs(float(sentinel.get("kl_after", 0.0)) - EXPECTED_POST) > 1e-6:
        raise RuntimeError("V3 post-crossing KL does not match the registered value")
    root_cause = json.loads((v3_dir / "root_cause.json").read_text(encoding="utf-8"))
    required_causes = {
        "B_GRADUAL_GLOBAL_POLICY_DRIFT",
        "C_LOCALIZED_STATE_POLICY_DRIFT",
        "G_MULTI_TERM_INTERACTION",
        "H_OPTIMIZER_ACCUMULATION",
    }
    observed_causes = set(root_cause.get("root_cause_of_kl_drift", []))
    if not required_causes.issubset(observed_causes):
        raise RuntimeError("V3 root-cause evidence is missing registered causes")

    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from audit_sac_first_kl_crossing import _capture_payload_at_actor_step, _new_agent

    device = torch.device(str(args.device))
    source_run = Path(v3_identity["source_run_dir"]).expanduser().resolve()
    source_config = json.loads((source_run / "training_config.json").read_text(encoding="utf-8"))
    source_input = json.loads((source_run / "input_identity.json").read_text(encoding="utf-8"))
    bc_path = Path(source_input["bc_checkpoint_path"]).expanduser().resolve()
    if file_sha256(bc_path) != EXPECTED_BC_SHA256:
        raise RuntimeError("frozen BC checkpoint SHA mismatch")
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    checkpoint0 = source_run / "checkpoint_step_00000000.pt"
    if not checkpoint0.is_file():
        raise RuntimeError("V2 checkpoint_step_00000000.pt is missing")

    # The V3 helper replays the immutable V2 batch/journal schedule and returns
    # the exact pre-step-143 Agent state plus its recorded actor batch.
    payload, actor_indices = _capture_payload_at_actor_step(
        run_dir=source_run,
        checkpoint_path=checkpoint0,
        config=source_config,
        bc_checkpoint=bc_checkpoint,
        journal=[
            json.loads(line)
            for line in (source_run / "training_step_journal.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        target_step=143,
        torch=torch,
        nn=nn,
        device=device,
    )
    replay = SACReplayBuffer.open(source_run / "replay", read_only=True)
    v3_manifest = json.loads((v3_dir / "sentinel_manifest.json").read_text(encoding="utf-8"))
    sentinel_rows = np.asarray(
        [int(item["replay_row"]) for item in v3_manifest["rows"]], dtype=np.int64
    )
    agent = _new_agent(torch, nn, device, bc_checkpoint, source_config)
    _restore_agent(agent, payload)
    sentinel_batch = replay.batch_from_indices(sentinel_rows, torch=torch, device=device)
    actor_batch = replay.batch_from_indices(actor_indices, torch=torch, device=device)
    agent.set_kl_sentinel(sentinel_batch)
    pre = agent.kl_sentinel_stats()
    result = agent.update_actor_transactional(
        actor_batch,
        hard_stop=1.0,
        entropy_floor=float(source_config.get("entropy_floor", 0.02)),
        trust_region_factors=SAC_KL_BACKTRACKING_FACTORS,
    )
    safe = bool(result.get("accepted")) and float(result.get("sentinel_kl_max", 2.0)) < 1.0
    reproduced = (
        abs(float(pre["kl_max"]) - EXPECTED_PRE) <= 5e-5
        and float(result.get("raw_factor_1_kl_max", 0.0)) >= 1.0
        and safe
    )
    payload_out = {
        "schema_id": "bc_initialized_discrete_sac_kl_trust_region_v4_crossing_regression",
        "status": "PASS" if reproduced else "FAIL",
        "source_v3_dir": str(v3_dir),
        "source_v2_run_dir": str(source_run),
        "source_v3_audit_sha256": file_sha256(v3_dir / "audit_summary.json"),
        "source_v3_first_crossing_sha256": file_sha256(v3_dir / "first_crossing.json"),
        "first_actor_step": 143,
        "expected_v3_pre_kl": EXPECTED_PRE,
        "expected_v3_post_kl": EXPECTED_POST,
        "replayed_pre_sentinel_kl_max": float(pre["kl_max"]),
        "raw_factor_1_kl_max": result.get("raw_factor_1_kl_max"),
        "accepted_factor": result.get("accepted_lr_factor"),
        "accepted_sentinel_kl_max": result.get("accepted_sentinel_kl_max"),
        "accepted_sentinel_kl_mean": result.get("accepted_sentinel_kl_mean"),
        "backtrack_count": result.get("backtrack_count"),
        "factor_results": result.get("factor_results", []),
        "sentinel_state_count": int(sentinel_rows.size),
        "fixed_factors": list(SAC_KL_BACKTRACKING_FACTORS),
        "root_causes": sorted(required_causes),
        "late_detection_root_cause": "I_TRANSACTION_ORDERING_BUG",
        "production_defaults_modified": False,
        "environment_steps": 0,
        "actor_step_reproduced": reproduced,
    }
    _write_json(out_dir / "v3_crossing_regression.json", payload_out)
    replay.close()
    print(json.dumps(_json_safe(payload_out), sort_keys=True, indent=2))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
