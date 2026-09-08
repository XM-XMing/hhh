"""Read-only evidence report for one formal AWAC run.

The diagnostic consumes an AWAC checkpoint and its persistent replay.  It does
not select checkpoints, alter the replay, launch Unity, or implement training
policy.  Historical candidate-selection and entropy diagnostics intentionally
have no representation in this owner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from planning.awac.checkpoint import validate_awac_checkpoint_payload
from planning.awac.model import build_actor, build_critic, masked_policy
from planning.awac.replay import AWACReplayBuffer
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256
from planning.contracts.reward import REWARD_CONTRACT_ID


def _json(path: Path) -> Dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object: {}".format(path))
    return payload


def _training_events(path: Path) -> list:
    events = []
    if not Path(path).is_file():
        return events
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid training event line {}: {}".format(line_number, path)
                ) from exc
            if isinstance(payload, Mapping):
                events.append(dict(payload))
    return events


def _last_finite(events: Sequence[Mapping], names: Iterable[str]):
    for event in reversed(events):
        for name in names:
            value = event.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def _quantiles(values: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("diagnostic values contain non-finite numbers")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "abs_p95": float(np.quantile(np.abs(values), 0.95)),
    }


def _replay_evidence(replay_dir: Path):
    replay = AWACReplayBuffer.open(replay_dir)
    size = int(replay.size)
    reward = np.asarray(replay.arrays["reward"][:size], dtype=np.float64)
    done = np.asarray(replay.arrays["done"][:size], dtype=np.bool_)
    behavior = np.asarray(replay.arrays["behavior_source"][:size], dtype=np.int64)
    if not np.isfinite(reward).all():
        raise ValueError("AWAC replay contains non-finite rewards")
    unique, counts = np.unique(behavior, return_counts=True)
    return replay, {
        "size": size,
        "done_rate": float(done.mean()) if size else 0.0,
        "reward": _quantiles(reward),
        "terminal_reward": _quantiles(reward[done]),
        "nonterminal_reward": _quantiles(reward[~done]),
        "behavior_source_counts": {
            str(int(source)): int(count)
            for source, count in zip(unique.tolist(), counts.tolist())
        },
    }


def _checkpoint_path(run_dir: Path) -> Path:
    candidates = (
        Path(run_dir) / "checkpoint_latest.pt",
        Path(run_dir) / "checkpoint_smoke.pt",
        Path(run_dir) / "learner" / "checkpoint_latest.pt",
        Path(run_dir) / "learner" / "checkpoint_final.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("AWAC checkpoint is missing under {}".format(run_dir))


def _replay_path(run_dir: Path) -> Path:
    candidates = (Path(run_dir) / "replay", Path(run_dir) / "learner" / "replay")
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    raise FileNotFoundError("AWAC replay metadata is missing under {}".format(run_dir))


def _model_evidence(
    checkpoint: Mapping,
    checkpoint_path: Path,
    replay: AWACReplayBuffer,
    *,
    torch,
    max_rows: int,
) -> Dict:
    validate_awac_checkpoint_payload(checkpoint)
    device = torch.device("cpu")
    depth_channels = int(checkpoint["depth_history_frames"])
    actor = build_actor(torch.nn, depth_channels=depth_channels).to(device).eval()
    critic1 = build_critic(torch.nn, depth_channels=depth_channels).to(device).eval()
    critic2 = build_critic(torch.nn, depth_channels=depth_channels).to(device).eval()
    target1 = build_critic(torch.nn, depth_channels=depth_channels).to(device).eval()
    target2 = build_critic(torch.nn, depth_channels=depth_channels).to(device).eval()
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic1.load_state_dict(checkpoint["critic1_state_dict"])
    critic2.load_state_dict(checkpoint["critic2_state_dict"])
    target1.load_state_dict(checkpoint["target_critic1_state_dict"])
    target2.load_state_dict(checkpoint["target_critic2_state_dict"])

    size = min(int(replay.size), int(max_rows))
    indices = np.arange(size, dtype=np.int64)
    q_values = []
    targets = []
    gamma = float(checkpoint["resolved_training_config"].get("gamma", 0.99))
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = replay.sample_indices(indices[start : start + 512], torch=torch, device=device)
            q1_all = critic1(batch["depth"], batch["vector"])
            q2_all = critic2(batch["depth"], batch["vector"])
            minimum_q = torch.minimum(q1_all, q2_all)
            q_values.append(
                minimum_q.gather(1, batch["action"][:, None]).squeeze(1).cpu().numpy()
            )
            probabilities, _, _ = masked_policy(
                actor(batch["next_depth"], batch["next_vector"]),
                batch["next_action_mask"],
                torch,
            )
            target_q = torch.minimum(
                target1(batch["next_depth"], batch["next_vector"]),
                target2(batch["next_depth"], batch["next_vector"]),
            )
            next_value = (probabilities * target_q).sum(dim=1)
            targets.append(
                (
                    batch["reward"]
                    + gamma * (1.0 - batch["done"]) * next_value
                ).cpu().numpy()
            )
    q_array = np.concatenate(q_values) if q_values else np.empty(0, dtype=np.float32)
    target_array = np.concatenate(targets) if targets else np.empty(0, dtype=np.float32)
    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": file_sha256(checkpoint_path),
            "algorithm_id": checkpoint.get("algorithm_id"),
            "global_step": int(checkpoint.get("global_step", 0)),
            "update_step": int(checkpoint.get("update_step", 0)),
            "reward_contract_id": checkpoint.get("reward_contract_id"),
        },
        "sample_rows": int(len(q_array)),
        "executed_q": _quantiles(q_array),
        "bellman_target": _quantiles(target_array),
        "td_residual": _quantiles(q_array - target_array),
    }


def diagnose(run_dir: Path, *, max_model_rows: int = 4096) -> Dict:
    """Build a read-only AWAC evidence report from one run directory."""
    root = Path(run_dir).expanduser().resolve()
    if int(max_model_rows) <= 0:
        raise ValueError("max-model-rows must be positive")
    checkpoint_path = _checkpoint_path(root)
    replay_path = _replay_path(root)
    import torch

    checkpoint = load_torch(checkpoint_path, torch=torch, map_location="cpu")
    validate_awac_checkpoint_payload(checkpoint)
    replay, replay_evidence = _replay_evidence(replay_path)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        summary_path = root / "learner" / "summary.json"
    summary = _json(summary_path) if summary_path.is_file() else {}
    events_path = root / "events.jsonl"
    if not events_path.is_file():
        events_path = root / "learner" / "events.jsonl"
    events = _training_events(events_path)
    model = _model_evidence(
        checkpoint,
        checkpoint_path,
        replay,
        torch=torch,
        max_rows=int(max_model_rows),
    )
    contract_checks = {
        "algorithm_id": checkpoint.get("algorithm_id") == "discrete_masked_awac",
        "reward_contract_id": checkpoint.get("reward_contract_id") == REWARD_CONTRACT_ID,
        "observation_contract": checkpoint.get("observation_contract")
        == checkpoint.get("observation_source"),
        "replay_size": int(checkpoint.get("replay_size", -1)) == int(replay.size),
    }
    passed = all(contract_checks.values())
    return {
        "verdict": "GREEN" if passed else "RED",
        "run_dir": str(root),
        "algorithm_id": checkpoint.get("algorithm_id"),
        "global_step": int(checkpoint.get("global_step", 0)),
        "latest_event": events[-1] if events else None,
        "summary": summary,
        "replay": replay_evidence,
        "model": model,
        "contract_checks": contract_checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--max-model-rows", type=int, default=4096)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = diagnose(args.run, max_model_rows=int(args.max_model_rows))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "AWAC_DIAGNOSIS verdict={} run={} global_step={}".format(
                report["verdict"], report["run_dir"], report["global_step"]
            )
        )
        print("replay_size={}".format(report["replay"]["size"]))
        print("RESULT={}".format(report["verdict"]))
    return 0 if report["verdict"] == "GREEN" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["diagnose", "main"]
