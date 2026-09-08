"""Read-only recovery audit for step-level reward provenance.

The production environment owns reward arithmetic, while collection/evaluation
artifacts own persistence.  This diagnostic joins only artifacts that preserve
episode identity and contiguous step rows.  It deliberately refuses to turn a
Replay storage order, a multi-action branch return, or a terminal flag in an
identity-free array into an episode reward sequence.

No learner, optimizer, Unity runtime, Bridge, production Replay, checkpoint,
or reward contract is modified by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.contracts.reward import reward_contract, reward_contract_sha256


REWARD_PROVENANCE_AUDIT_SCHEMA_ID = "reward_provenance_recovery_audit_v1"
REWARD_SEQUENCE_SCHEMA_ID = "episode_reward_sequence_v1"
CREDIT_METRICS_SCHEMA_ID = "reward_credit_metrics_v1"
DEFAULT_GAMMA = 0.99
DEFAULT_HORIZONS = (5, 10, 20)
TERMINAL_FLAGS = ("success", "collision", "dead_end", "timeout", "far", "hard_altitude")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("reward statistic input contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1.0e-12:
        return None
    return float(np.dot(x, y) / denominator)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _optional_float(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _step_bucket(step: int) -> str:
    if int(step) < 5:
        return "step_0_5"
    if int(step) < 10:
        return "step_5_10"
    if int(step) < 20:
        return "step_10_20"
    return "step_20_plus"


def validate_episode_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the public sequence seam before any return is computed."""

    if not rows:
        raise ValueError("episode has no step rows")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError("episode steps must be contiguous from zero")
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise ValueError("episode reward contains non-finite value")
    done_indices = [index for index, row in enumerate(rows) if bool(row.get("done", False))]
    if len(done_indices) != 1 or done_indices[0] != len(rows) - 1:
        raise ValueError("episode terminal row must be exactly the final row")


def compute_episode_returns(
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float = DEFAULT_GAMMA,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> List[Dict[str, Any]]:
    """Compute observed finite-horizon returns without a value bootstrap.

    ``return_N`` is the undiscounted sum of at most N observed rewards.  The
    companion ``discounted_return_N`` uses ``gamma``.  No value is invented at
    a truncation boundary; the episode-end row is the only terminal boundary.
    """

    validate_episode_rows(rows)
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    horizon_values = tuple(int(value) for value in horizons)
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise ValueError("horizons must be positive")
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    output: List[Dict[str, Any]] = []
    for start in range(len(rows)):
        remaining = rewards[start:]
        powers = np.power(gamma, np.arange(remaining.size, dtype=np.float64))
        item: Dict[str, Any] = {
            "step": int(rows[start]["step"]),
            "undiscounted_return": float(np.sum(remaining)),
            "discounted_return": float(np.dot(powers, remaining)),
            "future_return_after_action_undiscounted": float(np.sum(remaining[1:])),
            "future_return_after_action_discounted": float(
                np.dot(np.power(gamma, np.arange(max(0, remaining.size - 1), dtype=np.float64)), remaining[1:])
            ),
            "bootstrap_status": "NOT_USED_TERMINAL_OR_EPISODE_END",
        }
        for horizon in horizon_values:
            truncated = remaining[:horizon]
            truncated_powers = np.power(gamma, np.arange(truncated.size, dtype=np.float64))
            item["return_{}".format(horizon)] = float(np.sum(truncated))
            item["discounted_return_{}".format(horizon)] = float(
                np.dot(truncated_powers, truncated)
            )
        output.append(item)
    return output


def _component(value: Optional[float], status: str, source: str) -> Dict[str, Any]:
    return {"value": value, "status": status, "source": source}


def build_reward_component_breakdown(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Separate exact terms from terms absent in the persisted step schema."""

    defaults = reward_contract()["defaults"]
    before = _optional_float(row, "distance_before")
    after = _optional_float(row, "distance_after")
    if before is not None and after is not None:
        xy_progress = float(defaults["reward_progress_scale"]) * (before - after)
        xy_status = "EXACT_FROM_STEP_ARTIFACT"
    else:
        xy_progress = None
        xy_status = "UNAVAILABLE"

    previous = row.get("prev_action")
    action = row.get("action")
    action_changed = False
    if previous not in (None, "") and action not in (None, ""):
        action_changed = int(previous) >= 0 and int(previous) != int(action)

    result: Dict[str, Dict[str, Any]] = {
        "xy_progress": _component(xy_progress, xy_status, "reward.py:reward_progress_scale * (distance_before-distance_after)"),
        "goal_z_progress": _component(
            None, "UNAVAILABLE", "step CSV does not persist goal dz before/after"
        ),
        "clearance": _component(
            None, "UNAVAILABLE", "step CSV does not persist min_clearance/clearance_margin"
        ),
        "step": _component(
            float(defaults["reward_step"]), "EXACT_CONTRACT_DEFAULT", "contracts/reward.py"
        ),
        "action_change": _component(
            float(defaults["reward_action_change"]) if action_changed else 0.0,
            "EXACT_FROM_STEP_ARTIFACT",
            "prev_action/action columns",
        ),
    }
    terminal_defaults = {
        "success": float(defaults["reward_success"]),
        "collision": float(defaults["reward_collision"]),
        "altitude": float(defaults["reward_altitude_violation"]),
        "timeout": float(defaults["reward_timeout"]),
        "far": float(defaults["reward_far"]),
        "dead_end": float(defaults["reward_dead_end"]),
    }
    aliases = {
        "success": "terminal_success",
        "collision": "terminal_collision",
        "hard_altitude": "terminal_altitude",
        "timeout": "terminal_timeout",
        "far": "terminal_far",
        "dead_end": "terminal_dead_end",
    }
    for flag, output_name in aliases.items():
        default_key = "altitude" if flag == "hard_altitude" else flag
        result[output_name] = _component(
            terminal_defaults[default_key] if _bool_value(row.get(flag, False)) else 0.0,
            "EXACT_FROM_TERMINAL_FLAG",
            "step CSV terminal flag + contracts/reward.py defaults",
        )

    observed = float(row["reward"])
    known_sum = float(
        sum(
            float(item["value"])
            for item in result.values()
            if item["value"] is not None
        )
    )
    result["observed_reward"] = _component(observed, "EXACT_RECORDED", "step CSV reward")
    result["known_component_sum"] = _component(
        known_sum, "PARTIAL_EXACT", "sum of exact persisted/default terms"
    )
    result["unresolved_residual"] = _component(
        float(observed - known_sum),
        "RESIDUAL_CONTAINS_UNOBSERVED_TERMS",
        "observed reward - known component sum",
    )
    return result


def _terminal_reason(row: Mapping[str, Any]) -> str:
    for key, reason in (
        ("success", "success"),
        ("collision", "collision"),
        ("dead_end", "dead_end"),
        ("timeout", "timeout"),
        ("far", "far"),
        ("hard_altitude", "altitude_violation"),
    ):
        if _bool_value(row.get(key, False)):
            return reason
    return ""


def _episode_file_digest(step_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(step_root.glob("episode_*.csv")):
        files.append(
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update((item["path"] + "\0" + item["sha256"] + "\n").encode("utf-8"))
    return {"file_count": len(files), "files_sha256": digest.hexdigest(), "files": files}


def load_step_collection_artifact(root: Path) -> Dict[str, Any]:
    """Load and validate an existing evaluation collection's step CSVs."""

    root = Path(root)
    index_path = root / "rollout_index.csv"
    step_root = root / "steps"
    if not index_path.exists() or not step_root.is_dir():
        raise FileNotFoundError("collection artifact needs rollout_index.csv and steps/")
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    index_by_episode = {str(row["episode_id"]): row for row in index_rows}
    episodes: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    files = sorted(step_root.glob("episode_*.csv"))
    for path in files:
        episode_id = path.stem.split("episode_", 1)[1].lstrip("0") or "0"
        if episode_id not in index_by_episode:
            problems.append("step file {} has no rollout-index row".format(path.name))
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        rows: List[Dict[str, Any]] = []
        for raw in raw_rows:
            row = dict(raw)
            row["step"] = int(row["step"])
            row["reward"] = float(row["reward"])
            row["done"] = any(_bool_value(row.get(flag, False)) for flag in TERMINAL_FLAGS)
            row["episode_id"] = episode_id
            row["mission_id"] = str(index_by_episode[episode_id]["mission_id"])
            row["terminal_reason"] = _terminal_reason(row)
            rows.append(row)
        try:
            validate_episode_rows(rows)
        except ValueError as exc:
            problems.append("{}: {}".format(path.name, exc))
            continue
        returns = compute_episode_returns(rows)
        for row, return_row in zip(rows, returns):
            row.update(return_row)
            row["source_step_path"] = str(path.relative_to(root))
            row["known_components"] = build_reward_component_breakdown(row)
            flat_rows.append(row)
        episodes.append(
            {
                "episode_id": episode_id,
                "mission_id": str(index_by_episode[episode_id]["mission_id"]),
                "step_count": len(rows),
                "success": _bool_value(index_by_episode[episode_id].get("success", False)),
                "stop_reason": str(index_by_episode[episode_id].get("stop_reason", "")),
                "rows": rows,
            }
        )
    if problems:
        raise ValueError("step collection validation failed: {}".format("; ".join(problems[:5])))
    return {
        "root": str(root),
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "step_files": _episode_file_digest(step_root),
        "episode_count": len(episodes),
        "transition_count": len(flat_rows),
        "episodes": episodes,
        "rows": flat_rows,
        "schema": {
            "reward_t": "PRESENT",
            "terminal_flags": "PRESENT",
            "episode_identity": "PRESENT",
            "step_identity": "PRESENT",
            "done": "DERIVED_FROM_TERMINAL_FLAGS",
            "goal_z_progress": "ABSENT",
            "min_clearance": "ABSENT",
        },
    }


def inspect_multi_action_artifact(root: Path) -> Dict[str, Any]:
    root = Path(root)
    metadata_path = root / "replay" / "metadata.json"
    transitions_path = root / "replay" / "transitions.npz"
    branches_path = root / "candidate_branches.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(transitions_path, allow_pickle=False) as arrays:
        keys = sorted(arrays.files)
        count = int(arrays["reward"].shape[0])
        done_count = int(np.count_nonzero(arrays["done"][:count]))
    branch_count = sum(1 for line in branches_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "source_id": "multi_action_replay_v1_real_20260907",
        "root": str(root),
        "metadata": metadata,
        "artifact_sha256": {
            "metadata": _sha256_file(metadata_path),
            "transitions": _sha256_file(transitions_path),
            "candidate_branches": _sha256_file(branches_path),
        },
        "transition_count": count,
        "done_count": done_count,
        "branch_count": branch_count,
        "npz_fields": keys,
        "field_provenance": {
            "reward_t": {
                "status": "PRESENT_SINGLE_CANDIDATE_STEP",
                "field": "replay/transitions.npz:reward",
                "semantics": "candidate action immediate raw environment reward",
            },
            "terminal_reward": {
                "status": "UNAVAILABLE_AS_SEPARATE_TERM",
                "field": "candidate_branches.jsonl:terminal_reason only",
            },
            "episode_reward_sequence": {
                "status": "UNAVAILABLE",
                "reason": "only candidate reward is persisted; prefix and continuation rewards are not stored",
            },
            "episode_return": {
                "status": "PRESENT_AGGREGATE_ONLY",
                "field": "candidate_branches.jsonl:episode_return",
                "semantics": "prefix + candidate + deterministic BC continuation",
            },
        },
    }


def inspect_calibration_replay(root: Path) -> Dict[str, Any]:
    root = Path(root)
    metadata_path = root / "metadata.json"
    reward_path = root / "reward.npy"
    done_path = root / "done.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    valid_count = int(metadata.get("position", metadata.get("size", 0)))
    rewards = np.load(reward_path, mmap_mode="r")[:valid_count]
    done = np.load(done_path, mmap_mode="r")[:valid_count]
    return {
        "source_id": str(metadata.get("run_identity", "calibration_replay")),
        "root": str(root),
        "metadata": metadata,
        "artifact_sha256": {
            "metadata": _sha256_file(metadata_path),
            "reward": _sha256_file(reward_path),
            "done": _sha256_file(done_path),
        },
        "valid_row_count": valid_count,
        "reward_stats": _finite_stats(rewards.tolist()),
        "done_count": int(np.count_nonzero(done)),
        "field_provenance": {
            "reward_t": {
                "status": "PRESENT_IDENTITY_FREE_REPLAY_ROW",
                "field": "reward.npy",
                "semantics": "raw environment reward per stored transition",
            },
            "done": {
                "status": "PRESENT_IDENTITY_FREE_REPLAY_ROW",
                "field": "done.npy",
            },
            "episode_reward_sequence": {
                "status": "UNAVAILABLE",
                "reason": "no episode_id/step_id/transition identity arrays in calibration Replay",
            },
            "terminal_reward": {
                "status": "UNAVAILABLE_AS_SEPARATE_TERM",
                "reason": "done is present but terminal reason and reward components are absent",
            },
        },
    }


def inspect_teacher_collection_artifact(root: Path) -> Dict[str, Any]:
    root = Path(root)
    index_path = root / "rollout_index.csv"
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    episode_paths = sorted(root.glob("workers/*/episodes/*.npz"))
    key_union = set()
    reliable_rows = 0
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as arrays:
            key_union.update(arrays.files)
            metadata = json.loads(str(arrays["metadata_json"].item()))
            reliable_rows += int(metadata.get("reliable_rows", 0))
    return {
        "source_id": "teacher_recovery_replay_mini_experiment_20260907",
        "root": str(root),
        "artifact_sha256": {"rollout_index": _sha256_file(index_path)},
        "episode_count": len(index_rows),
        "episode_npz_count": len(episode_paths),
        "reliable_rows": reliable_rows,
        "npz_fields": sorted(key_union),
        "field_provenance": {
            "reward_t": {
                "status": "ABSENT",
                "reason": "episode NPZ stores observations/actions/identity but no reward array",
            },
            "episode_reward_sequence": {
                "status": "UNAVAILABLE",
                "reason": "collection report stores aggregate episode metadata only",
            },
            "terminal_reward": {
                "status": "UNAVAILABLE_AS_SEPARATE_TERM",
                "reason": "rollout index stores stop_reason but no per-step reward",
            },
        },
    }


def _correlation_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"row_count": 0, "action_id_vs_return_pearson": None, "action_id_vs_return_spearman": None, "reward_vs_return_pearson": None, "reward_vs_return_spearman": None}
    actions = [float(row["action"]) for row in rows]
    returns = [float(row["undiscounted_return"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    return {
        "row_count": len(rows),
        "action_id_vs_return_pearson": _pearson(actions, returns),
        "action_id_vs_return_spearman": _spearman(actions, returns),
        "reward_vs_return_pearson": _pearson(rewards, returns),
        "reward_vs_return_spearman": _spearman(rewards, returns),
    }


def _component_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [
        "xy_progress",
        "terminal_success",
        "terminal_collision",
        "terminal_dead_end",
        "terminal_timeout",
        "terminal_far",
        "terminal_altitude",
        "step",
        "action_change",
        "unresolved_residual",
    ]
    output: Dict[str, Any] = {}
    observed_abs = float(sum(abs(float(row["reward"])) for row in rows))
    for name in names:
        values = [float(row["known_components"][name]["value"]) for row in rows]
        output[name] = {
            "statistics": _finite_stats(values),
            "signed_sum": float(np.sum(values)) if values else 0.0,
            "absolute_sum": float(np.sum(np.abs(values))) if values else 0.0,
            "absolute_share_of_observed_reward_abs": (
                float(np.sum(np.abs(values)) / observed_abs) if observed_abs > 0.0 else None
            ),
            "status": "PARTIAL_EXACT" if name in ("xy_progress", "unresolved_residual") else "EXACT_FROM_PERSISTED_FIELDS_OR_CONTRACT",
        }
    output["observed_reward"] = {
        "statistics": _finite_stats([float(row["reward"]) for row in rows]),
        "absolute_sum": observed_abs,
        "status": "EXACT_RECORDED",
    }
    return output


def build_credit_metrics(collection: Mapping[str, Any], *, gamma: float = DEFAULT_GAMMA) -> Dict[str, Any]:
    rows = list(collection["rows"])
    by_bucket: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[_step_bucket(int(row["step"]))].append(row)
    per_episode_nonzero = [
        sum(float(row["reward"]) != 0.0 for row in episode["rows"])
        for episode in collection["episodes"]
    ]
    terminal_counts = Counter(str(row["terminal_reason"]) for row in rows if row["done"])
    contributions = _component_summary(rows)
    terminal_component_names = (
        "terminal_success",
        "terminal_collision",
        "terminal_dead_end",
        "terminal_timeout",
        "terminal_far",
        "terminal_altitude",
    )
    terminal_abs_share_sum = float(
        sum(
            contributions[name]["absolute_share_of_observed_reward_abs"] or 0.0
            for name in terminal_component_names
        )
    )
    terminal_episode_rate = float(
        sum(terminal_counts.values()) / collection["episode_count"]
        if collection["episode_count"]
        else 0.0
    )
    non_success_terminal_count = sum(
        terminal_counts.get(name, 0)
        for name in ("collision", "dead_end", "timeout", "far", "hard_altitude", "altitude")
    )
    return {
        "schema_id": CREDIT_METRICS_SCHEMA_ID,
        "status": "PASS_STEP_SEQUENCE_RECOVERED_FROM_EXISTING_COLLECTION_ARTIFACT",
        "source": collection["root"],
        "gamma": float(gamma),
        "episode_count": int(collection["episode_count"]),
        "transition_count": int(collection["transition_count"]),
        "terminal_episode_count": int(sum(1 for episode in collection["episodes"] if episode["rows"][-1]["done"])),
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "reward_density": {
            "per_episode_nonzero_reward_count": _finite_stats(per_episode_nonzero),
            "transition_nonzero_reward_count": int(sum(float(row["reward"]) != 0.0 for row in rows)),
            "transition_nonzero_reward_rate": float(np.mean([float(row["reward"]) != 0.0 for row in rows])),
            "per_episode_step_count": _finite_stats([episode["step_count"] for episode in collection["episodes"]]),
        },
        "credit_correlation": {
            "definition": "descriptive correlation; action_id is nominal and is not a causal/ordinal action embedding",
            "overall": _correlation_block(rows),
            "by_step_bucket": {bucket: _correlation_block(values) for bucket, values in sorted(by_bucket.items())},
        },
        "returns": {
            "undiscounted_G_t": "AVAILABLE_EXACT",
            "discounted_G_t": "AVAILABLE_EXACT",
            "truncated_horizons": {str(value): "AVAILABLE_EXACT_NO_BOOTSTRAP" for value in DEFAULT_HORIZONS},
            "bootstrap_value": "NOT_USED; no value estimate is introduced by this audit",
        },
        "reward_source_contributions": contributions,
        "interpretation": {
            "long_horizon_credit_signal": {
                "status": "SEQUENCE_RECOVERED_BUT_ACTION_LEVEL_CREDIT_NOT_IDENTIFIED",
                "sequence_evidence": "100 existing BC Dev100 episodes have contiguous reward_t and exact no-bootstrap G_t",
                "scope_limit": "calibration Replay and multi-action Replay do not provide an episode-identity-complete sequence",
                "action_level_conclusion": "NOT_ESTABLISHED",
            },
            "terminal_reward_dominance": {
                "status": "NOT_ESTABLISHED",
                "terminal_episode_rate": terminal_episode_rate,
                "non_success_terminal_episode_rate": float(
                    non_success_terminal_count / collection["episode_count"]
                    if collection["episode_count"]
                    else 0.0
                ),
                "terminal_component_absolute_share_sum_descriptive": terminal_abs_share_sum,
                "scope_limit": "absolute component shares are descriptive and non-additive; goal-z/clearance components are unresolved",
            },
            "awac_failure_cause": {
                "status": "UNDETERMINED_FROM_AVAILABLE_REWARD_PROVENANCE",
                "candidates": ["A_REWARD_HORIZON", "B_STATE", "C_ACTION", "D_DATA_COVERAGE"],
                "reason": "the recovered step artifact is not the calibration Replay and has no action-level counterfactual ground truth",
            },
        },
        "step_level_fields": collection["schema"],
    }


def _write_sequence_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "source_step_path",
        "episode_id",
        "mission_id",
        "step",
        "action",
        "reward",
        "done",
        "terminal_reason",
        "undiscounted_return",
        "discounted_return",
        "future_return_after_action_undiscounted",
        "future_return_after_action_discounted",
        "return_5",
        "discounted_return_5",
        "return_10",
        "discounted_return_10",
        "return_20",
        "discounted_return_20",
        "xy_progress_component",
        "terminal_success_component",
        "terminal_collision_component",
        "terminal_dead_end_component",
        "terminal_timeout_component",
        "terminal_altitude_component",
        "unresolved_reward_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            components = row["known_components"]
            writer.writerow(
                {
                    "source_step_path": row["source_step_path"],
                    "episode_id": row["episode_id"],
                    "mission_id": row["mission_id"],
                    "step": row["step"],
                    "action": row["action"],
                    "reward": row["reward"],
                    "done": row["done"],
                    "terminal_reason": row["terminal_reason"],
                    "undiscounted_return": row["undiscounted_return"],
                    "discounted_return": row["discounted_return"],
                    "future_return_after_action_undiscounted": row["future_return_after_action_undiscounted"],
                    "future_return_after_action_discounted": row["future_return_after_action_discounted"],
                    "return_5": row["return_5"],
                    "discounted_return_5": row["discounted_return_5"],
                    "return_10": row["return_10"],
                    "discounted_return_10": row["discounted_return_10"],
                    "return_20": row["return_20"],
                    "discounted_return_20": row["discounted_return_20"],
                    "xy_progress_component": components["xy_progress"]["value"],
                    "terminal_success_component": components["terminal_success"]["value"],
                    "terminal_collision_component": components["terminal_collision"]["value"],
                    "terminal_dead_end_component": components["terminal_dead_end"]["value"],
                    "terminal_timeout_component": components["terminal_timeout"]["value"],
                    "terminal_altitude_component": components["terminal_altitude"]["value"],
                    "unresolved_reward_residual": components["unresolved_residual"]["value"],
                }
            )


def _write_report(path: Path, provenance: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    multi = provenance["sources"]["multi_action_replay"]
    calibration = provenance["sources"]["calibration_replay"]
    teacher = provenance["sources"]["teacher_collection"]
    recovered = provenance["sources"]["recovered_step_collection"]
    density = metrics["reward_density"]
    contributions = metrics["reward_source_contributions"]
    terminal = metrics["terminal_reason_counts"]
    lines = [
        "# Reward Provenance Recovery Audit V1",
        "",
        "## 结论",
        "",
        "本审计未训练 AWAC/Actor/Critic，未启动 Unity/Bridge/ROS，未修改 reward 或 Replay。",
        "在既有 BC Dev100 collection artifact 的 `steps/episode_*.csv` 中恢复了 "
        "{} 个 episode、{} 条连续 step reward。".format(
            recovered["episode_count"], recovered["transition_count"]
        ),
        "因此该 artifact 支持对这 100 个既有 episode 计算观测到的 G_t 和有限 horizon return；"
        "它不是 calibration Replay 的逐行身份恢复。",
        "",
        "Calibration Replay 和 multi-action replay 均不能独立恢复合法的 episode reward sequence："
        "前者没有 episode/step identity，后者只保存候选动作即时 reward 和含 prefix/continuation 的汇总 `episode_return`。",
        "",
        "## 输入边界",
        "",
        "| 来源 | reward_t | episode sequence | terminal/component provenance |",
        "|---|---|---|---|",
        "| multi-action replay | 候选动作单步存在 | 不可用 | terminal_reason 有，独立 terminal reward 不有 |",
        "| calibration Replay | raw row reward 存在 | 不可用：无 episode/step identity | done 有，reason/component 无 |",
        "| teacher recovery collection | NPZ 无 reward 字段 | 不可用 | report 只有汇总 steps/stop_reason |",
        "| recovered BC Dev100 steps | 逐步 reward 存在 | 可用且连续 | terminal flags 可用；goal-z/clearance 分量缺失 |",
        "",
        "## 恢复序列统计",
        "",
        "- episodes: `{}`; transitions: `{}`。".format(metrics["episode_count"], metrics["transition_count"]),
        "- terminal reasons: `{}`。".format(json.dumps(terminal, ensure_ascii=False, sort_keys=True)),
        "- 每 episode 非零 reward 数：`{}`。".format(json.dumps(density["per_episode_nonzero_reward_count"], ensure_ascii=False)),
        "- 非零 reward transition 比例：`{:.6f}`。".format(density["transition_nonzero_reward_rate"]),
        "- 计算了 undiscounted G_t、gamma=0.99 discounted G_t，以及 5/10/20-step 截断回报；没有引入 bootstrap value。",
        "",
        "## reward source 限制",
        "",
        "生产合同 `terminal_failure_dominates_progress` 的默认 terminal bonus/penalty 已由源码确认："
        "success=+30，collision/dead_end/timeout/far/altitude=-100，step=-0.02。",
        "现有 step CSV 可精确给出 XY progress、step、action-change 和 terminal flag 的已知部分；"
        "goal-z progress 与 clearance 未持久化，因此 `unresolved_reward_residual` 不能归因给某一个分量。",
        "Terminal 贡献的 absolute share 仅是观测 reward absolute sum 的描述性比例，不是因果归因。",
        "按观测 reward absolute sum 的描述性比例：XY progress(部分可解释)={:.6f}，"
        "success={:.6f}，collision={:.6f}，dead_end={:.6f}，timeout={:.6f}，"
        "unresolved residual={:.6f}。".format(
            contributions["xy_progress"]["absolute_share_of_observed_reward_abs"],
            contributions["terminal_success"]["absolute_share_of_observed_reward_abs"],
            contributions["terminal_collision"]["absolute_share_of_observed_reward_abs"],
            contributions["terminal_dead_end"]["absolute_share_of_observed_reward_abs"],
            contributions["terminal_timeout"]["absolute_share_of_observed_reward_abs"],
            contributions["unresolved_residual"]["absolute_share_of_observed_reward_abs"],
        ),
        "",
        "## 三个问题的明确回答",
        "",
        "1. **reward 是否提供长期 credit signal？**：序列层面 `YES_FOR_RECOVERED_ARTIFACT`；"
        "但 action-level credit assignment `NOT_ESTABLISHED`。这 100 个既有 BC Dev100 episode 可计算完整 G_t，"
        "不能替代 calibration Replay 的 action/state 对齐证据。",
        "2. **terminal reward 是否主导？**：`NOT_ESTABLISHED`。terminal flags/合同项确实是重要信号，"
        "但 component absolute share 是描述性且非可加的，且 goal-z/clearance 尚未持久化，不能据此宣称 terminal dominance。",
        "3. **AWAC 失败更可能 A/B/C/D？**：`UNDETERMINED`。当前 provenance 不能在 A reward horizon、"
        "B state、C action、D data coverage 之间做证据充分的排序；本轮不做因果归因。",
        "",
        "## credit assignment 解读",
        "",
        "action_id 与 return 的 Pearson/Spearman 仅作描述性输出；105 个 action id 是名义类别，"
        "不能将其相关性解释为 action quality 或因果 credit。更可信的边界是：逐步 reward 与完整后续回报均已恢复，"
        "但 observation artifact 没有保存完整 reward component provenance。",
        "",
        "因此本轮不能仅凭该离线集合证明长期 credit assignment 充分，也不能把 terminal dominance 作为唯一根因。"
        "下一步若需生产级 attribution，应在不改变 reward 数值的前提下让 collection artifact 持久化 reward_t、"
        "done_reason、progress/z-progress、clearance 和 terminal component fields。",
        "",
        "## 证据身份",
        "",
        "- multi-action root: `{}`".format(multi["root"]),
        "- calibration replay: `{}`".format(calibration["root"]),
        "- teacher collection: `{}`".format(teacher["root"]),
        "- recovered step collection: `{}`".format(recovered["root"]),
        "- `episode_reward_sequences.csv` 仅写入恢复且通过连续性验证的 BC Dev100 step rows。",
        "",
        "## 运行边界",
        "",
        "`training_executed=false`, `actor_updates=0`, `critic_updates=0`, `unity_started=false`, "
        "`bridge_started=false`, `reward_modified=false`, `replay_modified=false`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reward_provenance_audit(
    *,
    multi_action_root: Path,
    calibration_replay_root: Path,
    teacher_collection_root: Path,
    recovered_step_collection_root: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run the read-only audit and publish only derived diagnostic artifacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    multi = inspect_multi_action_artifact(Path(multi_action_root))
    calibration = inspect_calibration_replay(Path(calibration_replay_root))
    teacher = inspect_teacher_collection_artifact(Path(teacher_collection_root))
    recovered = load_step_collection_artifact(Path(recovered_step_collection_root))
    metrics = build_credit_metrics(recovered)
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "reward.py"
    provenance = {
        "schema_id": REWARD_PROVENANCE_AUDIT_SCHEMA_ID,
        "status": "PASS_WITH_CALIBRATION_IDENTITY_BOUNDARY_EXPLICIT",
        "reward_contract": {
            "contract": reward_contract(),
            "sha256": reward_contract_sha256(),
            "source_path": str(contract_path),
            "source_sha256": _sha256_file(contract_path),
        },
        "sources": {
            "multi_action_replay": multi,
            "calibration_replay": calibration,
            "teacher_collection": teacher,
            "recovered_step_collection": {
                key: value
                for key, value in recovered.items()
                if key not in ("episodes", "rows")
            },
        },
        "field_provenance_table": {
            "reward_t": {
                "production_owner": "planning.runtime.unity_env.UnityForestEnv.step_primitive",
                "arithmetic_owner": "planning.contracts.reward.compute_reward",
                "recovered_from": "existing BC Dev100 steps/*.csv",
                "calibration_replay_status": "PRESENT_BUT_IDENTITY_FREE",
            },
            "terminal_reward": {
                "status": "PARTIALLY_RECOVERED",
                "exact": "terminal flag plus reward contract default",
                "missing": "separate persisted component field",
            },
            "success_reward": {"status": "EXACT_CONTRACT_DEFAULT_PLUS_SUCCESS_FLAG", "value": 30.0},
            "failure_penalty": {"status": "EXACT_CONTRACT_DEFAULTS", "value": -100.0},
            "collision_penalty": {"status": "EXACT_CONTRACT_DEFAULT_PLUS_COLLISION_FLAG", "value": -100.0},
            "dead_end_penalty": {"status": "EXACT_CONTRACT_DEFAULT_PLUS_DEAD_END_FLAG", "value": -100.0},
            "timeout_penalty": {"status": "EXACT_CONTRACT_DEFAULT_PLUS_TIMEOUT_FLAG", "value": -100.0},
            "progress": {"status": "PARTIAL_XY_ONLY", "missing": "goal-z progress field"},
            "clearance": {"status": "UNAVAILABLE", "missing": "min_clearance and clearance margin in step artifact"},
        },
        "sequence_recovery": {
            "status": "RECOVERED_FROM_EXISTING_COLLECTION_ARTIFACT",
            "exact_sequence_available": True,
            "source_is_calibration_replay": False,
            "episode_count": int(recovered["episode_count"]),
            "transition_count": int(recovered["transition_count"]),
            "validation": "contiguous step ids, finite rewards, exactly one terminal row at episode tail",
        },
        "runtime_guards": {
            "training_executed": False,
            "actor_updates": 0,
            "critic_updates": 0,
            "unity_started": False,
            "bridge_started": False,
            "ros_started": False,
            "reward_modified": False,
            "replay_modified": False,
        },
    }
    _json_dump(out_dir / "reward_provenance.json", provenance)
    _write_sequence_csv(out_dir / "episode_reward_sequences.csv", recovered["rows"])
    _json_dump(out_dir / "credit_metrics.json", metrics)
    _write_report(out_dir / "report_zh.md", provenance, metrics)
    _json_dump(
        out_dir / "manifest.json",
        {
            "schema_id": REWARD_PROVENANCE_AUDIT_SCHEMA_ID,
            "artifacts": {
                name: {"bytes": int((out_dir / name).stat().st_size), "sha256": _sha256_file(out_dir / name)}
                for name in ("reward_provenance.json", "episode_reward_sequences.csv", "credit_metrics.json", "report_zh.md")
            },
            "input_sequence_source": str(recovered_step_collection_root),
            "production_modified": False,
        },
    )
    return {
        "out_dir": str(out_dir),
        "provenance": provenance,
        "metrics": metrics,
    }


__all__ = [
    "CREDIT_METRICS_SCHEMA_ID",
    "REWARD_PROVENANCE_AUDIT_SCHEMA_ID",
    "REWARD_SEQUENCE_SCHEMA_ID",
    "build_credit_metrics",
    "build_reward_component_breakdown",
    "compute_episode_returns",
    "inspect_calibration_replay",
    "inspect_multi_action_artifact",
    "inspect_teacher_collection_artifact",
    "load_step_collection_artifact",
    "run_reward_provenance_audit",
    "validate_episode_rows",
]
