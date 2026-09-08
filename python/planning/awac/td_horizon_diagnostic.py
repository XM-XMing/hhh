"""Isolated helpers for the finite-horizon Critic diagnostic.

This module is deliberately not part of the production AWAC update path.  It
owns only the diagnostic representation of a known task budget, contiguous
Replay sequences, an expected TD target, and a zero-initialized Critic budget
column.  The deployable Actor/BC vector remains the 127-dimensional contract.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from planning.awac.model import build_critic
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM


DIAGNOSTIC_CRITIC_VECTOR_DIM = int(POLICY_VECTOR_DIM) + 1
DIAGNOSTIC_CRITIC_SCHEMA_ID = "critic_td_horizon_diagnostic"


def budget_steps(*, horizon: int, transition_index: int) -> int:
    """Return the current action budget ``T - t``.

    ``transition_index`` is the decision index before the current action.  The
    function intentionally does not accept an observed episode length: the
    latter is retrospective and would leak future termination information.
    """

    value = int(horizon)
    index = int(transition_index)
    if value <= 0:
        raise ValueError("horizon must be positive")
    if index < 0 or index >= value:
        raise ValueError("transition_index must satisfy 0 <= t < T")
    return value - index


def next_budget_steps(*, horizon: int, transition_index: int) -> int:
    """Return the budget after the current action, ``T - t - 1``."""

    return budget_steps(horizon=horizon, transition_index=transition_index) - 1


def budget_fraction(*, horizon: int, transition_index: int) -> float:
    return float(budget_steps(horizon=horizon, transition_index=transition_index)) / float(horizon)


def build_sequence_sidecar(
    *,
    arrays: Mapping[str, np.ndarray],
    count: int,
    horizon: int,
    episode_lengths: Optional[Sequence[int]] = None,
) -> Tuple[dict, dict]:
    """Reconstruct and validate contiguous train Replay episode sequences.

    The returned arrays contain only row identity, task budget, and up to five
    contiguous row indices.  ``steps_to_recorded_episode_end`` is retained as
    a retrospective audit column and is never used by the model adapter.
    """

    n = int(count)
    T = int(horizon)
    if n <= 0 or T <= 0:
        raise ValueError("count and horizon must be positive")
    required = ("depth", "vector", "action_mask", "next_depth", "next_vector", "next_action_mask", "done")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError("sequence sidecar missing arrays: {}".format(", ".join(missing)))
    done = np.asarray(arrays["done"][:n], dtype=bool)
    if not bool(done.any()) or not bool(done[-1]):
        raise ValueError("Replay must end at a committed terminal row")
    terminal_rows = np.flatnonzero(done).astype(np.int64)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), terminal_rows[:-1] + 1))
    episode_index = np.empty(n, dtype=np.int64)
    episode_start = np.empty(n, dtype=np.int64)
    episode_end = np.empty(n, dtype=np.int64)
    for episode, (start, end) in enumerate(zip(starts.tolist(), terminal_rows.tolist())):
        if end < start:
            raise ValueError("invalid episode interval")
        if int(done[start:end].sum()) != 0 or not bool(done[end]):
            raise ValueError("episode interval has an invalid done boundary")
        episode_index[start : end + 1] = int(episode)
        episode_start[start : end + 1] = int(start)
        episode_end[start : end + 1] = int(end)

    # Validate exact endpoint continuity at every nonterminal boundary.  This
    # prevents an array-adjacency assumption from creating a cross-episode
    # sequence or silently accepting a malformed committed prefix.
    continuity_failures = []
    for row in range(n - 1):
        if bool(done[row]):
            continue
        if not np.array_equal(np.asarray(arrays["next_depth"][row]), np.asarray(arrays["depth"][row + 1])):
            continuity_failures.append((row, "depth"))
        if not np.array_equal(np.asarray(arrays["next_vector"][row]), np.asarray(arrays["vector"][row + 1])):
            continuity_failures.append((row, "vector"))
        if not np.array_equal(np.asarray(arrays["next_action_mask"][row]), np.asarray(arrays["action_mask"][row + 1])):
            continuity_failures.append((row, "action_mask"))
    if continuity_failures:
        raise ValueError("Replay endpoint continuity failed: {}".format(continuity_failures[:5]))

    t = np.empty(n, dtype=np.int64)
    recorded_length = np.empty(n, dtype=np.int64)
    for start, end in zip(starts.tolist(), terminal_rows.tolist()):
        length = int(end - start + 1)
        t[start : end + 1] = np.arange(length, dtype=np.int64)
        recorded_length[start : end + 1] = length

    next_index = np.full(n, -1, dtype=np.int64)
    for row in range(n):
        if not bool(done[row]):
            candidate = row + 1
            if candidate >= n or int(episode_index[candidate]) != int(episode_index[row]):
                raise ValueError("nonterminal row has no same-episode next row")
            next_index[row] = candidate

    nstep_indices = np.full((n, 5), -1, dtype=np.int64)
    nstep_length = np.zeros(n, dtype=np.int64)
    nstep_terminal = np.zeros(n, dtype=np.uint8)
    bootstrap_index = np.full(n, -1, dtype=np.int64)
    for row in range(n):
        end = int(episode_end[row])
        cursor = row
        length = 0
        terminal = False
        for offset in range(5):
            if cursor > end:
                break
            nstep_indices[row, offset] = int(cursor)
            length += 1
            if bool(done[cursor]):
                terminal = True
                break
            cursor += 1
        if length <= 0:
            raise ValueError("empty n-step segment")
        nstep_length[row] = int(length)
        nstep_terminal[row] = 1 if terminal else 0
        if not terminal:
            if length != 5 or cursor >= n:
                raise ValueError("nonterminal n-step segment is not five rows")
            if int(episode_index[cursor]) != int(episode_index[row]):
                raise ValueError("n-step segment crosses an episode boundary")
            bootstrap_index[row] = int(cursor)

    expected_episode_lengths = [int(end - start + 1) for start, end in zip(starts, terminal_rows)]
    if episode_lengths is not None and list(map(int, episode_lengths)) != expected_episode_lengths:
        raise ValueError("provided episode lengths do not match Replay done boundaries")

    current_budget = np.asarray([budget_steps(horizon=T, transition_index=value) for value in t], dtype=np.int64)
    next_budget = current_budget - 1
    sidecar = {
        "episode_index": episode_index,
        "t": t,
        "T": np.full(n, T, dtype=np.int64),
        "next_index": next_index,
        "nstep_indices": nstep_indices,
        "nstep_length": nstep_length,
        "nstep_terminal": nstep_terminal,
        "bootstrap_index": bootstrap_index,
        "recorded_episode_length": recorded_length,
        "steps_to_recorded_episode_end": recorded_length - 1 - t,
        "remaining_budget_steps": current_budget,
        "next_remaining_budget_steps": next_budget,
        "remaining_budget_fraction": current_budget.astype(np.float32) / float(T),
        "terminal_row": done.astype(np.uint8),
    }
    report = {
        "schema_id": DIAGNOSTIC_CRITIC_SCHEMA_ID,
        "row_count": n,
        "episode_count": int(len(terminal_rows)),
        "horizon_T": T,
        "done_rows": int(done.sum()),
        "episode_lengths": expected_episode_lengths,
        "continuity_failures": 0,
        "cross_episode_segments": 0,
        "unterminated_tail": False,
        "retrospective_field": "steps_to_recorded_episode_end",
        "retrospective_field_is_model_input": False,
        "future_leakage_inputs": 0,
        "nstep_terminal_segments": int(nstep_terminal.sum()),
        "nstep_bootstrap_segments": int((nstep_terminal == 0).sum()),
    }
    return sidecar, report


def extend_critic_state_dict(state_dict: Mapping[str, Any], *, torch) -> MutableMapping[str, Any]:
    """Copy a 127-dimensional Critic into a zero-initialized 128-dimensional one."""

    if "vector_encoder.0.weight" not in state_dict:
        raise KeyError("Critic state dict has no vector_encoder.0.weight")
    old_weight = state_dict["vector_encoder.0.weight"]
    if int(old_weight.shape[1]) != int(POLICY_VECTOR_DIM):
        raise ValueError("unexpected source Critic vector dimension")
    extended = {}
    for name, value in state_dict.items():
        extended[name] = value.detach().clone() if hasattr(value, "detach") else copy.deepcopy(value)
    new_weight = torch.zeros(
        (int(old_weight.shape[0]), int(DIAGNOSTIC_CRITIC_VECTOR_DIM)),
        dtype=old_weight.dtype,
        device=old_weight.device,
    )
    new_weight[:, : int(POLICY_VECTOR_DIM)] = old_weight
    extended["vector_encoder.0.weight"] = new_weight
    return extended


class BudgetCritic(__import__("torch").nn.Module):
    """A diagnostic Critic accepting the original vector plus one budget scalar."""

    def __init__(self, *, nn, depth_channels: int, source_state_dict: Mapping[str, Any], device):
        super().__init__()
        self.base = build_critic(
            nn,
            depth_channels=int(depth_channels),
            vec_dim=int(DIAGNOSTIC_CRITIC_VECTOR_DIM),
            num_actions=int(NUM_ACTIONS),
        ).to(device)
        self.base.load_state_dict(extend_critic_state_dict(source_state_dict, torch=__import__("torch")), strict=True)

    def forward(self, depth, vector, budget):
        if vector.ndim != 2 or int(vector.shape[1]) != int(POLICY_VECTOR_DIM):
            raise ValueError("BudgetCritic requires the original 127-dimensional policy vector")
        value = budget.to(device=vector.device, dtype=vector.dtype).reshape(-1, 1)
        if int(value.shape[0]) != int(vector.shape[0]):
            raise ValueError("budget batch size differs from vector batch size")
        return self.base(depth, __import__("torch").cat((vector, value), dim=1))


def masked_expected_value_independent(*, logits, q1, q2, action_mask, torch):
    """Independent masked categorical expected min-Q value, without production helper."""

    masks = action_mask.bool()
    if masks.ndim != 2 or logits.shape != q1.shape or q1.shape != q2.shape or q1.shape != masks.shape:
        raise ValueError("independent expected value tensors have incompatible shapes")
    if bool((~masks.any(dim=1)).any().item()):
        raise ValueError("non-terminal action mask is empty")
    masked_logits = logits.masked_fill(~masks, -1.0e4)
    probabilities = torch.softmax(masked_logits, dim=1) * masks.to(dtype=logits.dtype)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    q_min = torch.minimum(q1, q2)
    return (probabilities * q_min).sum(dim=1), probabilities


def n_step_discounted_return(
    *,
    rewards: Sequence[float],
    done: Sequence[bool],
    gamma: float,
    reward_scale: float,
    bootstrap_value: float = 0.0,
) -> Tuple[float, int, bool]:
    """Compute one bounded n-step target prefix without crossing terminal rows."""

    if len(rewards) != len(done) or not rewards:
        raise ValueError("n-step rewards/done must be a non-empty equal-length sequence")
    total = 0.0
    terminal = False
    used = 0
    for index, (reward, is_done) in enumerate(zip(rewards, done)):
        total += (float(gamma) ** index) * float(reward_scale) * float(reward)
        used += 1
        if bool(is_done):
            terminal = True
            break
    if not terminal:
        total += (float(gamma) ** used) * float(bootstrap_value)
    return float(total), int(used), bool(terminal)


__all__ = [
    "BudgetCritic",
    "DIAGNOSTIC_CRITIC_SCHEMA_ID",
    "DIAGNOSTIC_CRITIC_VECTOR_DIM",
    "budget_fraction",
    "budget_steps",
    "build_sequence_sidecar",
    "extend_critic_state_dict",
    "masked_expected_value_independent",
    "n_step_discounted_return",
    "next_budget_steps",
]
