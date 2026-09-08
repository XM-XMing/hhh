"""Generic masked-discrete optimization helpers owned by AWAC."""

from __future__ import annotations

import math
from typing import Dict


def bc_topk_support_mask(bc_logits, action_mask, top_k: int, torch):
    """Return the valid-action intersection with the BC top-k support."""
    masks = action_mask.bool()
    if bc_logits.ndim != 2 or masks.shape != bc_logits.shape:
        raise ValueError("BC logits/action mask must have identical [B,A] shape")
    if int(top_k) <= 0:
        raise ValueError("BC support top-k must be positive")
    no_valid = ~masks.any(dim=1)
    if bool(no_valid.any()):
        masks = masks.clone()
        masks[no_valid, 0] = True
    count = min(int(top_k), int(bc_logits.shape[1]))
    ranked_logits = bc_logits.detach().masked_fill(~masks, -torch.inf)
    indices = torch.topk(ranked_logits, k=count, dim=1).indices
    support = torch.zeros_like(masks)
    support.scatter_(1, indices, True)
    support &= masks
    if bool((~support.any(dim=1)).any()):
        raise RuntimeError("BC top-k support unexpectedly became empty")
    return support


def masked_discrete_cql_loss(q_values, actions, action_mask, torch):
    """CQL penalty over valid actions relative to the executed data action."""
    masks = action_mask.bool()
    if q_values.ndim != 2 or masks.shape != q_values.shape:
        raise ValueError("Q values/action mask must have identical [B,A] shape")
    if actions.ndim != 1 or int(actions.shape[0]) != int(q_values.shape[0]):
        raise ValueError("actions must have shape [B]")
    executed_valid = masks.gather(1, actions[:, None]).squeeze(1)
    if not bool(executed_valid.all()):
        raise ValueError("executed replay action is outside its action mask")
    q_data = q_values.gather(1, actions[:, None]).squeeze(1)
    valid_q = q_values.masked_fill(~masks, -torch.inf)
    return (torch.logsumexp(valid_q, dim=1) - q_data).mean()


def actor_update_due(
    *,
    replay_total_added: int,
    learner_update_step: int,
    actor_learning_starts: int,
    critic_burnin_updates: int,
    actor_update_interval: int,
) -> bool:
    """Return whether the next scheduled update may change the Actor."""
    if int(actor_learning_starts) < 0 or int(critic_burnin_updates) < 0:
        raise ValueError("actor learning start and critic burn-in must be non-negative")
    if int(actor_update_interval) <= 0:
        raise ValueError("actor update interval must be positive")
    if int(replay_total_added) < int(actor_learning_starts):
        return False
    if int(learner_update_step) < int(critic_burnin_updates):
        return False
    post_burnin_update_number = (
        int(learner_update_step) - int(critic_burnin_updates) + 1
    )
    return post_burnin_update_number % int(actor_update_interval) == 0


def critic_update_target(
    *,
    replay_total_added: int,
    learning_starts: int,
    updates_per_step: float,
) -> int:
    """Return the cumulative Critic-update target for the replay cursor.

    The persistent replay's ``total_added`` is the schedule clock.  This is
    the same threshold semantics used by the Phase-0 calibration producer,
    expressed here as a public schedule owner so the Standard online phase
    cannot silently switch to a phase-local or replay-size clock.
    """

    total = int(replay_total_added)
    starts = int(learning_starts)
    rate = float(updates_per_step)
    if total < 0 or starts < 0 or rate <= 0.0 or not math.isfinite(rate):
        raise ValueError("invalid Critic update schedule inputs")
    if total <= starts:
        return 0
    return int(math.floor(float(total - starts + 1) * rate))


def minimum_online_transitions_for_first_actor_update(
    *,
    starting_replay_total_added: int,
    starting_critic_update_count: int,
    starting_learner_update_step: int,
    learning_starts: int,
    actor_learning_starts: int,
    critic_burnin_updates: int,
    actor_update_interval: int,
    updates_per_step: float,
) -> int:
    """Compute the first online transition that can schedule an Actor step.

    This simulates only the existing integer schedule predicates; it performs
    no learner work.  The returned value is therefore a budget calculation,
    not a tuning shortcut.  An online transition is counted when it advances
    the persistent ``total_added`` clock, and the Actor predicate is checked
    immediately before each scheduled learner update.
    """

    values = {
        "starting_replay_total_added": int(starting_replay_total_added),
        "starting_critic_update_count": int(starting_critic_update_count),
        "starting_learner_update_step": int(starting_learner_update_step),
        "learning_starts": int(learning_starts),
        "actor_learning_starts": int(actor_learning_starts),
        "critic_burnin_updates": int(critic_burnin_updates),
        "actor_update_interval": int(actor_update_interval),
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("AWAC schedule counters must be non-negative")
    if values["actor_update_interval"] <= 0:
        raise ValueError("actor_update_interval must be positive")
    if values["actor_learning_starts"] < values["learning_starts"]:
        raise ValueError("actor_learning_starts must be at least learning_starts")
    rate = float(updates_per_step)
    if rate <= 0.0 or not math.isfinite(rate):
        raise ValueError("updates_per_step must be finite and positive")

    # The bound is intentionally generous.  It keeps malformed schedules
    # fail-closed without making the ordinary formal values an unbounded
    # search, while the loop below remains the source of the exact answer.
    required_updates = max(
        values["starting_critic_update_count"] + 1,
        values["critic_burnin_updates"] + values["actor_update_interval"],
    )
    required_total = values["learning_starts"] - 1 + int(
        math.ceil(float(required_updates) / rate)
    )
    search_limit = max(
        1,
        values["actor_learning_starts"] - values["starting_replay_total_added"],
        required_total - values["starting_replay_total_added"],
    ) + values["actor_update_interval"] + 8

    critic_updates = values["starting_critic_update_count"]
    learner_update_step = values["starting_learner_update_step"]
    for online_count in range(1, int(search_limit) + 1):
        total_added = values["starting_replay_total_added"] + online_count
        target = critic_update_target(
            replay_total_added=total_added,
            learning_starts=values["learning_starts"],
            updates_per_step=rate,
        )
        while critic_updates < target:
            if actor_update_due(
                replay_total_added=total_added,
                learner_update_step=learner_update_step,
                actor_learning_starts=values["actor_learning_starts"],
                critic_burnin_updates=values["critic_burnin_updates"],
                actor_update_interval=values["actor_update_interval"],
            ):
                return int(online_count)
            critic_updates += 1
            learner_update_step += 1

    raise ValueError(
        "AWAC schedule cannot reach a first Actor update within the derived bound"
    )


def kl_proposal_is_acceptable(
    *,
    recovery_active: bool,
    budget: float,
    before_mean: float,
    before_max: float,
    proposed_mean: float,
    proposed_max: float,
    tolerance: float = 1.0e-8,
) -> bool:
    """Apply the BC KL budget to both expected and worst-row drift."""
    values = (
        float(budget),
        float(before_mean),
        float(before_max),
        float(proposed_mean),
        float(proposed_max),
        float(tolerance),
    )
    if not all(math.isfinite(value) for value in values):
        return False
    if float(budget) <= 0.0 or float(tolerance) < 0.0:
        raise ValueError("KL budget/tolerance are invalid")
    if not bool(recovery_active):
        return bool(
            float(proposed_mean) <= float(budget) + float(tolerance)
            and float(proposed_max) <= float(budget) + float(tolerance)
        )
    mean_non_regression = float(proposed_mean) <= float(before_mean) + float(tolerance)
    max_non_regression = float(proposed_max) <= float(before_max) + float(tolerance)
    strict_improvement = bool(
        float(proposed_mean) < float(before_mean) - float(tolerance)
        or float(proposed_max) < float(before_max) - float(tolerance)
    )
    return bool(mean_non_regression and max_non_regression and strict_improvement)


def kl_budget_exceeded(*, mean_kl: float, max_kl: float, budget: float) -> bool:
    """Use one hard budget for mean and worst-row KL evidence."""
    values = (float(mean_kl), float(max_kl), float(budget))
    if not all(math.isfinite(value) for value in values):
        return True
    if float(budget) <= 0.0:
        raise ValueError("KL budget must be positive")
    return bool(float(mean_kl) > float(budget) or float(max_kl) > float(budget))


def critic_update_budget_increment(
    *,
    replay_total_before: int,
    replay_total_after: int,
    learning_starts: int,
    updates_per_step: float,
) -> float:
    """Return critic update budget earned by one committed replay generation."""
    before = int(replay_total_before)
    after = int(replay_total_after)
    starts = int(learning_starts)
    rate = float(updates_per_step)
    if before < 0 or after < before or starts < 0 or rate <= 0.0:
        raise ValueError("invalid critic update schedule inputs")
    if after <= starts:
        return 0.0
    if before <= starts:
        eligible = after - starts
        return max(1.0, rate * float(eligible))
    return rate * float(after - before)


def first_critic_update_evidence(*, global_step: int, critic_update_step: int, metrics: Dict):
    """Return immutable evidence for the first critic update."""
    if int(critic_update_step) <= 0:
        return None
    required = (
        "critic_loss", "target_q_mean", "target_q_std", "q_mean", "q_std",
        "critic_gradient_norm", "critic_parameter_delta_norm", "parameters_finite",
    )
    evidence = {"global_step": int(global_step), "critic_update_step": int(critic_update_step)}
    for name in required:
        if name not in metrics:
            raise KeyError("missing first critic metric: {}".format(name))
        value = float(metrics[name])
        if not math.isfinite(value):
            raise FloatingPointError("non-finite first critic metric: {}".format(name))
        evidence[name] = value
    return evidence
