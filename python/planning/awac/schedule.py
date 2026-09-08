"""Scheduling contract for AWAC learner orchestration.

Training segment boundaries control collection-policy refreshes.  Full
development evaluations are an independent, less frequent cadence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AWACSchedule:
    """Resolved boundaries for one AWAC training run."""

    total_steps: int
    train_segment_steps: int
    full_eval_interval_steps: int
    training_segment_ends: tuple[int, ...]
    collection_refresh_ends: tuple[int, ...]
    full_eval_ends: tuple[int, ...]


def build_awac_schedule(
    *,
    total_steps: int,
    train_segment_steps: int,
    full_eval_interval_steps: int,
) -> AWACSchedule:
    """Build the deterministic AWAC run boundary schedule.

    The final total step is always a full-evaluation boundary, even when it
    is not an exact multiple of ``full_eval_interval_steps``.  Segment
    boundaries are clipped at the total so a partial final segment is still
    represented exactly once.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if train_segment_steps <= 0:
        raise ValueError("train_segment_steps must be positive")
    if full_eval_interval_steps <= 0:
        raise ValueError("full_eval_interval_steps must be positive")

    segment_ends = tuple(
        list(range(train_segment_steps, total_steps, train_segment_steps))
        + [total_steps]
    )
    full_eval_ends = tuple(
        list(range(full_eval_interval_steps, total_steps, full_eval_interval_steps))
        + [total_steps]
    )
    return AWACSchedule(
        total_steps=total_steps,
        train_segment_steps=train_segment_steps,
        full_eval_interval_steps=full_eval_interval_steps,
        training_segment_ends=segment_ends,
        collection_refresh_ends=segment_ends,
        full_eval_ends=full_eval_ends,
    )
