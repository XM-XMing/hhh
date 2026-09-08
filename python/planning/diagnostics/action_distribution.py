"""Shared, descriptive action-distribution audit for expert datasets."""

from __future__ import annotations
from typing import Any
import numpy as np

ACTION_DISTRIBUTION_AUDIT_CONTRACT_ID = "expert_action_distribution"

def summarize_action_histogram(action_histogram: Any, motion_primitives: Any) -> dict:
    histogram = np.asarray(action_histogram, dtype=np.int64).reshape(-1)
    if histogram.shape != (int(motion_primitives.num_actions),):
        raise ValueError("action histogram does not match motion primitive count")
    if np.any(histogram < 0):
        raise ValueError("action histogram cannot contain negative counts")
    total = int(histogram.sum())
    used = histogram > 0
    probabilities = histogram.astype(np.float64) / max(1, total)
    positive = probabilities[probabilities > 0.0]
    normalized_entropy = (
        float(-(positive * np.log(positive)).sum() / np.log(histogram.size))
        if positive.size and histogram.size > 1
        else 0.0
    )

    def rate(mask: np.ndarray) -> float:
        return float(histogram[np.asarray(mask, dtype=np.bool_)].sum() / max(1, total))

    tolerance = 1.0e-6
    lateral = np.abs(np.asarray(motion_primitives.y_end)) > tolerance
    vertical = np.abs(np.asarray(motion_primitives.z_end)) > tolerance
    turning = np.abs(np.asarray(motion_primitives.terminal_heading_rad)) > tolerance
    return {
        "contract_id": ACTION_DISTRIBUTION_AUDIT_CONTRACT_ID,
        "total_actions": total,
        "unique_actions": int(np.count_nonzero(used)),
        "normalized_entropy": normalized_entropy,
        "max_action_fraction": float(probabilities.max()) if total else 0.0,
        "center_action_fraction": float(
            histogram[int(motion_primitives.center_action_id)] / max(1, total)
        ),
        "lateral_action_fraction": rate(lateral),
        "vertical_action_fraction": rate(vertical),
        "turning_action_fraction": rate(turning),
        "used_horizontal_bins": int(
            np.unique(np.asarray(motion_primitives.horizontal_index)[used]).size
        ),
        "total_horizontal_bins": int(
            np.unique(np.asarray(motion_primitives.horizontal_index)).size
        ),
        "used_vertical_bins": int(
            np.unique(np.asarray(motion_primitives.vertical_index)[used]).size
        ),
        "total_vertical_bins": int(
            np.unique(np.asarray(motion_primitives.vertical_index)).size
        ),
        "histogram": histogram.tolist(),
    }
