import copy

import numpy as np
import pytest
import torch

from planning.diagnostics.multi_action_ranking_critic import (
    assert_state_dict_equal,
    build_pair_label,
    build_mission_group_split,
    compute_q_scale,
    lambda_rank_from_grad_norms,
    mission_cluster_bootstrap,
    ranking_enabled_for_branch,
    ranking_loss,
    sample_pair_schedule,
    validate_no_mc_target_without_sequence,
)


def _pair(mission, state, left_return, right_return, left_source="BC", right_source="RANDOM"):
    return {
        "mission_id": mission,
        "state_id": state,
        "action1": 1,
        "action2": 2,
        "action_source1": left_source,
        "action_source2": right_source,
        "return1": left_return,
        "return2": right_return,
    }


def test_pair_label_is_return_only_and_ties_are_excluded():
    assert build_pair_label(3.0, 1.0) == 1
    assert build_pair_label(1.0, 3.0) == -1
    assert build_pair_label(2.0, 2.0) == 0


def test_mission_split_keeps_all_states_and_pairs_together():
    pairs = [
        _pair("m0", "s0", 2.0, 1.0),
        _pair("m0", "s0", 1.0, 2.0),
        _pair("m1", "s1", 2.0, 1.0),
        _pair("m2", "s2", 2.0, 1.0),
    ]
    manifest = build_mission_group_split(pairs, seed=20260907, fold_count=3)
    assert len(manifest["missions"]) == 3
    assert len({row["fold_id"] for row in manifest["missions"]}) == 3
    assert all(row["mission_id"] in manifest["mission_to_fold"] for row in manifest["missions"])
    assert all(
        manifest["mission_to_fold"][row["mission_id"]]
        == manifest["mission_to_fold"][pairs[index]["mission_id"]]
        for index, row in enumerate(manifest["pair_assignments"])
    )


def test_ranking_loss_direction_and_tie_rejection():
    labels = torch.tensor([1.0])
    good = torch.tensor([2.0], requires_grad=True)
    bad = torch.tensor([1.0], requires_grad=True)
    lower = ranking_loss(good, bad, good, bad, labels, q_scale=1.0)
    higher_bad = ranking_loss(good, bad + 0.8, good, bad + 0.8, labels, q_scale=1.0)
    higher_good = ranking_loss(good + 0.8, bad, good + 0.8, bad, labels, q_scale=1.0)
    assert higher_good < lower
    assert higher_bad > lower
    with pytest.raises(ValueError):
        ranking_loss(good, bad, good, bad, torch.tensor([0.0]), q_scale=1.0)


def test_both_q_heads_receive_ranking_gradient():
    q1_i = torch.tensor([1.0], requires_grad=True)
    q1_j = torch.tensor([0.0], requires_grad=True)
    q2_i = torch.tensor([1.0], requires_grad=True)
    q2_j = torch.tensor([0.0], requires_grad=True)
    loss = ranking_loss(q1_i, q1_j, q2_i, q2_j, torch.tensor([1.0]), q_scale=1.0)
    loss.backward()
    assert q1_i.grad is not None and float(q1_i.grad.abs()) > 0.0
    assert q2_i.grad is not None and float(q2_i.grad.abs()) > 0.0


def test_pair_schedule_is_reproducible_and_mission_state_balanced():
    groups = {"m0": {"s0": [0, 1]}, "m1": {"s1": [2, 3]}}
    left = sample_pair_schedule(groups, updates=4, batch_size=8, seed=202609071)
    right = sample_pair_schedule(groups, updates=4, batch_size=8, seed=202609071)
    assert np.array_equal(left, right)
    assert left.shape == (4, 8)


def test_bootstrap_is_mission_clustered_and_reproducible():
    values = {
        "c1": np.asarray([0.60, 0.70, 0.80]),
        "c2": np.asarray([0.65, 0.75, 0.85]),
    }
    clusters = np.asarray(["m0", "m1", "m2"])
    one = mission_cluster_bootstrap(values, clusters, left="c2", right="c1", repeats=200, seed=4)
    two = mission_cluster_bootstrap(values, clusters, left="c2", right="c1", repeats=200, seed=4)
    assert one == two
    assert one["cluster_count"] == 3
    assert len(one["samples"]) == 200


def test_state_dict_invariance_seam():
    before = {"weight": torch.ones(2)}
    after = copy.deepcopy(before)
    assert_state_dict_equal(before, after)
    after["weight"][0] = 2.0
    with pytest.raises(AssertionError):
        assert_state_dict_equal(before, after)


def test_no_mc_target_without_step_sequence():
    with pytest.raises(ValueError):
        validate_no_mc_target_without_sequence(has_complete_step_sequence=False)
    assert validate_no_mc_target_without_sequence(has_complete_step_sequence=True) is True


def test_q_scale_and_lambda_are_fixed_calibration_rules():
    assert compute_q_scale([0.0, 0.002, 0.004]) == pytest.approx(0.002)
    assert compute_q_scale([0.0, 0.0]) == pytest.approx(1.0e-3)
    assert lambda_rank_from_grad_norms(4.0, 2.0) == pytest.approx(0.5)
    assert lambda_rank_from_grad_norms(1.0e9, 1.0e-12) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        compute_q_scale([np.nan])


def test_c1_and_c2_have_distinct_preregistered_ranking_flags():
    assert ranking_enabled_for_branch("C1_CONTINUED_TD") is False
    assert ranking_enabled_for_branch("C2_TD_PLUS_RANK") is True
    with pytest.raises(ValueError):
        ranking_enabled_for_branch("C0_FROZEN_N5")
