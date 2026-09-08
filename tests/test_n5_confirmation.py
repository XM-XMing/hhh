"""Regression tests for the N1/H0 versus N5/H0 confirmation contracts."""

from planning.awac.n5_confirmation import (
    cluster_bootstrap_delta,
    derive_episode_seed,
    episode_budget_audit,
    normalized_task_identity,
    select_confirmation_rows,
)


def _row(index, *, formal=True):
    row = {
        "episode_id": str(index),
        "mission_id": "declared-{}".format(index),
        "start_x": str(index),
        "start_y": "0",
        "start_z": "1.5",
        "start_yaw_deg": "0",
        "goal_x": str(index + 1),
        "goal_y": "0",
        "goal_z": "1.5",
    }
    if formal:
        for field in (
            "result", "steps", "final_distance_xy_m", "final_error_z_m",
            "route_length_m", "executed_path_length_m", "remaining_path_lower_bound_m",
            "straight_line_distance_m", "path_stretch", "path_length_contract_id",
            "plan_path_max_m", "teacher_path_length_contract_id", "teacher_plan_path_length_m",
            "teacher_plan_path_stretch", "mission_route_store_contract_id",
            "mission_route_store_schema_version", "mission_route_store",
            "mission_route_store_candidate_index_sha256", "mission_route_store_points_sha256",
            "mission_route_store_offsets_sha256", "mission_route_store_total_route_points",
        ):
            row[field] = "1"
    return row


def test_normalized_identity_ignores_declared_id_and_row_order():
    left = _row(1)
    right = dict(left, mission_id="different-file-id")
    assert normalized_task_identity(left) == normalized_task_identity(right)


def test_unknown_exclusion_identity_fails_closed():
    try:
        select_confirmation_rows([_row(1)], excluded_identities=set(), excluded_identity_status="UNKNOWN", count=1, seed=7)
    except ValueError as error:
        assert "not complete" in str(error)
    else:
        raise AssertionError("unknown exclusion identity was treated as zero overlap")


def test_nonformal_rows_cannot_fill_confirmation_budget():
    selected, audit = select_confirmation_rows([_row(1, formal=False), _row(2)], excluded_identities=set(), excluded_identity_status="COMPLETE", count=2, seed=7)
    assert len(selected) == 1
    assert audit["shortfall"] == 1
    assert audit["selection_status"] == "BLOCKED_INSUFFICIENT_FORMAL_TASKS"


def test_episode_seed_is_worker_order_independent():
    assert derive_episode_seed(202609074, "task-a", 0) == derive_episode_seed(202609074, "task-a", 0)
    assert derive_episode_seed(202609074, "task-a", 0) != derive_episode_seed(202609074, "task-b", 0)


def test_episode_budget_is_hard_bounded():
    report = episode_budget_audit(count=200, max_steps=45, max_decision_steps=9000)
    assert report["within_step_budget"] is True
    assert report["worst_case_steps"] == 9000


def test_cluster_bootstrap_keeps_models_paired():
    result = cluster_bootstrap_delta(
        {"a": [1.0, 2.0], "b": [2.0, 3.0]},
        {"a": [0.0, 1.0], "b": [1.0, 2.0]},
        {"a": [0.5, 1.5], "b": [1.5, 2.5]},
        seed=202609075,
        repeats=25,
    )
    assert result["cluster_count"] == 2
    assert result["valid"] == 25
