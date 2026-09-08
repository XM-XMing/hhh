"""Regression contract for audit-only primitive acknowledgement localization."""

import json
from pathlib import Path

import pytest


FRAME_COUNT = 25
EXECUTION_ID = 991
AUDIT_RUN_ID = "audit-run-991"
UNITY_RUNTIME_IDENTITY = "player=fixture;assembly_csharp=fixture"
P0_J_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "execution_transport"
    / "p0_j_real_execution_4435486932233309751.json"
)


def _frame(index, **extra):
    return {
        "execution_id": EXECUTION_ID,
        "frame_index": int(index),
        "command_id": 1000 + int(index),
        "state_id": 5000 + int(index),
        "sim_time_ns": 100000000 + 20000000 * int(index),
        **extra,
    }


def _unity(*, failed_send=()):
    return [
        _frame(
            index,
            applied=True,
            send_attempted=True,
            send_ok=index not in set(failed_send),
            execution_status="complete" if index == FRAME_COUNT - 1 else "frame_applied",
        )
        for index in range(FRAME_COUNT)
    ]


def _bridge(indices):
    return [_frame(index, received=True, forwarded=True) for index in indices]


def _planning(indices):
    return [_frame(index, received=True) for index in indices]


def _accepted(indices):
    return [_frame(index, accepted=True) for index in indices]


def _build(**kwargs):
    from planning.diagnostics.execution_transport_audit import build_execution_transport_ledger

    # Existing cases express a single terminal receipt set; for those cases
    # the collector and accounting boundaries deliberately coincide.
    accepted = kwargs.pop("accepted_records", None)
    if accepted is not None:
        indices = [record["frame_index"] for record in accepted]
        kwargs.setdefault("collector_records", _collector(indices))
        kwargs.setdefault("accounting_records", _accounted(indices))
    return build_execution_transport_ledger(
        execution_id=EXECUTION_ID,
        expected_frame_count=FRAME_COUNT,
        **kwargs,
    )


def _collector(indices):
    return [_frame(index, collected=True) for index in indices]


def _accounted(indices):
    return [_frame(index, accounted=True) for index in indices]


def _load_p0_j_fixture():
    return json.loads(P0_J_FIXTURE.read_text(encoding="utf-8"))


def test_unity_applied_25_planning_received_23_remains_exact_n_red():
    from planning.contracts.primitive_execution import (
        PrimitiveExecutionContractError,
        validate_primitive_execution,
    )

    indices = [index for index in range(FRAME_COUNT) if index not in {7, 19}]
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(indices),
        planning_records=_planning(indices),
        accepted_records=_accepted(indices),
    )

    assert ledger["first_loss_layer"] == "UNITY_PUB_TO_BRIDGE"
    assert ledger["missing_frame_indices"] == [7, 19]
    receipt = {
        "execution_id": EXECUTION_ID,
        "requested_frame_count": FRAME_COUNT,
        "applied_frames": _accepted(indices),
        "endpoint_state_id": 5000 + FRAME_COUNT - 1,
    }
    with pytest.raises(PrimitiveExecutionContractError):
        validate_primitive_execution(receipt, expected_frame_count=FRAME_COUNT)


def test_unity_send_false_is_localized_and_not_silent():
    indices = [index for index in range(FRAME_COUNT) if index not in {11, 22}]
    ledger = _build(
        unity_records=_unity(failed_send={11, 22}),
        bridge_records=_bridge(indices),
        planning_records=_planning(indices),
        accepted_records=_accepted(indices),
    )

    assert ledger["first_loss_layer"] == "UNITY_SEND_ATTEMPT"
    assert ledger["missing_frame_indices"] == [11, 22]
    assert ledger["frames"][11]["unity_send_ok"] is False
    assert ledger["frames"][22]["unity_send_ok"] is False


def test_bridge_to_planning_loss_is_not_attributed_to_unity_physics():
    indices = [index for index in range(FRAME_COUNT) if index not in {4, 17}]
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(range(FRAME_COUNT)),
        planning_records=_planning(indices),
        accepted_records=_accepted(indices),
    )

    assert ledger["first_loss_layer"] == "BRIDGE_TO_PLANNING"
    assert ledger["missing_frame_indices"] == [4, 17]


def test_planning_received_25_accepted_23_is_collector_rejection():
    indices = [index for index in range(FRAME_COUNT) if index not in {6, 23}]
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(range(FRAME_COUNT)),
        planning_records=_planning(range(FRAME_COUNT)),
        accepted_records=_accepted(indices),
    )

    assert ledger["first_loss_layer"] == "PLANNING_COLLECTOR"
    assert ledger["missing_frame_indices"] == [6, 23]


def test_unity_applied_prefix_is_execution_accounting_not_transport_loss():
    ledger = _build(
        unity_records=_unity()[:14],
        bridge_records=_bridge(range(14)),
        planning_records=_planning(range(14)),
        accepted_records=_accepted(range(14)),
    )

    assert ledger["first_loss_layer"] == "EXECUTION_ACCOUNTING"
    assert ledger["missing_frame_indices"] == list(range(14, FRAME_COUNT))


def test_missing_unity_ledger_remains_unresolved_not_execution_accounting():
    indices = [index for index in range(FRAME_COUNT) if index not in {2, 21}]
    ledger = _build(
        unity_records=[],
        bridge_records=_bridge(indices),
        planning_records=_planning(indices),
        accepted_records=_accepted(indices),
    )

    assert ledger["unity_audit_available"] is False
    assert ledger["first_loss_layer"] == "UNRESOLVED"


def test_first_loss_report_keeps_missing_unity_audit_unresolved():
    from planning.diagnostics.execution_transport_report import build_report

    indices = [index for index in range(FRAME_COUNT) if index not in {8, 20}]
    planning_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit",
        "audit_schema_version": 2,
        "audit_run_id": AUDIT_RUN_ID,
        "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
        "episode_id": 81660,
        "step": 11,
        "selected_action": 98,
        "failure_kind": "PrimitiveExecutionContractError",
        "executions": [
            {
                "execution_id": EXECUTION_ID,
                "requested_frame_count": FRAME_COUNT,
                "command_ids": [1000 + index for index in range(FRAME_COUNT)],
                "planning_received": _planning(indices),
                "collector_received": _collector(indices),
                "execution_accounted": _accounted(indices),
                "collector_result": {"kind": "PROTOCOL_ERROR"},
            }
        ],
    }
    report = build_report(
        planning_payload=planning_payload,
        bridge_payload={
            "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit",
            "audit_schema_version": 2,
            "audit_run_id": AUDIT_RUN_ID,
            "episode_id": 81660,
            "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
            "recv_hwm": 4,
            "send_hwm": 4,
            "records": _bridge(indices),
        },
        unity_payload=None,
    )

    assert report["decision_index"] == 11
    assert report["selected_action"] == 98
    assert report["bridge_recv_hwm"] == 4
    assert report["ledger"]["first_loss_layer"] == "UNRESOLVED"
    assert report["ledger"]["missing_frame_indices"] == [8, 20]


def test_first_loss_report_normalizes_unity_player_audit_fields():
    from planning.diagnostics.execution_transport_report import build_report

    unity_records = [
        {
            **_frame(index),
            "frame_count": FRAME_COUNT,
            "frame_applied": True,
            "serialization_attempted": True,
            "serialization_success": True,
            "state_publish_attempted": True,
            "try_send_return": index != 12,
        }
        for index in range(FRAME_COUNT)
    ]
    planning_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit",
        "audit_schema_version": 2,
        "audit_run_id": AUDIT_RUN_ID,
        "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
        "episode_id": 81660,
        "step": 7,
        "selected_action": 52,
        "executions": [
            {
                "execution_id": EXECUTION_ID,
                "requested_frame_count": FRAME_COUNT,
                "command_ids": [1000 + index for index in range(FRAME_COUNT)],
                "planning_received": _planning([index for index in range(FRAME_COUNT) if index != 12]),
                "collector_received": _collector([index for index in range(FRAME_COUNT) if index != 12]),
                "execution_accounted": _accounted([index for index in range(FRAME_COUNT) if index != 12]),
                "collector_result": {"kind": "PROTOCOL_ERROR"},
            }
        ],
    }

    report = build_report(
        planning_payload=planning_payload,
        bridge_payload={
            "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit",
            "audit_schema_version": 2,
            "audit_run_id": AUDIT_RUN_ID,
            "episode_id": 81660,
            "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
            "recv_hwm": 4,
            "send_hwm": 4,
            "records": _bridge([index for index in range(FRAME_COUNT) if index != 12]),
        },
        unity_payload={
            "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]unity_execution_transport_audit",
            "audit_schema_version": 2,
            "episode_id": 81660,
            "audit_run_id": AUDIT_RUN_ID,
            "runtime_identity": UNITY_RUNTIME_IDENTITY,
            "configured_send_hwm": 2,
            "configured_recv_hwm": 2,
            "records": unity_records,
        },
    )

    assert report["ledger"]["unity_applied_count"] == 25
    assert report["ledger"]["unity_send_success_count"] == 24
    assert report["ledger"]["first_loss_layer"] == "UNITY_SEND_ATTEMPT"
    assert report["ledger"]["first_loss_frame_index"] == 12


def test_terminal_abort_unity_audit_preserves_only_real_applied_prefix():
    from planning.diagnostics.execution_transport_report import build_report

    prefix = list(range(10))
    unity_records = [
        {
            **_frame(index),
            "frame_count": FRAME_COUNT,
            "frame_applied": True,
            "serialization_attempted": True,
            "serialization_success": True,
            "state_publish_attempted": True,
            "try_send_return": True,
        }
        for index in prefix
    ]
    planning_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit",
        "audit_schema_version": 2,
        "audit_run_id": AUDIT_RUN_ID,
        "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
        "episode_id": 12116,
        "executions": [
            {
                "execution_id": EXECUTION_ID,
                "requested_frame_count": FRAME_COUNT,
                "command_ids": [1000 + index for index in range(FRAME_COUNT)],
                "planning_received": _planning(prefix),
                "collector_received": _collector(prefix),
                "execution_accounted": _accounted(prefix),
                "collector_result": {"kind": "TERMINAL_ABORT"},
            }
        ],
    }

    report = build_report(
        planning_payload=planning_payload,
        bridge_payload={
            "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit",
            "audit_schema_version": 2,
            "audit_run_id": AUDIT_RUN_ID,
            "episode_id": 12116,
            "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
            "records": _bridge(prefix),
        },
        unity_payload={
            "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]unity_execution_transport_audit",
            "audit_schema_version": 2,
            "episode_id": 12116,
            "audit_run_id": AUDIT_RUN_ID,
            "runtime_identity": UNITY_RUNTIME_IDENTITY,
            "records": unity_records,
        },
    )

    assert report["ledger"]["unity_applied_count"] == 10
    assert report["requested_frame_count"] == FRAME_COUNT
    assert report["terminal_abort"] is True
    assert report["ledger"]["expected_frame_count"] == 10
    assert report["ledger"]["missing_frame_indices"] == []
    assert report["ledger"]["first_loss_layer"] is None


def test_send_failure_is_not_attributed_without_an_absent_bridge_frame():
    """A contradictory ledger must fail instead of inventing a loss layer."""
    from planning.diagnostics.execution_transport_audit import build_execution_transport_ledger

    with pytest.raises(ValueError, match="send failure"):
        build_execution_transport_ledger(
            execution_id=EXECUTION_ID,
            expected_frame_count=FRAME_COUNT,
            unity_records=_unity(failed_send={12}),
            bridge_records=_bridge(range(FRAME_COUNT)),
            planning_records=_planning(range(FRAME_COUNT)),
            collector_records=_collector(range(FRAME_COUNT)),
            accounting_records=_accounted(range(FRAME_COUNT)),
        )


def test_duplicate_ledger_frame_is_not_silently_collapsed():
    from planning.diagnostics.execution_transport_audit import build_execution_transport_ledger

    duplicated_unity = _unity() + [_unity()[7]]
    with pytest.raises(ValueError, match="duplicate Unity audit record"):
        build_execution_transport_ledger(
            execution_id=EXECUTION_ID,
            expected_frame_count=FRAME_COUNT,
            unity_records=duplicated_unity,
            bridge_records=_bridge(range(FRAME_COUNT)),
            planning_records=_planning(range(FRAME_COUNT)),
            collector_records=_collector(range(FRAME_COUNT)),
            accounting_records=_accounted(range(FRAME_COUNT)),
        )


def test_callback_present_collector_missing_is_planning_collector_loss():
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(range(FRAME_COUNT)),
        planning_records=_planning(range(FRAME_COUNT)),
        collector_records=_collector([index for index in range(FRAME_COUNT) if index != 9]),
        accounting_records=_accounted([index for index in range(FRAME_COUNT) if index != 9]),
    )

    assert ledger["first_loss_layer"] == "PLANNING_COLLECTOR"
    assert ledger["first_loss_frame_index"] == 9


def test_collector_present_accounting_missing_is_execution_accounting_loss():
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(range(FRAME_COUNT)),
        planning_records=_planning(range(FRAME_COUNT)),
        collector_records=_collector(range(FRAME_COUNT)),
        accounting_records=_accounted([index for index in range(FRAME_COUNT) if index != 9]),
    )

    assert ledger["first_loss_layer"] == "EXECUTION_ACCOUNTING"
    assert ledger["first_loss_frame_index"] == 9


@pytest.mark.parametrize(
    "bridge_indices,planning_indices,collector_indices,accounted_indices,expected_layer",
    [
        (
            range(24),
            range(24),
            range(24),
            (),
            "UNITY_PUB_TO_BRIDGE",
        ),
        (
            range(FRAME_COUNT),
            range(24),
            range(24),
            (),
            "BRIDGE_TO_PLANNING",
        ),
        (
            range(FRAME_COUNT),
            range(FRAME_COUNT),
            range(24),
            (),
            "PLANNING_COLLECTOR",
        ),
        (
            range(FRAME_COUNT),
            range(FRAME_COUNT),
            range(FRAME_COUNT),
            range(24),
            "EXECUTION_ACCOUNTING",
        ),
    ],
)
def test_p0_j_first_loss_precedes_downstream_empty_ledgers(
    bridge_indices,
    planning_indices,
    collector_indices,
    accounted_indices,
    expected_layer,
):
    ledger = _build(
        unity_records=_unity(),
        bridge_records=_bridge(bridge_indices),
        planning_records=_planning(planning_indices),
        collector_records=_collector(collector_indices),
        accounting_records=_accounted(accounted_indices),
    )

    assert ledger["first_loss_layer"] == expected_layer
    assert ledger["first_loss_frame_index"] == 24


def test_p0_j_real_red_fixture_classifies_unity_pub_to_bridge():
    from planning.diagnostics.execution_transport_report import build_report

    fixture = _load_p0_j_fixture()
    assert fixture["source_execution_id"] == 4435486932233309751
    assert fixture["bridge"]["episode_id"] == -1
    assert fixture["unity"]["episode_id"] is None
    report = build_report(
        planning_payload=fixture["planning"],
        bridge_payload=fixture["bridge"],
        unity_payload=fixture["unity"],
        execution_id=fixture["source_execution_id"],
    )

    assert report["ledger"]["unity_applied_count"] == 25
    assert report["ledger"]["unity_send_success_count"] == 25
    assert report["ledger"]["bridge_received_count"] == 24
    assert report["ledger"]["planning_received_count"] == 24
    assert report["ledger"]["first_loss_layer"] == "UNITY_PUB_TO_BRIDGE"
    assert report["ledger"]["first_loss_frame_index"] == 24


def test_p0_j_real_fixture_rejects_concrete_wrong_bridge_episode():
    from planning.diagnostics.execution_transport_report import build_report

    fixture = _load_p0_j_fixture()
    wrong_bridge = {**fixture["bridge"], "episode_id": 999999}
    with pytest.raises(ValueError, match="episode identities differ"):
        build_report(
            planning_payload=fixture["planning"],
            bridge_payload=wrong_bridge,
            unity_payload=fixture["unity"],
            execution_id=fixture["source_execution_id"],
        )


def test_report_rejects_wrong_audit_contract_or_schema_version():
    from planning.diagnostics.execution_transport_report import build_report

    planning_payload = {
        "contract_id": "wrong",
        "audit_schema_version": 2,
        "episode_id": 81660,
        "executions": [],
    }
    with pytest.raises(ValueError, match="planning audit contract"):
        build_report(
            planning_payload=planning_payload,
            bridge_payload=None,
            unity_payload=None,
        )


def test_report_rejects_mixed_run_identity_and_wrong_unity_frame_count():
    from planning.diagnostics.execution_transport_report import build_report

    records = _unity()
    records[0] = {**records[0], "frame_count": FRAME_COUNT - 1}
    planning_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit",
        "audit_schema_version": 2,
        "audit_run_id": AUDIT_RUN_ID,
        "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
        "episode_id": 81660,
        "executions": [{
            "execution_id": EXECUTION_ID,
            "requested_frame_count": FRAME_COUNT,
            "command_ids": [1000 + index for index in range(FRAME_COUNT)],
            "planning_received": _planning(range(FRAME_COUNT)),
            "collector_received": _collector(range(FRAME_COUNT)),
            "execution_accounted": _accounted(range(FRAME_COUNT)),
            "collector_result": {"kind": "COMPLETED"},
        }],
    }
    bridge_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit",
        "audit_schema_version": 2,
        "audit_run_id": AUDIT_RUN_ID,
        "episode_id": 81660,
        "unity_runtime_identity": UNITY_RUNTIME_IDENTITY,
        "records": _bridge(range(FRAME_COUNT)),
    }
    unity_payload = {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]unity_execution_transport_audit",
        "audit_schema_version": 2,
        "episode_id": 81660,
        "audit_run_id": AUDIT_RUN_ID,
        "runtime_identity": UNITY_RUNTIME_IDENTITY,
        "records": records,
    }

    with pytest.raises(ValueError, match="frame count"):
        build_report(
            planning_payload=planning_payload,
            bridge_payload=bridge_payload,
            unity_payload=unity_payload,
            execution_id=EXECUTION_ID,
        )

    unity_payload["records"][0]["frame_count"] = FRAME_COUNT
    unity_payload["audit_run_id"] = "different-run"
    with pytest.raises(ValueError, match="run identities"):
        build_report(
            planning_payload=planning_payload,
            bridge_payload=bridge_payload,
            unity_payload=unity_payload,
            execution_id=EXECUTION_ID,
        )
