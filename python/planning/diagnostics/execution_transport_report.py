#!/usr/bin/env python3
"""Produce the frame-level ledger for one audited primitive execution.

This command is deliberately offline and audit-only.  It never treats a
missing acknowledgement as a completed frame and it does not alter the
primitive execution contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from planning.diagnostics.execution_transport_audit import build_execution_transport_ledger


PLANNING_AUDIT_CONTRACT_ID = "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit"
BRIDGE_AUDIT_CONTRACT_ID = "[DEBUG-EXEC-TRANSPORT-81660]bridge_state_audit"
UNITY_AUDIT_CONTRACT_ID = "[DEBUG-EXEC-TRANSPORT-81660]unity_execution_transport_audit"
AUDIT_SCHEMA_VERSION = 2


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit payload must be an object: {}".format(path))
    return payload


def _records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("audit records must be a list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("audit records must contain only objects")
    return list(records)


def _validate_audit_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    contract_id: str,
) -> None:
    if payload.get("contract_id") != contract_id:
        raise ValueError("{} audit contract mismatch".format(label))
    if payload.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            "{} audit schema version must be {}".format(label, AUDIT_SCHEMA_VERSION)
        )
    _records(payload)
    if bool(payload.get("capture_overflow", False)):
        raise ValueError("{} audit capture overflow".format(label))


def _unity_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Map the Unity Player's audit schema onto the common ledger schema.

    ``try_send_return`` records only the local NetMQ API result.  It is not
    reinterpreted as delivery to the bridge.
    """

    normalized = []
    for record in _records(payload):
        normalized.append({
            **record,
            "applied": record.get("frame_applied", record.get("applied", False)),
            "send_attempted": record.get(
                "state_publish_attempted", record.get("send_attempted", False)
            ),
            "send_ok": record.get("try_send_return", record.get("send_ok", False)),
        })
    return normalized


def _select_execution(
    executions: Sequence[Mapping[str, Any]], execution_id: int | None
) -> Mapping[str, Any]:
    if execution_id is not None:
        matches = [
            execution
            for execution in executions
            if int(execution.get("execution_id", -1)) == int(execution_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one audited execution_id={}, found {}".format(
                    execution_id, len(matches)
                )
            )
        return matches[0]

    incomplete = []
    for execution in executions:
        expected = int(execution.get("requested_frame_count", 0))
        accounted = list(execution.get("execution_accounted", []))
        result = dict(execution.get("collector_result", {}))
        if len(accounted) != expected or result.get("kind") != "COMPLETED":
            incomplete.append(execution)
    if len(incomplete) != 1:
        raise ValueError(
            "cannot select one failed execution automatically; pass --execution-id "
            "(incomplete executions={})".format(len(incomplete))
        )
    return incomplete[0]


def _validate_execution_frame_contract(
    *,
    records: Sequence[Mapping[str, Any]],
    label: str,
    execution_id: int,
    requested_frame_count: int,
    command_ids: Sequence[int],
    require_frame_count: bool,
) -> None:
    """Bind ledger rows to the submitted execution before joining them."""

    for record in records:
        if int(record.get("execution_id", -1)) != int(execution_id):
            continue
        frame_index = int(record.get("frame_index", -1))
        if frame_index < 0:
            continue
        if frame_index >= requested_frame_count:
            raise ValueError("{} frame index exceeds requested frame count".format(label))
        if int(record.get("command_id", -1)) != int(command_ids[frame_index]):
            raise ValueError("{} command identity mismatches submitted execution".format(label))
        if require_frame_count and int(record.get("frame_count", -1)) != int(
            requested_frame_count
        ):
            raise ValueError("Unity frame count mismatches requested execution")


def build_report(
    *,
    planning_payload: Mapping[str, Any],
    bridge_payload: Mapping[str, Any] | None,
    unity_payload: Mapping[str, Any] | None,
    execution_id: int | None = None,
) -> dict[str, Any]:
    """Build one auditable first-loss report from saved endpoint ledgers."""

    _validate_audit_payload(
        planning_payload,
        label="planning",
        contract_id=PLANNING_AUDIT_CONTRACT_ID,
    )
    if bridge_payload is not None:
        _validate_audit_payload(
            bridge_payload,
            label="bridge",
            contract_id=BRIDGE_AUDIT_CONTRACT_ID,
        )
    if unity_payload is not None:
        _validate_audit_payload(
            unity_payload,
            label="Unity",
            contract_id=UNITY_AUDIT_CONTRACT_ID,
        )
        unity_episode_id = unity_payload.get("episode_id", -1)
        planning_episode_id = int(planning_payload.get("episode_id", -1))
        # Unity's runtime-global audit serializes an unbound episode as null
        # (older paths use -1).  A concrete wrong episode remains invalid.
        if unity_episode_id is not None and int(unity_episode_id) not in (
            -1,
            planning_episode_id,
        ):
            raise ValueError("Unity and planning audit episode identities differ")

    audit_run_id = str(planning_payload.get("audit_run_id", ""))
    unity_runtime_identity = str(
        planning_payload.get("unity_runtime_identity", "")
    )
    if not audit_run_id or not unity_runtime_identity:
        raise ValueError("planning audit is missing run/runtime identity")
    if bridge_payload is not None:
        if str(bridge_payload.get("audit_run_id", "")) != audit_run_id:
            raise ValueError("bridge and planning audit run identities differ")
        bridge_episode_id = int(bridge_payload.get("episode_id", -1))
        planning_episode_id = int(planning_payload.get("episode_id", -1))
        # The bridge audit is runtime-global in long single-worker runs and
        # therefore legitimately records episode_id=-1.  A concrete, wrong
        # episode identity must still fail before ledger classification.
        if bridge_episode_id not in (-1, planning_episode_id):
            raise ValueError("bridge and planning audit episode identities differ")
        if str(bridge_payload.get("unity_runtime_identity", "")) != (
            unity_runtime_identity
        ):
            raise ValueError("bridge and planning Unity runtime identities differ")
    if unity_payload is not None:
        if str(unity_payload.get("audit_run_id", "")) != audit_run_id:
            raise ValueError("Unity and planning audit run identities differ")
        if str(unity_payload.get("runtime_identity", "")) != unity_runtime_identity:
            raise ValueError("Unity runtime identity does not match planning expectation")

    executions = planning_payload.get("executions", [])
    if not isinstance(executions, list):
        raise ValueError("planning audit executions must be a list")
    execution = _select_execution(executions, execution_id)
    selected_id = int(execution["execution_id"])
    expected = int(execution["requested_frame_count"])
    command_ids = [int(value) for value in execution.get("command_ids", [])]
    if len(command_ids) != expected:
        raise ValueError("submitted command identity count mismatches requested frame count")
    _validate_execution_frame_contract(
        records=list(execution.get("planning_received", [])),
        label="planning callback",
        execution_id=selected_id,
        requested_frame_count=expected,
        command_ids=command_ids,
        require_frame_count=False,
    )
    _validate_execution_frame_contract(
        records=list(execution.get("collector_received", [])),
        label="planning collector",
        execution_id=selected_id,
        requested_frame_count=expected,
        command_ids=command_ids,
        require_frame_count=False,
    )
    _validate_execution_frame_contract(
        records=list(execution.get("execution_accounted", [])),
        label="execution accounting",
        execution_id=selected_id,
        requested_frame_count=expected,
        command_ids=command_ids,
        require_frame_count=False,
    )
    if bridge_payload is not None:
        _validate_execution_frame_contract(
            records=_records(bridge_payload),
            label="bridge",
            execution_id=selected_id,
            requested_frame_count=expected,
            command_ids=command_ids,
            require_frame_count=False,
        )
    if unity_payload is not None:
        _validate_execution_frame_contract(
            records=_unity_records(unity_payload),
            label="Unity",
            execution_id=selected_id,
            requested_frame_count=expected,
            command_ids=command_ids,
            require_frame_count=True,
        )
    collector_result = dict(execution.get("collector_result", {}))
    terminal_abort = collector_result.get("kind") == "TERMINAL_ABORT"
    ledger_expected = expected
    if terminal_abort:
        # A validated terminal abort intentionally has no completed N-frame
        # endpoint.  Its audit compares only the real applied prefix; treating
        # its unexecuted suffix as a transport loss would be false evidence.
        ledger_expected = len([
            record
            for record in execution.get("collector_received", [])
            if int(record.get("frame_index", -1)) >= 0
        ])
        if ledger_expected <= 0:
            raise ValueError("terminal-abort audit is missing its applied prefix")
    ledger = build_execution_transport_ledger(
        execution_id=selected_id,
        expected_frame_count=ledger_expected,
        unity_records=(
            _unity_records(unity_payload) if unity_payload is not None else []
        ),
        bridge_records=(
            _records(bridge_payload) if bridge_payload is not None else []
        ),
        planning_records=list(execution.get("planning_received", [])),
        collector_records=list(execution.get("collector_received", [])),
        accounting_records=list(execution.get("execution_accounted", [])),
    )
    first_loss_detail = None
    if ledger.get("first_loss_layer") == "UNITY_PUB_TO_BRIDGE":
        first_loss_detail = (
            "Unity TrySendFrame succeeded, but the bridge has no matching frame; "
            "the exact sublayer remains unresolved between the NetMQ PUB pipe, "
            "transport, and bridge receive boundary."
        )
    return {
        "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]first_loss_report",
        "episode_id": int(planning_payload.get("episode_id", -1)),
        "decision_index": planning_payload.get("step"),
        "selected_action": planning_payload.get("selected_action"),
        "failure_kind": planning_payload.get("failure_kind"),
        "failure_error": planning_payload.get("error"),
        "execution_id": selected_id,
        "requested_frame_count": expected,
        "terminal_abort": terminal_abort,
        "command_ids": command_ids,
        "collector_terminal": execution.get("collector_terminal"),
        "collector_result": collector_result,
        "bridge_audit_available": bridge_payload is not None,
        "unity_audit_available": unity_payload is not None,
        "first_loss_detail": first_loss_detail,
        "unity_runtime_identity": (
            unity_payload.get("runtime_identity") if unity_payload is not None else None
        ),
        "unity_configured_send_hwm": (
            unity_payload.get("configured_send_hwm")
            if unity_payload is not None
            else None
        ),
        "unity_configured_recv_hwm": (
            unity_payload.get("configured_recv_hwm")
            if unity_payload is not None
            else None
        ),
        "bridge_recv_hwm": (
            bridge_payload.get("recv_hwm") if bridge_payload is not None else None
        ),
        "bridge_send_hwm": (
            bridge_payload.get("send_hwm") if bridge_payload is not None else None
        ),
        "ledger": ledger,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--unity", type=Path)
    parser.add_argument("--execution-id", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    planning = _read_json(args.planning)
    bridge = _read_json(args.bridge) if args.bridge is not None else None
    unity = _read_json(args.unity) if args.unity is not None else None
    report = build_report(
        planning_payload=planning,
        bridge_payload=bridge,
        unity_payload=unity,
        execution_id=args.execution_id,
    )
    out = args.out or args.planning.with_name("execution_transport_first_loss.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ledger = report["ledger"]
    print(
        "EXECUTION_TRANSPORT_FIRST_LOSS episode={} decision={} action={} "
        "execution_id={} expected={} accepted={} missing={} first_loss={}".format(
            report["episode_id"],
            report["decision_index"],
            report["selected_action"],
            report["execution_id"],
            ledger["expected_frame_count"],
            ledger["execution_accounted_count"],
            ledger["missing_frame_indices"],
            ledger["first_loss_layer"],
        )
    )
    print("report={}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
