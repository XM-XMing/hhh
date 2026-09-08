"""Audit-only localization for schema-v3 primitive acknowledgements.

The execution contract remains the authority for whether a primitive completed.
This module only compares independently captured frame ledgers to identify the
first boundary at which an acknowledgement is absent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


EXECUTION_TRANSPORT_AUDIT_CONTRACT_ID = (
    "[DEBUG-EXEC-TRANSPORT-81660]execution_transport_audit"
)


def _records_by_frame(
    records: Iterable[Mapping[str, Any]],
    *,
    execution_id: int,
    ledger_name: str,
) -> Dict[int, Mapping[str, Any]]:
    """Keep the first audit record for each real applied frame index.

    Records outside the target execution and terminal-failure records with
    frame index ``-1`` are intentionally excluded from the exact-N ledger.
    """

    result: Dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("{} audit record is not an object".format(ledger_name))
        if int(record.get("execution_id", -1)) != int(execution_id):
            continue
        index = int(record.get("frame_index", -1))
        if index < 0:
            continue
        if index in result:
            raise ValueError(
                "duplicate {} audit record for execution_id={} frame_index={}".format(
                    ledger_name, execution_id, index
                )
            )
        result[index] = record
    return result


def _has(record: Mapping[str, Any] | None, key: str) -> bool:
    return record is not None and bool(record.get(key, False))


def _first_frame_matching(frames: Sequence[Mapping[str, Any]], predicate) -> int | None:
    """Return the first frame matching a boundary predicate, if any."""

    for frame in frames:
        if predicate(frame):
            return int(frame["frame_index"])
    return None


def _validate_shared_identity(
    *,
    frame_index: int,
    records: Sequence[tuple[str, Mapping[str, Any] | None]],
) -> None:
    """Reject a cross-run or corrupt join before diagnosing a loss boundary."""

    for field in ("command_id", "state_id", "sim_time_ns", "execution_status"):
        observed = {
            record[field]
            for _, record in records
            if record is not None and field in record
        }
        if len(observed) > 1:
            sources = ", ".join(
                "{}={}".format(name, record[field])
                for name, record in records
                if record is not None and field in record
            )
            raise ValueError(
                "cross-ledger {} mismatch at frame_index={}: {}".format(
                    field, frame_index, sources
                )
            )


def build_execution_transport_ledger(
    *,
    execution_id: int,
    expected_frame_count: int,
    unity_records: Sequence[Mapping[str, Any]],
    bridge_records: Sequence[Mapping[str, Any]],
    planning_records: Sequence[Mapping[str, Any]],
    collector_records: Sequence[Mapping[str, Any]],
    accounting_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a frame ledger and the earliest missing acknowledgement layer.

    This intentionally does not infer absent acknowledgements from endpoint
    motion or state IDs.  Every boolean describes a record actually observed
    at its named boundary.
    """

    count = int(expected_frame_count)
    if count <= 0:
        raise ValueError("expected_frame_count must be positive")

    unity = _records_by_frame(
        unity_records, execution_id=execution_id, ledger_name="Unity"
    )
    bridge = _records_by_frame(
        bridge_records, execution_id=execution_id, ledger_name="bridge"
    )
    planning = _records_by_frame(
        planning_records, execution_id=execution_id, ledger_name="planning callback"
    )
    collector = _records_by_frame(
        collector_records, execution_id=execution_id, ledger_name="planning collector"
    )
    accounting = _records_by_frame(
        accounting_records, execution_id=execution_id, ledger_name="execution accounting"
    )

    frames = []
    unity_available = bool(unity)
    first_loss_layer = None
    first_loss_index = None
    for index in range(count):
        unity_record = unity.get(index)
        bridge_record = bridge.get(index)
        planning_record = planning.get(index)
        collector_record = collector.get(index)
        accounting_record = accounting.get(index)
        _validate_shared_identity(
            frame_index=index,
            records=(
                ("Unity", unity_record),
                ("bridge", bridge_record),
                ("planning callback", planning_record),
                ("planning collector", collector_record),
                ("execution accounting", accounting_record),
            ),
        )
        if unity_available and bridge_record is not None and (
            not _has(unity_record, "send_attempted")
            or not _has(unity_record, "send_ok")
        ):
            raise ValueError(
                "Unity send failure conflicts with a bridge receipt at frame_index={}"
                .format(index)
            )
        row = {
            "frame_index": int(index),
            "unity_applied": _has(unity_record, "applied") if unity_available else None,
            "unity_send_attempted": (
                _has(unity_record, "send_attempted") if unity_available else None
            ),
            "unity_send_ok": _has(unity_record, "send_ok") if unity_available else None,
            "bridge_received": _has(bridge_record, "received"),
            "bridge_forwarded": _has(bridge_record, "forwarded"),
            "planning_received": _has(planning_record, "received"),
            "collector_received": _has(collector_record, "collected"),
            "execution_accounted": _has(accounting_record, "accounted"),
        }
        frames.append(row)

    # Diagnose the earliest missing *boundary*, not the first row whose
    # downstream accounting happens to be empty.  A failed primitive may have
    # no execution-accounted records at all; that is a consequence of a loss
    # upstream, not evidence that accounting was the first physical loss.
    if not unity_available:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["bridge_received"]
        )
        if first_loss_index is not None:
            first_loss_layer = "UNRESOLVED"
    else:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["unity_applied"]
        )
        if first_loss_index is not None:
            first_loss_layer = "EXECUTION_ACCOUNTING"
        else:
            first_loss_index = _first_frame_matching(
                frames,
                lambda frame: (
                    not frame["unity_send_attempted"]
                    and not frame["bridge_received"]
                )
                or (not frame["unity_send_ok"] and not frame["bridge_received"]),
            )
            if first_loss_index is not None:
                first_loss_layer = "UNITY_SEND_ATTEMPT"

    if first_loss_layer is None:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["bridge_received"]
        )
        if first_loss_index is not None:
            first_loss_layer = "UNITY_PUB_TO_BRIDGE"

    if first_loss_layer is None:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["planning_received"]
        )
        if first_loss_index is not None:
            first_loss_layer = "BRIDGE_TO_PLANNING"

    if first_loss_layer is None:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["collector_received"]
        )
        if first_loss_index is not None:
            first_loss_layer = "PLANNING_COLLECTOR"

    if first_loss_layer is None:
        first_loss_index = _first_frame_matching(
            frames, lambda frame: not frame["execution_accounted"]
        )
        if first_loss_index is not None:
            first_loss_layer = "EXECUTION_ACCOUNTING"

    missing = [
        int(row["frame_index"])
        for row in frames
        if not bool(row["execution_accounted"])
    ]
    return {
        "contract_id": EXECUTION_TRANSPORT_AUDIT_CONTRACT_ID,
        "execution_id": int(execution_id),
        "expected_frame_count": count,
        "unity_audit_available": unity_available,
        "unity_applied_count": sum(bool(row["unity_applied"]) for row in frames),
        "unity_send_attempt_count": sum(
            bool(row["unity_send_attempted"]) for row in frames
        ),
        "unity_send_success_count": sum(bool(row["unity_send_ok"]) for row in frames),
        "bridge_received_count": sum(bool(row["bridge_received"]) for row in frames),
        "bridge_forwarded_count": sum(bool(row["bridge_forwarded"]) for row in frames),
        "planning_received_count": sum(bool(row["planning_received"]) for row in frames),
        "collector_received_count": sum(bool(row["collector_received"]) for row in frames),
        "execution_accounted_count": sum(bool(row["execution_accounted"]) for row in frames),
        "missing_frame_indices": missing,
        "first_loss_layer": first_loss_layer,
        "first_loss_frame_index": first_loss_index,
        "frames": frames,
    }
