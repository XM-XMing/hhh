"""Run identity, aggregate-target, and resume contracts for formal collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Optional, Set

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    validate_reliable_exact_metadata,
)
from planning.common.config import parse_bool


class CollectionTargetError(RuntimeError):
    """The worker pool ended before its global accepted target was met."""


def aggregate_target_reached(accepted_total: int, target_accepted: int) -> bool:
    """Return whether the global accepted target is satisfied."""

    target = int(target_accepted)
    return target > 0 and int(accepted_total) >= target


def aggregate_target_status(
    accepted_total: int, target_accepted: int, *, active_workers: int
) -> str:
    """Classify aggregate progress without converting it to a per-worker quota."""

    target = int(target_accepted)
    if target <= 0:
        return "TARGET_DISABLED"
    if aggregate_target_reached(accepted_total, target):
        return "TARGET_REACHED"
    if int(active_workers) <= 0:
        return "TARGET_UNREACHED"
    return "RUNNING"


def require_aggregate_target(
    accepted_total: int, target_accepted: int, *, active_workers: int
) -> None:
    status = aggregate_target_status(
        accepted_total, target_accepted, active_workers=active_workers
    )
    if status == "TARGET_UNREACHED":
        raise CollectionTargetError(
            "global accepted target was not reached: accepted={} target={}".format(
                int(accepted_total), int(target_accepted)
            )
        )


def _transition_ids(row: dict) -> Set[str]:
    value = row.get("transition_ids", "")
    if not value:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(item) for item in value if str(item)}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = [part for part in text.split(",") if part.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("transition_ids must be a list")
    return {str(item) for item in parsed if str(item)}


@dataclass
class CollectionResumeLedger:
    """Idempotency ledger reconstructed from a worker report on resume."""

    runtime_instance_id: str
    observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT
    run_id: Optional[str] = None
    attempted_episode_ids: Set[int] = field(default_factory=set)
    attempted_mission_ids: Set[str] = field(default_factory=set)
    accepted_episode_ids: Set[int] = field(default_factory=set)
    transition_ids: Set[str] = field(default_factory=set)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[dict],
        *,
        runtime_instance_id: str,
        run_id: Optional[str] = None,
        observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    ) -> "CollectionResumeLedger":
        ledger = cls(
            runtime_instance_id=str(runtime_instance_id),
            observation_contract=str(observation_contract),
            run_id=None if run_id is None else str(run_id),
        )
        for row in rows:
            ledger.add_row(row)
        return ledger

    def contains_episode(self, episode_id: int) -> bool:
        return int(episode_id) in self.attempted_episode_ids

    def contains_mission(self, mission_id: str) -> bool:
        return str(mission_id) in self.attempted_mission_ids

    @property
    def attempted_total(self) -> int:
        return len(self.attempted_episode_ids)

    @property
    def accepted_total(self) -> int:
        return len(self.accepted_episode_ids)

    def next_transition_id_offset(self) -> int:
        """Return an offset that keeps a restarted runtime's IDs unique.

        Unity execution IDs are scoped to one runtime process and can restart
        at zero after a resume.  The persisted transition identity is scoped
        to the collection run, so a resumed session must allocate a namespace
        above the largest persisted numeric suffix for this runtime.
        """

        prefix = self.runtime_instance_id + ":"
        maximum = -1
        for transition_id in self.transition_ids:
            text = str(transition_id)
            if not text.startswith(prefix):
                continue
            suffix = text[len(prefix) :]
            try:
                maximum = max(maximum, int(suffix))
            except ValueError:
                continue
        return maximum + 1

    def add_row(self, row: dict) -> None:
        if not isinstance(row, dict):
            raise ValueError("resume row must be a mapping")
        validate_reliable_exact_metadata(row, path="collection report row")
        row_runtime = str(row.get("runtime_instance_id", "")).strip()
        if row_runtime != self.runtime_instance_id:
            raise ValueError(
                "resume row runtime identity mismatch: {} != {}".format(
                    row_runtime, self.runtime_instance_id
                )
            )
        row_contract = str(row.get("observation_contract", "")).strip()
        if row_contract != self.observation_contract:
            raise ValueError(
                "resume row observation contract mismatch: {} != {}".format(
                    row_contract, self.observation_contract
                )
            )
        if self.run_id is not None:
            row_run_id = str(row.get("collection_run_id", "")).strip()
            if not row_run_id or row_run_id != self.run_id:
                raise ValueError(
                    "resume row collection run identity mismatch: {} != {}".format(
                        row_run_id, self.run_id
                    )
                )
        try:
            episode_id = int(float(row["episode_id"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("resume row episode_id is invalid")
        mission_id = str(row.get("mission_id", "")).strip()
        if not mission_id:
            raise ValueError("resume row mission_id is missing")
        if episode_id in self.attempted_episode_ids:
            raise ValueError("duplicate episode_id {} in resume ledger".format(episode_id))
        if mission_id in self.attempted_mission_ids:
            raise ValueError("duplicate mission_id {} in resume ledger".format(mission_id))
        row_transition_ids = _transition_ids(row)
        duplicate_transitions = self.transition_ids.intersection(row_transition_ids)
        if duplicate_transitions:
            raise ValueError(
                "duplicate transition_id(s) {} in resume ledger".format(
                    sorted(duplicate_transitions)
                )
            )
        if parse_bool(row.get("execute_ok", False)) and not row_transition_ids:
            raise ValueError("accepted resume row has no transition identity")
        self.attempted_episode_ids.add(episode_id)
        self.attempted_mission_ids.add(mission_id)
        self.transition_ids.update(row_transition_ids)
        if parse_bool(row.get("execute_ok", False)):
            self.accepted_episode_ids.add(episode_id)


__all__ = [
    "CollectionResumeLedger",
    "CollectionTargetError",
    "aggregate_target_reached",
    "aggregate_target_status",
    "require_aggregate_target",
]
