"""Structured run logging with a stable cross-pipeline field contract."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_LOG_FIELDS = (
    "timestamp",
    "run_id",
    "mission_id",
    "episode",
    "step",
    "component",
    "duration_ms",
    "memory_mb",
    "gpu_memory_mb",
    "status",
    "error_type",
)


def _resident_memory_mb() -> Optional[float]:
    """Return current resident memory without adding a third-party dependency."""

    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return round(
            resident_pages * int(os.sysconf("SC_PAGE_SIZE")) / (1024.0 * 1024.0),
            3,
        )
    except (IndexError, OSError, ValueError):
        return None


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            record.structured_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class StructuredRunLogger:
    """Append JSONL events while guaranteeing the common monitoring fields."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        component: str,
        gpu_sample_interval_s: float = 30.0,
        sample_gpu: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.component = str(component)
        self.sample_gpu = bool(sample_gpu)
        self.gpu_sample_interval_s = max(1.0, float(gpu_sample_interval_s))
        self._gpu_memory_mb: Optional[float] = None
        self._gpu_sample_time = -float("inf")

        logger_name = "planning.run.{}.{}.{}".format(
            self.component, self.run_id, id(self)
        )
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = logging.FileHandler(str(self.path), mode="a", encoding="utf-8")
        handler.setFormatter(_JsonFormatter())
        self._logger.addHandler(handler)

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def _sample_gpu_memory_mb(self) -> Optional[float]:
        if not self.sample_gpu:
            return None
        now = time.monotonic()
        if now - self._gpu_sample_time < self.gpu_sample_interval_s:
            return self._gpu_memory_mb
        self._gpu_sample_time = now
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
            process_id = str(os.getpid())
            used = []
            for line in completed.stdout.splitlines():
                columns = [column.strip() for column in line.split(",")]
                if len(columns) == 2 and columns[0] == process_id:
                    used.append(float(columns[1]))
            self._gpu_memory_mb = round(sum(used), 3) if used else 0.0
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            self._gpu_memory_mb = None
        return self._gpu_memory_mb

    def event(
        self,
        status: str,
        *,
        mission_id: Any = "",
        episode: int = -1,
        step: int = -1,
        duration_ms: float = 0.0,
        error_type: str = "",
        **details: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "run_id": self.run_id,
            "mission_id": mission_id,
            "episode": int(episode),
            "step": int(step),
            "component": self.component,
            "duration_ms": round(float(duration_ms), 3),
            "memory_mb": _resident_memory_mb(),
            "gpu_memory_mb": self._sample_gpu_memory_mb(),
            "status": str(status),
            "error_type": str(error_type),
        }
        payload.update(details)
        self._logger.info("", extra={"structured_payload": payload})
        return payload
