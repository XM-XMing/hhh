"""Owned subprocess lifecycle for runtime workers."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, TextIO


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    stdout_handle: Optional[TextIO] = None

    @classmethod
    def start(
        cls,
        *,
        name: str,
        argv: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        log_path: Optional[Path] = None,
        append_log: bool = False,
    ) -> "ManagedProcess":
        handle = None
        stdout = None
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a" if append_log else "w", encoding="utf-8")
            stdout = handle
        try:
            process = subprocess.Popen(
                [str(value) for value in argv],
                env=dict(env) if env is not None else None,
                stdout=stdout,
                stderr=subprocess.STDOUT if stdout is not None else None,
                start_new_session=True,
                text=True,
            )
        except Exception:
            if handle is not None:
                handle.close()
            raise
        return cls(name=name, process=process, stdout_handle=handle)

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> Optional[int]:
        return self.process.poll()

    def is_alive(self) -> bool:
        return self.poll() is None

    def terminate(self) -> None:
        if not self.is_alive():
            return
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

    def kill(self) -> None:
        if not self.is_alive():
            return
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGKILL)
        except ProcessLookupError:
            return

    def wait(self, timeout: Optional[float] = None) -> int:
        try:
            return int(self.process.wait(timeout=timeout))
        finally:
            if self.stdout_handle is not None and not self.stdout_handle.closed:
                self.stdout_handle.close()
