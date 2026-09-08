"""Small append-only logging primitives for bounded pipeline runs.

The collection data contracts already own machine-readable progress and
summary files.  This module only owns human-facing session logs and the
append-only command/config identity records that make a resume diagnosable.
It intentionally does not configure the process-wide :mod:`logging` module or
capture the environment.
"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, TextIO, Tuple

from planning.common.io import write_json_atomic


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _worker_name(worker_id: int) -> str:
    return "worker_{:02d}".format(int(worker_id))


def collection_log_path(out_dir: Path) -> Path:
    return Path(out_dir) / "logs" / "collection.log"


def worker_log_path(out_dir: Path, worker_id: int) -> Path:
    return Path(out_dir) / "logs" / "workers" / (_worker_name(worker_id) + ".log")


def unity_log_path(out_dir: Path, worker_id: int) -> Path:
    return Path(out_dir) / "logs" / "unity" / (_worker_name(worker_id) + ".log")


def bridge_log_path(out_dir: Path, worker_id: int) -> Path:
    return Path(out_dir) / "logs" / "bridge" / (_worker_name(worker_id) + ".log")


def roscore_log_path(out_dir: Path, worker_id: int) -> Path:
    return Path(out_dir) / "logs" / "roscore" / (_worker_name(worker_id) + ".log")


def preparation_log_path(run_dir: Path) -> Path:
    return Path(run_dir) / "logs" / "preparation.log"


def prepare_log_directories(out_dir: Path, worker_count: int = 0) -> None:
    """Create the stable log layout without creating data artifacts."""

    root = Path(out_dir) / "logs"
    for name in ("workers", "unity", "bridge", "roscore"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if int(worker_count) > 0:
        for worker_id in range(int(worker_count)):
            for path in (
                worker_log_path(out_dir, worker_id),
                unity_log_path(out_dir, worker_id),
                bridge_log_path(out_dir, worker_id),
                roscore_log_path(out_dir, worker_id),
            ):
                path.touch(exist_ok=True)


def _format_fields(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    # JSON makes paths, booleans, and error details unambiguous while keeping
    # the line human-readable.  Callers pass only resolved, non-secret data.
    return " " + json.dumps(dict(fields), ensure_ascii=False, sort_keys=True, default=str)


class CollectionLogSession:
    """Timestamped console+file log with append-only resume sessions."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        command: str,
        console: Optional[TextIO] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._console = console if console is not None else sys.stdout
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.run_id = str(run_id)
        self.command = str(command)
        self._closed = False
        self._write_failed = False
        had_previous = self.path.stat().st_size > 0
        marker = "RESUME SESSION" if had_previous else "SESSION START"
        self.event(
            marker,
            run_id=self.run_id,
            command=self.command,
            session_kind="resume" if had_previous else "initial",
        )

    def event(self, message: str, *, level: str = "INFO", **fields: Any) -> str:
        if self._closed:
            raise RuntimeError("collection log session is closed")
        line = "{} [{}] {}{}".format(
            _timestamp(),
            str(level).upper(),
            str(message),
            _format_fields(fields),
        )
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
        except (OSError, ValueError) as error:
            self._write_failed = True
            _log_fallback("collection log write failed", error, line)
        if self._console is not None:
            try:
                print(line, file=self._console, flush=True)
            except (OSError, ValueError) as error:
                _log_fallback("collection console write failed", error, line)
        return line

    def exception(self, error: BaseException, *, message: str = "exception", **fields: Any) -> str:
        details = dict(fields)
        details.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )
        return self.event(message, level="ERROR", **details)

    def close(self, status: str = "UNKNOWN") -> None:
        if self._closed:
            return
        self.event("SESSION END", status=str(status))
        self._closed = True
        try:
            self._handle.flush()
            self._handle.close()
        except (OSError, ValueError) as error:
            self._write_failed = True
            _log_fallback("collection log close failed", error, str(self.path))

    def __enter__(self) -> "CollectionLogSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.exception(exc)
            self.close("FAIL")
        else:
            self.close("PASS")


def append_log_event(path: Path, message: str, *, level: str = "INFO", **fields: Any) -> None:
    """Append one timestamped event to a worker/process log."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = "{} [{}] {}{}\n".format(
        _timestamp(), str(level).upper(), str(message), _format_fields(fields)
    )
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    except (OSError, ValueError) as error:
        _log_fallback("append-only log write failed", error, line.rstrip("\n"))


def _log_fallback(message: str, error: BaseException, detail: str) -> None:
    """Best-effort stderr fallback that never masks runtime cleanup."""

    try:
        print(
            "{}: {}: {}".format(message, type(error).__name__, detail),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


def _json_projection(payload: Mapping[str, Any], keys: Sequence[str]) -> dict:
    return {str(key): payload.get(key) for key in keys}


def write_resume_manifest(
    path: Path,
    payload: Mapping[str, Any],
    *,
    immutable_keys: Sequence[str],
    schema: str,
) -> dict:
    """Persist a resume-safe manifest and reject identity changes.

    The first payload is retained as ``baseline``.  Later invocations append a
    session record to the same JSON document; no previous session log or
    command record is truncated.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current = dict(payload)
    keys = tuple(str(key) for key in immutable_keys)
    if not target.exists() or target.stat().st_size == 0:
        document = dict(current)
        document["schema"] = str(schema)
        document["baseline"] = _json_projection(current, keys)
        document["sessions"] = [
            {
                "session_index": 0,
                "timestamp": _timestamp(),
                **current,
            }
        ]
        write_json_atomic(target, document, trailing_newline=True)
        return document

    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("RESUME_IDENTITY_MISMATCH: unreadable manifest {}".format(target)) from error
    if not isinstance(existing, dict) or str(existing.get("schema", "")) != str(schema):
        raise ValueError("RESUME_IDENTITY_MISMATCH: manifest schema changed: {}".format(target))
    baseline = existing.get("baseline")
    if not isinstance(baseline, dict):
        baseline = _json_projection(existing, keys)
    actual = _json_projection(current, keys)
    mismatches = {
        key: {"expected": baseline.get(key), "actual": actual.get(key)}
        for key in keys
        if baseline.get(key) != actual.get(key)
    }
    if mismatches:
        raise ValueError(
            "RESUME_IDENTITY_MISMATCH: {}".format(
                json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
            )
        )
    sessions = existing.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("RESUME_IDENTITY_MISMATCH: manifest sessions are invalid")
    document = dict(existing)
    document.update(current)
    document["schema"] = str(schema)
    document["baseline"] = baseline
    document["sessions"] = sessions + [
        {
            "session_index": len(sessions),
            "timestamp": _timestamp(),
            **current,
        }
    ]
    write_json_atomic(target, document, trailing_newline=True)
    return document


class _TeeStream:
    def __init__(self, console: TextIO, file_handle: TextIO):
        self.console = console
        self.file_handle = file_handle
        self.file_write_failed = False

    def write(self, value: str) -> int:
        self.console.write(value)
        if not self.file_write_failed:
            try:
                self.file_handle.write(value)
            except (OSError, ValueError) as error:
                self.file_write_failed = True
                _log_fallback("mirrored log write failed", error, "output suppressed")
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        if not self.file_write_failed:
            try:
                self.file_handle.flush()
            except (OSError, ValueError) as error:
                self.file_write_failed = True
                _log_fallback("mirrored log flush failed", error, "output suppressed")

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())


@contextmanager
def mirror_command_output(
    path: Path,
    *,
    run_id: str,
    command: str,
) -> Iterator[CollectionLogSession]:
    """Mirror a preparation owner's existing stdout/stderr to one log."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", buffering=1) as handle:
        session = CollectionLogSession(
            target,
            run_id=run_id,
            command=command,
            console=None,
        )
        # The session opened its own append handle.  Keep the second handle
        # only for the algorithm's raw output mirror and close it after the
        # redirects are restored.
        stdout_stream = _TeeStream(sys.stdout, handle)
        stderr_stream = _TeeStream(sys.stderr, handle)
        status = "PASS"
        try:
            with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                yield session
        except BaseException as error:
            status = "FAIL"
            session.exception(error)
            raise
        finally:
            session.close(status)


__all__ = [
    "CollectionLogSession",
    "append_log_event",
    "bridge_log_path",
    "collection_log_path",
    "mirror_command_output",
    "preparation_log_path",
    "prepare_log_directories",
    "roscore_log_path",
    "unity_log_path",
    "worker_log_path",
    "write_resume_manifest",
]
