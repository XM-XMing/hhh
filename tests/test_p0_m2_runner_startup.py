"""P0-M2 runner startup orchestration tests.

These tests exercise the runner process boundary with tiny local stand-ins for
roscore, the bridge, and Unity. They do not exercise protocol semantics or
primitive execution.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_roscore(path: Path, pid_path: Path) -> Path:
    source = """#!/usr/bin/env python3
import os
import signal
import socket
import sys

port = int(sys.argv[sys.argv.index("-p") + 1])
open("__PID__", "w").write(str(os.getpid()))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen(8)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    server.settimeout(0.1)
    try:
        client, _ = server.accept()
        client.close()
    except socket.timeout:
        pass
"""
    return _write_executable(path, source.replace("__PID__", str(pid_path)))


def _fake_bridge(path: Path, *, pid_path: Path, fail: bool = False) -> Path:
    if fail:
        source = """#!/usr/bin/env python3
import os
open("__PID__", "w").write(str(os.getpid()))
raise SystemExit(7)
"""
        return _write_executable(path, source.replace("__PID__", str(pid_path)))
    source = """#!/usr/bin/env python3
import os
import signal
import sys
import time
import zmq

args = {
    item.split(":=", 1)[0]: item.split(":=", 1)[1]
    for item in sys.argv[1:] if ":=" in item
}
open("__PID__", "w").write(str(os.getpid()))
context = zmq.Context()
sockets = []
for key in ("_execution_result_port", "_python_result_port"):
    socket_obj = context.socket(zmq.ROUTER)
    socket_obj.bind("tcp://*:" + args[key])
    sockets.append(socket_obj)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
"""
    return _write_executable(path, source.replace("__PID__", str(pid_path)))


def _fake_unity(
    path: Path, *, pid_path: Path, bind_delay_s: float = 0.0
) -> Path:
    source = """#!/usr/bin/env python3
import os
import signal
import sys
import time
import zmq

args = {
    sys.argv[index]: sys.argv[index + 1]
    for index in range(1, len(sys.argv) - 1)
    if sys.argv[index].startswith("-")
}
open("__PID__", "w").write(str(os.getpid()))
time.sleep(__DELAY__)
context = zmq.Context()
command = context.socket(zmq.SUB)
command.setsockopt(zmq.SUBSCRIBE, b"")
command.bind("tcp://*:" + args["-cmdSubPort"])
state = context.socket(zmq.PUB)
state.bind("tcp://*:" + args["-statePubPort"])
depth = context.socket(zmq.PUB)
depth.bind("tcp://*:" + args["-depthPubPort"])
result = context.socket(zmq.DEALER)
result.connect("tcp://127.0.0.1:" + args["-executionResultPort"])
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
"""
    return _write_executable(
        path,
        source.replace("__PID__", str(pid_path)).replace(
            "__DELAY__", repr(float(bind_delay_s))
        ),
    )


def _pid_is_alive(pid_path: Path) -> bool:
    if not pid_path.is_file():
        return False
    pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _runtime(
    tmp_path: Path, *, unity: Path, bridge: Path, timeout: float
):
    from planning.diagnostics.reliable_single_worker import _RealRuntime

    return _RealRuntime(
        out_dir=tmp_path,
        unity_binary=unity,
        bridge_binary=bridge,
        runtime_instance_id="worker-startup-test",
        episode_id="startup-episode",
        reset_id="startup-reset",
        startup_timeout_s=timeout,
        primitive_timeout_s=1.0,
        telemetry_buffer_size=8,
    )


def test_delayed_unity_starts_after_bridge_ready(tmp_path, monkeypatch):
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    _fake_roscore(tmp_path / "roscore", pid_dir / "roscore.pid")
    bridge = _fake_bridge(tmp_path / "bridge", pid_path=pid_dir / "bridge.pid")
    unity = _fake_unity(
        tmp_path / "unity",
        pid_path=pid_dir / "unity.pid",
        bind_delay_s=0.25,
    )

    started = time.monotonic()
    runtime = _runtime(tmp_path, unity=unity, bridge=bridge, timeout=2.0)
    with runtime:
        elapsed = time.monotonic() - started
        assert elapsed >= 0.20
        assert runtime.unity is not None
        assert runtime.unity.poll() is None
        assert runtime.bridge.poll() is None
        assert runtime.startup_state == [
            "ENDPOINTS_ALLOCATED",
            "ROSCORE_STARTED",
            "ROS_MASTER_READY",
            "BRIDGE_STARTED",
            "PYTHON_SOCKETS_CONNECTED",
            "BRIDGE_READY",
            "PYTHON_RESULT_READY_SENT",
            "UNITY_STARTED",
            "UNITY_FACING_SOCKETS_READY",
        ]
    assert not _pid_is_alive(pid_dir / "roscore.pid")
    assert not _pid_is_alive(pid_dir / "bridge.pid")
    assert not _pid_is_alive(pid_dir / "unity.pid")


def test_unity_startup_timeout_cleans_all_children(tmp_path, monkeypatch):
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    _fake_roscore(tmp_path / "roscore", pid_dir / "roscore.pid")
    bridge = _fake_bridge(tmp_path / "bridge", pid_path=pid_dir / "bridge.pid")
    unity = _write_executable(
        tmp_path / "unity",
        """#!/usr/bin/env python3
import os
import signal
import sys
import time
open("__PID__", "w").write(str(os.getpid()))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
""".replace("__PID__", str(pid_dir / "unity.pid")),
    )

    runtime = _runtime(tmp_path, unity=unity, bridge=bridge, timeout=0.25)
    with pytest.raises(RuntimeError):
        runtime.__enter__()
    assert not _pid_is_alive(pid_dir / "roscore.pid")
    assert not _pid_is_alive(pid_dir / "bridge.pid")
    assert not _pid_is_alive(pid_dir / "unity.pid")


def test_bridge_startup_failure_cleans_roscore(tmp_path, monkeypatch):
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    _fake_roscore(tmp_path / "roscore", pid_dir / "roscore.pid")
    bridge = _fake_bridge(
        tmp_path / "bridge", pid_path=pid_dir / "bridge.pid", fail=True
    )
    unity = _fake_unity(tmp_path / "unity", pid_path=pid_dir / "unity.pid")

    runtime = _runtime(tmp_path, unity=unity, bridge=bridge, timeout=0.25)
    with pytest.raises(RuntimeError):
        runtime.__enter__()
    assert not _pid_is_alive(pid_dir / "roscore.pid")
    assert not _pid_is_alive(pid_dir / "bridge.pid")
    assert not _pid_is_alive(pid_dir / "unity.pid")


def test_normal_dry_run_remains_available(tmp_path):
    from planning.diagnostics.reliable_single_worker import run_dry_run

    summary = run_dry_run(out_dir=tmp_path, primitive_count=1)

    assert summary["execution_total"] == 1
    assert summary["complete_total"] == 1
