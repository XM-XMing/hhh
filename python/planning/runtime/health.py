"""Runtime health checks shared by collection workers."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Mapping, Sequence


def _run(argv: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def wait_for_ros_master(*, env: Mapping[str, str], timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        result = _run(("rosparam", "list"), env)
        if result.returncode == 0:
            return
        time.sleep(0.25)
    raise TimeoutError(f"ROS master did not become ready: {env.get('ROS_MASTER_URI', '')}")


def wait_for_ros_topics(
    *,
    env: Mapping[str, str],
    required_topics: Sequence[str],
    timeout_s: float,
) -> None:
    required = set(required_topics)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        result = _run(("rostopic", "list"), env)
        if result.returncode == 0:
            topics = {line.strip() for line in result.stdout.splitlines() if line.strip()}
            if required.issubset(topics):
                return
        time.sleep(0.5)
    missing = ", ".join(sorted(required))
    raise TimeoutError(
        f"ROS topics did not become ready ({missing}): {env.get('ROS_MASTER_URI', '')}"
    )
