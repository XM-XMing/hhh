#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XMflight ROS base system acceptance test.

The check supports --skip-map so normal runtime checks do not subscribe to the
full-resolution latched PointCloud2. It also separates functional checks from
heavy RViz/full-map checks.

Recommended:
  # Runtime/perception check, no full map subscription:
  rosrun planning check_base_system.py --no-motion --skip-map

  # Full functional check, still no full map subscription:
  rosrun planning check_base_system.py --skip-map

  # Full map check only:
  rosrun planning check_base_system.py --no-motion --map-only
"""


import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool

try:
    from planning.msg import XMState
except Exception as exc:
    XMState = None
    _XMSTATE_IMPORT_ERROR = exc
else:
    _XMSTATE_IMPORT_ERROR = None


@dataclass
class CheckResult:
    name: str
    ok: bool
    value: str
    target: str
    note: str = ""


class TopicMonitor:
    def __init__(self, topic: str, msg_type: Any, queue_size: int = 10):
        self.topic = topic
        self.msg_type = msg_type
        self.count = 0
        self.last_msg: Optional[Any] = None
        self.sub = rospy.Subscriber(topic, msg_type, self._cb, queue_size=queue_size)

    def _cb(self, msg: Any) -> None:
        self.count += 1
        self.last_msg = msg

    def wait_first(self, timeout: float) -> Optional[Any]:
        deadline = time.time() + timeout
        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.last_msg is not None:
                return self.last_msg
            rate.sleep()
        return None

    def measure_hz(self, duration: float, warmup: float = 0.2) -> float:
        self.count = 0
        time.sleep(warmup)
        self.count = 0
        t0 = time.time()
        time.sleep(duration)
        elapsed = max(1e-6, time.time() - t0)
        return float(self.count) / elapsed


def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def pos_tuple(state: Any) -> Tuple[float, float, float]:
    return (float(state.position.x), float(state.position.y), float(state.position.z))


def vel_norm(state: Any) -> float:
    return norm3(float(state.velocity.x), float(state.velocity.y), float(state.velocity.z))


def wait_state_near(
    monitor: TopicMonitor,
    target_xyz: Tuple[float, float, float],
    threshold: float,
    timeout: float,
) -> Tuple[bool, Optional[Any], float]:
    deadline = time.time() + timeout
    best_err = float("inf")
    best_state = None
    rate = rospy.Rate(50)
    while not rospy.is_shutdown() and time.time() < deadline:
        state = monitor.last_msg
        if state is not None:
            p = pos_tuple(state)
            err = norm3(p[0] - target_xyz[0], p[1] - target_xyz[1], p[2] - target_xyz[2])
            if err < best_err:
                best_err = err
                best_state = state
            if err <= threshold:
                return True, state, err
        rate.sleep()
    return False, best_state, best_err


def wait_speed_below(monitor: TopicMonitor, threshold: float, timeout: float) -> Tuple[bool, Optional[Any], float]:
    deadline = time.time() + timeout
    best_speed = float("inf")
    best_state = None
    rate = rospy.Rate(50)
    while not rospy.is_shutdown() and time.time() < deadline:
        state = monitor.last_msg
        if state is not None:
            speed = vel_norm(state)
            if speed < best_speed:
                best_speed = speed
                best_state = state
            if speed <= threshold:
                return True, state, speed
        rate.sleep()
    return False, best_state, best_speed


def make_pose(x: float, y: float, z: float, yaw: float = 0.0) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    half = yaw * 0.5
    msg.pose.orientation.z = math.sin(half)
    msg.pose.orientation.w = math.cos(half)
    return msg


def make_twist(vx: float, vy: float, vz: float, yaw_rate: float) -> Twist:
    msg = Twist()
    msg.linear.x = vx
    msg.linear.y = vy
    msg.linear.z = vz
    msg.angular.z = yaw_rate
    return msg


def pointcloud_fields_string(msg: PointCloud2) -> str:
    return ",".join([f.name for f in msg.fields])


def print_results(results):
    name_w = max(20, max(len(r.name) for r in results))
    print("")
    print("XMflight base system check")
    print("=" * 100)
    print(f"{'CHECK'.ljust(name_w)}  {'PASS'.ljust(6)}  {'VALUE'.ljust(28)}  {'TARGET'.ljust(24)}  NOTE")
    print("-" * 100)
    for r in results:
        ok = "PASS" if r.ok else "FAIL"
        print(f"{r.name.ljust(name_w)}  {ok.ljust(6)}  {r.value.ljust(28)}  {r.target.ljust(24)}  {r.note}")
    print("=" * 100)
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"summary: {passed}/{total} passed")
    print("")


def add_map_checks(results, args):
    map_mon = TopicMonitor("/xm/map_points", PointCloud2, queue_size=1)
    map_msg = map_mon.wait_first(args.wait_timeout)
    if map_msg is None:
        results.append(CheckResult("/xm/map_points", False, "timeout", f"<= {args.wait_timeout:.1f}s"))
        results.append(CheckResult("map fields", False, "no msg", "x,y,z required"))
        results.append(CheckResult("map height color", False, "no msg", "intensity or rgb"))
        results.append(CheckResult("map frame", False, "no msg", args.map_frame))
        return

    width = int(map_msg.width) * max(1, int(map_msg.height))
    fields = pointcloud_fields_string(map_msg)
    point_step = int(map_msg.point_step)
    raw_mib = width * point_step / (1024.0 * 1024.0)

    results.append(CheckResult("/xm/map_points", width > 0, f"{width} points", "> 0 points", fields))
    results.append(CheckResult("map raw size", raw_mib > 0, f"{raw_mib:.1f} MiB", "info"))
    results.append(CheckResult("map fields", all(k in fields.split(",") for k in ["x", "y", "z"]), fields, "x,y,z required"))
    results.append(CheckResult("map height color", ("intensity" in fields.split(",")) or ("rgb" in fields.split(",")), fields, "intensity or rgb"))
    results.append(CheckResult("map frame", map_msg.header.frame_id == args.map_frame, map_msg.header.frame_id, args.map_frame))


def check_tf(results, args, target: str, source: str, tf_buffer) -> None:
    try:
        ok = tf_buffer.can_transform(target, source, rospy.Time(0), rospy.Duration(args.wait_timeout))
        if ok:
            tr = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.2))
            t = tr.transform.translation
            value = f"({t.x:.3f},{t.y:.3f},{t.z:.3f})"
        else:
            value = "missing"
    except Exception as exc:
        ok = False
        value = f"error: {exc}"
    results.append(CheckResult(f"TF {target}->{source}", ok, value, "available"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-hz-min", type=float, default=45.0)
    parser.add_argument("--depth-hz-min", type=float, default=45.0)
    parser.add_argument("--marker-hz-min", type=float, default=25.0)
    parser.add_argument("--hz-duration", type=float, default=2.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--no-motion", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--skip-markers", action="store_true")
    parser.add_argument("--reset-x", type=float, default=0.0)
    parser.add_argument("--reset-y", type=float, default=0.0)
    parser.add_argument("--reset-z", type=float, default=2.0)
    parser.add_argument("--reset-threshold", type=float, default=0.10)
    parser.add_argument("--reset-timeout", type=float, default=3.0)
    parser.add_argument("--cmd-vx", type=float, default=1.0)
    parser.add_argument("--cmd-duration", type=float, default=3.0)
    parser.add_argument("--cmd-error-threshold", type=float, default=0.20)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.05)
    parser.add_argument("--stop-timeout", type=float, default=1.0)
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-frame", type=str, default="base_link")
    parser.add_argument("--camera-frame", type=str, default="d435i_depth_optical_frame")
    args = parser.parse_args()

    if XMState is None:
        print("FAIL: cannot import planning.msg.XMState")
        print(f"error: {_XMSTATE_IMPORT_ERROR}")
        print("Run from the catkin workspace: catkin_make && source devel/setup.bash")
        return 2

    rospy.init_node("check_base_system", anonymous=True)
    results = []

    if args.map_only:
        add_map_checks(results, args)
        print_results(results)
        return 0 if all(r.ok for r in results) else 1

    state_mon = TopicMonitor("/xm/state", XMState)
    depth_mon = TopicMonitor("/xm/depth/image_raw", Image)
    reset_pub = rospy.Publisher("/xm/reset_pose", PoseStamped, queue_size=1)
    cmd_pub = rospy.Publisher("/xm/cmd_vel", Twist, queue_size=10)
    stop_pub = rospy.Publisher("/xm/stop", Bool, queue_size=1)

    marker_mon = None
    if not args.skip_markers:
        try:
            from visualization_msgs.msg import MarkerArray
            marker_mon = TopicMonitor("/xm/markers", MarkerArray)
        except Exception:
            marker_mon = None

    state_msg = state_mon.wait_first(args.wait_timeout)
    results.append(CheckResult("/xm/state first msg", state_msg is not None, "received" if state_msg is not None else "timeout", f"<= {args.wait_timeout:.1f}s"))

    depth_msg = depth_mon.wait_first(args.wait_timeout)
    results.append(CheckResult("/xm/depth first msg", depth_msg is not None, "received" if depth_msg is not None else "timeout", f"<= {args.wait_timeout:.1f}s"))

    if not args.skip_map:
        add_map_checks(results, args)
    else:
        results.append(CheckResult("/xm/map_points", True, "skipped", "skipped"))

    if state_msg is not None:
        results.append(CheckResult("state collision", not bool(state_msg.collided), str(bool(state_msg.collided)), "false"))
        results.append(CheckResult("state altitude", not bool(state_msg.altitude_violation), str(bool(state_msg.altitude_violation)), "false"))

    if depth_msg is not None:
        depth_ok = depth_msg.width > 0 and depth_msg.height > 0 and depth_msg.encoding in ("16UC1", "mono16")
        results.append(CheckResult("depth metadata", depth_ok, f"{depth_msg.width}x{depth_msg.height} {depth_msg.encoding}", "width>0 height>0 16UC1"))
        expected_bytes = int(depth_msg.width) * int(depth_msg.height) * 2
        results.append(CheckResult("depth payload", len(depth_msg.data) == expected_bytes, f"{len(depth_msg.data)} bytes", f"{expected_bytes} bytes"))

    if state_msg is not None:
        hz = state_mon.measure_hz(args.hz_duration)
        results.append(CheckResult("/xm/state hz", hz >= args.state_hz_min, f"{hz:.2f} Hz", f">= {args.state_hz_min:.1f} Hz"))

    if depth_msg is not None:
        hz = depth_mon.measure_hz(args.hz_duration)
        results.append(CheckResult("/xm/depth hz", hz >= args.depth_hz_min, f"{hz:.2f} Hz", f">= {args.depth_hz_min:.1f} Hz"))

    if args.skip_markers:
        results.append(CheckResult("/xm/markers hz", True, "skipped", "skipped"))
    elif marker_mon is not None:
        marker_msg = marker_mon.wait_first(args.wait_timeout)
        if marker_msg is not None:
            hz = marker_mon.measure_hz(args.hz_duration)
            results.append(CheckResult("/xm/markers hz", hz >= args.marker_hz_min, f"{hz:.2f} Hz", f">= {args.marker_hz_min:.1f} Hz"))
        else:
            results.append(CheckResult("/xm/markers hz", False, "timeout", f">= {args.marker_hz_min:.1f} Hz"))

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    time.sleep(0.5)
    # Keep the listener alive while both transforms are checked.
    _ = tf_listener
    check_tf(results, args, args.map_frame, args.base_frame, tf_buffer)
    check_tf(results, args, args.base_frame, args.camera_frame, tf_buffer)

    if args.no_motion:
        results.append(CheckResult("reset_pose", True, "skipped", "skipped"))
        results.append(CheckResult("cmd_vel 3s", True, "skipped", "skipped"))
        results.append(CheckResult("stop", True, "skipped", "skipped"))
    else:
        target = (args.reset_x, args.reset_y, args.reset_z)
        time.sleep(0.2)
        reset_pub.publish(make_pose(*target))
        reset_ok, _, reset_err = wait_state_near(state_mon, target, args.reset_threshold, args.reset_timeout)
        results.append(CheckResult("reset_pose", reset_ok, f"err={reset_err:.3f} m", f"<= {args.reset_threshold:.2f} m"))

        p0 = pos_tuple(state_mon.last_msg) if state_mon.last_msg is not None else target
        cmd = make_twist(args.cmd_vx, 0.0, 0.0, 0.0)
        rate = rospy.Rate(50)
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < args.cmd_duration:
            cmd_pub.publish(cmd)
            rate.sleep()

        for _ in range(5):
            cmd_pub.publish(make_twist(0.0, 0.0, 0.0, 0.0))
            rate.sleep()

        time.sleep(0.1)
        p1 = pos_tuple(state_mon.last_msg) if state_mon.last_msg is not None else p0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        expected = abs(args.cmd_vx * args.cmd_duration)
        forward_dist = math.sqrt(dx * dx + dy * dy)
        err = abs(forward_dist - expected)
        results.append(CheckResult("cmd_vel 3s", err <= args.cmd_error_threshold, f"dist={forward_dist:.3f}m err={err:.3f}m", f"{expected:.2f}±{args.cmd_error_threshold:.2f}m", f"delta=({dx:.3f},{dy:.3f},{dz:.3f})"))

        stop_pub.publish(Bool(data=True))
        for _ in range(5):
            cmd_pub.publish(make_twist(0.0, 0.0, 0.0, 0.0))
            rate.sleep()
        stop_ok, _, speed = wait_speed_below(state_mon, args.stop_speed_threshold, args.stop_timeout)
        results.append(CheckResult("stop", stop_ok, f"speed={speed:.3f} m/s", f"<= {args.stop_speed_threshold:.2f} m/s"))

    print_results(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
