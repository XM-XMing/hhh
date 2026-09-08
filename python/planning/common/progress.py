"""One terminal progress contract for long-running pipeline commands."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple


def _finite_nonnegative(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def format_duration(seconds: float) -> str:
    """Format elapsed time without wrapping after 24 hours."""

    total = int(_finite_nonnegative(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)


class ProgressRateTracker:
    """Small rolling/EWMA rate tracker for runtime telemetry only.

    The tracker has no pipeline knowledge.  Callers provide one monotonically
    increasing completion counter; the returned stable rate is suitable for
    human ETA display while the caller remains responsible for its target.
    """

    def __init__(
        self,
        *,
        initial_completed: int = 0,
        window_s: float = 60.0,
        ewma_alpha: float = 0.30,
    ) -> None:
        if not math.isfinite(float(window_s)) or float(window_s) <= 0.0:
            raise ValueError("window_s must be positive")
        if not math.isfinite(float(ewma_alpha)) or not 0.0 < float(ewma_alpha) <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.initial_completed = max(0, int(initial_completed))
        self.window_s = float(window_s)
        self.ewma_alpha = float(ewma_alpha)
        self._samples: Deque[Tuple[float, int]] = deque()
        self._last: Optional[Tuple[float, int]] = None
        self._ewma_rate = 0.0

    def update(self, completed: int, *, now: Optional[float] = None) -> Dict[str, float]:
        timestamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(timestamp):
            timestamp = time.monotonic()
        value = max(self.initial_completed, int(completed))
        if self._last is not None:
            last_time, last_value = self._last
            elapsed = timestamp - last_time
            delta = value - last_value
            if elapsed > 0.0 and delta >= 0:
                instantaneous = delta / elapsed
                self._ewma_rate = (
                    instantaneous
                    if self._ewma_rate <= 0.0
                    else self.ewma_alpha * instantaneous
                    + (1.0 - self.ewma_alpha) * self._ewma_rate
                )
        self._last = (timestamp, value)
        self._samples.append((timestamp, value))
        cutoff = timestamp - self.window_s
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        oldest_time, oldest_value = self._samples[0]
        rolling_elapsed = timestamp - oldest_time
        rolling_delta = max(0, value - oldest_value)
        rolling_rate = (
            rolling_delta / rolling_elapsed
            if rolling_elapsed > 0.0 and rolling_delta > 0
            else 0.0
        )
        stable_rate = (
            0.5 * rolling_rate + 0.5 * self._ewma_rate
            if rolling_rate > 0.0 and self._ewma_rate > 0.0
            else max(rolling_rate, self._ewma_rate)
        )
        return {
            "rolling_rate_per_s": float(rolling_rate),
            "ewma_rate_per_s": float(self._ewma_rate),
            "stable_rate_per_s": float(stable_rate),
            "completed_this_run": float(max(0, value - self.initial_completed)),
            "sample_count": float(len(self._samples)),
        }

def progress_metrics(
    *,
    processed: int,
    passing: int,
    estimated_stop: int,
    elapsed_s: float,
    baseline_processed: int = 0,
    eta_processed: Optional[int] = None,
    eta_baseline_processed: Optional[int] = None,
    eta_target: Optional[int] = None,
    rolling_throughput_per_s: Optional[float] = None,
    ewma_throughput_per_s: Optional[float] = None,
    minimum_calibration_s: float = 0.0,
    minimum_calibration_items: int = 0,
) -> Dict[str, Any]:
    """Return stable pass-rate and ETA metrics.

    ``rate`` always means ``passing / processed``.  ETA uses only work
    completed during the current process, which keeps resumed runs honest.
    """

    processed_value = max(0, int(processed))
    passing_value = max(0, int(passing))
    stop_value = max(processed_value, int(estimated_stop))
    elapsed_value = _finite_nonnegative(elapsed_s)
    eta_processed_value = (
        processed_value
        if eta_processed is None
        else max(0, int(eta_processed))
    )
    eta_baseline_value = (
        int(baseline_processed)
        if eta_baseline_processed is None
        else int(eta_baseline_processed)
    )
    completed_this_run = max(0, eta_processed_value - eta_baseline_value)
    whole_run_throughput = (
        completed_this_run / elapsed_value
        if completed_this_run > 0 and elapsed_value > 0.0
        else 0.0
    )
    rolling_value = (
        _finite_nonnegative(rolling_throughput_per_s)
        if rolling_throughput_per_s is not None
        else whole_run_throughput
    )
    ewma_value = (
        _finite_nonnegative(ewma_throughput_per_s)
        if ewma_throughput_per_s is not None
        else whole_run_throughput
    )
    throughput = (
        0.5 * rolling_value + 0.5 * ewma_value
        if rolling_value > 0.0 and ewma_value > 0.0
        else max(rolling_value, ewma_value)
    )
    eta_stop_value = max(
        eta_processed_value,
        int(estimated_stop) if eta_target is None else int(eta_target),
    )
    remaining = max(0, eta_stop_value - eta_processed_value)
    calibrating = bool(
        remaining > 0
        and throughput > 0.0
        and (
            elapsed_value < max(0.0, float(minimum_calibration_s))
            or completed_this_run < max(0, int(minimum_calibration_items))
        )
    )
    eta_h = (
        0.0
        if remaining == 0
        else (
            remaining / throughput / 3600.0
            if throughput > 0.0 and not calibrating
            else None
        )
    )
    eta_status = (
        "READY"
        if remaining == 0
        else ("CALIBRATING" if calibrating else ("READY" if throughput > 0.0 else "UNKNOWN"))
    )
    return {
        "rate": passing_value / processed_value if processed_value else 0.0,
        "throughput_per_s": throughput,
        "eta_h": eta_h,
        "eta_status": eta_status,
        "rolling_throughput_per_s": rolling_value,
        "ewma_throughput_per_s": ewma_value,
        "eta_processed": float(eta_processed_value),
        "eta_baseline_processed": float(eta_baseline_value),
        "eta_target": float(eta_stop_value),
    }

def format_progress(
    *,
    component: str,
    processed: int,
    passing: int,
    estimated_stop: int,
    elapsed_s: float,
    baseline_processed: int = 0,
    eta_processed: Optional[int] = None,
    eta_baseline_processed: Optional[int] = None,
    eta_target: Optional[int] = None,
    rolling_throughput_per_s: Optional[float] = None,
    ewma_throughput_per_s: Optional[float] = None,
    minimum_calibration_s: float = 0.0,
    minimum_calibration_items: int = 0,
    **details: Any,
) -> str:
    """Build the canonical, grep-friendly terminal progress line."""

    metrics = progress_metrics(
        processed=processed,
        passing=passing,
        estimated_stop=estimated_stop,
        elapsed_s=elapsed_s,
        baseline_processed=baseline_processed,
        eta_processed=eta_processed,
        eta_baseline_processed=eta_baseline_processed,
        eta_target=eta_target,
        rolling_throughput_per_s=rolling_throughput_per_s,
        ewma_throughput_per_s=ewma_throughput_per_s,
        minimum_calibration_s=minimum_calibration_s,
        minimum_calibration_items=minimum_calibration_items,
    )
    stop_value = max(int(processed), int(estimated_stop))
    eta = metrics["eta_h"]
    if metrics["eta_status"] == "CALIBRATING":
        eta_text = "CALIBRATING"
    else:
        eta_text = (
            "{:.2f}".format(float(eta))
            if eta is not None and math.isfinite(float(eta))
            else "unknown"
        )
    fields = [
        "PROGRESS",
        "component={}".format(component),
        "processed={}".format(int(processed)),
        "passing={}".format(int(passing)),
        "rate={:.4f}".format(metrics["rate"]),
        "estimated_stop={}".format(stop_value),
        "eta_h={}".format(eta_text),
    ]
    fields.extend(
        "{}={}".format(key, str(value).replace(" ", "_"))
        for key, value in details.items()
        if value is not None
    )
    return " ".join(fields)

def print_progress(**kwargs: Any) -> str:
    """Print and return one canonical progress line."""

    line = format_progress(**kwargs)
    print(line, flush=True)
    return line
