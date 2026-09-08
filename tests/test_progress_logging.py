import math

from planning.common.progress import (
    ProgressRateTracker,
    format_duration,
    format_progress,
    progress_metrics,
)


def test_progress_contract_has_fixed_fields_and_semantics():
    line = format_progress(
        component="teacher_audit",
        processed=50,
        passing=20,
        estimated_stop=100,
        elapsed_s=10.0,
        worker=2,
    )
    assert line == (
        "PROGRESS component=teacher_audit processed=50 passing=20 "
        "rate=0.4000 estimated_stop=100 eta_h=0.00 worker=2"
    )


def test_progress_eta_uses_only_current_run_work():
    metrics = progress_metrics(
        processed=150,
        passing=75,
        estimated_stop=250,
        elapsed_s=100.0,
        baseline_processed=100,
    )
    assert metrics["rate"] == 0.5
    assert metrics["throughput_per_s"] == 0.5
    assert math.isclose(metrics["eta_h"], 200.0 / 3600.0)


def test_progress_eta_is_unknown_before_first_completed_item():
    line = format_progress(
        component="labeling",
        processed=0,
        passing=0,
        estimated_stop=10,
        elapsed_s=1.0,
    )
    assert "eta_h=unknown" in line


def test_progress_never_reports_stop_before_processed():
    line = format_progress(
        component="collection",
        processed=31,
        passing=31,
        estimated_stop=20,
        elapsed_s=10.0,
    )
    assert "processed=31" in line
    assert "estimated_stop=31" in line


def test_progress_eta_can_use_a_different_completion_counter():
    metrics = progress_metrics(
        processed=100,
        passing=20,
        estimated_stop=60,
        elapsed_s=100.0,
        eta_processed=20,
        eta_baseline_processed=0,
    )
    assert metrics["throughput_per_s"] == 0.2
    assert math.isclose(metrics["eta_h"], 40.0 / 0.2 / 3600.0)

    line = format_progress(
        component="collection",
        processed=100,
        passing=20,
        estimated_stop=60,
        elapsed_s=100.0,
        eta_processed=20,
        eta_baseline_processed=0,
    )
    assert "eta_h={:.2f}".format(40.0 / 0.2 / 3600.0) in line


def test_progress_eta_calibration_and_zero_rate_are_bounded():
    tracker = ProgressRateTracker()
    tracker.update(0, now=0.0)
    snapshot = tracker.update(1, now=1.0)
    metrics = progress_metrics(
        processed=10,
        passing=1,
        estimated_stop=10,
        elapsed_s=1.0,
        eta_processed=1,
        eta_target=10,
        rolling_throughput_per_s=snapshot["rolling_rate_per_s"],
        ewma_throughput_per_s=snapshot["ewma_rate_per_s"],
        minimum_calibration_s=10.0,
        minimum_calibration_items=5,
    )
    assert metrics["eta_status"] == "CALIBRATING"
    assert metrics["eta_h"] is None
    assert "eta_h=CALIBRATING" in format_progress(
        component="prepare",
        processed=10,
        passing=1,
        estimated_stop=10,
        elapsed_s=1.0,
        eta_processed=1,
        eta_target=10,
        rolling_throughput_per_s=snapshot["rolling_rate_per_s"],
        ewma_throughput_per_s=snapshot["ewma_rate_per_s"],
        minimum_calibration_s=10.0,
        minimum_calibration_items=5,
    )
    zero = progress_metrics(
        processed=10,
        passing=0,
        estimated_stop=10,
        elapsed_s=100.0,
        eta_processed=0,
        eta_target=10,
    )
    assert zero["eta_status"] == "UNKNOWN"
    assert zero["eta_h"] is None
    completed = progress_metrics(
        processed=10,
        passing=10,
        estimated_stop=10,
        elapsed_s=0.0,
        eta_processed=10,
        eta_target=10,
    )
    assert completed["eta_status"] == "READY"
    assert completed["eta_h"] == 0.0


def test_progress_rate_tracker_reflects_rate_drop_and_duration_over_24h():
    tracker = ProgressRateTracker(window_s=10.0, ewma_alpha=1.0)
    tracker.update(0, now=0.0)
    fast = tracker.update(10, now=1.0)
    slow = tracker.update(11, now=6.0)
    assert fast["stable_rate_per_s"] > slow["stable_rate_per_s"]
    assert format_duration(25 * 3600 + 61) == "25:01:01"


def test_progress_helper_never_emits_nonfinite_runtime_values():
    metrics = progress_metrics(
        processed=10,
        passing=2,
        estimated_stop=20,
        elapsed_s=float("nan"),
        rolling_throughput_per_s=float("inf"),
        ewma_throughput_per_s=float("nan"),
    )
    assert all(
        value is None or math.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, (int, float))
    )
    assert format_duration(float("nan")) == "00:00:00"
