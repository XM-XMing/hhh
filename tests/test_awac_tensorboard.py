"""Focused tests for the single Standard AWAC TensorBoard owner."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _snapshot(step: int):
    return {
        "progress": {
            "online_env_steps": step,
            "online_transitions_committed": step - 1,
            "replay_size": 5142 + step - 1,
            "completed_episodes": max(0, step // 10),
            "actor_update_count": max(0, step // 100),
            "critic_update_count": max(0, step // 5),
            "target_update_count": max(0, step // 5),
        },
        "actor": {
            "loss": 1.0,
            "entropy": 1.5,
            "bc_kl_mean": 0.01,
            "bc_top1_disagreement_rate": 0.02,
            "parameter_delta_norm": 0.03,
            "grad_norm": 0.04,
            "depth_grad_norm": 0.05,
            "vector_grad_norm": 0.06,
            "head_grad_norm": 0.07,
        },
        "awac": {
            "advantage_mean": 0.1,
            "advantage_std": 0.2,
            "weight_mean": 1.0,
            "weight_p50": 0.9,
        "weight_p95": 1.2,
        "weight_max": 2.0,
        "weight_clip_fraction": 0.0,
        "raw_weight_ess_fraction": 0.95,
        "normalized_clipped_weight_ess_fraction": 0.80,
        "ess_batch_size": 128.0,
        "ess_aggregation_window_updates": 4.0,
        },
        "critic": {
            "td_loss_mean": 1.0,
            "td_loss_p95": 2.0,
            "q_mean": 0.5,
            "q_std": 0.3,
            "q_p01": -1.0,
            "q_p99": 2.0,
            "twin_q_disagreement_mean": 0.1,
            "twin_q_disagreement_p95": 0.2,
            "grad_norm": 0.4,
        },
        "episode": {
            "return_mean": 3.0,
            "length_mean": 12.0,
            "success_rate_window": 0.8,
            "collision_rate_window": 0.1,
            "dead_end_rate_window": 0.1,
            "timeout_rate_window": 0.0,
        },
        "terminal": {
            "success_count": 8,
            "collision_count": 1,
            "dead_end_count": 1,
            "timeout_count": 0,
            "hard_altitude_count": 0,
            "runtime_drop_count": 0,
        },
        "replay": {
            "size": 5142 + step - 1,
            "bc_calibration_rows": 5142,
            "awac_online_rows": step - 1,
            "bc_calibration_fraction": 5142.0 / (5142.0 + max(0, step - 1)),
            "awac_online_fraction": max(0, step - 1) / (5142.0 + max(0, step - 1)),
        },
        "runtime": {
            "env_steps_per_sec": 10.0,
            "committed_transitions_per_sec": 9.0,
            "episodes_per_min": 20.0,
            "actor_updates_per_sec": 1.0,
            "critic_updates_per_sec": 2.0,
            "transition_drop_count": 0,
            "runtime_failure_count": 0,
        },
        "lr": {
            "actor_depth": 1.0e-6,
            "actor_vector": 3.0e-6,
            "actor_head": 1.0e-5,
            "critic_depth": 1.0e-5,
            "critic_vector": 1.0e-5,
            "critic_head": 1.0e-4,
        },
        "health": {
            "nan_count": 0,
            "inf_count": 0,
            "nonfinite_metric_count": 0,
            "q_divergence": 0,
            "actor_divergence": 0,
        },
        "confidence": {
            "mean": 0.48,
            "p05": 0.31,
            "p50": 0.49,
            "p95": 0.66,
            "q_margin_norm_mean": 0.10,
            "twin_disagreement_norm_mean": 0.25,
        },
        "adaptive_bc_kl": {
            "beta_mean": 0.051,
            "beta_p05": 0.021,
            "beta_p50": 0.050,
            "beta_p95": 0.079,
            "low_confidence_beta_mean": 0.070,
            "high_confidence_beta_mean": 0.031,
        },
    }


def test_tensorboard_writes_required_scalars_and_no_images(tmp_path: Path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    from planning.awac.tensorboard import AWACTensorBoardLogger

    logger = AWACTensorBoardLogger(tmp_path / "tensorboard")
    logger.log_snapshot(_snapshot(0), global_step=0)
    logger.log_snapshot(_snapshot(100), global_step=100)
    logger.close()

    events = list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    assert events
    accumulator = event_accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    required = AWACTensorBoardLogger.required_scalar_tags()
    assert required.issubset(set(accumulator.Tags()["scalars"]))
    assert {
        "confidence/mean",
        "confidence/p05",
        "confidence/p50",
        "confidence/p95",
        "confidence/q_margin_norm_mean",
        "confidence/twin_disagreement_norm_mean",
        "adaptive_bc_kl/beta_mean",
        "adaptive_bc_kl/beta_p05",
        "adaptive_bc_kl/beta_p50",
        "adaptive_bc_kl/beta_p95",
        "adaptive_bc_kl/low_confidence_beta_mean",
        "adaptive_bc_kl/high_confidence_beta_mean",
        "awac/raw_weight_ess_fraction",
        "awac/normalized_clipped_weight_ess_fraction",
        "awac/ess_batch_size",
        "awac/ess_aggregation_window_updates",
    }.issubset(set(accumulator.Tags()["scalars"]))
    assert not accumulator.Tags().get("images")
    for tag in required:
        assert [item.step for item in accumulator.Scalars(tag)] == [0, 100]


def test_tensorboard_resume_appends_without_resetting_steps(tmp_path: Path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    from planning.awac.tensorboard import AWACTensorBoardLogger

    first = AWACTensorBoardLogger(tmp_path / "tensorboard")
    first.log_snapshot(_snapshot(0), global_step=0)
    first.log_snapshot(_snapshot(100), global_step=100)
    first.close()

    resumed = AWACTensorBoardLogger(tmp_path / "tensorboard")
    resumed.log_snapshot(_snapshot(200), global_step=200)
    resumed.close()

    accumulator = event_accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    assert [item.step for item in accumulator.Scalars("progress/online_env_steps")] == [
        0,
        100,
        200,
    ]


def test_tensorboard_logs_dev100_at_requested_step(tmp_path: Path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    from planning.awac.tensorboard import AWACTensorBoardLogger

    logger = AWACTensorBoardLogger(tmp_path / "tensorboard")
    logger.log_dev100(
        {
            "success_rate": 0.70,
            "collision_rate": 0.08,
            "dead_end_rate": 0.20,
            "timeout_rate": 0.02,
            "episode_return_mean": 12.0,
            "episode_steps_mean": 18.0,
        },
        global_step=10000,
        bc_baseline={
            "success_rate": 0.69,
            "collision_rate": 0.09,
            "dead_end_rate": 0.22,
        },
    )
    logger.close()

    accumulator = event_accumulator.EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    assert accumulator.Scalars("dev100/success_rate")[0].step == 10000
    assert accumulator.Scalars("dev100/mean_steps")[0].value == 18.0
    assert accumulator.Scalars("dev100/delta_success_vs_bc")[0].value == pytest.approx(
        0.01
    )


def test_trainer_tensorboard_lifecycle_has_one_owner_and_safe_cleanup(
    tmp_path: Path, monkeypatch
):
    from planning.awac import tensorboard as tensorboard_module
    from planning.awac import trainer

    class FakeLogger:
        def __init__(self, log_dir, *, log_interval_steps):
            self.log_dir = Path(log_dir)
            self.log_interval_steps = int(log_interval_steps)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

        def manifest(self):
            return {"owner": "test"}

        def physical_size_mb(self):
            return 0.0

    created = []

    def build_logger(*args, **kwargs):
        logger = FakeLogger(*args, **kwargs)
        created.append(logger)
        return logger

    monkeypatch.setattr(tensorboard_module, "AWACTensorBoardLogger", build_logger)
    logger = trainer._create_tensorboard_logger(
        SimpleNamespace(
            tensorboard_log_dir=str(tmp_path / "tensorboard"),
            log_interval_steps=100,
        )
    )
    assert logger is created[0]
    assert logger.log_interval_steps == 100

    source = inspect.getsource(trainer._run_standard_awac_online)
    assert source.index("tensorboard_logger = None") < source.index("\n    try:")
    assert source.index("tensorboard_logger = _create_tensorboard_logger(") < source.index(
        "tensorboard_logger=tensorboard_logger"
    )

    trainer._close_tensorboard_logger(logger, output_dir=tmp_path / "out")
    assert logger.close_calls == 1


def test_trainer_disabled_tensorboard_path_returns_none():
    from planning.awac import trainer

    assert trainer._create_tensorboard_logger(
        SimpleNamespace(tensorboard_log_dir="", log_interval_steps=100)
    ) is None


def test_train_awac_standard_production_default_creates_event_file(tmp_path: Path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    from planning.awac import trainer

    out_dir = tmp_path / "standard"
    args = trainer.build_parser().parse_args(
        [
            "--bc-checkpoint", "bc.pt",
            "--phase", "awac_training",
            "--out-dir", str(out_dir),
        ]
    )
    logger = trainer._create_tensorboard_logger(
        args, default_to_output_dir=True
    )
    assert logger is not None
    assert logger.log_dir == (out_dir / "tensorboard").resolve()
    logger.log_snapshot(_snapshot(10000), global_step=10000)
    trainer._close_tensorboard_logger(logger, output_dir=out_dir)

    events = list((out_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert events
    accumulator = event_accumulator.EventAccumulator(str(out_dir / "tensorboard"))
    accumulator.Reload()
    assert accumulator.Scalars("progress/online_env_steps")[0].step == 10000
    manifest = (out_dir / "tensorboard_manifest.json").read_text(encoding="utf-8")
    assert '"enabled": true' in manifest


def test_no_tensorboard_disables_standard_production_default(tmp_path: Path):
    from planning.awac import trainer

    args = trainer.build_parser().parse_args(
        [
            "--bc-checkpoint", "bc.pt",
            "--phase", "awac_training",
            "--out-dir", str(tmp_path / "standard"),
            "--no-tensorboard",
        ]
    )
    assert trainer._create_tensorboard_logger(
        args, default_to_output_dir=True
    ) is None


def test_standard_resume_tensorboard_starts_at_restored_step(tmp_path: Path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    from planning.awac import trainer

    out_dir = tmp_path / "standard-25k"
    args = trainer.build_parser().parse_args(
        [
            "--bc-checkpoint", "bc.pt",
            "--phase", "awac_training",
            "--out-dir", str(out_dir),
        ]
    )
    logger = trainer._create_tensorboard_logger(
        args, default_to_output_dir=True
    )

    class RestoredRunner:
        online_env_steps = 10000
        next_tensorboard_step = 10100

        @staticmethod
        def tensorboard_snapshot():
            return _snapshot(10000)

    logger.log_runner(RestoredRunner(), force=True)
    trainer._close_tensorboard_logger(logger, output_dir=out_dir)
    accumulator = event_accumulator.EventAccumulator(str(out_dir / "tensorboard"))
    accumulator.Reload()
    steps = [
        item.step for item in accumulator.Scalars("progress/online_env_steps")
    ]
    assert steps == [10000]


def test_parser_exposes_explicit_tensorboard_disable_flag():
    from planning.awac import trainer

    args = trainer.build_parser().parse_args(
        ["--bc-checkpoint", "bc.pt", "--out-dir", "out", "--no-tensorboard"]
    )
    assert args.no_tensorboard is True


def test_cleanup_before_logger_creation_preserves_original_exception(tmp_path: Path):
    from planning.awac import trainer

    original = RuntimeError("runtime failed before TensorBoard creation")
    with pytest.raises(RuntimeError) as raised:
        try:
            raise original
        finally:
            trainer._close_tensorboard_logger(None, output_dir=tmp_path / "out")
    assert raised.value is original


def test_cleanup_after_logger_creation_preserves_original_exception(tmp_path: Path):
    from planning.awac import trainer

    class FakeLogger:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

        def manifest(self):
            return {"owner": "test"}

        def physical_size_mb(self):
            return 0.0

    logger = FakeLogger()
    original = RuntimeError("runtime failed after TensorBoard creation")
    with pytest.raises(RuntimeError) as raised:
        try:
            raise original
        finally:
            trainer._close_tensorboard_logger(logger, output_dir=tmp_path / "out")
    assert raised.value is original
    assert logger.close_calls == 1


def test_tensorboard_close_is_idempotent_after_backend_close(tmp_path: Path):
    pytest.importorskip("torch.utils.tensorboard")
    from planning.awac.tensorboard import AWACTensorBoardLogger

    logger = AWACTensorBoardLogger(tmp_path / "tensorboard")
    logger.close()
    logger.close()
    assert logger.closed is True
