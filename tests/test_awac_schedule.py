"""Public scheduling contract for AWAC learner orchestration."""

import pytest

from planning.awac.schedule import build_awac_schedule


@pytest.mark.unit
def test_training_segments_and_full_evaluations_are_independent():
    schedule = build_awac_schedule(
        total_steps=15000,
        train_segment_steps=1000,
        full_eval_interval_steps=5000,
    )

    assert schedule.training_segment_ends == tuple(range(1000, 15001, 1000))
    assert schedule.collection_refresh_ends == schedule.training_segment_ends
    assert schedule.full_eval_ends == (5000, 10000, 15000)
    assert schedule.full_eval_ends != schedule.training_segment_ends


@pytest.mark.unit
def test_full_evaluation_is_due_at_total_step_even_when_not_an_interval():
    schedule = build_awac_schedule(
        total_steps=6200,
        train_segment_steps=1000,
        full_eval_interval_steps=5000,
    )

    assert schedule.training_segment_ends[-1] == 6200
    assert schedule.full_eval_ends == (5000, 6200)


@pytest.mark.unit
def test_invalid_schedule_rejects_non_positive_values():
    with pytest.raises(ValueError, match="train_segment_steps"):
        build_awac_schedule(
            total_steps=15000,
            train_segment_steps=0,
            full_eval_interval_steps=5000,
        )
    with pytest.raises(ValueError, match="full_eval_interval_steps"):
        build_awac_schedule(
            total_steps=15000,
            train_segment_steps=1000,
            full_eval_interval_steps=0,
        )
