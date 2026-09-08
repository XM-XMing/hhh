"""Independent fixed-value tests for the canonical reward arithmetic."""

from __future__ import annotations

import pytest

from planning.contracts.reward import compute_reward, reward_contract_sha256


def _base(**overrides):
    values = dict(
        progress=0.5,
        z_progress=-0.25,
        min_clearance=0.25,
        clearance_margin_m=0.5,
        action_changed=True,
        success=False,
        collided=False,
        altitude_violation=False,
        timeout=False,
        far=False,
        dead_end=False,
    )
    values.update(overrides)
    return values


@pytest.mark.unit
def test_reward_contract_matches_frozen_normal_transition_value():
    # 2*.5 + 1*(-.25) -.02 + (-1)*(.5-.25) -.02 = .46
    assert compute_reward(**_base()) == pytest.approx(0.46, abs=1.0e-12)


@pytest.mark.unit
def test_reward_contract_preserves_terminal_penalty_and_success_precedence():
    collision = compute_reward(**_base(collided=True))
    assert collision == pytest.approx(-99.54, abs=1.0e-12)
    success = compute_reward(**_base(success=True))
    assert success == pytest.approx(30.46, abs=1.0e-12)
    unsafe_goal = compute_reward(**_base(success=True, collided=True))
    assert unsafe_goal == pytest.approx(-69.54, abs=1.0e-12)


@pytest.mark.unit
def test_reward_contract_has_stable_canonical_identity():
    value = reward_contract_sha256()
    assert len(value) == 64
    assert value == reward_contract_sha256()

