"""Focused tests for the formal observation provenance contract owner."""

from __future__ import annotations

import pytest

from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
    observation_contract,
)


@pytest.mark.unit
def test_observation_contract_requires_an_explicit_supported_value():
    assert (
        observation_contract(
            {
                "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
                "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            }
        )
        == EXACT_ENDPOINT_OBSERVATION_CONTRACT
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata",
    (
        {},
        {"observation_contract": "future_observation_v9"},
        {
            "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "observation_source": LEGACY_ASYNC_OBSERVATION_CONTRACT,
        },
    ),
)
def test_observation_contract_rejects_missing_unknown_or_mismatched_source(metadata):
    with pytest.raises(ValueError, match="observation contract"):
        observation_contract(metadata)
