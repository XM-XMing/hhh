"""Owner/interface tests for deferred AWAC capabilities."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_confidence_owner_is_typed_but_has_no_algorithm():
    from planning.awac.confidence import unavailable_confidence

    signal = unavailable_confidence()
    assert signal.available is False
    assert signal.value is None


@pytest.mark.unit
def test_primitive_neighbor_owner_is_bound_to_mpl_without_algorithm():
    from planning.primitives.neighbors import empty_neighbor_index

    index = empty_neighbor_index(mpl_contract_sha256="m" * 64)
    assert index.action_count == 105
    assert all(value == tuple() for value in index.neighbors.values())

