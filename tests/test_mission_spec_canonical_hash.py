"""Regression coverage for the mission-id canonical hash seam."""

from planning.mission.spec import mission_id_from_values


def test_mission_id_uses_the_common_canonical_json_owner():
    assert mission_id_from_values(
        0.0,
        0.0,
        2.0,
        0.0,
        40.0,
        0.0,
        2.0,
    ) == "d2fa94c04fc99b2e6383"
