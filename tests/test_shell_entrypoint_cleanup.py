"""Current shell surface and test-integrity report contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "shell_entrypoint_cleanup_and_test_suite_integrity_v1.json"

EXPECTED_RETAINED_SHELLS = {
    "evaluate_policy_unity_managed.sh",
    "run_admissible_pairing_audit.sh",
    "run_command_tick_audit.sh",
    "run_cross_mission_t2_baseline_audit.sh",
    "run_cross_mission_t2_risk_screen.sh",
    "run_live_pairing_reachability.sh",
    "run_policy_eval_repro.sh",
}
OBSOLETE_SHELLS = {
    "collect_rollouts_parallel.sh",
    "run_awac_guarded.sh",
    "run_awac_reliable_v4.sh",
}


def test_current_shell_surface_is_explicitly_closed():
    current = {path.name for path in (ROOT / "scripts").glob("*.sh")}
    assert current == EXPECTED_RETAINED_SHELLS
    assert not current.intersection(OBSOLETE_SHELLS)


def test_cleanup_report_closes_inventory_and_test_delta():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source_shell_count_before"] == 10
    assert payload["source_shell_count_after"] == 7
    assert payload["rl_awac_sac_production_shell_count"] == 0
    assert payload["generic_evaluation_shell_required"] is True
    assert {Path(item).name for item in payload["deleted_shells"]} == OBSOLETE_SHELLS

    test_audit = payload["test_suite_integrity"]
    assert test_audit["old_full_test_pass_count"] == 891
    assert test_audit["old_full_test_skip_count"] == 42
    assert test_audit["current_pre_audit_pass_count"] == 680
    assert test_audit["current_pre_audit_skip_count"] == 42
    assert test_audit["accidentally_missing_test_cases"] == 0
    assert test_audit["unexplained_test_count"] == 0
