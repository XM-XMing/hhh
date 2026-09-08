from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "aggregate_bc_scale_evaluation_300.py"
SPEC = importlib.util.spec_from_file_location("aggregate_bc_scale_evaluation_300", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _paired(left_fail_right_success=0, left_success_right_fail=0):
    return {
        "left_fail_right_success": left_fail_right_success,
        "left_success_right_fail": left_success_right_fail,
        "mcnemar_exact_two_sided_p": MODULE._COMMON._exact_mcnemar_pvalue(
            left_fail_right_success, left_success_right_fail
        ),
    }


def test_decision_rules_cover_saturation_clear_winner_and_inconclusive():
    assert (
        MODULE.classify_conclusion(0.78, 0.77, _paired(20, 20))
        == "40K_60K_STATISTICALLY_SIMILAR_DATA_SATURATION"
    )
    assert MODULE.classify_conclusion(0.70, 0.77, _paired(30, 2)) == "60K_CLEARLY_BETTER"
    assert MODULE.classify_conclusion(0.77, 0.70, _paired(2, 30)) == "40K_OUTPERFORMS_60K"
    assert MODULE.classify_conclusion(0.75, 0.79, _paired(5, 1)) == "INCONCLUSIVE_AT_300"


def test_path_length_uses_step_endpoint_segments(tmp_path: Path):
    steps = tmp_path / "steps"
    steps.mkdir()
    path = steps / "episode_000000.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "x_before",
                "y_before",
                "z_before",
                "x_after",
                "y_after",
                "z_after",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "0",
                "x_before": 0,
                "y_before": 0,
                "z_before": 0,
                "x_after": 3,
                "y_after": 4,
                "z_after": 0,
            }
        )
        writer.writerow(
            {
                "episode_id": "0",
                "x_before": 3,
                "y_before": 4,
                "z_before": 0,
                "x_after": 3,
                "y_after": 4,
                "z_after": 12,
            }
        )
    assert MODULE._row_path_length(tmp_path, {"episode_id": "0", "step_csv": "steps/episode_000000.csv"}) == 17.0
