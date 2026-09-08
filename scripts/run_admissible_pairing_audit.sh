#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CHECKPOINT MISSION_INDEX OUT_DIR [EPISODE_ID=451]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
MISSION_INDEX="$(realpath "$2")"
OUT_DIR="$(realpath -m "$3")"
EPISODE_ID="${4:-451}"

[[ -f "$CHECKPOINT" && -f "$MISSION_INDEX" ]] || {
  echo "checkpoint or mission index does not exist" >&2
  exit 2
}
[[ "$EPISODE_ID" =~ ^[0-9]+$ ]] || {
  echo "EPISODE_ID must be a non-negative integer" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_REPRO_EPISODE_ID="$EPISODE_ID" \
EVAL_ADMISSIBLE_PAIRING_AUDIT=1 \
  bash "$SCRIPT_DIR/evaluate_policy_unity_managed.sh" \
    "$CHECKPOINT" "$MISSION_INDEX" "$OUT_DIR" 1

REPORT="$OUT_DIR/pairing_counterfactual_episode_$(printf '%06d' "$EPISODE_ID").json"
[[ -f "$REPORT" ]] || {
  echo "pairing counterfactual report missing: $REPORT" >&2
  exit 1
}

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=/home/xm/anaconda3/envs/xm/bin/python
fi
"$PYTHON_EXECUTABLE" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = report["summary"]
print(
    "PAIRING_COUNTERFACTUAL_RESULT"
    " classification={classification} decisions={policy_decision_count}"
    " pairs={total_admissible_pairs} max_pairs={max_pairs}"
    " action_change_steps={action_steps}".format(
        classification=summary["classification"],
        policy_decision_count=summary["policy_decision_count"],
        total_admissible_pairs=summary["total_admissible_pairs"],
        max_pairs=summary["admissible_pair_count_distribution"]["max"],
        action_steps=summary["steps_with_multiple_top1_actions"],
    )
)
PY
