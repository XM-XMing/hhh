#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CHECKPOINT MISSION_INDEX OUT_ROOT EPISODE_ID [REPEATS=5]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
MISSION_INDEX="$(realpath "$2")"
OUT_ROOT="$(realpath -m "$3")"
EPISODE_ID="$4"
REPEATS="${5:-5}"

[[ -f "$CHECKPOINT" && -f "$MISSION_INDEX" ]] || {
  echo "checkpoint or mission index does not exist" >&2
  exit 2
}
[[ "$EPISODE_ID" =~ ^[0-9]+$ ]] || {
  echo "EPISODE_ID must be a non-negative integer" >&2
  exit 2
}
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 5 )) || {
  echo "REPEATS must be an integer >= 5" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_ROOT"

for ((repeat=1; repeat<=REPEATS; repeat++)); do
  repeat_tag="$(printf '%02d' "$repeat")"
  echo "EVAL_REPRO_REPEAT repeat=$repeat/$REPEATS episode_id=$EPISODE_ID"
  EVAL_REPRO_EPISODE_ID="$EPISODE_ID" \
    bash "$SCRIPT_DIR/evaluate_policy_unity_managed.sh" \
      "$CHECKPOINT" "$MISSION_INDEX" "$OUT_ROOT/repeat_$repeat_tag" 1
done

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=/home/xm/anaconda3/envs/xm/bin/python
fi
"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/diagnose_policy_eval_divergence.py" \
  --root "$OUT_ROOT" --min-repeats "$REPEATS"
