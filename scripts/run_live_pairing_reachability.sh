#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CHECKPOINT MISSION_INDEX OUT_ROOT EPISODE_ID [REPEATS=50]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
MISSION_INDEX="$(realpath "$2")"
OUT_ROOT="$(realpath -m "$3")"
EPISODE_ID="$4"
REPEATS="${5:-50}"

[[ -f "$CHECKPOINT" && -f "$MISSION_INDEX" ]] || {
  echo "checkpoint or mission index does not exist" >&2
  exit 2
}
[[ "$EPISODE_ID" =~ ^[0-9]+$ ]] || {
  echo "EPISODE_ID must be a non-negative integer" >&2
  exit 2
}
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 50 )) || {
  echo "REPEATS must be an integer >= 50" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=/home/xm/anaconda3/envs/xm/bin/python
fi
mkdir -p "$OUT_ROOT"

for ((repeat=1; repeat<=REPEATS; repeat++)); do
  repeat_tag="$(printf '%03d' "$repeat")"
  echo "LIVE_PAIRING_REPEAT repeat=$repeat/$REPEATS episode_id=$EPISODE_ID"
  EVAL_REPRO_EPISODE_ID="$EPISODE_ID" \
    bash "$SCRIPT_DIR/evaluate_policy_unity_managed.sh" \
      "$CHECKPOINT" "$MISSION_INDEX" "$OUT_ROOT/repeat_$repeat_tag" 1

  set +e
  "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/diagnose_live_pairing_reachability.py" \
    --root "$OUT_ROOT" --min-repeats "$repeat"
  diagnosis_status="$?"
  set -e
  if (( diagnosis_status == 1 )); then
    echo "LIVE_PAIRING_EARLY_STOP repeat=$repeat classification=T1"
    exit 1
  fi
  if (( diagnosis_status != 0 )); then
    exit "$diagnosis_status"
  fi
done

"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/diagnose_live_pairing_reachability.py" \
  --root "$OUT_ROOT" --min-repeats "$REPEATS"
