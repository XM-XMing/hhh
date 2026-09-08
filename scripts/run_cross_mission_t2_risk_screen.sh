#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT FIXED_DEV_100 SELECTION_JSON OUT_ROOT" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
INDEX="$(realpath "$2")"
SELECTION="$(realpath "$3")"
OUT_ROOT="$(realpath -m "$4")"
[[ -f "$CHECKPOINT" && -f "$INDEX" && -f "$SELECTION" ]] || {
  echo "checkpoint, index, or selection manifest missing" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/xm/anaconda3/envs/xm/bin/python}"
mapfile -t EPISODES < <("$PYTHON_EXECUTABLE" - "$SELECTION" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
selection = list(payload.get("selection", []))
if len(selection) != 20:
    raise SystemExit("selection must contain exactly 20 immutable missions")
for item in selection:
    print(int(item["episode_id"]))
PY
)

mkdir -p "$OUT_ROOT/missions"
for episode_id in "${EPISODES[@]}"; do
  mission_root="$OUT_ROOT/missions/$(printf 'episode_%06d' "$episode_id")"
  for repeat in 1 2 3 4 5; do
    repeat_dir="$mission_root/$(printf 'repeat_%03d' "$repeat")"
    echo "CROSS_MISSION_T2_REPEAT episode=$episode_id repeat=$repeat/5"
    EVAL_REPRO_EPISODE_ID="$episode_id" \
      bash "$SCRIPT_DIR/evaluate_policy_unity_managed.sh" \
        "$CHECKPOINT" "$INDEX" "$repeat_dir" 1
  done
  set +e
  "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/diagnose_cross_mission_t2_risk_screen.py" \
    --root "$OUT_ROOT" --selection "$SELECTION" --repeats 5
  diagnosis_status="$?"
  set -e
  if (( diagnosis_status == 1 )); then
    echo "CROSS_MISSION_T2_EARLY_STOP episode=$episode_id classification=T1" >&2
    exit 1
  fi
  (( diagnosis_status == 0 )) || exit "$diagnosis_status"
done

"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/diagnose_cross_mission_t2_risk_screen.py" \
  --root "$OUT_ROOT" --selection "$SELECTION" --repeats 5 --require-all
echo "RESULT=PASS"
