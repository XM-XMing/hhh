#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT FIXED_DEV_100 OUT_DIR" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
INDEX="$(realpath "$2")"
OUT_DIR="$(realpath -m "$3")"
[[ -f "$CHECKPOINT" && -f "$INDEX" ]] || { echo "checkpoint or index missing" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_FULL_AUDIT_ONLY=1 \
  EVAL_PAIRING_CANDIDATE_COUNT_AUDIT=1 \
  bash "$SCRIPT_DIR/evaluate_policy_unity_managed.sh" "$CHECKPOINT" "$INDEX" "$OUT_DIR" 100

for artifact in first_divergence_trace.json rollout_index.csv pairing_candidate_counts.json; do
  [[ -f "$OUT_DIR/$artifact" ]] || { echo "baseline audit missing $artifact" >&2; exit 1; }
done
echo "RESULT=PASS"
