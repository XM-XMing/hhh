#!/usr/bin/env bash
set -Eeuo pipefail

# Start an isolated ROS/Unity instance, run one formal fixed-holdout
# evaluation, and always tear the simulator down.  A quality-gate rejection
# is valid evaluation evidence and therefore does not make this wrapper fail.

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 CHECKPOINT HOLDOUT_INDEX OUT_DIR [EPISODES=100] [TENSORBOARD_DIR]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
HOLDOUT_INDEX="$(realpath "$2")"
OUT_DIR="$(realpath -m "$3")"
EPISODES="${4:-100}"
TENSORBOARD_DIR="${5:-}"
EVAL_REPRO_EPISODE_ID="${EVAL_REPRO_EPISODE_ID:-}"
EVAL_EPISODE_IDS="${EVAL_EPISODE_IDS:-}"
EVAL_REPRO_MAX_STEPS="${EVAL_REPRO_MAX_STEPS:-}"
EVAL_COMMAND_TICK_AUDIT="${EVAL_COMMAND_TICK_AUDIT:-0}"
EVAL_EXECUTION_TRANSPORT_AUDIT="${EVAL_EXECUTION_TRANSPORT_AUDIT:-0}"
EVAL_ADMISSIBLE_PAIRING_AUDIT="${EVAL_ADMISSIBLE_PAIRING_AUDIT:-0}"
EVAL_PAIRING_CANDIDATE_COUNT_AUDIT="${EVAL_PAIRING_CANDIDATE_COUNT_AUDIT:-0}"
EVAL_FULL_AUDIT_ONLY="${EVAL_FULL_AUDIT_ONLY:-0}"
EVAL_RELIABLE_V4="${EVAL_RELIABLE_V4:-1}"
EVAL_EXPECTED_OBSERVATION_CONTRACT="${EVAL_EXPECTED_OBSERVATION_CONTRACT:-reliable_exact_endpoint_snapshot}"
EVAL_AUDIT_ALLOW_RUNTIME_OVERRIDE="${EVAL_AUDIT_ALLOW_RUNTIME_OVERRIDE:-0}"
EVAL_COLLISION_CACHE="${EVAL_COLLISION_CACHE:-}"
EVAL_NORMALIZER_CHECKPOINT="${EVAL_NORMALIZER_CHECKPOINT:-}"
EVAL_SHADOW_POLICY_CHECKPOINT="${EVAL_SHADOW_POLICY_CHECKPOINT:-}"
EVAL_TRACE_JSONL="${EVAL_TRACE_JSONL:-}"
EVAL_PRIMARY_POLICY_LABEL="${EVAL_PRIMARY_POLICY_LABEL:-bc}"
EVAL_SHADOW_POLICY_LABEL="${EVAL_SHADOW_POLICY_LABEL:-awac}"

if [[ -n "$EVAL_TRACE_JSONL" && -z "$EVAL_SHADOW_POLICY_CHECKPOINT" ]]; then
  echo "EVAL_TRACE_JSONL requires EVAL_SHADOW_POLICY_CHECKPOINT" >&2
  exit 2
fi

if [[ -n "$EVAL_REPRO_EPISODE_ID" && -n "$EVAL_EPISODE_IDS" ]]; then
  echo "EVAL_REPRO_EPISODE_ID and EVAL_EPISODE_IDS are mutually exclusive" >&2
  exit 2
fi
if [[ "$EVAL_EXECUTION_TRANSPORT_AUDIT" == "1" && -z "$EVAL_REPRO_EPISODE_ID" ]]; then
  echo "execution transport audit requires EVAL_REPRO_EPISODE_ID so its fixed-capacity ledger stays bounded" >&2
  exit 2
fi

for path in "$CHECKPOINT" "$HOLDOUT_INDEX"; do
  [[ -f "$path" ]] || { echo "file not found: $path" >&2; exit 2; }
done
if [[ -n "$EVAL_REPRO_EPISODE_ID" ]]; then
  [[ "$EVAL_REPRO_EPISODE_ID" =~ ^[0-9]+$ ]] || {
    echo "EVAL_REPRO_EPISODE_ID must be a non-negative integer" >&2
    exit 2
  }
  [[ "$EPISODES" == "1" ]] || {
    echo "reproducibility diagnosis requires EPISODES=1" >&2
    exit 2
  }
  if [[ -n "$EVAL_REPRO_MAX_STEPS" ]]; then
    [[ "$EVAL_REPRO_MAX_STEPS" =~ ^[0-9]+$ ]] && (( EVAL_REPRO_MAX_STEPS >= 1 )) || {
      echo "EVAL_REPRO_MAX_STEPS must be a positive integer" >&2
      exit 2
    }
  fi
elif [[ -n "$EVAL_EPISODE_IDS" ]]; then
  [[ "$EVAL_EPISODE_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "EVAL_EPISODE_IDS must be a comma-separated list of non-negative integers" >&2
    exit 2
  }
  IFS=',' read -r -a EPISODE_ID_ARRAY <<< "$EVAL_EPISODE_IDS"
  [[ "$EPISODES" =~ ^[0-9]+$ ]] && (( EPISODES == ${#EPISODE_ID_ARRAY[@]} )) || {
    echo "EVAL_EPISODE_IDS count must equal EPISODES" >&2
    exit 2
  }
else
  [[ "$EPISODES" =~ ^[0-9]+$ ]] && (( EPISODES >= 100 )) || {
    echo "formal managed evaluation requires EPISODES >= 100" >&2
    exit 2
  }
fi

validate_summary_identity() {
  local summary_path="$1"
  python3 - "$summary_path" "$CHECKPOINT" "$HOLDOUT_INDEX" "$EPISODES" "$EVAL_REPRO_EPISODE_ID" "$EVAL_EPISODE_IDS" "$EVAL_FULL_AUDIT_ONLY" "$EVAL_EXPECTED_OBSERVATION_CONTRACT" "$EVAL_RELIABLE_V4" "$EVAL_AUDIT_ALLOW_RUNTIME_OVERRIDE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    summary_path,
    checkpoint_path,
    index_path,
    episodes,
    repro_episode_id,
    episode_ids,
    full_audit_only,
    expected_observation_contract,
    reliable_v4,
    allow_runtime_override,
) = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
runtime_observation_contract = (
    "reliable_exact_endpoint_snapshot"
    if reliable_v4 == "1"
    else "legacy_async_telemetry"
)
expected = {
    "evaluation_contract_id": "policy_unity_fixed_holdout",
    "checkpoint_sha256": sha256(checkpoint_path),
    "mission_index_sha256": sha256(index_path),
    "episodes": int(episodes),
    "quality_gate_applicable": not bool(repro_episode_id or episode_ids) and full_audit_only != "1",
    "policy_action_mode": "deterministic_argmax",
    "policy_temperature": 0.0,
    "observation_contract": runtime_observation_contract,
    "observation_source": runtime_observation_contract,
    "checkpoint_observation_contract": expected_observation_contract,
    "checkpoint_observation_source": expected_observation_contract,
    "expected_observation_contract": expected_observation_contract,
    "runtime_observation_contract": runtime_observation_contract,
    "runtime_observation_source": runtime_observation_contract,
    "reliable_execution": reliable_v4 == "1",
    "telemetry_observation": reliable_v4 != "1",
    "telemetry_fallback_enabled": reliable_v4 != "1",
    "snapshot_missing_count": 0,
    "telemetry_lookup_count": 0,
}
if allow_runtime_override != "1":
    expected["runtime_contract_override"] = False
if repro_episode_id:
    expected["episode_filter_ids"] = [int(repro_episode_id)]
elif episode_ids:
    expected["episode_filter_ids"] = sorted(int(value) for value in episode_ids.split(","))
mismatches = {
    key: {"expected": value, "actual": summary.get(key)}
    for key, value in expected.items()
    if summary.get(key) != value
}
if mismatches:
    print(json.dumps(mismatches, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
PY
}

if [[ -f "$OUT_DIR/summary.json" ]]; then
  if ! validate_summary_identity "$OUT_DIR/summary.json"
  then
    echo "existing evaluation summary does not match this checkpoint/index/episode contract: $OUT_DIR/summary.json" >&2
    exit 2
  fi
  echo "MANAGED_POLICY_UNITY_EVAL_ALREADY_COMPLETE"
  echo "  summary: $OUT_DIR/summary.json"
  exit 0
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "incomplete evaluation directory exists: $OUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../package.xml" ]]; then
  PLANNING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  source /opt/ros/noetic/setup.bash
  PLANNING_DIR="$(rospack find planning)"
fi
WORKSPACE="${WORKSPACE:-$(cd "${PLANNING_DIR}/../.." && pwd)}"
source "${WORKSPACE}/devel/setup.bash"

UNITY_BIN="${UNITY_BIN:-${WORKSPACE}/src/unity/XMflight.x86_64}"
RELIABLE_RUNTIME_ID="${EVAL_RELIABLE_RUNTIME_ID:-worker-00-eval}"
RELIABLE_EPISODE_ID="${EVAL_RELIABLE_EPISODE_ID:-episode-0}"
RELIABLE_RESET_ID="${EVAL_RELIABLE_RESET_ID:-reset-0}"
STARTUP_TIMEOUT="${EVAL_STARTUP_TIMEOUT:-60}"
WINDOW_WIDTH="${EVAL_WINDOW_WIDTH:-320}"
WINDOW_HEIGHT="${EVAL_WINDOW_HEIGHT:-240}"
EVALUATOR_DEVICE="${EVALUATOR_DEVICE:-cuda}"
if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]] \
    && [[ -x "${CONDA_PREFIX}/bin/python" ]] \
    && "${CONDA_PREFIX}/bin/python" -c 'import torch' >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
  elif "$(command -v python3)" -c 'import torch' >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="$(command -v python3)"
  elif [[ -x /home/xm/anaconda3/envs/xm/bin/python ]] \
    && /home/xm/anaconda3/envs/xm/bin/python \
      -c 'import torch' >/dev/null 2>&1; then
    PYTHON_EXECUTABLE=/home/xm/anaconda3/envs/xm/bin/python
else
    echo "cannot find a Python interpreter with PyTorch; set PYTHON_EXECUTABLE" >&2
    exit 2
  fi
fi
PRIMITIVE_RESULT_SCHEMA="$(
  PYTHONPATH="${PLANNING_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_EXECUTABLE" -c 'from planning.protocol.constants import PROTOCOL_VERSION; print(PROTOCOL_VERSION)'
)"
TEMP_DIR="${OUT_DIR}.incomplete"
PORT_PROFILE_VALUES="$(
  PYTHONPATH="${PLANNING_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_EXECUTABLE" - "$TEMP_DIR/ros" "$RELIABLE_RUNTIME_ID" <<'PY'
import os
import sys
from pathlib import Path

from planning.runtime.ports import build_managed_evaluation_profile

field_env = {
    "master_port": "EVAL_MASTER_PORT",
    "command_port": "EVAL_CMD_PORT",
    "state_port": "EVAL_STATE_PORT",
    "depth_port": "EVAL_DEPTH_PORT",
    "python_command_port": "EVAL_PYTHON_COMMAND_PORT",
    "unity_command_port": "EVAL_UNITY_COMMAND_PORT",
    "unity_result_port": "EVAL_EXECUTION_RESULT_PORT",
    "python_result_port": "EVAL_PYTHON_RESULT_PORT",
    "unity_snapshot_port": "EVAL_OBSERVATION_SNAPSHOT_PORT",
    "python_snapshot_port": "EVAL_PYTHON_SNAPSHOT_PORT",
}
overrides = {
    field: os.environ[env_name]
    for field, env_name in field_env.items()
    if env_name in os.environ
}
profile = build_managed_evaluation_profile(
    runtime_instance_id=sys.argv[2],
    ros_home=Path(sys.argv[1]),
    overrides=overrides,
)
names = {
    "master_port": "MASTER_PORT",
    "command_port": "CMD_PORT",
    "state_port": "STATE_PORT",
    "depth_port": "DEPTH_PORT",
    "python_command_port": "PYTHON_COMMAND_PORT",
    "unity_command_port": "UNITY_COMMAND_PORT",
    "unity_result_port": "EXECUTION_RESULT_PORT",
    "python_result_port": "PYTHON_RESULT_PORT",
    "unity_snapshot_port": "OBSERVATION_SNAPSHOT_PORT",
    "python_snapshot_port": "PYTHON_SNAPSHOT_PORT",
}
for field, name in names.items():
    print("{}\t{}".format(name, profile.port_mapping()[field]))
print("MASTER_URI\t{}".format(profile.ros_master_uri))
PY
)"
while IFS=$'\t' read -r profile_name profile_value; do
  case "$profile_name" in
    MASTER_PORT) MASTER_PORT="$profile_value" ;;
    CMD_PORT) CMD_PORT="$profile_value" ;;
    STATE_PORT) STATE_PORT="$profile_value" ;;
    DEPTH_PORT) DEPTH_PORT="$profile_value" ;;
    PYTHON_COMMAND_PORT) PYTHON_COMMAND_PORT="$profile_value" ;;
    UNITY_COMMAND_PORT) UNITY_COMMAND_PORT="$profile_value" ;;
    EXECUTION_RESULT_PORT) EXECUTION_RESULT_PORT="$profile_value" ;;
    PYTHON_RESULT_PORT) PYTHON_RESULT_PORT="$profile_value" ;;
    OBSERVATION_SNAPSHOT_PORT) OBSERVATION_SNAPSHOT_PORT="$profile_value" ;;
    PYTHON_SNAPSHOT_PORT) PYTHON_SNAPSHOT_PORT="$profile_value" ;;
    MASTER_URI) MASTER_URI="$profile_value" ;;
    *) echo "unknown managed port profile field: $profile_name" >&2; exit 2 ;;
  esac
done <<< "$PORT_PROFILE_VALUES"
ROS_HOME_DIR="${TEMP_DIR}/ros"
LOG_DIR="${TEMP_DIR}/runtime_logs"
WORKER_SPEC_FILE="${TEMP_DIR}/worker_runtime_specs.json"
BRIDGE_COMMAND_AUDIT_PATH=""
BRIDGE_EXECUTION_TRANSPORT_AUDIT_PATH=""
UNITY_EXECUTION_TRANSPORT_AUDIT_PATH=""
PLANNING_EXECUTION_TRANSPORT_AUDIT_PATH=""
EXECUTION_TRANSPORT_AUDIT_RUN_ID=""
UNITY_RUNTIME_IDENTITY=""
if [[ "$EVAL_COMMAND_TICK_AUDIT" == "1" ]]; then
  BRIDGE_COMMAND_AUDIT_PATH="$LOG_DIR/bridge_command_audit.jsonl"
fi
if [[ "$EVAL_EXECUTION_TRANSPORT_AUDIT" == "1" ]]; then
  BRIDGE_EXECUTION_TRANSPORT_AUDIT_PATH="$LOG_DIR/bridge_execution_transport_audit.json"
  UNITY_EXECUTION_TRANSPORT_AUDIT_PATH="${EVAL_UNITY_EXECUTION_TRANSPORT_AUDIT_PATH:-$LOG_DIR/unity_execution_transport_audit.json}"
  printf -v audit_episode_filename 'execution_transport_audit_episode_%06d.json' "$EVAL_REPRO_EPISODE_ID"
  PLANNING_EXECUTION_TRANSPORT_AUDIT_PATH="$TEMP_DIR/$audit_episode_filename"
  EXECUTION_TRANSPORT_AUDIT_RUN_ID="exec-transport-$(date +%s%N)-$$"
fi

[[ -x "$UNITY_BIN" ]] || {
  echo "Unity executable not found: $UNITY_BIN" >&2
  exit 2
}
PORTS_TO_CHECK=("$MASTER_PORT" "$CMD_PORT" "$STATE_PORT" "$DEPTH_PORT")
if [[ "$EVAL_RELIABLE_V4" == "1" ]]; then
  PORTS_TO_CHECK+=(
    "$PYTHON_COMMAND_PORT" "$UNITY_COMMAND_PORT" "$EXECUTION_RESULT_PORT"
    "$OBSERVATION_SNAPSHOT_PORT" "$PYTHON_RESULT_PORT" "$PYTHON_SNAPSHOT_PORT"
  )
fi
for port in "${PORTS_TO_CHECK[@]}"; do
  if ss -lntupH 2>/dev/null | grep -Eq "[:.]${port}[[:space:]]"; then
    echo "managed evaluation port already in use: $port" >&2
    exit 2
  fi
done

"$PYTHON_EXECUTABLE" - "$TEMP_DIR" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(str(path))
PY
mkdir -p "$ROS_HOME_DIR" "$LOG_DIR"

"$PYTHON_EXECUTABLE" - "$WORKER_SPEC_FILE" "$ROS_HOME_DIR" \
  "$RELIABLE_RUNTIME_ID" "$MASTER_PORT" "$CMD_PORT" "$STATE_PORT" \
  "$DEPTH_PORT" "$PYTHON_COMMAND_PORT" "$UNITY_COMMAND_PORT" \
  "$EXECUTION_RESULT_PORT" "$PYTHON_RESULT_PORT" \
  "$OBSERVATION_SNAPSHOT_PORT" "$PYTHON_SNAPSHOT_PORT" <<'PY'
import sys
from pathlib import Path

from planning.runtime.ports import build_managed_evaluation_profile
from planning.runtime.worker import write_worker_runtime_spec_file

fields = (
    "master_port",
    "command_port",
    "state_port",
    "depth_port",
    "python_command_port",
    "unity_command_port",
    "unity_result_port",
    "python_result_port",
    "unity_snapshot_port",
    "python_snapshot_port",
)
values = [int(value) for value in sys.argv[4:]]
if len(values) != len(fields):
    raise SystemExit("managed evaluation profile field count mismatch")
profile = build_managed_evaluation_profile(
    runtime_instance_id=sys.argv[3],
    ros_home=Path(sys.argv[2]),
    overrides=dict(zip(fields, values)),
)
write_worker_runtime_spec_file(Path(sys.argv[1]), (profile,))
PY

UNITY_AUDIT_ARGS=()
if [[ -n "$UNITY_EXECUTION_TRANSPORT_AUDIT_PATH" ]]; then
  UNITY_MANAGED_ASSEMBLY="${UNITY_BIN%.x86_64}_Data/Managed/Assembly-CSharp.dll"
  [[ -f "$UNITY_MANAGED_ASSEMBLY" ]] || {
    echo "Unity diagnostic audit requires Assembly-CSharp.dll: $UNITY_MANAGED_ASSEMBLY" >&2
    exit 2
  }
  UNITY_RUNTIME_IDENTITY="player=$(sha256sum "$UNITY_BIN" | awk '{print $1}');assembly_csharp=$(sha256sum "$UNITY_MANAGED_ASSEMBLY" | awk '{print $1}')"
  UNITY_AUDIT_EPISODE_ID="${EVAL_REPRO_EPISODE_ID:--1}"
  UNITY_AUDIT_ARGS=(
    -executionTransportAuditPath "$UNITY_EXECUTION_TRANSPORT_AUDIT_PATH"
    -executionTransportAuditEpisodeId "$UNITY_AUDIT_EPISODE_ID"
    -executionTransportAuditRunId "$EXECUTION_TRANSPORT_AUDIT_RUN_ID"
    -executionTransportAuditRuntimeIdentity "$UNITY_RUNTIME_IDENTITY"
  )
fi

UNITY_EXTRA_ARGS=()
for value in "${UNITY_AUDIT_ARGS[@]}"; do
  UNITY_EXTRA_ARGS+=(--unity-arg "$value")
done
BRIDGE_EXTRA_ARGS=(
  "command_audit_path:=$BRIDGE_COMMAND_AUDIT_PATH"
  "execution_transport_audit_path:=$BRIDGE_EXECUTION_TRANSPORT_AUDIT_PATH"
  "execution_transport_audit_run_id:=$EXECUTION_TRANSPORT_AUDIT_RUN_ID"
  "execution_transport_audit_episode_id:=$EVAL_REPRO_EPISODE_ID"
  "execution_transport_audit_unity_runtime_identity:=$UNITY_RUNTIME_IDENTITY"
)
BRIDGE_RUNTIME_ARGS=()
for value in "${BRIDGE_EXTRA_ARGS[@]}"; do
  BRIDGE_RUNTIME_ARGS+=(--bridge-arg "$value")
done

eval_args=(
  --checkpoint "$CHECKPOINT"
  --index "$HOLDOUT_INDEX"
  --out-dir "$TEMP_DIR"
  --max-episodes "$EPISODES"
  --device "$EVALUATOR_DEVICE"
)
if [[ -n "$EVAL_REPRO_EPISODE_ID" ]]; then
  eval_args+=(
    --audit-only
    --episode-ids "$EVAL_REPRO_EPISODE_ID"
    --first-divergence-trace
  )
  if [[ -n "$EVAL_REPRO_MAX_STEPS" ]]; then
    eval_args+=(--max-steps "$EVAL_REPRO_MAX_STEPS")
  fi
  if [[ "$EVAL_COMMAND_TICK_AUDIT" == "1" ]]; then
    eval_args+=(--command-tick-audit)
  fi
  if [[ "$EVAL_EXECUTION_TRANSPORT_AUDIT" == "1" ]]; then
    eval_args+=(
      --execution-transport-audit
      --execution-transport-audit-run-id "$EXECUTION_TRANSPORT_AUDIT_RUN_ID"
      --execution-transport-audit-unity-runtime-identity "$UNITY_RUNTIME_IDENTITY"
    )
  fi
  if [[ "$EVAL_ADMISSIBLE_PAIRING_AUDIT" == "1" ]]; then
    eval_args+=(--admissible-pairing-audit)
  fi
elif [[ -n "$EVAL_EPISODE_IDS" ]]; then
  eval_args+=(
    --audit-only
    --episode-ids "$EVAL_EPISODE_IDS"
    --first-divergence-trace
  )
fi
if [[ "$EVAL_FULL_AUDIT_ONLY" == "1" ]]; then
  eval_args+=(--audit-only --first-divergence-trace)
  if [[ "$EVAL_PAIRING_CANDIDATE_COUNT_AUDIT" == "1" ]]; then
    eval_args+=(--pairing-candidate-count-audit)
  fi
fi
if [[ "$EVAL_AUDIT_ALLOW_RUNTIME_OVERRIDE" == "1" ]]; then
  eval_args+=(--audit-allow-runtime-override)
fi
if [[ -n "$TENSORBOARD_DIR" ]]; then
  eval_args+=(--tensorboard-log-dir "$(realpath -m "$TENSORBOARD_DIR")")
fi
if [[ -n "$EVAL_EXPECTED_OBSERVATION_CONTRACT" ]]; then
  eval_args+=(--expected-observation-contract "$EVAL_EXPECTED_OBSERVATION_CONTRACT")
fi
if [[ -n "$EVAL_COLLISION_CACHE" ]]; then
  eval_args+=(--collision-cache "$EVAL_COLLISION_CACHE")
fi
if [[ -n "$EVAL_NORMALIZER_CHECKPOINT" ]]; then
  eval_args+=(--normalizer-checkpoint "$EVAL_NORMALIZER_CHECKPOINT")
fi
if [[ -n "$EVAL_SHADOW_POLICY_CHECKPOINT" ]]; then
  TRACE_PATH="${EVAL_TRACE_JSONL:-${OUT_DIR}/shadow_trace.jsonl}"
  eval_args+=(
    --shadow-policy-checkpoint "$(realpath "$EVAL_SHADOW_POLICY_CHECKPOINT")"
    --trace-jsonl "$TRACE_PATH"
    --primary-policy-label "$EVAL_PRIMARY_POLICY_LABEL"
    --shadow-policy-label "$EVAL_SHADOW_POLICY_LABEL"
  )
fi
if [[ "$EVAL_RELIABLE_V4" == "1" ]]; then
  eval_args+=(
    --reliable-v4
    --reliable-v4-runtime-instance-id "$RELIABLE_RUNTIME_ID"
    --reliable-v4-command-endpoint "tcp://127.0.0.1:$PYTHON_COMMAND_PORT"
    --reliable-v4-result-endpoint "tcp://127.0.0.1:$PYTHON_RESULT_PORT"
    --reliable-v4-snapshot-endpoint "tcp://127.0.0.1:$PYTHON_SNAPSHOT_PORT"
  )
fi

MANAGED_RUNTIME_MODE_ARGS=()
if [[ "$EVAL_RELIABLE_V4" == "1" ]]; then
  MANAGED_RUNTIME_MODE_ARGS+=(--reliable-v4)
fi

set +e
"$PYTHON_EXECUTABLE" scripts/run_managed_runtime.py \
  --worker-spec-file "$WORKER_SPEC_FILE" \
  --worker-count 1 \
  --worker-id 0 \
  --workspace "$WORKSPACE" \
  --log-dir "$LOG_DIR" \
  --unity-bin "$UNITY_BIN" \
  --observation-contract "$EVAL_EXPECTED_OBSERVATION_CONTRACT" \
  --max-steps "${EVAL_REPRO_MAX_STEPS:-45}" \
  --startup-timeout "$STARTUP_TIMEOUT" \
  "${MANAGED_RUNTIME_MODE_ARGS[@]}" \
  "${UNITY_EXTRA_ARGS[@]}" \
  "${BRIDGE_RUNTIME_ARGS[@]}" \
  -- "$PYTHON_EXECUTABLE" -m planning.evaluation.policy_evaluator "${eval_args[@]}"
evaluation_status="$?"
set -e

# The shared managed owner has already closed the isolated runtime before this
# wrapper validates or publishes TEMP_DIR.  This preserves the audit boundary
# without maintaining a second shell cleanup implementation.
if [[ "$EVAL_EXECUTION_TRANSPORT_AUDIT" == "1" ]]; then
  for audit_artifact in \
    "$UNITY_EXECUTION_TRANSPORT_AUDIT_PATH" \
    "$BRIDGE_EXECUTION_TRANSPORT_AUDIT_PATH" \
    "$PLANNING_EXECUTION_TRANSPORT_AUDIT_PATH"; do
    if [[ ! -s "$audit_artifact" ]]; then
      echo "managed evaluation is missing required execution transport audit artifact: $audit_artifact" >&2
      exit 1
    fi
  done
fi

if [[ ! -f "$TEMP_DIR/summary.json" ]]; then
  echo "managed evaluation did not produce summary.json" >&2
  exit "${evaluation_status:-1}"
fi
if ! validate_summary_identity "$TEMP_DIR/summary.json"; then
  echo "managed evaluation produced evidence with the wrong checkpoint/index/episode contract" >&2
  exit 1
fi
if (( evaluation_status > 1 )); then
  echo "managed evaluation failed with status $evaluation_status" >&2
  exit "$evaluation_status"
fi

mkdir -p "$(dirname "$OUT_DIR")"
mv "$TEMP_DIR" "$OUT_DIR"
echo "MANAGED_POLICY_UNITY_EVAL_COMPLETE"
echo "  checkpoint: $CHECKPOINT"
echo "  summary: $OUT_DIR/summary.json"
echo "  quality_gate_status: $evaluation_status"
echo "RESULT=PASS"
