#!/usr/bin/env bash
# Usage: ./run_queue.sh [--after PID]
set -u

# ---- edit this list: "working_dir|script [args]" ----
JOBS=(
  "08-25 - new - Hebb vs stochastic parameterization|1A - FF - cotuned.py --stage generate"
  "08-25 - new - Hebb vs stochastic parameterization|1A - FF - cotuned.py --stage plot"
  "08-25 - new - Hebb vs stochastic parameterization|1B - FF - random.py --stage generate"
  "08-25 - new - Hebb vs stochastic parameterization|1B - FF - random.py --stage plot"
#   "08-25 - new - Hebb vs stochastic parameterization|2A - rec - F to E plastic - cotuned.py --stage generate"
#     "08-25 - new - Hebb vs stochastic parameterization|2A - rec - F to E plastic - cotuned.py --stage plot"
)
STOP_ON_FAILURE=false
PYTHON="$HOME/miniforge3/envs/drift/bin/python"
# -----------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$ROOT/queue_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Optionally wait for an already-running process to finish first
if [[ "${1:-}" == "--after" && -n "${2:-}" ]]; then
  echo "Waiting for PID $2 ..."
  tail --pid="$2" -f /dev/null
fi

i=0
for job in "${JOBS[@]}"; do
  i=$((i+1))
  dir="${job%%|*}"; cmd="${job#*|}"
  script="${cmd%%.py*}.py"; args="${cmd#"$script"}"
  log="$LOGDIR/$(printf '%02d' $i) - ${script%.py}.log"

  echo "[$(date '+%F %T')] START ($i/${#JOBS[@]}): $cmd"
  start=$SECONDS
  ( cd "$ROOT/$dir" && eval "\"$PYTHON\" -u \"$script\" $args" ) > "$log" 2>&1
  code=$?
  echo "[$(date '+%F %T')] END   ($i/${#JOBS[@]}): exit=$code, $(( (SECONDS-start)/60 )) min -> $log"

  if [[ $code -ne 0 && "$STOP_ON_FAILURE" == true ]]; then
    echo "Stopping queue due to failure."; exit $code
  fi
done
echo "All jobs done."