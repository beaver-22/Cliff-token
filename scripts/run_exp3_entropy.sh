#!/usr/bin/env bash
# RQ2-1 Batch: run Phase B (cliff logprobs/rank/entropy) across all
# (model, dataset) combinations on a single GPU, sequentially.
# Phase C (greedy rollout) is run only for --phase_c_models (default: Qwen3-8B).
# After all runs, aggregate into 4 slim outputs.
set -euo pipefail

export VLLM_USE_V1=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ----- Cleanup trap: kill all descendants on exit/interrupt -----
cleanup() {
  local sig=$1
  echo ""
  echo "[cleanup] received $sig, killing all descendant processes..."
  kill_descendants() {
    local parent=$1
    local children
    children=$(pgrep -P "$parent" 2>/dev/null || true)
    for child in $children; do
      kill_descendants "$child"
    done
    [[ "$parent" != "$$" ]] && kill -TERM "$parent" 2>/dev/null || true
  }
  kill_descendants $$
  sleep 2
  kill_descendants_force() {
    local parent=$1
    local children
    children=$(pgrep -P "$parent" 2>/dev/null || true)
    for child in $children; do
      kill_descendants_force "$child"
    done
    [[ "$parent" != "$$" ]] && kill -KILL "$parent" 2>/dev/null || true
  }
  kill_descendants_force $$
  echo "[cleanup] done."
  exit 130
}
trap 'cleanup INT'  INT
trap 'cleanup TERM' TERM
trap 'cleanup HUP'  HUP

MODELS=""
DATASETS=""
GPU_LIST="0"
ROLLOUT_DIR="./output/03_rollout"
OUTPUT_DIR=""
NUM_SAMPLES=64
AGGREGATE_ONLY=0
PHASE_C_MODELS="Qwen3-8B"
RUNS_DIR_OVERRIDE=""
BASELINE_DIR_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp3_entropy.sh [options]

RQ2-1 Batch (multi-GPU free-pool scheduler).

Options:
  --models "m1,m2,..."      Model dir names (default: auto-discover)
  --datasets "d1,d2,..."    Dataset names (default: auto-discover)
  --gpus "G1,G2,..."        GPU ids comma-separated (default: 0)
                             alias: --gpu (singular)
  --rollout_dir DIR         Rollout root (default: ./output/03_rollout)
  --output_dir DIR          Output base (default: ./output/06_entropy_rank/<ts>_batch)
  --num_samples N           Phase C rollout samples (default: 64)
  --phase_c_models "m1,..." Models that run Phase C (default: Qwen3-8B)
  --aggregate_only          Skip GPU runs, only re-aggregate
  --runs_dir DIR            Override runs dir for aggregation (default: <output_dir>/runs)
  --baseline_dir DIR        Baseline dir with per-token rank/entropy data
                            (default: output/02_token_stats)

Examples:
  # Paper batch
  bash scripts/run_exp3_entropy.sh \
      --models "Qwen3-0.6B,Qwen3-4B,Qwen3-8B,Llama-3.2-1B-Instruct,Llama-3.2-3B-Instruct,Llama-3.1-8B-Instruct,gemma-3-4b-it" \
      --datasets "gsm1k_100,math500_100,aime25" \
      --gpus 0

  # Re-run aggregation only on existing output
  bash scripts/run_exp3_entropy.sh --aggregate_only \
      --output_dir output/06_entropy_rank/0410_batch
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)         MODELS="$2"; shift 2 ;;
    --datasets)       DATASETS="$2"; shift 2 ;;
    --gpu|--gpus)     GPU_LIST="$2"; shift 2 ;;
    --rollout_dir)    ROLLOUT_DIR="$2"; shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --num_samples)    NUM_SAMPLES="$2"; shift 2 ;;
    --phase_c_models) PHASE_C_MODELS="$2"; shift 2 ;;
    --aggregate_only) AGGREGATE_ONLY=1; shift ;;
    --runs_dir)       RUNS_DIR_OVERRIDE="$2"; shift 2 ;;
    --baseline_dir)   BASELINE_DIR_OVERRIDE="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  TS=$(date +%m%d_%H%M%S)
  OUTPUT_DIR="./output/06_entropy_rank/${TS}_batch"
fi

RUNS_DIR="$OUTPUT_DIR/runs"
mkdir -p "$RUNS_DIR"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "============================================================"
echo "RQ2-1 Batch (GPUs: ${GPU_LIST}, parallel free-pool)"
echo "============================================================"
echo "  rollout_dir:    $ROLLOUT_DIR"
echo "  output_dir:     $OUTPUT_DIR"
echo "  num_samples:    $NUM_SAMPLES"
echo "  phase_c_models: $PHASE_C_MODELS"
echo "  num_gpus:       $NUM_GPUS"
echo ""

# ---- Auto-discover models if not given ----
if [[ -z "$MODELS" ]]; then
  MODELS=$(ls -1 "$ROLLOUT_DIR" 2>/dev/null \
            | grep -vE '^(\.|Qwen3-8B-greedy)' \
            | tr '\n' ',' | sed 's/,$//')
fi
if [[ -z "$DATASETS" ]]; then
  DATASETS="gsm1k_100,math500_100,aime25"
fi
echo "  models:   $MODELS"
echo "  datasets: $DATASETS"
echo ""

# ---- Build (model, dataset) work queue ----
COMBOS_FILE=$(mktemp)
trap 'rm -f "$COMBOS_FILE"' EXIT
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

for M in "${MODEL_ARR[@]}"; do
  for D in "${DS_ARR[@]}"; do
    DATA_PATH="$ROLLOUT_DIR/$M/${D}_all_paths.json"
    if [[ -f "$DATA_PATH" ]]; then
      printf "%s\t%s\t%s\n" "$M" "$D" "$DATA_PATH" >> "$COMBOS_FILE"
    fi
  done
done

N_COMBOS=$(wc -l < "$COMBOS_FILE")
echo "Discovered $N_COMBOS (model, dataset) combinations:"
awk -F'\t' '{print "  " $1 "/" $2}' "$COMBOS_FILE"
echo ""

# Helper: is model in PHASE_C_MODELS list?
is_phase_c_model() {
  local m="$1"
  local x
  IFS=',' read -ra PCM <<< "$PHASE_C_MODELS"
  for x in "${PCM[@]}"; do
    [[ "$x" == "$m" ]] && return 0
  done
  return 1
}

# ---- Phase 1: GPU runs (free-pool parallel) ----
if [[ "$AGGREGATE_ONLY" -eq 0 ]]; then
  echo "============================================================"
  echo "Phase 1: Parallel runs on GPUs ${GPU_LIST}"
  echo "============================================================"

  # Free GPU pool
  declare -A PID_TO_GPU=()
  declare -A PID_TO_LABEL=()
  FREE_GPUS=("${GPUS[@]}")

  reap_one() {
    # Wait for any background job to finish, then return its GPU to the pool
    wait -n 2>/dev/null || true
    for pid in "${!PID_TO_GPU[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        FREE_GPUS+=("${PID_TO_GPU[$pid]}")
        echo "  [done] ${PID_TO_LABEL[$pid]} on GPU ${PID_TO_GPU[$pid]}"
        unset 'PID_TO_GPU[$pid]'
        unset 'PID_TO_LABEL[$pid]'
        return 0
      fi
    done
  }

  IDX=0
  while IFS=$'\t' read -r MODEL DS DATA_PATH; do
    IDX=$((IDX + 1))
    RUN_OUT="$RUNS_DIR/${MODEL}_${DS}"
    LOG="$RUN_OUT/run.log"

    # Decide Phase C
    if is_phase_c_model "$MODEL"; then
      RUN_PHASE_C=1
      MARKER="$RUN_OUT/greedy_replace_results.json"
    else
      RUN_PHASE_C=0
      MARKER="$RUN_OUT/cliff_logprobs.json"
    fi

    # skip-existing
    if [[ -f "$MARKER" ]]; then
      echo "  [$IDX/$N_COMBOS] [skip-existing] $MODEL/$DS  ($MARKER exists)"
      continue
    fi

    # Wait until a GPU is free
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
      reap_one
    done

    GPU="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    mkdir -p "$RUN_OUT"
    MODEL_ALIAS=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -e 's/-instruct$//' -e 's/-it$//')

    # Hard wall-clock cap as a safety net. The Python script ends with
    # os._exit(0), so this should rarely fire — only on init/runtime hangs.
    if [[ "$RUN_PHASE_C" == "1" ]]; then
      JOB_TIMEOUT="90m"   # Phase B + Phase C (greedy rollout)
    else
      JOB_TIMEOUT="30m"   # Phase B only
    fi

    echo "  [$IDX/$N_COMBOS] [start] $MODEL/$DS on GPU $GPU  alias=$MODEL_ALIAS  phase_c=$RUN_PHASE_C  log=$LOG"
    (
      CUDA_VISIBLE_DEVICES="$GPU" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        timeout --signal=KILL --kill-after=10s "$JOB_TIMEOUT" \
        python3 scripts/_run_exp3_entropy_phaseB.py \
          "$MODEL_ALIAS" "$DS" "$DATA_PATH" "$GPU" "$RUN_OUT" "$NUM_SAMPLES" "$RUN_PHASE_C" \
          > "$LOG" 2>&1 \
        || echo "  [$IDX/$N_COMBOS] [FAIL/TIMEOUT] $MODEL/$DS — see $LOG"
    ) &
    PID_TO_GPU[$!]=$GPU
    PID_TO_LABEL[$!]="$MODEL/$DS"
  done < "$COMBOS_FILE"

  # Wait for all remaining jobs
  while [[ ${#PID_TO_GPU[@]} -gt 0 ]]; do
    reap_one
  done
  echo "  All parallel runs finished."
else
  echo "Skipping Phase 1 (--aggregate_only)"
fi

# ---- Phase 2: Aggregation ----
echo ""
echo "============================================================"
echo "Phase 2: Aggregation"
echo "============================================================"
AGG_ARGS=("$OUTPUT_DIR")
[[ -n "$RUNS_DIR_OVERRIDE" ]]    && AGG_ARGS+=(--runs_dir "$RUNS_DIR_OVERRIDE")
[[ -n "$BASELINE_DIR_OVERRIDE" ]] && AGG_ARGS+=(--baseline_dir "$BASELINE_DIR_OVERRIDE")
python3 -m src.analysis.exp3_entropy_aggregator "${AGG_ARGS[@]}"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR"/*.{csv,png} 2>/dev/null || true
