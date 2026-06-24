#!/usr/bin/env bash
# RQ2-2: Cliff top-k candidate replacement experiment (cell-priority outputs).
set -euo pipefail

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODEL="qwen3-8b"
DATASETS="math500_100,gsm1k_100,aime25"
ROLLOUT_DIR="./output/03_rollout"
GPU_LIST="0"
PARALLEL_MODE="auto"   # auto | data | tensor
MODE=""
TEMPERATURE=""
NUM_SAMPLES=64
TOP_K=20
OUTPUT_DIR=""
ROLLOUT_CELLS="all"
EXTRACT_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=16
MAX_CLIFFS_PER_DATASET=0
MAX_CANDIDATES_PER_CLIFF=0
GPU_MEM=0.65
EXP3_RUNS_DIR=""
LOG_FILE=""
ANALYSIS_ONLY=0
ANALYSIS_SUBDIR="analysis"
TARGET_TEMPS="0,0.7,1,2,5"

declare -a CHILD_PIDS=()
declare -a SHARD_DIRS=()
CLEANUP_RUNNING=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp4_candidates.sh [options]

Options:
  --model NAME                 Model alias (default: qwen3-8b)
  --datasets "d1,d2,..."       Datasets to search (default: math500_100,gsm1k_100,aime25)
  --rollout_dir PATH           Root with <model_short>/<dataset>_all_paths.json
  --gpus "0"                   GPU IDs (e.g. "0,1" or "0-3")
  --parallel_mode MODE         auto|data|tensor (default: auto)
  --mode MODE                  thinking|non_thinking (default: model default)
  --temperature FLOAT          Sampling temperature override
  --num_samples N              Rollout samples per candidate token (default: 64)
  --top_k N                    Candidate top-k per cliff position (default: 20)
  --rollout_cells STR          all OR comma-separated cell labels (default: all)
  --extract_batch_size N       Top-k extraction prompt batch size (default: 128)
  --rollout_batch_size N       Candidate prompt batch size for rollout (default: 16)
  --max_cliffs_per_dataset N   0=no cap (smoke-test helper)
  --max_candidates_per_cliff N 0=use top_k (smoke-test helper)
  --gpu_mem F                  vLLM gpu_memory_utilization (default: 0.65)
  --exp3_runs_dir PATH         Optional exp3_entropy runs dir for consistency check
  --output_dir PATH            Output dir (default: output/07_candidate_replacement/<model_short>_<timestamp>)
  --log_file PATH              Log file path (default: <output_dir>/run_exp4_candidates.log)
  --analysis_only              Skip GPU work; re-run aggregation on --output_dir
  --analysis_subdir NAME       Subdir for analysis outputs (default: analysis)
  --target_temps FLOAT,...     Temperature values to include (default: 0,0.7,1,2,5)

Examples:
  bash scripts/run_exp4_candidates.sh --gpus 0
  bash scripts/run_exp4_candidates.sh --gpus 0,1 --parallel_mode data
  bash scripts/run_exp4_candidates.sh --gpus 0,1 --parallel_mode tensor
USAGE
}

_parse_gpu_list() {
  local raw="${1// /}"
  local -a arr=()
  if [[ "$raw" == *"-"* && "$raw" != *","* ]]; then
    local start end
    IFS='-' read -r start end <<<"$raw"
    for ((i=start; i<=end; i++)); do arr+=("$i"); done
  else
    IFS=',' read -r -a arr <<<"$raw"
  fi
  printf '%s\n' "${arr[@]}"
}

_kill_tree() {
  local sig="$1"
  local pid="$2"
  [[ -z "$pid" ]] && return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  local children
  children=$(ps -o pid= --ppid "$pid" 2>/dev/null | tr -s ' ' '\n' | awk '/^[0-9]+$/')
  for c in $children; do
    _kill_tree "$sig" "$c"
  done
  kill "-$sig" "$pid" 2>/dev/null || true
}

_cleanup_children() {
  local reason="$1"
  if [[ "$CLEANUP_RUNNING" -eq 1 ]]; then
    return 0
  fi
  CLEANUP_RUNNING=1

  echo ""
  echo "Caught ${reason}. Stopping exp4_candidates child processes..."

  for pid in "${CHILD_PIDS[@]:-}"; do
    _kill_tree TERM "$pid"
  done

  sleep 2

  for pid in "${CHILD_PIDS[@]:-}"; do
    _kill_tree KILL "$pid"
  done

  # Fallback: in case children were re-parented.
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "_run_exp4_candidates.py.*--output_dir ${OUTPUT_DIR}" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "_run_exp4_candidates.py.*--output_dir ${OUTPUT_DIR}" 2>/dev/null || true
  fi

  echo "All exp4_candidates child processes stopped."
}

_on_signal() {
  local sig="$1"
  _cleanup_children "$sig"
  exit 130
}

trap '_on_signal INT' INT
trap '_on_signal TERM' TERM
trap '_on_signal QUIT' QUIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --rollout_dir) ROLLOUT_DIR="$2"; shift 2 ;;
    --gpus|--gpu) GPU_LIST="$2"; shift 2 ;;
    --parallel_mode) PARALLEL_MODE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --rollout_cells) ROLLOUT_CELLS="$2"; shift 2 ;;
    --extract_batch_size) EXTRACT_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_batch_size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --max_cliffs_per_dataset) MAX_CLIFFS_PER_DATASET="$2"; shift 2 ;;
    --max_candidates_per_cliff) MAX_CANDIDATES_PER_CLIFF="$2"; shift 2 ;;
    --gpu_mem) GPU_MEM="$2"; shift 2 ;;
    --exp3_runs_dir) EXP3_RUNS_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log_file) LOG_FILE="$2"; shift 2 ;;
    --analysis_only) ANALYSIS_ONLY=1; shift ;;
    --analysis_subdir) ANALYSIS_SUBDIR="$2"; shift 2 ;;
    --target_temps) TARGET_TEMPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$ANALYSIS_ONLY" -eq 1 ]]; then
  if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output_dir required with --analysis_only"
    exit 1
  fi
  if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: $OUTPUT_DIR not found" >&2
    exit 1
  fi
  echo "============================================================"
  echo "RQ2-2 Analysis-only"
  echo "============================================================"
  echo "Input dir:       $OUTPUT_DIR"
  echo "Analysis subdir: $ANALYSIS_SUBDIR"
  echo "Target temps:    $TARGET_TEMPS"
  echo ""
  python3 -m src.analysis.exp4_candidates_aggregator "$OUTPUT_DIR" \
    --analysis_subdir "$ANALYSIS_SUBDIR" \
    --target_temps "$TARGET_TEMPS"
  echo ""
  echo "============================================================"
  echo "Analysis Complete"
  echo "============================================================"
  echo "Output: $OUTPUT_DIR/$ANALYSIS_SUBDIR"
  ls -la "$OUTPUT_DIR/$ANALYSIS_SUBDIR" | sed -n '1,200p'
  exit 0
fi

mapfile -t GPU_IDS < <(_parse_gpu_list "$GPU_LIST")
[[ "${#GPU_IDS[@]}" -eq 0 ]] && GPU_IDS=(0)
GPU_LIST_STR="$(IFS=','; echo "${GPU_IDS[*]}")"
NUM_GPUS="${#GPU_IDS[@]}"

case "$PARALLEL_MODE" in
  auto)
    if [[ "$NUM_GPUS" -gt 1 ]]; then
      PARALLEL_MODE="data"
    else
      PARALLEL_MODE="tensor"
    fi
    ;;
  data|tensor)
    ;;
  *)
    echo "Invalid --parallel_mode: $PARALLEL_MODE (expected auto|data|tensor)"
    exit 1
    ;;
esac

MODEL_SHORT=$(python3 - <<PY
import src.config as config
print(config.get_model_short_name(config.resolve_model_path("$MODEL")))
PY
)

if [[ -z "$OUTPUT_DIR" ]]; then
  TS=$(date +%m%d_%H%M%S)
  OUTPUT_DIR="./output/07_candidate_replacement/${MODEL_SHORT}_${TS}"
fi
mkdir -p "$OUTPUT_DIR"

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$OUTPUT_DIR/run_exp4_candidates.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "RQ2-2: Top-k Cliff Replacement"
echo "============================================================"
echo "Model:            $MODEL ($MODEL_SHORT)"
echo "Datasets:         $DATASETS"
echo "Rollout dir:      $ROLLOUT_DIR"
echo "GPU(s):           $GPU_LIST_STR"
echo "parallel_mode:    $PARALLEL_MODE"
[[ -n "$MODE" ]] && echo "mode:             $MODE"
[[ -n "$TEMPERATURE" ]] && echo "temperature:      $TEMPERATURE"
echo "top_k:            $TOP_K"
echo "num_samples:      $NUM_SAMPLES"
echo "rollout_cells:    $ROLLOUT_CELLS"
echo "Output dir:       $OUTPUT_DIR"
echo "Log file:         $LOG_FILE"
echo ""

_build_common_cmd() {
  local out_dir="$1"
  local gpus="$2"
  local num_shards="$3"
  local shard_index="$4"

  local -a cmd=(
    python3 -u scripts/_run_exp4_candidates.py
    --model "$MODEL"
    --datasets "$DATASETS"
    --rollout_dir "$ROLLOUT_DIR"
    --output_dir "$out_dir"
    --num_samples "$NUM_SAMPLES"
    --top_k "$TOP_K"
    --rollout_cells "$ROLLOUT_CELLS"
    --gpus "$gpus"
    --extract_batch_size "$EXTRACT_BATCH_SIZE"
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
    --max_cliffs_per_dataset "$MAX_CLIFFS_PER_DATASET"
    --max_candidates_per_cliff "$MAX_CANDIDATES_PER_CLIFF"
    --gpu_mem "$GPU_MEM"
    --num_shards "$num_shards"
    --shard_index "$shard_index"
  )

  if [[ -n "$MODE" ]]; then
    cmd+=(--mode "$MODE")
  fi
  if [[ -n "$TEMPERATURE" ]]; then
    cmd+=(--temperature "$TEMPERATURE")
  fi
  if [[ -n "$EXP3_RUNS_DIR" ]]; then
    cmd+=(--exp3_runs_dir "$EXP3_RUNS_DIR")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

if [[ "$PARALLEL_MODE" == "tensor" ]]; then
  TENSOR_SHARD_DIR="$OUTPUT_DIR/shards/shard0"
  mkdir -p "$TENSOR_SHARD_DIR"

  CMD=(
    python3 -u scripts/_run_exp4_candidates.py
    --model "$MODEL"
    --datasets "$DATASETS"
    --rollout_dir "$ROLLOUT_DIR"
    --output_dir "$TENSOR_SHARD_DIR"
    --num_samples "$NUM_SAMPLES"
    --top_k "$TOP_K"
    --rollout_cells "$ROLLOUT_CELLS"
    --gpus "$GPU_LIST_STR"
    --extract_batch_size "$EXTRACT_BATCH_SIZE"
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
    --max_cliffs_per_dataset "$MAX_CLIFFS_PER_DATASET"
    --max_candidates_per_cliff "$MAX_CANDIDATES_PER_CLIFF"
    --gpu_mem "$GPU_MEM"
    --num_shards 1
    --shard_index 0
  )

  if [[ -n "$MODE" ]]; then
    CMD+=(--mode "$MODE")
  fi
  if [[ -n "$TEMPERATURE" ]]; then
    CMD+=(--temperature "$TEMPERATURE")
  fi
  if [[ -n "$EXP3_RUNS_DIR" ]]; then
    CMD+=(--exp3_runs_dir "$EXP3_RUNS_DIR")
  fi

  CUDA_VISIBLE_DEVICES="$GPU_LIST_STR" VLLM_ENABLE_V1_MULTIPROCESSING=0 "${CMD[@]}" &
  TENSOR_PID=$!
  CHILD_PIDS+=("$TENSOR_PID")

  set +e
  wait "$TENSOR_PID"
  TENSOR_RC=$?
  set -e

  if [[ "$TENSOR_RC" -ne 0 ]]; then
    echo "Tensor-mode run failed (PID=$TENSOR_PID, rc=$TENSOR_RC)"
    exit "$TENSOR_RC"
  fi

  MERGE_CMD=(
    python3 -u scripts/_run_exp4_candidates.py
    --output_dir "$OUTPUT_DIR"
    --top_k "$TOP_K"
    --num_samples "$NUM_SAMPLES"
    --merge_shard_dirs "$TENSOR_SHARD_DIR"
  )
  if [[ -n "$EXP3_RUNS_DIR" ]]; then
    MERGE_CMD+=(--exp3_runs_dir "$EXP3_RUNS_DIR")
  fi

  echo "Finalizing tensor-mode output (merge) ..."
  "${MERGE_CMD[@]}"
else
  SHARD_BASE="$OUTPUT_DIR/shards"
  mkdir -p "$SHARD_BASE"

  echo "Launching data-parallel shards: $NUM_GPUS"
  for ((i=0; i<NUM_GPUS; i++)); do
    gpu="${GPU_IDS[$i]}"
    shard_dir="$SHARD_BASE/shard$i"
    shard_log="$OUTPUT_DIR/run_exp4_candidates_shard${i}.log"
    mkdir -p "$shard_dir"
    : > "$shard_log"

    cmd_line="$(_build_common_cmd "$shard_dir" "0" "$NUM_GPUS" "$i")"
    echo "  shard $i -> GPU $gpu"
    echo "    log: $shard_log"

    CUDA_VISIBLE_DEVICES="$gpu" VLLM_ENABLE_V1_MULTIPROCESSING=0 bash -lc "$cmd_line" > "$shard_log" 2>&1 &
    pid=$!
    CHILD_PIDS+=("$pid")
    SHARD_DIRS+=("$shard_dir")
  done

  fail=0
  for i in "${!CHILD_PIDS[@]}"; do
    pid="${CHILD_PIDS[$i]}"
    if ! wait "$pid"; then
      echo "Shard $i failed (PID=$pid). See $OUTPUT_DIR/run_exp4_candidates_shard${i}.log"
      fail=1
    else
      echo "Shard $i completed"
    fi
  done

  if [[ "$fail" -ne 0 ]]; then
    echo "One or more shards failed"
    exit 1
  fi

  SHARD_CSV="$(IFS=','; echo "${SHARD_DIRS[*]}")"
  MERGE_CMD=(
    python3 -u scripts/_run_exp4_candidates.py
    --output_dir "$OUTPUT_DIR"
    --top_k "$TOP_K"
    --num_samples "$NUM_SAMPLES"
    --merge_shard_dirs "$SHARD_CSV"
  )
  if [[ -n "$EXP3_RUNS_DIR" ]]; then
    MERGE_CMD+=(--exp3_runs_dir "$EXP3_RUNS_DIR")
  fi

  echo "Merging shard outputs..."
  "${MERGE_CMD[@]}"
fi

CHILD_PIDS=()

echo ""
echo "============================================================"
echo "Analysis"
echo "============================================================"
echo "Analysis subdir: $ANALYSIS_SUBDIR"
echo "Target temps:    $TARGET_TEMPS"
python3 -m src.analysis.exp4_candidates_aggregator "$OUTPUT_DIR" \
  --analysis_subdir "$ANALYSIS_SUBDIR" \
  --target_temps "$TARGET_TEMPS"

echo ""
echo "============================================================"
echo "RQ2-2 COMPLETE"
echo "============================================================"
echo "Output dir: $OUTPUT_DIR"
echo "Log file:   $LOG_FILE"
ls -la "$OUTPUT_DIR" | sed -n '1,200p'
