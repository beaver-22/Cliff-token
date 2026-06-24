#!/usr/bin/env bash
# Stage 2: Multi-GPU Rollout (Compute Tokenwise Potential)
# Shards paths across GPUs, computes tokenwise potential, merges results.
set -euo pipefail

# Enable vLLM V1 engine (separate scheduler, chunked prefill optimization)
# disable_cascade_attn is set in create_llm() (to avoid A100 FA2 LSE bug)
export VLLM_USE_V1=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="python3"

DATA_PATH=""
DATASET="math500"
GPU_LIST=""
OUTPUT_ROOT=""
ROLLOUT_SAMPLES=""
ROLLOUT_WINDOW=""
MODE=""
MODEL="qwen3-4b"
TEMPERATURE=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_rollout.sh --data_path PATH [options]

Stage 2: Compute tokenwise potential via rollout sampling (multi-GPU).

Options:
  --data_path PATH                Path to merged *_all_paths.json (required)
  --dataset NAME                  Dataset: gsm8k|gsm1k|gsm1k_100|math500|math500_100|aime24|aime25 (default: math500)
  --rollout_samples N             Rollout samples per token position (default: 64)
  --rollout_window N              Compute potential every N tokens (default: 1 = every token)
  --mode thinking|non_thinking    Reasoning mode (auto-detected if omitted)
  --model NAME                    Model alias or HF path (default: qwen3-4b)
  --temperature FLOAT             Override sampling temperature
  --gpus "0,1,2"                  GPU IDs for parallel shards (default: all available)
  --output_dir PATH               Output root directory

Examples:
  scripts/run_rollout.sh --model qwen3-4b --dataset math500 \
      --data_path ./output/01_inference/Qwen3-4B/math500_100_all_paths.json --gpus 0
  scripts/run_rollout.sh --model gemma-3-4b --dataset gsm8k \
      --data_path ./output/01_inference/gemma-3-4b-it/gsm8k_all_paths.json --rollout_samples 64 --gpus 0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)      DATA_PATH="$2";       shift 2 ;;
    --dataset)        DATASET="$2";         shift 2 ;;
    --rollout_samples) ROLLOUT_SAMPLES="$2"; shift 2 ;;
    --rollout_window) ROLLOUT_WINDOW="$2";  shift 2 ;;
    --mode)           MODE="$2";            shift 2 ;;
    --model)          MODEL="$2";           shift 2 ;;
    --temperature)    TEMPERATURE="$2";     shift 2 ;;
    --gpus)           GPU_LIST="$2";        shift 2 ;;
    --output_dir)     OUTPUT_ROOT="$2";     shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$DATA_PATH" ]]; then
  echo "Error: --data_path is required"; usage; exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Error: Data file not found: $DATA_PATH"; exit 1
fi

RUN_TAG="$(date +%m%d_%H%M)"

MODEL_SHORT=$(cd "$ROOT" && $PYTHON -c "
from src.config import resolve_model_path, get_model_short_name
print(get_model_short_name(resolve_model_path('$MODEL')))
" 2>/dev/null || echo "$MODEL")

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$ROOT/output/03_rollout/$MODEL_SHORT"
fi

# ── GPU list parsing ─────────────────────────────────────────────────────────
parse_gpu_list() {
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

GPU_IDS=()
if [[ -n "$GPU_LIST" ]]; then
  mapfile -t GPU_IDS < <(parse_gpu_list "$GPU_LIST")
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mapfile -t GPU_IDS < <(parse_gpu_list "${CUDA_VISIBLE_DEVICES}")
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("$i"); done
  fi
fi
[[ "${#GPU_IDS[@]}" -eq 0 ]] && GPU_IDS=(0)

GPU_LIST_STR="$(IFS=','; echo "${GPU_IDS[*]}")"
NUM_SHARDS="${#GPU_IDS[@]}"

LOG_DIR="$OUTPUT_ROOT/logs/${DATASET}"
SHARD_OUT_BASE="$OUTPUT_ROOT/shards/${DATASET}"
mkdir -p "$LOG_DIR" "$SHARD_OUT_BASE"

echo "=============================================="
echo "STAGE 2: ROLLOUT (Multi-GPU)"
echo "=============================================="
echo "Model:   $MODEL ($MODEL_SHORT)"
echo "Data:    $DATA_PATH"
echo "Dataset: $DATASET"
echo "GPUs:    $GPU_LIST_STR ($NUM_SHARDS shards)"
echo "Output:  $OUTPUT_ROOT"
[[ -n "$ROLLOUT_SAMPLES" ]] && echo "Rollout samples: $ROLLOUT_SAMPLES"
[[ -n "$ROLLOUT_WINDOW" ]]  && echo "Rollout window:  $ROLLOUT_WINDOW"
[[ -n "$TEMPERATURE" ]]     && echo "Temperature: $TEMPERATURE"
echo ""

# ── Split paths into shards ──────────────────────────────────────────────────
echo "Splitting paths into $NUM_SHARDS shards..."
SHARD_META="$LOG_DIR/rollout_shards.meta"
$PYTHON - "$DATA_PATH" "$SHARD_OUT_BASE" "$NUM_SHARDS" <<'PY' > "$SHARD_META"
import json, sys
from pathlib import Path

data_path   = sys.argv[1]
shard_base  = Path(sys.argv[2])
num_shards  = int(sys.argv[3])

all_paths = json.load(open(data_path))
total = len(all_paths)

# Token-balanced greedy assignment:
# Sort paths by token count descending, assign each to the shard with fewest total tokens
indexed = [(i, len(p.get("response_token_ids", []))) for i, p in enumerate(all_paths)]
indexed.sort(key=lambda x: -x[1])

buckets = [[] for _ in range(num_shards)]
bucket_tokens = [0] * num_shards

for orig_idx, tok_len in indexed:
    lightest = min(range(num_shards), key=lambda s: bucket_tokens[s])
    buckets[lightest].append(orig_idx)
    bucket_tokens[lightest] += tok_len

for i in range(num_shards):
    shard_paths = [all_paths[idx] for idx in sorted(buckets[i])]
    if not shard_paths:
        continue
    shard_dir = shard_base / f"shard{i}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_file = shard_dir / "input_paths.json"
    json.dump(shard_paths, open(shard_file, "w"), ensure_ascii=False)
    print(f"{shard_file}\t{len(shard_paths)}", file=sys.stdout)
    print(f"  Shard {i}: {len(shard_paths)} paths, {bucket_tokens[i]:,} tokens", file=sys.stderr)

print(f"Total paths: {total}", file=sys.stderr)
PY

mapfile -t SHARD_ROWS < "$SHARD_META"
if [[ "${#SHARD_ROWS[@]}" -eq 0 ]]; then
  echo "No paths to process."; exit 1
fi

# ── Launch parallel rollout jobs ─────────────────────────────────────────────
echo "Starting $NUM_SHARDS parallel rollout jobs..."
declare -a PIDS=()
declare -a SHARD_OUTS=()

idx=0
for row in "${SHARD_ROWS[@]}"; do
  shard_file="${row%%$'\t'*}"
  shard_count="${row#*$'\t'}"
  gpu="${GPU_IDS[$idx]}"
  out_dir="$SHARD_OUT_BASE/shard${idx}/output"
  mkdir -p "$out_dir"
  log_file="$LOG_DIR/rollout_shard$idx.log"

  echo "  Shard $idx -> GPU $gpu | $shard_count paths"
  SHARD_OUTS+=("$out_dir")

  CMD_ARGS=(
    rollout
    --model     "$MODEL"
    --data_path "$shard_file"
    --dataset   "$DATASET"
    --gpus 0
    --output_dir "$out_dir"
  )
  [[ -n "$ROLLOUT_SAMPLES" ]] && CMD_ARGS+=(--rollout_samples "$ROLLOUT_SAMPLES")
  [[ -n "$ROLLOUT_WINDOW" ]]  && CMD_ARGS+=(--rollout_window  "$ROLLOUT_WINDOW")
  [[ -n "$MODE" ]]            && CMD_ARGS+=(--mode "$MODE")
  [[ -n "$TEMPERATURE" ]]     && CMD_ARGS+=(--temperature "$TEMPERATURE")

  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" $PYTHON -m src.cli "${CMD_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$log_file" 2>&1 &
  PIDS+=("$!")
  idx=$((idx+1))
done

# ── Cleanup on Ctrl+C / SIGTERM ──────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Caught signal, killing all shard processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  echo "All shards killed."
  exit 1
}
trap cleanup SIGINT SIGTERM

# ── Wait with hang detection ─────────────────────────────────────────────────
HANG_GRACE=60

wait_or_kill() {
  local pid=$1 log_file=$2 idx=$3 marker=$4
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "$marker" "$log_file" 2>/dev/null; then
      local waited=0
      while kill -0 "$pid" 2>/dev/null && (( waited < HANG_GRACE )); do
        sleep 5; waited=$((waited + 5))
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "  Shard $idx: hanging after completion, killing (PID $pid)..."
        kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null || true
      fi
      break
    fi
    sleep 10
  done
  wait "$pid" 2>/dev/null || true
}

echo ""
echo "Waiting for all rollout shards... (logs: $LOG_DIR)"
fail=0
for i in "${!PIDS[@]}"; do
  log_file="$LOG_DIR/rollout_shard$i.log"
  wait_or_kill "${PIDS[$i]}" "$log_file" "$i" "ROLLOUT COMPLETE"
  if ! grep -q "ROLLOUT COMPLETE" "$log_file" 2>/dev/null; then
    echo "  Shard $i FAILED (see $log_file)"; fail=1
  else
    echo "  Shard $i completed"
  fi
done

[[ "$fail" -ne 0 ]] && { echo "One or more rollout shards failed!"; exit 1; }

# ── Merge rollout results ────────────────────────────────────────────────────
MERGED_DIR="$OUTPUT_ROOT"
mkdir -p "$MERGED_DIR"

echo ""
echo "Merging rollout results..."
$PYTHON - "$MERGED_DIR" "$DATASET" "${SHARD_OUTS[@]}" <<'PY'
import json, sys
from pathlib import Path

merged_dir = Path(sys.argv[1])
dataset    = sys.argv[2]
shard_dirs = sys.argv[3:]

all_paths = []; success_paths = []; failure_paths = []

for shard_dir in shard_dirs:
    for f in Path(shard_dir).glob("*_all_paths.json"):
        paths = json.load(open(f))
        all_paths.extend(paths)
        for p in paths:
            (success_paths if p.get("is_correct") else failure_paths).append(p)

json.dump(all_paths,     open(merged_dir / f"{dataset}_all_paths.json",     "w"), ensure_ascii=False)
json.dump(success_paths, open(merged_dir / f"{dataset}_success_paths.json", "w"), ensure_ascii=False)
json.dump(failure_paths, open(merged_dir / f"{dataset}_failure_paths.json", "w"), ensure_ascii=False)

print(f"Total: {len(all_paths)} paths")
print(f"  Success: {len(success_paths)}")
print(f"  Failure: {len(failure_paths)}")
print(f"Saved to: {merged_dir}")
PY

echo ""
echo "=============================================="
echo "ROLLOUT COMPLETE"
echo "=============================================="
echo "Output: $MERGED_DIR"
echo "Files:"
echo "  ${DATASET}_all_paths.json  (with tokenwise potential scores)"
echo "  ${DATASET}_success_paths.json"
echo "  ${DATASET}_failure_paths.json"
echo ""
echo "Next step:"
echo "  python -m src.cli experiment --experiment rq1_1 --model $MODEL --dataset $DATASET \\"
echo "      --data_path $MERGED_DIR/${DATASET}_all_paths.json"
