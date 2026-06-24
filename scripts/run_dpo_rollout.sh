#!/bin/bash
# Cliff-DPO Step 1 Candidate Rollout Wrapper (chunked, resumable)
#
# Usage:
#   bash scripts/run_dpo_rollout.sh --model qwen3-0.6b --dataset gsm8k \
#       --data_path ./output/03_rollout/Qwen3-0.6B/gsm8k_all_paths.json \
#       --gpus 0 --chunk_size 10

set -euo pipefail

# Defaults
MODEL=""
DATASET=""
DATA_PATH=""
OUTPUT_DIR=""
GPUS="0"
MODE="non_thinking"
K_CANDIDATES=10
NUM_SAMPLES=64
CHUNK_SIZE=10
FORCE=false
LOG_DIR="./output/09_cliff_dpo/logs"
LOG_LEVEL="INFO"

usage() {
    cat <<'USAGE'
Usage: scripts/run_dpo_rollout.sh [options]

Required:
  --model NAME                 Model alias or path
  --dataset NAME               Dataset name (e.g. gsm8k)
  --data_path PATH             Rollout Stage2 output (*_all_paths.json)

Options:
  --output_dir PATH            Output dir (default: ./output/09_cliff_dpo/01_candidates/{model_short})
  --gpus "0,1"              Comma-separated GPU IDs (default: 0)
  --mode MODE                  thinking|non_thinking (default: non_thinking)
  --k_candidates N             Top-k candidate count (default: 10)
  --num_samples N              Rollout samples per candidate (default: 64)
  --chunk_size N               Cliffs per save chunk (default: 10)
  --force                      Reset existing phase files before running
  --log_dir PATH               Log directory (default: ./output/09_cliff_dpo/logs)
  --log_level LVL              DEBUG|INFO|WARNING|ERROR (default: INFO)

Backward-compatible aliases:
  --k N                        Same as --k_candidates
  --samples N                  Same as --num_samples
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2;;
        --dataset)    DATASET="$2"; shift 2;;
        --data_path)  DATA_PATH="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --gpus)       GPUS="$2"; shift 2;;
        --mode)       MODE="$2"; shift 2;;
        --k|--k_candidates)      K_CANDIDATES="$2"; shift 2;;
        --samples|--num_samples) NUM_SAMPLES="$2"; shift 2;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2;;
        --force)      FORCE=true; shift;;
        --log_dir)    LOG_DIR="$2"; shift 2;;
        --log_level)  LOG_LEVEL="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "Unknown arg: $1"; usage; exit 1;;
    esac
done

if [[ -z "$MODEL" || -z "$DATASET" || -z "$DATA_PATH" ]]; then
    echo "Required: --model, --dataset, --data_path"
    usage
    exit 1
fi

# Canonicalize model path/name to avoid case drift in output directories.
MODEL_PATH=$(python3 -c "import src.config as config; print(config.resolve_model_path('$MODEL'))")
MODEL_SHORT=$(python3 -c "import src.config as config; print(config.get_model_short_name(config.resolve_model_path('$MODEL')))")

# Leave OUTPUT_DIR empty -> canonical default path.
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./output/09_cliff_dpo/01_candidates/${MODEL_SHORT}"
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "=========================================="
echo "Cliff-DPO Candidate Rollout (Step 1)"
echo "  Model:          $MODEL"
echo "  Resolved Model: $MODEL_PATH"
echo "  Model Short:    $MODEL_SHORT"
echo "  Dataset:        $DATASET"
echo "  Data:           $DATA_PATH"
echo "  Output:         $OUTPUT_DIR"
echo "  Logs:           $LOG_DIR"
echo "  GPUs:           $GPUS"
echo "  Candidates:     $K_CANDIDATES"
echo "  Samples:        $NUM_SAMPLES"
echo "  ChunkSize:      $CHUNK_SIZE"
echo "  Force:          $FORCE"
echo "=========================================="

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}

BASE_CMD=(
    python -m src.dpo.vllm_rollout
    --model "$MODEL_PATH"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --mode "$MODE"
    --k_candidates "$K_CANDIDATES"
    --num_samples "$NUM_SAMPLES"
    --chunk_size "$CHUNK_SIZE"
    --log_dir "$LOG_DIR"
    --log_level "$LOG_LEVEL"
)

if $FORCE; then
    BASE_CMD+=(--force)
fi

if [[ $NUM_SHARDS -le 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "${BASE_CMD[@]}" --gpus 0
    echo "Done."
    exit 0
fi

echo "Data-parallel mode: $NUM_SHARDS shards across GPUs ${GPU_ARR[*]}"
pids=()
for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    shard_log="$LOG_DIR/${DATASET}_shard${i}of${NUM_SHARDS}.stdout.log"
    echo "  -> shard $i on GPU $gpu  (log: $shard_log)"
    CUDA_VISIBLE_DEVICES="$gpu" "${BASE_CMD[@]}" \
        --gpus 0 \
        --shard_id "$i" \
        --num_shards "$NUM_SHARDS" \
        > "$shard_log" 2>&1 &
    pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Shard $i (pid ${pids[$i]}) FAILED — see $LOG_DIR/${DATASET}_shard${i}of${NUM_SHARDS}.stdout.log"
        fail=1
    else
        echo "Shard $i (pid ${pids[$i]}) ok"
    fi
done

if [[ $fail -ne 0 ]]; then
    echo "One or more shards failed; skipping merge. Re-run the same command to resume."
    exit 1
fi

echo "All shards complete. Merging..."
python -m src.dpo.vllm_rollout \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_SHARDS" \
    --log_dir "$LOG_DIR" \
    --log_level "$LOG_LEVEL" \
    --merge_only

echo "Done."
