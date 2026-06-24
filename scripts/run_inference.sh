#!/usr/bin/env bash
# Stage 1: Inference (Generate Reasoning Paths)
# Thin wrapper around src.inference for pipeline compatibility.
# Supports multiple models and datasets in a single call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (comma-separated for multiple values)
MODELS=""
DATASETS=""
GPU_LIST=""
OUTPUT_ROOT=""
PAPER_MODELS="qwen3-0.6b qwen3-4b qwen3-8b llama-3.2-1b llama-3.2-3b llama-3.1-8b gemma-3-4b"
MODE=""
TEMPERATURE=""
MAX_TOKENS=""
NUM_PROBLEMS=""
WITH_LOGPROBS=0
LOGPROBS_GPU=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_inference.sh [options]

Stage 1: Generate reasoning paths.
Supports multiple models (one per GPU) and multiple datasets.

Options:
  --model MODEL[,MODEL,...]       Model aliases (default: all 7 paper models)
                                   e.g. qwen3-4b  or  qwen3-0.6b,qwen3-4b,qwen3-8b,llama-3.2-1b,llama-3.2-3b,llama-3.1-8b,gemma-3-4b
  --dataset DS[,DS,...]           Datasets (default: gsm1k_100,math500_100,aime25)
                                   e.g. math500_100  or  gsm1k_100,math500_100,aime25
  --mode thinking|non_thinking    Reasoning mode (auto-detected if omitted)
  --temperature FLOAT             Override sampling temperature
  --max_tokens N                  Override max_tokens (default: use config)
  --num_problems N                Limit problems per dataset for smoke tests
  --gpus "0,1,2,3"               GPU IDs (default: all available)
  --output_dir PATH               Output root directory (default: output/01_inference)
  --with_logprobs                 After inference, compute per-token rank/logprob/entropy
                                   stats for each path and save to output/02_token_stats/.
                                   Required for the full exp3_entropy baseline comparison.
  --logprobs_gpu GPU              GPU to use for logprob computation (default: first GPU from --gpus)

Examples:
  # Paper models × all datasets
  scripts/run_inference.sh --gpus 0

  # Single model × single dataset
  scripts/run_inference.sh --model qwen3-4b --dataset math500_100 --gpus 0

  # With per-token logprob/entropy stats (needed for exp3_entropy full output)
  scripts/run_inference.sh --gpus 0 --with_logprobs
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODELS="$2";       shift 2 ;;
    --dataset)      DATASETS="$2";     shift 2 ;;
    --mode)         MODE="$2";         shift 2 ;;
    --temperature)  TEMPERATURE="$2";  shift 2 ;;
    --max_tokens)   MAX_TOKENS="$2";   shift 2 ;;
    --num_problems) NUM_PROBLEMS="$2"; shift 2 ;;
    --gpus)         GPU_LIST="$2";     shift 2 ;;
    --output_dir)   OUTPUT_ROOT="$2";  shift 2 ;;
    --with_logprobs) WITH_LOGPROBS=1;  shift ;;
    --logprobs_gpu) LOGPROBS_GPU="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="output/01_inference"
fi

# ── GPU list: comma-sep → space-sep ──────────────────────────────────────────
GPU_IDS_SPACE=""
if [[ -n "$GPU_LIST" ]]; then
  GPU_IDS_SPACE="${GPU_LIST//,/ }"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_IDS_SPACE="${CUDA_VISIBLE_DEVICES//,/ }"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    for ((i=0; i<GPU_COUNT; i++)); do
      GPU_IDS_SPACE+="$i "
    done
  fi
fi
[[ -z "$GPU_IDS_SPACE" ]] && GPU_IDS_SPACE="0"

# ── Model/dataset: comma-sep → space-sep ─────────────────────────────────────
MODELS_SPACE="${MODELS//,/ }"
DATASETS_SPACE="${DATASETS//,/ }"

echo "=============================================="
echo "STAGE 1: INFERENCE"
echo "=============================================="
echo "Models:  ${MODELS_SPACE:-all ($PAPER_MODELS)}"
echo "Datasets:${DATASETS_SPACE:-all (gsm1k_100 math500_100 aime25)}"
echo "GPUs:    $GPU_IDS_SPACE"
echo "Output:  $OUTPUT_ROOT"
[[ -n "$TEMPERATURE" ]]  && echo "Temperature: $TEMPERATURE"
[[ -n "$MODE" ]]         && echo "Mode: $MODE"
[[ -n "$MAX_TOKENS" ]]   && echo "Max tokens: $MAX_TOKENS"
[[ -n "$NUM_PROBLEMS" ]] && echo "Num problems: $NUM_PROBLEMS"
echo ""

# ── Build args for src.inference ─────────────────────────────────────────────
PY_ARGS=()

# shellcheck disable=SC2086
[[ -n "$MODELS_SPACE" ]]   && PY_ARGS+=(--model $MODELS_SPACE)
# shellcheck disable=SC2086
[[ -n "$DATASETS_SPACE" ]] && PY_ARGS+=(--dataset $DATASETS_SPACE)
# shellcheck disable=SC2086
PY_ARGS+=(--gpus $GPU_IDS_SPACE)

PY_ARGS+=(--output_dir "$OUTPUT_ROOT")
[[ -n "$TEMPERATURE" ]]  && PY_ARGS+=(--temperature "$TEMPERATURE")
[[ -n "$MAX_TOKENS" ]]   && PY_ARGS+=(--max_tokens "$MAX_TOKENS")
[[ -n "$NUM_PROBLEMS" ]] && PY_ARGS+=(--num_problems "$NUM_PROBLEMS")
[[ -n "$MODE" && "$MODE" == "thinking" ]] && PY_ARGS+=(--thinking)

cd "$ROOT"
python -m src.inference "${PY_ARGS[@]}"

echo ""
echo "=============================================="
echo "INFERENCE COMPLETE"
echo "=============================================="

# ── Optional: per-token logprob/rank/entropy stats ────────────────────────────
# Required for exp3_entropy baseline comparison plots.
# Runs sequentially on one GPU (one model at a time); can take 30-60 min total.
if [[ "$WITH_LOGPROBS" -eq 1 ]]; then
  FIRST_GPU="${GPU_IDS_SPACE%% *}"
  LP_GPU="${LOGPROBS_GPU:-$FIRST_GPU}"
  LP_SOURCE="${OUTPUT_ROOT:-output/01_inference}"

  echo ""
  echo "=============================================="
  echo "STAGE 1b: PER-TOKEN STATS (logprobs/rank/entropy)"
  echo "=============================================="
  echo "GPU:    $LP_GPU"
  echo "Source: $LP_SOURCE"
  echo "Output: output/02_token_stats"
  echo ""

  LP_ARGS=(
    python3 scripts/_compute_token_stats.py
    --gpu "$LP_GPU"
    --source "$LP_SOURCE"
    --skip-existing
  )
  [[ -n "$DATASETS_SPACE" ]] && LP_ARGS+=(--datasets "${DATASETS_SPACE// /,}")
  [[ -n "$MODELS_SPACE" ]]   && LP_ARGS+=(--models   "${MODELS_SPACE// /,}")

  VLLM_ENABLE_V1_MULTIPROCESSING=0 "${LP_ARGS[@]}"

  echo ""
  echo "=============================================="
  echo "PER-TOKEN STATS COMPLETE"
  echo "=============================================="
  echo "Output: output/02_token_stats"
fi
