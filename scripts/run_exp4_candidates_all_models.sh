#!/usr/bin/env bash
# Run exp4_candidates + analysis sequentially for multiple models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0}"
PARALLEL_MODE="${PARALLEL_MODE:-auto}"
MODELS_CSV=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp4_candidates_all_models.sh [options]

Options:
  --gpus "0,1,2,3"                 GPU IDs forwarded to run_exp4_candidates.sh
  --parallel_mode auto|data|tensor Parallel mode (default: auto)
  --models "m1,m2,..."             Model aliases
                                   (default: all 7 paper models)

Example:
  bash scripts/run_exp4_candidates_all_models.sh --gpus 0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --parallel_mode) PARALLEL_MODE="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$MODELS_CSV" ]]; then
  IFS=',' read -ra MODELS <<< "$MODELS_CSV"
else
  MODELS=(
    "qwen3-0.6b"
    "qwen3-4b"
    "qwen3-8b"
    "llama-3.2-1b"
    "llama-3.2-3b"
    "llama-3.1-8b"
    "gemma-3-4b"
  )
fi

for MODEL in "${MODELS[@]}"; do
  TS=$(date +%m%d_%H%M%S)
  MODEL_TAG="${MODEL//\//_}"
  OUT="./output/07_candidate_replacement/${MODEL_TAG}_${TS}"

  echo "============================================================"
  echo "[$(date '+%F %T')] START $MODEL -> $OUT"
  echo "============================================================"

  bash scripts/run_exp4_candidates.sh \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --parallel_mode "$PARALLEL_MODE" \
    --output_dir "$OUT"

  # Re-run analysis explicitly so this wrapper can refresh tables/plots even if
  # users rerun after manual edits to merged CSV files.
  bash scripts/run_exp4_candidates.sh --analysis_only --output_dir "$OUT"

  echo "[$(date '+%F %T')] DONE  $MODEL"
done

echo ""
echo "All models completed."
