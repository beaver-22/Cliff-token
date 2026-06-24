#!/usr/bin/env bash
# Cliff-DPO Step 6: Cliff Token Count Evaluation Wrapper
#
# Counts cliff tokens (deterministic / uncertain / sampled_off taxonomy) on a test set, comparing the
# base model and (optionally) one or more LoRA adapters trained via cliff-DPO.
#
# Usage:
#   # Smoke test: base only, 2 problems
#   bash scripts/run_dpo_cliff_num_eval.sh --num_problems 2 --gpus 0
#
#   # Baseline vs single trained adapter
#   bash scripts/run_dpo_cliff_num_eval.sh \
#       --adapter ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all/ \
#       --label Cliff-all --gpus 0
#
#   # Baseline vs multiple Cliff-DPO variants (comma-separated)
#   bash scripts/run_dpo_cliff_num_eval.sh \
#       --adapters ./.../cliff_all/,./.../cliff_deterministic_only/,./.../cliff_uncertainty_only/,./.../cliff_sampled_off_only/,./.../cliff_uncertainty_sampled_off_only/ \
#       --labels "Cliff-all,Cliff-deterministic,Cliff-uncertainty,Cliff-sampled-off,Cliff-uncertainty-sampled-off" --gpus 0

set -euo pipefail
export VLLM_USE_V1=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODEL="qwen3-0.6b"
DATASETS="gsm8k"
GPUS="0"
NUM_PROBLEMS=""
PATHS_PER_PROBLEM=""
ADAPTER=""
LABEL=""
ADAPTERS=""
LABELS=""
OUTPUT_DIR=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_dpo_cliff_num_eval.sh [options]

Options:
  --model NAME           Base model alias or path (default: qwen3-0.6b)
  --datasets "ds1 ds2"   Test sets, space-separated (default: gsm8k)
  --gpus "0,1"           GPU IDs (default: 0)
  --num_problems N       Limit test-set problems (default: full set)
  --paths_per_problem N  Reasoning paths per problem (default: 1)
  --adapter PATH         Single trained LoRA adapter (compares vs baseline)
  --label NAME           Label for --adapter (default: adapter dir name)
  --adapters "a,b,c"     Comma-separated adapter paths
  --labels  "x,y,z"      Comma-separated labels matching --adapters
  --output_dir PATH      Output directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2;;
    --datasets)      DATASETS="$2"; shift 2;;
    --gpus)          GPUS="$2"; shift 2;;
    --num_problems)  NUM_PROBLEMS="$2"; shift 2;;
    --paths_per_problem) PATHS_PER_PROBLEM="$2"; shift 2;;
    --adapter)       ADAPTER="$2"; shift 2;;
    --label)         LABEL="$2"; shift 2;;
    --adapters)      ADAPTERS="$2"; shift 2;;
    --labels)        LABELS="$2"; shift 2;;
    --output_dir)    OUTPUT_DIR="$2"; shift 2;;
    -h|--help)       usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

CMD=(python -m src.dpo.step6_cliff_num_eval --model "$MODEL" --gpus "$GPUS")

# datasets is space-separated already
read -ra DS_ARR <<< "$DATASETS"
CMD+=(--datasets "${DS_ARR[@]}")

if [[ -n "$NUM_PROBLEMS" ]]; then
  CMD+=(--num_problems "$NUM_PROBLEMS")
fi

if [[ -n "$PATHS_PER_PROBLEM" ]]; then
  CMD+=(--paths_per_problem "$PATHS_PER_PROBLEM")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output_dir "$OUTPUT_DIR")
fi

# Validate mutually-exclusive / dependent flags
if [[ -n "$ADAPTER" && -n "$ADAPTERS" ]]; then
  echo "Error: pass either --adapter or --adapters, not both." >&2
  exit 1
fi
if [[ -n "$LABELS" && -z "$ADAPTERS" ]]; then
  echo "Error: --labels only applies with --adapters (use --label for --adapter)." >&2
  exit 1
fi

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# Comparison-mode arg construction
if [[ -n "$ADAPTERS" ]]; then
  IFS=',' read -ra A_RAW <<< "$ADAPTERS"
  A_ARR=()
  for a in "${A_RAW[@]}"; do
    t="$(trim "$a")"
    [[ -n "$t" ]] && A_ARR+=("$t")
  done
  if [[ "${#A_ARR[@]}" -eq 0 ]]; then
    echo "Error: --adapters is empty after trimming." >&2
    exit 1
  fi
  CMD+=(--adapter_paths none "${A_ARR[@]}")
  if [[ -n "$LABELS" ]]; then
    IFS=',' read -ra L_RAW <<< "$LABELS"
    L_ARR=()
    for l in "${L_RAW[@]}"; do
      t="$(trim "$l")"
      [[ -n "$t" ]] && L_ARR+=("$t")
    done
    if [[ "${#L_ARR[@]}" -ne "${#A_ARR[@]}" ]]; then
      echo "Error: --labels count (${#L_ARR[@]}) must match --adapters count (${#A_ARR[@]})." >&2
      exit 1
    fi
    CMD+=(--labels Baseline "${L_ARR[@]}")
  fi
elif [[ -n "$ADAPTER" ]]; then
  CMD+=(--adapter_paths none "$ADAPTER")
  LABEL_NAME="${LABEL:-$(basename "${ADAPTER%/}")}"
  CMD+=(--labels Baseline "$LABEL_NAME")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
