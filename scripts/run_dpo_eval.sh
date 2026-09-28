#!/bin/bash
# Cliff-DPO evaluation on GSM8K (test), GSM1K, MATH500 (greedy accuracy) and
# AIME 2025 (avg@64, model sampling config), with the paper token profile.
#
#   Base model:     evaluated --base_runs times, one process per run
#                   -> <output_dir>/base_run<i>/
#   Cliff-DPO:      every <training_dir>/<dataset>/cliff_<variant>_only<SFX>_seed<seed>
#                   adapter evaluated once, in one comparison run
#                   -> <output_dir>/cliff_dpo<SFX>/<label>/
#
# Usage:
#   bash scripts/run_dpo_eval.sh --model ./model/Qwen3-0.6B --dataset gsm8k_train --gpus 0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_dpo_eval.sh [options]

  --model PATH              base model (required)
  --dataset NAME            training dataset name used in run_dpo_train.sh (default: gsm8k_train)
  --training_dir PATH       default: ./output/07_cliff_dpo/03_training/<model>
  --output_dir PATH         default: ./output/07_cliff_dpo/04_eval/<model>
  --variants LIST           default: deterministic,uncertainty,sampled_off
  --variant_suffix SFX      e.g. _token_matched
  --seeds LIST              default: 42,43,44
  --base_runs N             repeated base-model evaluations (default: 3; 0 = skip)
  --gpus "0"                GPU ids (multiple GPUs -> data-parallel shards)
  -h, --help

Any other options are forwarded to `python -m src.dpo.evaluate`.
EOF
}

MODEL=""
DATASET="gsm8k_train"
TRAINING_DIR=""
OUTPUT_DIR=""
VARIANTS="deterministic,uncertainty,sampled_off"
VARIANT_SUFFIX=""
SEEDS="42,43,44"
BASE_RUNS=3
GPU_LIST="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)        usage; exit 0;;
        --model)          MODEL="$2"; shift 2;;
        --dataset)        DATASET="$2"; shift 2;;
        --training_dir)   TRAINING_DIR="$2"; shift 2;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2;;
        --variants)       VARIANTS="$2"; shift 2;;
        --variant_suffix) VARIANT_SUFFIX="$2"; shift 2;;
        --seeds)          SEEDS="$2"; shift 2;;
        --base_runs)      BASE_RUNS="$2"; shift 2;;
        --gpus)           GPU_LIST="$2"; shift 2;;
        *) EXTRA_ARGS+=("$1"); shift;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Required: --model"
    usage
    exit 1
fi

MODEL_PATH=$(python3 -c "import src.config as config; print(config.resolve_model_path('$MODEL'))")
MODEL_SHORT=$(python3 -c "import src.config as config; print(config.get_model_short_name(config.resolve_model_path('$MODEL')))")
TRAINING_DIR="${TRAINING_DIR:-./output/07_cliff_dpo/03_training/${MODEL_SHORT}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/07_cliff_dpo/04_eval/${MODEL_SHORT}}"

COMMON_ARGS=(
    --model "$MODEL_PATH"
    --full_suite
    --token_profile paper
    --aime_samples 64
    --gpus "$GPU_LIST"
    --wandb_mode disabled
)

# ---- Base model: independent repeated runs ----
for ((run = 1; run <= BASE_RUNS; run++)); do
    out="$OUTPUT_DIR/base_run${run}"
    echo ""
    echo "=========================================="
    echo "Base model run $run/$BASE_RUNS -> $out"
    echo "=========================================="
    python -m src.dpo.evaluate "${COMMON_ARGS[@]}" \
        --output_dir "$out" --log_dir "$out/logs" "${EXTRA_ARGS[@]}"
done

# ---- Cliff-DPO adapters: one evaluation per training seed ----
IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
ADAPTERS=()
LABELS=()
for seed in "${SEED_ARR[@]}"; do
    for variant in "${VARIANT_ARR[@]}"; do
        stem="cliff_${variant}_only${VARIANT_SUFFIX}"
        adapter="$TRAINING_DIR/$DATASET/${stem}_seed${seed}"
        if [[ ! -f "$adapter/adapter_config.json" ]]; then
            echo "  SKIP $stem seed $seed: adapter not found at $adapter"
            continue
        fi
        ADAPTERS+=("$adapter")
        LABELS+=("${stem}_seed${seed}")
    done
done

if [[ ${#ADAPTERS[@]} -eq 0 ]]; then
    echo "No Cliff-DPO adapters found under $TRAINING_DIR/$DATASET"
    exit 0
fi

out="$OUTPUT_DIR/cliff_dpo${VARIANT_SUFFIX}"
echo ""
echo "=========================================="
echo "Cliff-DPO adapters (${#ADAPTERS[@]}) -> $out"
echo "=========================================="
python -m src.dpo.evaluate "${COMMON_ARGS[@]}" \
    --adapter_paths "${ADAPTERS[@]}" \
    --labels "${LABELS[@]}" \
    --output_dir "$out" --log_dir "$out/logs" "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results: $OUTPUT_DIR"
