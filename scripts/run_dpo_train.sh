#!/bin/bash
# Cliff-DPO training: cliff-type variants x seeds.
#
# Usage:
#   # Single run
#   bash scripts/run_dpo_train.sh --model ./model/Qwen3-0.6B \
#       --dataset_path ./output/07_cliff_dpo/02_pairs/Qwen3-0.6B/cliff_sampled_off_only_gsm8k_train.json \
#       --output_dir ./output/07_cliff_dpo/03_training/Qwen3-0.6B/gsm8k_train/cliff_sampled_off_only_seed42 \
#       --lr 5e-6 --seed 42
#
#   # Suite: deterministic / uncertainty / sampled-off variants x seeds 42,43,44
#   bash scripts/run_dpo_train.sh --suite --model ./model/Qwen3-0.6B \
#       --dataset gsm8k_train --lr 5e-6 --gpus 0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_dpo_train.sh [options]

Single run mode:
  --model PATH
  --dataset_path PATH
  --output_dir PATH

Suite mode:
  --suite
  --model PATH
  --dataset NAME            pair filename suffix (e.g. gsm8k_train)
  [--pairs_dir PATH]        default: ./output/07_cliff_dpo/02_pairs/<model>
  [--training_dir PATH]     default: ./output/07_cliff_dpo/03_training/<model>
  [--variants LIST]         default: deterministic,uncertainty,sampled_off
  [--variant_suffix SFX]    e.g. _token_matched (reads cliff_<v>_only<SFX>_<dataset>.json)
  [--seeds LIST]            default: 42,43,44
  Each run reads  <pairs_dir>/cliff_<variant>_only<SFX>_<dataset>.json
  and writes      <training_dir>/<dataset>/cliff_<variant>_only<SFX>_seed<seed>

Common options:
  --gpus "0"
  --wandb_project NAME
  --wandb_entity NAME
  --wandb_tags "tag1,tag2"
  --wandb_mode online|offline|disabled
  -h, --help

Any other options (e.g. --lr 5e-6, --seed 42) are forwarded to
`python -m src.dpo.train_dpo`.
EOF
}

# Defaults
MODEL=""
DATASET=""
DATASET_PATH=""
OUTPUT_DIR=""
SUITE=false
PAIRS_DIR=""
TRAINING_DIR=""
VARIANTS="deterministic,uncertainty,sampled_off"
VARIANT_SUFFIX=""
SEEDS="42,43,44"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_TAGS=""
WANDB_MODE="disabled"
GPU_LIST=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)        usage; exit 0;;
        --model)          MODEL="$2"; shift 2;;
        --dataset)        DATASET="$2"; shift 2;;
        --dataset_path)   DATASET_PATH="$2"; shift 2;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2;;
        --suite)          SUITE=true; shift;;
        --pairs_dir)      PAIRS_DIR="$2"; shift 2;;
        --training_dir)   TRAINING_DIR="$2"; shift 2;;
        --variants)       VARIANTS="$2"; shift 2;;
        --variant_suffix) VARIANT_SUFFIX="$2"; shift 2;;
        --seeds)          SEEDS="$2"; shift 2;;
        --wandb_project)  WANDB_PROJECT="$2"; shift 2;;
        --wandb_entity)   WANDB_ENTITY="$2"; shift 2;;
        --wandb_tags)     WANDB_TAGS="$2"; shift 2;;
        --wandb_mode)     WANDB_MODE="$2"; shift 2;;
        --gpus)           GPU_LIST="$2"; shift 2;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Required: --model"
    usage
    exit 1
fi

# Canonicalize model path/name to avoid case drift in output directories.
MODEL_PATH=$(python3 -c "import src.config as config; print(config.resolve_model_path('$MODEL'))")
MODEL_SHORT=$(python3 -c "import src.config as config; print(config.get_model_short_name(config.resolve_model_path('$MODEL')))")

if [[ -n "$GPU_LIST" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

RUN_COUNT=0
SKIP_COUNT=0

run_one() {
    local name="$1"
    local ds_path="$2"
    local out_dir="$3"
    shift 3

    if [[ ! -f "$ds_path" ]]; then
        echo "  SKIP $name: $ds_path not found"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    echo ""
    echo "=========================================="
    echo "Training: $name"
    echo "  Model:   $MODEL_PATH"
    echo "  Dataset: $ds_path"
    echo "  Output:  $out_dir"
    if [[ -n "$GPU_LIST" ]]; then
        echo "  GPUs:    $GPU_LIST"
    fi
    echo "=========================================="

    local wandb_args=""
    if [[ -n "$WANDB_PROJECT" ]]; then
        local run_name
        run_name=$(basename "$out_dir")
        wandb_args="--wandb_project $WANDB_PROJECT --wandb_run_name $run_name --wandb_mode $WANDB_MODE"
        if [[ -n "$WANDB_ENTITY" ]]; then
            wandb_args="$wandb_args --wandb_entity $WANDB_ENTITY"
        fi
        local tags="${name// /_}"
        if [[ -n "$WANDB_TAGS" ]]; then
            tags="$WANDB_TAGS,$tags"
        fi
        wandb_args="$wandb_args --wandb_tags $tags"
    fi

    python -m src.dpo.train_dpo \
        --model "$MODEL_PATH" \
        --dataset_path "$ds_path" \
        --output_dir "$out_dir" \
        $wandb_args \
        "$@" \
        $EXTRA_ARGS

    RUN_COUNT=$((RUN_COUNT + 1))
}

if $SUITE; then
    if [[ -z "$DATASET" ]]; then
        echo "ERROR: --dataset is required in --suite mode (e.g. --dataset gsm8k_train)."
        usage
        exit 1
    fi
    PAIRS_DIR="${PAIRS_DIR:-./output/07_cliff_dpo/02_pairs/${MODEL_SHORT}}"
    TRAINING_DIR="${TRAINING_DIR:-./output/07_cliff_dpo/03_training/${MODEL_SHORT}}"
    TRAINING_DIR_DS="${TRAINING_DIR}/${DATASET}"

    IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"
    IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

    echo "Running Cliff-DPO suite..."
    echo "  model:        $MODEL_PATH"
    echo "  dataset:      $DATASET"
    echo "  pairs_dir:    $PAIRS_DIR"
    echo "  training_dir: $TRAINING_DIR_DS"
    echo "  variants:     $VARIANTS${VARIANT_SUFFIX:+ (suffix $VARIANT_SUFFIX)}"
    echo "  seeds:        $SEEDS"

    for seed in "${SEED_ARR[@]}"; do
        for variant in "${VARIANT_ARR[@]}"; do
            stem="cliff_${variant}_only${VARIANT_SUFFIX}"
            run_one "Cliff-DPO ${variant}${VARIANT_SUFFIX} seed${seed}" \
                "$PAIRS_DIR/${stem}_${DATASET}.json" \
                "$TRAINING_DIR_DS/${stem}_seed${seed}" \
                --seed "$seed"
        done
    done

    echo ""
    echo "=========================================="
    echo "Suite complete."
    echo "  Trained: $RUN_COUNT"
    echo "  Skipped: $SKIP_COUNT"
    echo "=========================================="
else
    if [[ -z "$DATASET_PATH" || -z "$OUTPUT_DIR" ]]; then
        echo "Single mode requires --dataset_path and --output_dir"
        usage
        exit 1
    fi
    run_one "DPO Training" "$DATASET_PATH" "$OUTPUT_DIR"
fi

if [[ "$RUN_COUNT" -eq 0 ]]; then
    echo "ERROR: no training job was executed (all datasets missing or skipped)."
    exit 2
fi
