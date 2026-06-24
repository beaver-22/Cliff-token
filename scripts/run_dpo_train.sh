#!/bin/bash
# Cliff-DPO Training Wrapper (5-variant suite)
#
# Usage:
#   # Single experiment
#   bash scripts/run_dpo_train.sh --model ./model/Qwen3-0.6B \
#       --dataset_path ./output/09_cliff_dpo/02_pairs/Qwen3-0.6B/cliff_all_gsm8k.json \
#       --output_dir ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all
#
#   # Full 5-variant suite
#   bash scripts/run_dpo_train.sh --suite --model ./model/Qwen3-0.6B \
#       --dataset gsm8k \
#       --pairs_dir ./output/09_cliff_dpo/02_pairs/Qwen3-0.6B/ \
#       --training_dir ./output/09_cliff_dpo/03_training/Qwen3-0.6B/ \
#       --gpus 0

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
  --dataset NAME
  [--pairs_dir PATH]
  [--training_dir PATH]
  Runs 5 Cliff-DPO variants:
    cliff_all, cliff_deterministic_only, cliff_uncertainty_only,
    cliff_sampled_off_only, cliff_uncertainty_sampled_off_only

Common options:
  --gpus "0,1"
  --wandb_project NAME
  --wandb_entity NAME
  --wandb_tags "tag1,tag2"
  --wandb_mode online|offline|disabled
  -h, --help

Any unknown options are forwarded to `python -m src.dpo.train_dpo`.
EOF
}

# Defaults
MODEL=""
DATASET=""            # suite mode: dataset suffix for pair filenames (e.g. gsm8k)
DATASET_PATH=""
OUTPUT_DIR=""
SUITE=false
PAIRS_DIR=""
TRAINING_DIR=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_TAGS=""
WANDB_MODE="online"
GPU_LIST=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      usage; exit 0;;
        --model)        MODEL="$2"; shift 2;;
        --dataset)      DATASET="$2"; shift 2;;
        --dataset_path) DATASET_PATH="$2"; shift 2;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2;;
        --suite)        SUITE=true; shift;;
        --pairs_dir)    PAIRS_DIR="$2"; shift 2;;
        --training_dir) TRAINING_DIR="$2"; shift 2;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2;;
        --wandb_entity)  WANDB_ENTITY="$2"; shift 2;;
        --wandb_tags)    WANDB_TAGS="$2"; shift 2;;
        --wandb_mode)    WANDB_MODE="$2"; shift 2;;
        --gpus)         GPU_LIST="$2"; shift 2;;
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

# Optional GPU pinning for Step 4 training/eval.
# Example: --gpus 7 -> CUDA_VISIBLE_DEVICES=7
if [[ -n "$GPU_LIST" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

# Auto-fill default dirs for suite mode based on canonical model short name.
if $SUITE; then
    if [[ -z "$PAIRS_DIR" ]]; then
        PAIRS_DIR="./output/09_cliff_dpo/02_pairs/${MODEL_SHORT}"
    fi
    if [[ -z "$TRAINING_DIR" ]]; then
        TRAINING_DIR="./output/09_cliff_dpo/03_training/${MODEL_SHORT}"
    fi
fi

RUN_COUNT=0
SKIP_COUNT=0

run_one() {
    local name="$1"
    local ds_path="$2"
    local out_dir="$3"

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
        echo "  GPUs:    $GPU_LIST (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    fi
    echo "=========================================="

    local wandb_args=""
    if [[ -n "$WANDB_PROJECT" ]]; then
        # Each experiment gets its own run name based on output dir
        local run_name
        run_name=$(basename "$out_dir")
        wandb_args="--wandb_project $WANDB_PROJECT --wandb_run_name $run_name --wandb_mode $WANDB_MODE"
        if [[ -n "$WANDB_ENTITY" ]]; then
            wandb_args="$wandb_args --wandb_entity $WANDB_ENTITY"
        fi
        # Tag each run with its experiment name so they group nicely in wandb.
        # Replace spaces with underscores so the tag stays a single shell token.
        local name_tag="${name// /_}"
        local tags="$name_tag"
        if [[ -n "$WANDB_TAGS" ]]; then
            tags="$WANDB_TAGS,$name_tag"
        fi
        wandb_args="$wandb_args --wandb_tags $tags"
    fi

    python -m src.dpo.train_dpo \
        --model "$MODEL_PATH" \
        --dataset_path "$ds_path" \
        --output_dir "$out_dir" \
        $wandb_args \
        $EXTRA_ARGS

    RUN_COUNT=$((RUN_COUNT + 1))
}

if $SUITE; then
    if [[ -z "$DATASET" ]]; then
        echo "ERROR: --dataset is required in --suite mode so that pair/output filenames are unique."
        echo "       Example: --dataset gsm8k"
        usage
        exit 1
    fi

    # Dataset-suffixed filenames (matches build_dpo_pairs.py / build_baseline.py output)
    SFX="_${DATASET}"
    # Per-dataset training sub-directory to keep different dataset runs separate.
    TRAINING_DIR_DS="${TRAINING_DIR}/${DATASET}"

    echo "Running Cliff-DPO 5-variant suite..."
    echo "  model:        $MODEL_PATH"
    echo "  model_short:  $MODEL_SHORT"
    echo "  dataset:      $DATASET"
    echo "  pairs_dir:    $PAIRS_DIR"
    echo "  training_dir: $TRAINING_DIR_DS"
    if [[ -n "$GPU_LIST" ]]; then
        echo "  gpus:         $GPU_LIST (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    fi

    # Resolve canonical file first, then compatibility aliases if needed.
    resolve_pair_path() {
        local canonical="$1"
        shift
        if [[ -f "$canonical" ]]; then
            echo "$canonical"
            return
        fi
        for alt in "$@"; do
            if [[ -f "$alt" ]]; then
                echo "$alt"
                return
            fi
        done
        # Keep canonical path for clear SKIP logging in run_one.
        echo "$canonical"
    }

    # 1) all
    PAIR_ALL=$(resolve_pair_path \
        "$PAIRS_DIR/cliff_all${SFX}.json" \
        "$PAIRS_DIR/cliff_1N_all${SFX}.json")
    run_one "Cliff-DPO all" "$PAIR_ALL" "$TRAINING_DIR_DS/cliff_all"

    # 2) deterministic_only
    PAIR_DET=$(resolve_pair_path \
        "$PAIRS_DIR/cliff_deterministic_only${SFX}.json" \
        "$PAIRS_DIR/cliff_1N_deterministic_only${SFX}.json")
    run_one "Cliff-DPO deterministic-only" "$PAIR_DET" "$TRAINING_DIR_DS/cliff_deterministic_only"

    # 3) uncertainty_only (internal taxonomy key remains 'uncertain')
    PAIR_UNC=$(resolve_pair_path \
        "$PAIRS_DIR/cliff_uncertainty_only${SFX}.json" \
        "$PAIRS_DIR/cliff_uncertain_only${SFX}.json" \
        "$PAIRS_DIR/cliff_1N_uncertain_only${SFX}.json")
    run_one "Cliff-DPO uncertainty-only" "$PAIR_UNC" "$TRAINING_DIR_DS/cliff_uncertainty_only"

    # 4) sampled_off_only
    PAIR_SOFF=$(resolve_pair_path \
        "$PAIRS_DIR/cliff_sampled_off_only${SFX}.json" \
        "$PAIRS_DIR/cliff_1N_sampled_off_only${SFX}.json")
    run_one "Cliff-DPO sampled-off-only" "$PAIR_SOFF" "$TRAINING_DIR_DS/cliff_sampled_off_only"

    # 5) uncertainty_sampled_off_only
    PAIR_UNC_SOFF=$(resolve_pair_path \
        "$PAIRS_DIR/cliff_uncertainty_sampled_off_only${SFX}.json" \
        "$PAIRS_DIR/cliff_uncertain_sampled_off_only${SFX}.json")
    run_one "Cliff-DPO uncertainty-sampled-off-only" \
        "$PAIR_UNC_SOFF" \
        "$TRAINING_DIR_DS/cliff_uncertainty_sampled_off_only"

    echo ""
    echo "=========================================="
    echo "5-variant suite complete."
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
