#!/usr/bin/env bash
# RQ1-3 unified batch entrypoint.
# Runs all (model, dataset) combinations end-to-end:
#   Phase 1 (GPU): cliff detection + 4-method deletion experiments per pair
#   Phase 2 (CPU): aggregation, model x dataset grids, failure-only grids,
#                  exp2 methods grids, CSV/MD exports
#
# To re-run only Phase 2 against an existing batch dir, use
# --analysis_only --output_dir <batch_dir>.
set -euo pipefail

export VLLM_USE_V1=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ----- Cleanup trap: kill entire process group on exit/interrupt -----
# Without this, background children (python3 _run_exp1_deletion.py + its
# spawned vLLM workers) survive a Ctrl+C and leak GPU memory.
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
GPU_LIST=""
ROLLOUT_DIR="./output/03_rollout"
OUTPUT_DIR=""
NUM_SAMPLES=64
ANALYSIS_ONLY=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp1_deletion.sh [options]

RQ1-3 unified batch: run cliff-del/keep + critical/tangent/random
experiments across many (model, dataset) combinations on multiple GPUs,
then aggregate per-model and produce all grid plots, CSVs, and Markdown
tables. Combinations without rollout data are skipped automatically.

Options:
  --models "m1,m2,..."    Model names (default: auto-discover from rollout_dir)
  --datasets "d1,d2,..."  Dataset names (default: auto-discover)
  --gpus "0,1"            GPU IDs to use (round-robin); required (unless --analysis_only)
  --rollout_dir PATH      Rollout dir (default: ./output/03_rollout)
  --output_dir PATH       Batch output dir (default: ./output/05_deletion_ablation/<timestamp>_batch)
  --num_samples N         Greedy rollout samples per cliff (default: 64)
  --analysis_only         Skip GPU work; re-run Phase 2 on --output_dir

Examples:
  # Full paper batch
  scripts/run_exp1_deletion.sh \
      --models "Qwen3-0.6B,Qwen3-4B,Qwen3-8B,Llama-3.2-1B-Instruct,Llama-3.2-3B-Instruct,Llama-3.1-8B-Instruct,gemma-3-4b-it" \
      --datasets "gsm1k_100,math500_100,aime25" \
      --gpus 0

  # Re-run analysis only on an existing batch dir (no GPU)
  scripts/run_exp1_deletion.sh --analysis_only \
      --output_dir output/05_deletion_ablation/0407_140148_batch
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)         MODELS="$2"; shift 2 ;;
    --datasets)       DATASETS="$2"; shift 2 ;;
    --gpus)           GPU_LIST="$2"; shift 2 ;;
    --rollout_dir)    ROLLOUT_DIR="$2"; shift 2 ;;
    --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --num_samples)    NUM_SAMPLES="$2"; shift 2 ;;
    --analysis_only)  ANALYSIS_ONLY=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$ANALYSIS_ONLY" -eq 1 ]]; then
  if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output_dir required with --analysis_only"
    exit 1
  fi
  if [[ ! -d "$OUTPUT_DIR/runs" ]]; then
    echo "ERROR: $OUTPUT_DIR/runs not found" >&2
    exit 1
  fi
  echo "============================================================"
  echo "RQ1-3 Analysis-only (Phase 2)"
  echo "============================================================"
  echo "Batch dir: $OUTPUT_DIR"
  echo ""
  python3 scripts/_exp1_deletion_analyze.py "$OUTPUT_DIR"
  echo ""
  echo "============================================================"
  echo "ANALYSIS COMPLETE"
  echo "============================================================"
  ls -1 "$OUTPUT_DIR/grid/" 2>/dev/null || true
  exit 0
fi

if [[ -z "$GPU_LIST" ]]; then
  echo "ERROR: --gpus required"
  usage
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP=$(date +%m%d_%H%M%S)
  OUTPUT_DIR="./output/05_deletion_ablation/${TIMESTAMP}_batch"
fi
RUNS_DIR="$OUTPUT_DIR/runs"
mkdir -p "$RUNS_DIR"

echo "============================================================"
echo "RQ1-3 Batch (Phase 1: GPU rollout, Phase 2: analysis)"
echo "============================================================"
echo "Models:       ${MODELS:-(auto)}"
echo "Datasets:     ${DATASETS:-(auto)}"
echo "GPUs:         $GPU_LIST"
echo "Rollout dir:  $ROLLOUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Samples:      $NUM_SAMPLES"
echo ""

# ---------------------------------------------------------
# Discover (model, dataset) combinations with valid rollout
# ---------------------------------------------------------
COMBOS_FILE="$OUTPUT_DIR/_combos.tsv"
python3 - "$ROLLOUT_DIR" "$MODELS" "$DATASETS" "$COMBOS_FILE" <<'PY'
import sys, json
from pathlib import Path

rollout_dir = Path(sys.argv[1])
models_filter = [m.strip() for m in sys.argv[2].split(",") if m.strip()] if sys.argv[2] else []
datasets_filter = [d.strip() for d in sys.argv[3].split(",") if d.strip()] if sys.argv[3] else []
combos_file = sys.argv[4]

combos = []
for model_dir in sorted(rollout_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    model_name = model_dir.name
    if models_filter and model_name not in models_filter:
        continue
    for f in sorted(model_dir.glob("*_all_paths.json")):
        ds = f.name.replace("_all_paths.json", "")
        if datasets_filter and ds not in datasets_filter:
            continue
        try:
            data = json.load(open(f))
            if not data or len(data[0].get("all_position_scores", [])) == 0:
                print(f"  SKIP {model_name}/{ds}: no position scores")
                continue
            combos.append((model_name, ds, str(f)))
        except Exception as e:
            print(f"  SKIP {model_name}/{ds}: {e}")

if not combos:
    print("ERROR: no combinations to run")
    sys.exit(1)

print(f"\nDiscovered {len(combos)} (model, dataset) combinations:")
for m, d, _ in combos:
    print(f"  {m}/{d}")

with open(combos_file, "w") as f:
    for m, d, p in combos:
        f.write(f"{m}\t{d}\t{p}\n")
PY

# ---------------------------------------------------------
# Phase 1: GPU runs (round-robin across --gpus)
# ---------------------------------------------------------
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
echo ""
echo "============================================================"
echo "Phase 1: GPU runs ($NUM_GPUS GPUs in parallel)"
echo "============================================================"

declare -A PID_TO_GPU=()
FREE_GPUS=("${GPUS[@]}")

reap_one() {
  wait -n 2>/dev/null || true
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      FREE_GPUS+=("${PID_TO_GPU[$pid]}")
      unset 'PID_TO_GPU[$pid]'
      return 0
    fi
  done
}

# Hard wall-clock cap as a safety net per pair (vLLM hangs etc).
JOB_TIMEOUT="${JOB_TIMEOUT:-120m}"

while IFS=$'\t' read -r MODEL DS DATA_PATH; do
  RUN_OUT="$RUNS_DIR/${MODEL}_${DS}"

  # Skip if all 4 sub_exp_2 variants already exist (true completion marker)
  if [[ -f "$RUN_OUT/sub_exp_2/exp2_pass_at_k_all_failure.json" ]]; then
    echo "  [skip-existing] $MODEL/$DS already complete in $RUN_OUT"
    continue
  fi

  while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
    reap_one
  done

  GPU="${FREE_GPUS[0]}"
  FREE_GPUS=("${FREE_GPUS[@]:1}")

  mkdir -p "$RUN_OUT"
  LOG="$RUN_OUT/run.log"
  MODEL_ALIAS=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -e 's/-instruct$//' -e 's/-it$//')
  echo "  [start] $MODEL/$DS on GPU $GPU (alias=$MODEL_ALIAS) → log: $LOG"
  (
    CUDA_VISIBLE_DEVICES="$GPU" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        timeout --signal=KILL --kill-after=10s "$JOB_TIMEOUT" \
        python3 scripts/_run_exp1_deletion.py \
            "$MODEL_ALIAS" "$DS" "$DATA_PATH" "$GPU" "$NUM_SAMPLES" "$RUN_OUT" \
        > "$LOG" 2>&1 || echo "  [FAIL] $MODEL/$DS — see $LOG"
  ) &
  PID_TO_GPU[$!]=$GPU
done < "$COMBOS_FILE"

while [[ ${#PID_TO_GPU[@]} -gt 0 ]]; do
  reap_one
done
echo "  All GPU runs finished."

# ---------------------------------------------------------
# Phase 2: aggregation, plots, CSV, MD
# ---------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 2: Aggregation, plots, CSV, MD"
echo "============================================================"

python3 scripts/_exp1_deletion_analyze.py "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Per-model results:"
ls -1 "$OUTPUT_DIR/per_model/" 2>/dev/null || true
echo ""
echo "Grid:"
ls -1 "$OUTPUT_DIR/grid/" 2>/dev/null || true
echo ""
echo "Summary:"
ls -1 "$OUTPUT_DIR/summary_table.csv" 2>/dev/null && head -5 "$OUTPUT_DIR/summary_table.csv" || true
