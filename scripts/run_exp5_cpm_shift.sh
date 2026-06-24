#!/usr/bin/env bash
# RQ2-3: Cross-Model Cliff Probability Mass (CPM) Shift.
# Iterates (source, eval, dataset) triples, runs per-combo GPU worker, then
# invokes the analyzer.
set -euo pipefail

export VLLM_USE_V1=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

cleanup() {
  local sig=$1
  echo ""
  echo "[cleanup] received $sig, killing descendant processes..."
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
  exit 130
}
trap 'cleanup INT'  INT
trap 'cleanup TERM' TERM

SOURCES=""
EVALS=""
DATASETS="gsm1k_100,math500_100,aime25"
GPU_LIST=""
ROLLOUT_DIR="./output/03_rollout"
EXP4_CANDIDATES_ROOT="./output/07_candidate_replacement"
OUTPUT_DIR=""
ANALYSIS_ONLY=0
PAIR_DIR=""
EXP4_CANDIDATES_SMALLTOBIG_DIR=""
EXP4_CANDIDATES_BIGTOSMALL_DIR=""
EXTRA_OUTPUT_DIR=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp5_cpm_shift.sh [options]

RQ2-3 Cross-Model CPM Shift. For every (source, eval) pair where source != eval,
runs a vLLM forward pass with the eval model on the source's cliff-position
prefixes and records how the source's cliff token mass shifts in eval's top-20.

Options:
  --sources "m1,m2,..."          Source model aliases (required unless --analysis_only)
  --evals "m1,m2,..."            Eval model aliases (required unless --analysis_only)
  --datasets "d1,d2,..."         Datasets (default: gsm1k_100,math500_100,aime25)
  --gpus "0"                     GPU IDs passed to vLLM (comma list); required unless --analysis_only
  --rollout_dir PATH             (default: ./output/03_rollout)
  --exp4_candidates_root PATH             (default: ./output/07_candidate_replacement)
  --output_dir PATH              Batch output dir (default: ./output/08_cpm_shift/<ts>_batch)
  --analysis_only                Skip GPU work; re-run Phase 2 on --output_dir
  --pair PATH                    Second-direction batch dir; enables cross-direction extra plots
  --exp4_candidates_smalltobig_dir PATH   Override exp4_candidates dir for small-source direction
  --exp4_candidates_bigtosmall_dir PATH   Override exp4_candidates dir for big-source direction
  --extra_output_dir PATH        Output dir for extra cross-direction figures
                                 (default: <smalltobig_dir>/extra_figures)

Examples:
  bash scripts/run_exp5_cpm_shift.sh \
      --sources qwen3-0.6b,qwen3-8b --evals qwen3-0.6b,qwen3-8b \
      --datasets gsm1k_100,math500_100 --gpus 0

  # Re-run analysis only, with paired cross-direction plots
  bash scripts/run_exp5_cpm_shift.sh --analysis_only \
      --output_dir output/08_cpm_shift/0410_small_to_big \
      --pair output/08_cpm_shift/0410_big_to_small
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sources)                SOURCES="$2"; shift 2 ;;
    --evals)                  EVALS="$2"; shift 2 ;;
    --datasets)               DATASETS="$2"; shift 2 ;;
    --gpus)                   GPU_LIST="$2"; shift 2 ;;
    --rollout_dir)            ROLLOUT_DIR="$2"; shift 2 ;;
    --exp4_candidates_root)            EXP4_CANDIDATES_ROOT="$2"; shift 2 ;;
    --output_dir)             OUTPUT_DIR="$2"; shift 2 ;;
    --analysis_only)          ANALYSIS_ONLY=1; shift ;;
    --pair)                   PAIR_DIR="$2"; shift 2 ;;
    --exp4_candidates_smalltobig_dir)  EXP4_CANDIDATES_SMALLTOBIG_DIR="$2"; shift 2 ;;
    --exp4_candidates_bigtosmall_dir)  EXP4_CANDIDATES_BIGTOSMALL_DIR="$2"; shift 2 ;;
    --extra_output_dir)       EXTRA_OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)                usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$ANALYSIS_ONLY" -eq 0 ]] && [[ -z "$SOURCES" || -z "$EVALS" || -z "$GPU_LIST" ]]; then
  echo "ERROR: --sources, --evals, --gpus are required"
  usage
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP=$(date +%m%d_%H%M%S)
  OUTPUT_DIR="./output/08_cpm_shift/${TIMESTAMP}_batch"
fi
RUNS_DIR="$OUTPUT_DIR/runs"

# ---- Shared Phase 2 function ----
_run_phase2() {
  local batch_dir="$1"
  echo "[Phase 2] Aggregation on $batch_dir"
  python3 scripts/_exp5_cpm_shift_analyze.py "$batch_dir"

  if [[ -n "$PAIR_DIR" ]]; then
    if [[ ! -d "$PAIR_DIR/runs" ]]; then
      echo "ERROR: --pair $PAIR_DIR/runs not found" >&2
      exit 1
    fi
    echo ""
    echo "[Phase 2] Aggregation on $PAIR_DIR (pair)"
    python3 scripts/_exp5_cpm_shift_analyze.py "$PAIR_DIR"

    _src_model() {
      python3 - "$1" <<'PY'
import csv, os, sys
p = os.path.join(sys.argv[1], "results_per_cliff.csv")
if not os.path.exists(p):
    sys.exit(0)
with open(p) as f:
    for row in csv.DictReader(f):
        print(row.get("model_source", ""))
        break
PY
    }

    _model_size_score() {
      python3 - "$1" <<'PY'
import re, sys
name = sys.argv[1].lower()
m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
print(float(m.group(1)) if m else 0.0)
PY
    }

    SRC_A=$(_src_model "$batch_dir")
    SRC_B=$(_src_model "$PAIR_DIR")
    SIZE_A=$(_model_size_score "$SRC_A")
    SIZE_B=$(_model_size_score "$SRC_B")

    if python3 -c "import sys; sys.exit(0 if float('$SIZE_A') <= float('$SIZE_B') else 1)"; then
      SMALLTOBIG_DIR="$batch_dir"
      BIGTOSMALL_DIR="$PAIR_DIR"
    else
      SMALLTOBIG_DIR="$PAIR_DIR"
      BIGTOSMALL_DIR="$batch_dir"
    fi
    echo ""
    echo "Extra plots: smalltobig=$SMALLTOBIG_DIR (source ~${SIZE_A}B)  bigtosmall=$BIGTOSMALL_DIR (source ~${SIZE_B}B)"

    EXTRA_CMD=(
      python3 scripts/exp5_cpm_shift_extra_plots.py
      --smalltobig_dir "$SMALLTOBIG_DIR"
      --bigtosmall_dir "$BIGTOSMALL_DIR"
      --exp4_candidates_root "$EXP4_CANDIDATES_ROOT"
    )
    [[ -n "$EXP4_CANDIDATES_SMALLTOBIG_DIR" ]] && EXTRA_CMD+=(--exp4_candidates_smalltobig_dir "$EXP4_CANDIDATES_SMALLTOBIG_DIR")
    [[ -n "$EXP4_CANDIDATES_BIGTOSMALL_DIR" ]] && EXTRA_CMD+=(--exp4_candidates_bigtosmall_dir "$EXP4_CANDIDATES_BIGTOSMALL_DIR")
    [[ -n "$EXTRA_OUTPUT_DIR" ]]       && EXTRA_CMD+=(--output_dir "$EXTRA_OUTPUT_DIR")
    "${EXTRA_CMD[@]}"
  fi
}

if [[ "$ANALYSIS_ONLY" -eq 1 ]]; then
  if [[ ! -d "$OUTPUT_DIR/runs" ]]; then
    echo "ERROR: $OUTPUT_DIR/runs not found" >&2
    exit 1
  fi
  echo "============================================================"
  echo "RQ2-3 Analysis-only (Phase 2)"
  echo "============================================================"
  echo "Batch dir: $OUTPUT_DIR"
  [[ -n "$PAIR_DIR" ]] && echo "Pair dir:  $PAIR_DIR"
  echo ""
  _run_phase2 "$OUTPUT_DIR"
  echo ""
  echo "============================================================"
  echo "ANALYSIS COMPLETE"
  echo "============================================================"
  ls -1 "$OUTPUT_DIR/figures/" 2>/dev/null || true
  exit 0
fi

mkdir -p "$RUNS_DIR"

echo "============================================================"
echo "RQ2-3 Cross-Model CPM Shift"
echo "============================================================"
echo "Sources:      $SOURCES"
echo "Evals:        $EVALS"
echo "Datasets:     $DATASETS"
echo "GPUs:         $GPU_LIST"
echo "Rollout dir:  $ROLLOUT_DIR"
echo "Exp2_2 root:  $EXP4_CANDIDATES_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo ""

# Resolve alias → directory basename via Python helper.
_model_dir_name() {
  python3 - "$1" <<'PY'
import sys, os
sys.path.insert(0, ".")
from src.config import resolve_model_path
alias = sys.argv[1]
print(os.path.basename(resolve_model_path(alias).rstrip("/")))
PY
}

_latest_exp4_candidates_dir() {
  local model_dir="$1"
  ls -1d "$EXP4_CANDIDATES_ROOT/${model_dir}_"*/ 2>/dev/null | sort | tail -n 1
}

IFS=',' read -ra SRC_ARR <<< "$SOURCES"
IFS=',' read -ra EVL_ARR <<< "$EVALS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

n_total=0
n_run=0
n_skip=0

for SRC in "${SRC_ARR[@]}"; do
  SRC_DIR_NAME=$(_model_dir_name "$SRC")
  for EVL in "${EVL_ARR[@]}"; do
    if [[ "$SRC" == "$EVL" ]]; then
      continue
    fi
    EVL_DIR_NAME=$(_model_dir_name "$EVL")
    for DS in "${DS_ARR[@]}"; do
      n_total=$((n_total + 1))
      COMBO_DIR="$RUNS_DIR/${SRC_DIR_NAME}__${EVL_DIR_NAME}__${DS}"

      # Existence checks
      SRC_ROLLOUT="$ROLLOUT_DIR/$SRC_DIR_NAME/${DS}_all_paths.json"
      EVL_ROLLOUT="$ROLLOUT_DIR/$EVL_DIR_NAME/${DS}_all_paths.json"
      if [[ ! -f "$SRC_ROLLOUT" ]]; then
        echo "  [skip-missing-source-rollout] $SRC/$EVL/$DS ($SRC_ROLLOUT)"
        n_skip=$((n_skip + 1)); continue
      fi
      if [[ ! -f "$EVL_ROLLOUT" ]]; then
        echo "  [skip-missing-eval-rollout] $SRC/$EVL/$DS ($EVL_ROLLOUT)"
        n_skip=$((n_skip + 1)); continue
      fi
      EXP4_CANDIDATES_DIR=$(_latest_exp4_candidates_dir "$SRC_DIR_NAME" | sed 's:/$::')
      if [[ -z "$EXP4_CANDIDATES_DIR" || ! -f "$EXP4_CANDIDATES_DIR/cliff_instances.csv" ]]; then
        echo "  [skip-missing-exp4_candidates] $SRC/$EVL/$DS (no dir under $EXP4_CANDIDATES_ROOT/${SRC_DIR_NAME}_*)"
        n_skip=$((n_skip + 1)); continue
      fi

      mkdir -p "$COMBO_DIR"
      LOG="$COMBO_DIR/run.log"
      echo "  [run] $SRC → $EVL / $DS  (exp4_candidates=$EXP4_CANDIDATES_DIR)  log=$LOG"
      if python3 scripts/_exp5_cpm_shift_eval.py \
            --source "$SRC" --eval "$EVL" --dataset "$DS" \
            --exp4_candidates_source_dir "$EXP4_CANDIDATES_DIR" \
            --rollout_dir "$ROLLOUT_DIR" \
            --gpus "$GPU_LIST" \
            --output_dir "$COMBO_DIR" \
            > "$LOG" 2>&1; then
        n_run=$((n_run + 1))
      else
        echo "  [FAIL] $SRC/$EVL/$DS — see $LOG"
      fi
      # Give vLLM spawn workers a moment to release GPU memory before the
      # next combo on the same GPUs.
      sleep 3
    done
  done
done

echo ""
echo "============================================================"
echo "Phase 1 complete: $n_run/$n_total combos run, $n_skip skipped."
echo "============================================================"

# Phase 2: aggregate + plot
echo ""
echo "============================================================"
echo "Phase 2: Aggregation"
echo "============================================================"
_run_phase2 "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR/figures/" 2>/dev/null || true
