#!/usr/bin/env bash
# RQ1-1: Cliff Token Occurrence Analysis
# Discovers rollout results, runs multi-model analysis, generates all outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ROLLOUT_DIR="./output/03_rollout"
OUTPUT_DIR=""
MODELS=""
DATASETS=""

usage() {
  cat <<'USAGE'
Usage: scripts/run_exp1_occurrence.sh [options]

RQ1-1: Cliff Token Occurrence Analysis (Success vs Failure).
Auto-discovers completed rollout results from output/03_rollout/.

Options:
  --models "m1,m2,..."     Model names (default: auto-discover all)
  --datasets "d1,d2,..."   Dataset names (default: auto-discover all)
  --rollout_dir PATH       Rollout results directory (default: ./output/03_rollout)
  --output_dir PATH        Output directory (default: ./output/04_cliff_occurrence/<timestamp>)

Examples:
  # Auto-discover all models and datasets
  scripts/run_exp1_occurrence.sh

  # Specific models and datasets
  scripts/run_exp1_occurrence.sh --models "Qwen3-0.6B,Qwen3-4B,Qwen3-8B" --datasets "gsm1k_100"

  # All models, specific dataset
  scripts/run_exp1_occurrence.sh --datasets "gsm1k_100,math500_100"
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)      MODELS="$2"; shift 2 ;;
    --datasets)    DATASETS="$2"; shift 2 ;;
    --rollout_dir) ROLLOUT_DIR="$2"; shift 2 ;;
    --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Default output dir with timestamp
if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP=$(date +%m%d_%H%M%S)
  OUTPUT_DIR="./output/04_cliff_occurrence/${TIMESTAMP}"
fi

echo "============================================================"
echo "RQ1-1: Cliff Token Occurrence Analysis"
echo "============================================================"
echo "Rollout dir: $ROLLOUT_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Build model_dataset_map as JSON via Python
# Auto-discovers available *_all_paths.json files
python3 - "$ROLLOUT_DIR" "$OUTPUT_DIR" "$MODELS" "$DATASETS" <<'PY'
import sys, json, os
from pathlib import Path

rollout_dir = Path(sys.argv[1])
output_dir = sys.argv[2]
models_filter = [m.strip() for m in sys.argv[3].split(",") if m.strip()] if sys.argv[3] else []
datasets_filter = [d.strip() for d in sys.argv[4].split(",") if d.strip()] if sys.argv[4] else []

# Discover available results
model_dataset_map = {}
for model_dir in sorted(rollout_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    model_name = model_dir.name
    # Skip models not used in main analysis
    SKIP_MODELS = {"Qwen3-8B-greedy", "gemma-3-1b-it", "gemma-3-12b-it"}
    if model_name in SKIP_MODELS:
        print(f"  SKIP {model_name} (excluded)")
        continue
    if models_filter and model_name not in models_filter:
        continue

    datasets = {}
    for f in sorted(model_dir.glob("*_all_paths.json")):
        # Extract dataset name: gsm1k_all_paths.json → gsm1k
        ds = f.name.replace("_all_paths.json", "")
        if datasets_filter and ds not in datasets_filter:
            continue
        # Verify file has scores
        try:
            data = json.load(open(f))
            if data and len(data[0].get("all_position_scores", [])) > 0:
                datasets[ds] = str(f)
            else:
                print(f"  SKIP {model_name}/{ds}: no position scores")
        except Exception as e:
            print(f"  SKIP {model_name}/{ds}: {e}")

    if datasets:
        model_dataset_map[model_name] = datasets

if not model_dataset_map:
    print("ERROR: No completed rollout results found!")
    print(f"  Searched: {rollout_dir}")
    if models_filter:
        print(f"  Models filter: {models_filter}")
    if datasets_filter:
        print(f"  Datasets filter: {datasets_filter}")
    sys.exit(1)

print("Discovered rollout results:")
for model, datasets in model_dataset_map.items():
    for ds, path in datasets.items():
        print(f"  {model}/{ds} → {path}")
print()

# Run analysis
from src.analysis.curves import run_multi_model_analysis
run_multi_model_analysis(model_dataset_map, output_dir)
PY

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo "Files:"
ls -1 "$OUTPUT_DIR"/ 2>/dev/null || true
echo ""
if [[ -d "$OUTPUT_DIR/case_studies" ]]; then
  echo "Case studies:"
  ls -1 "$OUTPUT_DIR/case_studies/" 2>/dev/null || true
fi
