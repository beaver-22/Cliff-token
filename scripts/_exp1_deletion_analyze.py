"""Phase 2 entrypoint for exp1_3: aggregation, plots, CSV/MD.

Walks an existing exp1_3 batch directory's `runs/` subfolder and produces all
post-rollout artifacts (per-model aggregations, semantic union, model x dataset
grids, failure-only grid + tables, exp2 methods grid + tables). Performs no
GPU work.

Usage:
    python3 scripts/_exp1_deletion_analyze.py <batch_dir>
"""
import os
import sys

sys.path.insert(0, ".")

from src.decoding.evaluator import (
    DATASET_DISPLAY_ORDER,
    MODEL_DISPLAY_ORDER,
    _ordered,
    aggregate_per_model,
    aggregate_per_model_exp2,
    aggregate_semantic_all,
    plot_exp2_methods_grid,
    plot_failure_only_grid,
    plot_model_dataset_grid,
)


def discover_runs(runs_dir):
    """Scan runs/ for `<model>_<dataset>` subdirs with cliff_results.json.

    Returns (result_dirs, all_models, all_datasets).
    """
    result_dirs = {}
    all_models, all_datasets = [], []
    for entry in sorted(os.listdir(runs_dir)):
        rp = os.path.join(runs_dir, entry)
        if not os.path.isdir(rp):
            continue
        matched_ds = None
        model = None
        for ds in DATASET_DISPLAY_ORDER:
            suffix = f"_{ds}"
            if entry.endswith(suffix):
                matched_ds = ds
                model = entry[: -len(suffix)]
                break
        if matched_ds is None:
            print(f"  WARN: cannot parse model/dataset from {entry}")
            continue
        if not os.path.exists(os.path.join(rp, "sub_exp_1", "cliff_results.json")):
            print(f"  WARN: missing sub_exp_1/cliff_results.json for {entry}")
            continue
        result_dirs.setdefault(model, {})[matched_ds] = rp
        if model not in all_models:
            all_models.append(model)
        if matched_ds not in all_datasets:
            all_datasets.append(matched_ds)
    return result_dirs, all_models, all_datasets


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <batch_dir>")
        sys.exit(1)

    batch_dir = sys.argv[1]
    runs_dir = os.path.join(batch_dir, "runs")
    if not os.path.isdir(runs_dir):
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    result_dirs, all_models, all_datasets = discover_runs(runs_dir)
    if not result_dirs:
        print("ERROR: no completed runs found for aggregation")
        sys.exit(1)

    all_models = _ordered(all_models, MODEL_DISPLAY_ORDER)
    all_datasets = _ordered(all_datasets, DATASET_DISPLAY_ORDER)

    print(
        f"Aggregating {sum(len(v) for v in result_dirs.values())} runs across "
        f"{len(result_dirs)} models, {len(all_datasets)} datasets"
    )
    for m in all_models:
        for d in all_datasets:
            if d in result_dirs.get(m, {}):
                print(f"  {m}/{d}")

    print("\n[1/5] aggregate_per_model")
    aggregate_per_model(result_dirs, batch_dir)

    print("\n[2/5] aggregate_per_model_exp2")
    aggregate_per_model_exp2(result_dirs, batch_dir)

    print("\n[3/5] aggregate_semantic_all")
    aggregate_semantic_all(result_dirs, batch_dir)

    grid_dir = os.path.join(batch_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)

    print("\n[4/5] plot_model_dataset_grid")
    plot_model_dataset_grid(
        result_dirs,
        os.path.join(grid_dir, "pass_at_k_grid.png"),
        all_models, all_datasets,
    )

    print("\n[5/5] failure-only + exp2 methods grids")
    plot_failure_only_grid(result_dirs, all_models, all_datasets, grid_dir)
    plot_exp2_methods_grid(result_dirs, all_models, all_datasets, grid_dir)

    print(f"\nDone. Output: {batch_dir}")


if __name__ == "__main__":
    main()
