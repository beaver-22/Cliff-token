"""Aggregation for RQ1 Cliff-del / Cliff-keep (Figure 3). CPU only.

Walks ``<batch_dir>/runs/<model>_<dataset>/cliff_del_keep/cliff_results.json``
and writes, for incorrect traces, pass@k of Cliff-del and Cliff-keep at the
first cliff token of each trace:

  <batch_dir>/pass_at_k_first_cliff_incorrect.csv
      columns: model, dataset, k, cliff_del, cliff_keep, n_traces

Usage:
    python3 scripts/_exp1_deletion_analyze.py <batch_dir>
"""
import csv
import json
import os
import sys

sys.path.insert(0, ".")

from src import config
from src.decoding.evaluator import (
    DATASET_DISPLAY_ORDER,
    MODEL_DISPLAY_ORDER,
    _ordered,
    first_cliff_pass_at_k,
)


def discover_runs(runs_dir):
    """Scan runs/ for `<model>_<dataset>` subdirs with cliff_results.json."""
    result_dirs = {}
    all_models, all_datasets = [], []
    for entry in sorted(os.listdir(runs_dir)):
        rp = os.path.join(runs_dir, entry)
        if not os.path.isdir(rp):
            continue
        matched_ds = next((d for d in DATASET_DISPLAY_ORDER if entry.endswith(f"_{d}")), None)
        if matched_ds is None:
            print(f"  WARN: cannot parse model/dataset from {entry}")
            continue
        model = entry[: -len(f"_{matched_ds}")]
        if not os.path.exists(os.path.join(rp, "cliff_del_keep", "cliff_results.json")):
            print(f"  WARN: missing cliff_del_keep/cliff_results.json for {entry}")
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
    k_values = list(config.PASS_K_VALUES)

    out_csv = os.path.join(batch_dir, "pass_at_k_first_cliff_incorrect.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "k", "cliff_del", "cliff_keep", "n_traces"])
        for ds in all_datasets:
            for model in all_models:
                run_dir = result_dirs.get(model, {}).get(ds)
                if run_dir is None:
                    continue
                with open(os.path.join(run_dir, "cliff_del_keep", "cliff_results.json")) as fh:
                    rows = json.load(fh)
                res = first_cliff_pass_at_k(rows, k_values=k_values)
                if res["n_traces"] == 0:
                    print(f"  {model:<24}/{ds:<8} n=0")
                    continue
                for k in k_values:
                    w.writerow([model, ds, k,
                                f"{res['cliff_del'][k]:.4f}",
                                f"{res['cliff_keep'][k]:.4f}",
                                res["n_traces"]])
                print(f"  {model:<24}/{ds:<8} n={res['n_traces']:<3} "
                      f"pass@1 del={res['cliff_del'][1]:.3f} keep={res['cliff_keep'][1]:.3f}")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
