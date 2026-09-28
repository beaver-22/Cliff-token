"""
RQ1: Cliff token occurrence in correct vs. incorrect traces (Figure 2).

For each model, traces from all given datasets are pooled and split by
correctness. Cliff tokens are detected with the adaptive z-test threshold
(``find_all_cliff_tokens_statistical``).

Outputs (in ``output_dir``):
  avg_cliff_tokens.png / .json   average cliff tokens per trace
  cliff_stats_all_models.csv     per-model counts and averages

Usage (from scripts/run_exp1_occurrence.sh):
  run_multi_model_analysis(model_dataset_map, output_dir)
"""

import os
import csv
import json
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.detector import find_all_cliff_tokens_statistical


# ============================================================
# Style
# ============================================================
SUCCESS_COLOR = "#90CAF9"
FAILURE_COLOR = "#EF9A9A"
MODEL_ORDER = [
    "Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B",
    "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct",
    "gemma-3-4b-it",
]
MODEL_DISPLAY = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "gemma-3-4b-it": "Gemma-3-4B",
}


def _reorder_models(model_data: Dict) -> Dict:
    """Return a new dict ordered by MODEL_ORDER; unknown models appended in original order."""
    ordered = {m: model_data[m] for m in MODEL_ORDER if m in model_data}
    for m in model_data:
        if m not in ordered:
            ordered[m] = model_data[m]
    return ordered


def _display_model_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 9,
    })
    sns.set_style("whitegrid")


# ============================================================
# Data helpers
# ============================================================

def _compute_cliff_stats(paths: List[Dict]) -> Dict:
    num_cliffs = [
        len(find_all_cliff_tokens_statistical(
            p.get("all_position_scores", []),
            tokens=p.get("response_tokens"),
            token_ids=p.get("response_token_ids"),
        ))
        for p in paths
    ]
    n = len(num_cliffs)
    if n == 0:
        return {"num_paths": 0, "paths_with_cliff": 0, "total_cliffs": 0,
                "avg_cliffs_per_path": 0.0}
    return {
        "num_paths": n,
        "paths_with_cliff": sum(1 for c in num_cliffs if c > 0),
        "total_cliffs": sum(num_cliffs),
        "avg_cliffs_per_path": sum(num_cliffs) / n,
    }


def _collect_model_stats(model_data: Dict) -> Dict:
    stats = {}
    for model in model_data:
        all_s, all_f = [], []
        for ds in model_data[model]:
            all_s.extend(model_data[model][ds]["success_paths"])
            all_f.extend(model_data[model][ds]["failure_paths"])
        stats[model] = {
            "success": _compute_cliff_stats(all_s),
            "failure": _compute_cliff_stats(all_f),
        }
    return stats


# ============================================================
# Figure 2: average cliff tokens per trace
# ============================================================

def _plot_avg_cliff_tokens(stats: Dict, output_path: str):
    models = list(stats.keys())
    if not models:
        return

    _apply_style()
    fs = {"tick": 16, "label": 20, "legend": 16, "annot": 13}
    x = np.arange(len(models))
    bar_w = 0.35

    s_avg = [stats[m]["success"]["avg_cliffs_per_path"] for m in models]
    f_avg = [stats[m]["failure"]["avg_cliffs_per_path"] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(models)), 5))
    bars_s = ax.bar(
        x - bar_w / 2, s_avg, bar_w,
        color=SUCCESS_COLOR, edgecolor="#1565C0", linewidth=1.0,
        label="Correct trace", zorder=3,
    )
    bars_f = ax.bar(
        x + bar_w / 2, f_avg, bar_w,
        color=FAILURE_COLOR, edgecolor="#C62828", linewidth=1.0,
        label="Incorrect trace", zorder=3,
    )
    for bars, color in ((bars_s, "#1565C0"), (bars_f, "#C62828")):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom",
                fontsize=fs["annot"], color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_display_model_name(m) for m in models], fontsize=fs["tick"])
    ax.set_ylabel("Avg cliff tokens per trace", fontsize=fs["label"])
    y_max = max(f_avg + s_avg) if models else 0.0
    ax.set_ylim(0, max(0.1, y_max * 1.15))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=fs["legend"], loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    json_path = output_path.replace(".png", ".json")
    rates_avg = {
        m: {
            "success": stats[m]["success"]["avg_cliffs_per_path"],
            "failure": stats[m]["failure"]["avg_cliffs_per_path"],
            "n_success": stats[m]["success"]["num_paths"],
            "n_failure": stats[m]["failure"]["num_paths"],
        }
        for m in models
    }
    with open(json_path, "w") as f:
        json.dump(rates_avg, f, indent=2)
    print(f"  Saved: {json_path}")


def _write_summary_table(stats: Dict, output_path: str):
    fields = [
        "Model", "n_correct", "n_incorrect",
        "cliffs_correct", "cliffs_incorrect",
        "avg_cliffs/trace (correct)", "avg_cliffs/trace (incorrect)",
    ]
    rows = []
    for m, st in stats.items():
        s, f = st["success"], st["failure"]
        rows.append({
            "Model": m,
            "n_correct": s["num_paths"],
            "n_incorrect": f["num_paths"],
            "cliffs_correct": s["total_cliffs"],
            "cliffs_incorrect": f["total_cliffs"],
            "avg_cliffs/trace (correct)": f"{s['avg_cliffs_per_path']:.2f}",
            "avg_cliffs/trace (incorrect)": f"{f['avg_cliffs_per_path']:.2f}",
        })
    with open(output_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Model':<24} {'n_C':>5} {'n_I':>5} {'avg(C)':>8} {'avg(I)':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r['Model']:<24} {r['n_correct']:>5} {r['n_incorrect']:>5} "
              f"{r['avg_cliffs/trace (correct)']:>8} {r['avg_cliffs/trace (incorrect)']:>8}")
    print(f"\n  Saved: {output_path}")


# ============================================================
# Public API (scripts/run_exp1_occurrence.sh)
# ============================================================

def run_multi_model_analysis(model_dataset_map: Dict[str, Dict[str, str]], output_dir: str):
    """Cross-model cliff occurrence analysis.

    Args:
        model_dataset_map: {model_name: {dataset_name: path_to_all_paths_json}}
        output_dir: directory for combined outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    model_data = {}
    for model, datasets in model_dataset_map.items():
        model_data[model] = {}
        for ds, json_path in datasets.items():
            all_paths = json.load(open(json_path))
            s = [p for p in all_paths if p.get("is_correct")]
            f = [p for p in all_paths if not p.get("is_correct")]
            model_data[model][ds] = {"success_paths": s, "failure_paths": f}
            print(f"  {model}/{ds}: {len(s)} correct, {len(f)} incorrect")

    stats = _collect_model_stats(_reorder_models(model_data))

    print("\n--- Average cliff tokens per trace ---")
    _plot_avg_cliff_tokens(stats, os.path.join(output_dir, "avg_cliff_tokens.png"))
    _write_summary_table(stats, os.path.join(output_dir, "cliff_stats_all_models.csv"))

    print(f"\nAll outputs saved to: {output_dir}")
