"""
RQ1-1: Cliff Token Occurrence Analysis (Success vs Failure)

Outputs:
  1. Cliff token relative position distribution (per-model + combined)
  2. Case study curves: individual failure paths with max_cliff >= 0.5
  3. Threshold comparison: grouped bar chart (0.1, 0.2, 0.3, 0.4)
  4. Summary table: avg cliffs/path, avg cliff density (CSV + terminal)

Usage:
  # Single-model (from cli.py):
  run_experiment1_analysis(success_paths, failure_paths, output_dir)
  plot_sample_curves(success_paths, failure_paths, output_dir)

  # Multi-model (from run_exp1_occurrence.sh):
  run_multi_model_analysis(model_dataset_map, output_dir)
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.detector import (
    find_all_cliff_tokens,
    find_all_cliff_tokens_statistical,
    find_max_cliff,
)
from src import config

# Default cliff detection method (statistical z-test)
USE_STATISTICAL_CLIFF = True


# ============================================================
# Style
# ============================================================
SUCCESS_COLOR = "#90CAF9"
FAILURE_COLOR = "#EF9A9A"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
MODEL_ORDER = ["Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B", "Llama-3.1-8B-Instruct", "gemma-3-4b-it"]


def _reorder_models(model_data: Dict) -> Dict:
    """Return a new dict ordered by MODEL_ORDER; unknown models appended in original order."""
    ordered = {m: model_data[m] for m in MODEL_ORDER if m in model_data}
    for m in model_data:
        if m not in ordered:
            ordered[m] = model_data[m]
    return ordered

REGION_LABELS = ["Early\n(0–1/3)", "Mid\n(1/3–2/3)", "Late\n(2/3–1)"]
DATASET_HATCHES = {"gsm1k": "", "math500": "//", "aime25": "xx"}
DATASET_COLORS_S = {"gsm1k": "#64B5F6", "math500": "#2196F3", "aime25": "#0D47A1"}
DATASET_COLORS_F = {"gsm1k": "#EF9A9A", "math500": "#F44336", "aime25": "#B71C1C"}


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 9,
    })
    sns.set_style("whitegrid")


def _classify_region(rel_pos: float) -> int:
    if rel_pos < 1 / 3:
        return 0
    elif rel_pos < 2 / 3:
        return 1
    return 2


# ============================================================
# Data helpers
# ============================================================

def _extract_cliff_info(paths: List[Dict], threshold: float = config.DEFAULT_CLIFF_THRESHOLD,
                        use_statistical: Optional[bool] = None):
    if use_statistical is None:
        use_statistical = USE_STATISTICAL_CLIFF
    results = []
    for p in paths:
        scores = p.get("all_position_scores", [])
        total = p.get("total_tokens", len(scores) + 1)
        if use_statistical:
            cliffs = find_all_cliff_tokens_statistical(
                scores,
                tokens=p.get("response_tokens"),
                token_ids=p.get("response_token_ids"),
            )
        else:
            cliffs = find_all_cliff_tokens(
                scores, threshold,
                tokens=p.get("response_tokens"),
                token_ids=p.get("response_token_ids"),
            )
        rel_positions = [c.position / total for c in cliffs]
        results.append({
            "path_id": p["id"],
            "total_tokens": total,
            "is_correct": p.get("is_correct", False),
            "num_cliffs": len(cliffs),
            "rel_positions": rel_positions,
            "regions": [_classify_region(r) for r in rel_positions],
            "drop_magnitudes": [c.drop_magnitude for c in cliffs],
        })
    return results


def _compute_cliff_stats(paths: List[Dict], threshold: float = config.DEFAULT_CLIFF_THRESHOLD,
                         use_statistical: Optional[bool] = None):
    infos = _extract_cliff_info(paths, threshold, use_statistical=use_statistical)
    n = len(infos)
    if n == 0:
        return {"num_paths": 0, "paths_with_cliff": 0, "cliff_rate": 0,
                "avg_cliffs_per_path": 0, "avg_cliff_density": 0,
                "region_counts": [0, 0, 0], "all_rel_positions": []}

    paths_with_cliff = sum(1 for i in infos if i["num_cliffs"] > 0)
    total_cliffs = sum(i["num_cliffs"] for i in infos)
    densities = [i["num_cliffs"] / i["total_tokens"] if i["total_tokens"] > 0 else 0
                 for i in infos]
    region_counts = [0, 0, 0]
    all_rel = []
    for i in infos:
        for r in i["regions"]:
            region_counts[r] += 1
        all_rel.extend(i["rel_positions"])

    return {
        "num_paths": n,
        "paths_with_cliff": paths_with_cliff,
        "cliff_rate": paths_with_cliff / n * 100,
        "avg_cliffs_per_path": total_cliffs / n,
        "avg_cliff_density": float(np.mean(densities)),
        "region_counts": region_counts,
        "all_rel_positions": all_rel,
    }


# ============================================================
# Output 1: Position Distribution
# ============================================================

def _plot_position_histogram(success_rel, failure_rel, output_path, title_suffix=""):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)

    if success_rel:
        ax.hist(success_rel, bins=bins, alpha=0.4, color=SUCCESS_COLOR,
                label=f"Success (n={len(success_rel)})", density=True)
        if len(success_rel) >= 2:
            sns.kdeplot(success_rel, ax=ax, color=SUCCESS_COLOR, linewidth=2)
    if failure_rel:
        ax.hist(failure_rel, bins=bins, alpha=0.4, color=FAILURE_COLOR,
                label=f"Failure (n={len(failure_rel)})", density=True)
        if len(failure_rel) >= 2:
            sns.kdeplot(failure_rel, ax=ax, color=FAILURE_COLOR, linewidth=2)

    ax.axvline(1 / 3, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(2 / 3, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Relative Position (token position / total tokens)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cliff Token Position Distribution{title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_region_bar_chart(success_stats, failure_stats, output_path, title_suffix=""):
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    s_counts = np.array(success_stats["region_counts"], dtype=float)
    f_counts = np.array(failure_stats["region_counts"], dtype=float)
    s_total = s_counts.sum() if s_counts.sum() > 0 else 1
    f_total = f_counts.sum() if f_counts.sum() > 0 else 1

    region_names = ["Early (0–1/3)", "Mid (1/3–2/3)", "Late (2/3–1)"]
    # Light → dark shades for early/mid/late
    s_shades = ["#BBDEFB", "#64B5F6", "#1976D2"]
    f_shades = ["#FFCDD2", "#EF5350", "#B71C1C"]

    x = np.arange(2)
    w = 0.5
    labels = ["Success", "Failure"]
    totals = [s_total, f_total]
    counts_list = [s_counts, f_counts]
    shades_list = [s_shades, f_shades]

    for i in range(2):
        bottom = 0.0
        for r in range(3):
            pct = counts_list[i][r] / totals[i] * 100
            ax.bar(x[i], pct, w, bottom=bottom, color=shades_list[i][r],
                   edgecolor="white", linewidth=1.2,
                   label=region_names[r] if i == 0 else None)
            # Count label inside the segment
            if pct > 0:
                ax.text(x[i], bottom + pct / 2,
                        f"{int(counts_list[i][r])}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if r >= 1 else "black",
                        fontweight="bold")
            bottom += pct

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Cliff Tokens")
    ax.set_ylim(0, 105)
    ax.set_title(f"Cliff Token Region Distribution{title_suffix}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# Output 2: Case Study Curves
# ============================================================

def _plot_single_curve(path: Dict, output_path: str, model_name: str = "", dataset_name: str = ""):
    _apply_style()
    scores = path.get("all_position_scores", [])
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    positions = list(range(len(scores)))
    ax.plot(positions, scores, color="black", linewidth=0.8, alpha=0.8)
    ax.fill_between(positions, scores, alpha=0.15, color="steelblue")

    if USE_STATISTICAL_CLIFF:
        cliffs = find_all_cliff_tokens_statistical(scores)
    else:
        cliffs = find_all_cliff_tokens(scores, config.DEFAULT_CLIFF_THRESHOLD)
    max_cliff = find_max_cliff(scores)

    # Cliff tokens as red circles
    cliff_plotted = False
    for c in cliffs:
        idx = c.position - 1
        if 0 <= idx < len(scores):
            ax.plot(idx, scores[idx], "o", color=FAILURE_COLOR, markersize=7, alpha=0.7,
                    markeredgecolor="darkred", markeredgewidth=1,
                    label="Cliff Token" if not cliff_plotted else None)
            cliff_plotted = True

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Potential")

    # Title: model_dataset_problemid
    pid = path.get("id", "?")
    parts = [p for p in [model_name, dataset_name, pid] if p]
    title = "_".join(parts)
    drop_str = f"  |  Max Potential Drop = {max_cliff.drop_magnitude:.3f}" if max_cliff else ""
    ax.set_title(f"{title}{drop_str}", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sample_curves(success_paths: List[Dict], failure_paths: List[Dict], output_dir: str):
    """Generate individual curve plots for all failure paths with max_cliff >= 0.5."""
    case_dir = os.path.join(output_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)

    print("\n[Case Study] Finding failure paths with max cliff >= 0.5...")
    cases = []
    for p in failure_paths:
        scores = p.get("all_position_scores", [])
        mc = find_max_cliff(scores)
        if mc and mc.drop_magnitude >= 0.5:
            cases.append((p, mc))

    if not cases:
        print("  No failure paths with max cliff >= 0.5 found.")
        return

    print(f"  Found {len(cases)} case(s). Generating plots...")
    manifest = []
    for p, mc in cases:
        model = p.get("_model", "")
        dataset = p.get("_dataset", "")
        safe_id = p["id"].replace("/", "_").replace(" ", "_")
        parts = [x for x in [model, dataset, safe_id] if x]
        fname = f"case_{'_'.join(parts)}.png"
        fpath = os.path.join(case_dir, fname)
        _plot_single_curve(p, fpath, model_name=model, dataset_name=dataset)
        manifest.append({
            "path_id": p["id"],
            "total_tokens": p.get("total_tokens"),
            "max_cliff_magnitude": round(mc.drop_magnitude, 4),
            "max_cliff_position": mc.position,
            "filename": fname,
        })
        print(f"    {fname} (drop={mc.drop_magnitude:.3f} at pos={mc.position})")

    manifest_path = os.path.join(case_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"  Manifest: {manifest_path}")


# ============================================================
# Output 3: Threshold Comparison
# ============================================================

def _plot_threshold_comparison(model_data, output_path):
    """Single unified bar chart: threshold × model, success behind failure.

    Per model: 4 threshold groups (0.1, 0.2, 0.3, 0.4).
    Each threshold has 2 bars (failure in front, success behind — overlapping).
    All datasets aggregated per model.
    """
    _apply_style()
    models = list(model_data.keys())
    n_models = len(models)
    n_thresholds = len(THRESHOLDS)

    # Precompute rates: {model: {threshold: {success: rate, failure: rate}}}
    rates = {}
    for model in models:
        rates[model] = {}
        all_s, all_f = [], []
        for ds in model_data[model]:
            all_s.extend(model_data[model][ds].get("success_paths", []))
            all_f.extend(model_data[model][ds].get("failure_paths", []))
        for th in THRESHOLDS:
            # Threshold comparison always uses legacy (fixed-threshold) cliff detection
            s_stats = _compute_cliff_stats(all_s, th, use_statistical=False)
            f_stats = _compute_cliff_stats(all_f, th, use_statistical=False)
            rates[model][th] = {"success": s_stats["cliff_rate"], "failure": f_stats["cliff_rate"]}

    # Layout: each model gets a group, within group 4 threshold pairs
    fig, ax = plt.subplots(figsize=(max(3 * n_models, 10), 6))

    group_width = 0.8
    bar_width = group_width / n_thresholds
    th_colors = ["#A5D6A7", "#66BB6A", "#388E3C", "#1B5E20"]  # light→dark green for thresholds

    for m_idx, model in enumerate(models):
        for t_idx, th in enumerate(THRESHOLDS):
            x = m_idx + (t_idx - n_thresholds / 2 + 0.5) * bar_width
            s_rate = rates[model][th]["success"]
            f_rate = rates[model][th]["failure"]

            # Failure behind (full width, drawn first)
            ax.bar(x, f_rate, bar_width * 0.9, color=FAILURE_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=2,
                   label="Failure" if m_idx == 0 and t_idx == 0 else None)
            # Success in front (slightly narrower, drawn second)
            ax.bar(x, s_rate, bar_width * 0.78, color=SUCCESS_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label="Success" if m_idx == 0 and t_idx == 0 else None)

            # Threshold label on top
            top_val = max(s_rate, f_rate)
            ax.text(x, top_val + 1.5, f"{th}", ha="center", va="bottom", fontsize=7,
                    color="gray")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("% Paths with ≥1 Cliff Token")
    ax.set_title("Cliff Token Occurrence Rate by Absolute Threshold (all datasets combined)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", fontsize=10)

    # Add threshold legend explanation
    ax.text(0.98, 0.97, "Numbers above bars = threshold",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Save rates as JSON
    json_path = output_path.replace(".png", ".json")
    rates_json = {
        model: {str(th): vals for th, vals in ths.items()}
        for model, ths in rates.items()
    }
    with open(json_path, "w") as f:
        json.dump(rates_json, f, indent=2)
    print(f"  Saved: {json_path}")


# ============================================================
# Output 3 (statistical): Threshold Comparison — statistical cliff
# ============================================================

def _plot_threshold_comparison_statistical(model_data, output_path):
    """Statistical-cliff sibling of `_plot_threshold_comparison`.

    One group per model, one success/failure bar pair per group (no inner
    threshold groups, since statistical detection has no threshold knob).
    Mirrors the styling of `_plot_threshold_comparison`.
    """
    _apply_style()
    models = list(model_data.keys())
    n_models = len(models)

    rates = {}
    for model in models:
        all_s, all_f = [], []
        for ds in model_data[model]:
            all_s.extend(model_data[model][ds].get("success_paths", []))
            all_f.extend(model_data[model][ds].get("failure_paths", []))
        s_stats = _compute_cliff_stats(all_s, use_statistical=True)
        f_stats = _compute_cliff_stats(all_f, use_statistical=True)
        rates[model] = {
            "success": s_stats["cliff_rate"],
            "failure": f_stats["cliff_rate"],
            "n_success": s_stats["num_paths"],
            "n_failure": f_stats["num_paths"],
        }

    fig, ax = plt.subplots(figsize=(max(1.8 * n_models, 7), 6))
    bar_width = 0.5

    for m_idx, model in enumerate(models):
        x = m_idx
        s_rate = rates[model]["success"]
        f_rate = rates[model]["failure"]

        # Failure behind (full width, drawn first)
        ax.bar(x, f_rate, bar_width * 0.9, color=FAILURE_COLOR,
               edgecolor="white", linewidth=0.5, zorder=2,
               label="Failure" if m_idx == 0 else None)
        # Success in front (slightly narrower, drawn second)
        ax.bar(x, s_rate, bar_width * 0.78, color=SUCCESS_COLOR,
               edgecolor="white", linewidth=0.5, zorder=3,
               label="Success" if m_idx == 0 else None)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("% Paths with ≥1 Cliff Token")
    ax.set_title("Cliff Token Occurrence Rate by Statistical Threshold (all datasets combined)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    json_path = output_path.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(rates, f, indent=2)
    print(f"  Saved: {json_path}")


# ============================================================
# Output 4: Summary Table
# ============================================================

def _load_fullset_accuracy():
    """Load fullset accuracy from model_performance_fullset.json if available.

    Searches multiple candidate directories (newest first).
    """
    candidates = [
        os.path.join("output", "model_performance_test", "paper_max_length", "model_performance_fullset.json"),
        os.path.join("output", "inference_fullset", "model_performance_fullset.json"),
    ]
    fullset_path = None
    for c in candidates:
        if os.path.exists(c):
            fullset_path = c
            break
    if fullset_path is None:
        return {}
    data = json.load(open(fullset_path))
    acc_map = {}
    for r in data:
        model = r["Model"]
        accs = []
        for key in r:
            if key != "Model" and not key.endswith("_n") and not key.endswith("_n_problems"):
                val = r[key]
                if val != "-":
                    accs.append((key, val))
        acc_map[model] = {k: v for k, v in accs}
    return acc_map


def _generate_summary_table(model_data, output_path):
    """CSV + terminal table. Aggregates all datasets per model. Includes fullset accuracy."""
    fullset_acc = _load_fullset_accuracy()

    rows = []
    for model in model_data:
        all_success, all_failure = [], []
        for ds in model_data[model]:
            all_success.extend(model_data[model][ds].get("success_paths", []))
            all_failure.extend(model_data[model][ds].get("failure_paths", []))

        s = _compute_cliff_stats(all_success)
        f = _compute_cliff_stats(all_failure)

        # Fullset accuracy (from inference_fullset)
        model_acc = fullset_acc.get(model, {})
        gsm1k_acc = str(model_acc.get("GSM1K", "-"))
        math500_acc = str(model_acc.get("MATH500", "-"))
        aime_val = None
        aime_se = None
        for k, v in model_acc.items():
            if k.endswith("_SE") and k.startswith("AIME"):
                aime_se = v
            elif k.startswith("AIME"):
                aime_val = v
        if aime_val is None:
            aime_acc = "-"
        elif aime_se is not None:
            aime_acc = f"{aime_val}({aime_se})"
        else:
            aime_acc = str(aime_val)

        rows.append({
            "Model": model,
            "GSM1K(accuracy)": gsm1k_acc,
            "MATH-500(accuracy)": math500_acc,
            "AIME 2025(avg@64)": aime_acc,
            "avg_cliffs/path (S)": f"{s['avg_cliffs_per_path']:.2f}",
            "avg_cliffs/path (F)": f"{f['avg_cliffs_per_path']:.2f}",
            "n_success": s["num_paths"],
            "n_failure": f["num_paths"],
        })

    with open(output_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    header = "Performance"
    print(f"\n{'Model':<28} {'GSM1K':>8} {'MATH-500':>9} {'AIME 2025':>12} {'cliffs/path(S)':>15} {'cliffs/path(F)':>15} {'n_S':>5} {'n_F':>5}")
    print(f"{'':28} {'(acc)':>8} {'(acc)':>9} {'(avg@64)':>12}")
    print("-" * 105)
    for r in rows:
        print(f"{r['Model']:<28} {r['GSM1K(accuracy)']:>8} {r['MATH-500(accuracy)']:>9} {r['AIME 2025(avg@64)']:>12} "
              f"{r['avg_cliffs/path (S)']:>15} {r['avg_cliffs/path (F)']:>15} {r['n_success']:>5} {r['n_failure']:>5}")
    print(f"\n  Saved: {output_path}")


# ============================================================
# Public API: Single-model (compatible with cli.py)
# ============================================================

def run_experiment1_analysis(success_paths: List[Dict], failure_paths: List[Dict], output_dir: str):
    """RQ1-1 single-model analysis. Called by cli.py cmd_experiment()."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Cliff Token Position Analysis ---")
    s_stats = _compute_cliff_stats(success_paths)
    f_stats = _compute_cliff_stats(failure_paths)

    print(f"  Success: {s_stats['num_paths']} paths, "
          f"{s_stats['paths_with_cliff']} with cliff ({s_stats['cliff_rate']:.1f}%), "
          f"avg {s_stats['avg_cliffs_per_path']:.2f} cliffs/path")
    print(f"  Failure: {f_stats['num_paths']} paths, "
          f"{f_stats['paths_with_cliff']} with cliff ({f_stats['cliff_rate']:.1f}%), "
          f"avg {f_stats['avg_cliffs_per_path']:.2f} cliffs/path")

    _plot_position_histogram(
        s_stats["all_rel_positions"], f_stats["all_rel_positions"],
        os.path.join(output_dir, "cliff_position_histogram.png"),
    )
    _plot_region_bar_chart(
        s_stats, f_stats,
        os.path.join(output_dir, "cliff_position_regions.png"),
    )


# ============================================================
# Public API: Multi-model (run_exp1_occurrence.sh)
# ============================================================

def run_multi_model_analysis(model_dataset_map: Dict[str, Dict[str, str]], output_dir: str):
    """Cross-model aggregation analysis.

    Args:
        model_dataset_map: {model_name: {dataset_name: path_to_all_paths_json}}
        output_dir: directory for combined outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    _apply_style()

    # Load all data
    print("Loading data...")
    model_data = {}
    for model, datasets in model_dataset_map.items():
        model_data[model] = {}
        for ds, json_path in datasets.items():
            all_paths = json.load(open(json_path))
            s = [p for p in all_paths if p.get("is_correct")]
            f = [p for p in all_paths if not p.get("is_correct")]
            model_data[model][ds] = {"success_paths": s, "failure_paths": f}
            print(f"  {model}/{ds}: {len(s)} success, {len(f)} failure")

    model_data = _reorder_models(model_data)

    # Output 1: Position Distribution — single combined region bar + table
    print("\n--- Output 1: Position Distribution ---")
    all_s = [p for m in model_data for ds in model_data[m] for p in model_data[m][ds]["success_paths"]]
    all_f = [p for m in model_data for ds in model_data[m] for p in model_data[m][ds]["failure_paths"]]
    _plot_region_bar_chart(
        _compute_cliff_stats(all_s), _compute_cliff_stats(all_f),
        os.path.join(output_dir, "cliff_position_regions.png"),
        title_suffix=" (All Models & Datasets)",
    )

    # Output 2: Case studies
    print("\n--- Output 2: Case Studies ---")
    all_failure = [p for m in model_data for ds in model_data[m]
                   for p in model_data[m][ds]["failure_paths"]]
    plot_sample_curves([], all_failure, output_dir)

    # Output 3: Threshold comparison
    print("\n--- Output 3: Threshold Comparison ---")
    _plot_threshold_comparison(model_data, os.path.join(output_dir, "threshold_comparison.png"))

    # Output 4: Summary table
    print("\n--- Output 4: Summary Table ---")
    _generate_summary_table(model_data, os.path.join(output_dir, "cliff_stats_all_models.csv"))

    # Output 5: Threshold comparison (statistical)
    print("\n--- Output 5: Threshold Comparison (statistical) ---")
    _plot_threshold_comparison_statistical(
        model_data, os.path.join(output_dir, "threshold_comparison_statistical.png")
    )

    print(f"\nAll outputs saved to: {output_dir}")
