"""RQ2-1 slim aggregator — 4 focused outputs.

Reads cliff position data from a runs dir (produced by
scripts/_run_exp3_entropy_phaseB.py) and full per-token baseline data from
output/02_token_stats/ (produced by scripts/compute_path_token_stats.py),
then produces:

  1. greedy_ratio_table.csv                    — per-model cliff greedy ratio
                                                 vs all-token baseline greedy ratio
  2. entropy_density_combined.png              — pooled cliff vs baseline KDE
                                                 with E*_90 split + E*_95 reference
  3. entropy_density_per_model.png             — 5 per-model cliff KDE curves
  4. entropy_cliff_eq_greedy_table.csv         — 2×2 (H ≤/> E*_90 × eq/neq)

Boundary definitions (binary-entropy lower bounds on p_1 of the full
vocab distribution):

    E*_95 = H_b(0.95) = -0.95 ln 0.95 - 0.05 ln 0.05 ≈ 0.1985 nats
    E*_90 = H_b(0.90) = -0.90 ln 0.90 - 0.10 ln 0.10 ≈ 0.3251 nats

H ≤ E*_90 implies p_1 ≥ 0.90 under the tight extreme-case. We use partial
top-20 entropy (computed via src/analysis/entropy.py:compute_entropy_from_logprobs)
so this is a heuristic for sharp distributions (≈ exact in practice).
E*_90 is the SPLIT boundary used for stats. E*_95 is shown on plots as
a dotted reference only.
"""
import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================

# Canonical model display order: Qwen → Llama → gemma; larger first within family.
MODEL_ORDER = [
    "Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B",
    "Llama-3.1-8B-Instruct", "gemma-3-4b-it",
]
DATASET_ORDER = ["gsm1k", "math500", "aime25"]


# Canonical source of truth for taxonomy entropy boundary is
# `src.analysis.entropy`. Re-exported here so legacy callers that import
# `from src.analysis.exp3_entropy_aggregator import GREEDY_99_BOUND_NATS` keep
# working. Prefer importing from `src.analysis.entropy` in new code.
from src.analysis.entropy import (  # noqa: E402
    _binary_entropy_nats,
    GREEDY_PROB_THRESHOLD,
    GREEDY_BOUND_NATS,
    GREEDY_99_BOUND_NATS,
    GREEDY_95_BOUND_NATS,
    GREEDY_90_BOUND_NATS,
)


# ============================================================
# Helpers
# ============================================================

def _backfill_path_is_correct(cliff_logprobs, rollout_data_path):
    """Legacy backward-compat: if cliffs lack `path_is_correct`, look it up
    in the rollout data file. Mutates in place."""
    if not cliff_logprobs:
        return
    if all("path_is_correct" in c for c in cliff_logprobs):
        return
    if not rollout_data_path or not os.path.exists(rollout_data_path):
        return
    try:
        paths = json.load(open(rollout_data_path))
    except Exception:
        return
    pid_to_correct = {p["id"]: bool(p.get("is_correct", False)) for p in paths}
    for c in cliff_logprobs:
        if "path_is_correct" not in c:
            c["path_is_correct"] = pid_to_correct.get(c.get("path_id"), False)


def discover_runs_at(runs_dir: str) -> Dict[Tuple[str, str], Dict]:
    """Load cliff_logprobs.json for every <runs_dir>/<model>_<dataset>/.

    Returns: {(model, dataset): {"cliff_logprobs": [...]}}
    """
    if not os.path.isdir(runs_dir):
        print(f"  WARN: {runs_dir} not found")
        return {}

    out: Dict[Tuple[str, str], Dict] = {}
    for entry in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_path):
            continue

        cfg_path = os.path.join(run_path, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        model = cfg.get("model_short") or cfg.get("model") or "?"
        dataset = cfg.get("dataset", "?")

        cl_path = os.path.join(run_path, "cliff_logprobs.json")
        if not os.path.exists(cl_path):
            continue
        cliff_logprobs = json.load(open(cl_path))

        _backfill_path_is_correct(cliff_logprobs, cfg.get("rollout_data"))

        out[(model, dataset)] = {"cliff_logprobs": cliff_logprobs}
    return out


def discover_runs(output_dir: str) -> Dict[Tuple[str, str], Dict]:
    """Default runs discovery at <output_dir>/runs (backward-compat)."""
    return discover_runs_at(os.path.join(output_dir, "runs"))


def load_baseline_token_stats(
    models: List[str],
    datasets: List[str],
    base_dir: str = "output/02_token_stats",
) -> Dict[Tuple[str, str], Dict]:
    """Load per-token stats from inference_with_logprobs for (model, dataset) pairs.

    Returns: {(model, dataset): {"entropies": [...], "ranks": [...]}}
    Missing files are silently skipped.
    """
    out: Dict[Tuple[str, str], Dict] = {}
    for m in models:
        for d in datasets:
            f = os.path.join(base_dir, m, f"{d}_all_paths.json")
            if not os.path.exists(f):
                continue
            try:
                paths = json.load(open(f))
            except Exception as e:
                print(f"  WARN: failed to read {f}: {e}")
                continue
            ents: List[float] = []
            ranks: List[int] = []
            for p in paths:
                ents.extend(p.get("response_token_entropies", []))
                ranks.extend(p.get("response_token_ranks", []))
            if ents or ranks:
                out[(m, d)] = {"entropies": ents, "ranks": ranks}
    return out


def _ordered_models_from(runs_keys: List[Tuple[str, str]]) -> List[str]:
    seen = set(m for (m, _) in runs_keys)
    ordered = [m for m in MODEL_ORDER if m in seen]
    extras = sorted(m for m in seen if m not in MODEL_ORDER)
    return ordered + extras


def _ordered_models(runs: Dict[Tuple[str, str], Dict]) -> List[str]:
    return _ordered_models_from(list(runs.keys()))


def _merge_ordered(a: List[str], b: List[str]) -> List[str]:
    """Combine two model lists, preserving MODEL_ORDER."""
    seen = set(a) | set(b)
    ordered = [m for m in MODEL_ORDER if m in seen]
    extras = sorted(m for m in seen if m not in MODEL_ORDER)
    return ordered + extras


# ============================================================
# Output 1: Greedy ratio table (CSV only)
# ============================================================

def make_greedy_ratio_table(runs, baseline_data, output_dir):
    """CSV: per-model cliff greedy ratio vs all-token baseline greedy ratio.

    For each model (rows in canonical order), aggregates across all its
    (model, dataset) cells and reports two ratios:
      - cliff greedy ratio  = (# cliff tokens with is_cliff_eq_greedy) / N(cliff)
      - baseline greedy ratio = (# baseline tokens with rank==1) / N(baseline)
    """
    cliff_models = [m for (m, _) in runs.keys()]
    base_models = [m for (m, _) in baseline_data.keys()]
    models = _merge_ordered(cliff_models, base_models)

    rows = []
    for m in models:
        # Cliff aggregate
        cliffs = []
        for (mm, _), r in runs.items():
            if mm == m:
                cliffs.extend(r["cliff_logprobs"])
        n_cliff = len(cliffs)
        n_cliff_eq = sum(1 for c in cliffs if c.get("is_cliff_eq_greedy"))
        cliff_ratio = (n_cliff_eq / n_cliff) if n_cliff else None

        # Baseline aggregate (rank == 1 == greedy)
        n_base = 0
        n_base_top1 = 0
        for (mm, _), b in baseline_data.items():
            if mm == m:
                ranks = b["ranks"]
                n_base += len(ranks)
                n_base_top1 += sum(1 for r in ranks if r == 1)
        base_ratio = (n_base_top1 / n_base) if n_base else None

        rows.append({
            "Model": m,
            "Cliff_greedy_ratio": f"{cliff_ratio:.4f}" if cliff_ratio is not None else "-",
            "Baseline_greedy_ratio": f"{base_ratio:.4f}" if base_ratio is not None else "-",
        })

    csv_path = os.path.join(output_dir, "greedy_ratio_table.csv")
    fields = ["Model", "Cliff_greedy_ratio", "Baseline_greedy_ratio"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {csv_path}")

    # Terminal preview
    print("\n  Greedy ratio summary (cliff vs baseline):")
    print(f"  {'Model':<25} {'cliff_greedy':>14} {'base_greedy':>14}")
    for r in rows:
        print(f"  {r['Model']:<25} {r['Cliff_greedy_ratio']:>14} {r['Baseline_greedy_ratio']:>14}")


# ============================================================
# Output 2-1: Combined entropy density (cliff vs baseline)
# ============================================================

def _draw_boundary_lines(ax, *, primary=True, reference=True):
    """Draw E*_99 (dashed primary, p₁ ≥ 0.99) and E*_95 (dotted reference)."""
    if primary:
        ax.axvline(GREEDY_99_BOUND_NATS, color="black", linestyle="--",
                   linewidth=1.8,
                   label=f"E*₉₉ = {GREEDY_99_BOUND_NATS:.3f} nats (split, p₁ ≥ 0.99)")
    if reference:
        ax.axvline(GREEDY_95_BOUND_NATS, color="black", linestyle=":",
                   linewidth=1.3,
                   label=f"E*₉₅ = {GREEDY_95_BOUND_NATS:.3f} nats (reference)")


def _ecdf(values):
    """Empirical CDF as (x, y) step coordinates.

    For sorted unique data x_(1) ≤ ... ≤ x_(n),
        F̂(x) = (# values ≤ x) / n   ∈ [0, 1]

    Returns x_sorted (length n) and y = i/n (length n), suitable for
    ax.step(x, y, where='post'). Prepends a (0, 0) point so the curve
    starts at the origin for non-negative entropy data.
    """
    if not values:
        return np.array([0.0]), np.array([0.0])
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    y = np.arange(1, n + 1) / n
    # Prepend (0, 0) so the step starts from the x-axis at x=0
    x_out = np.concatenate(([0.0], arr))
    y_out = np.concatenate(([0.0], y))
    return x_out, y_out


def _ecdf_at(values, threshold):
    """F̂(threshold) = fraction of values ≤ threshold."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float((arr <= threshold).sum()) / len(arr)


def _annotate_ecdf_at_boundary(ax, cliff_ents, base_ents, boundary, label):
    """Mark cliff & baseline F̂(boundary) on the eCDF plot."""
    fc = _ecdf_at(cliff_ents, boundary)
    fb = _ecdf_at(base_ents, boundary) if base_ents else None
    # Cliff value
    ax.plot([boundary], [fc], marker="o", color="#E53935", markersize=7,
            zorder=10, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(
        f"  F_cliff({label}) = {fc:.3f}",
        xy=(boundary, fc), xytext=(8, -4), textcoords="offset points",
        fontsize=9, color="#E53935", fontweight="bold",
    )
    if fb is not None:
        ax.plot([boundary], [fb], marker="o", color="#546E7A", markersize=7,
                zorder=10, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(
            f"  F_base({label})  = {fb:.3f}",
            xy=(boundary, fb), xytext=(8, 4), textcoords="offset points",
            fontsize=9, color="#37474F", fontweight="bold",
        )
    return fc, fb


def make_entropy_density_combined(runs, baseline_data, output_dir):
    """Per-model entropy distributions (cliff vs all baseline tokens).

    Produces a single PNG with 5 subplots, one per model. Each subplot
    overlays two density-normalized histograms (integral = 1) of the
    entropy at token position: red = cliff tokens, gray = all baseline
    tokens. Subplot title is the model name.
    """
    models = _merge_ordered(
        [m for (m, _) in runs.keys()],
        [m for (m, _) in baseline_data.keys()],
    )

    per_model: Dict[str, Tuple[List[float], List[float]]] = {}
    for m in models:
        cliff_ents: List[float] = []
        for (mm, _), r in runs.items():
            if mm == m:
                cliff_ents.extend(float(c["entropy_at_t"]) for c in r["cliff_logprobs"])
        base_ents: List[float] = []
        for (mm, _), b in baseline_data.items():
            if mm == m:
                base_ents.extend(float(e) for e in b["entropies"])
        if cliff_ents or base_ents:
            per_model[m] = (cliff_ents, base_ents)

    if not per_model:
        print("  no entropy data; skipping entropy_density_combined")
        return

    from scipy.stats import gaussian_kde

    # Shared x-axis range across panels for visual comparability
    x_max = 0.0
    for cents, bents in per_model.values():
        if cents:
            x_max = max(x_max, max(cents))
        if bents:
            x_max = max(x_max, max(bents))
    x_max *= 1.05
    if x_max <= 0:
        x_max = 1.0

    rng = np.random.default_rng(42)
    SUBSAMPLE = 20_000  # cap baseline samples for KDE speed
    grid_x = np.linspace(0.0, x_max, 400)

    def _kde_curve(values):
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            return None
        if np.std(arr) < 1e-8:
            return None  # degenerate (all identical) — KDE fails
        if arr.size > SUBSAMPLE:
            arr = rng.choice(arr, size=SUBSAMPLE, replace=False)
        try:
            kde = gaussian_kde(arr, bw_method="scott")
            return kde(grid_x)
        except Exception as e:
            print(f"  WARN: KDE failed: {e}")
            return None

    # Pre-compute KDE curves once per model so we can reuse them across both
    # the linear-y and log-y figures.
    curves: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]] = {}
    for m, (cents, bents) in per_model.items():
        curves[m] = (
            _kde_curve(bents) if bents else None,
            _kde_curve(cents) if cents else None,
            len(bents),
            len(cents),
        )

    n_models = len(per_model)

    def _render(use_log: bool, out_path: str):
        fig, axes = plt.subplots(1, n_models,
                                  figsize=(4.2 * n_models, 5.6),
                                  sharey=False, squeeze=False)
        axes = axes[0]
        for ax, (m, (base_y, cliff_y, n_base, n_cliff)) in zip(axes, curves.items()):
            if base_y is not None:
                ax.fill_between(grid_x, base_y, color="#546E7A", alpha=0.30,
                                linewidth=0, zorder=2)
                ax.plot(grid_x, base_y, color="#37474F", linewidth=1.6,
                        label=f"All tokens (n={n_base:,})", zorder=3)
            if cliff_y is not None:
                ax.fill_between(grid_x, cliff_y, color="#E53935", alpha=0.30,
                                linewidth=0, zorder=4)
                ax.plot(grid_x, cliff_y, color="#C62828", linewidth=2.0,
                        label=f"Cliff tokens (n={n_cliff})", zorder=5)

            _draw_boundary_lines(ax)

            ax.set_xlim(0, x_max)
            if use_log:
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-3)
                ax.set_ylabel("Probability density (log)")
            else:
                ax.set_ylim(0, 8)
                ax.set_ylabel("Probability density")
            ax.set_xlabel("Entropy (nats, partial top-20)")
            ax.set_title(m, fontsize=11)
            ax.grid(True, which="both" if use_log else "major", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        scale_label = "log y" if use_log else "linear y"
        fig.suptitle(
            f"Cliff vs All Token Entropy Distributions (per model, {scale_label})",
            fontsize=13, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    _render(use_log=True,
            out_path=os.path.join(output_dir, "entropy_density_combined.png"))
    _render(use_log=False,
            out_path=os.path.join(output_dir, "entropy_density_combined_linear.png"))


# ============================================================
# Output 2-2: Per-model entropy density (cliff only, all on one plot)
# ============================================================

def make_entropy_density_per_model(runs, output_dir):
    """Single PNG: 5 cliff stepped histograms (stat=proportion), one per model.

    Each model's bin proportions sum to 1, so the curves are directly
    comparable regardless of cliff count.
    """
    models = _ordered_models(runs)

    # First pass: collect cliff_ents per model and find global x_max
    model_ents: Dict[str, List[float]] = {}
    for m in models:
        ents: List[float] = []
        for (mm, _), r in runs.items():
            if mm == m:
                ents.extend(float(c["entropy_at_t"]) for c in r["cliff_logprobs"])
        if ents:
            model_ents[m] = ents
    if not model_ents:
        print("  no per-model cliff data; skipping entropy_density_per_model")
        return

    x_max = max(max(e) for e in model_ents.values()) * 1.05
    b99 = GREEDY_99_BOUND_NATS

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(model_ents), 2)))

    for (m, ents), color in zip(model_ents.items(), cmap):
        cx, cy = _ecdf(ents)
        f_at = _ecdf_at(ents, b99)
        ax.step(cx, cy, where="post", color=color, linewidth=2.2,
                label=f"{m}  n={len(ents)}, F̂(E*₉₉)={f_at:.3f}")

    _draw_boundary_lines(ax)
    ax.set_xlabel("Cliff token entropy (nats, partial top-20)")
    ax.set_ylabel("F̂(x) = Pr[H ≤ x]")
    ax.set_title("Cliff Token Entropy — Empirical CDF per Model")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "entropy_density_per_model.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ============================================================
# Output 3: 2x2 entropy × cliff=greedy table
# ============================================================

def _partition_cliffs_4cell(cliffs, boundary):
    """Return 4 counts (low_eq, low_neq, high_eq, high_neq) for cliffs."""
    low_eq = low_neq = high_eq = high_neq = 0
    for c in cliffs:
        is_low = float(c["entropy_at_t"]) <= boundary
        is_eq = bool(c.get("is_cliff_eq_greedy"))
        if is_low and is_eq:
            low_eq += 1
        elif is_low:
            low_neq += 1
        elif is_eq:
            high_eq += 1
        else:
            high_neq += 1
    return low_eq, low_neq, high_eq, high_neq


def _partition_baseline_4cell(baseline_data, boundary):
    """Return 4 counts for baseline tokens, using paired (entropy, rank) arrays.

    "eq" (equivalent to cliff_eq_greedy) means rank == 1, i.e. the sampled
    baseline token IS the argmax. Iterate (entropy, rank) in parallel.
    """
    low_eq = low_neq = high_eq = high_neq = 0
    for b in baseline_data.values():
        ents = b["entropies"]
        ranks = b["ranks"]
        for e, r in zip(ents, ranks):
            is_low = float(e) <= boundary
            is_eq = (int(r) == 1)
            if is_low and is_eq:
                low_eq += 1
            elif is_low:
                low_neq += 1
            elif is_eq:
                high_eq += 1
            else:
                high_neq += 1
    return low_eq, low_neq, high_eq, high_neq


def make_entropy_cliff_eq_greedy_table(runs, output_dir):
    """2x2 CSV: rows = H ≤/> E*_90, cols = cliff==greedy / cliff!=greedy.

    Split boundary is E*_90 only (E*_95 is not used for stats).
    """
    cliffs = []
    for r in runs.values():
        cliffs.extend(r["cliff_logprobs"])
    if not cliffs:
        print("  no cliffs; skipping entropy_cliff_eq_greedy_table")
        return

    b99 = GREEDY_99_BOUND_NATS
    n_total = len(cliffs)
    low_eq, low_neq, high_eq, high_neq = _partition_cliffs_4cell(cliffs, b99)

    def fmt(n):
        return f"{n} ({n / n_total * 100:.1f}%)"

    csv_path = os.path.join(output_dir, "entropy_cliff_eq_greedy_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "cliff == greedy", "cliff != greedy", "row total"])
        w.writerow([
            f"H ≤ {b99:.3f} nats (low-H, p₁ ≥ 0.99)",
            fmt(low_eq), fmt(low_neq), fmt(low_eq + low_neq),
        ])
        w.writerow([
            f"H > {b99:.3f} nats (high-H, p₁ < 0.99)",
            fmt(high_eq), fmt(high_neq), fmt(high_eq + high_neq),
        ])
        w.writerow([
            "col total",
            fmt(low_eq + high_eq), fmt(low_neq + high_neq), fmt(n_total),
        ])
    print(f"  Saved {csv_path}")

    # Terminal preview
    print(f"\n  Entropy × cliff=greedy (split at E*₉₉ = {b99:.3f} nats):")
    print(f"  {'':36} {'cliff == greedy':>20} {'cliff != greedy':>20}")
    print(f"  {'H ≤ E*₉₉ (low-H)':36} {fmt(low_eq):>20} {fmt(low_neq):>20}")
    print(f"  {'H > E*₉₉ (high-H)':36} {fmt(high_eq):>20} {fmt(high_neq):>20}")


# ============================================================
# Output 5: Per-model taxonomy table (Deterministic Failure /
#           Ambiguous greedy / Sampling Slip)
# ============================================================

TAXONOMY_ROWS = [
    # (display name, key in counts dict)
    ("Deterministic Failure (low-H, greedy)",  "low_eq"),
    ("Ambiguous Greedy (high-H, greedy)",      "high_eq"),
    ("Sampling Slip (high-H, non-greedy)",     "high_neq"),
    ("(low-H, non-greedy)",                    "low_neq"),
]


def _per_model_4cell(runs, baseline_data, boundary):
    """Return {model: {"cliff": (le, ln, he, hn), "base": (le, ln, he, hn)}}."""
    out: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
    models = _merge_ordered(
        [m for (m, _) in runs.keys()],
        [m for (m, _) in baseline_data.keys()],
    )
    for m in models:
        cliffs_m = []
        for (mm, _), r in runs.items():
            if mm == m:
                cliffs_m.extend(r["cliff_logprobs"])
        base_m = {k: v for k, v in baseline_data.items() if k[0] == m}
        out[m] = {
            "cliff": _partition_cliffs_4cell(cliffs_m, boundary),
            "base":  _partition_baseline_4cell(base_m, boundary),
        }
    return out


def make_taxonomy_table(runs, baseline_data, output_dir):
    """Per-model taxonomy table (3 rows × per-model {cliff, baseline, ratio} columns).

    Saves:
      - taxonomy_table.csv : wide-form, headers <model>_cliff_pct /
                             <model>_baseline_pct / <model>_ratio
      - taxonomy_table.md  : same data with multi-level HTML headers

    Rows are the 3 named taxonomy cells:
      Deterministic Failure (low-H, cliff == greedy)
      Ambiguous greedy      (high-H, cliff == greedy)
      Sampling Slip         (high-H, cliff != greedy)
    The 4th cell (low-H, cliff != greedy) is omitted as it has no semantic
    interpretation in this taxonomy and is empirically negligible.

    `ratio` is `cliff_pct / baseline_pct` — how over/under-represented the cell
    is in cliff tokens relative to the baseline population.
    """
    if not runs and not baseline_data:
        print("  no data; skipping taxonomy_table")
        return

    b99 = GREEDY_99_BOUND_NATS
    per_model = _per_model_4cell(runs, baseline_data, b99)
    if not per_model:
        print("  no per-model data; skipping taxonomy_table")
        return

    # Pre-compute totals so we can convert counts → percentages of each model's
    # respective cliff or baseline population.
    totals: Dict[str, Tuple[int, int]] = {}
    for m, d in per_model.items():
        totals[m] = (sum(d["cliff"]), sum(d["base"]))

    def _cell_pct(model: str, key: str, source: str) -> Optional[float]:
        le, ln, he, hn = per_model[model][source]
        idx = {"low_eq": 0, "low_neq": 1, "high_eq": 2, "high_neq": 3}[key]
        n_total = totals[model][0 if source == "cliff" else 1]
        if n_total == 0:
            return None
        return (le, ln, he, hn)[idx] / n_total * 100.0

    def _cell_count(model: str, key: str, source: str) -> int:
        le, ln, he, hn = per_model[model][source]
        idx = {"low_eq": 0, "low_neq": 1, "high_eq": 2, "high_neq": 3}[key]
        return (le, ln, he, hn)[idx]

    models = list(per_model.keys())

    def _ratio(cp: Optional[float], bp: Optional[float]) -> Optional[float]:
        if cp is None or bp is None or bp == 0.0:
            return None
        return cp / bp

    # ---- CSV (wide form) ----
    csv_path = os.path.join(output_dir, "taxonomy_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["taxonomy"]
        for m in models:
            header.append(f"{m}_cliff_pct")
            header.append(f"{m}_baseline_pct")
            header.append(f"{m}_ratio")
        w.writerow(header)
        for label, key in TAXONOMY_ROWS:
            row = [label]
            for m in models:
                cp = _cell_pct(m, key, "cliff")
                bp = _cell_pct(m, key, "base")
                r = _ratio(cp, bp)
                row.append(f"{cp:.2f}%" if cp is not None else "—")
                row.append(f"{bp:.4f}%" if bp is not None else "—")
                row.append(f"{r:.4f}" if r is not None else "—")
            w.writerow(row)
        # n totals reference
        n_row = ["n (total)"]
        for m in models:
            nc, nb = totals[m]
            n_row.append(str(nc))
            n_row.append(str(nb))
            n_row.append("—")
        w.writerow(n_row)
    print(f"  Saved {csv_path}")

    # ---- Markdown (HTML for multi-level headers) ----
    md_path = os.path.join(output_dir, "taxonomy_table.md")
    sep = ' style="border-right:2px solid #333"'

    lines = []
    lines.append("# Cliff vs Baseline taxonomy (per model)\n")
    lines.append(
        "Each cliff token is classified into one of three semantic categories "
        f"using the entropy split E*₉₉ ≈ {b99:.3f} nats and the "
        "`cliff == greedy?` flag. The same partition is applied to every "
        "baseline token (rank=1 ⇔ greedy). Cell values are percentages within "
        "each model's own population (cliff cells sum within the cliff column; "
        "baseline cells within the baseline column). The `ratio` column is "
        "`cliff token / baseline`. The 4th cell (low-H, non-greedy) is omitted "
        "from this taxonomy.\n"
    )

    lines.append("## Percentages\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="2">Taxonomy</th>'
    for m in models:
        hdr1 += f'<th colspan="3" align="center"{sep}>{m}</th>'
    hdr1 += "</tr>"
    lines.append(hdr1)
    hdr2 = "<tr>"
    for _ in models:
        hdr2 += f'<th>cliff token</th><th>baseline</th><th{sep}>ratio</th>'
    hdr2 += "</tr>"
    lines.append(hdr2)
    for label, key in TAXONOMY_ROWS:
        row = f"<tr><td>{label}</td>"
        for m in models:
            cp = _cell_pct(m, key, "cliff")
            bp = _cell_pct(m, key, "base")
            r = _ratio(cp, bp)
            row += (f"<td>{cp:.1f}%</td>" if cp is not None else "<td>—</td>")
            row += (f"<td>{bp:.1f}%</td>" if bp is not None else "<td>—</td>")
            row += (f"<td{sep}>{r:.2f}</td>" if r is not None else f"<td{sep}>—</td>")
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    lines.append("## n (total tokens classified per model)\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="2">model</th>'
    hdr1 += '<th>cliff token</th><th>baseline</th></tr>'
    lines.append(hdr1)
    for m in models:
        nc, nb = totals[m]
        lines.append(f"<tr><td>{m}</td><td>{nc:,}</td><td>{nb:,}</td></tr>")
    lines.append("</table>\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


# ============================================================
# Main entrypoint
# ============================================================

def main(
    output_dir: str,
    runs_dir: Optional[str] = None,
    baseline_dir: str = "output/02_token_stats",
):
    if runs_dir is None:
        runs_dir = os.path.join(output_dir, "runs")

    print("=== exp3_entropy slim aggregation ===")
    print(f"  output:          {output_dir}")
    print(f"  cliff source:    {runs_dir}")
    print(f"  baseline source: {baseline_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Discover cliff runs
    runs = discover_runs_at(runs_dir)
    print(f"  discovered cliff runs: {len(runs)}")
    for (m, d), r in sorted(runs.items()):
        print(f"    {m}/{d}: {len(r['cliff_logprobs'])} cliffs")

    # Discover baseline cells — use all canonical models + observed runs
    base_model_set = set(MODEL_ORDER)
    if os.path.isdir(baseline_dir):
        base_model_set |= set(
            d for d in os.listdir(baseline_dir)
            if os.path.isdir(os.path.join(baseline_dir, d))
        )
    models = [m for m in MODEL_ORDER if m in base_model_set]
    models += sorted(m for m in base_model_set if m not in MODEL_ORDER)
    baseline_data = load_baseline_token_stats(models, DATASET_ORDER, baseline_dir)
    print(f"  discovered baseline cells: {len(baseline_data)}")
    for (m, d), b in sorted(baseline_data.items()):
        print(f"    {m}/{d}: {len(b['entropies']):,} tokens")
    print()

    if not runs and not baseline_data:
        print("ERROR: no data to aggregate")
        return

    make_greedy_ratio_table(runs, baseline_data, output_dir)
    make_entropy_density_combined(runs, baseline_data, output_dir)
    make_entropy_density_per_model(runs, output_dir)
    make_entropy_cliff_eq_greedy_table(runs, output_dir)
    make_taxonomy_table(runs, baseline_data, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2-1 slim aggregator")
    parser.add_argument("output_dir", help="Directory to write outputs into")
    parser.add_argument(
        "--runs_dir", default=None,
        help="Source dir containing <model>_<dataset>/cliff_logprobs.json. "
             "Default: <output_dir>/runs",
    )
    parser.add_argument(
        "--baseline_dir", default="output/02_token_stats",
        help="Source dir containing <model>/<dataset>_all_paths.json with "
             "per-token ranks/entropies (from compute_path_token_stats.py)",
    )
    args = parser.parse_args()
    main(args.output_dir, runs_dir=args.runs_dir, baseline_dir=args.baseline_dir)
