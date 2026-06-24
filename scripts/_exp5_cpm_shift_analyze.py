"""exp5_cpm_shift Phase 2 — merge per-combo CSVs and render CPM/rank/prob shift scatters.

Usage:
    python3 scripts/_exp5_cpm_shift_analyze.py <batch_dir>
"""
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAXONOMY_COLORS = {
    "deterministic": "#1E88E5",  # blue
    "uncertain":     "#FB8C00",  # orange
    "sampled_off":   "#E53935",  # red
}
TAXONOMY_ORDER = ["deterministic", "uncertain", "sampled_off"]
TAXONOMY_LABELS = {
    "deterministic": "Deterministic",
    "uncertain":     "Uncertain",
    "sampled_off":   "Sampled-off",
}
DIRECTION_LABELS = {
    "smalltobig": "Small→Big  (0.6B→8B)",
    "bigtosmall": "Big→Small  (8B→0.6B)",
}
# Drawing order (back-to-front) for scatter overlays — largest population first,
# deterministic last so the (1, 1) cluster is always visible on top.
TAXONOMY_DRAW_ORDER = ["sampled_off", "uncertain", "deterministic"]


def _read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _cast(row: Dict, float_cols=(), int_cols=()) -> Dict:
    for c in float_cols:
        if c in row and row[c] != "":
            row[c] = float(row[c])
    for c in int_cols:
        if c in row and row[c] != "":
            row[c] = int(row[c])
    return row


def _merge_runs(batch_dir: str):
    runs_dir = os.path.join(batch_dir, "runs")
    if not os.path.isdir(runs_dir):
        print(f"ERROR: runs dir not found: {runs_dir}")
        sys.exit(1)

    per_cliff_rows: List[Dict] = []
    per_token_rows: List[Dict] = []
    for entry in sorted(os.listdir(runs_dir)):
        combo_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(combo_dir):
            continue
        for r in _read_csv(os.path.join(combo_dir, "per_cliff.csv")):
            _cast(r,
                  float_cols=("cpm_source", "cpm_eval", "delta_cpm"),
                  int_cols=("cliff_position", "n_cliff_tokens"))
            per_cliff_rows.append(r)
        for r in _read_csv(os.path.join(combo_dir, "per_token.csv")):
            _cast(r,
                  float_cols=("source_prob", "source_logprob",
                              "eval_prob", "eval_logprob", "delta_prob"),
                  int_cols=("cliff_position", "cliff_token_id",
                            "source_rank", "eval_rank", "delta_rank"))
            per_token_rows.append(r)
    return per_cliff_rows, per_token_rows


def _write_merged_csv(rows: List[Dict], path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _scatter_cpm(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    for tax in TAXONOMY_DRAW_ORDER:
        xs = [r["cpm_source"] for r in rows if r["taxonomy_type"] == tax]
        ys = [r["cpm_eval"] for r in rows if r["taxonomy_type"] == tax]
        if not xs:
            continue
        ax.scatter(xs, ys, c=TAXONOMY_COLORS[tax], s=28, alpha=0.65,
                   edgecolors="black", linewidths=0.3,
                   label=f"{tax}  (n={len(xs)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray",
            label="y = x", zorder=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("cpm_source  (sum of source cliff-token probs, top-20)")
    ax.set_ylabel("cpm_eval    (eval model top-20 mass on source cliff tokens)")
    ax.set_title("Cross-model CPM shift — all (source, eval) pairs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _scatter_cpm_facet(rows: List[Dict], out_path: str):
    # Group by (source, eval)
    pairs: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        key = (r["model_source"], r["model_eval"])
        pairs.setdefault(key, []).append(r)
    if not pairs:
        return
    n = len(pairs)
    cols = min(n, 3)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols,
                              figsize=(4.5 * cols, 4.5 * rows_n),
                              squeeze=False)
    for i, (key, group) in enumerate(sorted(pairs.items())):
        ax = axes[i // cols][i % cols]
        for tax in TAXONOMY_DRAW_ORDER:
            xs = [r["cpm_source"] for r in group if r["taxonomy_type"] == tax]
            ys = [r["cpm_eval"] for r in group if r["taxonomy_type"] == tax]
            if not xs:
                continue
            ax.scatter(xs, ys, c=TAXONOMY_COLORS[tax], s=22, alpha=0.7,
                       edgecolors="black", linewidths=0.25, label=tax)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{key[0]} → {key[1]}", fontsize=11)
        ax.set_xlabel("cpm_source")
        ax.set_ylabel("cpm_eval")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    # Hide unused cells
    total = rows_n * cols
    for j in range(n, total):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle("Cross-model CPM shift by (source → eval) pair", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _scatter_rank(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    for tax in TAXONOMY_DRAW_ORDER:
        xs = [r["source_rank"] for r in rows if r["taxonomy_type"] == tax]
        ys = [r["eval_rank"] for r in rows if r["taxonomy_type"] == tax]
        if not xs:
            continue
        # Small jitter on both axes so overlapping points are visible
        import random
        rnd = random.Random(42 + TAXONOMY_ORDER.index(tax))
        xj = [x + rnd.uniform(-0.15, 0.15) for x in xs]
        yj = [y + rnd.uniform(-0.15, 0.15) for y in ys]
        ax.scatter(xj, yj, c=TAXONOMY_COLORS[tax], s=24, alpha=0.55,
                   edgecolors="black", linewidths=0.25,
                   label=f"{tax}  (n={len(xs)})")
    ax.plot([1, 21], [1, 21], linestyle="--", linewidth=1.0, color="gray",
            label="y = x", zorder=0)
    ax.axhline(21, linestyle=":", linewidth=1.0, color="#455A64",
               label="eval_rank = 21 (outside top-20)", zorder=0)
    ax.set_xlim(0.5, 21.5)
    ax.set_ylim(0.5, 22)
    ax.set_xlabel("source_rank  (cliff candidate rank in source top-20)")
    ax.set_ylabel("eval_rank    (same token's rank in eval top-20; 21 = out)")
    ax.set_title("Cross-model rank shift of cliff tokens (per-token)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _scatter_prob(rows: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    for tax in TAXONOMY_DRAW_ORDER:
        xs = [r["source_prob"] for r in rows if r["taxonomy_type"] == tax]
        ys = [r["eval_prob"] for r in rows if r["taxonomy_type"] == tax]
        if not xs:
            continue
        ax.scatter(xs, ys, c=TAXONOMY_COLORS[tax], s=24, alpha=0.6,
                   edgecolors="black", linewidths=0.25,
                   label=f"{tax}  (n={len(xs)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray",
            label="y = x", zorder=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("source_prob  (cliff candidate prob in source top-20)")
    ax.set_ylabel("eval_prob    (same token's prob in eval top-20; 0 if out)")
    ax.set_title("Cross-model probability shift of cliff tokens (per-token)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-direction plots (require smalltobig + bigtosmall data)
# ─────────────────────────────────────────────────────────────────────────────

def _load_selected_cliff_tokens(exp4_candidates_dir: str) -> Dict[Tuple, int]:
    """Load selected cliff token IDs from exp4_candidates candidate_results.csv."""
    selected: Dict[Tuple, int] = {}
    cands_csv = os.path.join(exp4_candidates_dir, "candidate_results.csv")
    if not os.path.exists(cands_csv):
        return selected
    with open(cands_csv) as f:
        for row in csv.DictReader(f):
            if row.get("is_candidate_selected_cliff", "").lower() in (
                "true", "1", "t", "yes"
            ):
                key = (row["path_id"], int(row["cliff_position"]))
                selected[key] = int(row["candidate_token_id"])
    return selected


def _find_exp4_candidates_dirs(batch_dir: str) -> Dict[str, str]:
    """Auto-detect exp4_candidates output dirs from batch_dir's sibling exp4_candidates/ folder."""
    exp_root = os.path.dirname(batch_dir)
    # Walk up if batch_dir is like .../exp5_cpm_shift/0410_combined
    if os.path.basename(exp_root).startswith("exp5_cpm_shift"):
        exp_root = os.path.dirname(exp_root)
    elif not os.path.isdir(os.path.join(exp_root, "exp4_candidates")):
        exp_root = os.path.dirname(exp_root)

    exp4_candidates_root = os.path.join(exp_root, "exp4_candidates")
    if not os.path.isdir(exp4_candidates_root):
        return {}

    dirs = {}
    for entry in os.listdir(exp4_candidates_root):
        full = os.path.join(exp4_candidates_root, entry)
        if os.path.isdir(full) and os.path.exists(
            os.path.join(full, "candidate_results.csv")
        ):
            dirs[entry] = full
    return dirs


def _load_directional_data(batch_dir: str):
    """Load smalltobig/bigtosmall directional data for cross-direction plots.

    Looks for sibling `<prefix>_smalltobig` / `<prefix>_bigtosmall` dirs
    (fallback: legacy `<prefix>_slow` / `<prefix>_fast`) relative to batch_dir.
    Returns dict with "smalltobig"/"bigtosmall" keys, each containing cliff/token
    rows, or None if directional data is unavailable.
    """
    parent = os.path.dirname(batch_dir)
    prefix = os.path.basename(batch_dir).split("_")[0]  # e.g. "0410"

    def _pick(primary: str, legacy: str) -> str:
        p = os.path.join(parent, f"{prefix}_{primary}")
        if os.path.isdir(p):
            return p
        return os.path.join(parent, f"{prefix}_{legacy}")

    smalltobig_dir = _pick("smalltobig", "slow")
    bigtosmall_dir = _pick("bigtosmall", "fast")

    if not os.path.isdir(smalltobig_dir) or not os.path.isdir(bigtosmall_dir):
        return None

    # Find exp4_candidates selected cliff tokens
    exp4_candidates_dirs = _find_exp4_candidates_dirs(batch_dir)
    selected_map: Dict[str, Dict[Tuple, int]] = {}
    for name, path in exp4_candidates_dirs.items():
        sel = _load_selected_cliff_tokens(path)
        if "0.6B" in name:
            selected_map["smalltobig"] = sel
        elif "8B" in name:
            selected_map["bigtosmall"] = sel

    data = {}
    for key, sub_dir in [("smalltobig", smalltobig_dir),
                          ("bigtosmall", bigtosmall_dir)]:
        cliff_csv = os.path.join(sub_dir, "results_per_cliff.csv")
        token_csv = os.path.join(sub_dir, "results_per_token.csv")

        cliff_rows = []
        for r in _read_csv(cliff_csv):
            _cast(r,
                  float_cols=("cpm_source", "cpm_eval", "delta_cpm"),
                  int_cols=("cliff_position", "n_cliff_tokens"))
            cliff_rows.append(r)

        all_token_rows = []
        for r in _read_csv(token_csv):
            _cast(r,
                  float_cols=("source_prob", "source_logprob",
                              "eval_prob", "eval_logprob", "delta_prob"),
                  int_cols=("cliff_position", "cliff_token_id",
                            "source_rank", "eval_rank", "delta_rank"))
            all_token_rows.append(r)

        # Filter to selected cliff tokens only, fix tie-induced rank anomalies
        sel = selected_map.get(key, {})
        token_rows = []
        for r in all_token_rows:
            rkey = (r["data_idx"], r["cliff_position"])
            if rkey in sel and sel[rkey] == r["cliff_token_id"]:
                tax = r["taxonomy_type"]
                if tax in ("uncertain", "deterministic") and r["source_rank"] != 1:
                    r["source_rank"] = 1
                elif tax == "sampled_off" and r["source_rank"] == 1:
                    r["source_rank"] = 2
                token_rows.append(r)

        data[key] = {"cliff": cliff_rows, "token": token_rows}

    return data


def _plot_violin(data: Dict, figures_dir: str):
    """Plot 1: Delta CPM violin by taxonomy × direction."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ti, tax in enumerate(TAXONOMY_ORDER):
        ax = axes[ti]
        parts_data = []
        for direction in ["smalltobig", "bigtosmall"]:
            vals = [r["delta_cpm"] for r in data[direction]["cliff"]
                    if r["taxonomy_type"] == tax]
            parts_data.append(vals)

        if all(len(d) > 0 for d in parts_data):
            vp = ax.violinplot(parts_data, positions=[0, 1],
                               showmeans=True, showmedians=True,
                               showextrema=False)
            for i, body in enumerate(vp["bodies"]):
                color = "#5C6BC0" if i == 0 else "#EF5350"
                body.set_facecolor(color)
                body.set_alpha(0.5)
            vp["cmeans"].set_color("black")
            vp["cmedians"].set_color("gray")
            vp["cmedians"].set_linestyle("--")

            rng = np.random.RandomState(42)
            for i, vals in enumerate(parts_data):
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           s=12, alpha=0.4,
                           c="#5C6BC0" if i == 0 else "#EF5350",
                           edgecolors="none", zorder=3)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Small→Big", "Big→Small"], fontsize=9)
        n_s2b = len([r for r in data["smalltobig"]["cliff"]
                     if r["taxonomy_type"] == tax])
        n_b2s = len([r for r in data["bigtosmall"]["cliff"]
                     if r["taxonomy_type"] == tax])
        ax.set_title(f"{TAXONOMY_LABELS[tax]}\n(n={n_s2b}, {n_b2s})",
                     fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Δ CPM  (eval − source)", fontsize=11)
    fig.suptitle("Delta CPM Distribution by Taxonomy × Direction",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(figures_dir, "delta_cpm_violin.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_rank_heatmap(data: Dict, figures_dir: str):
    """Plot 5: Per-taxonomy rank shift heatmap (selected cliff token only)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap_red = LinearSegmentedColormap.from_list(
        "cliff_red", ["#FFFFFF", "#FFCDD2", "#EF5350", "#B71C1C"])

    for tax in TAXONOMY_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        for di, direction in enumerate(["smalltobig", "bigtosmall"]):
            ax = axes[di]
            rows_filt = [r for r in data[direction]["token"]
                         if r["taxonomy_type"] == tax]
            source_ranks = [r["source_rank"] for r in rows_filt]
            eval_ranks = [r["eval_rank"] for r in rows_filt]

            bins_x = np.arange(0.5, 21.5, 1)
            bins_y = np.arange(0.5, 22.5, 1)
            h, _, _ = np.histogram2d(source_ranks, eval_ranks,
                                     bins=[bins_x, bins_y])
            im = ax.imshow(h.T, origin="lower", aspect="auto",
                           extent=[0.5, 20.5, 0.5, 21.5],
                           cmap=cmap_red, interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.8, label="count")

            for xi in range(20):
                for yi in range(21):
                    val = int(h[xi, yi])
                    if val > 0:
                        tc = "white" if val > h.max() * 0.5 else "#B71C1C"
                        ax.text(xi + 1, yi + 1, str(val),
                                ha="center", va="center", fontsize=6,
                                color=tc, fontweight="bold")

            ax.plot([0.5, 20.5], [0.5, 20.5], color="#555555",
                    linestyle="--", linewidth=1.2, alpha=0.6)
            ax.axhline(21, color="#555555", linestyle=":",
                       linewidth=1.2, alpha=0.6)
            ax.set_xlabel("Source Model Rank (cliff token)", fontsize=11)
            ax.set_ylabel("Eval Model Rank (21 = out of top-20)", fontsize=11)
            ax.set_title(DIRECTION_LABELS[direction], fontsize=12)

            n_total = len(source_ranks)
            n_out = sum(1 for r in eval_ranks if r == 21)
            r1_total = sum(1 for r in rows_filt if r["source_rank"] == 1)
            r1_keep = sum(1 for r in rows_filt
                          if r["source_rank"] == 1 and r["eval_rank"] == 1)
            info = [f"n = {n_total}"]
            info.append(
                f"Out of top-20: {n_out}/{n_total} ({n_out/n_total:.0%})"
                if n_total > 0 else "Out of top-20: 0")
            if r1_total > 0:
                info.append(
                    f"rank 1→1: {r1_keep}/{r1_total} ({r1_keep/r1_total:.0%})")
            ax.text(0.98, 0.02, "\n".join(info),
                    transform=ax.transAxes, fontsize=9, va="bottom",
                    ha="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        fig.suptitle(
            f"Cliff Token Rank Shift — {TAXONOMY_LABELS[tax]}",
            fontsize=13, y=1.02)
        fig.tight_layout()
        fname = f"rank_heatmap_{tax}.png"
        out = os.path.join(figures_dir, fname)
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def _plot_asymmetry_bar(data: Dict, figures_dir: str):
    """Plot 6: Direction asymmetry bar with IQR whiskers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    dir_colors = {"smalltobig": "#5C6BC0", "bigtosmall": "#EF5350"}

    x = np.arange(len(TAXONOMY_ORDER))
    width = 0.35

    for panel, stat_name, stat_fn in [
        (0, "Mean Δ CPM", np.mean),
        (1, "Median Δ CPM", np.median),
    ]:
        ax = axes[panel]
        for di, direction in enumerate(["smalltobig", "bigtosmall"]):
            centers = []
            q25s = []
            q75s = []
            for tax in TAXONOMY_ORDER:
                vals = [r["delta_cpm"] for r in data[direction]["cliff"]
                        if r["taxonomy_type"] == tax]
                centers.append(stat_fn(vals) if vals else 0)
                q25s.append(np.percentile(vals, 25) if vals else 0)
                q75s.append(np.percentile(vals, 75) if vals else 0)

            offset = -width / 2 + di * width
            color = dir_colors[direction]

            ax.bar(x + offset, centers, width,
                   label=DIRECTION_LABELS[direction],
                   color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

            cap_w = width * 0.3
            for ti in range(len(TAXONOMY_ORDER)):
                ax.plot([x[ti] + offset, x[ti] + offset],
                        [q25s[ti], q75s[ti]],
                        color="black", linewidth=1.8,
                        solid_capstyle="round", zorder=4)
                ax.plot([x[ti] + offset - cap_w, x[ti] + offset + cap_w],
                        [q25s[ti], q25s[ti]],
                        color="black", linewidth=1.2, zorder=4)
                ax.plot([x[ti] + offset - cap_w, x[ti] + offset + cap_w],
                        [q75s[ti], q75s[ti]],
                        color="black", linewidth=1.2, zorder=4)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TAXONOMY_ORDER, fontsize=11)
        ax.set_ylabel(stat_name, fontsize=11)
        ax.set_title(stat_name, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Direction Asymmetry — Small→Big vs Big→Small\n(whiskers = IQR: Q25–Q75)",
        fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(figures_dir, "asymmetry_bar.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _summarize(per_cliff: List[Dict], per_token: List[Dict]) -> Dict:
    def _group(rows, keys, float_col):
        out: Dict[Tuple, List[float]] = {}
        for r in rows:
            k = tuple(r[k0] for k0 in keys)
            out.setdefault(k, []).append(r[float_col])
        return out

    summary = {
        "n_cliffs_total": len(per_cliff),
        "n_tokens_total": len(per_token),
        "per_combo": {},
        "per_taxonomy_delta_cpm": {},
        "per_taxonomy_delta_rank": {},
    }

    combo_groups: Dict[Tuple, Dict] = {}
    for r in per_cliff:
        key = (r["model_source"], r["model_eval"], r["dataset"])
        combo_groups.setdefault(key, {"n": 0, "by_tax": {}})
        combo_groups[key]["n"] += 1
        tax = r["taxonomy_type"]
        combo_groups[key]["by_tax"].setdefault(tax, []).append(r["delta_cpm"])
    for key, data in combo_groups.items():
        s_key = f"{key[0]}__{key[1]}__{key[2]}"
        summary["per_combo"][s_key] = {
            "n_cliffs": data["n"],
            "delta_cpm_by_taxonomy": {
                tax: {
                    "n": len(vals),
                    "mean": statistics.fmean(vals) if vals else 0.0,
                    "median": statistics.median(vals) if vals else 0.0,
                }
                for tax, vals in data["by_tax"].items()
            },
        }

    for tax in TAXONOMY_ORDER:
        deltas = [r["delta_cpm"] for r in per_cliff if r["taxonomy_type"] == tax]
        summary["per_taxonomy_delta_cpm"][tax] = {
            "n": len(deltas),
            "mean": statistics.fmean(deltas) if deltas else 0.0,
            "median": statistics.median(deltas) if deltas else 0.0,
        }
        ranks = [r["delta_rank"] for r in per_token if r["taxonomy_type"] == tax]
        summary["per_taxonomy_delta_rank"][tax] = {
            "n": len(ranks),
            "mean": statistics.fmean(ranks) if ranks else 0.0,
            "median": statistics.median(ranks) if ranks else 0.0,
        }
    return summary


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <batch_dir>")
        sys.exit(1)
    batch_dir = sys.argv[1]
    if not os.path.isdir(batch_dir):
        print(f"ERROR: batch dir not found: {batch_dir}")
        sys.exit(1)

    print(f"[exp5_cpm_shift analyze] {batch_dir}")
    per_cliff, per_token = _merge_runs(batch_dir)
    print(f"  merged {len(per_cliff)} per_cliff rows, {len(per_token)} per_token rows")

    if per_cliff:
        _write_merged_csv(
            per_cliff,
            os.path.join(batch_dir, "results_per_cliff.csv"),
            fieldnames=[
                "model_source", "model_eval", "dataset", "data_idx",
                "cliff_position", "taxonomy_type", "n_cliff_tokens",
                "cpm_source", "cpm_eval", "delta_cpm",
            ],
        )
    if per_token:
        _write_merged_csv(
            per_token,
            os.path.join(batch_dir, "results_per_token.csv"),
            fieldnames=[
                "model_source", "model_eval", "dataset", "data_idx",
                "cliff_position", "taxonomy_type", "cliff_token_id",
                "cliff_token_str", "source_rank", "source_prob", "source_logprob",
                "eval_rank", "eval_prob", "eval_logprob",
                "delta_rank", "delta_prob",
            ],
        )

    figures_dir = os.path.join(batch_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    if per_cliff:
        _scatter_cpm(per_cliff, os.path.join(figures_dir, "cpm_shift_scatter.png"))
        _scatter_cpm_facet(per_cliff, os.path.join(figures_dir, "cpm_shift_scatter_facet.png"))
    if per_token:
        _scatter_rank(per_token, os.path.join(figures_dir, "rank_shift_scatter.png"))
        _scatter_prob(per_token, os.path.join(figures_dir, "prob_shift_scatter.png"))

    # ── Cross-direction plots (violin, rank heatmap, asymmetry bar) ──────────
    dir_data = _load_directional_data(batch_dir)
    if dir_data is not None:
        print(f"  Directional data loaded: "
              f"smalltobig={len(dir_data['smalltobig']['cliff'])} cliffs / "
              f"{len(dir_data['smalltobig']['token'])} selected tokens, "
              f"bigtosmall={len(dir_data['bigtosmall']['cliff'])} cliffs / "
              f"{len(dir_data['bigtosmall']['token'])} selected tokens")
        _plot_violin(dir_data, figures_dir)
        _plot_rank_heatmap(dir_data, figures_dir)
        _plot_asymmetry_bar(dir_data, figures_dir)
    else:
        print("  Skipping cross-direction plots "
              "(no smalltobig/bigtosmall sibling dirs)")

    summary = _summarize(per_cliff, per_token)
    with open(os.path.join(batch_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {os.path.join(batch_dir, 'summary.json')}")
    print("  done.")


if __name__ == "__main__":
    main()
