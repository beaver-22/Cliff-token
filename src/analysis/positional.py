"""
RQ1-2: Cliff Token vs Prior Work Definitions (Critical Token + Reasoning Tangent)

Part 1: Critical Token vs Cliff Token — positional difference analysis
Part 2: Reasoning Tangent vs Cliff Token — rate of cliff tokens within tangent regions

Outputs:
  - distance_scatter.png: distribution of cliff_pos - critical_pos
  - distance_summary.csv: before/same/after ratio table
  - tangent_summary.csv: rate of cliff tokens within tangent chunks (per model + combined)
  - case_studies/*.png: cases where a cliff exists outside the tangent region
"""

import os
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.detector import (
    find_first_cliff_token, find_all_cliff_tokens, find_critical_token,
    find_first_cliff_token_statistical, find_all_cliff_tokens_statistical,
)

# Default to statistical cliff (z-test)
USE_STATISTICAL_CLIFF = True
from src import config

# Reuse style from curves.py
SUCCESS_COLOR = "#90CAF9"
FAILURE_COLOR = "#EF9A9A"
TANGENT_COLOR = "#FFB74D"       # orange for tangent shading
TANGENT_EDGE = "#E65100"        # deep orange for tangent border
BOUNDARY_COLOR = "#4CAF50"      # green for chunk boundary line
CLIFF_COLOR = "#E53935"         # red for cliff markers

TANGENT_THRESHOLD = 0.3
TANGENT_N_CHUNKS = 20


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 9,
    })
    sns.set_style("whitegrid")


# ============================================================
# Data structures
# ============================================================

@dataclass
class TangentChunk:
    chunk_index: int
    start_idx: int
    end_idx: int
    start_score: float
    end_score: float
    score_drop: float
    is_tangent: bool
    contains_cliff: bool


@dataclass
class PathAnalysisResult:
    path_id: str
    model: str
    dataset: str
    total_tokens: int
    # Part 1
    has_cliff: bool
    has_critical: bool
    cliff_position: Optional[int]
    critical_position: Optional[int]
    distance: Optional[int]           # cliff_pos - critical_pos (absolute tokens)
    category: Optional[str]           # "before" / "same" / "after"
    # Part 2
    all_cliff_positions: List[int]    # all cliff positions (1-indexed)
    boundary_indices: List[int]       # 21 boundary indices (0-indexed into scores)
    boundary_scores: List[float]      # 21 scores at boundaries
    chunks: List[TangentChunk]        # 20 chunks
    cliffs_outside_tangent: List[int] # cliff positions not inside any tangent chunk


@dataclass
class Experiment2Results:
    total_failure_paths: int
    paths_with_both: int
    paths_cliff_only: int
    paths_critical_only: int
    paths_neither: int
    distances: List[int]
    category_counts: Dict[str, int]
    total_cliffs: int
    cliffs_inside_tangent: int
    tangent_cliff_rate: float           # cliffs_inside / total_cliffs * 100
    path_results: List[PathAnalysisResult]
    # Per-model tangent stats
    model_tangent_stats: Dict[str, Dict]


# ============================================================
# Part 1: Critical-Cliff distance
# ============================================================

def _analyze_single_path(
    path: Dict, model: str = "", dataset: str = "",
) -> PathAnalysisResult:
    """Analyze a single failure path for both Part 1 and Part 2."""
    scores = path.get("all_position_scores", [])
    total = path.get("total_tokens", len(scores) + 1)
    pid = path.get("id", "?")

    # Part 1: first cliff + critical token
    if USE_STATISTICAL_CLIFF:
        cliff = find_first_cliff_token_statistical(scores)
        all_cliffs = find_all_cliff_tokens_statistical(
            scores,
            tokens=path.get("response_tokens"),
            token_ids=path.get("response_token_ids"),
        )
    else:
        cliff = find_first_cliff_token(scores, config.DEFAULT_CLIFF_THRESHOLD)
        all_cliffs = find_all_cliff_tokens(
            scores, config.DEFAULT_CLIFF_THRESHOLD,
            tokens=path.get("response_tokens"),
            token_ids=path.get("response_token_ids"),
        )
    critical = find_critical_token(scores, config.CRITICAL_TOKEN_THRESHOLD)
    all_cliff_positions = [c.position for c in all_cliffs]

    has_cliff = cliff is not None
    has_critical = critical is not None
    distance = None
    category = None
    if has_cliff and has_critical:
        distance = cliff.position - critical.position
        if distance < 0:
            category = "before"
        elif distance == 0:
            category = "same"
        else:
            category = "after"

    # Part 2: tangent detection
    boundary_indices, boundary_scores, chunks, cliffs_outside = _detect_tangents(
        scores, all_cliff_positions
    )

    return PathAnalysisResult(
        path_id=pid,
        model=model,
        dataset=dataset,
        total_tokens=total,
        has_cliff=has_cliff,
        has_critical=has_critical,
        cliff_position=cliff.position if cliff else None,
        critical_position=critical.position if critical else None,
        distance=distance,
        category=category,
        all_cliff_positions=all_cliff_positions,
        boundary_indices=boundary_indices,
        boundary_scores=boundary_scores,
        chunks=chunks,
        cliffs_outside_tangent=cliffs_outside,
    )


# ============================================================
# Part 2: Tangent detection
# ============================================================

def _compute_chunk_boundaries(n_scores: int, n_chunks: int = TANGENT_N_CHUNKS) -> List[int]:
    """21 boundary indices (0-indexed) dividing scores into n_chunks equal parts."""
    if n_scores < 2:
        return [0]
    return np.linspace(0, n_scores - 1, n_chunks + 1).astype(int).tolist()


def _detect_tangents(
    scores: List[float],
    cliff_positions: List[int],
    tangent_threshold: float = TANGENT_THRESHOLD,
    n_chunks: int = TANGENT_N_CHUNKS,
) -> Tuple[List[int], List[float], List[TangentChunk], List[int]]:
    """Detect tangent chunks and check cliff containment.

    Returns: (boundary_indices, boundary_scores, chunks, cliffs_outside_tangent)
    """
    n_scores = len(scores)
    boundaries = _compute_chunk_boundaries(n_scores, n_chunks)
    boundary_scores = [scores[i] if i < n_scores else 0.0 for i in boundaries]

    tangent_ranges = []  # (start_idx, end_idx) for tangent chunks
    chunks = []
    for i in range(len(boundaries) - 1):
        s_idx, e_idx = boundaries[i], boundaries[i + 1]
        s_score = boundary_scores[i]
        e_score = boundary_scores[i + 1]
        drop = s_score - e_score
        is_tangent = drop >= tangent_threshold

        # Check if any cliff falls within [s_idx, e_idx]
        contains_cliff = any(s_idx <= (cp - 1) <= e_idx for cp in cliff_positions)

        chunks.append(TangentChunk(
            chunk_index=i, start_idx=s_idx, end_idx=e_idx,
            start_score=s_score, end_score=e_score,
            score_drop=drop, is_tangent=is_tangent, contains_cliff=contains_cliff,
        ))
        if is_tangent:
            tangent_ranges.append((s_idx, e_idx))

    # Find cliffs outside all tangent ranges
    cliffs_outside = []
    for cp in cliff_positions:
        cp_idx = cp - 1  # 1-indexed → 0-indexed
        inside = any(s <= cp_idx <= e for s, e in tangent_ranges)
        if not inside:
            cliffs_outside.append(cp)

    return boundaries, boundary_scores, chunks, cliffs_outside


# ============================================================
# Aggregation
# ============================================================

def _aggregate_results(path_results: List[PathAnalysisResult]) -> Experiment2Results:
    total = len(path_results)
    both = sum(1 for r in path_results if r.has_cliff and r.has_critical)
    cliff_only = sum(1 for r in path_results if r.has_cliff and not r.has_critical)
    critical_only = sum(1 for r in path_results if not r.has_cliff and r.has_critical)
    neither = sum(1 for r in path_results if not r.has_cliff and not r.has_critical)

    distances = [r.distance for r in path_results if r.distance is not None]
    cat_counts = {"before": 0, "same": 0, "after": 0}
    for r in path_results:
        if r.category:
            cat_counts[r.category] += 1

    # Tangent stats — cliff-based: cliffs inside tangent / total cliffs
    total_cliffs = sum(len(r.all_cliff_positions) for r in path_results)
    cliffs_inside_tangent = total_cliffs - sum(len(r.cliffs_outside_tangent) for r in path_results)
    tangent_cliff_rate = (cliffs_inside_tangent / total_cliffs * 100) if total_cliffs > 0 else 0

    # Per-model tangent stats (cliff-based)
    model_stats = {}
    for r in path_results:
        m = r.model or "unknown"
        if m not in model_stats:
            model_stats[m] = {"total_cliffs": 0, "cliffs_inside_tangent": 0}
        n_cliffs = len(r.all_cliff_positions)
        n_outside = len(r.cliffs_outside_tangent)
        model_stats[m]["total_cliffs"] += n_cliffs
        model_stats[m]["cliffs_inside_tangent"] += (n_cliffs - n_outside)
    for m in model_stats:
        t = model_stats[m]["total_cliffs"]
        model_stats[m]["rate"] = (model_stats[m]["cliffs_inside_tangent"] / t * 100) if t > 0 else 0

    return Experiment2Results(
        total_failure_paths=total,
        paths_with_both=both,
        paths_cliff_only=cliff_only,
        paths_critical_only=critical_only,
        paths_neither=neither,
        distances=distances,
        category_counts=cat_counts,
        total_cliffs=total_cliffs,
        cliffs_inside_tangent=cliffs_inside_tangent,
        tangent_cliff_rate=tangent_cliff_rate,
        path_results=path_results,
        model_tangent_stats=model_stats,
    )


# ============================================================
# Plots
# ============================================================

def _plot_distance_scatter(path_results: List["PathAnalysisResult"], output_dir: str):
    """Plot signed (cliff_pos - critical_pos) for ALL cliffs in paths that
    have a critical token. Mirrors `_plot_relative_distance` denominator so
    the two figures share the same population."""
    _apply_style()

    distances: List[int] = []
    for r in path_results:
        if r.critical_position is None:
            continue
        for cp in r.all_cliff_positions:
            distances.append(cp - r.critical_position)

    if not distances:
        print("  No paths with both cliff and critical token — skipping plots.")
        return

    from collections import Counter
    d = np.array(distances)  # negative: cliff before critical
    n = len(d)
    counts = Counter(d.tolist())
    max_count = max(counts.values())
    median_d = np.median(d)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    for dist in sorted(counts.keys()):
        cnt = counts[dist]
        alpha = 0.3 + 0.5 * (cnt / max_count)
        ax.plot([dist, dist], [-0.3, 0.3], color="#5C6BC0", linewidth=1.5, alpha=alpha, zorder=2)
    ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.8, zorder=1)
    ax.axvline(median_d, color=FAILURE_COLOR, linestyle="--", linewidth=1.5,
               label=f"Median = {median_d:.0f} tokens", zorder=3)
    ax.set_xlabel("Cliff Token Position − Critical Token Position (tokens)")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"Cliff Token → Critical Token Distance  (n={n})")
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate("Critical Token", xy=(0, -0.45), fontsize=8, ha="center", color="black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distance.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: distance.png")


def _plot_relative_distance(path_results: List["PathAnalysisResult"], output_dir: str):
    """Relative-position version of distance.png.

    For each (cliff, critical) pair where both exist, compute
        rel = cliff_position / critical_position
    so that:
        rel = 0  → first token of response (1-indexed pos 1 / large critical)
        rel = 1  → cliff exactly at critical token
        rel > 1  → cliff AFTER critical token

    Aggregates ALL cliffs (not just first) across paths and draws each as a
    vertical line whose opacity encodes occurrence count, mirroring the style
    of `_plot_distance_scatter`.
    """
    _apply_style()

    rel_positions: List[float] = []
    n_excluded_after = 0
    for r in path_results:
        if r.critical_position is None or r.critical_position <= 0:
            continue
        for cp in r.all_cliff_positions:
            rel = cp / r.critical_position
            if rel > 1.0:
                n_excluded_after += 1
                continue
            rel_positions.append(rel)

    if not rel_positions:
        print("  No (cliff, critical) pairs in [0, 1] — skipping distance_relative.png.")
        return

    rel_arr = np.array(rel_positions)
    n = len(rel_arr)
    median_r = float(np.median(rel_arr))
    mean_r = float(rel_arr.mean())

    # Histogram with 20 bins on [0, 1] — enough resolution without noise
    n_bins = 20
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(rel_arr, bins=bin_edges)
    pct = counts / n * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.95
    ax.bar(
        bin_edges[:-1] + bar_width / 2, pct, width=bar_width,
        color="#5C6BC0", edgecolor="white", linewidth=0.6, zorder=2,
    )

    # Reference markers — bold black vertical lines at the boundaries
    ax.axvline(0.0, color="black", linestyle="-", linewidth=3.0, alpha=1.0, zorder=4)
    ax.axvline(1.0, color="black", linestyle="-", linewidth=3.0, alpha=1.0, zorder=4)

    ax.set_xlabel("Relative position  (0 = first token, 1 = critical token)")
    ax.set_ylabel("% of cliff tokens")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, max(pct.max() * 1.18, 5))
    title = f"Cliff Token Relative Position to Critical Token  (n={n})"
    if n_excluded_after:
        title += f"   [{n_excluded_after} cliffs after critical excluded]"
    ax.set_title(title)
    # Labels inside the plot, ~30% down from the top, just inside the boundary lines
    ax.text(0.012, 0.70, "first token",
            transform=ax.transAxes, fontsize=9, ha="left", va="center",
            color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            zorder=5)
    ax.text(0.988, 0.70, "critical token",
            transform=ax.transAxes, fontsize=9, ha="right", va="center",
            color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            zorder=5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distance_relative.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: distance_relative.png  (n_cliffs={n}, excluded_after={n_excluded_after})")

    # ----- Companion: scatter-style with one line per cliff (no aggregation) -----
    # Mirror distance.png: all lines parallel, full vertical range, alpha makes
    # dense regions darker naturally.
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    for x in rel_arr:
        ax2.plot([x, x], [-0.3, 0.3],
                 color="#5C6BC0", linewidth=1.0, alpha=0.35, zorder=2)

    ax2.axvline(0.0, color="black", linestyle="-", linewidth=1.5, alpha=0.8, zorder=1)
    ax2.axvline(1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.8, zorder=1)
    ax2.axvline(median_r, color=FAILURE_COLOR, linestyle="--", linewidth=1.5,
                label=f"Median = {median_r:.2f}", zorder=3)
    ax2.axvline(mean_r, color="#2E7D32", linestyle=":", linewidth=1.5,
                label=f"Mean = {mean_r:.2f}", zorder=3)

    ax2.set_xlabel("Relative position  (0 = first token, 1 = critical token)")
    ax2.set_yticks([])
    ax2.set_ylim(-0.55, 0.55)
    ax2.set_xlim(-0.02, 1.02)
    title2 = f"Cliff Token Relative Position (per-cliff scatter, n={n})"
    if n_excluded_after:
        title2 += f"   [{n_excluded_after} after critical excluded]"
    ax2.set_title(title2)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.annotate("first token", xy=(0.0, -0.05), xycoords=("data", "axes fraction"),
                 fontsize=8, ha="center", color="black", annotation_clip=False)
    ax2.annotate("critical token", xy=(1.0, -0.05), xycoords=("data", "axes fraction"),
                 fontsize=8, ha="center", color="black", annotation_clip=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "distance_relative_scatter.png"), bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: distance_relative_scatter.png  ({n} individual cliff lines)")


def _plot_tangent_case_study(
    path: Dict, result: PathAnalysisResult, output_path: str,
):
    """Case study: potential curve + tangent overlay + cliff markers."""
    _apply_style()
    scores = path.get("all_position_scores", [])
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    positions = list(range(len(scores)))

    # Tangent shading (behind everything)
    tangent_labeled = False
    for c in result.chunks:
        if c.is_tangent:
            lbl = "Tangent Region (drop ≥ 0.3)" if not tangent_labeled else None
            ax.axvspan(c.start_idx, c.end_idx, alpha=0.3, facecolor=TANGENT_COLOR,
                       edgecolor=TANGENT_EDGE, linewidth=2, zorder=1, label=lbl)
            tangent_labeled = True

    # Token-level potential curve
    ax.plot(positions, scores, color="black", linewidth=0.8, alpha=0.8, zorder=2)
    ax.fill_between(positions, scores, alpha=0.1, color="steelblue", zorder=1)

    # Chunk boundary line (green dots + line)
    ax.plot(result.boundary_indices, result.boundary_scores, "o-",
            color=BOUNDARY_COLOR, markersize=4, linewidth=1.5, alpha=0.8,
            label="20-Chunk Potential", zorder=3)

    # Cliff tokens (red circles)
    cliff_plotted = False
    for cp in result.all_cliff_positions:
        idx = cp - 1
        if 0 <= idx < len(scores):
            in_tangent = any(c.start_idx <= idx <= c.end_idx for c in result.chunks if c.is_tangent)
            lbl = "Cliff Token" if not cliff_plotted else None
            marker_edge = "darkred" if not in_tangent else "red"
            ax.plot(idx, scores[idx], "o", color=CLIFF_COLOR, markersize=8,
                    markeredgecolor=marker_edge, markeredgewidth=1.5,
                    label=lbl, zorder=4)
            cliff_plotted = True

    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Potential")
    ax.set_ylim(-0.05, 1.05)

    model = result.model or path.get("_model", "")
    ds = result.dataset or path.get("_dataset", "")
    parts = [p for p in [model, ds, result.path_id] if p]
    outside_n = len(result.cliffs_outside_tangent)
    title = f"{'_'.join(parts)}  |  Cliffs outside tangent: {outside_n}/{len(result.all_cliff_positions)}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CSV / Summary helpers
# ============================================================

def _save_distance_summary(path_results: List[PathAnalysisResult], output_path: str):
    """Per-cliff category counts restricted to paths that have a critical token.

    For each path with a critical token, every cliff in ``all_cliff_positions``
    is categorized by sign of (cliff_pos - critical_pos).
    """
    per_cliff_counts = {"before": 0, "same": 0, "after": 0}
    for r in path_results:
        if r.critical_position is None:
            continue
        for cp in r.all_cliff_positions:
            d = cp - r.critical_position
            if d < 0:
                per_cliff_counts["before"] += 1
            elif d == 0:
                per_cliff_counts["same"] += 1
            else:
                per_cliff_counts["after"] += 1

    total = sum(per_cliff_counts.values())
    rows = []
    for cat in ["before", "same", "after"]:
        n = per_cliff_counts[cat]
        pct = n / total * 100 if total > 0 else 0
        rows.append({"category": f"cliff {cat} critical", "count": n, "percentage": f"{pct:.1f}%"})
    rows.append({"category": "total cliffs (paths with critical)", "count": total, "percentage": "100%"})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")


def _save_tangent_summary(results: Experiment2Results, output_path: str):
    n_outside = results.total_cliffs - results.cliffs_inside_tangent
    rows = [
        {"metric": "Total Cliff Tokens", "inside_tangent": results.cliffs_inside_tangent, "outside_tangent": n_outside, "total": results.total_cliffs},
        {"metric": "Rate (%)", "inside_tangent": f"{results.tangent_cliff_rate:.1f}", "outside_tangent": f"{100 - results.tangent_cliff_rate:.1f}", "total": "100.0"},
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")


def _save_tangent_bidirectional(path_results: List[PathAnalysisResult], output_path: str):
    """Bidirectional cliff/tangent overlap.

    Direction A — Tangent → Cliff: of all tangent chunks across paths,
    what fraction contain ≥1 cliff token.
    Direction B — Cliff → Tangent: of all cliff tokens, what fraction land
    inside any tangent chunk (mirrors `tangent_summary.csv`).
    """
    n_tangent_chunks = 0
    n_tangent_with_cliff = 0
    for r in path_results:
        for c in r.chunks:
            if c.is_tangent:
                n_tangent_chunks += 1
                if c.contains_cliff:
                    n_tangent_with_cliff += 1
    tangent_to_cliff_rate = (n_tangent_with_cliff / n_tangent_chunks * 100) if n_tangent_chunks else 0.0

    n_cliffs_total = sum(len(r.all_cliff_positions) for r in path_results)
    n_cliffs_inside = n_cliffs_total - sum(len(r.cliffs_outside_tangent) for r in path_results)
    cliff_to_tangent_rate = (n_cliffs_inside / n_cliffs_total * 100) if n_cliffs_total else 0.0

    rows = [
        {
            "view": "Tangent -> Cliff",
            "numerator": n_tangent_with_cliff,
            "denominator": n_tangent_chunks,
            "rate_pct": f"{tangent_to_cliff_rate:.1f}",
            "description": "Of all tangent chunks, % containing >=1 cliff token",
        },
        {
            "view": "Cliff -> Tangent",
            "numerator": n_cliffs_inside,
            "denominator": n_cliffs_total,
            "rate_pct": f"{cliff_to_tangent_rate:.1f}",
            "description": "Of all cliff tokens, % located inside a tangent chunk",
        },
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")


# ============================================================
# Public API: Single-model (compatible with cli.py)
# ============================================================

def run_experiment2_analysis(
    failure_paths: List[Dict], output_dir: str,
) -> Experiment2Results:
    os.makedirs(output_dir, exist_ok=True)
    path_results = [_analyze_single_path(p) for p in failure_paths]
    results = _aggregate_results(path_results)

    _plot_distance_scatter(results.path_results, output_dir)
    _plot_relative_distance(results.path_results, output_dir)
    _save_distance_summary(results.path_results,
                           os.path.join(output_dir, "distance_summary.csv"))
    _save_tangent_summary(results, os.path.join(output_dir, "tangent_summary.csv"))
    _save_tangent_bidirectional(results.path_results,
                                os.path.join(output_dir, "tangent_bidirectional.csv"))

    return results


def print_experiment2_summary(results: Experiment2Results):
    print(f"\n{'='*60}")
    print("Part 1: Critical Token vs Cliff Token")
    print(f"{'='*60}")
    print(f"  Total failure paths: {results.total_failure_paths}")
    print(f"  Both cliff + critical: {results.paths_with_both}")
    print(f"  Cliff only: {results.paths_cliff_only}")
    print(f"  Critical only: {results.paths_critical_only}")
    print(f"  Neither: {results.paths_neither}")

    if results.paths_with_both > 0:
        total = results.paths_with_both
        print(f"\n  Cliff BEFORE critical: {results.category_counts['before']} "
              f"({results.category_counts['before']/total*100:.1f}%)")
        print(f"  Same position:         {results.category_counts['same']} "
              f"({results.category_counts['same']/total*100:.1f}%)")
        print(f"  Cliff AFTER critical:  {results.category_counts['after']} "
              f"({results.category_counts['after']/total*100:.1f}%)")
        if results.distances:
            d = np.array(results.distances)
            print(f"\n  Distance (tokens): mean={d.mean():.1f}, median={np.median(d):.0f}, "
                  f"min={d.min()}, max={d.max()}")

    print(f"\n{'='*60}")
    print("Part 2: Reasoning Tangent vs Cliff Token")
    print(f"{'='*60}")
    print(f"  Total cliff tokens: {results.total_cliffs}")
    print(f"  Cliffs inside tangent: {results.cliffs_inside_tangent} "
          f"({results.tangent_cliff_rate:.1f}%)")
    print(f"  Cliffs outside tangent: {results.total_cliffs - results.cliffs_inside_tangent} "
          f"({100 - results.tangent_cliff_rate:.1f}%)")

    for model, stats in results.model_tangent_stats.items():
        print(f"    {model}: {stats['cliffs_inside_tangent']}/{stats['total_cliffs']} "
              f"inside ({stats['rate']:.1f}%)")


def create_all_visualizations(failure_paths: List[Dict], output_dir: str):
    """Generate tangent case study plots. Prioritize cliffs OUTSIDE tangent."""
    case_dir = os.path.join(output_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)

    results = [_analyze_single_path(p) for p in failure_paths]

    # Case study: paths where cliff exists BOTH inside AND outside tangent
    n_inside = lambda r: len(r.all_cliff_positions) - len(r.cliffs_outside_tangent)
    both_cases = [(p, r) for p, r in zip(failure_paths, results)
                  if r.cliffs_outside_tangent and n_inside(r) > 0]

    print(f"\n[Case Study] Paths with cliff both inside AND outside tangent: {len(both_cases)}")

    manifest = []
    for p, r in both_cases:
        safe_id = r.path_id.replace("/", "_").replace(" ", "_")
        parts = [x for x in [r.model, r.dataset, safe_id] if x]
        fname = f"case_{'_'.join(parts)}.png"
        _plot_tangent_case_study(p, r, os.path.join(case_dir, fname))
        n_in = n_inside(r)
        manifest.append({
            "path_id": r.path_id,
            "model": r.model,
            "cliffs_inside": n_in,
            "cliffs_outside": len(r.cliffs_outside_tangent),
            "total_cliffs": len(r.all_cliff_positions),
            "filename": fname,
        })
        print(f"    {fname} (inside={n_in}, outside={len(r.cliffs_outside_tangent)})")

    if manifest:
        manifest_path = os.path.join(case_dir, "manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)
        print(f"  Manifest: {manifest_path}")
    else:
        print("  No case studies generated.")


# ============================================================
# Public API: Multi-model (run_exp1_occurrence.sh)
# ============================================================

def run_multi_model_analysis(
    model_dataset_map: Dict[str, Dict[str, str]], output_dir: str,
):
    """Cross-model aggregation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    _apply_style()

    print("Loading data...")
    all_path_results = []
    all_failure_paths = []

    for model, datasets in model_dataset_map.items():
        for ds, json_path in datasets.items():
            all_paths = json.load(open(json_path))
            failures = [p for p in all_paths if not p.get("is_correct")]
            print(f"  {model}/{ds}: {len(failures)} failure paths")

            for p in failures:
                p["_model"] = model
                p["_dataset"] = ds
                r = _analyze_single_path(p, model=model, dataset=ds)
                all_path_results.append(r)
                all_failure_paths.append(p)

    results = _aggregate_results(all_path_results)

    # Part 1: Distance analysis
    print("\n--- Part 1: Critical Token vs Cliff Token ---")
    _plot_distance_scatter(results.path_results, output_dir)
    _plot_relative_distance(results.path_results, output_dir)
    _save_distance_summary(results.path_results,
                           os.path.join(output_dir, "distance_summary.csv"))
    print_experiment2_summary(results)

    # Part 2: Tangent analysis
    print("\n--- Part 2: Tangent Case Studies ---")
    _save_tangent_summary(results, os.path.join(output_dir, "tangent_summary.csv"))
    _save_tangent_bidirectional(results.path_results,
                                os.path.join(output_dir, "tangent_bidirectional.csv"))

    # Case studies
    case_dir = os.path.join(output_dir, "case_studies")
    os.makedirs(case_dir, exist_ok=True)

    n_inside = lambda r: len(r.all_cliff_positions) - len(r.cliffs_outside_tangent)
    both_cases = [(p, r) for p, r in zip(all_failure_paths, all_path_results)
                  if r.cliffs_outside_tangent and n_inside(r) > 0]

    print(f"\n  Paths with cliff both inside AND outside tangent: {len(both_cases)}")

    manifest = []
    for p, r in both_cases:
        safe_id = r.path_id.replace("/", "_").replace(" ", "_")
        parts = [x for x in [r.model, r.dataset, safe_id] if x]
        fname = f"case_{'_'.join(parts)}.png"
        _plot_tangent_case_study(p, r, os.path.join(case_dir, fname))
        n_in = n_inside(r)
        manifest.append({
            "path_id": r.path_id, "model": r.model,
            "cliffs_inside": n_in, "cliffs_outside": len(r.cliffs_outside_tangent),
            "total_cliffs": len(r.all_cliff_positions), "filename": fname,
        })
        print(f"    {fname} (inside={n_in}, outside={len(r.cliffs_outside_tangent)})")

    if manifest:
        mp = os.path.join(case_dir, "manifest.csv")
        with open(mp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)
        print(f"  Manifest: {mp}")

    print(f"\nAll outputs saved to: {output_dir}")
