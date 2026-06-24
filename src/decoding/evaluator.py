"""
RQ1-3 Evaluator: pass@k computation, comparison plots, semantic analysis.

Exp 1: Cliff-del vs Cliff-keep (success + failure)
Exp 2: Cliff-del vs Critical-del vs Tangent-del vs Random-del
Exp 3: Semantic analysis (token context + taxonomy)

Cross-run aggregation:
  aggregate_per_model — combine multiple (model, dataset) runs into per-model summaries
  plot_per_model_pass_at_k — single chart per model with datasets aggregated
  plot_model_dataset_grid — grid of (model × dataset) cells
"""

import os
import csv
import json
from math import comb
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config


# ============================================================
# Style
# ============================================================
SUCCESS_COLOR = "#90CAF9"
FAILURE_COLOR = "#EF9A9A"
METHOD_COLORS = {
    "Cliff-del": "#E53935",
    "Critical-del": "#1E88E5",
    "Tangent-del": "#43A047",
    "Random-del": "#8E24AA",
}

# Exp1 line styles: dark = del, light = keep; blue = success, red = failure
EXP1_LINES = {
    "del_success":  {"color": "#1565C0", "ls": "-", "marker": "o", "label": "Cliff-del (Success)"},
    "keep_success": {"color": "#64B5F6", "ls": "-", "marker": "s", "label": "Cliff-keep (Success)"},
    "del_failure":  {"color": "#C62828", "ls": "-", "marker": "o", "label": "Cliff-del (Failure)"},
    "keep_failure": {"color": "#EF9A9A", "ls": "-", "marker": "s", "label": "Cliff-keep (Failure)"},
}
GAP_ARROW_KS = [1, 2, 4, 8, 16, 32, 64]  # k values where gap arrows are drawn

# Display order for grid plots (used by plot_model_dataset_grid)
MODEL_DISPLAY_ORDER = [
    "Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B",
    "Llama-3.1-8B-Instruct", "gemma-3-4b-it",
]
DATASET_DISPLAY_ORDER = ["gsm1k", "math500", "aime25"]


def _ordered(items, ordering):
    """Return items in `ordering` order, with any extras appended at the end."""
    in_order = [x for x in ordering if x in items]
    extras = [x for x in items if x not in ordering]
    return in_order + extras


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    })
    sns.set_style("whitegrid")


# ============================================================
# Pass@k computation
# ============================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k)."""
    if c == 0:
        return 0.0
    if c >= n or n - c < k:
        return 1.0
    if k > n:
        k = n
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_avg_pass_at_k(
    results: List[Dict],
    correct_key: str = "del_num_correct",
    samples_key: str = "num_samples",
    k_values: List[int] = None,
) -> Dict[int, float]:
    """Compute average pass@k across all result instances."""
    if k_values is None:
        k_values = config.PASS_K_VALUES
    if not results:
        return {k: 0.0 for k in k_values}

    avg = {k: 0.0 for k in k_values}
    for r in results:
        n = r[samples_key]
        c = r[correct_key]
        for k in k_values:
            avg[k] += pass_at_k(n, c, k)
    for k in k_values:
        avg[k] /= len(results)
    return avg


def compute_avg_at_k(
    results: List[Dict],
    correct_key: str = "del_num_correct",
    samples_key: str = "num_samples",
) -> float:
    """Average accuracy across problems: mean of c/n.

    avg@k is the expected fraction of correct answers; it does not depend
    on k (= c/n regardless of k), so a single scalar is returned.
    """
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        n = r[samples_key]
        c = r[correct_key]
        total += (c / n) if n > 0 else 0.0
    return total / len(results)


# ============================================================
# Exp 1: Cliff-del vs Cliff-keep
# ============================================================

def _plot_pass_at_k_chart(
    data: Dict[str, Dict[int, float]],
    line_keys: List[str],
    title: str,
    save_path: str,
    n_success: int = 0,
    n_failure: int = 0,
):
    """Shared helper to draw pass@k chart with gap arrows.

    Args:
        data: dict with keys like "del_success", "keep_failure", etc.
              Each value maps k -> avg pass@k.
        line_keys: which keys from data to plot (e.g. all 4, or just failure 2).
        title: chart title.
        save_path: output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in line_keys:
        style = EXP1_LINES[key]
        d = data[key]
        ks = sorted(d.keys())
        vals = [d[k] for k in ks]
        ax.plot(
            ks, vals,
            marker=style["marker"], markersize=5,
            color=style["color"], linestyle="-",
            linewidth=1.8, label=style["label"], zorder=2,
        )

    # Gap arrows between del and keep
    # Determine del/keep pairs present in line_keys
    # Only draw gap lines when showing a single group (failure-only or success-only)
    has_success = any("success" in k for k in line_keys)
    has_failure = any("failure" in k for k in line_keys)
    pairs = []
    if has_success and has_failure:
        pass  # skip gaps — too cluttered with both groups
    elif "del_failure" in line_keys and "keep_failure" in line_keys:
        pairs.append(("del_failure", "keep_failure"))
    elif "del_success" in line_keys and "keep_success" in line_keys:
        pairs.append(("del_success", "keep_success"))

    for del_key, keep_key in pairs:
        for k in GAP_ARROW_KS:
            if k not in data[del_key]:
                continue
            v_del = data[del_key][k]
            v_keep = data[keep_key][k]
            gap = v_del - v_keep
            if abs(gap) < 0.005:
                continue

            mid_y = (v_del + v_keep) / 2
            # Simple black line (no arrowheads)
            ax.plot(
                [k, k], [v_keep, v_del],
                color="black", linewidth=1.0, zorder=4,
            )
            ax.text(
                k + 1.5, mid_y, f"{gap:+.3f}",
                fontsize=7, color="black", fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                zorder=5,
            )

    subtitle_parts = []
    if n_success:
        subtitle_parts.append(f"n_success={n_success}")
    if n_failure:
        subtitle_parts.append(f"n_failure={n_failure}")
    full_title = title
    if subtitle_parts:
        full_title += f"  ({', '.join(subtitle_parts)})"

    ax.set_xlabel("k")
    ax.set_ylabel("Mean Pass@k")
    ax.set_title(full_title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(handlelength=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_experiment1_evaluation(cliff_results: List[Dict], output_dir: str) -> Dict:
    """Evaluate Cliff-del vs Cliff-keep for success and failure paths."""
    os.makedirs(output_dir, exist_ok=True)
    _apply_style()

    success = [r for r in cliff_results if r["path_is_correct"]]
    failure = [r for r in cliff_results if not r["path_is_correct"]]

    k_values = config.PASS_K_VALUES
    result = {
        "del_success": compute_avg_pass_at_k(success, "del_num_correct", k_values=k_values),
        "del_failure": compute_avg_pass_at_k(failure, "del_num_correct", k_values=k_values),
        "keep_success": compute_avg_pass_at_k(success, "keep_num_correct", k_values=k_values),
        "keep_failure": compute_avg_pass_at_k(failure, "keep_num_correct", k_values=k_values),
        "n_success_instances": len(success),
        "n_failure_instances": len(failure),
    }

    # Save JSON
    json.dump(result, open(os.path.join(output_dir, "exp1_pass_at_k.json"), "w"), indent=2)

    # Chart A: all 4 lines (success + failure)
    _plot_pass_at_k_chart(
        result,
        line_keys=["del_success", "keep_success", "del_failure", "keep_failure"],
        title="Cliff-del vs Cliff-keep",
        save_path=os.path.join(output_dir, "pass_at_k_all.png"),
        n_success=len(success), n_failure=len(failure),
    )

    # Chart B: failure only
    _plot_pass_at_k_chart(
        result,
        line_keys=["del_failure", "keep_failure"],
        title="Cliff-del vs Cliff-keep (Failure Paths Only)",
        save_path=os.path.join(output_dir, "pass_at_k_failure.png"),
        n_failure=len(failure),
    )

    return result


# ============================================================
# Exp 2: 4-method comparison
# ============================================================

def run_experiment2_evaluation(
    cliff_results: List[Dict],
    critical_results: List[Dict],
    tangent_results: List[Dict],
    random_results: List[Dict],
    output_dir: str,
    suffix: str = "",
    title_extra: str = "",
    upper_bounds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compare 4 deletion methods.

    Args:
        suffix: appended to output filenames (e.g. "_cliff_first").
        title_extra: appended to chart title in parentheses.
        upper_bounds: optional {method_name: max possible pass@k}. Drawn as
            horizontal dashed lines in matching color (skipped if >= 0.999).
    """
    os.makedirs(output_dir, exist_ok=True)
    _apply_style()

    k_values = config.PASS_K_VALUES
    methods = {
        "Cliff-del": compute_avg_pass_at_k(cliff_results, "del_num_correct", k_values=k_values),
        "Critical-del": compute_avg_pass_at_k(critical_results, "del_num_correct", k_values=k_values),
        "Tangent-del": compute_avg_pass_at_k(tangent_results, "del_num_correct", k_values=k_values),
        "Random-del": compute_avg_pass_at_k(random_results, "del_num_correct", k_values=k_values),
    }
    avg_methods = {
        "Cliff-del": compute_avg_at_k(cliff_results, "del_num_correct"),
        "Critical-del": compute_avg_at_k(critical_results, "del_num_correct"),
        "Tangent-del": compute_avg_at_k(tangent_results, "del_num_correct"),
        "Random-del": compute_avg_at_k(random_results, "del_num_correct"),
    }

    result = {
        "methods": methods,
        "avg_methods": avg_methods,
        "n_paths": {
            "cliff": len(cliff_results),
            "critical": len(critical_results),
            "tangent": len(tangent_results),
            "random": len(random_results),
        },
        "upper_bounds": upper_bounds or {},
    }

    json.dump(result, open(os.path.join(output_dir, f"exp2_pass_at_k{suffix}.json"), "w"), indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name, data in methods.items():
        color = METHOD_COLORS.get(method_name, "gray")
        ks = sorted(data.keys())
        vals = [data[k] for k in ks]
        ax.plot(ks, vals, marker="o", markersize=4, color=color, linewidth=2, label=method_name)

    # Upper bound dashed lines
    if upper_bounds:
        ks_all = sorted(next(iter(methods.values())).keys())
        xmax = max(ks_all)
        for method_name, ub in upper_bounds.items():
            if ub >= 0.999:
                continue
            color = METHOD_COLORS.get(method_name, "gray")
            ax.axhline(y=ub, color=color, linestyle="--",
                       linewidth=1.0, alpha=0.4, zorder=1)
            ax.text(xmax, ub, f" max={ub:.2f}",
                    color=color, alpha=0.75, fontsize=8,
                    va="center", ha="left")

    ax.set_xlabel("k")
    ax.set_ylabel("Average Pass@k")
    title = "Cliff-del vs Critical-del vs Tangent-del vs Random-del"
    if title_extra:
        title += f"  {title_extra}"
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(output_dir, f"pass_at_k_comparison{suffix}.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")

    return result


def print_experiment3_summary(exp1_result: Dict, exp2_result: Dict = None):
    """Print summary of Exp 1 and Exp 2 results."""
    print(f"\n{'='*60}")
    print("Exp 1: Cliff-del vs Cliff-keep")
    print(f"{'='*60}")
    print(f"  Success instances: {exp1_result['n_success_instances']}")
    print(f"  Failure instances: {exp1_result['n_failure_instances']}")
    for label, key in [("Cliff-del Success", "del_success"), ("Cliff-del Failure", "del_failure"),
                        ("Cliff-keep Success", "keep_success"), ("Cliff-keep Failure", "keep_failure")]:
        data = exp1_result[key]
        p1 = data.get(1, 0)
        p64 = data.get(64, 0)
        print(f"    {label}: pass@1={p1:.3f}, pass@64={p64:.3f}")

    if exp2_result:
        print(f"\n{'='*60}")
        print("Exp 2: 4-Method Comparison (failure paths)")
        print(f"{'='*60}")
        for method, data in exp2_result["methods"].items():
            p1 = data.get(1, 0)
            p64 = data.get(64, 0)
            print(f"    {method}: pass@1={p1:.3f}, pass@64={p64:.3f}")


# ============================================================
# Exp 3: Semantic Analysis
# ============================================================

TRANSITION_WORDS = {
    "however", "but", "instead", "therefore", "thus", "hence",
    "so", "then", "since", "because", "although", "yet",
    "actually", "wait", "no", "let", "now", "first",
    "next", "finally", "alternatively", "note",
    "after", "before", "while", "when", "if", "not",
    "this", "that", "here", "also", "still",
}

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be",
    "to", "of", "in", "for", "on", "with", "at", "by",
    "and", "or", "as", "from", "into", "each", "per",
    "we", "she", "he", "it", "they", "her", "his",
    "has", "have", "had", "do", "does", "did",
    "can", "will", "would", "could", "should",
}

MATH_KEYWORDS = {
    "equation", "solve", "calculate", "compute", "find",
    "substitute", "simplify", "factor", "expand", "multiply",
    "divide", "add", "subtract", "equals", "total",
    "number", "sum", "difference", "product", "remainder",
    "frac", "sqrt", "log", "sin", "cos", "tan",
    "hours", "hour", "minutes", "days", "points", "miles",
    "regular", "possible", "remaining", "currently",
}


# Binary classification: which fine-grained types are "math-related"
MATH_RELATED_TYPES = {
    "number", "math_symbol", "math_operator", "math_keyword",
    "latex_command", "variable",
}


def classify_math_binary(token_type: str) -> str:
    """Coarse binary: math-related vs non-math (based on fine-grained type)."""
    return "math" if token_type in MATH_RELATED_TYPES else "non-math"


def classify_token_type(token_str: str) -> str:
    """Classify a token into semantic categories for math reasoning analysis."""
    stripped = token_str.strip().lower()
    if not stripped:
        return "whitespace"
    # Formatting: markdown bold/headers
    if stripped in ("**", "***", "###", "##", "#", "---"):
        return "formatting"
    # Math operators
    if stripped in "+-*/=<>^%":
        return "math_operator"
    # Math symbols: brackets, dollar signs
    if stripped.replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("$", "") == "":
        return "math_symbol"
    # Numbers (with commas, decimals)
    if stripped.replace(".", "").replace("-", "").replace(",", "").isdigit():
        return "number"
    # LaTeX commands
    if stripped.startswith("\\") or stripped in ("frac", "sqrt", "boxed", "cdot", "times"):
        return "latex_command"
    # Transition / reasoning connectors
    if stripped in TRANSITION_WORDS:
        return "transition"
    # Function words (grammar glue)
    if stripped in FUNCTION_WORDS:
        return "function_word"
    # Math domain keywords
    if stripped in MATH_KEYWORDS:
        return "math_keyword"
    # Single letter = variable
    if len(stripped) == 1 and stripped.isalpha():
        return "variable"
    # Punctuation
    if stripped in ".,;:!?\\":
        return "punctuation"
    # Newlines, special tokens
    if "\n" in token_str or stripped.startswith("<"):
        return "formatting"
    # Subword / partial tokens (e.g., "'s", "-e", "arks")
    if stripped.startswith("'") or stripped.startswith("-") or (len(stripped) <= 3 and stripped.isalpha()):
        return "subword"
    return "word"


def extract_cliff_contexts(
    paths: List[Dict],
    cliff_results: List[Dict],
    window_size: int = 10,
) -> List[Dict]:
    """Extract cliff token context (before 10, cliff, after 10) for each cliff instance."""
    path_map = {p["id"]: p for p in paths}
    contexts = []

    for r in cliff_results:
        pid = r["path_id"]
        p = path_map.get(pid)
        if not p:
            continue

        tokens = p.get("response_tokens", [])
        token_ids = p.get("response_token_ids", [])
        cliff_idx = r["cliff_position"] - 1  # 0-indexed

        if cliff_idx < 0 or cliff_idx >= len(tokens):
            continue

        before_start = max(0, cliff_idx - window_size)
        after_end = min(len(tokens), cliff_idx + window_size + 1)

        cliff_token = tokens[cliff_idx]
        before_tokens = tokens[before_start:cliff_idx]
        after_tokens = tokens[cliff_idx + 1:after_end]

        contexts.append({
            "path_id": pid,
            "cliff_position": r["cliff_position"],
            "cliff_token": cliff_token,
            "cliff_token_id": token_ids[cliff_idx] if cliff_idx < len(token_ids) else None,
            "token_type": classify_token_type(cliff_token),
            "drop_magnitude": r.get("drop_magnitude", 0),
            "path_is_correct": r.get("path_is_correct", False),
            "del_num_correct": r.get("del_num_correct", 0),
            "del_success_response": r.get("del_success_response"),
            "before_tokens": " ".join(before_tokens),
            "after_tokens": " ".join(after_tokens),
            "before_10": "|".join(before_tokens),
            "cliff": cliff_token,
            "after_10": "|".join(after_tokens),
        })

    return contexts


def generate_semantic_analysis(contexts: List[Dict], output_dir: str):
    """Generate semantic analysis outputs."""
    os.makedirs(output_dir, exist_ok=True)
    _apply_style()

    if not contexts:
        print("  No cliff token contexts to analyze.")
        return

    # 1. Save full context table
    csv_path = os.path.join(output_dir, "cliff_token_contexts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=contexts[0].keys())
        writer.writeheader()
        writer.writerows(contexts)
    print(f"  Saved: {csv_path}")

    # 1b. Specific cliff token strings (raw, not category).
    # Save TWO files:
    #   - all_cliff_tokens.csv: every unique token (full dump)
    #   - top_k_cliff_tokens.csv: top 30 preview (also used for the bar chart)
    top_k = 30
    token_counter = Counter(c["cliff_token"] for c in contexts)
    # Map each token to its category for reference
    token_to_type: Dict[str, str] = {}
    for c in contexts:
        token_to_type.setdefault(c["cliff_token"], c["token_type"])
    sorted_tokens = token_counter.most_common()  # ALL, sorted by count desc
    top_tokens = sorted_tokens[:top_k]
    total_ctx = len(contexts)

    # Full table — every unique cliff token
    all_path = os.path.join(output_dir, "all_cliff_tokens.csv")
    with open(all_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token_repr", "token_type", "count", "percentage"])
        for rank, (tok, cnt) in enumerate(sorted_tokens, 1):
            w.writerow([rank, repr(tok), token_to_type.get(tok, "?"),
                        cnt, f"{cnt / total_ctx * 100:.2f}%"])
    print(f"  Saved: {all_path}  ({len(sorted_tokens)} unique tokens)")

    # Top-K preview
    top_path = os.path.join(output_dir, "top_k_cliff_tokens.csv")
    with open(top_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token_repr", "token_type", "count", "percentage"])
        for rank, (tok, cnt) in enumerate(top_tokens, 1):
            w.writerow([rank, repr(tok), token_to_type.get(tok, "?"),
                        cnt, f"{cnt / total_ctx * 100:.2f}%"])
    print(f"  Saved: {top_path}")
    print(f"\n  Top-{min(top_k, len(top_tokens))} cliff tokens (n={total_ctx}):")
    for rank, (tok, cnt) in enumerate(top_tokens[:15], 1):
        bar = "█" * max(1, int(cnt / top_tokens[0][1] * 28))
        print(f"    {rank:>2}. {repr(tok):<22} {token_to_type.get(tok, '?'):<14} "
              f"{cnt:>4} ({cnt / total_ctx * 100:>4.1f}%) {bar}")

    # Bar chart of top-K. Escape $ to prevent matplotlib mathtext parsing.
    def _safe_label(t):
        return repr(t).replace("$", r"\$")
    fig_top, ax_top = plt.subplots(figsize=(9, max(4, 0.30 * len(top_tokens))))
    labels = [f"{_safe_label(t)}  [{token_to_type.get(t, '?')}]" for t, _ in top_tokens]
    values = [c for _, c in top_tokens]
    colors_top = plt.cm.viridis(np.linspace(0.15, 0.85, len(top_tokens)))
    ax_top.barh(labels[::-1], values[::-1], color=colors_top[::-1])
    ax_top.set_xlabel("Count")
    ax_top.set_title(f"Top-{len(top_tokens)} cliff tokens (n={total_ctx})")
    for i, v in enumerate(values[::-1]):
        ax_top.text(v + max(values) * 0.01, i,
                    f"{v / total_ctx * 100:.1f}%", va="center", fontsize=8)
    fig_top.tight_layout()
    fig_top.savefig(os.path.join(output_dir, "top_k_cliff_tokens.png"),
                    bbox_inches="tight")
    plt.close(fig_top)
    print(f"  Saved: {output_dir}/top_k_cliff_tokens.png")

    # 2. Taxonomy summary
    type_counts = Counter(c["token_type"] for c in contexts)
    total = len(contexts)
    taxonomy_rows = sorted(type_counts.items(), key=lambda x: -x[1])

    tax_path = os.path.join(output_dir, "taxonomy_summary.csv")
    with open(tax_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_type", "count", "percentage"])
        for ttype, count in taxonomy_rows:
            writer.writerow([ttype, count, f"{count/total*100:.1f}%"])
    print(f"  Saved: {tax_path}")

    # Terminal print
    print(f"\n  Cliff Token Taxonomy (n={total}):")
    for ttype, count in taxonomy_rows:
        bar = "█" * int(count / total * 30)
        print(f"    {ttype:<16} {count:>4} ({count/total*100:>5.1f}%) {bar}")

    # 2b. Binary math vs non-math summary (coarse view)
    binary_counts = Counter(classify_math_binary(c["token_type"]) for c in contexts)
    n_math = binary_counts.get("math", 0)
    n_non = binary_counts.get("non-math", 0)
    bin_path = os.path.join(output_dir, "math_vs_nonmath.csv")
    with open(bin_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "percentage", "fine_grained_types"])
        w.writerow(["math", n_math, f"{n_math/total*100:.1f}%",
                    "|".join(sorted(MATH_RELATED_TYPES))])
        w.writerow(["non-math", n_non, f"{n_non/total*100:.1f}%",
                    "|".join(sorted(t for t in type_counts if t not in MATH_RELATED_TYPES))])
        w.writerow(["TOTAL", total, "100.0%", ""])
    print(f"  Saved: {bin_path}")
    print(f"  Math vs Non-math: {n_math} ({n_math/total*100:.1f}%) / "
          f"{n_non} ({n_non/total*100:.1f}%)")

    # Binary bar chart
    from matplotlib.ticker import MaxNLocator
    fig_bin, ax_bin = plt.subplots(figsize=(5, 3.5))
    bin_labels = ["math", "non-math"]
    bin_values = [n_math, n_non]
    bin_colors = ["#1E88E5", "#FB8C00"]
    bars = ax_bin.bar(bin_labels, bin_values, color=bin_colors,
                       edgecolor="white", linewidth=1.5)
    ax_bin.set_ylabel("Count")
    ax_bin.set_title(f"Cliff Token: Math vs Non-Math (n={total})")
    ax_bin.yaxis.set_major_locator(MaxNLocator(integer=True))
    for bar, v in zip(bars, bin_values):
        ax_bin.text(bar.get_x() + bar.get_width() / 2, v + max(bin_values) * 0.01,
                    f"{v} ({v/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig_bin.tight_layout()
    fig_bin.savefig(os.path.join(output_dir, "math_vs_nonmath.png"),
                    bbox_inches="tight")
    plt.close(fig_bin)
    print(f"  Saved: {output_dir}/math_vs_nonmath.png")

    # 3. Taxonomy bar chart
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(figsize=(8, 5))
    types = [t for t, _ in taxonomy_rows]
    counts = [c for _, c in taxonomy_rows]
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    ax.barh(types[::-1], counts[::-1], color=colors[::-1])
    ax.set_xlabel("Count")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Cliff Token Taxonomy (n={total})")
    for i, (t, c) in enumerate(zip(types[::-1], counts[::-1])):
        ax.text(c + 0.5, i, f"{c/total*100:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "taxonomy_chart.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/taxonomy_chart.png")

    # 4. Divergence analysis: compare original failure path vs cliff-del success path
    divergence = []
    for c in contexts:
        if not c["path_is_correct"] and c.get("del_success_response"):
            # Original response (after cliff)
            original_after = c["after_tokens"]
            # Cliff-del success response (from cliff position onward)
            del_resp = c["del_success_response"]
            # The del response starts from truncation point (before cliff)
            # Extract what was generated after the prefix
            divergence.append({
                "path_id": c["path_id"],
                "cliff_position": c["cliff_position"],
                "cliff_token": c["cliff_token"],
                "token_type": c["token_type"],
                "drop_magnitude": c["drop_magnitude"],
                "original_after_cliff": original_after,
                "del_success_response_snippet": del_resp[:500],  # first 500 chars
                "before_10": c["before_10"],
            })

    if divergence:
        div_path = os.path.join(output_dir, "divergence_analysis.csv")
        with open(div_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=divergence[0].keys())
            writer.writeheader()
            writer.writerows(divergence)
        print(f"  Saved: {div_path}")
        print(f"  Divergence cases (failure→success via cliff-del): {len(divergence)}/{len(contexts)}")
    else:
        print("  No divergence cases found (no failure→success via cliff-del).")


# ============================================================
# Plot-only mode: regenerate plots from saved JSON
# ============================================================

def replot_from_json(result_dir: str):
    """Regenerate all plots from saved JSON results (no GPU needed)."""
    _apply_style()

    # Exp 1 (support both old "exp1" and new "sub_exp_1" naming)
    exp1_json = os.path.join(result_dir, "sub_exp_1", "exp1_pass_at_k.json")
    if not os.path.exists(exp1_json):
        exp1_json = os.path.join(result_dir, "exp1", "exp1_pass_at_k.json")
    if os.path.exists(exp1_json):
        exp1 = json.load(open(exp1_json))
        # Convert string keys back to int
        for key in ["del_success", "del_failure", "keep_success", "keep_failure"]:
            exp1[key] = {int(k): v for k, v in exp1[key].items()}

        out_dir = os.path.dirname(exp1_json)
        n_s = exp1.get("n_success_instances", 0)
        n_f = exp1.get("n_failure_instances", 0)

        # Chart A: all 4 lines
        _plot_pass_at_k_chart(
            exp1,
            line_keys=["del_success", "keep_success", "del_failure", "keep_failure"],
            title="Cliff-del vs Cliff-keep",
            save_path=os.path.join(out_dir, "pass_at_k_all.png"),
            n_success=n_s, n_failure=n_f,
        )

        # Chart B: failure only
        _plot_pass_at_k_chart(
            exp1,
            line_keys=["del_failure", "keep_failure"],
            title="Cliff-del vs Cliff-keep (Failure Paths Only)",
            save_path=os.path.join(out_dir, "pass_at_k_failure.png"),
            n_failure=n_f,
        )

    # Exp 2: try each variant + legacy unsuffixed
    sub2 = os.path.join(result_dir, "sub_exp_2")
    if not os.path.isdir(sub2):
        sub2 = os.path.join(result_dir, "exp2")
    candidates = [(f"_{v}", v) for v in EXP2_VARIANTS] + [("", "legacy")]
    for suffix, label in candidates:
        exp2_json = os.path.join(sub2, f"exp2_pass_at_k{suffix}.json")
        if not os.path.exists(exp2_json):
            continue
        exp2 = json.load(open(exp2_json))
        methods = {m: {int(k): v for k, v in data.items()} for m, data in exp2["methods"].items()}
        ub = exp2.get("upper_bounds") or {}

        fig, ax = plt.subplots(figsize=(8, 5))
        for method_name, data in methods.items():
            color = METHOD_COLORS.get(method_name, "gray")
            ks = sorted(data.keys())
            vals = [data[k] for k in ks]
            ax.plot(ks, vals, marker="o", markersize=4, color=color, linewidth=2, label=method_name)

        if ub:
            xmax = max(next(iter(methods.values())).keys())
            for method_name, val in ub.items():
                if val >= 0.999:
                    continue
                color = METHOD_COLORS.get(method_name, "gray")
                ax.axhline(y=val, color=color, linestyle="--",
                           linewidth=1.0, alpha=0.4, zorder=1)
                ax.text(xmax, val, f" max={val:.2f}",
                        color=color, alpha=0.75, fontsize=8,
                        va="center", ha="left")

        ax.set_xlabel("k")
        ax.set_ylabel("Average Pass@k")
        ax.set_title(f"4-Method Comparison ({label})")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(sub2, f"pass_at_k_comparison{suffix}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Re-plotted: sub_exp_2/pass_at_k_comparison{suffix}.png")

    print("  Plot-only mode complete.")


# ============================================================
# Cross-run aggregation: per-model and model x dataset grid
# ============================================================

def _load_cliff_results(run_dir: str) -> Optional[List[Dict]]:
    """Load sub_exp_1/cliff_results.json from a run directory."""
    p = os.path.join(run_dir, "sub_exp_1", "cliff_results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"  WARN: failed to load {p}: {e}")
        return None


def aggregate_per_model(
    result_dirs: Dict[str, Dict[str, str]],
    output_dir: str,
):
    """Aggregate exp1_deletion results by model (combine datasets per model).

    Args:
        result_dirs: {model_name: {dataset_name: run_directory}}
        output_dir: output base dir for aggregated results
    """
    _apply_style()
    per_model_dir = os.path.join(output_dir, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)

    summary_rows = []  # for summary_table.csv

    for model_name, ds_map in result_dirs.items():
        # Collect cliff_results.json from each dataset for this model
        combined: List[Dict] = []
        per_dataset_n: Dict[str, int] = {}
        for dataset_name, run_dir in ds_map.items():
            cliffs = _load_cliff_results(run_dir)
            if cliffs is None:
                print(f"  WARN: no cliff_results for {model_name}/{dataset_name}")
                continue
            # Tag each entry with dataset for traceability
            for c in cliffs:
                c.setdefault("_dataset", dataset_name)
            combined.extend(cliffs)
            per_dataset_n[dataset_name] = len(cliffs)

            # Per-(model,dataset) row in summary table
            success = [r for r in cliffs if r.get("path_is_correct")]
            failure = [r for r in cliffs if not r.get("path_is_correct")]
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "N(cliff)": len(cliffs),
                "n_success": len(success),
                "n_failure": len(failure),
            }
            if cliffs:
                k_values = config.PASS_K_VALUES
                if failure:
                    fp = compute_avg_pass_at_k(failure, "del_num_correct", k_values=k_values)
                    row["Failure_del_pass@1"] = round(fp.get(1, 0), 3)
                    row["Failure_del_pass@64"] = round(fp.get(64, 0), 3)
                    fk = compute_avg_pass_at_k(failure, "keep_num_correct", k_values=k_values)
                    row["Failure_keep_pass@1"] = round(fk.get(1, 0), 3)
                    row["Failure_keep_pass@64"] = round(fk.get(64, 0), 3)
            summary_rows.append(row)

        if not combined:
            continue

        # Per-model aggregated chart (uses run_experiment1_evaluation logic)
        model_out = os.path.join(per_model_dir, model_name)
        os.makedirs(model_out, exist_ok=True)
        print(f"\n  [{model_name}] aggregated: {len(combined)} cliffs across {len(per_dataset_n)} datasets")
        for ds, n in per_dataset_n.items():
            print(f"    {ds}: {n}")

        # Reuse the existing exp1 evaluation function
        result = run_experiment1_evaluation(combined, model_out)
        # Save per-dataset breakdown alongside
        json.dump(per_dataset_n, open(os.path.join(model_out, "per_dataset_n.json"), "w"), indent=2)

    # Save the summary table
    if summary_rows:
        # Use union of keys (some rows may not have failure metrics)
        all_keys = set()
        for r in summary_rows:
            all_keys.update(r.keys())
        ordered = ["Model", "Dataset", "N(cliff)", "n_success", "n_failure",
                    "Failure_del_pass@1", "Failure_del_pass@64",
                    "Failure_keep_pass@1", "Failure_keep_pass@64"]
        ordered = [k for k in ordered if k in all_keys]
        with open(os.path.join(output_dir, "summary_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in ordered})
        print(f"\n  Saved: {os.path.join(output_dir, 'summary_table.csv')}")


EXP2_VARIANTS = ["population_tangent", "cliff_first", "cliff_random", "all_failure"]


def aggregate_semantic_all(
    result_dirs: Dict[str, Dict[str, str]],
    output_dir: str,
):
    """Cross-run semantic aggregation: union ALL cliff token contexts from
    every (model, dataset) run into one combined sub_exp_3 analysis.

    Each run's sub_exp_3_semantic/cliff_token_contexts.csv contains the
    per-cliff token context that `extract_cliff_contexts()` produced.
    We concatenate them, tag with model+dataset, and re-run
    `generate_semantic_analysis()` on the union.
    """
    rows: List[Dict] = []
    n_runs_with_data = 0
    for model_name, ds_map in result_dirs.items():
        for ds_name, run_dir in ds_map.items():
            csv_path = os.path.join(run_dir, "sub_exp_3_semantic",
                                    "cliff_token_contexts.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    run_rows = list(reader)
            except Exception as e:
                print(f"  WARN: failed to read {csv_path}: {e}")
                continue
            if not run_rows:
                continue
            n_runs_with_data += 1
            # Tag with model + dataset for traceability + restore types
            for r in run_rows:
                r["_model"] = model_name
                r["_dataset"] = ds_name
                if "path_is_correct" in r:
                    r["path_is_correct"] = (
                        str(r["path_is_correct"]).strip().lower() == "true"
                    )
                if "del_num_correct" in r:
                    try:
                        r["del_num_correct"] = int(r["del_num_correct"])
                    except (TypeError, ValueError):
                        r["del_num_correct"] = 0
                if "drop_magnitude" in r:
                    try:
                        r["drop_magnitude"] = float(r["drop_magnitude"])
                    except (TypeError, ValueError):
                        r["drop_magnitude"] = 0.0
            rows.extend(run_rows)

    print(f"\n  [semantic-all] {len(rows)} cliffs from {n_runs_with_data} runs")
    if not rows:
        print("  No semantic context data found; skipping aggregate semantic.")
        return

    out = os.path.join(output_dir, "semantic_all")
    os.makedirs(out, exist_ok=True)
    generate_semantic_analysis(rows, out)


def aggregate_per_model_exp2(
    result_dirs: Dict[str, Dict[str, str]],
    output_dir: str,
):
    """Per-model aggregation for sub_exp_2 across all 4 variants.

    For each model and variant, concat per-instance results
    (cliff_del, critical_del, tangent_del, random_del) from each dataset's
    sub_exp_2/<method>_results_<variant>.json and re-run
    run_experiment2_evaluation on the combined set.
    """
    per_model_dir = os.path.join(output_dir, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)

    for variant in EXP2_VARIANTS:
        for model_name, ds_map in result_dirs.items():
            cliff_all, critical_all, tangent_all, random_all = [], [], [], []
            loaded_any = False
            for ds_name, run_dir in ds_map.items():
                base = os.path.join(run_dir, "sub_exp_2")

                def _load(name):
                    p = os.path.join(base, f"{name}_results_{variant}.json")
                    if not os.path.exists(p):
                        return None
                    try:
                        return json.load(open(p))
                    except Exception as e:
                        print(f"  WARN: failed to load {p}: {e}")
                        return None

                c = _load("cliff_del")
                cr = _load("critical_del")
                tg = _load("tangent_del")
                rn = _load("random_del")
                if c is None:
                    continue
                loaded_any = True
                cliff_all.extend(c)
                critical_all.extend(cr or [])
                tangent_all.extend(tg or [])
                random_all.extend(rn or [])

            if not loaded_any or not cliff_all:
                continue

            n_total = len(cliff_all)
            ub = {
                "Cliff-del":    sum(1 for r in cliff_all    if not r.get("_stub")) / n_total,
                "Critical-del": sum(1 for r in critical_all if not r.get("_stub")) / n_total,
                "Tangent-del":  sum(1 for r in tangent_all  if not r.get("_stub")) / n_total,
                "Random-del":   sum(1 for r in random_all   if not r.get("_stub")) / n_total,
            }
            out = os.path.join(per_model_dir, model_name)
            os.makedirs(out, exist_ok=True)
            print(f"\n  [{model_name}/{variant}] aggregated: n={n_total}")
            run_experiment2_evaluation(
                cliff_all, critical_all, tangent_all, random_all,
                out, suffix=f"_{variant}",
                title_extra=f"({model_name}, {variant}, n={n_total})",
                upper_bounds=ub,
            )


def plot_model_dataset_grid(
    result_dirs: Dict[str, Dict[str, str]],
    output_path: str,
    all_models: List[str],
    all_datasets: List[str],
):
    """Grid of pass@k charts: rows = models, cols = datasets.

    Distinguishes three cell states:
      - no rollout (file absent)        → gray "(no rollout)"
      - rollout exists, 0 cliffs found  → orange "(0 cliffs)"
      - normal                          → 4-line pass@k plot

    Rows/cols are sorted by MODEL_DISPLAY_ORDER / DATASET_DISPLAY_ORDER.
    Figure-level legend explains the 4 line colors.
    """
    _apply_style()
    all_models = _ordered(all_models, MODEL_DISPLAY_ORDER)
    all_datasets = _ordered(all_datasets, DATASET_DISPLAY_ORDER)

    n_rows = len(all_models)
    n_cols = len(all_datasets)
    if n_rows == 0 or n_cols == 0:
        print("  Empty grid, skipping.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                              squeeze=False)
    k_values = config.PASS_K_VALUES
    K_TICKS = [1, 2, 4, 8, 16, 32, 64]
    Y_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def _apply_axes_style(ax):
        """Force consistent ticks/grid on every cell (data or placeholder)."""
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_TICKS)
        ax.set_xticklabels([str(k) for k in K_TICKS], fontsize=9)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels([f"{y:.1f}" for y in Y_TICKS], fontsize=9)
        ax.set_xlim(K_TICKS[0] * 0.85, K_TICKS[-1] * 1.18)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="major", axis="both",
                color="#CCCCCC", linewidth=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    for i, model in enumerate(all_models):
        for j, dataset in enumerate(all_datasets):
            ax = axes[i][j]
            run_dir = result_dirs.get(model, {}).get(dataset)
            cliffs = _load_cliff_results(run_dir) if run_dir else None

            _apply_axes_style(ax)

            if cliffs is None:
                # No rollout / file not found
                ax.text(0.5, 0.5, "(no rollout)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="gray",
                        zorder=5)
            elif len(cliffs) == 0:
                # Rollout exists but no statistical cliff token detected
                ax.text(0.5, 0.5, "(0 cliffs)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="#FB8C00",
                        fontweight="bold", zorder=5)
            else:
                success = [r for r in cliffs if r.get("path_is_correct")]
                failure = [r for r in cliffs if not r.get("path_is_correct")]
                data = {
                    "del_success":  compute_avg_pass_at_k(success, "del_num_correct", k_values=k_values),
                    "del_failure":  compute_avg_pass_at_k(failure, "del_num_correct", k_values=k_values),
                    "keep_success": compute_avg_pass_at_k(success, "keep_num_correct", k_values=k_values),
                    "keep_failure": compute_avg_pass_at_k(failure, "keep_num_correct", k_values=k_values),
                }
                for key in ["del_success", "keep_success", "del_failure", "keep_failure"]:
                    style = EXP1_LINES[key]
                    d = data.get(key, {})
                    if not d:
                        continue
                    ks = sorted(d.keys())
                    vals = [d[k] for k in ks]
                    ax.plot(ks, vals,
                            color=style["color"],
                            linestyle="-", linewidth=1.5,
                            marker=style["marker"], markersize=4,
                            zorder=3)
                # Bottom-right n label (so plot lines stay unobstructed)
                ax.text(0.98, 0.02, f"n(cliff tokens)={len(cliffs)}",
                        transform=ax.transAxes, fontsize=8,
                        ha="right", va="bottom", color="#555555", zorder=5)

            # Headers / row labels
            if i == 0:
                ax.set_title(dataset, fontsize=11)
            if j == 0:
                ax.set_ylabel(model, fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("k")

    # Figure-level legend explaining the 4 lines
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0],
               color=EXP1_LINES[k]["color"],
               marker=EXP1_LINES[k]["marker"],
               linestyle="-", linewidth=1.8, markersize=6,
               label=EXP1_LINES[k]["label"])
        for k in ["del_success", "keep_success", "del_failure", "keep_failure"]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.015), frameon=True, fontsize=10)

    fig.suptitle("Cliff-del vs Cliff-keep pass@k (model × dataset)",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================
# Failure-only grid + CSV/MD (formerly _replot_exp1_deletion_failure_only.py)
# ============================================================

_GRID_K_TICKS = [1, 2, 4, 8, 16, 32, 64]
_GRID_Y_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_MD_SEP_STYLE = ' style="border-right:2px solid #333"'


def _failure_grid_axes_style(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(_GRID_K_TICKS)
    ax.set_xticklabels([str(k) for k in _GRID_K_TICKS], fontsize=9)
    ax.set_yticks(_GRID_Y_TICKS)
    ax.set_yticklabels([f"{y:.1f}" for y in _GRID_Y_TICKS], fontsize=9)
    ax.set_xlim(_GRID_K_TICKS[0] * 0.85, _GRID_K_TICKS[-1] * 1.18)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", axis="both",
            color="#CCCCCC", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_failure_only_grid(
    result_dirs: Dict[str, Dict[str, str]],
    all_models: List[str],
    all_datasets: List[str],
    output_dir: str,
):
    """Failure-path-only Cliff-del/Cliff-keep grid + CSV + Markdown.

    Produces under output_dir:
        pass_at_k_grid_failure_only.png            (rows=models, cols=datasets)
        pass_at_k_grid_failure_only_transposed.png (rows=datasets, cols=models)
        pass_at_k_failure_only.csv                 (long-form)
        pass_at_k_failure_only.md                  (HTML tables, multi-level)
    """
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)
    all_models = _ordered(all_models, MODEL_DISPLAY_ORDER)
    all_datasets = _ordered(all_datasets, DATASET_DISPLAY_ORDER)
    k_values = config.PASS_K_VALUES

    pass_data: Dict[str, Dict[str, Optional[Dict[str, Dict[int, float]]]]] = {}
    n_failures: Dict[str, Dict[str, int]] = {}
    n_total: Dict[str, Dict[str, int]] = {}
    status: Dict[str, Dict[str, str]] = {}

    for model in all_models:
        pass_data[model] = {}
        n_failures[model] = {}
        n_total[model] = {}
        status[model] = {}
        for ds in all_datasets:
            run_dir = result_dirs.get(model, {}).get(ds)
            cliffs = _load_cliff_results(run_dir) if run_dir else None
            if cliffs is None:
                pass_data[model][ds] = None
                n_failures[model][ds] = 0
                n_total[model][ds] = 0
                status[model][ds] = "no_rollout"
                continue
            if len(cliffs) == 0:
                pass_data[model][ds] = None
                n_failures[model][ds] = 0
                n_total[model][ds] = 0
                status[model][ds] = "no_cliffs"
                continue
            failure = [r for r in cliffs if not r.get("path_is_correct")]
            n_total[model][ds] = len(cliffs)
            n_failures[model][ds] = len(failure)
            if not failure:
                pass_data[model][ds] = None
                status[model][ds] = "no_failures"
                continue
            pass_data[model][ds] = {
                "del": compute_avg_pass_at_k(failure, "del_num_correct", k_values=k_values),
                "keep": compute_avg_pass_at_k(failure, "keep_num_correct", k_values=k_values),
            }
            status[model][ds] = "ok"

    placeholder = {
        "no_rollout": ("(no rollout)", "gray"),
        "no_cliffs": ("(0 cliffs)", "#FB8C00"),
        "no_failures": ("(no failures)", "gray"),
    }

    def _draw_grid(row_items, col_items, get_md, out_png, row_label, col_label):
        _apply_style()
        n_rows = len(row_items)
        n_cols = len(col_items)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 3 * n_rows),
                                  squeeze=False)
        for i, ri in enumerate(row_items):
            for j, ci in enumerate(col_items):
                ax = axes[i][j]
                _failure_grid_axes_style(ax)
                model, ds = get_md(ri, ci)
                st = status[model][ds]
                d = pass_data[model][ds]
                if d is None:
                    text, color = placeholder[st]
                    weight = "bold" if st == "no_cliffs" else "normal"
                    ax.text(0.5, 0.5, text, ha="center", va="center",
                            transform=ax.transAxes, fontsize=11, color=color,
                            fontweight=weight, zorder=5)
                else:
                    for line_key, src_key in [("del_failure", "del"),
                                              ("keep_failure", "keep")]:
                        style = EXP1_LINES[line_key]
                        vals_dict = d[src_key]
                        ks = sorted(vals_dict.keys())
                        vals = [vals_dict[k] for k in ks]
                        ax.plot(ks, vals,
                                color=style["color"],
                                linestyle="-", linewidth=1.5,
                                marker=style["marker"], markersize=4,
                                zorder=3)
                    ax.text(0.98, 0.02,
                            f"n(failure)={n_failures[model][ds]}/{n_total[model][ds]}",
                            transform=ax.transAxes, fontsize=8,
                            ha="right", va="bottom", color="#555555", zorder=5)
                if i == 0:
                    ax.set_title(str(ci), fontsize=11)
                if j == 0:
                    ax.set_ylabel(str(ri), fontsize=10)
                if i == n_rows - 1:
                    ax.set_xlabel("k")

        legend_handles = [
            Line2D([0], [0],
                   color=EXP1_LINES[k]["color"],
                   marker=EXP1_LINES[k]["marker"],
                   linestyle="-", linewidth=1.8, markersize=6,
                   label=EXP1_LINES[k]["label"])
            for k in ["del_failure", "keep_failure"]
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.015), frameon=True, fontsize=10)
        fig.suptitle(
            f"Cliff-del vs Cliff-keep pass@k (failure paths only) "
            f"— rows: {row_label}, cols: {col_label}",
            fontsize=13, y=1.00,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png}")

    _draw_grid(
        row_items=all_models, col_items=all_datasets,
        get_md=lambda m, d: (m, d),
        out_png=os.path.join(output_dir, "pass_at_k_grid_failure_only.png"),
        row_label="model", col_label="dataset",
    )
    _draw_grid(
        row_items=all_datasets, col_items=all_models,
        get_md=lambda d, m: (m, d),
        out_png=os.path.join(output_dir, "pass_at_k_grid_failure_only_transposed.png"),
        row_label="dataset", col_label="model",
    )

    # ---- CSV ----
    csv_path = os.path.join(output_dir, "pass_at_k_failure_only.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "dataset", "k",
            "cliff_del_pass_at_k", "cliff_keep_pass_at_k",
            "n_failure", "n_total_cliffs",
        ])
        for model in all_models:
            for ds in all_datasets:
                d = pass_data[model][ds]
                if d is None:
                    continue
                for k in k_values:
                    w.writerow([
                        model, ds, k,
                        f"{d['del'].get(k, 0):.4f}",
                        f"{d['keep'].get(k, 0):.4f}",
                        n_failures[model][ds],
                        n_total[model][ds],
                    ])
    print(f"  Saved: {csv_path}")

    # ---- Markdown ----
    md_path = os.path.join(output_dir, "pass_at_k_failure_only.md")

    def _cell(d, k, sep=False):
        sep_attr = _MD_SEP_STYLE if sep else ""
        if d is None:
            return f"<td>—</td><td{sep_attr}>—</td>"
        return (f"<td>{d['del'].get(k, 0):.3f}</td>"
                f"<td{sep_attr}>{d['keep'].get(k, 0):.3f}</td>")

    lines = []
    lines.append("# Pass@k on failure paths only\n")
    lines.append(
        "Cliff-del vs Cliff-keep pass@k computed only over reasoning paths "
        "whose original rollout was incorrect (`path_is_correct == false`).\n"
    )

    lines.append("## Combined (Model x Dataset x Variant)\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="3">k</th>'
    for model in all_models:
        hdr1 += (f'<th colspan="{2 * len(all_datasets)}" align="center"'
                 f'{_MD_SEP_STYLE}>{model}</th>')
    hdr1 += "</tr>"
    lines.append(hdr1)
    hdr2 = "<tr>"
    for _ in all_models:
        for di, ds in enumerate(all_datasets):
            sep = _MD_SEP_STYLE if di == len(all_datasets) - 1 else ""
            hdr2 += f'<th colspan="2" align="center"{sep}>{ds}</th>'
    hdr2 += "</tr>"
    lines.append(hdr2)
    hdr3 = "<tr>"
    for _ in all_models:
        for di, _ in enumerate(all_datasets):
            sep = _MD_SEP_STYLE if di == len(all_datasets) - 1 else ""
            hdr3 += f'<th>Cliff-del</th><th{sep}>Cliff-keep</th>'
    hdr3 += "</tr>"
    lines.append(hdr3)
    for k in k_values:
        row = f"<tr><td><b>{k}</b></td>"
        for model in all_models:
            for di, ds in enumerate(all_datasets):
                row += _cell(pass_data[model][ds], k,
                             sep=(di == len(all_datasets) - 1))
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    for ds in all_datasets:
        lines.append(f"## {ds}\n")
        lines.append("<table>")
        hdr1 = '<tr><th rowspan="2">k</th>'
        for model in all_models:
            hdr1 += f'<th colspan="2" align="center"{_MD_SEP_STYLE}>{model}</th>'
        hdr1 += "</tr>"
        lines.append(hdr1)
        hdr2 = "<tr>"
        for _ in all_models:
            hdr2 += f'<th>Cliff-del</th><th{_MD_SEP_STYLE}>Cliff-keep</th>'
        hdr2 += "</tr>"
        lines.append(hdr2)
        for k in k_values:
            row = f"<tr><td><b>{k}</b></td>"
            for model in all_models:
                row += _cell(pass_data[model][ds], k, sep=True)
            row += "</tr>"
            lines.append(row)
        lines.append("</table>\n")

    lines.append("## n(failure) per (model, dataset)\n")
    lines.append("<table>")
    hdr = "<tr><th>model \\ dataset</th>"
    for ds in all_datasets:
        hdr += f"<th>{ds}</th>"
    hdr += "</tr>"
    lines.append(hdr)
    for model in all_models:
        row = f"<tr><td>{model}</td>"
        for ds in all_datasets:
            row += f"<td>{n_failures[model][ds]} / {n_total[model][ds]}</td>"
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")


# ============================================================
# Exp2 methods grid + CSV/MD (formerly _plot_exp1_deletion_exp2_methods_grid.py)
# ============================================================

_EXP2_METHOD_MARKERS = {
    "Cliff-del":    "o",
    "Critical-del": "s",
    "Tangent-del":  "^",
    "Random-del":   "D",
}


def _exp2_grid_axes_style(ax):
    ax.set_xticks(_GRID_K_TICKS)
    ax.set_xticklabels([str(k) for k in _GRID_K_TICKS], fontsize=9)
    ax.set_yticks(_GRID_Y_TICKS)
    ax.set_yticklabels([f"{y:.1f}" for y in _GRID_Y_TICKS], fontsize=9)
    ax.set_xlim(0, _GRID_K_TICKS[-1] + 2)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", axis="both",
            color="#CCCCCC", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_exp2_methods_grid(
    result_dirs: Dict[str, Dict[str, str]],
    all_models: List[str],
    all_datasets: List[str],
    output_dir: str,
):
    """Exp2 (Cliff/Critical/Tangent/Random)-del pass@k grid on all-failure subset.

    Produces under output_dir:
        exp2_methods_grid_all_failure.png            (4 lines: incl. Random-del)
        exp2_methods_grid_all_failure_no_random.png  (3 lines: no Random-del)
        exp2_methods_all_failure.csv                 (long-form, all 4 methods)
        exp2_methods_all_failure.md                  (HTML tables, 3 inner cols)
    """
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)
    all_models = _ordered(all_models, MODEL_DISPLAY_ORDER)
    all_datasets = _ordered(all_datasets, DATASET_DISPLAY_ORDER)
    k_values = config.PASS_K_VALUES
    methods_all = ["Cliff-del", "Critical-del", "Tangent-del", "Random-del"]

    def _load_exp2(run_dir):
        p = os.path.join(run_dir, "sub_exp_2", "exp2_pass_at_k_all_failure.json")
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p))
        except Exception as e:
            print(f"  WARN: failed to load {p}: {e}")
            return None

    pass_data: Dict[str, Dict[str, Optional[Dict[str, Dict[int, float]]]]] = {}
    avg_data: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    n_paths: Dict[str, Dict[str, int]] = {}
    for model in all_models:
        pass_data[model] = {}
        avg_data[model] = {}
        n_paths[model] = {}
        for ds in all_datasets:
            run_dir = result_dirs.get(model, {}).get(ds)
            data = _load_exp2(run_dir) if run_dir else None
            if data is None or "methods" not in data:
                pass_data[model][ds] = None
                avg_data[model][ds] = None
                n_paths[model][ds] = 0
                continue
            methods = data["methods"]
            out = {}
            for m in methods_all:
                if m not in methods:
                    continue
                out[m] = {int(k): float(v) for k, v in methods[m].items()}
            pass_data[model][ds] = out if out else None
            avg_block = data.get("avg_methods", {})
            avg_data[model][ds] = {m: float(v) for m, v in avg_block.items()} if avg_block else None
            np_block = data.get("n_paths", {})
            n_paths[model][ds] = max(np_block.values()) if np_block else 0

    def _draw(methods_to_plot, out_png, title_suffix):
        _apply_style()
        n_rows = len(all_datasets)
        n_cols = len(all_models)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 3 * n_rows),
                                  squeeze=False)
        for i, ds in enumerate(all_datasets):
            for j, model in enumerate(all_models):
                ax = axes[i][j]
                _exp2_grid_axes_style(ax)
                d = pass_data[model][ds]
                if d is None:
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                            transform=ax.transAxes, fontsize=11, color="gray",
                            zorder=5)
                else:
                    for m in methods_to_plot:
                        if m not in d:
                            continue
                        ks = sorted(d[m].keys())
                        vals = [d[m][k] for k in ks]
                        ax.plot(ks, vals,
                                color=METHOD_COLORS[m],
                                linestyle="-", linewidth=1.5,
                                marker=_EXP2_METHOD_MARKERS[m], markersize=4,
                                zorder=3)
                    ax.text(0.98, 0.02, f"n={n_paths[model][ds]}",
                            transform=ax.transAxes, fontsize=8,
                            ha="right", va="bottom", color="#555555", zorder=5)
                if i == 0:
                    ax.set_title(model, fontsize=11)
                if j == 0:
                    ax.set_ylabel(ds, fontsize=10)
                if i == n_rows - 1:
                    ax.set_xlabel("k")

        legend_handles = [
            Line2D([0], [0],
                   color=METHOD_COLORS[m],
                   marker=_EXP2_METHOD_MARKERS[m],
                   linestyle="-", linewidth=1.8, markersize=6,
                   label=m)
            for m in methods_to_plot
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(methods_to_plot),
                   bbox_to_anchor=(0.5, -0.015), frameon=True, fontsize=10)
        fig.suptitle(
            f"Average pass@k on all-failure subset {title_suffix} "
            f"— rows: dataset, cols: model",
            fontsize=13, y=1.00,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png}")

    _draw(
        methods_to_plot=["Cliff-del", "Critical-del", "Tangent-del", "Random-del"],
        out_png=os.path.join(output_dir, "exp2_methods_grid_all_failure.png"),
        title_suffix="(Cliff/Critical/Tangent/Random-del)",
    )
    _draw(
        methods_to_plot=["Cliff-del", "Critical-del", "Tangent-del"],
        out_png=os.path.join(output_dir, "exp2_methods_grid_all_failure_no_random.png"),
        title_suffix="(Cliff/Critical/Tangent-del, no Random)",
    )

    # ---- CSV (long-form, all 4 methods) ----
    csv_path = os.path.join(output_dir, "exp2_methods_all_failure.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "dataset", "k",
            "cliff_del", "critical_del", "tangent_del", "random_del",
            "n_paths",
        ])
        for model in all_models:
            for ds in all_datasets:
                d = pass_data[model][ds]
                if d is None:
                    continue
                for k in k_values:
                    w.writerow([
                        model, ds, k,
                        f"{d.get('Cliff-del', {}).get(k, 0):.4f}",
                        f"{d.get('Critical-del', {}).get(k, 0):.4f}",
                        f"{d.get('Tangent-del', {}).get(k, 0):.4f}",
                        f"{d.get('Random-del', {}).get(k, 0):.4f}",
                        n_paths[model][ds],
                    ])
    print(f"  Saved: {csv_path}")

    # ---- Markdown (3 inner cols) ----
    md_path = os.path.join(output_dir, "exp2_methods_all_failure.md")
    inner = ["Cliff-del", "Critical-del", "Tangent-del"]

    def _cells(d, k, sep_last=False):
        parts = []
        for idx, m in enumerate(inner):
            sep = _MD_SEP_STYLE if (sep_last and idx == len(inner) - 1) else ""
            if d is None or m not in d:
                parts.append(f"<td{sep}>—</td>")
            else:
                parts.append(f"<td{sep}>{d[m].get(k, 0):.3f}</td>")
        return "".join(parts)

    lines = []
    lines.append("# Exp2 methods pass@k on all-failure subset\n")
    lines.append(
        "Average pass@k for Cliff-del / Critical-del / Tangent-del on the "
        "all-failure subset of cliff paths. Random-del is omitted from the "
        "tables (see CSV for the full 4-method data).\n"
    )

    lines.append("## Combined (Model x Dataset x Variant)\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="3">k</th>'
    for model in all_models:
        hdr1 += (f'<th colspan="{len(inner) * len(all_datasets)}" '
                 f'align="center"{_MD_SEP_STYLE}>{model}</th>')
    hdr1 += "</tr>"
    lines.append(hdr1)
    hdr2 = "<tr>"
    for _ in all_models:
        for di, ds in enumerate(all_datasets):
            sep = _MD_SEP_STYLE if di == len(all_datasets) - 1 else ""
            hdr2 += f'<th colspan="{len(inner)}" align="center"{sep}>{ds}</th>'
    hdr2 += "</tr>"
    lines.append(hdr2)
    hdr3 = "<tr>"
    for _ in all_models:
        for di, _ in enumerate(all_datasets):
            last_ds = (di == len(all_datasets) - 1)
            for vi, v in enumerate(inner):
                sep = _MD_SEP_STYLE if (last_ds and vi == len(inner) - 1) else ""
                hdr3 += f"<th{sep}>{v}</th>"
    hdr3 += "</tr>"
    lines.append(hdr3)
    for k in k_values:
        row = f"<tr><td><b>{k}</b></td>"
        for model in all_models:
            for di, ds in enumerate(all_datasets):
                row += _cells(pass_data[model][ds], k,
                              sep_last=(di == len(all_datasets) - 1))
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    for ds in all_datasets:
        lines.append(f"## {ds}\n")
        lines.append("<table>")
        hdr1 = '<tr><th rowspan="2">k</th>'
        for model in all_models:
            hdr1 += (f'<th colspan="{len(inner)}" align="center"'
                     f'{_MD_SEP_STYLE}>{model}</th>')
        hdr1 += "</tr>"
        lines.append(hdr1)
        hdr2 = "<tr>"
        for _ in all_models:
            for vi, v in enumerate(inner):
                sep = _MD_SEP_STYLE if vi == len(inner) - 1 else ""
                hdr2 += f"<th{sep}>{v}</th>"
        hdr2 += "</tr>"
        lines.append(hdr2)
        for k in k_values:
            row = f"<tr><td><b>{k}</b></td>"
            for model in all_models:
                row += _cells(pass_data[model][ds], k, sep_last=True)
            row += "</tr>"
            lines.append(row)
        lines.append("</table>\n")

    lines.append("## pass@1 summary (rows: dataset, cols: model x variant)\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="2">dataset</th>'
    for model in all_models:
        hdr1 += (f'<th colspan="{len(inner)}" align="center"'
                 f'{_MD_SEP_STYLE}>{model}</th>')
    hdr1 += "</tr>"
    lines.append(hdr1)
    hdr2 = "<tr>"
    for _ in all_models:
        for vi, v in enumerate(inner):
            sep = _MD_SEP_STYLE if vi == len(inner) - 1 else ""
            hdr2 += f"<th{sep}>{v}</th>"
    hdr2 += "</tr>"
    lines.append(hdr2)
    for ds in all_datasets:
        row = f"<tr><td><b>{ds}</b></td>"
        for model in all_models:
            d = pass_data[model][ds]
            for vi, m in enumerate(inner):
                sep = _MD_SEP_STYLE if vi == len(inner) - 1 else ""
                if d is None or m not in d:
                    row += f"<td{sep}>—</td>"
                else:
                    row += f"<td{sep}>{d[m].get(1, 0):.3f}</td>"
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    # ---- avg@64 summary (= mean c/n across failure problems) ----
    lines.append("## avg@64 summary (rows: dataset, cols: model x variant)\n")
    lines.append("<table>")
    hdr1 = '<tr><th rowspan="2">dataset</th>'
    for model in all_models:
        hdr1 += (f'<th colspan="{len(inner)}" align="center"'
                 f'{_MD_SEP_STYLE}>{model}</th>')
    hdr1 += "</tr>"
    lines.append(hdr1)
    hdr2 = "<tr>"
    for _ in all_models:
        for vi, v in enumerate(inner):
            sep = _MD_SEP_STYLE if vi == len(inner) - 1 else ""
            hdr2 += f"<th{sep}>{v}</th>"
    hdr2 += "</tr>"
    lines.append(hdr2)
    for ds in all_datasets:
        row = f"<tr><td><b>{ds}</b></td>"
        for model in all_models:
            a = avg_data[model][ds]
            for vi, m in enumerate(inner):
                sep = _MD_SEP_STYLE if vi == len(inner) - 1 else ""
                if a is None or m not in a:
                    row += f"<td{sep}>—</td>"
                else:
                    row += f"<td{sep}>{a[m]:.3f}</td>"
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    lines.append("## n(paths) per (model, dataset)\n")
    lines.append("<table>")
    hdr = "<tr><th>model \\ dataset</th>"
    for ds in all_datasets:
        hdr += f"<th>{ds}</th>"
    hdr += "</tr>"
    lines.append(hdr)
    for model in all_models:
        row = f"<tr><td>{model}</td>"
        for ds in all_datasets:
            row += f"<td>{n_paths[model][ds]}</td>"
        row += "</tr>"
        lines.append(row)
    lines.append("</table>\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")
