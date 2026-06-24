"""
RQ2-1: Entropy threshold computation and 2x2 analysis (Cliff vs Greedy x High/Low Entropy).

Phase A: compute_dataset_entropy_threshold — top-20 logprob entropy at every token in dataset
Phase D: 2x2 analysis, plots, tables
"""

import os
import csv
import json
import math
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vllm import SamplingParams


# ============================================================
# Style
# ============================================================
SUCCESS_COLOR = "#90CAF9"
FAILURE_COLOR = "#EF9A9A"
CASE_A_COLOR = "#1E88E5"
CASE_B_COLOR = "#E53935"


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    })
    sns.set_style("whitegrid")


# ============================================================
# Entropy computation utilities
# ============================================================

def _binary_entropy_nats(p: float) -> float:
    """Binary entropy H_b(p) in nats. Returns 0 for degenerate p."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


# ----------------------------------------------------------------
# Taxonomy entropy boundary (canonical source of truth)
# ----------------------------------------------------------------
# Threshold probability for "effectively deterministic" classification.
# Deterministic/uncertain/sampled_off taxonomy (and exp3_entropy 2x2 cells) split tokens by whether entropy
# is <= H_b(GREEDY_PROB_THRESHOLD). Swap this single constant to experiment
# with alternative boundaries (e.g., 0.95 → H_b(0.95) ≈ 0.1985 nats).
GREEDY_PROB_THRESHOLD = 0.99
GREEDY_BOUND_NATS = _binary_entropy_nats(GREEDY_PROB_THRESHOLD)

# Legacy alias: older callers import `GREEDY_99_BOUND_NATS`. Kept for
# backward compatibility — value tracks whatever GREEDY_PROB_THRESHOLD is,
# so the name becomes stale if the threshold is changed. Prefer
# `GREEDY_BOUND_NATS` in new code.
GREEDY_99_BOUND_NATS = GREEDY_BOUND_NATS

# Reference boundaries (fixed regardless of GREEDY_PROB_THRESHOLD). Used as
# dotted reference lines on plots or for alternative-threshold ablations.
GREEDY_95_BOUND_NATS = _binary_entropy_nats(0.95)   # ≈ 0.1985 nats
GREEDY_90_BOUND_NATS = _binary_entropy_nats(0.90)   # ≈ 0.3251 nats


def _logprob_value(lp_entry) -> float:
    """Extract float logprob from vLLM Logprob object or float."""
    if hasattr(lp_entry, "logprob"):
        return lp_entry.logprob
    return float(lp_entry)


def compute_tie_aware_ranks(logprob_dict: Dict, eps: float = 1e-9) -> Dict[int, int]:
    """Competition rank (1, 1, 3, 4) by descending logprob; ties share the
    smaller rank.

    vLLM's `entry.rank` arbitrarily breaks probability ties, assigning rank=1
    to one tied-top token and rank=2 to another. Downstream `rank == 1`
    classification (deterministic/uncertain/sampled_off taxonomy, exp2 baseline cells) then misclassifies
    the latter as non-greedy. This helper recomputes ranks so all tokens
    within `eps` of the top logprob share rank=1.

    Returns: {token_id: rank} for every token in `logprob_dict`. Callers must
    fall back to (k+1) for tokens absent from the dict.
    """
    if not logprob_dict:
        return {}
    sorted_entries = sorted(
        ((tid, _logprob_value(e)) for tid, e in logprob_dict.items()),
        key=lambda x: -x[1],
    )
    ranks: Dict[int, int] = {}
    rank = 1
    prev_lp = sorted_entries[0][1]
    for i, (tid, lp) in enumerate(sorted_entries):
        if (prev_lp - lp) > eps:
            rank = i + 1  # competition rank: skip over tied positions
            prev_lp = lp
        ranks[tid] = rank
    return ranks


def compute_entropy_from_logprobs(logprob_dict: Dict) -> float:
    """Shannon entropy lower bound from top-k logprobs (partial sum).

    H_partial = -sum p_i * log(p_i)  over the top-k tokens, WITHOUT
    renormalizing. This is a lower bound on the true full-vocab entropy:
    it omits the positive contribution from tail tokens (vocab \\ top-k).
    Standard in API-restricted uncertainty literature
    (e.g., Shi et al. 2024 "Min-K% Prob"). Returns 0.0 for empty dict.
    """
    if not logprob_dict:
        return 0.0
    h = 0.0
    for v in logprob_dict.values():
        lp = _logprob_value(v)
        if lp == float("-inf"):
            continue
        p = math.exp(lp)
        if p > 0:
            # -p * log(p)  ==  -p * lp  (since log p == lp)
            h -= p * lp
    return h


# ============================================================
# Phase A: Dataset entropy threshold
# ============================================================

def compute_dataset_entropy_threshold(
    llm,
    tokenizer,
    paths_by_dataset: Dict[str, List[Dict]],
    prompt_logprobs_k: int = 20,
    percentiles: List[int] = (60, 70, 80, 90),
    max_tokens_per_batch: int = 4000,
    max_seq_len: int = 4000,
) -> Dict:
    """Compute entropy at every response token, return percentile thresholds.

    Args:
        paths_by_dataset: {"gsm1k": [paths], "math500": [paths], ...}

    Returns:
        {
            "combined": {"60": float, "70": ..., "80": ..., "90": ...},
            "per_dataset": {"gsm1k": {...}, ...},
            "all_entropies": [float, ...],  # combined
            "per_dataset_entropies": {ds: [...]}
        }
    """
    print(f"\n[Phase A] Computing entropy thresholds (top-{prompt_logprobs_k} logprobs)")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=prompt_logprobs_k,
    )

    per_dataset_entropies: Dict[str, List[float]] = {}

    for ds_name, paths in paths_by_dataset.items():
        print(f"  [{ds_name}] {len(paths)} paths (max_tokens_per_batch={max_tokens_per_batch}, max_seq_len={max_seq_len})")

        # Build prompts with full token length, truncating long sequences
        all_items = []  # (prompt_dict, prompt_len, total_len)
        n_truncated = 0
        for p in paths:
            prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
            response_ids = p["response_token_ids"]
            full_ids = prompt_ids + response_ids
            # Truncate if too long (entropy stats unchanged by length)
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
                n_truncated += 1
            all_items.append(({"prompt_token_ids": full_ids}, len(prompt_ids), len(full_ids)))
        if n_truncated > 0:
            print(f"  [{ds_name}] {n_truncated} paths truncated to {max_seq_len} tokens")

        # Token-budget based dynamic batching
        ds_entropies: List[float] = []
        n_total = len(all_items)
        i = 0
        n_processed = 0
        while i < n_total:
            # Greedily pack items until token budget or batch limit
            batch_items = []
            batch_tokens = 0
            while i < n_total:
                item = all_items[i]
                total_len = item[2]
                # Single item exceeds budget — process alone
                if not batch_items and total_len > max_tokens_per_batch:
                    batch_items.append(item)
                    batch_tokens = total_len
                    i += 1
                    break
                if batch_tokens + total_len > max_tokens_per_batch:
                    break
                batch_items.append(item)
                batch_tokens += total_len
                i += 1

            batch_prompts = [it[0] for it in batch_items]
            batch_lens = [it[1] for it in batch_items]

            try:
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            except Exception as e:
                print(f"\n  [{ds_name}] batch failed ({len(batch_items)} items, {batch_tokens} tokens): {e}")
                # Skip this batch
                n_processed += len(batch_items)
                continue

            for plen, output in zip(batch_lens, outputs):
                prompt_logprobs = output.prompt_logprobs
                if prompt_logprobs is None:
                    continue
                for j in range(plen, len(prompt_logprobs)):
                    lp_dict = prompt_logprobs[j]
                    if lp_dict is None:
                        continue
                    ent = compute_entropy_from_logprobs(lp_dict)
                    ds_entropies.append(ent)

            n_processed += len(batch_items)
            print(f"  [{ds_name}] {n_processed}/{n_total} paths processed", end="\r")

        per_dataset_entropies[ds_name] = ds_entropies
        print(f"  [{ds_name}] {len(ds_entropies):,} response tokens, "
              f"mean entropy={np.mean(ds_entropies):.3f}")

    # Combined
    all_entropies: List[float] = []
    for ents in per_dataset_entropies.values():
        all_entropies.extend(ents)

    def _percentile_dict(arr):
        return {str(p): float(np.percentile(arr, p)) for p in percentiles}

    result = {
        "combined": _percentile_dict(all_entropies),
        "per_dataset": {ds: _percentile_dict(e) for ds, e in per_dataset_entropies.items()},
        "n_tokens_combined": len(all_entropies),
        "n_tokens_per_dataset": {ds: len(e) for ds, e in per_dataset_entropies.items()},
    }

    print(f"\n  Combined ({len(all_entropies):,} tokens): " +
          ", ".join(f"{p}th={v:.3f}" for p, v in result["combined"].items()))
    for ds, percs in result["per_dataset"].items():
        print(f"  {ds:<10}: " + ", ".join(f"{p}th={v:.3f}" for p, v in percs.items()))

    # Save raw entropies separately (large)
    return result, per_dataset_entropies, all_entropies


def compute_per_token_stats(
    llm,
    tokenizer,
    paths: List[Dict],
    prompt_logprobs_k: int = 20,
    max_tokens_per_batch: int = 4000,
    max_model_len: int = 12288,
) -> List[Optional[Dict]]:
    """For each path, compute per-token rank/logprob/entropy at every
    response position via vLLM's prompt_logprobs.

    Returns: list (length == len(paths)) where each entry is None (vLLM
    rejected) or a dict with three parallel arrays:
        - response_token_ranks:    List[int]    (1-indexed; k+1 if outside top-k)
        - response_token_logprobs: List[float]  (-inf if outside top-k)
        - response_token_entropies: List[float] (partial-sum top-k Shannon)

    Length of each array == len(paths[i]['response_token_ids']) when
    the path was not truncated by max_model_len.

    Uses the same token-budget batching pattern as
    `compute_dataset_entropy_threshold` for OOM safety.
    """
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=prompt_logprobs_k,
    )

    # Build items: (orig_idx, full_token_ids, prompt_len)
    items: List[Tuple[int, List[int], int]] = []
    n_truncated = 0
    for idx, p in enumerate(paths):
        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        response_ids = list(p["response_token_ids"])
        full_ids = prompt_ids + response_ids
        if len(full_ids) > max_model_len:
            full_ids = full_ids[:max_model_len]
            n_truncated += 1
        items.append((idx, full_ids, len(prompt_ids)))

    if n_truncated:
        print(f"  WARNING: {n_truncated}/{len(paths)} paths truncated to {max_model_len} tokens")

    results: List[Optional[Dict]] = [None] * len(paths)

    n_total = len(items)
    i = 0
    while i < n_total:
        # Greedy token-budget packing
        batch: List[Tuple[int, List[int], int]] = []
        batch_tokens = 0
        while i < n_total:
            it = items[i]
            full_len = len(it[1])
            if not batch and full_len > max_tokens_per_batch:
                batch.append(it)
                i += 1
                break
            if batch_tokens + full_len > max_tokens_per_batch:
                break
            batch.append(it)
            batch_tokens += full_len
            i += 1

        prompts = [{"prompt_token_ids": it[1]} for it in batch]
        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as e:
            print(f"\n  batch failed ({len(batch)} items, {batch_tokens} tok): {e}")
            continue

        for it, output in zip(batch, outputs):
            orig_idx, full_ids, plen = it
            ranks: List[int] = []
            logprobs_list: List[float] = []
            entropies: List[float] = []
            prompt_lp = output.prompt_logprobs
            if prompt_lp is None:
                results[orig_idx] = {
                    "response_token_ranks": [],
                    "response_token_logprobs": [],
                    "response_token_entropies": [],
                }
                continue
            for j in range(plen, len(prompt_lp)):
                lp_dict = prompt_lp[j]
                if lp_dict is None:
                    ranks.append(prompt_logprobs_k + 1)
                    logprobs_list.append(float("-inf"))
                    entropies.append(0.0)
                    continue
                actual_id = full_ids[j]
                if actual_id in lp_dict:
                    entry = lp_dict[actual_id]
                    tie_ranks = compute_tie_aware_ranks(lp_dict)
                    ranks.append(tie_ranks.get(actual_id, prompt_logprobs_k + 1))
                    logprobs_list.append(_logprob_value(entry))
                else:
                    # Sampled token outside returned top-k (rare with k=20)
                    ranks.append(prompt_logprobs_k + 1)
                    logprobs_list.append(float("-inf"))
                entropies.append(compute_entropy_from_logprobs(lp_dict))
            results[orig_idx] = {
                "response_token_ranks": ranks,
                "response_token_logprobs": logprobs_list,
                "response_token_entropies": entropies,
            }

        n_done = sum(1 for r in results if r is not None)
        print(f"  processed {n_done}/{n_total}", end="\r")

    print()
    return results


# ============================================================
# Phase D: 2x2 Analysis
# ============================================================

def analyze_2x2(
    results: List[Dict],
    entropy_threshold: float,
    filter_potential_min: Optional[float] = None,
) -> Dict:
    """2x2: Case A/B x High/Low entropy.

    Args:
        results: list of GreedyReplaceResult dicts (with case, entropy_at_t, etc.)
        entropy_threshold: float, e.g. 80th percentile
        filter_potential_min: if set, only include results with potential_t_minus_1 >= this

    Returns:
        {
            "case_A_high": int, "case_A_low": int,
            "case_B_high": int, "case_B_low": int,
            "total": int,
            "filter_applied": filter_potential_min,
        }
    """
    if filter_potential_min is not None:
        filtered = [r for r in results if r["potential_t_minus_1"] >= filter_potential_min]
    else:
        filtered = results

    counts = {"case_A_high": 0, "case_A_low": 0, "case_B_high": 0, "case_B_low": 0}
    for r in filtered:
        is_high = r["entropy_at_t"] >= entropy_threshold
        if r["case"] == "A":
            counts["case_A_high" if is_high else "case_A_low"] += 1
        else:
            counts["case_B_high" if is_high else "case_B_low"] += 1

    counts["total"] = len(filtered)
    counts["filter_applied"] = filter_potential_min
    return counts


# ============================================================
# Tables
# ============================================================

def generate_table1_main(results: List[Dict], output_path: str) -> Dict:
    """Table 1: Model, Dataset, N(cliff), Cliff=Greedy(%), Case A(%), Case B(%)."""
    if not results:
        return {}

    # Group by (model, dataset)
    by_md = defaultdict(list)
    for r in results:
        by_md[(r.get("model", "?"), r.get("dataset", "?"))].append(r)

    rows = []
    for (model, dataset), items in by_md.items():
        n = len(items)
        n_eq = sum(1 for r in items if r["is_cliff_eq_greedy"])
        n_a = sum(1 for r in items if r["case"] == "A")
        n_b = sum(1 for r in items if r["case"] == "B")
        rows.append({
            "Model": model,
            "Dataset": dataset,
            "N(cliff)": n,
            "Cliff=Greedy(%)": f"{n_eq/n*100:.1f}",
            "Case A(%)": f"{n_a/n*100:.1f}",
            "Case B(%)": f"{n_b/n*100:.1f}",
        })

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {output_path}")
    return rows


def generate_table_2x2(counts: Dict, output_path: str, label: str = ""):
    """Table 2 or 3: 2x2 crosstab."""
    total = counts.get("total", 1) or 1

    def pct(n):
        return f"{n} ({n/total*100:.1f}%)"

    rows = [
        {"": "Case A (Δ<0.2)",
         "High Entropy": pct(counts["case_A_high"]),
         "Low Entropy": pct(counts["case_A_low"]),
         "Total": pct(counts["case_A_high"] + counts["case_A_low"])},
        {"": "Case B (Δ≥0.2)",
         "High Entropy": pct(counts["case_B_high"]),
         "Low Entropy": pct(counts["case_B_low"]),
         "Total": pct(counts["case_B_high"] + counts["case_B_low"])},
        {"": "Total",
         "High Entropy": pct(counts["case_A_high"] + counts["case_B_high"]),
         "Low Entropy": pct(counts["case_A_low"] + counts["case_B_low"]),
         "Total": pct(total)},
    ]

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {output_path}  (filter={counts.get('filter_applied')})")


def generate_table4_sensitivity(
    results: List[Dict],
    threshold_dict: Dict[str, float],
    output_path: str,
    filter_potential_min: Optional[float] = None,
):
    """Table 4: Sensitivity analysis across entropy threshold percentiles."""
    rows = []
    for percentile_str in sorted(threshold_dict.keys(), key=int):
        thr = threshold_dict[percentile_str]
        counts = analyze_2x2(results, thr, filter_potential_min)
        total = counts["total"] or 1
        n_high = counts["case_A_high"] + counts["case_B_high"]
        rows.append({
            "Percentile": f"{percentile_str}th",
            "Threshold": f"{thr:.3f}",
            "High Ent (%)": f"{n_high/total*100:.1f}",
            "Case A High": counts["case_A_high"],
            "Case A Low": counts["case_A_low"],
            "Case B High": counts["case_B_high"],
            "Case B Low": counts["case_B_low"],
        })

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {output_path}")


def generate_tableA1_per_dataset_percentile(threshold_result: Dict, output_path: str):
    """Appendix A1: Per-dataset percentile vs combined."""
    rows = []
    combined = threshold_result["combined"]
    per_ds = threshold_result["per_dataset"]
    n_per_ds = threshold_result["n_tokens_per_dataset"]
    n_combined = threshold_result["n_tokens_combined"]

    # Header row
    cols = ["Dataset", "N tokens"] + [f"{p}th" for p in sorted(combined.keys(), key=int)]

    rows.append({**{"Dataset": "COMBINED", "N tokens": n_combined},
                 **{f"{p}th": f"{combined[p]:.3f}" for p in sorted(combined.keys(), key=int)}})
    for ds in sorted(per_ds.keys()):
        rows.append({**{"Dataset": ds, "N tokens": n_per_ds[ds]},
                     **{f"{p}th": f"{per_ds[ds][p]:.3f}" for p in sorted(per_ds[ds].keys(), key=int)}})

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {output_path}")


# ============================================================
# Figures
# ============================================================

def plot_delta_histogram(results: List[Dict], output_path: str):
    """Figure 1: Δpotential distribution histogram."""
    _apply_style()
    deltas = [r["delta_potential"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=40, color="#5C6BC0", edgecolor="white", alpha=0.8)
    ax.axvline(0.2, color="red", linestyle="--", label="Δ=0.2 (Case A/B threshold)")
    ax.set_xlabel("Δpotential = potential_{t-1} − potential_t_greedy")
    ax.set_ylabel("Count")
    ax.set_title(f"Δpotential Distribution (n={len(deltas)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_entropy_delta_scatter(results: List[Dict], threshold: float, output_path: str):
    """Figure 2: scatter X=entropy, Y=Δpotential, color=potential_{t-1} bin."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    high_pot = [(r["entropy_at_t"], r["delta_potential"]) for r in results if r["potential_t_minus_1"] >= 0.5]
    low_pot = [(r["entropy_at_t"], r["delta_potential"]) for r in results if r["potential_t_minus_1"] < 0.5]

    if low_pot:
        x, y = zip(*low_pot)
        ax.scatter(x, y, c="gray", alpha=0.4, s=30, label=f"potential_{{t-1}} < 0.5 (n={len(low_pot)})")
    if high_pot:
        x, y = zip(*high_pot)
        ax.scatter(x, y, c="#1E88E5", alpha=0.7, s=40, label=f"potential_{{t-1}} ≥ 0.5 (n={len(high_pot)})")

    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Entropy at t (top-20 approx)")
    ax.set_ylabel("Δpotential")
    ax.set_title("2×2 Quadrant: Entropy vs Δpotential")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rank_histogram(results: List[Dict], output_path: str):
    """Figure 3: Cliff token rank distribution."""
    _apply_style()
    ranks = [r["cliff_token_rank"] for r in results]
    max_rank = max(ranks)
    bins = list(range(1, max_rank + 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ranks, bins=bins, color="#7E57C2", edgecolor="white", alpha=0.8)
    n_eq = sum(1 for r in ranks if r == 1)
    ax.axvline(1.5, color="red", linestyle="--", label=f"Rank=1 (greedy): {n_eq}/{len(ranks)} ({n_eq/len(ranks)*100:.1f}%)")
    ax.set_xlabel("Cliff Token Rank in Output Distribution")
    ax.set_ylabel("Count")
    ax.set_title(f"Cliff Token Rank Distribution (n={len(ranks)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_case_entropy_overlay(results: List[Dict], threshold: float, output_path: str):
    """Figure 5: Case A vs Case B entropy distribution overlay."""
    _apply_style()
    case_a = [r["entropy_at_t"] for r in results if r["case"] == "A"]
    case_b = [r["entropy_at_t"] for r in results if r["case"] == "B"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(case_a, default=0), max(case_b, default=0)) + 0.5, 30)
    if case_a:
        ax.hist(case_a, bins=bins, color=CASE_A_COLOR, alpha=0.5,
                label=f"Case A (Δ<0.2): n={len(case_a)}", density=True)
    if case_b:
        ax.hist(case_b, bins=bins, color=CASE_B_COLOR, alpha=0.5,
                label=f"Case B (Δ≥0.2): n={len(case_b)}", density=True)
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"80th percentile = {threshold:.2f}")
    ax.set_xlabel("Entropy at cliff position")
    ax.set_ylabel("Density")
    ax.set_title("Entropy Distribution: Case A vs Case B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_recovery_rate(results: List[Dict], output_path: str):
    """Figure 6: Recovery rate distribution."""
    _apply_style()
    rates = [r["recovery_rate"] for r in results if r.get("recovery_rate") is not None]
    if not rates:
        print("  No recovery rates to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rates, bins=40, color="#26A69A", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6, label="0 = no recovery")
    ax.axvline(1, color="green", linestyle="--", alpha=0.6, label="1 = full recovery")
    ax.set_xlabel("Recovery Rate = (p_{t-1} − p_greedy) / (p_{t-1} − p_cliff)")
    ax.set_ylabel("Count")
    ax.set_title(f"Recovery Rate Distribution (n={len(rates)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_slopegraph(results: List[Dict], output_path: str):
    """Figure 7: Slopegraph showing potential at 3 stages for each cliff."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    x_positions = [0, 1, 2]
    x_labels = ["potential_{t-1}", "potential_t_cliff", "potential_t_greedy"]

    for r in results:
        ys = [r["potential_t_minus_1"], r["potential_t_cliff"], r["potential_t_greedy"]]
        color = CASE_A_COLOR if r["case"] == "A" else CASE_B_COLOR
        ax.plot(x_positions, ys, color=color, alpha=0.3, linewidth=1)

    # Legend handles
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=CASE_A_COLOR, label=f"Case A (Δ<0.2): n={sum(1 for r in results if r['case']=='A')}"),
        Line2D([0], [0], color=CASE_B_COLOR, label=f"Case B (Δ≥0.2): n={sum(1 for r in results if r['case']=='B')}"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.set_ylabel("Potential")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Greedy Replacement Effect (per cliff trajectory)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_trajectory_examples(results: List[Dict], paths_lookup: Dict[str, Dict],
                              output_path: str, n_examples: int = 3):
    """Figure 4: Potential trajectory examples (Case A and B)."""
    _apply_style()
    case_a = [r for r in results if r["case"] == "A"][:n_examples]
    case_b = [r for r in results if r["case"] == "B"][:n_examples]

    n_total = len(case_a) + len(case_b)
    if n_total == 0:
        print("  No trajectory examples to plot.")
        return

    fig, axes = plt.subplots(n_total, 1, figsize=(10, 2.5 * n_total))
    if n_total == 1:
        axes = [axes]

    for idx, r in enumerate(case_a + case_b):
        ax = axes[idx]
        path = paths_lookup.get(r["path_id"])
        if not path:
            continue
        scores = path["all_position_scores"]
        positions = list(range(len(scores)))
        ax.plot(positions, scores, color="black", linewidth=0.8, alpha=0.8)

        cliff_idx = r["cliff_position"] - 1
        if 0 <= cliff_idx < len(scores):
            ax.plot(cliff_idx, scores[cliff_idx], "o", color="red", markersize=10,
                    markeredgecolor="darkred", label=f"original cliff (p={r['potential_t_cliff']:.2f})")
            ax.plot(cliff_idx, r["potential_t_greedy"], "o", color="blue", markersize=10,
                    markeredgecolor="darkblue", label=f"greedy (p={r['potential_t_greedy']:.2f})")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Potential")
        ax.set_title(f"Case {r['case']}: {r['path_id']}  Δ={r['delta_potential']:.2f}", fontsize=10)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_dataset_entropy_overlay(per_dataset_entropies: Dict[str, List[float]], output_path: str):
    """Appendix A2: Per-dataset entropy distribution overlay."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"gsm1k": "#1E88E5", "math500": "#E53935", "aime25": "#43A047"}
    for ds, ents in per_dataset_entropies.items():
        if not ents:
            continue
        sample = np.array(ents)
        if len(sample) > 50000:
            sample = np.random.choice(sample, 50000, replace=False)
        ax.hist(sample, bins=80, alpha=0.4, density=True,
                color=colors.get(ds, "gray"), label=f"{ds} (n={len(ents):,})")

    ax.set_xlabel("Token Entropy (top-20 approx)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Dataset Entropy Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
