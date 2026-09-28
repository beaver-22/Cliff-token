"""
RQ1 evaluator: pass@k for Cliff-del vs. Cliff-keep (Figure 3).

  evaluate_cliff_del_keep  per-run pass@k over all cliffs (correct / incorrect)
  first_cliff_pass_at_k       pass@k over the first cliff of each incorrect trace
"""

import os
import json
from math import comb
from typing import List, Dict

from src import config


MODEL_DISPLAY_ORDER = [
    "Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B",
    "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct",
    "gemma-3-4b-it",
]
DATASET_DISPLAY_ORDER = ["gsm1k_100", "math500_100", "aime25"]


def _ordered(items, ordering):
    """Return items in `ordering` order, with any extras appended at the end."""
    in_order = [x for x in ordering if x in items]
    extras = [x for x in items if x not in ordering]
    return in_order + extras


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
    """Average pass@k across result instances."""
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


def first_cliff_rows(cliff_results: List[Dict]) -> List[Dict]:
    """One row per trace: the cliff with the smallest position."""
    first = {}
    for r in cliff_results:
        pid = r["path_id"]
        if pid not in first or r["cliff_position"] < first[pid]["cliff_position"]:
            first[pid] = r
    return list(first.values())


def first_cliff_pass_at_k(cliff_results: List[Dict], k_values: List[int] = None) -> Dict:
    """Cliff-del / Cliff-keep pass@k over the first cliff of each incorrect trace."""
    rows = first_cliff_rows([r for r in cliff_results if not r["path_is_correct"]])
    return {
        "cliff_del": compute_avg_pass_at_k(rows, "del_num_correct", k_values=k_values),
        "cliff_keep": compute_avg_pass_at_k(rows, "keep_num_correct", k_values=k_values),
        "n_traces": len(rows),
    }


# ============================================================
# Per-run evaluation
# ============================================================

def evaluate_cliff_del_keep(cliff_results: List[Dict], output_dir: str) -> Dict:
    """Cliff-del vs Cliff-keep pass@k over all cliffs, split by trace correctness."""
    os.makedirs(output_dir, exist_ok=True)

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
    with open(os.path.join(output_dir, "exp1_pass_at_k.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def print_cliff_del_keep_summary(result: Dict):
    print(f"\n{'='*60}")
    print("Cliff-del vs Cliff-keep")
    print(f"{'='*60}")
    print(f"  Correct-trace cliffs:   {result['n_success_instances']}")
    print(f"  Incorrect-trace cliffs: {result['n_failure_instances']}")
    for label, key in [("Cliff-del  (correct)", "del_success"),
                       ("Cliff-del  (incorrect)", "del_failure"),
                       ("Cliff-keep (correct)", "keep_success"),
                       ("Cliff-keep (incorrect)", "keep_failure")]:
        data = result[key]
        print(f"    {label}: pass@1={data.get(1, 0):.3f}, pass@64={data.get(64, 0):.3f}")
