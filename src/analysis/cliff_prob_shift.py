"""Cliff-token probability before and after Cliff-DPO (Figure 5, right).

A frozen set of held-out cliffs (GSM1K / MATH500 / AIME 2025 stem traces) is scored with
the base model and with every Cliff-DPO adapter. At each cliff, the model is
teacher-forced on prompt + response[:t-1] and the probability of the cliff
token is read from the full next-token distribution (HF forward pass, full
vocabulary log-softmax; no top-k censoring).
"""

import json
import math
import os
from typing import Dict, List

from src.analysis.detector import find_all_cliff_tokens_statistical
from src.dpo.vllm_rollout import classify_cliff

EVAL_DATASETS = ["gsm1k_100", "math500_100", "aime25"]
NUM_SAMPLES = 64
TOP_K = 20


# ============================================================
# Cliff set construction (frozen once, shared by every pass)
# ============================================================

def build_eval_cliff_set(
    model_short: str,
    tokenizer,
    datasets: List[str] = None,
    rollout_dir: str = "./output/03_rollouts",
    token_stats_dir: str = "./output/02_token_stats",
) -> List[Dict]:
    """Detect every cliff on the held-out sets and label its cliff type.

    Cliff positions come from the 64-sample success-probability curve in
    `rollout_dir`; the cliff type needs the entropy and rank of the cliff token,
    read from the token-stats copy of the same traces.
    """
    datasets = datasets or EVAL_DATASETS
    cliffs: List[Dict] = []

    for dataset in datasets:
        roll_path = os.path.join(rollout_dir, model_short, f"{dataset}_all_paths.json")
        stats_path = os.path.join(token_stats_dir, model_short, f"{dataset}_all_paths.json")
        if not os.path.exists(roll_path):
            print(f"  [{dataset}] missing rollout file, skipping: {roll_path}")
            continue
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"No token stats for {model_short}/{dataset} at {stats_path}; "
                "cliff-type labels need per-token entropy and rank."
            )

        with open(roll_path) as f:
            paths = json.load(f)
        with open(stats_path) as f:
            stats_by_id = {p["id"]: p for p in json.load(f)}

        n_before = len(cliffs)
        for p in paths:
            scores = p.get("all_position_scores") or []
            if not scores:
                continue
            found = find_all_cliff_tokens_statistical(
                scores,
                tokens=p.get("response_tokens"),
                token_ids=p.get("response_token_ids"),
                N=NUM_SAMPLES,
            )
            if not found:
                continue

            st = stats_by_id.get(p["id"])
            if st is None:
                raise KeyError(f"{p['id']} present in rollout but absent from {stats_path}")
            entropies = st.get("response_token_entropies") or []
            ranks = st.get("response_token_ranks") or []

            prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
            response_ids = p["response_token_ids"]

            for c in found:
                idx = c.position - 1  # 0-indexed into the response
                if idx >= len(entropies) or idx >= len(ranks):
                    raise IndexError(
                        f"{p['id']} cliff at {c.position} beyond token-stats trace "
                        f"({len(entropies)} entries) -- traces are out of sync."
                    )
                entropy = entropies[idx]
                cliffs.append({
                    "cliff_uid": f"{dataset}|{p['id']}|{c.position}",
                    "dataset": dataset,
                    "path_id": p["id"],
                    "path_is_correct": bool(p.get("is_correct", False)),
                    "cliff_position": c.position,
                    "cliff_token_id": c.token_id,
                    "cliff_token_str": c.token_str,
                    "category": classify_cliff(entropy, ranks[idx] == 1),
                    "base_entropy_at_cliff": entropy,
                    "base_rank_at_cliff": ranks[idx],
                    "base_prev_score": c.prev_score,
                    "base_cliff_score": c.curr_score,
                    # prefix = prompt + response[:t-1]; the cliff token itself is excluded
                    "prefix_token_ids": prompt_ids + response_ids[:idx],
                })
        print(f"  [{dataset}] {len(cliffs) - n_before} cliffs")

    return cliffs


def cliff_set_summary(cliffs: List[Dict]) -> Dict:
    """Counts by dataset and by cliff type."""
    by_ds: Dict[str, int] = {}
    by_cat: Dict[str, int] = {}
    for c in cliffs:
        by_ds[c["dataset"]] = by_ds.get(c["dataset"], 0) + 1
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    return {"total": len(cliffs), "by_dataset": by_ds, "by_category": by_cat}


# ============================================================
# Teacher-forced cliff-token probability
# ============================================================

def score_token_probs_hf(
    model,
    cliffs: List[Dict],
    top_k: int = TOP_K,
    device: str = "cuda",
) -> List[Dict]:
    """One HF forward pass per cliff; reads P(cliff token) from the full
    next-token distribution at the cliff position."""
    import torch

    rows = []
    for i, c in enumerate(cliffs):
        ids = torch.tensor([c["prefix_token_ids"]], device=device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)

        cliff_lp = logprobs[c["cliff_token_id"]].item()
        # competition rank with a tie tolerance, matching compute_tie_aware_ranks
        rank = int((logprobs > cliff_lp + 1e-9).sum().item()) + 1

        top = torch.topk(logprobs, top_k)
        top_lp = top.values
        rows.append({
            "cliff_uid": c["cliff_uid"],
            "p_cliff": math.exp(cliff_lp),
            "logp_cliff": cliff_lp,
            "rank_cliff": rank,
            "cliff_in_top_k": rank <= top_k,
            "entropy_top_k": float(-(top_lp.exp() * top_lp).sum().item()),
            "greedy_token_id": int(top.indices[0].item()),
            "p_greedy": float(top_lp[0].exp().item()),
        })
        if (i + 1) % 50 == 0 or i + 1 == len(cliffs):
            print(f"    {i + 1}/{len(cliffs)}")
    return rows
