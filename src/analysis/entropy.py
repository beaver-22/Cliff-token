"""
Token-level entropy / rank utilities.

  compute_entropy_from_logprobs  partial-sum Shannon entropy over top-k logprobs
  compute_tie_aware_ranks        competition rank (ties share rank 1)
  compute_per_token_stats        per-token rank / logprob / entropy of a trace
"""

import math
from typing import List, Dict, Optional, Tuple

from vllm import SamplingParams


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
# Tokens with entropy <= H_b(GREEDY_PROB_THRESHOLD) are near-deterministic.
# The deterministic / uncertain / sampled-off taxonomy splits on this boundary
# (H_b(0.99) ~= 0.0561 nats).
GREEDY_PROB_THRESHOLD = 0.99
GREEDY_BOUND_NATS = _binary_entropy_nats(GREEDY_PROB_THRESHOLD)


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
    classification (deterministic / uncertain / sampled-off taxonomy) then misclassifies
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
# Per-token statistics
# ============================================================

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

    Batches requests under a token budget for OOM safety.
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
