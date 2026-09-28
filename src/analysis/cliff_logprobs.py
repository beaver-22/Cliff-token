"""
Cliff taxonomy inputs (Table 1).

For each cliff token at position t, query the model once on
prompt + response[:t-1] (temperature 0, one token, top-20 logprobs) to obtain
the greedy token, the cliff token's tie-aware rank and probability, and the
token entropy H_t at that position.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional

from vllm import SamplingParams

from src.analysis.detector import find_all_cliff_tokens_statistical
from src.analysis.entropy import (
    compute_entropy_from_logprobs,
    _logprob_value,
    compute_tie_aware_ranks,
)


@dataclass
class CliffLogprobsInfo:
    path_id: str
    model: str
    dataset: str
    cliff_position: int            # 1-indexed
    cliff_token_id: int
    cliff_token_str: Optional[str]
    drop_magnitude: float
    potential_t_minus_1: float
    potential_t_cliff: float
    # Next-token distribution at the cliff position
    greedy_token_id: int
    greedy_token_str: Optional[str]
    is_cliff_eq_greedy: bool
    cliff_token_rank: int          # 1-indexed; k+1 if outside top-k
    cliff_token_prob: float
    greedy_token_prob: float
    entropy_at_t: float
    # Path-level metadata
    path_is_correct: bool = False



# ============================================================
# Cliff logprobs + greedy token extraction
# ============================================================

def extract_cliff_logprobs_and_greedy(
    llm,
    tokenizer,
    paths: List[Dict],
    model_name: str,
    dataset_name: str,
    top_k: int = 20,
) -> List[CliffLogprobsInfo]:
    """For each cliff token in paths, extract logprobs at cliff position.

    Workflow:
    - For each cliff at position t in path p:
      prefix = prompt + response[:t-1]
      Run vLLM with temperature=0, max_tokens=1, logprobs=top_k
      → generated token = greedy token
      → output.outputs[0].logprobs[0] = top-k logprobs at position t
      → find cliff_token in this dict to get rank, prob, etc.
    """
    print(f"\nExtracting cliff logprobs (top-{top_k})")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=top_k,
    )

    # Build requests
    requests = []  # (path_idx, cliff, prefix_ids)
    for path_idx, p in enumerate(paths):
        scores = p.get("all_position_scores", [])
        cliffs = find_all_cliff_tokens_statistical(
            scores,
            tokens=p.get("response_tokens"),
            token_ids=p.get("response_token_ids"),
        )
        if not cliffs:
            continue

        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        for cliff in cliffs:
            # prefix = prompt + response[:cliff_pos-1] (0-indexed slice, excluding cliff)
            truncate_pos = cliff.position - 1
            prefix_ids = prompt_ids + p["response_token_ids"][:truncate_pos]
            requests.append((path_idx, cliff, prefix_ids))

    if not requests:
        print("  No cliff tokens found.")
        return []

    print(f"  {len(requests)} cliff positions × 1 token gen")
    prompts = [{"prompt_token_ids": req[2]} for req in requests]
    outputs = llm.generate(prompts, sampling_params)

    # Extract logprobs
    results = []
    for (path_idx, cliff, _), output in zip(requests, outputs):
        p = paths[path_idx]
        scores = p["all_position_scores"]
        cliff_idx = cliff.position - 1  # 0-indexed
        # potential_{t-1} = scores at position t-1 (1-indexed) = scores[cliff_idx - 1] (0-indexed)
        # But scores[i] = P(correct | response[:i+1]); so scores[cliff_idx - 1] = potential at the token BEFORE cliff
        # And scores[cliff_idx] = potential AT cliff token (after seeing cliff)
        # Cliff drop: scores[cliff_idx-1] - scores[cliff_idx] >= threshold
        if cliff_idx > 0 and cliff_idx < len(scores):
            pot_t_minus_1 = scores[cliff_idx - 1] if scores[cliff_idx - 1] is not None else 0.0
            pot_t_cliff = scores[cliff_idx] if scores[cliff_idx] is not None else 0.0
        else:
            pot_t_minus_1 = cliff.prev_score
            pot_t_cliff = cliff.curr_score

        # Generated greedy token from output
        sample = output.outputs[0]
        greedy_token_id = sample.token_ids[0] if sample.token_ids else cliff.token_id
        greedy_token_str = tokenizer.decode([greedy_token_id], skip_special_tokens=True)

        # Logprobs at position 0 of the output (= top-k at cliff position)
        logprob_dict = sample.logprobs[0] if sample.logprobs else {}

        # Tie-aware competition rank: tokens within EPS of the top logprob
        # share rank=1. Avoids vLLM's arbitrary tie-break for tied-top
        # tokens, matching the deterministic/uncertain/sampled_off taxonomy semantics used elsewhere.
        tie_ranks = compute_tie_aware_ranks(logprob_dict) if logprob_dict else {}

        # Cliff token rank/prob
        cliff_token_id = cliff.token_id
        cliff_rank = tie_ranks.get(cliff_token_id, top_k + 1)
        if cliff_token_id in logprob_dict:
            cliff_lp = _logprob_value(logprob_dict[cliff_token_id])
            cliff_prob = math.exp(cliff_lp)
        else:
            cliff_prob = 0.0

        # Greedy token prob
        if greedy_token_id in logprob_dict:
            greedy_lp = _logprob_value(logprob_dict[greedy_token_id])
            greedy_prob = math.exp(greedy_lp)
        else:
            greedy_prob = 0.0

        entropy = compute_entropy_from_logprobs(logprob_dict)

        results.append(CliffLogprobsInfo(
            path_id=p["id"],
            model=model_name,
            dataset=dataset_name,
            cliff_position=cliff.position,
            cliff_token_id=cliff_token_id,
            cliff_token_str=cliff.token_str,
            drop_magnitude=cliff.drop_magnitude,
            potential_t_minus_1=pot_t_minus_1,
            potential_t_cliff=pot_t_cliff,
            greedy_token_id=greedy_token_id,
            greedy_token_str=greedy_token_str,
            # Tie-aware: classify as greedy if the cliff token is in the top
            # probability tier (rank==1, including ties). Comparing token_id
            # directly can produce false negatives due to vLLM's arbitrary tie-breaking.
            is_cliff_eq_greedy=(cliff_rank == 1),
            cliff_token_rank=cliff_rank,
            cliff_token_prob=cliff_prob,
            greedy_token_prob=greedy_prob,
            entropy_at_t=entropy,
            path_is_correct=bool(p.get("is_correct", False)),
        ))

    n_eq = sum(1 for r in results if r.is_cliff_eq_greedy)
    print(f"  Done: {len(results)} cliff instances, {n_eq} ({n_eq/len(results)*100:.1f}%) where cliff==greedy")
    return results
