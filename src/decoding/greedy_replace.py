"""
RQ2-1 Phase B+C: Greedy replacement experiment.

Phase B: For each cliff token, extract logprobs at position t (using prefix response[:t-1])
         to get greedy_token, cliff_rank, cliff_prob, greedy_prob, entropy.
Phase C: Replace cliff token with greedy token, run rollout 64 times.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from vllm import SamplingParams

from src.analysis.detector import find_all_cliff_tokens, find_all_cliff_tokens_statistical
from src.analysis.entropy import (
    compute_entropy_from_logprobs,
    _logprob_value,
    compute_tie_aware_ranks,
)
from src.utils.grader import batch_grade_responses_mathverify
from src import config

# Default to statistical cliff detection (z-test based)
USE_STATISTICAL_CLIFF = True


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
    # Phase B
    greedy_token_id: int
    greedy_token_str: Optional[str]
    is_cliff_eq_greedy: bool
    cliff_token_rank: int          # 1-indexed; k+1 if outside top-k
    cliff_token_prob: float
    greedy_token_prob: float
    entropy_at_t: float
    # Path-level metadata
    path_is_correct: bool = False


@dataclass
class GreedyReplaceResult:
    # Inherited from CliffLogprobsInfo
    path_id: str
    model: str
    dataset: str
    cliff_position: int
    cliff_token_id: int
    cliff_token_str: Optional[str]
    drop_magnitude: float
    potential_t_minus_1: float
    potential_t_cliff: float
    greedy_token_id: int
    greedy_token_str: Optional[str]
    is_cliff_eq_greedy: bool
    cliff_token_rank: int
    cliff_token_prob: float
    greedy_token_prob: float
    entropy_at_t: float
    # Phase C
    greedy_correctness: List[bool] = field(default_factory=list)
    greedy_num_correct: int = 0
    potential_t_greedy: float = 0.0
    delta_potential: float = 0.0
    recovery_rate: float = 0.0
    case: str = "?"               # "A" or "B"
    num_samples: int = 64


# ============================================================
# Phase B: Cliff logprobs + greedy token extraction
# ============================================================

def extract_cliff_logprobs_and_greedy(
    llm,
    tokenizer,
    paths: List[Dict],
    model_name: str,
    dataset_name: str,
    drop_threshold: float = config.DEFAULT_CLIFF_THRESHOLD,
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
    print(f"\n[Phase B] Extracting cliff logprobs (top-{top_k})")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=top_k,
    )

    # Build requests
    requests = []  # (path_idx, cliff, prefix_ids)
    for path_idx, p in enumerate(paths):
        scores = p.get("all_position_scores", [])
        if USE_STATISTICAL_CLIFF:
            cliffs = find_all_cliff_tokens_statistical(
                scores,
                tokens=p.get("response_tokens"),
                token_ids=p.get("response_token_ids"),
            )
        else:
            cliffs = find_all_cliff_tokens(
                scores, drop_threshold,
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
            import math
            cliff_prob = math.exp(cliff_lp)
        else:
            cliff_prob = 0.0

        # Greedy token prob
        if greedy_token_id in logprob_dict:
            greedy_lp = _logprob_value(logprob_dict[greedy_token_id])
            import math
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


# ============================================================
# Phase C: Greedy replacement rollout
# ============================================================

def run_greedy_replacement_rollout(
    llm,
    tokenizer,
    cliff_info_list: List[CliffLogprobsInfo],
    paths: List[Dict],
    model_name: str,
    dataset_name: str,
    num_samples: int = 64,
    mode: str = "non_thinking",
    model_path: str = None,
    delta_threshold: float = 0.20,
) -> List[GreedyReplaceResult]:
    """For each cliff, replace with greedy token and run rollout."""
    print(f"\n[Phase C] Greedy replacement rollout ({num_samples} samples each)")

    sampling_cfg = config.get_sampling_config(mode, model_path)
    source_name = config.get_dataset_source_name(dataset_name)
    max_new_tokens = config.get_max_tokens(dataset_name, mode)

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=sampling_cfg.temperature,
        top_p=sampling_cfg.top_p,
        top_k=sampling_cfg.top_k if sampling_cfg.top_k > 0 else -1,
        presence_penalty=sampling_cfg.presence_penalty,
        repetition_penalty=sampling_cfg.repetition_penalty,
        max_tokens=max_new_tokens,
        stop=config.STOP_TOKENS,
    )

    paths_by_id = {p["id"]: p for p in paths}

    # Build requests
    requests = []  # (cliff_info_idx, prefix_ids, golden, path)
    for ci_idx, ci in enumerate(cliff_info_list):
        p = paths_by_id.get(ci.path_id)
        if not p:
            continue
        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        truncate_pos = ci.cliff_position - 1
        # prefix = prompt + response[:t-1] + [greedy_token]
        prefix_ids = prompt_ids + p["response_token_ids"][:truncate_pos] + [ci.greedy_token_id]
        requests.append((ci_idx, prefix_ids, p.get("golden_answer", []), p, prompt_ids))

    print(f"  {len(requests)} cliffs × {num_samples} samples = {len(requests) * num_samples} generations")
    prompts = [{"prompt_token_ids": req[1]} for req in requests]
    outputs = llm.generate(prompts, sampling_params)

    # Build all responses for grading
    print("  Grading...")
    all_responses = []
    all_golden = []
    response_map = []  # (req_idx, sample_idx)
    for req_idx, output in enumerate(outputs):
        ci_idx, prefix_ids, golden, p, prompt_ids = requests[req_idx]
        prefix_response_ids = prefix_ids[len(prompt_ids):]
        prefix_text = tokenizer.decode(prefix_response_ids, skip_special_tokens=True)
        for sample in output.outputs:
            all_responses.append(prefix_text + sample.text)
            all_golden.append(golden)
            response_map.append(req_idx)

    correctness = batch_grade_responses_mathverify(all_responses, all_golden, source_name)

    # Assemble results
    results = [GreedyReplaceResult(
        **asdict(ci),
        num_samples=num_samples,
    ) for ci in cliff_info_list]

    # Group correctness by cliff
    cliff_correctness = [[] for _ in cliff_info_list]
    for resp_idx, is_correct in enumerate(correctness):
        req_idx = response_map[resp_idx]
        ci_idx = requests[req_idx][0]
        cliff_correctness[ci_idx].append(is_correct)

    for i, r in enumerate(results):
        r.greedy_correctness = cliff_correctness[i]
        r.greedy_num_correct = sum(cliff_correctness[i])
        r.potential_t_greedy = r.greedy_num_correct / num_samples if num_samples else 0.0
        r.delta_potential = r.potential_t_minus_1 - r.potential_t_greedy
        # Recovery rate
        cliff_drop = r.potential_t_minus_1 - r.potential_t_cliff
        if cliff_drop > 0:
            r.recovery_rate = (r.potential_t_greedy - r.potential_t_cliff) / cliff_drop
        else:
            r.recovery_rate = 0.0
        r.case = "A" if r.delta_potential < delta_threshold else "B"

    n_a = sum(1 for r in results if r.case == "A")
    n_b = sum(1 for r in results if r.case == "B")
    print(f"  Done: Case A (Δ<{delta_threshold}): {n_a}, Case B (Δ≥{delta_threshold}): {n_b}")
    return results


def greedy_replace_results_to_dicts(results: List[GreedyReplaceResult]) -> List[Dict]:
    """Convert to JSON. Keeps full greedy_correctness but excludes greedy responses."""
    out = []
    for r in results:
        d = asdict(r)
        # greedy_correctness is List[bool] — keep
        out.append(d)
    return out
