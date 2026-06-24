"""
Cliff Token Decoder: Cliff-del and Cliff-keep regeneration.

Cliff-del: truncate before cliff token, regenerate N times.
Cliff-keep: truncate after cliff token (including it), regenerate N times.

Each cliff token in a path is a separate experiment point.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

from vllm import SamplingParams

from src.analysis.detector import find_all_cliff_tokens, find_all_cliff_tokens_statistical
from src.utils.grader import batch_grade_responses_mathverify
from src import config

# Default to statistical cliff detection (z-test based)
USE_STATISTICAL_CLIFF = True


@dataclass
class CliffRegenerationResult:
    path_id: str
    problem_id: str
    question: str
    golden_answer: List[str]
    cliff_position: int             # 1-indexed
    cliff_token_str: Optional[str]
    cliff_token_id: Optional[int]
    drop_magnitude: float
    path_is_correct: bool           # original path correctness
    original_response: str = ""     # full original response

    # Cliff-del results
    del_responses: List[str] = field(default_factory=list)
    del_correctness: List[bool] = field(default_factory=list)
    del_num_correct: int = 0

    # Cliff-keep results
    keep_responses: List[str] = field(default_factory=list)
    keep_correctness: List[bool] = field(default_factory=list)
    keep_num_correct: int = 0

    num_samples: int = 64


def _build_prefix(path: Dict, tokenizer, truncate_pos: int) -> List[int]:
    """Build prefix token IDs: prompt + response[:truncate_pos].

    truncate_pos is 0-indexed into response_token_ids.
    """
    prompt_ids = tokenizer.encode(path["full_prompt"], add_special_tokens=False)
    response_ids = path["response_token_ids"][:truncate_pos]
    return prompt_ids + response_ids


def run_cliff_on_paths(
    llm,
    tokenizer,
    paths: List[Dict],
    dataset_name: str,
    num_samples: int = 64,
    mode: str = "non_thinking",
    drop_threshold: float = config.DEFAULT_CLIFF_THRESHOLD,
    model_path: str = None,
) -> List[CliffRegenerationResult]:
    """Run Cliff-del and Cliff-keep for all cliff tokens in all paths.

    Returns one CliffRegenerationResult per cliff token per path.
    """
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

    # Phase 1: Collect all requests
    method_str = "statistical z-test" if USE_STATISTICAL_CLIFF else f"fixed threshold={drop_threshold}"
    print(f"  Detecting cliff tokens ({method_str})...")
    requests = []  # (result_idx, mode, prefix_ids, golden_answers)
    results = []

    for p in paths:
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

        for cliff in cliffs:
            result_idx = len(results)
            result = CliffRegenerationResult(
                path_id=p["id"],
                problem_id=p.get("problem_id", ""),
                question=p.get("question", ""),
                golden_answer=p.get("golden_answer", []),
                cliff_position=cliff.position,
                cliff_token_str=cliff.token_str,
                cliff_token_id=cliff.token_id,
                drop_magnitude=cliff.drop_magnitude,
                path_is_correct=p.get("is_correct", False),
                original_response=p.get("response", ""),
                num_samples=num_samples,
            )
            results.append(result)

            # Cliff-del: truncate BEFORE cliff (position is 1-indexed, so [:pos-1] in 0-indexed)
            del_pos = cliff.position - 1  # 0-indexed, exclusive of cliff
            del_prefix = _build_prefix(p, tokenizer, del_pos)
            requests.append((result_idx, "del", del_prefix, p.get("golden_answer", [])))

            # Cliff-keep: truncate AFTER cliff (include cliff token)
            keep_pos = cliff.position  # 0-indexed slice [:pos] includes cliff
            keep_prefix = _build_prefix(p, tokenizer, keep_pos)
            requests.append((result_idx, "keep", keep_prefix, p.get("golden_answer", [])))

    if not requests:
        print("  No cliff tokens found in any path.")
        return results

    print(f"  {len(results)} cliff instances × 2 modes = {len(requests)} prompts × {num_samples} samples")

    # Phase 2: Generate
    print("  Generating responses...")
    prompts = [{"prompt_token_ids": req[2]} for req in requests]
    outputs = llm.generate(prompts, sampling_params)

    # Phase 3: Grade
    print("  Grading responses...")
    all_responses = []
    all_golden = []
    response_map = []  # (request_idx, sample_idx)

    for req_idx, output in enumerate(outputs):
        golden = requests[req_idx][3]
        for s_idx, sample in enumerate(output.outputs):
            # Reconstruct full response for grading
            prefix_ids = requests[req_idx][2]
            prompt_ids = tokenizer.encode(
                [p for p in paths if p["id"] == results[requests[req_idx][0]].path_id][0]["full_prompt"],
                add_special_tokens=False,
            )
            prefix_response_ids = prefix_ids[len(prompt_ids):]
            prefix_text = tokenizer.decode(prefix_response_ids, skip_special_tokens=True)
            full_response = prefix_text + sample.text
            all_responses.append(full_response)
            all_golden.append(golden)
            response_map.append((req_idx, s_idx))

    correctness = batch_grade_responses_mathverify(
        all_responses, all_golden, source_name, wall_timeout=2.0,
    )

    # Phase 4: Assemble results
    print("  Assembling results...")
    for resp_idx, is_correct in enumerate(correctness):
        req_idx, s_idx = response_map[resp_idx]
        result_idx = requests[req_idx][0]
        mode_str = requests[req_idx][1]
        result = results[result_idx]

        if mode_str == "del":
            result.del_responses.append(all_responses[resp_idx])
            result.del_correctness.append(is_correct)
        else:
            result.keep_responses.append(all_responses[resp_idx])
            result.keep_correctness.append(is_correct)

    for r in results:
        r.del_num_correct = sum(r.del_correctness)
        r.keep_num_correct = sum(r.keep_correctness)

    n_del_success = sum(1 for r in results if r.del_num_correct > 0)
    n_keep_success = sum(1 for r in results if r.keep_num_correct > 0)
    print(f"  Done: {len(results)} cliff instances")
    print(f"    Cliff-del: {n_del_success}/{len(results)} have ≥1 correct")
    print(f"    Cliff-keep: {n_keep_success}/{len(results)} have ≥1 correct")

    return results


def cliff_results_to_dicts(results: List[CliffRegenerationResult]) -> List[Dict]:
    """Convert to JSON-serializable dicts. Saves first success response for divergence analysis."""
    out = []
    for r in results:
        # Find first successful del/keep response (for semantic comparison)
        del_success_response = None
        for resp, correct in zip(r.del_responses, r.del_correctness):
            if correct:
                del_success_response = resp
                break

        keep_success_response = None
        for resp, correct in zip(r.keep_responses, r.keep_correctness):
            if correct:
                keep_success_response = resp
                break

        d = {
            "path_id": r.path_id,
            "problem_id": r.problem_id,
            "question": r.question,
            "golden_answer": r.golden_answer,
            "cliff_position": r.cliff_position,
            "cliff_token_str": r.cliff_token_str,
            "cliff_token_id": r.cliff_token_id,
            "drop_magnitude": r.drop_magnitude,
            "path_is_correct": r.path_is_correct,
            "original_response": r.original_response,
            "del_num_correct": r.del_num_correct,
            "del_correctness": r.del_correctness,
            "del_success_response": del_success_response,
            "keep_num_correct": r.keep_num_correct,
            "keep_correctness": r.keep_correctness,
            "keep_success_response": keep_success_response,
            "num_samples": r.num_samples,
        }
        out.append(d)
    return out


# Backward-compatible aliases for cli.py
run_cliff_del_on_paths = run_cliff_on_paths
cliff_del_results_to_dicts = cliff_results_to_dicts
