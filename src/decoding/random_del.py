"""
Random Decoder: truncate at a random position, regenerate N times.
Baseline for comparison with cliff/critical/tangent deletion.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from vllm import SamplingParams

from src.utils.grader import batch_grade_responses_mathverify
from src import config


@dataclass
class RandomRegenerationResult:
    path_id: str
    golden_answer: List[str]
    truncation_position: Optional[int]  # 0-indexed
    path_is_correct: bool

    del_responses: List[str] = field(default_factory=list)
    del_correctness: List[bool] = field(default_factory=list)
    del_num_correct: int = 0
    num_samples: int = 64


def run_random_del_on_paths(
    llm, tokenizer, paths: List[Dict], dataset_name: str,
    num_samples: int = 64, mode: str = "non_thinking", model_path: str = None,
    seed: int = 42,
) -> List[RandomRegenerationResult]:
    """Truncate at random position, regenerate."""
    sampling_cfg = config.get_sampling_config(mode, model_path)
    source_name = config.get_dataset_source_name(dataset_name)
    max_new_tokens = config.get_max_tokens(dataset_name, mode)
    rng = random.Random(seed)

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

    print(f"  Picking random truncation positions (seed={seed})...")
    requests = []
    results = []

    for p in paths:
        n_tokens = len(p.get("response_token_ids", []))
        if n_tokens < 2:
            continue
        trunc_pos = rng.randint(1, n_tokens - 1)  # at least 1 token kept

        result = RandomRegenerationResult(
            path_id=p["id"],
            golden_answer=p.get("golden_answer", []),
            truncation_position=trunc_pos,
            path_is_correct=p.get("is_correct", False),
            num_samples=num_samples,
        )
        results.append(result)

        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        prefix_ids = prompt_ids + p["response_token_ids"][:trunc_pos]
        requests.append((len(results) - 1, prefix_ids, p.get("golden_answer", []), p))

    if not requests:
        print("  No valid paths for random-del.")
        return results

    print(f"  {len(requests)} paths × {num_samples} samples")
    print("  Generating responses...")
    prompts = [{"prompt_token_ids": req[1]} for req in requests]
    outputs = llm.generate(prompts, sampling_params)

    print("  Grading responses...")
    all_responses, all_golden, response_map = [], [], []
    for req_idx, output in enumerate(outputs):
        golden = requests[req_idx][2]
        p = requests[req_idx][3]
        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        prefix_ids = requests[req_idx][1]
        prefix_response_ids = prefix_ids[len(prompt_ids):]
        prefix_text = tokenizer.decode(prefix_response_ids, skip_special_tokens=True)
        for sample in output.outputs:
            all_responses.append(prefix_text + sample.text)
            all_golden.append(golden)
            response_map.append(req_idx)

    correctness = batch_grade_responses_mathverify(all_responses, all_golden, source_name)

    for resp_idx, is_correct in enumerate(correctness):
        req_idx = response_map[resp_idx]
        result_idx = requests[req_idx][0]
        results[result_idx].del_responses.append(all_responses[resp_idx])
        results[result_idx].del_correctness.append(is_correct)

    for r in results:
        r.del_num_correct = sum(r.del_correctness)

    n_success = sum(1 for r in results if r.del_num_correct > 0)
    print(f"  Done: {n_success}/{len(results)} have ≥1 correct after random-del")
    return results


def random_del_results_to_dicts(results: List[RandomRegenerationResult]) -> List[Dict]:
    return [{
        "path_id": r.path_id, "golden_answer": r.golden_answer,
        "truncation_position": r.truncation_position,
        "path_is_correct": r.path_is_correct,
        "del_num_correct": r.del_num_correct, "del_correctness": r.del_correctness,
        "num_samples": r.num_samples,
    } for r in results]
