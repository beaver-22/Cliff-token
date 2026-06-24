"""
Tangent Token Decoder: truncate at first tangent chunk start, regenerate N times.

Tangent: 20-chunk division, chunk with start-end score drop >= 0.3.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from vllm import SamplingParams

from src.analysis.detector import find_all_cliff_tokens, find_all_cliff_tokens_statistical
from src.analysis.positional import _compute_chunk_boundaries, _detect_tangents

# Default to statistical cliff detection
USE_STATISTICAL_CLIFF = True
from src.utils.grader import batch_grade_responses_mathverify
from src import config


@dataclass
class TangentRegenerationResult:
    path_id: str
    golden_answer: List[str]
    tangent_start_position: Optional[int]  # 0-indexed
    tangent_found: bool
    path_is_correct: bool

    del_responses: List[str] = field(default_factory=list)
    del_correctness: List[bool] = field(default_factory=list)
    del_num_correct: int = 0
    num_samples: int = 64


def run_tangent_del_on_paths(
    llm, tokenizer, paths: List[Dict], dataset_name: str,
    num_samples: int = 64, mode: str = "non_thinking", model_path: str = None,
) -> List[TangentRegenerationResult]:
    """Run Tangent-del: truncate at first tangent chunk start, regenerate."""
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

    print(f"  Detecting tangent chunks...")
    requests = []
    results = []

    for p in paths:
        scores = p.get("all_position_scores", [])
        if USE_STATISTICAL_CLIFF:
            cliffs = find_all_cliff_tokens_statistical(scores)
        else:
            cliffs = find_all_cliff_tokens(scores, config.DEFAULT_CLIFF_THRESHOLD)
        cliff_positions = [c.position for c in cliffs]
        _, _, chunks, _ = _detect_tangents(scores, cliff_positions)

        first_tangent = next((c for c in chunks if c.is_tangent), None)
        result = TangentRegenerationResult(
            path_id=p["id"],
            golden_answer=p.get("golden_answer", []),
            tangent_start_position=first_tangent.start_idx if first_tangent else None,
            tangent_found=first_tangent is not None,
            path_is_correct=p.get("is_correct", False),
            num_samples=num_samples,
        )
        results.append(result)

        if first_tangent:
            prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
            truncate_pos = first_tangent.start_idx  # 0-indexed, truncate at chunk start
            prefix_ids = prompt_ids + p["response_token_ids"][:truncate_pos]
            requests.append((len(results) - 1, prefix_ids, p.get("golden_answer", []), p))

    if not requests:
        print("  No tangent chunks found.")
        return results

    print(f"  {len(requests)} paths with tangent × {num_samples} samples")
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

    n_success = sum(1 for r in results if r.del_num_correct > 0 and r.tangent_found)
    n_with = sum(1 for r in results if r.tangent_found)
    print(f"  Done: {n_success}/{n_with} have ≥1 correct after tangent-del")
    return results


def tangent_del_results_to_dicts(results: List[TangentRegenerationResult]) -> List[Dict]:
    return [{
        "path_id": r.path_id, "golden_answer": r.golden_answer,
        "tangent_start_position": r.tangent_start_position,
        "tangent_found": r.tangent_found, "path_is_correct": r.path_is_correct,
        "del_num_correct": r.del_num_correct, "del_correctness": r.del_correctness,
        "num_samples": r.num_samples,
    } for r in results]
