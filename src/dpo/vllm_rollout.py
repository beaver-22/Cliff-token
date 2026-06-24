"""
Step 1: Top-10 Candidate Extraction & Potential Rollout (Chunked + Resumable)

Pipeline:
  Phase A — Extract Top-10 candidates at each cliff via logprobs (saved to phase_a.json).
  Phase B — For each non-cliff candidate, run 64 rollouts → measure potential.
             Processed in chunks of N cliffs (default 10). Each chunk writes to
             partial.jsonl + checkpoint.jsonl incrementally, so a crash loses
             at most one chunk's work.
  Merge    — Read partial.jsonl to produce the final cliff_candidates.json.

Cliff tokens themselves are NOT re-rolled (their potential = cliff_score from
the existing rollout data).

Usage:
    python -m src.dpo.vllm_rollout \
        --model qwen3-0.6b --dataset gsm8k \
        --data_path ./output/03_rollout/Qwen3-0.6B/gsm8k_all_paths.json \
        --gpus 0 --chunk_size 10
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src import config
from src.analysis.cliff_threshold import score_to_k
from src.analysis.detector import find_all_cliff_tokens_statistical
from src.analysis.entropy import (
    _logprob_value,
    compute_entropy_from_logprobs,
    compute_tie_aware_ranks,
    GREEDY_BOUND_NATS,
)
from src.dpo.logging_utils import parse_log_level, setup_logger
from src.utils.grader import batch_grade_responses_mathverify

# Deterministic/Uncertain/Sampled-off entropy boundary. Canonical source lives in
# `src.analysis.entropy.GREEDY_BOUND_NATS` (derived from
# `GREEDY_PROB_THRESHOLD`; default 0.99, swap to 0.95 in one place to
# experiment with alternative boundary).
GREEDY_99_BOUND_NATS = GREEDY_BOUND_NATS  # legacy alias used within this module

# Module-level logger (reconfigured in main())
logger = logging.getLogger("dpo.step1_rollout")


# ============================================================
# Data structures
# ============================================================

@dataclass
class CandidateResult:
    """Rollout result for a single candidate token."""
    token_id: int
    token_str: str
    rank: int
    prob: float
    is_cliff_token: bool
    num_correct: int
    num_samples: int
    potential: float


@dataclass
class CliffCandidateAnalysis:
    """All candidates for one cliff position."""
    path_id: str
    problem_id: str
    question: str
    golden_answer: List[str]
    cliff_position: int
    cliff_token_id: int
    cliff_token_str: Optional[str]
    prev_score: float
    cliff_score: float
    drop_magnitude: float
    category: str
    entropy_at_cliff: float
    is_greedy: bool
    candidates: List[CandidateResult] = field(default_factory=list)
    prefix_text: str = ""
    # IMPORTANT: store actual token IDs to avoid BPE re-tokenization issues.
    # When using single-token DPO, these IDs must be used directly so loss
    # flows on exactly the cliff token.
    prefix_token_ids: List[int] = field(default_factory=list)


# ============================================================
# Cliff token classification (deterministic / uncertain / sampled_off)
# ============================================================

def classify_cliff(entropy: float, is_greedy: bool) -> str:
    if entropy <= GREEDY_99_BOUND_NATS and is_greedy:
        return "deterministic"
    elif entropy > GREEDY_99_BOUND_NATS and is_greedy:
        return "uncertain"
    elif entropy > GREEDY_99_BOUND_NATS and not is_greedy:
        return "sampled_off"
    else:
        return "other"


# ============================================================
# Phase A: Top-10 Candidate Extraction
# ============================================================

def extract_top_k_candidates(
    llm,
    tokenizer,
    paths: List[Dict],
    k: int = 10,
    logprobs_k: int = 20,
    lora_request=None,
) -> List[CliffCandidateAnalysis]:
    """For each cliff in each path, extract top-k candidate tokens at that position.

    When `lora_request` is provided, the single forward pass used to get
    next-token logprobs at each cliff position is routed through the LoRA
    adapter so classification reflects the adapter's behavior.
    """
    logger.info(f"[Phase A] Extracting top-{k} candidates at cliff positions"
                + (f" (lora={lora_request.lora_name})" if lora_request else ""))

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=logprobs_k,
    )

    requests = []  # (path_idx, cliff_info, prefix_ids, prompt_ids)
    for path_idx, p in enumerate(paths):
        scores = p.get("all_position_scores", [])
        if not scores:
            continue
        cliffs = find_all_cliff_tokens_statistical(
            scores,
            tokens=p.get("response_tokens"),
            token_ids=p.get("response_token_ids"),
        )
        if not cliffs:
            continue

        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        for cliff in cliffs:
            truncate_pos = cliff.position - 1  # 0-indexed
            prefix_ids = prompt_ids + p["response_token_ids"][:truncate_pos]
            requests.append((path_idx, cliff, prefix_ids, prompt_ids))

    if not requests:
        logger.warning("No cliff tokens found.")
        return []

    logger.info(f"  {len(requests)} cliff positions found")

    prompts = [{"prompt_token_ids": req[2]} for req in requests]
    _gen_kwargs = {"lora_request": lora_request} if lora_request else {}
    outputs = llm.generate(prompts, sampling_params, **_gen_kwargs)

    MISSING_LOGPROBS_WARN_LIMIT = 5  # first N at WARN, rest at DEBUG
    n_missing_logprobs = 0

    analyses = []
    for (path_idx, cliff, prefix_ids, prompt_ids), output in zip(requests, outputs):
        p = paths[path_idx]
        sample = output.outputs[0]
        logprob_dict = sample.logprobs[0] if sample.logprobs else {}

        entropy = compute_entropy_from_logprobs(logprob_dict)

        if logprob_dict:
            # Tie-aware competition rank: tokens within EPS of the top logprob
            # all get rank=1, so a cliff token that happens to share the top
            # probability with another token is still classified as greedy.
            tie_ranks = compute_tie_aware_ranks(logprob_dict)
            cliff_rank = tie_ranks.get(cliff.token_id, logprobs_k + 1)
            is_greedy = (cliff_rank == 1)
        else:
            # Defensive fallback: vLLM may rarely omit logprobs (e.g., engine
            # edge case). sample.token_ids is independent and still allows
            # sampling-time greedy comparison. Log first few at WARN, rest at
            # DEBUG to avoid log explosion on large runs.
            n_missing_logprobs += 1
            _msg = (f"[Phase A] Missing logprobs for path={p['id']} "
                    f"cliff_pos={cliff.position}; fallback to token_id comparison.")
            if n_missing_logprobs <= MISSING_LOGPROBS_WARN_LIMIT:
                logger.warning(_msg)
            else:
                logger.debug(_msg)
            tie_ranks = {}
            greedy_token_id = sample.token_ids[0] if sample.token_ids else None
            cliff_rank = logprobs_k + 1
            is_greedy = (greedy_token_id is not None
                         and cliff.token_id == greedy_token_id)

        # Top-k candidates from logprobs, sorted by probability desc
        candidates_raw = []
        for tid, entry in logprob_dict.items():
            lp = _logprob_value(entry)
            prob = math.exp(lp) if lp > float("-inf") else 0.0
            rank = tie_ranks.get(tid, logprobs_k + 1)
            token_str = tokenizer.decode([tid], skip_special_tokens=False)
            candidates_raw.append((tid, token_str, rank, prob))
        candidates_raw.sort(key=lambda x: x[3], reverse=True)
        candidates_raw = candidates_raw[:k]

        candidate_results = []
        for tid, tok_str, rank, prob in candidates_raw:
            is_cliff_tok = (tid == cliff.token_id)
            if is_cliff_tok:
                # Reuse existing cliff_score as potential (no new rollout)
                potential = cliff.curr_score
                k_correct = score_to_k(cliff.curr_score)
            else:
                potential = -1.0
                k_correct = 0
            candidate_results.append(CandidateResult(
                token_id=tid,
                token_str=tok_str,
                rank=rank,
                prob=prob,
                is_cliff_token=is_cliff_tok,
                num_correct=k_correct,
                num_samples=64,
                potential=potential,
            ))

        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        category = classify_cliff(entropy, is_greedy)

        analyses.append(CliffCandidateAnalysis(
            path_id=p["id"],
            problem_id=p["problem_id"],
            question=p["question"],
            golden_answer=p.get("golden_answer", []),
            cliff_position=cliff.position,
            cliff_token_id=cliff.token_id,
            cliff_token_str=cliff.token_str,
            prev_score=cliff.prev_score,
            cliff_score=cliff.curr_score,
            drop_magnitude=cliff.drop_magnitude,
            category=category,
            entropy_at_cliff=entropy,
            is_greedy=is_greedy,
            candidates=candidate_results,
            prefix_text=prefix_text,
            prefix_token_ids=list(prefix_ids),
        ))

    if n_missing_logprobs > 0:
        logger.warning(
            f"[Phase A] {n_missing_logprobs}/{len(requests)} cliffs had missing "
            f"logprobs and used token_id fallback for is_greedy "
            f"(first {min(MISSING_LOGPROBS_WARN_LIMIT, n_missing_logprobs)} at WARN, rest at DEBUG)."
        )

    n_deterministic = sum(1 for a in analyses if a.category == "deterministic")
    n_uncertain = sum(1 for a in analyses if a.category == "uncertain")
    n_sampled_off = sum(1 for a in analyses if a.category == "sampled_off")
    logger.info(
        f"  Categories: deterministic={n_deterministic}, uncertain={n_uncertain}, "
        f"sampled_off={n_sampled_off}, other={len(analyses)-n_deterministic-n_uncertain-n_sampled_off}"
    )
    return analyses


# ============================================================
# Phase B: Chunked Candidate Rollout with incremental checkpoint
# ============================================================

def _cliff_key(a: CliffCandidateAnalysis) -> str:
    return f"{a.path_id}_{a.cliff_position}"


def _load_completed_keys(checkpoint_path: str) -> set:
    completed = set()
    if not os.path.exists(checkpoint_path):
        return completed
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "key" in rec:
                    completed.add(rec["key"])
            except json.JSONDecodeError:
                continue
    return completed


def _append_chunk_results(
    partial_path: str,
    checkpoint_path: str,
    chunk: List[CliffCandidateAnalysis],
):
    """Atomically append a chunk's results to both partial and checkpoint files."""
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    with open(partial_path, "a", encoding="utf-8") as pf:
        for analysis in chunk:
            pf.write(json.dumps(asdict(analysis), ensure_ascii=False) + "\n")
    with open(checkpoint_path, "a", encoding="utf-8") as cf:
        for analysis in chunk:
            cf.write(json.dumps({"key": _cliff_key(analysis)}) + "\n")


def _load_partial_analyses(partial_path: str) -> List[Dict]:
    """Load all previously-saved analyses from partial JSONL."""
    results = []
    if not os.path.exists(partial_path):
        return results
    with open(partial_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def run_candidate_rollouts_chunked(
    llm,
    tokenizer,
    analyses: List[CliffCandidateAnalysis],
    paths: List[Dict],
    dataset_name: str,
    model_path: str,
    partial_path: str,
    checkpoint_path: str,
    mode: str = "non_thinking",
    num_samples: int = 64,
    chunk_size: int = 10,
) -> None:
    """Chunked Phase B. Writes results to partial_path and checkpoint_path incrementally.

    Does NOT mutate the returned list — callers should reload from partial_path.
    """
    logger.info(f"[Phase B] Running candidate rollouts (chunk_size={chunk_size}, samples={num_samples})")

    sampling_cfg = config.get_sampling_config(mode, model_path)
    source_name = config.get_dataset_source_name(dataset_name)
    max_new_tokens = config.get_rollout_max_tokens(dataset_name, mode)

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

    completed_keys = _load_completed_keys(checkpoint_path)
    if completed_keys:
        logger.info(f"  Resuming: {len(completed_keys)} cliff positions already completed")

    # Filter to pending analyses
    pending = [a for a in analyses if _cliff_key(a) not in completed_keys]
    logger.info(f"  Pending: {len(pending)} cliffs to process")
    if not pending:
        logger.info("  Nothing to do.")
        return

    paths_by_id = {p["id"]: p for p in paths}
    total_chunks = (len(pending) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk = pending[chunk_start:chunk_start + chunk_size]

        # Collect requests for this chunk
        requests = []  # (a_idx_in_chunk, c_idx, prefix_ids, golden, prompt_ids)
        for ci_local, analysis in enumerate(chunk):
            p = paths_by_id.get(analysis.path_id)
            if not p:
                logger.warning(f"  [chunk {chunk_idx+1}] path {analysis.path_id} not found in rollout data, skipping")
                continue
            prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
            truncate_pos = analysis.cliff_position - 1
            for c_idx, cand in enumerate(analysis.candidates):
                if cand.is_cliff_token:
                    continue  # skip cliff itself — reuse cliff_score
                prefix_ids = prompt_ids + p["response_token_ids"][:truncate_pos] + [cand.token_id]
                requests.append((ci_local, c_idx, prefix_ids, p.get("golden_answer", []), prompt_ids))

        if not requests:
            # All candidates in this chunk are cliff-only (unlikely), mark done anyway
            logger.info(f"  [chunk {chunk_idx+1}/{total_chunks}] no non-cliff candidates, marking complete")
            _append_chunk_results(partial_path, checkpoint_path, chunk)
            continue

        # Sort by prefix length for prefix caching efficiency
        requests.sort(key=lambda r: len(r[2]))

        logger.info(
            f"  [chunk {chunk_idx+1}/{total_chunks}] {len(chunk)} cliffs, "
            f"{len(requests)} candidates × {num_samples} = {len(requests)*num_samples} rollouts"
        )

        prompts = [{"prompt_token_ids": req[2]} for req in requests]
        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            logger.error(f"  [chunk {chunk_idx+1}] vLLM generate failed: {e}")
            raise  # re-raise so user can investigate; already-saved chunks are safe

        # Grade
        all_responses = []
        all_golden = []
        response_map = []
        for req_idx, output in enumerate(outputs):
            _, _, prefix_ids, golden, prompt_ids = requests[req_idx]
            prefix_response_ids = prefix_ids[len(prompt_ids):]
            prefix_text = tokenizer.decode(prefix_response_ids, skip_special_tokens=True)
            for sample in output.outputs:
                all_responses.append(prefix_text + sample.text)
                all_golden.append(golden)
                response_map.append(req_idx)

        correctness = batch_grade_responses_mathverify(all_responses, all_golden, source_name)

        # Aggregate per candidate
        cand_correct: Dict[tuple, List[bool]] = {}
        for resp_idx, is_correct in enumerate(correctness):
            req_idx = response_map[resp_idx]
            ci_local, c_idx, *_ = requests[req_idx]
            key = (ci_local, c_idx)
            cand_correct.setdefault(key, []).append(is_correct)

        for (ci_local, c_idx), correct_list in cand_correct.items():
            cand = chunk[ci_local].candidates[c_idx]
            cand.num_correct = sum(correct_list)
            cand.num_samples = len(correct_list)
            cand.potential = cand.num_correct / cand.num_samples if cand.num_samples > 0 else 0.0

        # Incremental save (crash-safe)
        _append_chunk_results(partial_path, checkpoint_path, chunk)
        logger.info(f"  [chunk {chunk_idx+1}/{total_chunks}] saved {len(chunk)} cliffs to partial+checkpoint")

    logger.info("[Phase B] Complete.")


# ============================================================
# Full pipeline
# ============================================================

def _shard_paths(output_dir: str, dataset_name: str, shard_id: int, num_shards: int):
    suffix = f"_shard{shard_id}of{num_shards}" if num_shards > 1 else ""
    return (
        os.path.join(output_dir, f"{dataset_name}{suffix}_phase_a.json"),
        os.path.join(output_dir, f"{dataset_name}{suffix}_candidates.partial.jsonl"),
        os.path.join(output_dir, f"{dataset_name}{suffix}_checkpoint.jsonl"),
        os.path.join(output_dir, f"{dataset_name}_cliff_candidates.json"),
    )


def merge_shards(output_dir: str, dataset_name: str, num_shards: int) -> str:
    """Merge per-shard partial.jsonl files into the final cliff_candidates.json."""
    final_path = os.path.join(output_dir, f"{dataset_name}_cliff_candidates.json")
    all_results = []
    for i in range(num_shards):
        _, partial_path, _, _ = _shard_paths(output_dir, dataset_name, i, num_shards)
        shard_results = _load_partial_analyses(partial_path)
        logger.info(f"  shard {i}: {len(shard_results)} cliffs from {partial_path}")
        all_results.extend(shard_results)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Merged {num_shards} shards → {len(all_results)} cliffs → {final_path}")
    return final_path


def run_full_pipeline(
    model_path: str,
    dataset_name: str,
    data_path: str,
    output_dir: str,
    mode: str = "non_thinking",
    gpu_ids: Optional[List[int]] = None,
    k_candidates: int = 10,
    num_samples: int = 64,
    chunk_size: int = 10,
    force: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
) -> str:
    """End-to-end: load → Phase A (cached) → Phase B (chunked) → merge.

    When num_shards > 1, this process handles only paths[shard_id::num_shards]
    and writes shard-suffixed phase_a/partial/checkpoint files. The final
    cliff_candidates.json is NOT written here — call merge_shards() after all
    shards finish.
    """
    if gpu_ids is None:
        gpu_ids = [0]

    os.makedirs(output_dir, exist_ok=True)

    phase_a_path, partial_path, checkpoint_path, final_path = _shard_paths(
        output_dir, dataset_name, shard_id, num_shards
    )

    if force:
        logger.info("--force: clearing existing phase_a/partial/checkpoint files")
        for p in [phase_a_path, partial_path, checkpoint_path]:
            if os.path.exists(p):
                os.remove(p)

    # Load rollout data
    logger.info(f"Loading rollout data from {data_path}")
    with open(data_path) as f:
        paths = json.load(f)
    logger.info(f"  {len(paths)} paths loaded")
    paths = [p for p in paths if p.get("all_position_scores")]
    logger.info(f"  {len(paths)} paths with position scores")

    if num_shards > 1:
        sharded = paths[shard_id::num_shards]
        logger.info(
            f"Shard {shard_id}/{num_shards}: processing {len(sharded)}/{len(paths)} paths"
        )
        paths = sharded

    # Init vLLM. Skip CUDA_VISIBLE_DEVICES override in shard mode — the parent
    # shell pins each shard to its own GPU via env, and overriding here would
    # confuse the device remapping.
    if num_shards <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    logger.info(f"Initializing vLLM on GPU(s) {gpu_ids}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        max_num_seqs=config.MAX_NUM_SEQS,
        max_num_batched_tokens=config.MAX_NUM_BATCHED_TOKENS,
        disable_cascade_attn=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Phase A: extract candidates (cached)
    if os.path.exists(phase_a_path):
        logger.info(f"[Phase A] Loading cached result from {phase_a_path}")
        with open(phase_a_path) as f:
            raw = json.load(f)
        analyses = []
        for d in raw:
            cands = [CandidateResult(**c) for c in d["candidates"]]
            d2 = {**d, "candidates": cands}
            analyses.append(CliffCandidateAnalysis(**d2))
        logger.info(f"  Loaded {len(analyses)} cliff analyses from cache")
    else:
        analyses = extract_top_k_candidates(llm, tokenizer, paths, k=k_candidates)
        if not analyses:
            logger.warning("No cliff positions found. Exiting.")
            return ""
        with open(phase_a_path, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in analyses], f, indent=2, ensure_ascii=False)
        logger.info(f"[Phase A] Saved to {phase_a_path}")

    # Phase B: chunked rollout with checkpointing
    run_candidate_rollouts_chunked(
        llm=llm,
        tokenizer=tokenizer,
        analyses=analyses,
        paths=paths,
        dataset_name=dataset_name,
        model_path=model_path,
        partial_path=partial_path,
        checkpoint_path=checkpoint_path,
        mode=mode,
        num_samples=num_samples,
        chunk_size=chunk_size,
    )

    if num_shards > 1:
        logger.info(
            f"Shard {shard_id}/{num_shards} done. Run with --merge_only after all shards finish."
        )
        return ""

    # Merge partial → final
    logger.info(f"Merging partial → final: {final_path}")
    partial_results = _load_partial_analyses(partial_path)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(partial_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(partial_results)} cliff analyses to {final_path}")

    # Summary
    total_candidates = sum(len(a["candidates"]) for a in partial_results)
    cliff_tokens = sum(1 for a in partial_results for c in a["candidates"] if c["is_cliff_token"])
    logger.info(f"Summary: {len(partial_results)} cliffs, {total_candidates} candidates ({cliff_tokens} cliff tokens reused)")

    return final_path


# ============================================================
# CLI
# ============================================================

def _default_output_dir(model_path: str) -> str:
    model_short = config.get_model_short_name(model_path)
    return f"./output/09_cliff_dpo/01_candidates/{model_short}"


def main():
    parser = argparse.ArgumentParser(description="DPO candidate rollout (chunked, resumable)")
    parser.add_argument("--model", required=True, help="Model alias or path")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. gsm8k)")
    parser.add_argument("--data_path", required=True, help="Path to rollout all_paths.json")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Default: ./output/09_cliff_dpo/01_candidates/{model_short}/")
    parser.add_argument("--mode", default="non_thinking", choices=["thinking", "non_thinking"])
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--k_candidates", type=int, default=10, help="Top-k candidates")
    parser.add_argument("--num_samples", type=int, default=64, help="Rollout samples per candidate")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Cliffs per chunk before incremental save (default 10)")
    parser.add_argument("--force", action="store_true",
                        help="Clear existing phase_a/partial/checkpoint files before running")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Index of this shard (0-based). Used with --num_shards for data parallel.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of data-parallel shards. Paths are split via paths[shard_id::num_shards].")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip rollout; just merge existing per-shard partial files into the final output.")
    parser.add_argument("--log_dir", default="./output/09_cliff_dpo/logs",
                        help="Directory for log files")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    model_path = config.resolve_model_path(args.model)
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    output_dir = args.output_dir or _default_output_dir(model_path)

    global logger
    logger = setup_logger(
        name=f"step1_rollout_{config.get_model_short_name(model_path)}_{args.dataset}",
        log_dir=args.log_dir,
        level=parse_log_level(args.log_level),
    )
    logger.info(f"Config: model={model_path}, dataset={args.dataset}, output={output_dir}, "
                f"chunk_size={args.chunk_size}, gpus={gpu_ids}, "
                f"shard={args.shard_id}/{args.num_shards}")

    if args.merge_only:
        if args.num_shards <= 1:
            logger.error("--merge_only requires --num_shards > 1")
            return
        merge_shards(output_dir, args.dataset, args.num_shards)
        return

    run_full_pipeline(
        model_path=model_path,
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_dir=output_dir,
        mode=args.mode,
        gpu_ids=gpu_ids,
        k_candidates=args.k_candidates,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        force=args.force,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
