"""
Generation & Evaluation Framework for Cliff Token Analysis

=============================================================================
Overall Pipeline Overview
=============================================================================

                        ┌─────────────────────┐
                        │   Dataset (JSONL)    │
                        │ question / answer    │
                        └────────┬────────────┘
                                 │ sample_problems()
                                 ▼
┌══════════════════════════════════════════════════════════════════════════┐
║  Phase 1 — Inference (generate_reasoning_paths)                         ║
║                                                                          ║
║  build_chat_prompt()                                                     ║
║  ├─ zeroshot : system="reason step by step, put answer in \\boxed{}"    ║
║  ├─ fewshot  : few-shot CoT examples (MATH_PROMPT / GSM8K_PROMPT)       ║
║  └─ direct   : few-shot, answer-only (no reasoning)                     ║
║       │                                                                  ║
║       ▼  tokenizer.apply_chat_template()                                 ║
║  vLLM generate() ─────────────────────────────────────────────────────  ║
║       │  temperature / top_p / top_k / presence_penalty                 ║
║       ▼                                                                  ║
║  check_answer()  [main thread → signal timeout 5s available]            ║
║  ├─ Stage 1: _extract_for_grading()  last \\boxed{} slicing             ║
║  └─ Stage 2: math-verify parse() + verify()  (SymPy symbolic)           ║
║       Fallback: Hendrycks strip_string + numeric isclose                 ║
║                                                                          ║
║  → ReasoningPath(response, is_correct, tokens, ...)                     ║
╚══════════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
┌═══════════════════════════════════════════════════════════════════════════
║  Phase 2 — Rollout Scoring (compute_position_scores_optimized)          ║
║                                                                          ║
║  collect_rollout_requests()                                              ║
║    All paths × token positions → RolloutRequest (prefix_token_ids)      ║
║    Sort: prefix_length ascending (minimize vLLM padding)                ║
║                                                                          ║
║  _run_pipelined_batches()  ── GPU/CPU parallel pipeline ──              ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │   GPU thread (producer)          CPU thread (consumer)          │    ║
║  │                                                                  │    ║
║  │  for batch in all_requests:    while True:                      │    ║
║  │    outputs = llm.generate()      item = queue.get()            │    ║
║  │    queue.put(outputs)  ──────►   batch_grade_responses_mv()    │    ║
║  │    # GIL released during         # Stage1+Stage2 grading       │    ║
║  │    # vLLM IPC wait               # _grade_with_timeout_guard() │    ║
║  │                                  all_scores[(idx,pos)] = score │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║  queue(maxsize=2): GPU can run at most 1 batch ahead, memory bounded    ║
║                                                                          ║
║  → ReasoningPath.all_position_scores[t] = P(correct | prefix[:t])      ║
╚══════════════════════════════════════════════════════════════════════════╝

=============================================================================
Grading Pipeline Details (see grader.py)
=============================================================================

  Model output
      │
      ▼ Stage 1: _extract_for_grading()
  last \\boxed{} rfind slicing  → keep wrapper (preserve anchor)
  "the answer is X" pattern     → GSM1K fallback
  full response                 → last resort fallback
      │
      ▼ Stage 2: math-verify
  mv_parse(candidate, [LatexExtractionConfig(), ExprExtractionConfig()])
      │ ANTLR4 grammar → SymPy object
      ▼
  mv_verify(gold_parsed, pred_parsed)
      1. string equality
      2. numeric precision (float_rounding=6, numeric_precision=15)
      3. symbolic equality (SymPy simplify)

threading constraints:
  - Inference (main thread): parsing_timeout=5, timeout_seconds=5 (signal.SIGALRM)
  - Rollout   (worker thread): timeout=0 + daemon thread join (wall-clock guard)

=============================================================================
Key Data Structures
=============================================================================

  ReasoningPath
  ├─ id, problem_id, question, golden_answer
  ├─ response          : model-generated text
  ├─ is_correct        : Phase 1 grading result
  ├─ response_tokens   : token-level decomposition
  ├─ response_token_ids
  ├─ all_position_scores : Phase 2 rollout score [t] = P(correct | prefix[:t])
  ├─ total_tokens
  └─ full_prompt       : actual input prompt (after chat template is applied)

  RolloutRequest
  ├─ path_idx, position
  ├─ prompt_token_ids  : full_prompt + response_prefix concatenated
  └─ prefix_length     : sort key (minimize padding)
"""

import json
import multiprocessing as mp
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from src import config
from src.config import (
    SamplingConfig, MATH_PROMPT, GSM8K_PROMPT, ZEROSHOT_SYSTEM_PROMPT,
    MATH_DIRECT_PROMPT, GSM8K_DIRECT_PROMPT, DIRECT_SYSTEM_PROMPT,
)
from src.utils.grader import _grade_with_timeout_guard, batch_grade_responses_mathverify


@dataclass
class ReasoningPath:
    """Data class for a single reasoning path."""
    id: str
    problem_id: str
    question: str
    golden_answer: List[str]
    response: str
    is_correct: bool
    response_tokens: List[str]
    response_token_ids: List[int]
    all_position_scores: List[Optional[float]]
    total_tokens: int
    full_prompt: str
    # Optional: per-token rank in vocab distribution at sampling time
    # (1-indexed; populated only when capture_ranks=True is passed to
    # generate_reasoning_paths). Empty list otherwise to keep existing
    # inference / rollout pipelines unaffected.
    response_token_ranks: List[int] = field(default_factory=list)


@dataclass
class RolloutRequest:
    """Metadata for a single rollout request (used in optimized batching)."""
    path_idx: int           # Index into original paths list
    position: int           # Token position (1-indexed)
    prompt_token_ids: List[int] = field(repr=False)  # Full prompt + prefix token IDs
    prefix_length: int = 0  # Length of prefix (for sorting)
    golden_answers: List[str] = field(default_factory=list)  # For grading

    def __post_init__(self):
        if self.prefix_length == 0:
            self.prefix_length = len(self.prompt_token_ids)

    def __lt__(self, other):
        """Enable sorting by prefix length."""
        return self.prefix_length < other.prefix_length


def collect_rollout_requests(
    tokenizer,
    paths: List[ReasoningPath],
    rollout_window: int = 1,
) -> Tuple[List[RolloutRequest], Dict[Tuple[int, int], int], Dict[int, List[int]]]:
    """
    Collect all rollout requests from all paths for global batching.

    This function enables the optimized rollout by:
    1. Collecting ALL prefix requests from ALL paths upfront
    2. Sorting by prefix length to minimize padding overhead
    3. Creating an index for efficient result reconstruction
    4. Caching prompt token IDs to avoid redundant encoding

    Args:
        tokenizer: Tokenizer instance
        paths: List of ReasoningPath objects
        rollout_window: Compute scores every N tokens (1 = every token)

    Returns:
        all_requests: List of RolloutRequest sorted by prefix length
        request_index: Mapping from (path_idx, position) to request index
        prompt_token_cache: Mapping from path_idx to prompt token IDs
    """
    all_requests = []
    prompt_token_cache = {}

    for path_idx, path in enumerate(paths):
        if path.total_tokens < 2:
            continue

        # Encode full prompt once per path and cache
        prompt_token_ids = tokenizer.encode(path.full_prompt, add_special_tokens=False)
        prompt_token_cache[path_idx] = prompt_token_ids

        for pos in range(1, len(path.response_token_ids), rollout_window):
            prefix_ids = path.response_token_ids[:pos]
            full_ids = prompt_token_ids + prefix_ids

            all_requests.append(RolloutRequest(
                path_idx=path_idx,
                position=pos,
                prompt_token_ids=full_ids,
                prefix_length=len(full_ids),
                golden_answers=path.golden_answer,
            ))

    # Sort by prefix length for efficient padding
    all_requests.sort()

    # Build index for result reconstruction
    request_index = {
        (req.path_idx, req.position): i
        for i, req in enumerate(all_requests)
    }

    return all_requests, request_index, prompt_token_cache


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data: Any, path: str, indent=None):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def check_answer(response: str, golden_answers: List[str], dataset_name: str) -> bool:
    """Check if response contains correct answer.

    Protected by a thread-based 5-second timeout.
    Unlike SIGALRM, this reliably interrupts even C extension hangs.
    """
    return _grade_with_timeout_guard(response, golden_answers, dataset_name, wall_timeout=5.0)


def sample_problems(dataset_path: str, num_problems: int, seed: int = 42) -> List[Dict]:
    """Sample problems from the dataset."""
    all_problems = load_jsonl(dataset_path)
    random.seed(seed)

    if num_problems >= len(all_problems):
        return all_problems

    return random.sample(all_problems, num_problems)


def build_chat_prompt(
    question: str,
    tokenizer,
    dataset_name: str,
    enable_thinking: bool = True,
    use_chat_template: bool = True,
    prompt_type: str = "fewshot",
) -> str:
    """Build prompt using chat template or raw few-shot format.

    For instruct/chat models, wraps in chat template.
    For base models (use_chat_template=False), uses raw few-shot completion format.

    Args:
        prompt_type: "fewshot" (few-shot CoT), "zeroshot" (CoT instruction), or "direct" (few-shot, answer only)
    """
    source_name = config.get_dataset_source_name(dataset_name)

    if prompt_type == "zeroshot":
        # Qwen3 official style: CoT instruction in system prompt, bare question in user.
        user_content = question
        system_content = ZEROSHOT_SYSTEM_PROMPT
    elif prompt_type == "direct":
        few_shot = MATH_DIRECT_PROMPT if source_name in ("MATH", "AIME") else GSM8K_DIRECT_PROMPT
        user_content = few_shot + f"Q: {question}\nA:"
        system_content = DIRECT_SYSTEM_PROMPT
    else:
        few_shot = MATH_PROMPT if source_name in ("MATH", "AIME") else GSM8K_PROMPT
        if source_name in ("MATH", "AIME"):
            user_content = few_shot + f"Problem: {question}\nSolution:"
        else:
            user_content = few_shot + f"Question: {question}\nAnswer:"
        system_content = None

    if not use_chat_template:
        if system_content:
            return system_content + "\n\n" + user_content
        return user_content

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def generate_reasoning_paths(
    llm: LLM,
    tokenizer,
    problems: List[Dict],
    paths_per_problem: int,
    dataset_name: str,
    sampling_config: SamplingConfig,
    max_new_tokens: Optional[int] = None,
    capture_ranks: bool = False,
    lora_request=None,
) -> List[ReasoningPath]:
    """Generate multiple reasoning paths for each problem.

    Args:
        capture_ranks: When True, asks vLLM to also return top-K logprobs at
            each generated position so we can record each sampled token's rank
            in the model's vocab distribution. Stored in
            `ReasoningPath.response_token_ranks`.
    """
    actual_max_tokens = max_new_tokens if max_new_tokens else sampling_config.max_tokens
    source_name = config.get_dataset_source_name(dataset_name)

    prompts = []
    prompt_to_problem = []
    for problem in problems:
        full_prompt = build_chat_prompt(
            question=problem["question"],
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            enable_thinking=sampling_config.enable_thinking,
            use_chat_template=sampling_config.use_chat_template,
            prompt_type=sampling_config.prompt_type,
        )
        for path_idx in range(paths_per_problem):
            prompts.append(full_prompt)
            prompt_to_problem.append((problem, path_idx, full_prompt))

    sp_kwargs = dict(
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        top_k=sampling_config.top_k,
        presence_penalty=sampling_config.presence_penalty,
        repetition_penalty=sampling_config.repetition_penalty,
        max_tokens=actual_max_tokens,
        stop=config.STOP_TOKENS,
    )
    if capture_ranks:
        # top-20 covers all sampled tokens because top_k=20 sampling
        sp_kwargs["logprobs"] = 20
    sampling_params = SamplingParams(**sp_kwargs)

    mode_str = "thinking" if sampling_config.enable_thinking else "non-thinking"
    rank_msg = " +rank-capture" if capture_ranks else ""
    lora_msg = f" (lora={lora_request.lora_name})" if lora_request else ""
    print(f"Generating {len(prompts)} reasoning paths ({mode_str} mode, T={sampling_config.temperature}){rank_msg}{lora_msg}...")
    _gen_kwargs = {"lora_request": lora_request} if lora_request else {}
    outputs = llm.generate(prompts, sampling_params, **_gen_kwargs)

    paths = []
    for i, output in enumerate(outputs):
        problem, path_idx, full_prompt = prompt_to_problem[i]
        sample = output.outputs[0]
        response = sample.text

        if capture_ranks:
            # Use vLLM's exact generated IDs (avoid lossy text→encode round-trip
            # so logprobs[t] aligns 1:1 with token_ids[t]).
            response_token_ids = list(sample.token_ids)
            response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]
            ranks: List[int] = []
            if sample.logprobs:
                from src.analysis.entropy import compute_tie_aware_ranks
                for t, tok_id in enumerate(response_token_ids):
                    lp_dict = sample.logprobs[t] if t < len(sample.logprobs) else None
                    if lp_dict and tok_id in lp_dict:
                        tie_ranks = compute_tie_aware_ranks(lp_dict)
                        ranks.append(tie_ranks.get(tok_id, 21))
                    else:
                        # Sampled token not in returned top-K (shouldn't happen
                        # with top_k=20 sampling, but be defensive)
                        ranks.append(21)
        else:
            response_token_ids = tokenizer.encode(response, add_special_tokens=False)
            response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]
            ranks = []

        is_correct = check_answer(response, problem["answer"], dataset_name)

        path = ReasoningPath(
            id=f"{problem['id']}_path_{path_idx}",
            problem_id=problem["id"],
            question=problem["question"],
            golden_answer=problem["answer"],
            response=response,
            is_correct=is_correct,
            response_tokens=response_tokens,
            response_token_ids=response_token_ids,
            all_position_scores=[],
            total_tokens=len(response_token_ids),
            full_prompt=full_prompt,
            response_token_ranks=ranks,
        )
        paths.append(path)

    return paths


def _cpu_grading_worker(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    event_queue: mp.Queue,
    source_name: str,
    rollout_samples: int,
    wall_timeout: float,
):
    """Run CPU grading in a separate process.

    As a GIL-independent process, it runs truly in parallel with GPU generation.
    Grading results are passed via result_queue; monitor events via event_queue.
    """
    from src.utils.grader import batch_grade_responses_mathverify

    all_scores = {}
    batch_num = 0

    while True:
        item = work_queue.get()
        if item is None:
            break

        event_queue.put(("cpu_grade_start", {"batch": batch_num}))

        batch_keys, all_responses, all_golden_answers, response_to_batch_idx = item

        correctness_results = batch_grade_responses_mathverify(
            all_responses,
            all_golden_answers,
            source_name,
            wall_timeout=wall_timeout,
        )

        position_correct_counts: Dict[int, int] = {}
        for resp_idx, is_correct in enumerate(correctness_results):
            bidx = response_to_batch_idx[resp_idx]
            if bidx not in position_correct_counts:
                position_correct_counts[bidx] = 0
            if is_correct:
                position_correct_counts[bidx] += 1

        for bidx, (path_idx, position) in enumerate(batch_keys):
            correct_count = position_correct_counts.get(bidx, 0)
            all_scores[(path_idx, position)] = correct_count / rollout_samples

        event_queue.put(("cpu_grade_end", {"batch": batch_num}))
        batch_num += 1

    result_queue.put(all_scores)


def _drain_event_queue(event_queue: mp.Queue, monitor):
    """Drain accumulated CPU grading events from event_queue into the monitor."""
    while True:
        try:
            name, meta = event_queue.get_nowait()
            monitor.mark(name, **meta)
        except queue.Empty:
            break


def _run_pipelined_batches(
    llm,
    all_requests: List[RolloutRequest],
    prompt_token_cache: Dict[int, List[int]],
    sampling_params,
    source_name: str,
    rollout_samples: int,
    global_batch_size: int,
    tokenizer,
    wall_timeout: float = 2.0,
    monitor=None,
) -> Dict[Tuple[int, int], float]:
    """Run GPU generation and CPU grading in parallel using a producer-consumer pattern.

    GPU (main process): generates batches and puts extracted text into work_queue.
    CPU (child process): dequeues from work_queue, grades, and returns scores via result_queue.

    Uses multiprocessing: GIL-independent process enables true parallelism.
    work_queue(maxsize=2): GPU can run at most 1 batch ahead. Memory usage bounded.

    Args:
        monitor: Optional PipelineMonitor instance for GPU/CPU utilization tracking.

    Returns:
        all_scores: {(path_idx, position): correctness_score} dict
    """
    work_queue: mp.Queue = mp.Queue(maxsize=2)
    result_queue: mp.Queue = mp.Queue()
    event_queue: mp.Queue = mp.Queue()
    total_batches = (len(all_requests) + global_batch_size - 1) // global_batch_size

    # Start CPU grading in a separate process (GIL-free parallelism)
    consumer = mp.Process(
        target=_cpu_grading_worker,
        args=(work_queue, result_queue, event_queue,
              source_name, rollout_samples, wall_timeout),
    )
    consumer.start()

    # GPU producer (main process)
    try:
        batch_num = 0
        for batch_start in tqdm(range(0, len(all_requests), global_batch_size),
                                total=total_batches, desc="GPU batches"):
            if monitor:
                monitor.mark("gpu_batch_start", batch=batch_num)

            batch_end = min(batch_start + global_batch_size, len(all_requests))
            batch_requests = all_requests[batch_start:batch_end]

            prompts = [{"prompt_token_ids": req.prompt_token_ids}
                       for req in batch_requests]
            outputs = llm.generate(prompts, sampling_params)

            if monitor:
                monitor.mark("gpu_batch_end", batch=batch_num)

            # Extract serializable data for child process
            batch_keys: List[Tuple[int, int]] = []
            all_responses: List[str] = []
            all_golden_answers: List[List[str]] = []
            response_to_batch_idx: List[int] = []

            for batch_idx, (req, output) in enumerate(zip(batch_requests, outputs)):
                prompt_len = len(prompt_token_cache[req.path_idx])
                prefix_token_ids = req.prompt_token_ids[prompt_len:]
                prefix_text = tokenizer.decode(prefix_token_ids, skip_special_tokens=True)
                batch_keys.append((req.path_idx, req.position))

                for sample_output in output.outputs:
                    all_responses.append(prefix_text + sample_output.text)
                    all_golden_answers.append(req.golden_answers)
                    response_to_batch_idx.append(batch_idx)

            work_queue.put((batch_keys, all_responses, all_golden_answers, response_to_batch_idx))

            # Drain monitor events from child process
            if monitor:
                _drain_event_queue(event_queue, monitor)

            batch_num += 1
    finally:
        work_queue.put(None)  # sentinel

    # Wait for CPU grading to finish
    all_scores = result_queue.get()
    consumer.join()

    # Drain remaining monitor events
    if monitor:
        _drain_event_queue(event_queue, monitor)

    return all_scores


def compute_position_scores_optimized(
    llm: LLM,
    tokenizer,
    paths: List[ReasoningPath],
    dataset_name: str,
    rollout_samples: int,
    sampling_config: SamplingConfig,
    global_batch_size: int = 512,
    rollout_max_tokens: int = None,
    max_grading_workers: int = 8,
    memory_limit_requests: int = 50000,
    rollout_window: int = 1,
    monitor=None,
    early_termination_k: int = 0,
    checkpoint_path: str = None,
    lora_request=None,
) -> List[ReasoningPath]:
    """
    Compute correctness scores via path-sequential rollout with early termination.

    Key optimizations:
    1. Processes paths sequentially for prefix cache reuse
    2. Positions processed in forward order within each path
    3. Early termination: stops a path after K consecutive score=0.0 positions
    4. Synchronous grading (CPU grading ≈ 0s, no pipeline overhead)
    5. Per-path checkpoint: saves each completed path to JSONL for crash recovery

    Args:
        llm: vLLM instance
        tokenizer: Tokenizer instance
        paths: List of ReasoningPath objects
        dataset_name: Dataset name for grading
        rollout_samples: Number of samples per position
        sampling_config: Sampling configuration
        global_batch_size: Number of unique prefixes per batch
        rollout_max_tokens: Max tokens for rollout generation
        max_grading_workers: Parallel workers for grading
        memory_limit_requests: Max requests before chunking (unused, kept for API compat)
        rollout_window: Compute scores every N tokens (1 = every token)
        monitor: Optional PipelineMonitor instance
        early_termination_k: Stop path after K consecutive score=0.0. 0 = disabled.

    Returns:
        Updated paths with all_position_scores filled
    """
    source_name = config.get_dataset_source_name(dataset_name)
    mode_str = "thinking" if sampling_config.enable_thinking else "non-thinking"
    mode_key = "thinking" if sampling_config.enable_thinking else "non_thinking"
    actual_rollout_tokens = rollout_max_tokens if rollout_max_tokens else config.get_rollout_max_tokens(dataset_name, mode_key)

    sampling_params = SamplingParams(
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        top_k=sampling_config.top_k,
        presence_penalty=sampling_config.presence_penalty,
        repetition_penalty=sampling_config.repetition_penalty,
        max_tokens=actual_rollout_tokens,
        n=rollout_samples,
        stop=config.STOP_TOKENS,
    )

    window_str = f", window={rollout_window}" if rollout_window > 1 else ""
    et_str = f", early_term_k={early_termination_k}" if early_termination_k > 0 else ""
    print(f"Computing tokenwise potential ({rollout_samples} rollouts{window_str}{et_str}, {mode_str} mode)...")

    total_positions = sum(
        max(0, (len(p.response_token_ids) - 1 + rollout_window - 1) // rollout_window)
        for p in paths if p.total_tokens >= 2
    )
    print(f"  Total paths: {len(paths)}, Total positions (before early term): {total_positions:,}")

    if total_positions == 0:
        print("  No requests to process.")
        return paths

    # Load checkpoint: skip already-completed paths
    completed_ids: set = set()
    if checkpoint_path:
        import os
        if os.path.exists(checkpoint_path):
            path_by_id = {p.id: p for p in paths}
            with open(checkpoint_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        pid = obj["id"]
                        completed_ids.add(pid)
                        if pid in path_by_id:
                            path_by_id[pid].all_position_scores = obj.get("all_position_scores", [])
                    except (json.JSONDecodeError, KeyError):
                        continue
            if completed_ids:
                print(f"  Checkpoint: {len(completed_ids)} paths already completed, resuming...")

    # Process each path sequentially (path-sequential for prefix cache reuse)
    positions_computed = 0
    positions_skipped = 0
    batch_num = 0

    paths_pbar = tqdm(total=len(paths), desc="Paths", initial=len(completed_ids))
    for path_idx, path in enumerate(paths):
        if path.id in completed_ids:
            continue

        if path.total_tokens < 2:
            path.all_position_scores = []
            continue

        # Encode prompt once for this path
        prompt_token_ids = tokenizer.encode(path.full_prompt, add_special_tokens=False)

        # Build requests for this path (forward order)
        requests = []
        for pos in range(1, len(path.response_token_ids), rollout_window):
            prefix_ids = path.response_token_ids[:pos]
            full_ids = prompt_token_ids + prefix_ids
            requests.append(RolloutRequest(
                path_idx=path_idx,
                position=pos,
                prompt_token_ids=full_ids,
                golden_answers=path.golden_answer,
            ))

        scores_dict: Dict[int, float] = {}  # position → score
        consecutive_zeros = 0
        terminated = False

        # Process in batches within this path
        for batch_start in range(0, len(requests), global_batch_size):
            batch = requests[batch_start:batch_start + global_batch_size]

            if monitor:
                monitor.mark("gpu_batch_start", batch=batch_num, path=path_idx)

            # GPU: generate rollout samples
            prompts = [{"prompt_token_ids": req.prompt_token_ids} for req in batch]
            _gen_kwargs = {"lora_request": lora_request} if lora_request else {}
            outputs = llm.generate(prompts, sampling_params, **_gen_kwargs)

            if monitor:
                monitor.mark("gpu_batch_end", batch=batch_num, path=path_idx)

            # Build responses for batch grading
            all_responses: List[str] = []
            all_golden_answers: List[List[str]] = []
            response_to_batch_idx: List[int] = []

            for batch_idx, (req, output) in enumerate(zip(batch, outputs)):
                prefix_token_ids = req.prompt_token_ids[len(prompt_token_ids):]
                prefix_text = tokenizer.decode(prefix_token_ids, skip_special_tokens=True)

                for sample_output in output.outputs:
                    all_responses.append(prefix_text + sample_output.text)
                    all_golden_answers.append(req.golden_answers)
                    response_to_batch_idx.append(batch_idx)

            if monitor:
                monitor.mark("cpu_grade_start", batch=batch_num, path=path_idx)

            # CPU: synchronous batch grading (≈ 0s for GSM/AIME, fast for MATH)
            correctness_results = batch_grade_responses_mathverify(
                all_responses, all_golden_answers, source_name, wall_timeout=2.0,
            )

            if monitor:
                monitor.mark("cpu_grade_end", batch=batch_num, path=path_idx)

            # Compute scores and check early termination
            position_correct_counts: Dict[int, int] = {}
            for resp_idx, is_correct in enumerate(correctness_results):
                bidx = response_to_batch_idx[resp_idx]
                if bidx not in position_correct_counts:
                    position_correct_counts[bidx] = 0
                if is_correct:
                    position_correct_counts[bidx] += 1

            for batch_idx, req in enumerate(batch):
                correct_count = position_correct_counts.get(batch_idx, 0)
                score = correct_count / rollout_samples
                scores_dict[req.position] = score
                positions_computed += 1

                if early_termination_k > 0:
                    if score == 0.0:
                        consecutive_zeros += 1
                        if consecutive_zeros >= early_termination_k:
                            terminated = True
                            break
                    else:
                        consecutive_zeros = 0

            batch_num += 1

            if terminated:
                remaining = len(requests) - (batch_start + len(batch))
                positions_skipped += remaining
                break

        # Reconstruct position scores (fill terminated positions with 0.0)
        path.all_position_scores = []
        for pos in range(1, len(path.response_token_ids)):
            path.all_position_scores.append(scores_dict.get(pos, 0.0))

        if terminated:
            print(f"  Path {path.id}: early terminated at position "
                  f"{max(scores_dict.keys())}/{len(path.response_token_ids) - 1} "
                  f"(K={early_termination_k} consecutive zeros)")

        # Checkpoint: append completed path to JSONL
        if checkpoint_path:
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps(asdict(path), ensure_ascii=False) + "\n")

        paths_pbar.update(1)

    paths_pbar.close()
    print(f"  Computed: {positions_computed:,}, Skipped (early term): {positions_skipped:,}")
    if positions_computed + positions_skipped > 0:
        print(f"  Savings: {positions_skipped / (positions_computed + positions_skipped) * 100:.1f}%")

    return paths


def _compute_scores_chunked(
    llm: LLM,
    tokenizer,
    paths: List[ReasoningPath],
    source_name: str,
    rollout_samples: int,
    sampling_config: SamplingConfig,
    global_batch_size: int,
    rollout_max_tokens: int,
    max_grading_workers: int,
    memory_limit_requests: int,
    rollout_window: int = 1,
    monitor=None,
) -> List[ReasoningPath]:
    """Process paths in chunks when total requests exceed memory limit."""
    total_tokens = sum(p.total_tokens for p in paths)
    avg_tokens = total_tokens / len(paths) if paths else 1
    paths_per_chunk = max(1, int(memory_limit_requests / avg_tokens))

    print(f"  Processing {len(paths)} paths in chunks of ~{paths_per_chunk}")

    for chunk_start in range(0, len(paths), paths_per_chunk):
        chunk_end = min(chunk_start + paths_per_chunk, len(paths))
        chunk_paths = paths[chunk_start:chunk_end]

        print(f"  Chunk: paths {chunk_start}-{chunk_end}")

        _compute_chunk_scores(
            llm, tokenizer, chunk_paths, source_name,
            rollout_samples, sampling_config, global_batch_size,
            rollout_max_tokens, max_grading_workers, rollout_window,
            monitor=monitor,
        )

    return paths


def _compute_chunk_scores(
    llm: LLM,
    tokenizer,
    paths: List[ReasoningPath],
    source_name: str,
    rollout_samples: int,
    sampling_config: SamplingConfig,
    global_batch_size: int,
    rollout_max_tokens: int,
    max_grading_workers: int,
    rollout_window: int = 1,
    monitor=None,
) -> None:
    """Process a chunk of paths (modifies paths in place)."""
    sampling_params = SamplingParams(
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        top_k=sampling_config.top_k,
        presence_penalty=sampling_config.presence_penalty,
        repetition_penalty=sampling_config.repetition_penalty,
        max_tokens=rollout_max_tokens,
        n=rollout_samples,
        stop=config.STOP_TOKENS,
    )

    all_requests, _, prompt_token_cache = collect_rollout_requests(tokenizer, paths, rollout_window)
    if not all_requests:
        return

    all_scores = _run_pipelined_batches(
        llm=llm,
        all_requests=all_requests,
        prompt_token_cache=prompt_token_cache,
        sampling_params=sampling_params,
        source_name=source_name,
        rollout_samples=rollout_samples,
        global_batch_size=global_batch_size,
        tokenizer=tokenizer,
        monitor=monitor,
    )

    for path_idx, path in enumerate(paths):
        if path.total_tokens < 2:
            path.all_position_scores = []
            continue
        path.all_position_scores = [
            all_scores.get((path_idx, pos), None)
            for pos in range(1, len(path.response_token_ids))
        ]


def compute_position_scores(
    llm: LLM,
    tokenizer,
    paths: List[ReasoningPath],
    dataset_name: str,
    rollout_samples: int,
    sampling_config: SamplingConfig,
    rollout_max_tokens: int = None,
    global_batch_size: int = 512,
    max_grading_workers: int = 8,
    rollout_window: int = 1,
    monitor=None,
    early_termination_k: int = 0,
    checkpoint_path: str = None,
    lora_request=None,
) -> List[ReasoningPath]:
    """
    Compute correctness scores for each token position via rollout sampling.

    Args:
        llm: vLLM instance
        tokenizer: Tokenizer instance
        paths: List of ReasoningPath objects
        dataset_name: Dataset name for grading
        rollout_samples: Number of rollout samples per position
        sampling_config: Sampling configuration
        rollout_max_tokens: Max tokens for rollout generation
        global_batch_size: Batch size for optimized processing
        max_grading_workers: Workers for parallel grading
        rollout_window: Compute scores every N tokens (1 = every token, token-wise)
        monitor: Optional PipelineMonitor for GPU/CPU utilization tracking
        early_termination_k: Stop path after K consecutive score=0.0. 0 = disabled.
        checkpoint_path: Path to JSONL checkpoint file for crash recovery.

    Returns:
        Updated paths with all_position_scores filled
    """
    source_name = config.get_dataset_source_name(dataset_name)
    mode_key = "thinking" if sampling_config.enable_thinking else "non_thinking"
    actual_rollout_tokens = rollout_max_tokens if rollout_max_tokens else config.get_rollout_max_tokens(dataset_name, mode_key)

    return compute_position_scores_optimized(
        llm=llm,
        tokenizer=tokenizer,
        paths=paths,
        dataset_name=source_name,
        rollout_samples=rollout_samples,
        sampling_config=sampling_config,
        global_batch_size=global_batch_size,
        rollout_max_tokens=actual_rollout_tokens,
        max_grading_workers=max_grading_workers,
        rollout_window=rollout_window,
        monitor=monitor,
        early_termination_k=early_termination_k,
        checkpoint_path=checkpoint_path,
        lora_request=lora_request,
    )


def split_success_failure(paths: List[ReasoningPath]) -> Dict[str, List[ReasoningPath]]:
    """Split paths into success and failure groups."""
    success = [p for p in paths if p.is_correct]
    failure = [p for p in paths if not p.is_correct]

    return {
        "success": success,
        "failure": failure,
    }


def paths_to_dicts(paths: List[ReasoningPath]) -> List[Dict]:
    """Convert ReasoningPath objects to dictionaries."""
    return [asdict(p) for p in paths]
