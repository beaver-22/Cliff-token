"""
Math Answer Grading Utilities

=============================================================================
Grading Pipeline Architecture
=============================================================================

A 2-phase pipeline to extract answers from model outputs (long CoT + answer)
and compare them against the gold answer.

  batch_grade_responses_mathverify()
              │
              ▼
  Phase 1 — Light Grade  (_light_grade)
  ┌────────────────────────────────────────────────────────────┐
  │ Extract answer from \boxed{} or "the answer is X",        │
  │ then immediately judge by int() comparison if both sides  │
  │ are integers. No thread/SymPy/math-verify — runs in μs.   │
  │                                                            │
  │ AIME: 100%, GSM1K: ~99%, MATH500: ~63% complete here.     │
  │                                                            │
  │ Returns:                                                   │
  │   True/False → definitive result (Phase 2 skipped)        │
  │   None       → undecidable (non-integer) → pass to Phase 2│
  └────────────────────────────────────────────────────────────┘
              │ None only
              ▼
  Phase 2 — math-verify  (HuggingFace, github.com/huggingface/Math-Verify)
  ┌────────────────────────────────────────────────────────────┐
  │ Stage 2a: Pre-extraction  (_extract_for_grading)           │
  │   1. last \boxed{} slicing (wrapper preserved)             │
  │   2. "the answer is X" pattern                             │
  │   3. full response fallback                                │
  │                                                            │
  │ Stage 2b: math-verify parse + verify                       │
  │   parse(candidate, [LatexExtractionConfig,                 │
  │                      ExprExtractionConfig])                 │
  │   → ANTLR4 grammar → SymPy object conversion              │
  │   verify(gold_parsed, pred_parsed)                         │
  │   → string equality / numeric precision / symbolic (SymPy) │
  │                                                            │
  │ Executed in parallel via ThreadPoolExecutor +              │
  │ _grade_with_timeout_guard inside _GradingPool              │
  │ (separate child process), so zombie threads do not         │
  │ affect the main process GIL.                               │
  └────────────────────────────────────────────────────────────┘

Why is Pre-extraction necessary?
  - math-verify's default extraction_mode='any_match' stops at the first
    successful parse point.
  - In long CoT responses, an intermediate \boxed{} for a sub-calculation
    may be found before the final answer.
  - We use rfind('\boxed') to explicitly select the last \boxed{}.
  - Stripping the wrapper (extract_boxed_answer) and passing the content
    can cause math-verify parse failures for anchor-less LaTeX
    (e.g. \sqrt{4} standalone).
  → Slicing with the wrapper preserved is optimal for both accuracy and speed.

Threading constraints in rollout grading:
  - math-verify's timeout is based on signal.SIGALRM → only safe on the main thread.
  - In background threads, use parsing_timeout=None, timeout_seconds=None.
  - _grade_with_timeout_guard() provides a wall-clock guard via daemon thread + join.
  - SIGALRM may not fire inside a tight loop in a C extension.
  → The thread join approach is reliable even for C extension hangs.

=============================================================================
"""
import io
import atexit
import queue
import threading
import contextlib
import multiprocessing as mp
from functools import lru_cache
from typing import List, Optional

from math_verify import verify as mv_verify, parse as mv_parse
from math_verify import LatexExtractionConfig, ExprExtractionConfig


# Extraction config covers both LaTeX (\boxed{}) and plain expressions ("42")
_MV_EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]


def _extract_for_grading(response: str) -> str:
    """Stage 1: Extract a candidate string from model output for grading.

    Strategy (in priority order):
    1. Last \\boxed{} slicing
       response.rfind('\\boxed') → response[idx:]
       ★ Pass the string with the \\boxed{} wrapper preserved.
         - Extracting only the content (like extract_boxed_answer()) can cause
           math-verify parse failures for anchor-less LaTeX (e.g. \\sqrt{4} standalone).
         - Preserving the wrapper allows LatexExtractionConfig to work correctly.
         - Using rfind selects the last \\boxed, preventing misidentification of
           intermediate sub-calculation results in CoT.

    2. "the answer is X" pattern
       Handles GSM1K few-shot responses that do not contain \\boxed{}.

    3. Full response fallback
       If all else fails, return the full response (math-verify retries internally).
    """
    # Strategy 1: last \boxed{} — wrapper preserved
    idx = response.rfind(r'\boxed')
    if idx != -1:
        return response[idx:]

    # Strategy 2: "the answer is X" pattern
    if 'he answer is' in response:
        return response.split('he answer is')[-1].strip()

    # Strategy 3: fallback
    return response


@lru_cache(maxsize=2048)
def _parse_gold_cached(gold_text: str):
    """Cache the parsed result of a gold answer.

    Prevents repeated parsing since 64 rollout samples share the same gold answer.
    Thread-based 3-second timeout: safely interrupts C extension hangs without SIGALRM.
    """
    result = [None]

    def _worker():
        try:
            # Gold answers in datasets (MATH-500, etc.) are stored as raw LaTeX
            # without \boxed{}/$...$ anchors. Wrapping in $...$ is required for
            # LatexExtractionConfig to parse correctly.
            wrapped = f"${gold_text}$"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result[0] = mv_parse(wrapped, extraction_config=_MV_EXTRACTION_CONFIG,
                                     parsing_timeout=None)
        except Exception:
            result[0] = []

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=3.0)
    return result[0] if result[0] is not None else []


def grade_with_mathverify(
    response: str,
    golden_answers: List[str],
    dataset_name: str,
    timeout: int = 5,
) -> bool:
    """Grade a single response using math-verify.

    Args:
        timeout: parsing/verify timeout in seconds.
            > 0: uses signal.SIGALRM → only safe on the main thread.
            0 or None: disables signal → safe for background threads.
            Must use timeout=0 when calling from a background thread.
    """
    parsing_timeout = None if not timeout else timeout
    verify_timeout  = None if not timeout else timeout

    try:
        candidate = _extract_for_grading(response)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pred_parsed = mv_parse(
                candidate,
                extraction_config=_MV_EXTRACTION_CONFIG,
                parsing_timeout=parsing_timeout,
            )
        if not pred_parsed:
            return False

        for gold_text in golden_answers:
            gold_parsed = _parse_gold_cached(gold_text)
            if not gold_parsed:
                continue
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if mv_verify(gold_parsed, pred_parsed, timeout_seconds=verify_timeout):
                    return True
        return False
    except Exception:
        return False


def _grade_with_timeout_guard(
    response: str,
    golden_answers: List[str],
    dataset_name: str,
    wall_timeout: float = 2.0,
) -> bool:
    """Thread-based wall-clock timeout guard.

    Runs grade_with_mathverify(timeout=0) as a daemon thread and joins
    after wall_timeout seconds. Safe for background threads without signal.SIGALRM.
    If the timeout expires, the thread continues in the background, but since
    daemon=True it is automatically cleaned up when the interpreter exits.
    """
    result = [False]

    def _worker():
        result[0] = grade_with_mathverify(
            response, golden_answers, dataset_name, timeout=0
        )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(wall_timeout)
    return result[0]


def _batch_grade_phase2(
    responses: List[str],
    golden_answers_list: List[List[str]],
    dataset_name: str,
    wall_timeout: float = 2.0,
    max_workers: int = 32,
) -> List[bool]:
    """Phase 2 grading: ThreadPoolExecutor + _grade_with_timeout_guard.

    This function runs inside a child process.
    Accumulated zombie threads do not affect the main process GIL.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [False] * len(responses)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _grade_with_timeout_guard,
                responses[i], golden_answers_list[i], dataset_name, wall_timeout
            ): i
            for i in range(len(responses))
        }
        for future in as_completed(futures, timeout=300.0):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    return results


def _grading_pool_worker(work_queue, result_queue):
    """Child process worker: processes Phase 2 grading in a loop.

    Runs the same ThreadPoolExecutor logic, but in a separate process so
    zombie threads do not affect the main process GIL.
    """
    while True:
        item = work_queue.get()
        if item is None:
            break
        responses, golden_answers_list, dataset_name, wall_timeout, max_workers = item
        try:
            results = _batch_grade_phase2(
                responses, golden_answers_list, dataset_name, wall_timeout, max_workers
            )
        except Exception:
            results = [False] * len(responses)
        result_queue.put(results)


class _GradingPool:
    """Singleton pool that runs Phase 2 grading in a separate process.

    Zombie threads are isolated inside the child process and do not affect the main process GIL.
    If the child hangs, it is immediately killed with SIGKILL and a new worker is spawned.
    Uses the spawn context to prevent CUDA context from being copied.
    """

    def __init__(self):
        self._mp_ctx = mp.get_context("spawn")
        self._work_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._worker: Optional[mp.Process] = None
        self._ensure_worker()

    def _ensure_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return
        if self._worker is not None:
            self._worker.kill()
            self._worker.join(timeout=5.0)
        self._work_queue = self._mp_ctx.Queue()
        self._result_queue = self._mp_ctx.Queue()
        self._worker = self._mp_ctx.Process(
            target=_grading_pool_worker,
            args=(self._work_queue, self._result_queue),
            daemon=True,
        )
        self._worker.start()

    def grade_batch(
        self,
        responses: List[str],
        golden_answers_list: List[List[str]],
        dataset_name: str,
        wall_timeout: float = 2.0,
        max_workers: int = 32,
        batch_timeout: float = 300.0,
    ) -> List[bool]:
        self._ensure_worker()
        self._work_queue.put((
            responses, golden_answers_list, dataset_name, wall_timeout, max_workers
        ))
        try:
            return self._result_queue.get(timeout=batch_timeout)
        except queue.Empty:
            self._kill_and_restart()
            return [False] * len(responses)

    def _kill_and_restart(self):
        if self._worker is not None:
            self._worker.kill()
            self._worker.join(timeout=5.0)
        self._worker = None
        self._ensure_worker()

    def shutdown(self):
        if self._worker is not None and self._worker.is_alive():
            self._work_queue.put(None)
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                self._worker.kill()
                self._worker.join(timeout=5.0)
        self._worker = None


_grading_pool: Optional[_GradingPool] = None


def _get_grading_pool() -> _GradingPool:
    global _grading_pool
    if _grading_pool is None:
        _grading_pool = _GradingPool()
    return _grading_pool


def shutdown_grading_pool():
    global _grading_pool
    if _grading_pool is not None:
        _grading_pool.shutdown()
        _grading_pool = None


atexit.register(shutdown_grading_pool)


def _light_grade(response: str, golden_answers: List[str]):
    """Lightweight grader: extract integer from \\boxed{} and compare.

    Grades in μs without math-verify/SymPy/thread for
    AIME (100% integer), GSM1K (~99% integer), and MATH500 (63% integer).

    Returns:
        True/False: judgment complete via integer comparison
        None: undecidable (non-integer) → math-verify fallback required
    """
    # 1. Extract contents of last \boxed{}
    idx = response.rfind(r'\boxed')
    if idx == -1:
        # "the answer is X" pattern (GSM1K)
        if 'he answer is' not in response:
            return None
        tail = response.split('he answer is')[-1]
        pred = tail.strip().rstrip('.').strip()
    else:
        after = response[idx + len(r'\boxed'):]
        if not after or after[0] != '{':
            return None
        stack = 1
        chars = []
        for c in after[1:]:
            if c == '{':
                stack += 1
            elif c == '}':
                stack -= 1
                if stack == 0:
                    break
            chars.append(c)
        else:
            return None  # unmatched brace
        pred = ''.join(chars).strip()

    # 2. Attempt integer comparison
    #    Normalize common patterns found in GSM1K model outputs: $, %, commas, .00, etc.
    pred = pred.replace(',', '').replace('\\,', '')
    pred = pred.strip('$').strip('\\$').strip()
    pred = pred.rstrip('%').rstrip('\\%').strip()
    if pred.endswith('.00'):
        pred = pred[:-3]
    elif pred.endswith('.0'):
        pred = pred[:-2]

    try:
        pred_int = int(pred)
    except (ValueError, OverflowError):
        return None  # pred is non-integer → math-verify fallback

    any_gold_is_int = False
    for gold in golden_answers:
        try:
            gold_int = int(gold.strip())
            any_gold_is_int = True
            if pred_int == gold_int:
                return True
        except (ValueError, OverflowError):
            continue

    # If any gold was an integer but none matched → definitive False
    # If no gold was an integer → fallback (gold may be in a form like "x=5")
    return False if any_gold_is_int else None


def _is_light_only(dataset_name: str) -> bool:
    """GSM1K and AIME families have 100% integer answers, so only the light grader is used."""
    key = dataset_name.lower()
    return key.startswith("gsm") or key.startswith("aime")


def batch_grade_responses_mathverify(
    responses: List[str],
    golden_answers_list: List[List[str]],
    dataset_name: str,
    wall_timeout: float = 2.0,
    max_workers: int = 32,
    chunk_timeout: float = 300.0,
) -> List[bool]:
    """Batch grading.

    GSM1K/AIME: uses only _light_grade(). Undecidable (None) results are treated as False.
                No math-verify calls.
    MATH (and others): runs _light_grade() first, then math-verify fallback for undecidable items only.
    """
    if len(responses) != len(golden_answers_list):
        raise ValueError("Responses and golden_answers must have same length")

    if not responses:
        return []

    light_only = _is_light_only(dataset_name)

    results = [False] * len(responses)
    slow_indices = []

    # Phase 1: light grading (integer answers — processed immediately without thread/SymPy)
    for i, (resp, golds) in enumerate(zip(responses, golden_answers_list)):
        fast = _light_grade(resp, golds)
        if fast is not None:
            results[i] = fast
        elif not light_only:
            slow_indices.append(i)
        # if light_only, None → False (keep default value)

    # Phase 2: math-verify fallback (only for datasets with many non-integer answers, e.g. MATH)
    # Run in a separate child process to isolate zombie threads from the main process GIL.
    if slow_indices:
        pool = _get_grading_pool()
        slow_responses = [responses[i] for i in slow_indices]
        slow_golds = [golden_answers_list[i] for i in slow_indices]
        phase2_results = pool.grade_batch(
            slow_responses, slow_golds, dataset_name,
            wall_timeout, max_workers, chunk_timeout,
        )
        for j, i in enumerate(slow_indices):
            results[i] = phase2_results[j]

    return results
