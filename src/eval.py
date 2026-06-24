#!/usr/bin/env python3
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _kst_timestamp() -> str:
    """Return the current time in KST (UTC+9) formatted as MMDD_HHMMSS."""
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(kst)
    return now.strftime("%m%d_%H%M%S")
"""
Multi-GPU Parallel Accuracy Evaluation for Math Reasoning Benchmarks

Loads one model per GPU and evaluates all (model × dataset × temperature)
combinations, then outputs results as an accuracy matrix.

Usage:
  python src/eval.py                                          # full evaluation
  python src/eval.py --model qwen3-4b --dataset gsm1k        # specific model/dataset
  python src/eval.py --model qwen3-4b qwen3-8b --gpus 0 1    # specify GPUs
  python src/eval.py --temperature 0.6                        # single temperature
  python src/eval.py --thinking --model qwen3-4b             # Qwen3 thinking mode
  python src/eval.py --num_problems 50 --force               # 50 problems, ignore cache
"""

import os
import sys
import json
import argparse
import csv
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Safe (no CUDA) imports only at module level
from src.config import (
    MODEL_ALIASES, MODEL_CONFIGS, PAPER_MODEL_ALIASES,
    get_model_short_name, resolve_model_path,
    GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_EVAL_MODELS     = list(PAPER_MODEL_ALIASES)
DEFAULT_EVAL_DATASETS   = ["gsm1k_100", "math500_100", "aime25"]
DEFAULT_TEMPERATURES    = [0.0, 0.6, 1.0]

DATASET_DISPLAY = {
    "gsm1k":     "GSM1K",
    "gsm1k_100": "GSM1K-100",
    "math500":   "MATH-500",
    "math500_100": "MATH-100",
    "aime24":    "AIME-2024",
    "aime25":  "AIME-2025",
}


# =============================================================================
# Helpers
# =============================================================================

def resolve_mode(model_path: str, thinking: bool) -> str:
    """Return 'thinking' or 'non_thinking'. Falls back safely for Gemma3."""
    model_cfg = MODEL_CONFIGS.get(model_path, {})
    modes = [k for k in model_cfg if k != "name"]
    if thinking and "thinking" in modes:
        return "thinking"
    if "non_thinking" in modes:
        return "non_thinking"
    return modes[0] if modes else "non_thinking"


def get_cache_path(output_dir: str, model_alias: str,
                   dataset: str, temperature: float, prompt_type: str) -> Path:
    return Path(output_dir) / model_alias / f"{dataset}_temp{temperature:.1f}_{prompt_type}.json"


def load_cached(cache_path: Path) -> Optional[dict]:
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
    except Exception:
        pass
    return None


def save_cached(cache_path: Path, result: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


# =============================================================================
# Worker (runs in spawned subprocess — CUDA_VISIBLE_DEVICES set as first line)
# =============================================================================

def eval_worker(
    gpu_id: int,
    model_alias: str,
    model_path: str,
    datasets: List[str],
    temperatures: List[float],
    num_problems: int,
    paths_per_problem: int,
    mode: str,
    prompt_type: str,
    output_dir: str,
    force: bool,
    result_dict,   # multiprocessing.Manager().dict()
    error_dict,
) -> None:
    # ── MUST be the very first action: isolate GPU before any CUDA import ──
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Deferred heavy imports (vLLM, transformers, generator)
    from vllm import LLM
    from transformers import AutoTokenizer
    from src.analysis.generator import generate_reasoning_paths, sample_problems, load_jsonl
    from src.config import (
        get_sampling_config_with_temperature,
        get_max_tokens,
        get_dataset_path,
    )

    print(f"[{model_alias}] Loading model on GPU {gpu_id} (mode={mode}) ...")
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        msg = f"Model load failed: {exc}"
        print(f"[{model_alias}] ERROR: {msg}")
        error_dict[model_alias] = msg
        return

    print(f"[{model_alias}] Model loaded. Starting evaluation.")

    for dataset in datasets:
        dataset_path = get_dataset_path(dataset)
        max_new_tokens = get_max_tokens(dataset, mode)

        # Resolve num_problems=-1 → use all
        try:
            all_probs = load_jsonl(dataset_path)
        except FileNotFoundError:
            msg = f"Dataset file not found: {dataset_path}"
            print(f"[{model_alias}/{dataset}] ERROR: {msg}")
            for temp in temperatures:
                result_dict[(model_alias, dataset, temp)] = {
                    "error": msg, "accuracy": None,
                    "model_alias": model_alias, "dataset": dataset,
                    "temperature": temp, "mode": mode,
                }
            continue

        n = len(all_probs) if num_problems == -1 else num_problems
        problems = sample_problems(dataset_path, n)

        for temperature in temperatures:
            key = (model_alias, dataset, temperature)
            cache_path = get_cache_path(output_dir, model_alias, dataset, temperature, prompt_type)

            # Cache hit
            if not force:
                cached = load_cached(cache_path)
                if cached is not None:
                    print(f"[{model_alias}] Cache hit: {dataset} T={temperature}")
                    result_dict[key] = cached
                    continue

            print(f"[{model_alias}] Evaluating {dataset} | T={temperature} | prompt={prompt_type} | {len(problems)} problems ...")
            try:
                from dataclasses import replace as dc_replace
                sampling_config = get_sampling_config_with_temperature(
                    mode=mode,
                    model_path=model_path,
                    temperature=temperature,
                )
                # Override prompt_type from CLI
                sampling_config = dc_replace(sampling_config, prompt_type=prompt_type)

                paths = generate_reasoning_paths(
                    llm=llm,
                    tokenizer=tokenizer,
                    problems=problems,
                    paths_per_problem=paths_per_problem,
                    dataset_name=dataset,
                    sampling_config=sampling_config,
                    max_new_tokens=max_new_tokens,
                )

                n_total   = len(paths)
                n_correct = sum(p.is_correct for p in paths)
                accuracy  = n_correct / n_total if n_total > 0 else 0.0

                result = {
                    "accuracy":     accuracy,
                    "n_correct":    n_correct,
                    "n_total":      n_total,
                    "n_problems":   len(problems),
                    "model_alias":  model_alias,
                    "model_name":   get_model_short_name(model_path),
                    "dataset":      dataset,
                    "temperature":  temperature,
                    "mode":         mode,
                    "prompt_type":  prompt_type,
                    "error":        None,
                }
                save_cached(cache_path, result)
                result_dict[key] = result
                print(f"[{model_alias}] {dataset} T={temperature}: "
                      f"{accuracy * 100:.1f}% ({n_correct}/{n_total})")

                # Save per-problem details for error analysis
                details_path = cache_path.with_suffix('.details.jsonl')
                with open(details_path, 'w') as df:
                    for p in paths:
                        detail = {
                            "id": p.problem_id,
                            "question": p.question[:200],
                            "gold": p.golden_answer,
                            "response_tail": p.response[-500:],
                            "is_correct": p.is_correct,
                            "response_len": p.total_tokens,
                            "has_boxed": r'\boxed' in p.response,
                            "has_think": '<think>' in p.response or '</think>' in p.response,
                        }
                        df.write(json.dumps(detail, ensure_ascii=False) + '\n')
                print(f"[{model_alias}] Saved details: {details_path}")

            except Exception as exc:
                msg = str(exc)
                print(f"[{model_alias}] ERROR on {dataset} T={temperature}: {msg}")
                result_dict[key] = {
                    "error":       msg,
                    "accuracy":    None,
                    "model_alias": model_alias,
                    "model_name":  get_model_short_name(model_path),
                    "dataset":     dataset,
                    "temperature": temperature,
                    "mode":        mode,
                    "prompt_type": prompt_type,
                }

    print(f"[{model_alias}] Done.")


# =============================================================================
# Result aggregation
# =============================================================================

def aggregate(
    result_dict: dict,
    models: List[str],
    datasets: List[str],
    temperatures: List[float],
) -> Dict[float, Dict[str, Dict[str, dict]]]:
    """Reshape flat result_dict into {temp: {model: {dataset: result}}}."""
    out: Dict[float, Dict[str, Dict[str, dict]]] = {}
    for temp in temperatures:
        out[temp] = {}
        for model in models:
            out[temp][model] = {}
            for dataset in datasets:
                key = (model, dataset, temp)
                if key in result_dict:
                    out[temp][model][dataset] = dict(result_dict[key])
    return out


# =============================================================================
# Output: console table
# =============================================================================

def print_table(
    results: Dict[float, Dict[str, Dict[str, dict]]],
    temperature: float,
    models: List[str],
    datasets: List[str],
    mode: str,
    prompt_type: str = "boxed",
) -> None:
    col_w   = 12
    model_w = 18
    ds_headers = [DATASET_DISPLAY.get(d, d).center(col_w) for d in datasets]

    print()
    print(f"=== Accuracy Matrix (T={temperature}, mode={mode}, prompt={prompt_type}) ===")
    print(f"{'Model':<{model_w}}| " + " | ".join(ds_headers))
    print("-" * model_w + "|" + ("".join("-" * (col_w + 2) + "|" for _ in datasets)))

    for alias in models:
        model_name = get_model_short_name(resolve_model_path(alias))
        cells = []
        for dataset in datasets:
            r = results.get(temperature, {}).get(alias, {}).get(dataset)
            if r is None:
                cells.append("N/A".center(col_w))
            elif r.get("error"):
                cells.append("ERROR".center(col_w))
            elif r.get("accuracy") is not None:
                cells.append(f"{r['accuracy'] * 100:6.1f}%".center(col_w))
            else:
                cells.append("N/A".center(col_w))
        print(f"{model_name:<{model_w}}| " + " | ".join(cells))


# =============================================================================
# Output: save files
# =============================================================================

def save_json(results: dict, output_dir: str, temperatures: List[float]) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for temp in temperatures:
        path = out_path / f"results_temp{temp:.1f}.json"
        path.write_text(json.dumps(results.get(temp, {}), indent=2, ensure_ascii=False))
        print(f"Saved: {path}")


def save_csv(
    results: dict,
    output_dir: str,
    models: List[str],
    datasets: List[str],
    temperatures: List[float],
) -> None:
    out_path = Path(output_dir) / "results_summary.csv"
    fieldnames = [
        "model_alias", "model_name", "dataset",
        "temperature", "mode",
        "accuracy", "n_correct", "n_total", "n_problems",
        "error",
    ]
    rows = []
    for temp in temperatures:
        for alias in models:
            for dataset in datasets:
                r = results.get(temp, {}).get(alias, {}).get(dataset)
                if r is None:
                    continue
                rows.append({
                    "model_alias": alias,
                    "model_name":  r.get("model_name", get_model_short_name(resolve_model_path(alias))),
                    "dataset":     dataset,
                    "temperature": temp,
                    "mode":        r.get("mode", ""),
                    "accuracy":    f"{r['accuracy']:.6f}" if r.get("accuracy") is not None else "",
                    "n_correct":   r.get("n_correct", ""),
                    "n_total":     r.get("n_total", ""),
                    "n_problems":  r.get("n_problems", ""),
                    "error":       r.get("error", ""),
                })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel accuracy evaluation for math reasoning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval.py --model qwen3-4b --dataset gsm1k --num_problems 50 --gpus 0
  python src/eval.py --model qwen3-4b qwen3-8b --dataset math500 --temperature 0.6 --gpus 0 1
  python src/eval.py --thinking --model qwen3-4b --temperature 0.6 --gpus 0
  python src/eval.py  # full evaluation: all 7 paper models × datasets × temperatures
        """,
    )
    parser.add_argument(
        "--model", nargs="+",
        choices=list(MODEL_ALIASES.keys()),
        default=DEFAULT_EVAL_MODELS,
        metavar="MODEL",
        help=f"Model alias(es). Choices: {list(MODEL_ALIASES.keys())}. Default: all 7 paper models.",
    )
    parser.add_argument(
        "--dataset", nargs="+",
        choices=DEFAULT_EVAL_DATASETS,
        default=DEFAULT_EVAL_DATASETS,
        metavar="DATASET",
        help="Dataset(s) to evaluate. Default: all.",
    )
    parser.add_argument(
        "--temperature", nargs="+", type=float,
        default=DEFAULT_TEMPERATURES,
        metavar="T",
        help="Temperature value(s). Default: 0 0.6 1.0",
    )
    parser.add_argument(
        "--num_problems", type=int, default=-1,
        metavar="N",
        help="Problems per dataset (-1 = all). Default: -1",
    )
    parser.add_argument(
        "--paths_per_problem", type=int, default=1,
        metavar="K",
        help="Reasoning paths per problem (pass@1 style). Default: 1",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable thinking mode (Qwen3 only; Gemma3 silently uses non_thinking)",
    )
    parser.add_argument(
        "--prompt_type",
        choices=["zeroshot", "fewshot", "direct"],
        default="zeroshot",
        help=(
            "Prompt template to use. Default: zeroshot\n"
            "  zeroshot — 0-shot, CoT instruction in system prompt (unified default)\n"
            "             apply_chat_template is handled automatically for each model\n"
            "  fewshot  — few-shot CoT (MATH_PROMPT / GSM8K_PROMPT)\n"
            "  direct   — few-shot, answer-only (no reasoning)"
        ),
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int,
        default=[0, 1, 2, 3],
        metavar="GPU_ID",
        help="GPU IDs to use. Model i → gpus[i %% len(gpus)]. Default: 0 1 2 3",
    )
    parser.add_argument(
        "--output_dir", type=str, default=f"./results/model_eval_{_kst_timestamp()}",
        help="Output directory for results. Default: ./results/model_eval_MMDD_HHMMSS (KST)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if cached results exist",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Must set spawn method before any multiprocessing usage
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()

    # Resolve model aliases → (alias, hf_path) pairs
    model_pairs: List[Tuple[str, str]] = [
        (alias, resolve_model_path(alias)) for alias in args.model
    ]

    # Warn if temperature=0 with Qwen3 (official docs advise against greedy)
    if 0.0 in args.temperature and any("qwen3" in m for m in args.model):
        print("[WARNING] Qwen3 official docs advise against greedy decoding (T=0). "
              "Results may contain repetitions or degraded quality.")

    # Warn if GPU cycling will share GPUs
    if len(model_pairs) > len(args.gpus):
        print(f"[WARNING] {len(model_pairs)} models but only {len(args.gpus)} GPUs. "
              f"Multiple models will share GPUs — risk of OOM.")

    print(f"\nEvaluation plan:")
    print(f"  Models     : {[a for a, _ in model_pairs]}")
    print(f"  Datasets   : {args.dataset}")
    print(f"  Temperatures: {args.temperature}")
    print(f"  Mode       : {'thinking (Qwen3) / non_thinking (others)' if args.thinking else 'non_thinking'}")
    print(f"  Prompt type: {args.prompt_type}")
    print(f"  Num problems: {'all' if args.num_problems == -1 else args.num_problems}")
    print(f"  Paths/problem: {args.paths_per_problem}")
    print(f"  GPUs       : {args.gpus}")
    print(f"  Output dir : {args.output_dir}")
    print()

    # Shared result stores
    manager     = multiprocessing.Manager()
    result_dict = manager.dict()
    error_dict  = manager.dict()

    # Launch one process per model
    processes = []
    for i, (alias, model_path) in enumerate(model_pairs):
        gpu_id = args.gpus[i % len(args.gpus)]
        mode   = resolve_mode(model_path, args.thinking)

        if args.thinking and mode == "non_thinking" and "gemma" in alias:
            print(f"[INFO] {alias} does not support thinking mode — using non_thinking.")

        p = multiprocessing.Process(
            target=eval_worker,
            args=(
                gpu_id, alias, model_path,
                args.dataset, args.temperature,
                args.num_problems, args.paths_per_problem,
                mode, args.prompt_type, args.output_dir, args.force,
                result_dict, error_dict,
            ),
            name=f"worker-{alias}",
            daemon=False,
        )
        p.start()
        print(f"Started worker: {alias} on GPU {gpu_id}")
        processes.append((p, alias))

    # Wait for all workers
    print()
    any_failure = False
    for p, alias in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[WARNING] Worker '{alias}' exited with code {p.exitcode}")
            any_failure = True

    # Report worker-level errors
    if error_dict:
        print("\nWorker errors:")
        for alias, msg in error_dict.items():
            print(f"  {alias}: {msg}")

    # Aggregate and display results
    results = aggregate(dict(result_dict), args.model, args.dataset, args.temperature)

    # Determine display mode label
    mode_label = "thinking" if args.thinking else "non_thinking"

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for temp in sorted(args.temperature):
        print_table(results, temp, args.model, args.dataset, mode_label, args.prompt_type)

    # Save outputs
    print()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_json(results, args.output_dir, args.temperature)
    save_csv(results, args.output_dir, args.model, args.dataset, args.temperature)

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
