#!/usr/bin/env python3
"""
All-model × All-dataset Inference

Runs inference over all model × dataset combinations and saves:
  - Per-problem responses (JSONL): id, question, gold, response, is_correct, token_len, ...
  - Rollout-compatible JSON: *_all_paths.json, *_success_paths.json, *_failure_paths.json
  - Accuracy summary (CSV + JSON)
  - Response length histograms (PNG, one figure per dataset with model overlay)

Usage:
  python -m src.inference --gpus 0 1 2 3
  python -m src.inference --gpus 0 --model qwen3-4b
  python -m src.inference --gpus 0 1 2 3 --prompt_type fewshot
"""
import os, datetime, argparse, json, csv, multiprocessing
from pathlib import Path

from src.config import (
    MODEL_ALIASES, MODEL_CONFIGS, PAPER_MODEL_ALIASES,
    get_model_short_name, resolve_model_path,
    GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODELS   = list(PAPER_MODEL_ALIASES)
DEFAULT_DATASETS = ["gsm1k_100", "math500_100", "aime25"]


def _kst_timestamp() -> str:
    kst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(kst).strftime("%m%d_%H%M%S")


# =============================================================================
# Worker (spawned subprocess per model)
# =============================================================================

def inference_worker(
    gpu_id: int,
    model_alias: str,
    model_path: str,
    datasets: list,
    temperature: float,
    prompt_type: str,
    output_dir: str,
    result_dict,
    max_tokens_override: int = None,
    thinking: bool = False,
    num_problems: int = -1,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM
    from transformers import AutoTokenizer
    from src.analysis.generator import (
        generate_reasoning_paths, load_jsonl, sample_problems,
        paths_to_dicts, split_success_failure,
    )
    from src.config import (
        get_sampling_config_with_temperature, get_dataset_path, get_max_tokens,
    )
    from dataclasses import replace as dc_replace

    capture_ranks = os.environ.get("CAPTURE_RANKS", "0") == "1"
    # When capturing logprobs, vLLM allocates an extra top-K workspace per
    # generated token; lowering gpu_memory_utilization + max_model_len +
    # max_num_seqs leaves room for it.
    if capture_ranks:
        gpu_mem = 0.70
        max_seqs = 64
        max_model_len = 8192
    else:
        gpu_mem = GPU_MEMORY_UTILIZATION
        max_seqs = MAX_NUM_SEQS
        max_model_len = None  # let vLLM use model default

    print(f"[{model_alias}] Loading model on GPU {gpu_id} "
          f"(mem_util={gpu_mem}, max_seqs={max_seqs}, max_model_len={max_model_len})...")
    try:
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            max_num_seqs=max_seqs,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            disable_cascade_attn=True,  # Workaround for A100 FA2 LSE bug (V1)
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm = LLM(**llm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"[{model_alias}] Model load FAILED: {exc}")
        return

    print(f"[{model_alias}] Model loaded.")

    # Resolve mode
    model_cfg = MODEL_CONFIGS.get(model_path, {})
    modes = [k for k in model_cfg if k != "name"]
    if thinking and "thinking" in modes:
        mode = "thinking"
    else:
        mode = "non_thinking" if "non_thinking" in modes else modes[0]
        if thinking and "thinking" not in modes:
            print(f"[{model_alias}] WARNING: no thinking mode, using {mode}")

    for dataset in datasets:
        dataset_path = get_dataset_path(dataset)
        try:
            problems = load_jsonl(dataset_path)
        except FileNotFoundError:
            print(f"[{model_alias}] Dataset not found: {dataset_path}")
            continue
        if num_problems and num_problems > 0:
            problems = problems[:num_problems]

        sampling_config = get_sampling_config_with_temperature(
            mode=mode, model_path=model_path, temperature=temperature,
        )
        sampling_config = dc_replace(sampling_config, prompt_type=prompt_type)
        # Use override if given, otherwise dataset-specific max_tokens from config
        max_new_tokens = max_tokens_override if max_tokens_override else get_max_tokens(dataset, mode)

        print(f"[{model_alias}] {dataset} | {len(problems)} problems | "
              f"T={temperature} | max_tokens={max_new_tokens} | mode={mode} | prompt={prompt_type}")

        try:
            capture_ranks = os.environ.get("CAPTURE_RANKS", "0") == "1"
            paths = generate_reasoning_paths(
                llm=llm,
                tokenizer=tokenizer,
                problems=problems,
                paths_per_problem=1,
                dataset_name=dataset,
                sampling_config=sampling_config,
                max_new_tokens=max_new_tokens,
                capture_ranks=capture_ranks,
            )

            out_dir = Path(output_dir) / get_model_short_name(model_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save per-problem JSONL (lightweight, for quick analysis)
            with open(out_dir / f"{dataset}.jsonl", "w") as f:
                for p in paths:
                    resp = p.response
                    # Measure think block length
                    think_len = 0
                    if '</think>' in resp:
                        think_end = resp.index('</think>')
                        think_start = resp.index('<think>') if '<think>' in resp else 0
                        think_len = len(resp[think_start:think_end])
                    rec = {
                        "id": p.problem_id,
                        "question": p.question,
                        "gold": p.golden_answer,
                        "response": resp,
                        "is_correct": p.is_correct,
                        "token_len": p.total_tokens,
                        "has_boxed": r'\boxed' in resp,
                        "has_think": '<think>' in resp or '</think>' in resp,
                        "think_chars": think_len,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Save rollout-compatible format (*_all_paths.json)
            all_dicts = paths_to_dicts(paths)
            groups = split_success_failure(paths)
            with open(out_dir / f"{dataset}_all_paths.json", "w") as f:
                json.dump(all_dicts, f, ensure_ascii=False)
            with open(out_dir / f"{dataset}_success_paths.json", "w") as f:
                json.dump(paths_to_dicts(groups["success"]), f, ensure_ascii=False)
            with open(out_dir / f"{dataset}_failure_paths.json", "w") as f:
                json.dump(paths_to_dicts(groups["failure"]), f, ensure_ascii=False)

            n_correct = sum(p.is_correct for p in paths)
            accuracy = n_correct / len(paths)
            key = (model_alias, dataset)
            result_dict[key] = {
                "model_alias": model_alias,
                "model_name": get_model_short_name(model_path),
                "dataset": dataset,
                "temperature": temperature,
                "mode": mode,
                "prompt_type": prompt_type,
                "accuracy": accuracy,
                "n_correct": n_correct,
                "n_total": len(paths),
                "max_tokens": max_new_tokens,
            }
            print(f"[{model_alias}] {dataset}: {accuracy*100:.1f}% ({n_correct}/{len(paths)})")

        except Exception as exc:
            print(f"[{model_alias}] ERROR on {dataset}: {exc}")

    print(f"[{model_alias}] Done.")


# =============================================================================
# Histogram generation (runs after all workers finish, no GPU needed)
# =============================================================================

def generate_histograms(output_dir: str, models: list, datasets: list):
    """One figure per dataset, models distinguished by color and overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[histogram] matplotlib not installed, skipping.")
        return

    hist_dir = Path(output_dir) / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Model display names & colors
    colors = {
        "qwen3-0.6b":   "#1f77b4",
        "qwen3-4b":     "#ff7f0e",
        "qwen3-8b":     "#2ca02c",
        "llama-3.2-1b": "#9467bd",
        "llama-3.2-3b": "#8c564b",
        "llama-3.1-8b": "#d62728",
        "gemma-3-4b":   "#17becf",
    }
    DATASET_DISPLAY = {
        "gsm1k": "GSM1K", "gsm1k_100": "GSM1K-100",
        "math500": "MATH-500", "math500_100": "MATH-100",
        "aime24": "AIME-2024", "aime25": "AIME-2025",
    }

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        for model_alias in models:
            model_name = get_model_short_name(resolve_model_path(model_alias))
            jsonl_path = Path(output_dir) / model_name / f"{dataset}.jsonl"
            if not jsonl_path.exists():
                continue

            with open(jsonl_path) as f:
                records = [json.loads(l) for l in f if l.strip()]
            if not records:
                continue

            lengths = [r["token_len"] for r in records]
            n_correct = sum(1 for r in records if r["is_correct"])
            accuracy = n_correct / len(records)
            label = f"{model_name} ({accuracy*100:.1f}%, avg={np.mean(lengths):.0f}tok)"
            color = colors.get(model_alias, None)

            ax.hist(lengths, bins=50, alpha=0.45, label=label, color=color, edgecolor="white", linewidth=0.3)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ds_display = DATASET_DISPLAY.get(dataset, dataset)
        ax.set_title(f"Response Length Distribution — {ds_display} (T=0.6)", fontsize=13)
        ax.set_xlabel("Response Length (tokens)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig_path = hist_dir / f"{dataset}_length_dist.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram: {fig_path}")


# =============================================================================
# Summary output
# =============================================================================

def save_summary(output_dir: str, result_dict: dict, models: list, datasets: list):
    out = Path(output_dir)
    DATASET_DISPLAY = {
        "gsm1k": "GSM1K", "gsm1k_100": "GSM1K-100",
        "math500": "MATH-500", "math500_100": "MATH-100",
        "aime24": "AIME-2024", "aime25": "AIME-2025",
    }

    # CSV
    csv_path = out / "summary.csv"
    fieldnames = ["model_alias", "model_name", "dataset", "temperature",
                  "mode", "prompt_type", "accuracy", "n_correct", "n_total", "max_tokens"]
    rows = []
    for model in models:
        for dataset in datasets:
            key = (model, dataset)
            if key in result_dict:
                r = dict(result_dict[key])
                r["accuracy"] = f"{r['accuracy']:.4f}"
                rows.append(r)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    # JSON (accuracy matrix)
    json_path = out / "summary.json"
    matrix = {}
    for model in models:
        matrix[model] = {}
        for dataset in datasets:
            key = (model, dataset)
            if key in result_dict:
                matrix[model][dataset] = dict(result_dict[key])
    with open(json_path, "w") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # Console table
    col_w = 12
    model_w = 18
    ds_headers = [DATASET_DISPLAY.get(d, d).center(col_w) for d in datasets]
    print(f"\n{'Model':<{model_w}}| " + " | ".join(ds_headers))
    print("-" * model_w + "|" + "".join("-" * (col_w + 2) + "|" for _ in datasets))
    for alias in models:
        model_name = get_model_short_name(resolve_model_path(alias))
        cells = []
        for dataset in datasets:
            key = (alias, dataset)
            if key in result_dict:
                acc = result_dict[key]["accuracy"]
                cells.append(f"{acc*100:6.1f}%".center(col_w))
            else:
                cells.append("N/A".center(col_w))
        print(f"{model_name:<{model_w}}| " + " | ".join(cells))
    print()


# =============================================================================
# CLI + Main
# =============================================================================

def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Run inference for all model × dataset combos")
    parser.add_argument("--model", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODEL_ALIASES.keys()), metavar="MODEL")
    from src.config import DATASET_PATHS
    parser.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS,
                        choices=sorted(DATASET_PATHS.keys()), metavar="DS")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--prompt_type", default="zeroshot",
                        choices=["zeroshot", "fewshot", "direct"])
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (Qwen3 only)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: use model config)")
    parser.add_argument("--num_problems", type=int, default=-1,
                        help="Limit problems per dataset for smoke tests (-1 = all)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./output/inference_{_kst_timestamp()}"

    model_pairs = [(alias, resolve_model_path(alias)) for alias in args.model]

    print(f"Inference Plan:")
    print(f"  Models      : {args.model}")
    print(f"  Datasets    : {args.dataset}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Prompt      : {args.prompt_type}")
    print(f"  Thinking    : {args.thinking}")
    print(f"  max_tokens  : {args.max_tokens or 'model default'}")
    print(f"  num_problems: {args.num_problems if args.num_problems > 0 else 'all'}")
    print(f"  GPUs        : {args.gpus}")
    print(f"  Output      : {args.output_dir}")
    print()

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    processes = []
    for i, (alias, model_path) in enumerate(model_pairs):
        gpu_id = args.gpus[i % len(args.gpus)]
        p = multiprocessing.Process(
            target=inference_worker,
            args=(gpu_id, alias, model_path, args.dataset,
                  args.temperature, args.prompt_type,
                  args.output_dir, result_dict, args.max_tokens,
                  args.thinking, args.num_problems),
            name=f"worker-{alias}",
            daemon=False,
        )
        p.start()
        print(f"Started: {alias} on GPU {gpu_id}")
        processes.append((p, alias))

    print()
    for p, alias in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[WARNING] {alias} exited with code {p.exitcode}")

    # Post-processing (no GPU needed)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    save_summary(args.output_dir, dict(result_dict), args.model, args.dataset)
    generate_histograms(args.output_dir, args.model, args.dataset)
    print("Done.")


if __name__ == "__main__":
    main()
