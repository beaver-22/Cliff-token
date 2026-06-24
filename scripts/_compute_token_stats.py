"""Compute per-token rank/logprob/entropy on existing inference outputs.

For each (model, dataset) under output/01_inference/, runs vLLM with
prompt_logprobs=20 on (prompt + response) and saves an augmented copy at
output/02_token_stats/<model>/<dataset>_all_paths.json with three
new parallel arrays per path:
    - response_token_ranks:     List[int]
    - response_token_logprobs:  List[float]
    - response_token_entropies: List[float]  (partial-sum top-K Shannon)

Each model is loaded once, processes all its datasets, then unloaded.

Usage:
    python3 scripts/_compute_token_stats.py [--gpu N]
                                             [--models m1,m2,...]
                                             [--datasets d1,d2,...]
                                             [--source DIR]
                                             [--output_dir DIR]
                                             [--skip-existing]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, ".")

DEFAULT_SOURCE = "output/01_inference"
DEFAULT_OUTPUT = "output/02_token_stats"
DEFAULT_DATASETS = ["gsm1k_100", "math500_100", "aime25"]

# Map directory name → CLI alias used by config.resolve_model_path().
# Preserves canonical order (Qwen → Llama → Gemma, larger first within family).
MODEL_DIR_TO_ALIAS = {
    "Qwen3-8B":               "qwen3-8b",
    "Qwen3-8B-greedy":        "qwen3-8b",
    "Qwen3-4B":               "qwen3-4b",
    "Qwen3-0.6B":             "qwen3-0.6b",
    "Llama-3.2-1B-Instruct":  "llama-3.2-1b",
    "Llama-3.2-3B-Instruct":  "llama-3.2-3b",
    "Llama-3.1-8B-Instruct":  "llama-3.1-8b",
    "gemma-3-4b-it":          "gemma-3-4b",
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token rank/logprob/entropy on inference outputs."
    )
    parser.add_argument("--gpu", default="0",
                        help="GPU device ID (default: 0)")
    parser.add_argument("--models", default="",
                        help="Comma-separated model dir names; default = all found in --source")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="Comma-separated dataset names")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Input directory with inference results")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT,
                        help="Output root directory")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (model, dataset) pairs whose output already exists")
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_tokens_per_batch", type=int, default=4000)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--gpu_mem", type=float, default=0.65)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from src import config
    from src.cli import _init_heavy_imports
    _init_heavy_imports()
    from vllm import LLM
    from transformers import AutoTokenizer
    from src.analysis.entropy import compute_per_token_stats

    # Determine which model directories to process, in canonical order.
    if args.models:
        wanted_dirs = [m.strip() for m in args.models.split(",") if m.strip()]
        # Accept both dir names (Qwen3-8B) and aliases (qwen3-8b)
        alias_to_dir = {v: k for k, v in MODEL_DIR_TO_ALIAS.items()}
        resolved = []
        for m in wanted_dirs:
            if m in MODEL_DIR_TO_ALIAS:
                resolved.append(m)
            elif m in alias_to_dir:
                resolved.append(alias_to_dir[m])
            else:
                print(f"  WARN: unknown model '{m}', skipping")
        wanted_models = [m for m in MODEL_DIR_TO_ALIAS if m in resolved]
    else:
        # Auto-discover from source directory
        wanted_models = [
            m for m in MODEL_DIR_TO_ALIAS
            if os.path.isdir(os.path.join(args.source, m))
        ]

    wanted_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"Source:    {args.source}")
    print(f"Output:    {args.output_dir}")
    print(f"GPU:       {args.gpu}")
    print(f"Models:    {wanted_models}")
    print(f"Datasets:  {wanted_datasets}")
    print()

    for model_dir in wanted_models:
        # Build (dataset, src_path, dst_path) work list for this model.
        work = []
        for ds in wanted_datasets:
            src = os.path.join(args.source, model_dir, f"{ds}_all_paths.json")
            if not os.path.exists(src):
                continue
            dst = os.path.join(args.output_dir, model_dir, f"{ds}_all_paths.json")
            if args.skip_existing and os.path.exists(dst):
                print(f"  [skip] {model_dir}/{ds}")
                continue
            work.append((ds, src, dst))

        if not work:
            print(f"  [{model_dir}] no work to do, skipping")
            continue

        alias = MODEL_DIR_TO_ALIAS[model_dir]
        model_path = config.resolve_model_path(alias)
        print(f"\n[{model_dir}] loading {model_path} ...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_mem,
            trust_remote_code=True,
            enable_prefix_caching=False,
            enforce_eager=False,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_tokens_per_batch,
            max_model_len=args.max_model_len,
            disable_cascade_attn=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[{model_dir}] model loaded.")

        for ds, src, dst in work:
            paths = json.load(open(src))
            print(f"\n[{model_dir}/{ds}] {len(paths)} paths — computing per-token stats...")
            stats_list = compute_per_token_stats(
                llm, tokenizer, paths,
                prompt_logprobs_k=20,
                max_tokens_per_batch=args.max_tokens_per_batch,
                max_model_len=args.max_model_len,
            )
            n_ok = sum(1 for s in stats_list if s is not None)
            print(f"[{model_dir}/{ds}] {n_ok}/{len(paths)} paths processed")

            # Augment paths in place with rank/logprob/entropy arrays.
            for p, stats in zip(paths, stats_list):
                if stats is None:
                    p["response_token_ranks"] = []
                    p["response_token_logprobs"] = []
                    p["response_token_entropies"] = []
                else:
                    p.update(stats)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                json.dump(paths, f, ensure_ascii=False)
            print(f"[{model_dir}/{ds}] saved → {dst}")

        # Release GPU memory before loading the next model.
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{model_dir}] done, GPU released")

    print("\n=== All models done ===")
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip Python shutdown to bypass vLLM teardown hang.
    os._exit(0)


if __name__ == "__main__":
    main()
