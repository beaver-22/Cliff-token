"""
Step 6: Cliff Token Count Evaluation (3-Taxonomy)

Counts cliff tokens produced by a model on a test set, broken down into the
3-taxonomy categories from `src.dpo.vllm_rollout.classify_cliff`:

  - deterministic: entropy <= 0.0560 and is_greedy=True
  - uncertain:     entropy >  0.0560 and is_greedy=True
  - sampled_off:   entropy >  0.0560 and is_greedy=False

Pipeline (per model x dataset):
  1. (optional) merge LoRA adapter into a temp dir so vLLM can load it as a
     plain HF model.
  2. Sample test-set problems, generate reasoning paths.
  3. Run rollout to fill `all_position_scores` (statistical cliff signal).
  4. Run Phase A of `extract_top_k_candidates` (k=1) to classify each cliff.
  5. Tally categories, save JSON.

Usage:
    # Smoke test: base model, 2 problems
    python -m src.dpo.step6_cliff_num_eval \
        --model qwen3-0.6b --datasets gsm8k --num_problems 2 --gpus 0

    # Baseline vs trained cliff-DPO LoRA
    python -m src.dpo.step6_cliff_num_eval \
        --model qwen3-0.6b \
        --adapter_paths none ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all/ \
        --labels Baseline Cliff-all \
        --datasets gsm8k --gpus 0
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional

from src import config
from src.analysis.generator import (
    compute_position_scores,
    generate_reasoning_paths,
    paths_to_dicts,
    sample_problems,
)
from src.dpo.logging_utils import parse_log_level, setup_logger
from src.dpo.vllm_rollout import extract_top_k_candidates

logger = logging.getLogger("dpo.step6_cliff_num_eval")

CATEGORIES = ("deterministic", "uncertain", "sampled_off", "other")
EVAL_GPU_MEMORY_UTILIZATION = 0.9


# ============================================================
# LoRA merge helper
# ============================================================

def _merge_lora_adapter(base_path: str, adapter_path: str, out_dir: str) -> None:
    """Merge a LoRA adapter into the base model and save to out_dir."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Merging LoRA adapter {adapter_path} into base {base_path} -> {out_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True, torch_dtype="auto"
    )
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tok.save_pretrained(out_dir)
    del base, merged
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info(f"Merged model saved at {out_dir}")


# ============================================================
# Core: count cliffs for one model on one dataset
# ============================================================

def _build_paths_for_dataset(
    llm,
    tokenizer,
    dataset_name: str,
    sampling_lookup_path: str,
    num_problems: int,
    paths_per_problem: int,
    mode: str,
) -> List[Dict]:
    """Generate reasoning paths and fill `all_position_scores`.

    `sampling_lookup_path` is the path used to resolve the sampling config — it
    must always be the registered base model path so baseline and LoRA-merged
    runs share identical generation settings (temperature, top_p, etc.). The
    actual weights live in the `llm` object passed in.
    """
    sampling_cfg = config.get_sampling_config(mode, sampling_lookup_path)
    max_new_tokens = config.get_max_tokens(dataset_name, mode)

    dataset_path = config.get_dataset_path(dataset_name)
    problems = sample_problems(dataset_path, num_problems=num_problems)
    logger.info(f"  {dataset_name}: {len(problems)} problems sampled")

    logger.info(
        f"  Generating reasoning paths (max_new_tokens={max_new_tokens}, "
        f"paths_per_problem={paths_per_problem})"
    )
    paths = generate_reasoning_paths(
        llm=llm,
        tokenizer=tokenizer,
        problems=problems,
        paths_per_problem=paths_per_problem,
        dataset_name=dataset_name,
        sampling_config=sampling_cfg,
        max_new_tokens=max_new_tokens,
    )

    logger.info(f"  Computing position scores (rollout_samples={config.ROLLOUT_SAMPLES})")
    paths = compute_position_scores(
        llm=llm,
        tokenizer=tokenizer,
        paths=paths,
        dataset_name=dataset_name,
        rollout_samples=config.ROLLOUT_SAMPLES,
        sampling_config=sampling_cfg,
    )
    return paths_to_dicts(paths)


def count_cliffs_for_model(
    model_path: str,
    base_model_path: str,
    adapter_path: Optional[str],
    dataset_name: str,
    label: str,
    output_dir: str,
    num_problems: int,
    paths_per_problem: int,
    mode: str,
    gpu_ids: List[int],
    llm=None,
    tokenizer=None,
    gpu_memory_utilization: float = EVAL_GPU_MEMORY_UTILIZATION,
) -> Dict:
    """Generate paths, detect cliffs, classify, tally. Returns the per-(model,ds) result dict."""
    from transformers import AutoTokenizer
    from vllm import LLM

    own_llm = False
    if llm is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        logger.info(f"Initializing vLLM for {label} on GPUs {gpu_ids}")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            max_num_seqs=config.MAX_NUM_SEQS,
            max_num_batched_tokens=config.MAX_NUM_BATCHED_TOKENS,
            disable_cascade_attn=True,
        )
        own_llm = True
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    paths_dicts = _build_paths_for_dataset(
        llm=llm,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sampling_lookup_path=base_model_path,
        num_problems=num_problems,
        paths_per_problem=paths_per_problem,
        mode=mode,
    )

    paths_with_scores = [p for p in paths_dicts if p.get("all_position_scores")]
    logger.info(f"  {len(paths_with_scores)}/{len(paths_dicts)} paths have position scores")

    n_tokens_total = sum(p.get("total_tokens", 0) for p in paths_with_scores)

    analyses = []
    if paths_with_scores:
        analyses = extract_top_k_candidates(llm, tokenizer, paths_with_scores, k=1)

    counts = {c: 0 for c in CATEGORIES}
    for a in analyses:
        cat = a.category if a.category in counts else "other"
        counts[cat] += 1

    n_paths = len(paths_with_scores)
    n_cliffs_total = sum(counts.values())
    per_path = {c: (counts[c] / n_paths if n_paths else 0.0) for c in CATEGORIES}
    per_1k_tokens = {
        c: (counts[c] / n_tokens_total * 1000.0 if n_tokens_total else 0.0)
        for c in CATEGORIES
    }

    result = {
        "model_path": base_model_path,
        "adapter_path": adapter_path,
        "label": label,
        "dataset": dataset_name,
        "n_paths": n_paths,
        "paths_per_problem": paths_per_problem,
        "n_tokens_total": n_tokens_total,
        "n_cliffs_total": n_cliffs_total,
        "counts": counts,
        "per_path": per_path,
        "per_1k_tokens": per_1k_tokens,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cliff_counts_{dataset_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {out_path}")

    if own_llm:
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result


# ============================================================
# Reporting
# ============================================================

def _log_table(rows: List[Dict]) -> None:
    if not rows:
        return
    logger.info("")
    if any(r["n_paths"] < 32 for r in rows):
        logger.warning(
            "Small-N WARNING: at least one row has n_paths < 32. "
            "Cliff counts have high variance at this scale — bump --num_problems "
            "and/or --paths_per_problem before drawing conclusions."
        )
    logger.info("=" * 86)
    header = (
        f"{'Model':<18}{'Dataset':>10}{'Paths':>8}{'Cliffs':>8}"
        f"{'determ.':>10}{'uncert.':>10}{'samp-off':>10}{'cliffs/path':>14}"
    )
    logger.info(header)
    logger.info("-" * 86)
    for r in rows:
        cliffs_per_path = r["n_cliffs_total"] / r["n_paths"] if r["n_paths"] else 0.0
        logger.info(
            f"{r['label']:<18}{r['dataset']:>10}{r['n_paths']:>8}"
            f"{r['n_cliffs_total']:>8}{r['counts']['deterministic']:>10}{r['counts']['uncertain']:>10}"
            f"{r['counts']['sampled_off']:>10}{cliffs_per_path:>14.2f}"
        )
    logger.info("=" * 86)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DPO Step 6: cliff token count evaluation")
    parser.add_argument("--model", required=True, help="Base model alias or path")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"],
                        help="Test set name(s). Default: ['gsm8k']")
    parser.add_argument("--mode", default="non_thinking", choices=["thinking", "non_thinking"])
    parser.add_argument("--num_problems", type=int, default=sys.maxsize,
                        help="Number of test-set problems to evaluate (default: full set)")
    parser.add_argument("--paths_per_problem", type=int, default=1,
                        help="Reasoning paths per problem (default: 1). Increase to amortize "
                             "sampling variance when comparing models on small test sets.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Default: ./output/09_cliff_dpo/05_cliff_count/{model_short}/")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=EVAL_GPU_MEMORY_UTILIZATION)

    parser.add_argument("--adapter_path", default=None,
                        help="Single LoRA adapter path (single-model eval)")
    parser.add_argument("--adapter_paths", nargs="+", default=None,
                        help="Multiple adapter paths for comparison (use 'none' for base)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each adapter (must match --adapter_paths)")

    parser.add_argument("--log_dir", default="./output/09_cliff_dpo/logs")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.paths_per_problem < 1:
        parser.error("--paths_per_problem must be >= 1")
    if args.num_problems < 1:
        parser.error("--num_problems must be >= 1")

    base_model_path = config.resolve_model_path(args.model)
    base_short = config.get_model_short_name(base_model_path)
    if args.output_dir is None:
        args.output_dir = f"./output/09_cliff_dpo/05_cliff_count/{base_short}"

    global logger
    logger = setup_logger(
        name=f"step6_cliff_num_eval_{base_short}",
        log_dir=args.log_dir,
        level=parse_log_level(args.log_level),
    )

    # Resolve (label, adapter_path) entries.
    if args.adapter_paths is not None:
        adapter_entries = [
            (None if p.lower() == "none" else p) for p in args.adapter_paths
        ]
        if args.labels:
            if len(args.labels) != len(adapter_entries):
                parser.error("--labels must match --adapter_paths in length")
            labels = args.labels
        else:
            labels = [
                "Baseline" if p is None else os.path.basename(p.rstrip("/"))
                for p in adapter_entries
            ]
    else:
        adapter_entries = [args.adapter_path]
        labels = ["Baseline" if args.adapter_path is None
                  else os.path.basename(args.adapter_path.rstrip("/"))]

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Models: {list(zip(labels, adapter_entries))}")
    logger.info(f"num_problems: {args.num_problems}")

    all_rows: List[Dict] = []
    tmp_dirs: List[str] = []

    try:
        for label, adapter_path in zip(labels, adapter_entries):
            # Resolve effective model weights for this label
            if adapter_path is None:
                effective_model_path = base_model_path
            else:
                tmp_dir = tempfile.mkdtemp(
                    prefix=f"step6_merged_{base_short}_{label.replace(' ', '_')}_"
                )
                tmp_dirs.append(tmp_dir)
                _merge_lora_adapter(base_model_path, adapter_path, tmp_dir)
                effective_model_path = tmp_dir

            label_dir = os.path.join(args.output_dir, label.replace(" ", "_"))
            for dataset_name in args.datasets:
                logger.info("")
                logger.info("#" * 60)
                logger.info(f"# {label} | {dataset_name}")
                logger.info("#" * 60)
                row = count_cliffs_for_model(
                    model_path=effective_model_path,
                    base_model_path=base_model_path,
                    adapter_path=adapter_path,
                    dataset_name=dataset_name,
                    label=label,
                    output_dir=label_dir,
                    num_problems=args.num_problems,
                    paths_per_problem=args.paths_per_problem,
                    mode=args.mode,
                    gpu_ids=gpu_ids,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                all_rows.append(row)
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Removed merged-model temp dir: {d}")

    _log_table(all_rows)

    if len(all_rows) > 1:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_path = os.path.join(args.output_dir, "comparison_all.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        logger.info(f"Saved comparison: {combined_path}")


if __name__ == "__main__":
    main()
