"""
Step 4: Post-training Evaluation

Evaluates baseline and DPO-trained models on math benchmarks using accuracy.
Supports single model eval, LoRA adapter eval, and multi-model comparison.

Default dataset: gsm8k (test set)
Full evaluation suite: gsm8k, gsm1k, math500, aime25

Usage:
    # Default: gsm8k test set only
    python -m src.dpo.evaluate --model ./model/Qwen3-0.6B

    # Full suite (4 datasets)
    python -m src.dpo.evaluate --model ./model/Qwen3-0.6B \
        --datasets gsm8k gsm1k math500 aime25

    # Paper-profile token limits (long-context one-shot report run)
    python -m src.dpo.evaluate --model ./model/Qwen3-0.6B \
        --full_suite --token_profile paper

    # LoRA adapter
    python -m src.dpo.evaluate \
        --model ./model/Qwen3-0.6B \
        --adapter_path ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all/

    # Compare multiple Cliff-DPO variants on full suite
    python -m src.dpo.evaluate \
        --model ./model/Qwen3-0.6B \
        --adapter_paths none \
            ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_all/ \
            ./output/09_cliff_dpo/03_training/Qwen3-0.6B/gsm8k/cliff_uncertainty_sampled_off_only/ \
        --labels Baseline "Cliff-all" "Cliff-uncertainty-sampled-off" \
        --datasets gsm8k gsm1k math500 aime25
"""

# Full evaluation suite (gsm8k testset + 3 OOD benchmarks)
FULL_EVAL_DATASETS = ["gsm8k", "gsm1k", "math500", "aime25"]
DEFAULT_EVAL_DATASETS = ["gsm8k"]  # initial phase: only gsm8k testset

# Evaluate-time vLLM memory budget. Assumes eval has the GPU(s) exclusively
# (no sibling training process on the same device). Override at the CLI with
# `--gpu_memory_utilization` if you need to coexist with other workloads.
EVAL_GPU_MEMORY_UTILIZATION = 0.9
import argparse
import json
import logging
import os
import subprocess
import sys
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from src import config
TOKEN_PROFILE_CHOICES = tuple(sorted(config.DATASET_MAX_TOKENS_PROFILES.keys()))
from src.analysis.generator import build_chat_prompt, sample_problems
from src.dpo.logging_utils import parse_log_level, setup_logger
from src.utils.grader import batch_grade_responses_mathverify

logger = logging.getLogger("dpo.step5_eval")


# ============================================================
# Core evaluation
# ============================================================

def evaluate_model(
    model_path: str,
    dataset_name: str,
    adapter_path: Optional[str] = None,
    mode: str = "non_thinking",
    token_profile: str = "default",
    output_dir: Optional[str] = None,
    gpu_ids: List[int] = None,
    llm: Optional[LLM] = None,
    tokenizer=None,
    lora_request: Optional[LoRARequest] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    gpu_memory_utilization: float = EVAL_GPU_MEMORY_UTILIZATION,
    temperature: Optional[float] = None,
    aime_samples: int = 1,
    aime_temperature: Optional[float] = None,
) -> Dict:
    """Evaluate a single model (optionally with LoRA adapter) on a dataset.

    Returns dict with accuracy, n_correct, n_total.

    When num_shards > 1, only problems[shard_id::num_shards] are evaluated and
    the saved file is suffixed with `_shard{i}of{N}` so a separate merge step
    can aggregate per-shard counts into the final eval_{dataset}.json.
    """
    if gpu_ids is None:
        gpu_ids = [0]

    # Load dataset (sample_problems treats num_problems >= len(dataset) as "all";
    # passing 0 returns an empty list because random.sample(_, 0) is empty.)
    dataset_path = config.get_dataset_path(dataset_name)
    problems = sample_problems(dataset_path, num_problems=sys.maxsize)
    source_name = config.get_dataset_source_name(dataset_name)
    if num_shards > 1:
        problems = problems[shard_id::num_shards]
        logger.info(
            f"  Dataset: {dataset_name} (shard {shard_id}/{num_shards}, {len(problems)} problems)"
        )
    else:
        logger.info(f"  Dataset: {dataset_name} ({len(problems)} problems)")

    # Init vLLM if not provided
    own_llm = False
    if llm is None:
        # In shard mode the parent shell pins CUDA_VISIBLE_DEVICES; don't override.
        if num_shards <= 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        enable_lora = adapter_path is not None
        logger.info(
            f"  vLLM init: gpu_memory_utilization={gpu_memory_utilization} "
            f"(eval default; rollout uses {config.GPU_MEMORY_UTILIZATION})"
        )
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
            enable_lora=enable_lora,
            max_lora_rank=64 if enable_lora else None,
        )
        own_llm = True

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build LoRA request if adapter provided
    if adapter_path and lora_request is None:
        lora_request = LoRARequest("dpo_adapter", 1, adapter_path)

    # Sampling config
    sampling_cfg = config.get_sampling_config(mode, model_path)
    max_new_tokens = config.get_max_tokens(dataset_name, mode, token_profile=token_profile)

    # AIME-style datasets: multi-sample avg@N with model's own stochastic temp.
    # Matches scripts/run_fullset_eval.sh (aime_samples=64, Qwen3 non_thinking=0.7).
    is_aime_multi = dataset_name.lower().startswith("aime") and aime_samples > 1

    if is_aime_multi:
        resolved_aime_temp = (
            aime_temperature if aime_temperature is not None else sampling_cfg.temperature
        )
        eval_temp = resolved_aime_temp
        eval_top_p = sampling_cfg.top_p
        eval_top_k = sampling_cfg.top_k if sampling_cfg.top_k > 0 else -1
        n_samples = aime_samples
    else:
        # Temperature override: default=0 (greedy) for reproducible eval.
        # When temp=0, force top_p=1 and top_k=-1 to ensure pure greedy.
        eval_temp = temperature if temperature is not None else 0.0
        if eval_temp == 0.0:
            eval_top_p = 1.0
            eval_top_k = -1
        else:
            eval_top_p = sampling_cfg.top_p
            eval_top_k = sampling_cfg.top_k if sampling_cfg.top_k > 0 else -1
        n_samples = 1

    logger.info(
        f"  Token profile: {token_profile} | max_new_tokens={max_new_tokens} "
        f"| temperature={eval_temp} | n={n_samples}"
        + (f" (aime avg@{n_samples})" if is_aime_multi else "")
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=eval_temp,
        top_p=eval_top_p,
        top_k=eval_top_k,
        presence_penalty=sampling_cfg.presence_penalty,
        repetition_penalty=sampling_cfg.repetition_penalty,
        max_tokens=max_new_tokens,
        stop=config.STOP_TOKENS,
    )

    # Build prompts
    prompts = []
    for problem in problems:
        prompt = build_chat_prompt(
            question=problem["question"],
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            enable_thinking=sampling_cfg.enable_thinking,
            use_chat_template=sampling_cfg.use_chat_template,
            prompt_type=sampling_cfg.prompt_type,
        )
        prompts.append(prompt)

    # Generate
    logger.info(f"  Generating {len(prompts)} responses...")
    generate_kwargs = {}
    if lora_request:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    # Grade — flatten n samples/problem into parallel arrays, then aggregate.
    flat_responses: List[str] = []
    flat_goldens: List[str] = []
    per_problem_span: List[int] = []  # number of samples per problem
    for out, problem in zip(outputs, problems):
        sample_texts = [o.text for o in out.outputs]
        flat_responses.extend(sample_texts)
        flat_goldens.extend([problem["answer"]] * len(sample_texts))
        per_problem_span.append(len(sample_texts))

    correctness = batch_grade_responses_mathverify(flat_responses, flat_goldens, source_name)

    per_problem: List[Dict] = []
    cursor = 0
    for problem, span in zip(problems, per_problem_span):
        samples_correct = sum(correctness[cursor:cursor + span])
        per_problem.append({
            "problem_id": problem.get("id"),
            "n_correct": samples_correct,
            "n_samples": span,
        })
        cursor += span

    n_correct = sum(correctness)
    n_total = len(correctness)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    result = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
    }
    if is_aime_multi:
        result["aime_samples"] = aime_samples
        result["aime_temperature"] = eval_temp
        result["per_problem"] = per_problem

    # Save if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_shard{shard_id}of{num_shards}" if num_shards > 1 else ""
        result_path = os.path.join(output_dir, f"eval_{dataset_name}{suffix}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {result_path}")

    return result


# ============================================================
# Comparison evaluation
# ============================================================

def compare_models(
    model_path: str,
    adapter_paths: List[Optional[str]],
    labels: List[str],
    dataset_names: List[str],
    mode: str = "non_thinking",
    token_profile: str = "default",
    output_dir: Optional[str] = None,
    gpu_ids: List[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    gpu_memory_utilization: float = EVAL_GPU_MEMORY_UTILIZATION,
    temperature: Optional[float] = None,
    aime_samples: int = 1,
    aime_temperature: Optional[float] = None,
) -> Dict[str, List[Dict]]:
    """Evaluate multiple models × multiple datasets. Prints per-dataset tables.

    Returns: {dataset_name: [result_per_model, ...]}
    """
    if gpu_ids is None:
        gpu_ids = [0]

    if num_shards <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    has_adapters = any(p is not None for p in adapter_paths)

    # Single LLM instance, swap adapters & iterate datasets
    logger.info(
        f"Initializing vLLM (enable_lora={has_adapters}, "
        f"gpu_memory_utilization={gpu_memory_utilization})..."
    )
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
        enable_lora=has_adapters,
        max_lora_rank=64 if has_adapters else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_results: Dict[str, List[Dict]] = {}

    for dataset_name in dataset_names:
        logger.info(f"\n{'#'*60}\n# Dataset: {dataset_name}\n{'#'*60}")
        ds_results = []
        for adapter_idx, (label, adapter_path) in enumerate(zip(labels, adapter_paths)):
            logger.info(f"\n--- Evaluating: {label} on {dataset_name} ---")
            lora_request = None
            if adapter_path:
                # Each adapter MUST get a unique lora_int_id. vLLM caches
                # LoRA weights by this integer; reusing id=1 for every adapter
                # causes only the FIRST one to load — all subsequent silently
                # reuse the first adapter's weights, producing identical
                # (wrong) scores for every row in the comparison table.
                lora_request = LoRARequest(
                    label.replace(" ", "_"), adapter_idx + 1, adapter_path
                )

            eval_output_dir = None
            if output_dir:
                eval_output_dir = os.path.join(output_dir, label.replace(" ", "_"))

            result = evaluate_model(
                model_path=model_path,
                dataset_name=dataset_name,
                adapter_path=adapter_path,
                mode=mode,
                token_profile=token_profile,
                output_dir=eval_output_dir,
                llm=llm,
                tokenizer=tokenizer,
                lora_request=lora_request,
                shard_id=shard_id,
                num_shards=num_shards,
                gpu_memory_utilization=gpu_memory_utilization,
                temperature=temperature,
                aime_samples=aime_samples,
                aime_temperature=aime_temperature,
            )
            result["label"] = label
            ds_results.append(result)

        # Per-dataset table
        logger.info(f"\n{'='*60}")
        logger.info(f"DPO Evaluation Results ({dataset_name})")
        logger.info(f"{'='*60}")
        logger.info(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        logger.info(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for r in ds_results:
            logger.info(f"{r['label']:<20} {r['accuracy']:>9.1%} {r['n_correct']:>10} {r['n_total']:>10}")
        logger.info(f"{'='*60}")

        all_results[dataset_name] = ds_results

    # Combined summary table across datasets
    if len(dataset_names) > 1:
        logger.info(f"\n{'#'*60}\n# Summary: Accuracy across all datasets\n{'#'*60}")
        header = f"{'Model':<20}" + "".join(f"{ds:>12}" for ds in dataset_names)
        logger.info(header)
        logger.info("-" * len(header))
        for i, label in enumerate(labels):
            row = f"{label:<20}"
            for ds in dataset_names:
                acc = all_results[ds][i]["accuracy"]
                row += f"{acc:>11.1%} "
            logger.info(row)
        logger.info("=" * len(header))

    # Save combined results (skipped in shard mode — merged later)
    if output_dir and num_shards <= 1:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, "comparison_all.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved comparison: {combined_path}")

    return all_results


# ============================================================
# Data-parallel sharding
# ============================================================

def _merge_shard_files(base_dir: str, dataset_name: str, num_shards: int) -> Dict:
    n_correct = 0
    n_total = 0
    merged_per_problem: List[Dict] = []
    base_record = None
    for i in range(num_shards):
        path = os.path.join(base_dir, f"eval_{dataset_name}_shard{i}of{num_shards}.json")
        with open(path) as f:
            r = json.load(f)
        n_correct += r["n_correct"]
        n_total += r["n_total"]
        if "per_problem" in r:
            merged_per_problem.extend(r["per_problem"])
        if base_record is None:
            base_record = {k: v for k, v in r.items()
                           if k not in ("n_correct", "n_total", "accuracy", "per_problem")}
    base_record["n_correct"] = n_correct
    base_record["n_total"] = n_total
    base_record["accuracy"] = n_correct / n_total if n_total else 0.0
    if merged_per_problem:
        base_record["per_problem"] = merged_per_problem
        # Canonical avg@N metric: mean of per-problem accuracies.
        # Equals n_correct/n_total when every problem shares the same n_samples,
        # but we compute it explicitly so future heterogeneous runs still work.
        per_accs = [p["n_correct"] / p["n_samples"] for p in merged_per_problem if p["n_samples"]]
        base_record["avg_at_n"] = sum(per_accs) / len(per_accs) if per_accs else 0.0
    final_path = os.path.join(base_dir, f"eval_{dataset_name}.json")
    with open(final_path, "w") as f:
        json.dump(base_record, f, indent=2)
    logger.info(f"  merged {num_shards} shards → {final_path} "
                f"({n_correct}/{n_total} = {base_record['accuracy']:.1%})")
    return base_record


def _log_dataset_table(dataset_name: str, ds_results: List[Dict]) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"DPO Evaluation Results ({dataset_name})")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    logger.info(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for r in ds_results:
        logger.info(f"{r['label']:<20} {r['accuracy']:>9.1%} {r['n_correct']:>10} {r['n_total']:>10}")
    logger.info(f"{'='*60}")


def _log_summary_table(dataset_names: List[str], labels: List[str], all_results: Dict) -> None:
    logger.info(f"\n{'#'*60}\n# Summary: Accuracy across all datasets\n{'#'*60}")
    header = f"{'Model':<20}" + "".join(f"{ds:>12}" for ds in dataset_names)
    logger.info(header)
    logger.info("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:<20}"
        for ds in dataset_names:
            acc = all_results[ds][i]["accuracy"]
            row += f"{acc:>11.1%} "
        logger.info(row)
    logger.info("=" * len(header))


def _merge_all_shards(
    output_dir: str,
    dataset_names: List[str],
    labels: Optional[List[str]],
    num_shards: int,
) -> None:
    """Merge per-shard outputs. labels=None → single-model layout (output_dir/);
    otherwise comparison layout (output_dir/<label>/)."""
    if labels is None:
        for ds in dataset_names:
            _merge_shard_files(output_dir, ds, num_shards)
        return

    all_results: Dict[str, List[Dict]] = {}
    for ds in dataset_names:
        ds_results = []
        for label in labels:
            label_dir = os.path.join(output_dir, label.replace(" ", "_"))
            merged = _merge_shard_files(label_dir, ds, num_shards)
            merged["label"] = label
            ds_results.append(merged)
        all_results[ds] = ds_results
        _log_dataset_table(ds, ds_results)

    if len(dataset_names) > 1:
        _log_summary_table(dataset_names, labels, all_results)

    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "comparison_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved comparison: {combined_path}")


def _spawn_data_parallel(args, gpu_ids: List[int], log_dir: str) -> int:
    """Launch one subprocess per GPU, each handling one shard. Returns 0 on success."""
    n = len(gpu_ids)
    logger.info(f"Data-parallel mode: {n} shards across GPUs {gpu_ids}")

    # Rebuild argv: replace --gpus with "0", strip any existing --shard_id/--num_shards.
    base_args: List[str] = []
    skip = 0
    src = sys.argv[1:]
    i = 0
    while i < len(src):
        a = src[i]
        if a in ("--shard_id", "--num_shards"):
            i += 2
            continue
        if a == "--gpus":
            i += 2
            continue
        base_args.append(a)
        i += 1
    base_args += ["--gpus", "0"]

    cmd_prefix = [sys.executable, "-m", "src.dpo.evaluate"]
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for shard_id, gid in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        cmd = cmd_prefix + base_args + ["--shard_id", str(shard_id), "--num_shards", str(n)]
        log_path = os.path.join(log_dir, f"eval_shard{shard_id}of{n}.log")
        f = open(log_path, "w")
        logger.info(f"  -> shard {shard_id} on GPU {gid} (log: {log_path})")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, f, shard_id, gid))

    fail = False
    for p, f, shard_id, gid in procs:
        rc = p.wait()
        f.close()
        if rc != 0:
            logger.error(f"  shard {shard_id} (GPU {gid}, pid {p.pid}) FAILED rc={rc}")
            fail = True
        else:
            logger.info(f"  shard {shard_id} (GPU {gid}) ok")
    return 1 if fail else 0


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cliff-DPO Model Evaluation")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Dataset names. Default: {DEFAULT_EVAL_DATASETS}. "
                             f"Full suite: {FULL_EVAL_DATASETS}")
    parser.add_argument("--dataset", default=None,
                        help="(legacy) single dataset name; prefer --datasets")
    parser.add_argument("--full_suite", action="store_true",
                        help=f"Shortcut for --datasets {' '.join(FULL_EVAL_DATASETS)}")
    parser.add_argument("--mode", default="non_thinking", choices=["thinking", "non_thinking"])
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for eval generation. Default: 0.0 (greedy, "
                             "deterministic). Set >0 (e.g. 0.7) for stochastic sampling.")
    parser.add_argument("--aime_samples", type=int, default=1,
                        help="Samples per problem for aime* datasets (avg@N). Default: 1 "
                             "(greedy, same as other datasets). Set to 64 to match "
                             "scripts/run_fullset_eval.sh paper-profile aime25 evaluation.")
    parser.add_argument("--aime_temperature", type=float, default=None,
                        help="Override temperature for aime* multi-sample runs. "
                             "Default: None (auto from MODEL_CONFIGS[<mode>].temperature, "
                             "e.g. 0.7 for Qwen3 non_thinking).")
    parser.add_argument("--token_profile", default="default", choices=TOKEN_PROFILE_CHOICES,
                        help="Token-limit profile. default=fast daily profile, paper=long-context report profile")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=EVAL_GPU_MEMORY_UTILIZATION,
                        help=f"vLLM gpu_memory_utilization for eval. "
                             f"Default: {EVAL_GPU_MEMORY_UTILIZATION} (lower than the "
                             f"rollout pipeline's {config.GPU_MEMORY_UTILIZATION} so "
                             f"auto-eval can coexist with sibling training processes "
                             f"on the same GPU). Bump up if you have the GPU exclusively.")

    # Single adapter
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter path (single eval)")

    # Comparison mode
    parser.add_argument("--adapter_paths", nargs="+", default=None,
                        help="Multiple adapter paths (use 'none' for base model)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each adapter (must match --adapter_paths)")

    # Logging
    parser.add_argument("--log_dir", default="./output/09_cliff_dpo/logs")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # wandb (optional; primarily used by train_dpo --auto_eval handoff)
    parser.add_argument("--wandb_project", default=None,
                        help="If set, log per-(label, dataset) accuracy to a wandb run")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_tags", default=None,
                        help="Comma-separated tags")
    parser.add_argument("--wandb_mode", default="online",
                        choices=["online", "offline", "disabled"])

    # Data-parallel sharding (multi-gpu --gpus auto-spawns one shard per GPU)
    parser.add_argument("--shard_id", type=int, default=0,
                        help="(internal) index of this shard when running data-parallel")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="(internal) total number of data-parallel shards")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip evaluation; merge existing per-shard outputs into final files")

    args = parser.parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    model_path = config.resolve_model_path(args.model)
    model_short = config.get_model_short_name(model_path)

    # Default output dir: ./output/09_cliff_dpo/04_eval/{model_short}/
    if args.output_dir is None:
        args.output_dir = f"./output/09_cliff_dpo/04_eval/{model_short}"

    # Setup logger
    global logger
    logger = setup_logger(
        name=f"step5_eval_{model_short}",
        log_dir=args.log_dir,
        level=parse_log_level(args.log_level),
    )

    # Resolve dataset list
    if args.full_suite:
        dataset_names = FULL_EVAL_DATASETS
    elif args.datasets:
        dataset_names = args.datasets
    elif args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = DEFAULT_EVAL_DATASETS
    logger.info(f"Evaluation datasets: {dataset_names}")
    logger.info(f"Token profile: {args.token_profile}")

    # Resolve labels/adapters once so DP merge can reuse them
    is_comparison = args.adapter_paths is not None
    if is_comparison:
        adapter_paths = [None if p.lower() == "none" else p for p in args.adapter_paths]
        labels = args.labels or [
            os.path.basename(p.rstrip("/")) if p else "Baseline" for p in adapter_paths
        ]
        if len(labels) != len(adapter_paths):
            parser.error("--labels must match --adapter_paths in length")
    else:
        adapter_paths = None
        labels = None

    # Auto data-parallel: multi-gpu, not already a sub-shard, not merge-only
    is_dp_root = (
        args.num_shards == 1
        and args.shard_id == 0
        and len(gpu_ids) > 1
        and not args.merge_only
    )
    if is_dp_root:
        rc = _spawn_data_parallel(args, gpu_ids, args.log_dir)
        if rc != 0:
            logger.error("One or more shards failed; skipping merge.")
            sys.exit(1)
        logger.info("\nAll shards complete. Merging...")
        _merge_all_shards(args.output_dir, dataset_names, labels, len(gpu_ids))
        return

    if args.merge_only:
        logger.info(f"Merge-only: aggregating shards from {args.output_dir}")
        if args.num_shards <= 1:
            parser.error("--merge_only requires --num_shards > 1")
        _merge_all_shards(args.output_dir, dataset_names, labels, args.num_shards)
        return

    # Collect results in a uniform shape: {dataset: [{label, accuracy, ...}, ...]}
    final_results: Dict[str, List[Dict]] = {}

    if is_comparison:
        final_results = compare_models(
            model_path=model_path,
            adapter_paths=adapter_paths,
            labels=labels,
            dataset_names=dataset_names,
            mode=args.mode,
            token_profile=args.token_profile,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
            aime_samples=args.aime_samples,
            aime_temperature=args.aime_temperature,
        )
    else:
        # Single model eval (possibly across multiple datasets).
        # Pre-init LLM once so we don't pay vLLM startup per dataset.
        if args.num_shards <= 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        enable_lora = args.adapter_path is not None
        logger.info(
            f"Initializing vLLM (enable_lora={enable_lora}, "
            f"gpu_memory_utilization={args.gpu_memory_utilization})..."
        )
        llm = LLM(
            model=model_path,
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            max_num_seqs=config.MAX_NUM_SEQS,
            max_num_batched_tokens=config.MAX_NUM_BATCHED_TOKENS,
            disable_cascade_attn=True,
            enable_lora=enable_lora,
            max_lora_rank=64 if enable_lora else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        lora_request = None
        if args.adapter_path:
            lora_request = LoRARequest("dpo_adapter", 1, args.adapter_path)

        for ds in dataset_names:
            result = evaluate_model(
                model_path=model_path,
                dataset_name=ds,
                adapter_path=args.adapter_path,
                mode=args.mode,
                token_profile=args.token_profile,
                output_dir=args.output_dir,
                llm=llm,
                tokenizer=tokenizer,
                lora_request=lora_request,
                temperature=args.temperature,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                gpu_memory_utilization=args.gpu_memory_utilization,
                aime_samples=args.aime_samples,
                aime_temperature=args.aime_temperature,
            )
            if args.num_shards <= 1:
                logger.info(
                    f"\n[{ds}] Accuracy: {result['accuracy']:.1%} "
                    f"({result['n_correct']}/{result['n_total']})"
                )
            single_label = (
                os.path.basename((args.adapter_path or "Base").rstrip("/"))
            )
            result_with_label = {**result, "label": single_label}
            final_results.setdefault(ds, []).append(result_with_label)

    # Optional wandb logging — fires when train_dpo handed off via --auto_eval
    # or when the user invoked evaluate.py directly with --wandb_project. Only
    # the dispatch root (not individual shards) logs; otherwise N shards each
    # spawn their own wandb run.
    if args.wandb_project and args.num_shards <= 1 and final_results:
        try:
            import wandb
            os.environ["WANDB_MODE"] = args.wandb_mode
            os.environ["WANDB_PROJECT"] = args.wandb_project
            if args.wandb_entity:
                os.environ["WANDB_ENTITY"] = args.wandb_entity
            run_name = args.wandb_run_name or "dpo_eval"
            tags = args.wandb_tags.split(",") if args.wandb_tags else None
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                entity=args.wandb_entity,
                tags=tags,
                reinit=True,
            )
            summary: Dict[str, float] = {}
            single_label = (
                None if is_comparison
                else os.path.basename((args.adapter_path or "Base").rstrip("/"))
            )
            for ds_name, ds_results in final_results.items():
                for r in ds_results:
                    label_part = (
                        r.get("label", "")
                        if (is_comparison and len(ds_results) > 1)
                        else ""
                    )
                    prefix = f"eval/{ds_name}"
                    if label_part:
                        safe_label = label_part.replace(" ", "_")
                        prefix = f"eval/{safe_label}/{ds_name}"
                    summary[f"{prefix}/accuracy"] = r["accuracy"]
                    summary[f"{prefix}/n_correct"] = r["n_correct"]
                    summary[f"{prefix}/n_total"] = r["n_total"]
            wandb.log(summary)
            wandb.finish()
            logger.info(f"[wandb] logged {len(summary)} eval metrics to {run_name}")
        except Exception as e:
            logger.warning(f"[wandb] eval logging failed: {e}")


if __name__ == "__main__":
    main()
