#!/usr/bin/env python3
"""
Cliff Token Analysis (CTA) CLI

Main entry point for running inference, rollout, and analysis experiments.

Commands:
- inference: Generate reasoning paths (Stage 1)
- rollout:   Compute tokenwise potential via rollout sampling (Stage 2)
- experiment: Run analysis experiments (RQ1-1, RQ1-2, RQ1-3)

Usage:
    python -m src.cli inference --model qwen3-4b --dataset math500 --num_problems 500
    python -m src.cli rollout --data_path ./output/.../math500_all_paths.json --model qwen3-4b
    python -m src.cli experiment --experiment rq1_1 --data_path ./output/.../math500_all_paths.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from math import comb
from collections import defaultdict

from src import config

# Heavy imports (vLLM, transformers, analysis) are deferred to
# _init_heavy_imports() so multiprocessing spawn workers do not load vLLM/torch.
LLM = SamplingParams = AutoTokenizer = LoRARequest = None
sample_problems = generate_reasoning_paths = compute_position_scores = None
split_success_failure = paths_to_dicts = ReasoningPath = None


def _init_heavy_imports():
    """Load vLLM, transformers, and pipeline modules."""
    global LLM, SamplingParams, AutoTokenizer, LoRARequest
    global sample_problems, generate_reasoning_paths, compute_position_scores
    global split_success_failure, paths_to_dicts, ReasoningPath

    if LLM is not None:
        return

    from vllm import LLM as _LLM, SamplingParams as _SP
    from vllm.lora.request import LoRARequest as _LR
    from transformers import AutoTokenizer as _AT
    LLM = _LLM
    SamplingParams = _SP
    AutoTokenizer = _AT
    LoRARequest = _LR

    from src.analysis.generator import (
        sample_problems as _sp,
        generate_reasoning_paths as _grp,
        compute_position_scores as _cps,
        split_success_failure as _ssf,
        paths_to_dicts as _ptd,
        ReasoningPath as _RP,
    )
    sample_problems = _sp
    generate_reasoning_paths = _grp
    compute_position_scores = _cps
    split_success_failure = _ssf
    paths_to_dicts = _ptd
    ReasoningPath = _RP


# =============================================================================
# Utility Functions
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate Pass@K probability."""
    if c == 0:
        return 0.0
    if c >= n or n - c < k:
        return 1.0
    if k > n:
        k = n
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str, indent=None):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def create_llm(
    model_path: str,
    gpu_ids: list,
    memory_utilization: float = 0.9,
    max_num_seqs: int = None,
    max_num_batched_tokens: int = None,
    adapter_path: str = None,
    max_lora_rank: int = 64,
):
    """Create vLLM instance with optimized settings.

    When adapter_path is provided, enables LoRA serving so callers can pass
    LoRARequest per-generate-call (no merge-to-disk needed). The caller is
    responsible for binding the process to the desired physical GPUs via
    CUDA_VISIBLE_DEVICES before invocation; tensor_parallel_size here uses
    len(gpu_ids) only.
    """
    kwargs = dict(
        model=model_path,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        max_num_seqs=max_num_seqs or config.MAX_NUM_SEQS,
        max_num_batched_tokens=max_num_batched_tokens or config.MAX_NUM_BATCHED_TOKENS,
        disable_cascade_attn=True,   # Workaround for A100 FA2 LSE bug (when using V1)
    )
    if adapter_path:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max_lora_rank
    return LLM(**kwargs)


# =============================================================================
# Stage 1: Inference (Generate Reasoning Paths)
# =============================================================================

def cmd_inference(args):
    """Generate reasoning paths without position scores.

    Returns:
        Path: The output directory where inference results were saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = config.resolve_model_path(args.model)
    mode = args.mode or config.get_default_mode(model_path)
    model_short = config.get_model_short_name(model_path)
    temperature = getattr(args, 'temperature', None)
    temp_suffix = f"_temp{temperature}" if temperature is not None else ""
    output_dir = Path(args.output_dir) / f"inference_{model_short}_{args.dataset}{temp_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_config = config.get_sampling_config_with_temperature(mode, model_path, temperature)

    print("=" * 60)
    print("STAGE 1: INFERENCE (Generate Reasoning Paths)")
    print("=" * 60)
    print(f"Model:        {model_path}")
    print(f"Dataset:      {args.dataset}")
    print(f"Problems:     {args.num_problems}")
    print(f"Paths/Problem:{args.paths_per_problem}")
    print(f"Mode:         {mode}")
    if temperature is not None:
        print(f"Temperature:  {temperature} (override)")
    else:
        print(f"Temperature:  {sampling_config.temperature} (from config)")
    print(f"Output:       {output_dir}")
    print()

    save_json({**vars(args), "resolved_model": model_path, "resolved_mode": mode},
              str(output_dir / "inference_config.json"), indent=2)

    adapter_path = getattr(args, 'adapter_path', None)
    if adapter_path:
        print(f"Adapter:      {adapter_path}")

    print(f"Loading model ({model_path})...")
    gpu_list = [int(g) for g in args.gpus.split(",")]
    llm = create_llm(model_path, gpu_list, config.GPU_MEMORY_UTILIZATION,
                     adapter_path=adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")

    lora_request = None
    if adapter_path:
        lora_request = LoRARequest("adapter", 1, adapter_path)

    split = getattr(args, 'split', config.DEFAULT_SPLIT)
    dataset_path = getattr(args, 'dataset_path', None) or config.get_dataset_path(args.dataset, split)
    problems = sample_problems(dataset_path, args.num_problems)
    print(f"Sampled {len(problems)} problems from {dataset_path}")

    print(f"\nGenerating {len(problems) * args.paths_per_problem} reasoning paths...")
    paths = generate_reasoning_paths(
        llm=llm,
        tokenizer=tokenizer,
        problems=problems,
        paths_per_problem=args.paths_per_problem,
        dataset_name=args.dataset,
        sampling_config=sampling_config,
        max_new_tokens=config.get_max_tokens(args.dataset, mode, token_profile=getattr(args, 'token_profile', 'default')),
        lora_request=lora_request,
    )

    paths_dicts = paths_to_dicts(paths)
    groups = split_success_failure(paths)
    success_dicts = paths_to_dicts(groups["success"])
    failure_dicts = paths_to_dicts(groups["failure"])

    save_json(paths_dicts,   str(output_dir / f"{args.dataset}_all_paths.json"))
    save_json(success_dicts, str(output_dir / f"{args.dataset}_success_paths.json"))
    save_json(failure_dicts, str(output_dir / f"{args.dataset}_failure_paths.json"))

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total paths: {len(paths_dicts)}")
    print(f"  Success:   {len(success_dicts)}")
    print(f"  Failure:   {len(failure_dicts)}")
    print(f"\nOutput: {output_dir}")
    print(f"\nNext step:")
    print(f"  python -m src.cli rollout --model {args.model} --data_path {output_dir / f'{args.dataset}_all_paths.json'}")

    return output_dir


# =============================================================================
# Stage 2: Rollout (Compute Tokenwise Potential)
# =============================================================================

def cmd_rollout(args):
    """Compute tokenwise potential via rollout sampling."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = config.resolve_model_path(args.model)
    model_short = config.get_model_short_name(model_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.OUTPUT_DIR) / "03_rollout" / model_short
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode or config.get_default_mode(model_path)
    temperature = getattr(args, 'temperature', None)
    sampling_config = config.get_sampling_config_with_temperature(mode, model_path, temperature)

    use_optimized = not getattr(args, 'no_optimized', False)
    global_batch_size = getattr(args, 'global_batch_size', config.GLOBAL_BATCH_SIZE)
    max_grading_workers = getattr(args, 'max_grading_workers', config.MAX_GRADING_WORKERS)
    rollout_window = getattr(args, 'rollout_window', config.ROLLOUT_WINDOW_SIZE)
    early_termination_k = getattr(args, 'early_termination_k', config.EARLY_TERMINATION_K)

    print("=" * 60)
    print("STAGE 2: ROLLOUT (Compute Tokenwise Potential)")
    print("=" * 60)
    print(f"Model:          {model_path}")
    print(f"Data:           {args.data_path}")
    print(f"Rollout samples:{args.rollout_samples}")
    print(f"Rollout window: {rollout_window}")
    print(f"Early term K:   {early_termination_k}")
    print(f"Mode:           {mode}")
    if temperature is not None:
        print(f"Temperature:    {temperature} (override)")
    print(f"Optimized:      {use_optimized}")
    if use_optimized:
        print(f"  Global batch: {global_batch_size}")
        print(f"  Grade workers:{max_grading_workers}")
    else:
        print(f"  Batch size:   {args.batch_size}")
    print(f"Output:         {output_dir}")
    print()

    print("Loading inference data...")
    paths_dicts = load_json(args.data_path)
    print(f"Loaded {len(paths_dicts)} paths.")

    if args.failure_only:
        paths_dicts = [p for p in paths_dicts if not p.get("is_correct", True)]
        print(f"Filtered to {len(paths_dicts)} failure paths.")

    paths = [
        ReasoningPath(
            id=p["id"],
            problem_id=p["problem_id"],
            question=p["question"],
            golden_answer=p["golden_answer"],
            response=p["response"],
            is_correct=p["is_correct"],
            response_tokens=p["response_tokens"],
            response_token_ids=p["response_token_ids"],
            all_position_scores=p.get("all_position_scores", []),
            total_tokens=p["total_tokens"],
            full_prompt=p["full_prompt"],
        )
        for p in paths_dicts
    ]

    save_json({**vars(args), "output_dir": str(output_dir), "num_paths": len(paths)},
              str(output_dir / f"{args.dataset}_rollout_config.json"), indent=2)

    adapter_path = getattr(args, 'adapter_path', None)
    if adapter_path:
        print(f"Adapter:        {adapter_path}")

    print(f"\nLoading model ({model_path})...")
    gpu_list = [int(g) for g in args.gpus.split(",")]

    # Pipeline monitor (GPU/CPU utilization tracking)
    from src.utils.monitor import PipelineMonitor
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor = PipelineMonitor(
        gpu_ids=gpu_list,
        interval=0.5,
        output_dir=str(monitor_dir),
        prefix=f"{args.dataset}_",
    )
    llm = create_llm(model_path, gpu_list, config.GPU_MEMORY_UTILIZATION,
                     adapter_path=adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")

    lora_request = None
    if adapter_path:
        lora_request = LoRARequest("adapter", 1, adapter_path)

    checkpoint_path = str(output_dir / f"{args.dataset}_checkpoint.jsonl")
    print(f"\nComputing tokenwise potential ({args.rollout_samples} rollouts/position, rollout_max_tokens={config.get_rollout_max_tokens(args.dataset, mode)})...")
    print(f"Checkpoint: {checkpoint_path}")
    monitor.start()
    paths = compute_position_scores(
        llm=llm,
        tokenizer=tokenizer,
        paths=paths,
        dataset_name=args.dataset,
        rollout_samples=args.rollout_samples,
        sampling_config=sampling_config,
        global_batch_size=global_batch_size,
        max_grading_workers=max_grading_workers,
        rollout_window=rollout_window,
        monitor=monitor,
        early_termination_k=early_termination_k,
        checkpoint_path=checkpoint_path,
        lora_request=lora_request,
    )
    monitor.stop()
    monitor.save_raw()
    monitor.plot()

    paths_dicts = paths_to_dicts(paths)
    groups = split_success_failure(paths)
    success_dicts = paths_to_dicts(groups["success"])
    failure_dicts = paths_to_dicts(groups["failure"])

    save_json(paths_dicts,   str(output_dir / f"{args.dataset}_all_paths.json"))
    save_json(success_dicts, str(output_dir / f"{args.dataset}_success_paths.json"))
    save_json(failure_dicts, str(output_dir / f"{args.dataset}_failure_paths.json"))

    print()
    print("=" * 60)
    print("ROLLOUT COMPLETE")
    print("=" * 60)
    print(f"Processed:   {len(paths_dicts)} paths")
    print(f"  Success:   {len(success_dicts)}")
    print(f"  Failure:   {len(failure_dicts)}")
    print(f"\nOutput: {output_dir}")
    print(f"\nNext step:")
    print(f"  python -m src.cli experiment --experiment rq1_1 --data_path {output_dir / f'{args.dataset}_all_paths.json'}")


# =============================================================================
# Experiments (Analysis — Phase 2)
# =============================================================================

def cmd_experiment(args):
    """Run Cliff Token Analysis experiments on pre-computed rollout data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = config.resolve_model_path(args.model)
    model_short = config.get_model_short_name(model_path)
    output_dir = Path(args.output_dir) / f"experiment_{model_short}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CLIFF TOKEN ANALYSIS EXPERIMENTS")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Data:       {args.data_path}")
    print(f"Output:     {output_dir}")
    print()

    print("Loading data...")
    all_paths = load_json(args.data_path)
    success_paths = [p for p in all_paths if p["is_correct"]]
    failure_paths = [p for p in all_paths if not p["is_correct"]]
    print(f"Loaded {len(all_paths)} paths (Success: {len(success_paths)}, Failure: {len(failure_paths)})")

    sample_path = all_paths[0] if all_paths else {}
    has_scores = len(sample_path.get("all_position_scores", [])) > 0
    if not has_scores:
        print("\nWARNING: No tokenwise potential found. Run 'rollout' first.")

    save_json(vars(args), str(output_dir / "experiment_config.json"), indent=2)

    VALID = ["rq1_1", "rq1_2", "rq1_3", "all"]
    if args.experiment not in VALID:
        print(f"Unknown experiment '{args.experiment}'. Valid: {VALID}")
        return

    # --- RQ1-1: Cliff token occurrence in success vs failure paths ---
    if args.experiment in ["all", "rq1_1"]:
        try:
            from src.analysis.curves import run_experiment1_analysis, plot_sample_curves
            exp_dir = output_dir / "rq1_1_cliff_occurrence"
            exp_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 60)
            print("RQ1-1: Cliff Token Occurrence (Success vs Failure)")
            print("=" * 60)
            run_experiment1_analysis(success_paths, failure_paths, str(exp_dir))
            plot_sample_curves(success_paths, failure_paths, str(exp_dir))
        except ImportError:
            print("\n[RQ1-1] src/analysis/curves.py not yet available (Phase 2).")

    # --- RQ1-2: Cliff token vs Critical token positional analysis ---
    if args.experiment in ["all", "rq1_2"]:
        try:
            from src.analysis.positional import (
                run_experiment2_analysis, print_experiment2_summary, create_all_visualizations
            )
            exp_dir = output_dir / "rq1_2_positional"
            exp_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 60)
            print("RQ1-2: Cliff Token vs Critical Token Positional Analysis")
            print("=" * 60)
            results = run_experiment2_analysis(failure_paths, str(exp_dir))
            print_experiment2_summary(results)
            create_all_visualizations(failure_paths, str(exp_dir))
        except ImportError:
            print("\n[RQ1-2] src/analysis/positional.py not yet available (Phase 2).")

    # --- RQ1-3: Cliff-del vs Critical-del decoding comparison ---
    if args.experiment in ["all", "rq1_3"]:
        try:
            from src.decoding.cliff import run_cliff_del_on_paths, cliff_del_results_to_dicts
            from src.decoding.critical import run_critical_del_on_paths, critical_del_results_to_dicts
            from src.decoding.evaluator import run_experiment3_evaluation, print_experiment3_summary
            exp_dir = output_dir / "rq1_3_cliff_vs_critical"
            exp_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 60)
            print("RQ1-3: Cliff-Del vs Critical-Del Decoding Comparison")
            print("=" * 60)
            mode = args.mode or config.get_default_mode(model_path)
            gpu_list = [int(g) for g in args.gpus.split(",")]
            llm = create_llm(model_path, gpu_list, config.GPU_MEMORY_UTILIZATION)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            print("Running Cliff-Del decoding...")
            cliff_results = run_cliff_del_on_paths(
                llm, tokenizer, all_paths, args.dataset,
                num_samples=args.num_regen_samples, mode=mode,
                drop_threshold=args.cliff_threshold,
            )
            cliff_dicts = cliff_del_results_to_dicts(cliff_results)
            save_json(cliff_dicts, str(exp_dir / "cliff_del_results.json"))

            print("Running Critical-Del decoding...")
            critical_results = run_critical_del_on_paths(
                llm, tokenizer, all_paths, args.dataset,
                num_samples=args.num_regen_samples, mode=mode,
            )
            critical_dicts = critical_del_results_to_dicts(critical_results)
            save_json(critical_dicts, str(exp_dir / "critical_del_results.json"))

            eval_results = run_experiment3_evaluation(
                critical_dicts, cliff_dicts, str(exp_dir)
            )
            print_experiment3_summary(eval_results)
        except ImportError:
            print("\n[RQ1-3] src/decoding/ modules not yet available (Phase 2).")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cliff Token Analysis (CTA) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  inference   Generate reasoning paths (Stage 1)
  rollout     Compute tokenwise potential (Stage 2)
  experiment  Run Cliff Token analysis experiments

Examples:
  # Stage 1: Generate reasoning paths
  python -m src.cli inference --model qwen3-4b --dataset math500 --num_problems 500 --gpus 0,1

  # Stage 2: Compute tokenwise potential
  python -m src.cli rollout --model qwen3-4b --dataset math500 \\
      --data_path ./output/.../math500_all_paths.json --gpus 0,1

  # Multi-GPU pipeline via script
  scripts/run_data.sh --model qwen3-4b --dataset math500 --temperature 0.6 --gpus 0,1,2,3
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -------------------------------------------------------------------------
    # Shared arguments helper
    # -------------------------------------------------------------------------
    def add_model_args(p):
        p.add_argument("--model", type=str, default="qwen3-4b",
                       help="Model alias or HuggingFace path (default: qwen3-4b)")
        p.add_argument("--mode", type=str, default=None,
                       choices=["thinking", "non_thinking"],
                       help="Reasoning mode (auto-detected from model if omitted)")
        p.add_argument("--temperature", type=float, default=None,
                       help="Override sampling temperature (default: from model config)")
        p.add_argument("--gpus", type=str, default="0",
                       help="Comma-separated GPU IDs (default: 0)")
        p.add_argument("--output_dir", type=str, default="./output")

    # -------------------------------------------------------------------------
    # inference
    # -------------------------------------------------------------------------
    inf_parser = subparsers.add_parser("inference", help="Generate reasoning paths")
    add_model_args(inf_parser)
    inf_parser.add_argument("--dataset", type=str, default=config.DATASET_NAME,
                            help="Dataset: gsm1k | math500 | aime24 | aime25")
    inf_parser.add_argument("--split", type=str, default=config.DEFAULT_SPLIT,
                            choices=["train", "test"])
    inf_parser.add_argument("--num_problems", type=int, default=500)
    inf_parser.add_argument("--paths_per_problem", type=int, default=1)
    inf_parser.add_argument("--dataset_path", type=str, default=None,
                            help="Override dataset file path (default: from config)")
    inf_parser.add_argument("--token_profile", type=str, default="default",
                            choices=config.TOKEN_PROFILE_CHOICES,
                            help="Token-limit profile: default (short) or paper (long)")
    inf_parser.add_argument("--adapter_path", type=str, default=None,
                            help="LoRA adapter directory. When set, vLLM serves the "
                                 "adapter natively via enable_lora + LoRARequest (no merge).")

    # -------------------------------------------------------------------------
    # rollout
    # -------------------------------------------------------------------------
    roll_parser = subparsers.add_parser("rollout", help="Compute tokenwise potential")
    add_model_args(roll_parser)
    roll_parser.add_argument("--data_path", type=str, required=True,
                             help="Path to inference output (*_all_paths.json)")
    roll_parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    roll_parser.add_argument("--rollout_samples", type=int, default=config.ROLLOUT_SAMPLES,
                             help=f"Rollout samples per token position (default: {config.ROLLOUT_SAMPLES})")
    roll_parser.add_argument("--rollout_window", type=int, default=config.ROLLOUT_WINDOW_SIZE,
                             help="Compute potential every N tokens (default: 1 = every token)")
    roll_parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    roll_parser.add_argument("--global_batch_size", type=int, default=config.GLOBAL_BATCH_SIZE)
    roll_parser.add_argument("--max_grading_workers", type=int, default=config.MAX_GRADING_WORKERS)
    roll_parser.add_argument("--no_optimized", action="store_true",
                             help="Disable global batching (use legacy per-path mode)")
    roll_parser.add_argument("--failure_only", action="store_true",
                             help="Only compute potential for failure paths")
    roll_parser.add_argument("--early_termination_k", type=int, default=config.EARLY_TERMINATION_K,
                             help=f"Stop path after K consecutive score=0.0 (default: {config.EARLY_TERMINATION_K}, 0=disabled)")
    roll_parser.add_argument("--adapter_path", type=str, default=None,
                             help="LoRA adapter directory. When set, vLLM serves the "
                                  "adapter natively via enable_lora + LoRARequest (no merge).")

    # -------------------------------------------------------------------------
    # experiment
    # -------------------------------------------------------------------------
    exp_parser = subparsers.add_parser("experiment", help="Run Cliff Token analysis experiments")
    add_model_args(exp_parser)
    exp_parser.add_argument("--experiment", type=str, default="rq1_1",
                            choices=["all", "rq1_1", "rq1_2", "rq1_3"],
                            help="Which experiment to run")
    exp_parser.add_argument("--data_path", type=str, required=True,
                            help="Path to rollout output (*_all_paths.json with potential)")
    exp_parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    exp_parser.add_argument("--cliff_threshold", type=float, default=config.DEFAULT_CLIFF_THRESHOLD,
                            help=f"Cliff token threshold (default: {config.DEFAULT_CLIFF_THRESHOLD})")
    exp_parser.add_argument("--num_regen_samples", type=int, default=32,
                            help="Regeneration samples for RQ1-3 decoding comparison")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    _init_heavy_imports()

    if args.command == "inference":
        cmd_inference(args)
    elif args.command == "rollout":
        cmd_rollout(args)
    elif args.command == "experiment":
        cmd_experiment(args)


if __name__ == "__main__":
    main()
