"""RQ1 Cliff-del / Cliff-keep entry point (one model x dataset run).

Invoked by scripts/run_exp1_deletion.sh as a real .py file (not a heredoc) so
that multiprocessing's spawn context can re-import __main__ without hitting
FileNotFoundError on '<stdin>'. All work is guarded under __main__ so spawned
grading workers (which re-import this module) skip top-level execution.
"""
import sys, json, os
sys.path.insert(0, ".")


if __name__ == "__main__":
    model_alias = sys.argv[1]
    dataset = sys.argv[2]
    data_path = sys.argv[3]
    gpu_list = sys.argv[4]
    num_samples = int(sys.argv[5])
    output_dir = sys.argv[6]

    os.makedirs(output_dir, exist_ok=True)

    import src.config as config
    from src.cli import load_json, save_json, _init_heavy_imports
    _init_heavy_imports()
    from src.cli import create_llm

    model_path = config.resolve_model_path(model_alias)
    mode = config.get_default_mode(model_path)

    print("Loading data...")
    all_paths = load_json(data_path)
    success_paths = [p for p in all_paths if p.get("is_correct")]
    failure_paths = [p for p in all_paths if not p.get("is_correct")]
    print(f"  Total: {len(all_paths)}, Success: {len(success_paths)}, Failure: {len(failure_paths)}")

    print("\nLoading model...")
    gpu_ids = [int(g) for g in gpu_list.split(",")]
    from transformers import AutoTokenizer
    llm = create_llm(model_path, gpu_ids, config.GPU_MEMORY_UTILIZATION)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.\n")

    # Save config
    save_json({
        "model": model_alias, "model_path": model_path, "dataset": dataset,
        "data_path": data_path, "num_samples": num_samples, "mode": mode,
    }, os.path.join(output_dir, "experiment_config.json"), indent=2)

    print("=" * 60)
    print("Cliff-del vs Cliff-keep")
    print("=" * 60)

    from src.decoding.cliff import run_cliff_on_paths, cliff_results_to_dicts
    cliff_results = run_cliff_on_paths(
        llm, tokenizer, all_paths, dataset,
        num_samples=num_samples, mode=mode, model_path=model_path,
    )
    cliff_dicts = cliff_results_to_dicts(cliff_results)

    exp1_dir = os.path.join(output_dir, "cliff_del_keep")
    os.makedirs(exp1_dir, exist_ok=True)
    save_json(cliff_dicts, os.path.join(exp1_dir, "cliff_results.json"))

    from src.decoding.evaluator import evaluate_cliff_del_keep, print_cliff_del_keep_summary
    exp1_result = evaluate_cliff_del_keep(cliff_dicts, exp1_dir)

    print_cliff_del_keep_summary(exp1_result)

    print(f"\nAll results saved to: {output_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip Python's normal shutdown to bypass vLLM teardown hang.
    # (vLLM V1 in-process mode leaves worker threads stuck in futex_wait,
    # preventing the interpreter from exiting. All output files are already
    # fsynced via `with open() as f` context managers above.)
    os._exit(0)
