"""RQ1-3 entry point.

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

    # =============================================
    # Exp 1: Cliff-del vs Cliff-keep (all paths)
    # =============================================
    print("=" * 60)
    print("Exp 1: Cliff-del vs Cliff-keep")
    print("=" * 60)

    from src.decoding.cliff import run_cliff_on_paths, cliff_results_to_dicts
    cliff_results = run_cliff_on_paths(
        llm, tokenizer, all_paths, dataset,
        num_samples=num_samples, mode=mode, model_path=model_path,
    )
    cliff_dicts = cliff_results_to_dicts(cliff_results)

    exp1_dir = os.path.join(output_dir, "sub_exp_1")
    os.makedirs(exp1_dir, exist_ok=True)
    save_json(cliff_dicts, os.path.join(exp1_dir, "cliff_results.json"))

    from src.decoding.evaluator import run_experiment1_evaluation, print_experiment3_summary
    exp1_result = run_experiment1_evaluation(cliff_dicts, exp1_dir)

    # =============================================
    # Exp 2: 4-method comparison across 4 populations × cliff selectors
    # =============================================
    print("\n" + "=" * 60)
    print("Exp 2: Cliff/Critical/Tangent/Random-del across 4 variants")
    print("=" * 60)

    import random as _rnd
    from collections import defaultdict
    from src.analysis.detector import find_critical_token, find_all_cliff_tokens_statistical
    from src.analysis.positional import _detect_tangents
    from src.decoding.critical import run_critical_del_on_paths, critical_del_results_to_dicts
    from src.decoding.tangent import run_tangent_del_on_paths, tangent_del_results_to_dicts
    from src.decoding.random_del import run_random_del_on_paths, random_del_results_to_dicts
    from src.decoding.evaluator import run_experiment2_evaluation

    exp2_dir = os.path.join(output_dir, "sub_exp_2")
    os.makedirs(exp2_dir, exist_ok=True)

    # Group sub_exp_1 cliff_dicts by path_id (sorted by position) — REUSED for cliff-del
    cliffs_by_path = defaultdict(list)
    for c in cliff_dicts:
        cliffs_by_path[c["path_id"]].append(c)
    for k in cliffs_by_path:
        cliffs_by_path[k].sort(key=lambda x: x["cliff_position"])

    # Population A: tangent-eligible (failure ∧ cliff ∧ critical ∧ tangent)
    eligible_tangent = []
    for p in failure_paths:
        if not cliffs_by_path.get(p["id"]):
            continue
        scores = p.get("all_position_scores", [])
        if not find_critical_token(scores):
            continue
        cliffs_all_obj = find_all_cliff_tokens_statistical(scores)
        _, _, chunks, _ = _detect_tangents(scores, [c.position for c in cliffs_all_obj])
        if any(c.is_tangent for c in chunks):
            eligible_tangent.append(p)

    # Population B: failure paths with at least one cliff
    failure_with_cliff = [p for p in failure_paths if cliffs_by_path.get(p["id"])]

    # Population C: all failure paths
    all_failure = list(failure_paths)

    print(f"  eligible_tangent:   {len(eligible_tangent)}")
    print(f"  failure_with_cliff: {len(failure_with_cliff)}")
    print(f"  all_failure:        {len(all_failure)}")

    if not all_failure:
        print("  No failure paths — skipping Exp 2.")
        exp2_result = None
    else:
        # Run critical/tangent/random ONCE on broadest population
        print("\n  [GPU] critical-del on all_failure...")
        critical_dicts_all = critical_del_results_to_dicts(
            run_critical_del_on_paths(llm, tokenizer, all_failure, dataset,
                                       num_samples=num_samples, mode=mode, model_path=model_path)
        )
        print("  [GPU] tangent-del on all_failure...")
        tangent_dicts_all = tangent_del_results_to_dicts(
            run_tangent_del_on_paths(llm, tokenizer, all_failure, dataset,
                                      num_samples=num_samples, mode=mode, model_path=model_path)
        )
        print("  [GPU] random-del on all_failure...")
        random_dicts_all = random_del_results_to_dicts(
            run_random_del_on_paths(llm, tokenizer, all_failure, dataset,
                                     num_samples=num_samples, mode=mode, model_path=model_path)
        )

        def fill_zero_stubs(method_results, paths):
            have = {r["path_id"] for r in method_results}
            out = list(method_results)
            for p in paths:
                if p["id"] not in have:
                    out.append({
                        "path_id": p["id"],
                        "del_num_correct": 0,
                        "num_samples": num_samples,
                        "_stub": True,
                    })
            return out

        def cliff_first(pid):
            return cliffs_by_path[pid][0] if cliffs_by_path.get(pid) else None

        _rng = _rnd.Random(42)
        def cliff_random(pid):
            return _rng.choice(cliffs_by_path[pid]) if cliffs_by_path.get(pid) else None

        def build_evaluation(paths, cliff_picker, variant):
            n_total = len(paths)
            pids = {p["id"] for p in paths}

            cliff_picks = []
            for p in paths:
                c = cliff_picker(p["id"])
                if c is not None:
                    cliff_picks.append(c)
                else:
                    cliff_picks.append({
                        "path_id": p["id"], "del_num_correct": 0,
                        "num_samples": num_samples, "_stub": True,
                    })

            crit_filtered = [r for r in critical_dicts_all if r.get("ct_found") and r["path_id"] in pids]
            tang_filtered = [r for r in tangent_dicts_all if r.get("tangent_found") and r["path_id"] in pids]
            rand_filtered = [r for r in random_dicts_all if r["path_id"] in pids]

            crit_with_stubs = fill_zero_stubs(crit_filtered, paths)
            tang_with_stubs = fill_zero_stubs(tang_filtered, paths)
            rand_with_stubs = fill_zero_stubs(rand_filtered, paths)

            save_json(cliff_picks,     os.path.join(exp2_dir, f"cliff_del_results_{variant}.json"))
            save_json(crit_with_stubs, os.path.join(exp2_dir, f"critical_del_results_{variant}.json"))
            save_json(tang_with_stubs, os.path.join(exp2_dir, f"tangent_del_results_{variant}.json"))
            save_json(rand_with_stubs, os.path.join(exp2_dir, f"random_del_results_{variant}.json"))

            upper_bounds = {
                "Cliff-del":    sum(1 for r in cliff_picks if not r.get("_stub")) / n_total,
                "Critical-del": len(crit_filtered) / n_total,
                "Tangent-del":  len(tang_filtered) / n_total,
                "Random-del":   len(rand_filtered) / n_total,
            }
            print(f"\n  [{variant}] n={n_total}  upper_bounds={ {k: round(v, 3) for k, v in upper_bounds.items()} }")
            return run_experiment2_evaluation(
                cliff_picks, crit_with_stubs, tang_with_stubs, rand_with_stubs,
                exp2_dir, suffix=f"_{variant}",
                title_extra=f"({variant}, n={n_total})",
                upper_bounds=upper_bounds,
            )

        exp2_result = None
        if eligible_tangent:
            exp2_result = build_evaluation(eligible_tangent, cliff_first, "population_tangent")
        if failure_with_cliff:
            build_evaluation(failure_with_cliff, cliff_first, "cliff_first")
            build_evaluation(failure_with_cliff, cliff_random, "cliff_random")
        build_evaluation(all_failure, cliff_first, "all_failure")

    # =============================================
    # Exp 3: Semantic Analysis
    # =============================================
    print("\n" + "=" * 60)
    print("Exp 3: Semantic Analysis")
    print("=" * 60)

    from src.decoding.evaluator import extract_cliff_contexts, generate_semantic_analysis
    contexts = extract_cliff_contexts(all_paths, cliff_dicts)
    exp3_dir = os.path.join(output_dir, "sub_exp_3_semantic")
    generate_semantic_analysis(contexts, exp3_dir)

    # Summary
    print_experiment3_summary(exp1_result, exp2_result)

    print(f"\nAll results saved to: {output_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip Python's normal shutdown to bypass vLLM teardown hang.
    # (vLLM V1 in-process mode leaves worker threads stuck in futex_wait,
    # preventing the interpreter from exiting. All output files are already
    # fsynced via `with open() as f` context managers above.)
    os._exit(0)
