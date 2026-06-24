"""RQ2-1 single (model, dataset) runner.

Phase B: cliff logprobs (rank, entropy, greedy token).
Phase C (optional): greedy replacement rollout (n=64).

Args:
    1: model_alias (e.g. qwen3-8b)
    2: dataset (e.g. math500)
    3: rollout_data path
    4: gpu id (string)
    5: output_dir
    6: num_samples (Phase C)
    7: run_phase_c ("1" or "0")
"""
import sys
import os
import json

sys.path.insert(0, ".")

if len(sys.argv) != 8:
    print(f"Usage: python3 {sys.argv[0]} model dataset rollout_data gpu output_dir num_samples run_phase_c")
    sys.exit(1)

model_alias = sys.argv[1]
dataset = sys.argv[2]
rollout_data = sys.argv[3]
gpu = sys.argv[4]
output_dir = sys.argv[5]
num_samples = int(sys.argv[6])
run_phase_c = sys.argv[7] == "1"

os.makedirs(output_dir, exist_ok=True)

from src import config
from src.cli import _init_heavy_imports
_init_heavy_imports()
from src.cli import create_llm
from transformers import AutoTokenizer
from src.decoding.greedy_replace import (
    extract_cliff_logprobs_and_greedy,
    run_greedy_replacement_rollout,
    greedy_replace_results_to_dicts,
)

model_path = config.resolve_model_path(model_alias)
mode = config.get_default_mode(model_path)
model_short = os.path.basename(model_path.rstrip('/'))

print(f"============================================================")
print(f"RQ2-1: {model_alias} / {dataset}")
print(f"  rollout_data: {rollout_data}")
print(f"  output_dir:   {output_dir}")
print(f"  phase_c:      {run_phase_c}")
print(f"============================================================")

# vLLM load (memory_utilization=0.65 to leave room for prompt_logprobs workspace)
print("\nLoading model (gpu_memory_utilization=0.65)...")
llm = create_llm(model_path, [0], memory_utilization=0.65)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded.\n")

# Load rollout data
target_paths = json.load(open(rollout_data))
print(f"Loaded {len(target_paths)} paths from {rollout_data}")

# Phase B
cliff_info = extract_cliff_logprobs_and_greedy(
    llm, tokenizer, target_paths,
    model_name=model_short, dataset_name=dataset,
    drop_threshold=config.DEFAULT_CLIFF_THRESHOLD, top_k=20,
)

cliff_dicts = [
    {k: v for k, v in vars(c).items() if not k.startswith("_")}
    for c in cliff_info
]
out_b = os.path.join(output_dir, "cliff_logprobs.json")
with open(out_b, "w") as f:
    json.dump(cliff_dicts, f, indent=2)
print(f"\n  Saved {out_b}: {len(cliff_dicts)} cliffs")

with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump({
        "model": model_alias,
        "model_short": model_short,
        "dataset": dataset,
        "rollout_data": rollout_data,
        "num_samples": num_samples,
        "phase_c": run_phase_c,
        "n_cliffs": len(cliff_dicts),
    }, f, indent=2)

# Phase C (Qwen3-8B only by default)
if run_phase_c and cliff_info:
    print(f"\n[Phase C] greedy replacement rollout (n={num_samples})")
    results = run_greedy_replacement_rollout(
        llm, tokenizer, cliff_info, target_paths,
        model_name=model_short, dataset_name=dataset,
        num_samples=num_samples, mode=mode, model_path=model_path,
    )
    result_dicts = greedy_replace_results_to_dicts(results)
    out_c = os.path.join(output_dir, "greedy_replace_results.json")
    with open(out_c, "w") as f:
        json.dump(result_dicts, f, indent=2)
    print(f"  Saved {out_c}: {len(result_dicts)} cliffs")

print("\n=== Done ===")
sys.stdout.flush()
sys.stderr.flush()
# Skip Python's normal shutdown to bypass vLLM teardown hang.
# (vLLM V1 in-process mode leaves ~500 worker threads stuck in futex_wait,
# preventing the interpreter from exiting. All output files are already
# fsynced via `with open() as f` context managers above.)
os._exit(0)
