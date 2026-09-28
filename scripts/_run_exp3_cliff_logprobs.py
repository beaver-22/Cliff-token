"""RQ2 single (model, dataset) runner: cliff-token logprobs for the taxonomy.

For every cliff token, records the greedy token, the cliff token's tie-aware
rank and probability, and the token entropy at the cliff position.

Args:
    1: model_alias (e.g. qwen3-8b)
    2: dataset (e.g. math500_100)
    3: rollout_data path (*_all_paths.json with all_position_scores)
    4: gpu id (string)
    5: output_dir
"""
import sys
import os
import json

sys.path.insert(0, ".")

if len(sys.argv) != 6:
    print(f"Usage: python3 {sys.argv[0]} model dataset rollout_data gpu output_dir")
    sys.exit(1)

model_alias = sys.argv[1]
dataset = sys.argv[2]
rollout_data = sys.argv[3]
gpu = sys.argv[4]
output_dir = sys.argv[5]

os.makedirs(output_dir, exist_ok=True)

from src import config
from src.cli import _init_heavy_imports
_init_heavy_imports()
from src.cli import create_llm
from transformers import AutoTokenizer
from src.analysis.cliff_logprobs import extract_cliff_logprobs_and_greedy

model_path = config.resolve_model_path(model_alias)
model_short = os.path.basename(model_path.rstrip('/'))

print(f"============================================================")
print(f"RQ2 cliff logprobs: {model_alias} / {dataset}")
print(f"  rollout_data: {rollout_data}")
print(f"  output_dir:   {output_dir}")
print(f"============================================================")

# vLLM load (memory_utilization=0.65 to leave room for the logprobs workspace)
print("\nLoading model (gpu_memory_utilization=0.65)...")
llm = create_llm(model_path, [0], memory_utilization=0.65)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded.\n")

target_paths = json.load(open(rollout_data))
print(f"Loaded {len(target_paths)} paths from {rollout_data}")

cliff_info = extract_cliff_logprobs_and_greedy(
    llm, tokenizer, target_paths,
    model_name=model_short, dataset_name=dataset, top_k=20,
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
        "n_cliffs": len(cliff_dicts),
    }, f, indent=2)

print("\n=== Done ===")
sys.stdout.flush()
sys.stderr.flush()
# Skip Python's normal shutdown to bypass vLLM teardown hang.
# (vLLM V1 in-process mode leaves worker threads stuck in futex_wait,
# preventing the interpreter from exiting. All output files are already
# fsynced via `with open() as f` context managers above.)
os._exit(0)
