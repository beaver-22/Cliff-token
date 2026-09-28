#!/usr/bin/env python
"""Cliff-token probability before / after Cliff-DPO (Figure 5, right).

Scores a frozen held-out cliff set with the base model and every Cliff-DPO
adapter (variants x seeds) trained by scripts/run_dpo_train.sh.

    python scripts/run_cliff_prob_shift.py --model ./model/Qwen3-0.6B --gpu 0

Outputs (under <out>/<model_short>/):
    cliff_set.json          frozen held-out cliffs (built once, then reused)
    probs/base.jsonl        per-cliff P(cliff token) under the base model
    probs/<label>.jsonl     same, per adapter (label = cliff_<variant>_only_seed<seed>)

Aggregate with scripts/_cliff_prob_shift_aggregate.py.
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402


def load_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def verify_adapter(path: str, stem: str, seed: int) -> None:
    """Confirm the adapter was trained on this variant's pairs with this seed."""
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(f"No LoRA adapter in {path}")
    with open(os.path.join(path, "train_config.json")) as f:
        cfg = json.load(f)
    if f"{stem}_" not in os.path.basename(cfg.get("dataset_path", "")):
        raise AssertionError(f"{path}: expected a '{stem}_*' pair file, got {cfg.get('dataset_path')}")
    if int(cfg.get("seed", -1)) != seed:
        raise AssertionError(f"{path}: expected seed {seed}, train_config.json says {cfg.get('seed')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model alias or path")
    ap.add_argument("--dataset", default="gsm8k_train", help="training dataset name used in run_dpo_train.sh")
    ap.add_argument("--training_dir", default=None,
                    help="default: ./output/07_cliff_dpo/03_training/<model_short>")
    ap.add_argument("--variants", default="deterministic,uncertainty,sampled_off")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--eval_datasets", default="gsm1k_100,math500_100,aime25")
    ap.add_argument("--rollout_dir", default="./output/03_rollouts")
    ap.add_argument("--token_stats_dir", default="./output/02_token_stats")
    ap.add_argument("--out", default="./output/07_cliff_dpo/05_cliff_prob_shift")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.analysis.cliff_prob_shift import (
        build_eval_cliff_set,
        cliff_set_summary,
        score_token_probs_hf,
    )

    model_path = config.resolve_model_path(args.model)
    short = config.get_model_short_name(model_path)
    training_dir = args.training_dir or f"./output/07_cliff_dpo/03_training/{short}"
    out_dir = os.path.join(args.out, short)
    os.makedirs(out_dir, exist_ok=True)

    passes = [("base", None)]
    for seed in [int(s) for s in args.seeds.split(",")]:
        for variant in args.variants.split(","):
            stem = f"cliff_{variant}_only"
            path = os.path.join(training_dir, args.dataset, f"{stem}_seed{seed}")
            verify_adapter(path, stem, seed)
            passes.append((f"{stem}_seed{seed}", path))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    cliff_set_path = os.path.join(out_dir, "cliff_set.json")
    if os.path.exists(cliff_set_path):
        with open(cliff_set_path) as f:
            cliffs = json.load(f)
        print(f"Reusing frozen cliff set: {cliff_set_path} ({len(cliffs)} cliffs)")
    else:
        print("Building cliff set from held-out rollouts...")
        cliffs = build_eval_cliff_set(
            short, tokenizer,
            datasets=args.eval_datasets.split(","),
            rollout_dir=args.rollout_dir,
            token_stats_dir=args.token_stats_dir,
        )
        with open(cliff_set_path, "w") as f:
            json.dump(cliffs, f)
        print(f"Wrote {cliff_set_path}")
    print(f"  {json.dumps(cliff_set_summary(cliffs))}")

    print(f"\nLoading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    for label, path in passes:
        if path is None:
            continue
        if isinstance(model, PeftModel):
            model.load_adapter(os.path.abspath(path), adapter_name=label)
        else:
            model = PeftModel.from_pretrained(model, os.path.abspath(path), adapter_name=label)
    print(f"  loaded {len(passes) - 1} LoRA adapters onto one base copy")

    for i, (label, path) in enumerate(passes):
        out_path = os.path.join(out_dir, "probs", f"{label}.jsonl")
        print(f"\n[{i + 1}/{len(passes)}] {label}")
        if args.skip_existing and len(load_jsonl(out_path)) >= len(cliffs):
            print("  already complete, skipping")
            continue
        if path is None:
            ctx = model.disable_adapter() if isinstance(model, PeftModel) else contextlib.nullcontext()
            with ctx:
                rows = score_token_probs_hf(model, cliffs)
        else:
            model.set_adapter(label)
            rows = score_token_probs_hf(model, cliffs)
        write_jsonl(out_path, rows)
        print(f"  wrote {out_path}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
