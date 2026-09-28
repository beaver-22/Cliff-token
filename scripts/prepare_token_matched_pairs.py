#!/usr/bin/env python3
"""Token-matched Cliff-DPO pairs (Table 21).

Each Cliff-DPO pair updates exactly two tokens (chosen / rejected at the cliff
position). The uncertain and sampled-off pair sets are subsampled without
replacement to the size of the deterministic set, so every cliff type trains
on the same number of updated tokens. The deterministic set is used in full,
so its token-matched run is the regular deterministic run from
``run_dpo_train.sh`` (no separate adapter is trained).

Reads  (from --pairs_dir, written by src.dpo.build_dpo_pairs):
    cliff_deterministic_only_<dataset>.json
    cliff_uncertainty_only_<dataset>.json
    cliff_sampled_off_only_<dataset>.json
Writes (to --pairs_dir):
    cliff_uncertainty_only_token_matched_<dataset>.json
    cliff_sampled_off_only_token_matched_<dataset>.json
    token_matched_<dataset>.meta.json

Sampling seeds: uncertain = --seed, sampled-off = --seed + 1 (default 42 / 43).

Usage:
    python3 scripts/prepare_token_matched_pairs.py \
        --pairs_dir output/07_cliff_dpo/02_pairs/Llama-3.1-8B-Instruct \
        --dataset gsm8k_train
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SOURCES = {
    "deterministic": ("cliff_deterministic_only", "deterministic"),
    "uncertainty": ("cliff_uncertainty_only", "uncertain"),
    "sampled_off": ("cliff_sampled_off_only", "sampled_off"),
}
SUBSAMPLED = ("uncertainty", "sampled_off")


def load_pairs(path: Path, expected_category: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        pairs = json.load(handle)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    wrong = {p.get("category") for p in pairs if p.get("category") != expected_category}
    if wrong:
        raise ValueError(f"{path}: expected only category {expected_category}, found {sorted(wrong)}")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_dir", type=Path, required=True)
    parser.add_argument("--dataset", default="gsm8k_train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = {}
    for variant, (stem, category) in SOURCES.items():
        path = args.pairs_dir / f"{stem}_{args.dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pair file: {path}")
        sources[variant] = load_pairs(path, category)

    target = len(sources["deterministic"])
    outputs: dict[str, str] = {}
    selected_indices: dict[str, list[int]] = {}
    for offset, variant in enumerate(SUBSAMPLED):
        population = sources[variant]
        if len(population) < target:
            raise ValueError(f"{variant} has only {len(population)} pairs; need {target}")
        rng = random.Random(args.seed + offset)
        indices = sorted(rng.sample(range(len(population)), target))
        selected_indices[variant] = indices
        out_path = args.pairs_dir / f"{SOURCES[variant][0]}_token_matched_{args.dataset}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump([population[i] for i in indices], handle, ensure_ascii=False)
            handle.write("\n")
        outputs[variant] = str(out_path)

    meta = {
        "dataset": args.dataset,
        "pairs_per_variant": target,
        "updated_tokens_per_pair": 2,
        "source_counts": {k: len(v) for k, v in sources.items()},
        "sampling_seed": {"uncertainty": args.seed, "sampled_off": args.seed + 1},
        "output_files": outputs,
        "selected_indices": selected_indices,
    }
    meta_path = args.pairs_dir / f"token_matched_{args.dataset}.meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print(f"pairs per variant: {target:,} (= deterministic set)")
    for variant in SOURCES:
        print(f"  {variant:<13} source={len(sources[variant]):,}")
    for variant, path in outputs.items():
        print(f"  -> {path}")
    print(f"  -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
