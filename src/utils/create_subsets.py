#!/usr/bin/env python3
"""
Dataset Subset Creator for Cliff Token Analysis

Creates reproducible subsets with a fixed seed.

Subsets:
  GSM1K   → 100 problems, random sampling (seed=42)
  MATH500 → 100 problems, proportional stratified sampling by level (seed=42)
            preserving original proportions: L1:8, L2:18, L3:21, L4:26, L5:27 = 100
  AIME25  → no subset file; the paper uses the full 30-problem aime25.jsonl

Usage:
  python create_subsets.py                          # create all subsets
  python create_subsets.py --dataset gsm1k          # GSM1K only
  python create_subsets.py --dataset math500        # MATH-500 only
  python create_subsets.py --seed 42                # specify seed (default: 42)
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def create_gsm1k_subset(input_dir: Path, output_dir: Path, seed: int = 42, num_samples: int = 100):
    """Randomly sample num_samples problems from GSM1K."""
    input_path = input_dir / "GSM1K_test.jsonl"
    output_path = output_dir / "GSM1K_test_100.jsonl"

    problems = load_jsonl(input_path)
    print(f"GSM1K: {len(problems)} problems loaded")

    random.seed(seed)
    subset = random.sample(problems, num_samples)

    save_jsonl(subset, output_path)
    print(f"  → {len(subset)} problems saved: {output_path}")
    return subset



# MATH-500 proportional stratified sampling allocation
# Original: L1:43, L2:90, L3:105, L4:128, L5:134 (total 500)
# Allocate 100 problems while preserving proportions (largest remainder method):
#   L1: 43/500*100=8.6  → 8
#   L2: 90/500*100=18.0 → 18
#   L3: 105/500*100=21.0 → 21
#   L4: 128/500*100=25.6 → 26
#   L5: 134/500*100=26.8 → 27
#   Total: 100
MATH_SAMPLES_PER_LEVEL = {1: 8, 2: 18, 3: 21, 4: 26, 5: 27}


def create_math500_subset(input_dir: Path, output_dir: Path, seed: int = 42):
    """Proportional stratified sampling from MATH-500 while preserving original level ratios."""
    input_path = input_dir / "MATH_test.jsonl"
    output_path = output_dir / "MATH_test_100.jsonl"

    problems = load_jsonl(input_path)
    print(f"MATH-500: {len(problems)} problems loaded")

    # Group by level
    by_level = defaultdict(list)
    for p in problems:
        by_level[p["level"]].append(p)

    # Level-proportional sampling
    subset = []
    for level in sorted(by_level.keys()):
        level_problems = by_level[level]
        num_samples = MATH_SAMPLES_PER_LEVEL[level]
        random.seed(seed)
        sampled = random.sample(level_problems, num_samples)
        subset.extend(sampled)
        print(f"  Level {level}: sampled {num_samples} from {len(level_problems)} problems")

    save_jsonl(subset, output_path)
    print(f"  → Total {len(subset)} problems saved: {output_path}")
    return subset


SUBSET_FUNCS = {
    "gsm1k": create_gsm1k_subset,
    "math500": create_math500_subset,
}


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible subsets for Cliff Token Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subsets:
  gsm1k   → GSM1K_test_100.jsonl   (100 problems, random sampling)
  math500 → MATH_test_100.jsonl    (100 problems, proportional by level)

AIME25 uses the downloaded aime25.jsonl directly (full 30 problems).

Examples:
  python create_subsets.py
  python create_subsets.py --dataset gsm1k
  python create_subsets.py --seed 42
        """,
    )
    parser.add_argument(
        "--dataset", nargs="+",
        choices=list(SUBSET_FUNCS.keys()),
        default=list(SUBSET_FUNCS.keys()),
        help="Dataset(s) to create subsets for (default: all)",
    )
    parser.add_argument(
        "--input_dir", type=str, default="./data/input",
        help="Input directory (default: ./data/input)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/input",
        help="Output directory (default: ./data/input)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Seed:   {args.seed}")
    print()

    for name in args.dataset:
        print(f"{'=' * 50}")
        print(f"Subset: {name.upper()}")
        print(f"{'=' * 50}")
        SUBSET_FUNCS[name](input_dir, output_dir, seed=args.seed)  # noqa: all funcs accept (input_dir, output_dir, seed)
        print()

    print("=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
