#!/usr/bin/env python3
"""Cliff taxonomy distribution (Table 1). CPU only.

Cliff types (entropy boundary H_b(0.99) ~= 0.0561 nats):
  deterministic   greedy token,     H_t <= H_b(0.99)
  uncertain       greedy token,     H_t >  H_b(0.99)
  sampled-off     non-greedy token, H_t >  H_b(0.99)
  (non-greedy with H_t <= H_b(0.99) is counted but not reported)

Inputs:
  cliff tokens   <output_dir>/runs/<model>_<dataset>/cliff_logprobs.json
                 greedy status = is_cliff_eq_greedy, entropy = entropy_at_t
  all tokens     <baseline_dir>/<model>/<dataset>_all_paths.json
                 greedy status = (response_token_ranks == 1),
                 entropy = response_token_entropies

Outputs:
  <output_dir>/cliff_type_distribution.csv

Usage:
    python3 scripts/_exp3_cliff_type_table.py <output_dir> \
        [--baseline_dir output/02_token_stats] \
        [--datasets gsm1k_100,math500_100,aime25]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = [
    "Qwen3-8B",
    "Qwen3-4B",
    "Qwen3-0.6B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct",
    "gemma-3-4b-it",
]
MODEL_DISPLAY = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "gemma-3-4b-it": "Gemma-3-4B",
}
DEFAULT_DATASETS = ["gsm1k_100", "math500_100", "aime25"]
CATEGORIES = [
    ("deterministic", "low_greedy"),
    ("uncertain", "high_greedy"),
    ("sampled_off", "high_non_greedy"),
    ("low_entropy_non_greedy", "low_non_greedy"),
]
GREEDY_PROB_THRESHOLD = 0.99


def binary_entropy_nats(p: float) -> float:
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def classify(entropy: float, is_greedy: bool, boundary: float) -> str:
    if entropy <= boundary:
        return "low_greedy" if is_greedy else "low_non_greedy"
    return "high_greedy" if is_greedy else "high_non_greedy"


def load_cliffs(runs_dir: Path, datasets: list[str]) -> dict[str, list[tuple[float, bool]]]:
    by_model: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        cliffs_path = run_dir / "cliff_logprobs.json"
        if not config_path.exists() or not cliffs_path.exists():
            continue
        config = json.loads(config_path.read_text())
        model = config.get("model_short") or config.get("model")
        dataset = config.get("dataset")
        if model not in MODEL_ORDER or dataset not in datasets:
            continue
        for cliff in json.loads(cliffs_path.read_text()):
            by_model[model].append(
                (float(cliff["entropy_at_t"]), bool(cliff.get("is_cliff_eq_greedy")))
            )
    return by_model


def load_baseline(baseline_dir: Path, datasets: list[str]) -> dict[str, list[tuple[float, bool]]]:
    by_model: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for model in MODEL_ORDER:
        for dataset in datasets:
            source = baseline_dir / model / f"{dataset}_all_paths.json"
            if not source.exists():
                continue
            for path in json.loads(source.read_text()):
                entropies = path.get("response_token_entropies", [])
                ranks = path.get("response_token_ranks", [])
                if len(entropies) != len(ranks):
                    raise ValueError(
                        f"Unpaired baseline arrays in {source}: "
                        f"{len(entropies)} entropies vs {len(ranks)} ranks"
                    )
                by_model[model].extend(
                    (float(entropy), int(rank) == 1)
                    for entropy, rank in zip(entropies, ranks)
                )
    return by_model


def counts_for(rows, boundary: float) -> dict[str, int]:
    counts = {key: 0 for _, key in CATEGORIES}
    for entropy, is_greedy in rows:
        counts[classify(entropy, is_greedy, boundary)] += 1
    return counts


def pct(count: int, total: int) -> float:
    return count / total * 100.0 if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--runs_dir", type=Path, default=None)
    parser.add_argument("--baseline_dir", type=Path, default=Path("output/02_token_stats"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    runs_dir: Path = args.runs_dir or output_dir / "runs"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    cliffs = load_cliffs(runs_dir, datasets)
    baseline = load_baseline(args.baseline_dir, datasets)
    models = [m for m in MODEL_ORDER if cliffs.get(m) and baseline.get(m)]
    skipped = [m for m in MODEL_ORDER if m not in models]
    if not models:
        sys.exit("ERROR: no model has both cliff and baseline data")
    if skipped:
        print(f"Skipping (missing cliff or baseline data): {', '.join(skipped)}")

    boundary = binary_entropy_nats(GREEDY_PROB_THRESHOLD)
    csv_path = output_dir / "cliff_type_distribution.csv"
    fields = ["model", "category", "cliff_count", "cliff_total", "cliff_pct",
              "all_token_count", "all_token_total", "all_token_pct"]

    print(f"Entropy boundary H_b({GREEDY_PROB_THRESHOLD}) = {boundary:.4f} nats\n")
    print(f"{'Model':<16} {'Cliff D':>8} {'U':>6} {'S':>6}   {'All D':>6} {'U':>6} {'S':>6}")
    print("-" * 62)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            c = counts_for(cliffs[model], boundary)
            b = counts_for(baseline[model], boundary)
            n_c, n_b = sum(c.values()), sum(b.values())
            for label, key in CATEGORIES:
                writer.writerow({
                    "model": model,
                    "category": label,
                    "cliff_count": c[key],
                    "cliff_total": n_c,
                    "cliff_pct": f"{pct(c[key], n_c):.4f}",
                    "all_token_count": b[key],
                    "all_token_total": n_b,
                    "all_token_pct": f"{pct(b[key], n_b):.4f}",
                })
            row = [pct(c[k], n_c) for k in ("low_greedy", "high_greedy", "high_non_greedy")]
            row += [pct(b[k], n_b) for k in ("low_greedy", "high_greedy", "high_non_greedy")]
            print(f"{MODEL_DISPLAY.get(model, model):<16} "
                  f"{row[0]:>8.1f} {row[1]:>6.1f} {row[2]:>6.1f}   "
                  f"{row[3]:>6.1f} {row[4]:>6.1f} {row[5]:>6.1f}")

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
