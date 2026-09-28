#!/usr/bin/env python
"""Aggregate cliff-token probability shift (Figure 5, right). CPU only.

Each Cliff-DPO variant is evaluated on held-out cliffs of its own cliff type.
For every (model, variant, seed), the mean P(cliff token) over those cliffs is
computed under the base model and under the adapter; seeds are then averaged.

Reads   <root>/<model_short>/{cliff_set.json, probs/*.jsonl}
Writes  <root>/cliff_prob_shift_per_seed.csv
        <root>/cliff_prob_shift.csv
            model, variant, cliff_type, n_cliffs, n_seeds,
            base_mean_prob_pct, cliff_dpo_mean_prob_pct, change_pp

    python scripts/_cliff_prob_shift_aggregate.py \
        --root ./output/07_cliff_dpo/05_cliff_prob_shift
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

VARIANT_TO_TYPE = {
    "deterministic": "deterministic",
    "uncertainty": "uncertain",
    "sampled_off": "sampled_off",
}
LABEL_RE = re.compile(r"^cliff_(?P<variant>.+)_only_seed(?P<seed>\d+)$")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./output/07_cliff_dpo/05_cliff_prob_shift")
    args = ap.parse_args()

    per_seed = []
    for model in sorted(os.listdir(args.root)):
        mdir = os.path.join(args.root, model)
        cliff_path = os.path.join(mdir, "cliff_set.json")
        base_path = os.path.join(mdir, "probs", "base.jsonl")
        if not (os.path.isfile(cliff_path) and os.path.isfile(base_path)):
            continue
        with open(cliff_path) as f:
            category = {c["cliff_uid"]: c["category"] for c in json.load(f)}
        base = {r["cliff_uid"]: r["p_cliff"] for r in read_jsonl(base_path)}

        for fname in sorted(os.listdir(os.path.join(mdir, "probs"))):
            m = LABEL_RE.match(fname[: -len(".jsonl")])
            if not m:
                continue
            variant, seed = m.group("variant"), int(m.group("seed"))
            cliff_type = VARIANT_TO_TYPE.get(variant)
            if cliff_type is None:
                continue
            rows = read_jsonl(os.path.join(mdir, "probs", fname))
            uids = [r["cliff_uid"] for r in rows if category.get(r["cliff_uid"]) == cliff_type]
            adapter = {r["cliff_uid"]: r["p_cliff"] for r in rows}
            per_seed.append({
                "model": model,
                "variant": variant,
                "cliff_type": cliff_type,
                "seed": seed,
                "n_cliffs": len(uids),
                "base_mean_prob": mean([base[u] for u in uids]),
                "cliff_dpo_mean_prob": mean([adapter[u] for u in uids]),
            })

    if not per_seed:
        raise SystemExit(f"No results found under {args.root}")

    with open(os.path.join(args.root, "cliff_prob_shift_per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    grouped = defaultdict(list)
    for r in per_seed:
        grouped[(r["model"], r["variant"], r["cliff_type"])].append(r)

    out_rows = []
    for (model, variant, cliff_type), rs in grouped.items():
        base_pct = 100 * mean([r["base_mean_prob"] for r in rs])
        dpo_pct = 100 * mean([r["cliff_dpo_mean_prob"] for r in rs])
        out_rows.append({
            "model": model,
            "variant": variant,
            "cliff_type": cliff_type,
            "n_cliffs": rs[0]["n_cliffs"],
            "n_seeds": len(rs),
            "base_mean_prob_pct": f"{base_pct:.2f}",
            "cliff_dpo_mean_prob_pct": f"{dpo_pct:.2f}",
            "change_pp": f"{dpo_pct - base_pct:+.2f}",
        })

    out_path = os.path.join(args.root, "cliff_prob_shift.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{'model':<24} {'type':<14} {'n':>5} {'base%':>7} {'dpo%':>7} {'Δpp':>7}")
    for r in out_rows:
        print(f"{r['model']:<24} {r['cliff_type']:<14} {r['n_cliffs']:>5} "
              f"{r['base_mean_prob_pct']:>7} {r['cliff_dpo_mean_prob_pct']:>7} {r['change_pp']:>7}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
