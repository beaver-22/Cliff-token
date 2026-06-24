#!/usr/bin/env python3
"""Render cross-direction extra plots for exp5_cpm_shift.

This script is used by scripts/run_exp5_cpm_shift.sh --pair mode.
It loads two analyzed batch directories:
  - smalltobig_dir: source smaller model -> eval larger model
  - bigtosmall_dir: source larger model -> eval smaller model

Expected inputs in each dir:
  - results_per_cliff.csv
  - results_per_token.csv

It filters token rows to the selected cliff token from exp4_candidates
candidate_results.csv when available, then renders:
  - delta_cpm_violin.png
  - rank_heatmap_{deterministic,uncertain,sampled_off}.png
  - asymmetry_bar.png
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _load_analyzer_module():
    script_path = Path(__file__).resolve().parent / "_exp5_cpm_shift_analyze.py"
    spec = importlib.util.spec_from_file_location("exp5_cpm_shift_analyze", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_first_source_model(batch_dir: str) -> str:
    path = os.path.join(batch_dir, "results_per_cliff.csv")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        for row in csv.DictReader(f):
            return row.get("model_source", "")
    return ""


def _auto_find_exp4_dir(exp4_candidates_root: str, source_model_name: str) -> str:
    if not source_model_name:
        return ""
    root = Path(exp4_candidates_root)
    if not root.exists():
        return ""
    candidates = sorted(root.glob(f"{source_model_name}_*"))
    candidates = [p for p in candidates if (p / "candidate_results.csv").exists()]
    if not candidates:
        return ""
    return str(candidates[-1])


def _load_direction_data(
    analyzer,
    batch_dir: str,
    selected_map: Dict[Tuple[str, int], int],
) -> Dict[str, List[Dict]]:
    cliff_csv = os.path.join(batch_dir, "results_per_cliff.csv")
    token_csv = os.path.join(batch_dir, "results_per_token.csv")
    if not os.path.exists(cliff_csv):
        raise FileNotFoundError(f"Missing file: {cliff_csv}")
    if not os.path.exists(token_csv):
        raise FileNotFoundError(f"Missing file: {token_csv}")

    cliff_rows = []
    for row in analyzer._read_csv(cliff_csv):
        analyzer._cast(
            row,
            float_cols=("cpm_source", "cpm_eval", "delta_cpm"),
            int_cols=("cliff_position", "n_cliff_tokens"),
        )
        cliff_rows.append(row)

    token_rows_all = []
    for row in analyzer._read_csv(token_csv):
        analyzer._cast(
            row,
            float_cols=("source_prob", "source_logprob", "eval_prob", "eval_logprob", "delta_prob"),
            int_cols=("cliff_position", "cliff_token_id", "source_rank", "eval_rank", "delta_rank"),
        )
        token_rows_all.append(row)

    # If selected map is available, keep only selected cliff token rows.
    if selected_map:
        token_rows = []
        for row in token_rows_all:
            key = (row["data_idx"], row["cliff_position"])
            if key in selected_map and selected_map[key] == row["cliff_token_id"]:
                tax = row["taxonomy_type"]
                # Keep tie-aware rank normalization consistent with analyzer.
                if tax in ("uncertain", "deterministic") and row["source_rank"] != 1:
                    row["source_rank"] = 1
                elif tax == "sampled_off" and row["source_rank"] == 1:
                    row["source_rank"] = 2
                token_rows.append(row)
    else:
        token_rows = token_rows_all

    return {"cliff": cliff_rows, "token": token_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exp5 cross-direction extra plots")
    parser.add_argument("--smalltobig_dir", required=True, help="Analyzed batch dir for small->big direction")
    parser.add_argument("--bigtosmall_dir", required=True, help="Analyzed batch dir for big->small direction")
    parser.add_argument(
        "--exp4_candidates_root",
        default="./output/07_candidate_replacement",
        help="Root directory containing exp4_candidates outputs",
    )
    parser.add_argument(
        "--exp4_candidates_smalltobig_dir",
        default="",
        help="Explicit exp4_candidates dir for small->big source model",
    )
    parser.add_argument(
        "--exp4_candidates_bigtosmall_dir",
        default="",
        help="Explicit exp4_candidates dir for big->small source model",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory for extra figures (default: <smalltobig_dir>/extra_figures)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = _load_analyzer_module()

    smalltobig_dir = os.path.abspath(args.smalltobig_dir)
    bigtosmall_dir = os.path.abspath(args.bigtosmall_dir)

    if not os.path.isdir(smalltobig_dir):
        raise FileNotFoundError(f"Directory not found: {smalltobig_dir}")
    if not os.path.isdir(bigtosmall_dir):
        raise FileNotFoundError(f"Directory not found: {bigtosmall_dir}")

    out_dir = args.output_dir or os.path.join(smalltobig_dir, "extra_figures")
    os.makedirs(out_dir, exist_ok=True)

    src_small = _read_first_source_model(smalltobig_dir)
    src_big = _read_first_source_model(bigtosmall_dir)

    exp4_small_dir = args.exp4_candidates_smalltobig_dir or _auto_find_exp4_dir(
        args.exp4_candidates_root, src_small
    )
    exp4_big_dir = args.exp4_candidates_bigtosmall_dir or _auto_find_exp4_dir(
        args.exp4_candidates_root, src_big
    )

    selected_small = {}
    if exp4_small_dir and os.path.exists(os.path.join(exp4_small_dir, "candidate_results.csv")):
        selected_small = analyzer._load_selected_cliff_tokens(exp4_small_dir)
    selected_big = {}
    if exp4_big_dir and os.path.exists(os.path.join(exp4_big_dir, "candidate_results.csv")):
        selected_big = analyzer._load_selected_cliff_tokens(exp4_big_dir)

    data = {
        "smalltobig": _load_direction_data(analyzer, smalltobig_dir, selected_small),
        "bigtosmall": _load_direction_data(analyzer, bigtosmall_dir, selected_big),
    }

    analyzer._plot_violin(data, out_dir)
    analyzer._plot_rank_heatmap(data, out_dir)
    analyzer._plot_asymmetry_bar(data, out_dir)

    summary = {
        "smalltobig_dir": smalltobig_dir,
        "bigtosmall_dir": bigtosmall_dir,
        "exp4_candidates_smalltobig_dir": exp4_small_dir,
        "exp4_candidates_bigtosmall_dir": exp4_big_dir,
        "n_smalltobig_cliffs": len(data["smalltobig"]["cliff"]),
        "n_smalltobig_tokens": len(data["smalltobig"]["token"]),
        "n_bigtosmall_cliffs": len(data["bigtosmall"]["cliff"]),
        "n_bigtosmall_tokens": len(data["bigtosmall"]["token"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[exp5 extra plots] complete")
    print(f"  output_dir: {out_dir}")
    print(f"  smalltobig cliffs/tokens: {summary['n_smalltobig_cliffs']}/{summary['n_smalltobig_tokens']}")
    print(f"  bigtosmall cliffs/tokens: {summary['n_bigtosmall_cliffs']}/{summary['n_bigtosmall_tokens']}")


if __name__ == "__main__":
    main()
