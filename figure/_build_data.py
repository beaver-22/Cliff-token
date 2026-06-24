"""Pre-extract minimal data for /workspace/figure/figure.ipynb.

Run once from the repo root (or anywhere — paths are absolute):
    python /workspace/figure/_build_data.py

Idempotent: rerunning overwrites outputs. Reads from /workspace/output/...
and writes only into /workspace/figure/data/.

Each section is independent — comment out blocks if a source is missing.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path("/workspace")
OUT_BASE = REPO / "output"
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

MODELS = ["Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B", "Llama-3.1-8B-Instruct", "gemma-3-4b-it"]
DATASETS = ["gsm1k", "math500", "aime25"]


# ──────────────────────────────────────────────────────────────────────
# Section 1: static copies
# ──────────────────────────────────────────────────────────────────────

def copy_static() -> None:
    moves = [
        # fig02
        (OUT_BASE / "exp1_1/0418_004503/threshold_comparison_statistical.json",
         DATA / "02_threshold_comparison_pct/rates.json"),
        # fig03
        (OUT_BASE / "exp1_1/0418_004503/threshold_comparison_statistical_avg_cliffs.json",
         DATA / "03_threshold_comparison_avg/rates.json"),
        # fig11-14
        (OUT_BASE / "exp2_3/0418/smalltobig/results_per_cliff.csv",
         DATA / "11_14_exp2_3_extra_plots/smalltobig_per_cliff.csv"),
        (OUT_BASE / "exp2_3/0418/smalltobig/results_per_token.csv",
         DATA / "11_14_exp2_3_extra_plots/smalltobig_per_token.csv"),
        (OUT_BASE / "exp2_3/0418/bigtosmall/results_per_cliff.csv",
         DATA / "11_14_exp2_3_extra_plots/bigtosmall_per_cliff.csv"),
        (OUT_BASE / "exp2_3/0418/bigtosmall/results_per_token.csv",
         DATA / "11_14_exp2_3_extra_plots/bigtosmall_per_token.csv"),
        # fig16 (failure-only pass@k transposed grid)
        (OUT_BASE / "exp1_3/0417_034659_batch/grid/pass_at_k_failure_only.csv",
         DATA / "16_pass_at_k_failure_only/pass_at_k_failure_only.csv"),
    ]
    for src, dst in moves:
        if not src.exists():
            print(f"  [WARN] missing: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied {src.name} -> {dst.relative_to(DATA.parent)}")


# ──────────────────────────────────────────────────────────────────────
# Section 2: fig04 pass@k grid CSV
# ──────────────────────────────────────────────────────────────────────

def build_pass_at_k_csv() -> None:
    runs_root = OUT_BASE / "exp1_3/0417_034659_batch/runs"
    out_csv = DATA / "04_pass_at_k_grid/pass_at_k.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for model in MODELS:
        for ds in DATASETS:
            run_dir = runs_root / f"{model}_{ds}" / "sub_exp_1"
            pk_path = run_dir / "exp1_pass_at_k.json"
            cliffs_path = run_dir / "cliff_results.json"
            if not pk_path.exists() or not cliffs_path.exists():
                print(f"  [skip] {model}/{ds} (no rollout)")
                continue
            pk = json.loads(pk_path.read_text())
            n_cliffs = len(json.loads(cliffs_path.read_text()))
            for line_type in ("del_success", "keep_success", "del_failure", "keep_failure"):
                d = pk.get(line_type, {})
                for k_str, val in d.items():
                    rows.append({
                        "model": model, "dataset": ds, "line_type": line_type,
                        "k": int(k_str), "pass_at_k": float(val),
                        "n_cliff_tokens": n_cliffs,
                    })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "line_type", "k", "pass_at_k", "n_cliff_tokens"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {out_csv.relative_to(DATA.parent)}")


# ──────────────────────────────────────────────────────────────────────
# Section 3: fig05-10 exp2_2 boxplot points (sum_p_cliff per cliff)
#            + fig06 type3 recovery (Qwen3-8B only)
# ──────────────────────────────────────────────────────────────────────

EXP2_2_RUNS = {
    "Qwen3-8B": "Qwen3-8B_0417_140533",
    "Qwen3-4B": "Qwen3-4B_0418_021224",
    "Qwen3-0.6B": "Qwen3-0.6B_0418_020315",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct_0418_142517",
    "gemma-3-4b-it": "gemma-3-4b-it_0418_053349",
}

# cell_label → (taxonomy_short, full_name) — taken from TYPE_ORDER in exp2_2_aggregator.py
CELL_TO_TAXONOMY = {
    "low-H + greedy":     ("DF", "deterministic failure"),
    "high-H + greedy":    ("AG", "ambiguous greedy"),
    "high-H + non-greedy": ("SS", "sampling slip"),
}

DATA_DIR_BY_MODEL = {
    "Qwen3-8B": "05_06_exp2_2_Qwen3-8B",
    "Qwen3-4B": "07_exp2_2_Qwen3-4B",
    "Qwen3-0.6B": "08_exp2_2_Qwen3-0.6B",
    "Llama-3.1-8B-Instruct": "09_exp2_2_Llama-3.1-8B-Instruct",
    "gemma-3-4b-it": "10_exp2_2_gemma-3-4b-it",
}


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_csv_dict(p: Path) -> List[Dict[str, str]]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def build_exp2_2_per_model() -> None:
    for model, run_name in EXP2_2_RUNS.items():
        run_dir = OUT_BASE / "exp2_2" / run_name
        cliffs_csv = run_dir / "cliff_instances.csv"
        cands_csv = run_dir / "candidate_results.csv"
        if not cliffs_csv.exists() or not cands_csv.exists():
            print(f"  [skip] {model}: missing CSVs in {run_dir}")
            continue

        cliffs = _read_csv_dict(cliffs_csv)
        cands = _read_csv_dict(cands_csv)

        # Group candidates by cliff_uid
        cands_by_uid: Dict[str, List[Dict]] = {}
        for r in cands:
            uid = r.get("cliff_uid", "")
            if not uid:
                continue
            cands_by_uid.setdefault(uid, []).append(r)

        # Compute sum_p_cliff per cliff_uid (matches _build_cliff_metrics)
        out_box: List[Dict] = []
        for c in cliffs:
            uid = c["cliff_uid"]
            cell = c.get("cell_label", "")
            tax = CELL_TO_TAXONOMY.get(cell)
            if tax is None:
                continue  # exclude low-H + non-greedy and unknown
            tax_short, _ = tax
            rows = cands_by_uid.get(uid, [])
            sum_p_cliff = 0.0
            for r in rows:
                if _parse_bool(r.get("is_candidate_cliff_stat", "")) or \
                   _parse_bool(r.get("is_candidate_selected_cliff", "")):
                    try:
                        sum_p_cliff += float(r.get("candidate_prob", "0") or 0.0)
                    except ValueError:
                        pass
            out_box.append({
                "cliff_uid": uid,
                "model": model,
                "taxonomy_type": tax_short,
                "cell_label": cell,
                "sum_p_cliff": sum_p_cliff,
            })

        out_dir = DATA / DATA_DIR_BY_MODEL[model]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "boxplot_points.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cliff_uid", "model", "taxonomy_type", "cell_label", "sum_p_cliff"])
            w.writeheader()
            w.writerows(out_box)
        print(f"  {model}: {len(out_box)} boxplot points -> {out_path.relative_to(DATA.parent)}")

        # Type3 recovery: just copy the existing aggregator output
        if model == "Qwen3-8B":
            t3_src = run_dir / "analysis/tables/exp2_2_type3_greedy_recovery_points.csv"
            t3_dst = DATA / "05_06_exp2_2_Qwen3-8B/type3_recovery_points.csv"
            if t3_src.exists():
                shutil.copy2(t3_src, t3_dst)
                with t3_dst.open() as f:
                    n = sum(1 for _ in f) - 1
                print(f"  {model}: type3 scatter copied ({n} points)")
            else:
                print(f"  [WARN] {t3_src} not found")


# ──────────────────────────────────────────────────────────────────────
# Section 3b: fig12-14 selected-cliff token filter
#   The original heatmap (scripts/exp2_3_extra_plots.py:plot5) only includes
#   tokens that match the *selected* cliff candidate per (path_id, cliff_pos)
#   in exp2_2/<source_model>/candidate_results.csv. We bundle just those keys
#   so the notebook can apply the same filter without copying the full CSVs.
# ──────────────────────────────────────────────────────────────────────

EXP2_3_SOURCE_RUN = {
    # smalltobig source = small model (Qwen3-0.6B); bigtosmall source = big (Qwen3-8B)
    "smalltobig": "Qwen3-0.6B_0418_020315",
    "bigtosmall": "Qwen3-8B_0417_140533",
}


def build_selected_cliff_keys() -> None:
    out_dir = DATA / "11_14_exp2_3_extra_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for direction, run_name in EXP2_3_SOURCE_RUN.items():
        src = OUT_BASE / "exp2_2" / run_name / "candidate_results.csv"
        if not src.exists():
            print(f"  [WARN] missing candidate_results: {src}")
            continue
        dst = out_dir / f"{direction}_selected_cliff_keys.csv"
        n = 0
        with src.open() as fi, dst.open("w", newline="") as fo:
            r = csv.DictReader(fi)
            w = csv.DictWriter(fo, fieldnames=["path_id", "cliff_position", "candidate_token_id"])
            w.writeheader()
            for row in r:
                if row.get("is_candidate_selected_cliff", "").lower() in {"true", "1", "t", "yes"}:
                    w.writerow({
                        "path_id": row["path_id"],
                        "cliff_position": row["cliff_position"],
                        "candidate_token_id": row["candidate_token_id"],
                    })
                    n += 1
        print(f"  {direction}: {n} selected-cliff keys -> {dst.relative_to(DATA.parent)}")


# ──────────────────────────────────────────────────────────────────────
# Section 4: fig15 entropy density (cliff + subsampled baseline per model)
# ──────────────────────────────────────────────────────────────────────

def build_entropy_arrays() -> None:
    cliff_runs_root = OUT_BASE / "exp2_1/0417_120425_batch/runs"
    base_root = OUT_BASE / "inference_with_logprobs"
    out_path = DATA / "15_entropy_density/entropy_arrays.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    SUBSAMPLE = 20_000
    rng = np.random.default_rng(42)
    arrays: Dict[str, np.ndarray] = {}

    for model in MODELS:
        # Cliff entropies (across all datasets for this model)
        cliff_ents: List[float] = []
        for ds in DATASETS:
            cliff_json = cliff_runs_root / f"{model}_{ds}/cliff_logprobs.json"
            if not cliff_json.exists():
                continue
            for c in json.loads(cliff_json.read_text()):
                try:
                    cliff_ents.append(float(c["entropy_at_t"]))
                except (KeyError, TypeError, ValueError):
                    pass

        # Baseline entropies (across datasets) — flatten + subsample
        base_ents: List[float] = []
        for ds in DATASETS:
            bp = base_root / model / f"{ds}_all_paths.json"
            if not bp.exists():
                continue
            data = json.loads(bp.read_text())
            for entry in data:
                ents = entry.get("response_token_entropies") or []
                base_ents.extend(float(e) for e in ents if e is not None)

        cliff_arr = np.asarray(cliff_ents, dtype=np.float32)
        base_arr_full = np.asarray(base_ents, dtype=np.float32)
        n_base_full = base_arr_full.size
        if n_base_full > SUBSAMPLE:
            idx = rng.choice(n_base_full, size=SUBSAMPLE, replace=False)
            base_arr = base_arr_full[idx]
        else:
            base_arr = base_arr_full
        arrays[f"cliff_{model}"] = cliff_arr
        arrays[f"base_{model}"] = base_arr
        # Original baseline count (pre-subsample) so the legend can show
        # "All tokens (n=138,590)" etc. like the original figure.
        arrays[f"n_base_full_{model}"] = np.array([n_base_full])
        print(f"  {model}: cliff={cliff_arr.size}  base(sub)={base_arr.size}  base(full)={n_base_full}")

    np.savez_compressed(out_path, **arrays)
    print(f"  wrote {out_path.relative_to(DATA.parent)}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 1. static copies ===")
    copy_static()
    print("\n=== 2. fig04 pass@k CSV ===")
    build_pass_at_k_csv()
    print("\n=== 3. fig05-10 exp2_2 per-model ===")
    build_exp2_2_per_model()
    print("\n=== 3b. fig12-14 selected-cliff filter keys ===")
    build_selected_cliff_keys()
    print("\n=== 4. fig15 entropy npz ===")
    build_entropy_arrays()
    print("\nDone. Artifacts written under:", DATA)


if __name__ == "__main__":
    main()
