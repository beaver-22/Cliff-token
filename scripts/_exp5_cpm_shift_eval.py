"""exp5_cpm_shift — Cross-Model Cliff Probability Mass (CPM) shift (single combo).

Given a (source_model, eval_model, dataset) triple, this script:
  1. Loads the source model's exp4_candidates candidate-level cliff data to get per-cliff
     cliff-token sets and cpm_source.
  2. Loads rollout paths for the source model, reconstructs prefix token
     IDs at each cliff position.
  3. Feeds prefixes into the eval model via vLLM with logprobs=20, max_tokens=1
     (forward-pass only, no generation).
  4. For each cliff instance, computes cpm_eval (sum of eval top-20 probs for
     source cliff token IDs) and per-token rank/prob shift.
  5. Writes per_cliff.csv, per_token.csv, eval_top20.jsonl under output_dir.

Usage:
    python3 scripts/_exp5_cpm_shift_eval.py \
        --source qwen3-8b --eval qwen3-0.6b --dataset gsm1k \
        --exp4_candidates_source_dir output/07_candidate_replacement/Qwen3-8B_0409_072941 \
        --rollout_dir ./output/03_rollout \
        --gpus 0 \
        --output_dir output/08_cpm_shift/_smoke/runs/Qwen3-8B__Qwen3-0.6B__gsm1k
"""
import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, ".")

from src import config
from src.analysis.entropy import _logprob_value, compute_tie_aware_ranks
from src.analysis.exp4_candidates_aggregator import TYPE_ORDER


CELL_LABEL_TO_TAXONOMY = {
    "low-H + greedy": "deterministic",
    "high-H + greedy": "uncertain",
    "high-H + non-greedy": "sampled_off",
}


def _model_dir_name(alias: str) -> str:
    """Return the capitalized directory name used under output/03_rollout/ etc."""
    return os.path.basename(config.resolve_model_path(alias).rstrip("/"))


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_exp4_candidates_source(exp4_candidates_dir: str, dataset: str):
    """Load per-cliff data from a source exp4_candidates output dir, filtered by dataset.

    Returns: dict cliff_uid -> {
        "dataset", "path_id", "cliff_position", "cell_label", "taxonomy",
        "cliff_candidates": list of {tid, token_str, rank, prob, logprob},
        "cpm_source": float,
    }
    """
    cliffs_csv = os.path.join(exp4_candidates_dir, "cliff_instances.csv")
    cands_csv = os.path.join(exp4_candidates_dir, "candidate_results.csv")
    if not os.path.exists(cliffs_csv):
        raise SystemExit(f"missing: {cliffs_csv}")
    if not os.path.exists(cands_csv):
        raise SystemExit(f"missing: {cands_csv}")

    # Build cliff meta from cliff_instances.csv (dataset filter applied here)
    cliff_meta: Dict[str, Dict] = {}
    with open(cliffs_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("dataset", "") != dataset:
                continue
            cell_label = row.get("cell_label", "")
            taxonomy = CELL_LABEL_TO_TAXONOMY.get(cell_label)
            if taxonomy is None:
                # Skip 4th cell (low-H + non-greedy) — not in the 3-class taxonomy.
                continue
            uid = row["cliff_uid"]
            cliff_meta[uid] = {
                "dataset": row.get("dataset", ""),
                "path_id": row.get("path_id", ""),
                "cliff_position": int(row.get("cliff_position", "0") or 0),
                "cell_label": cell_label,
                "taxonomy": taxonomy,
                "cliff_candidates": [],
            }

    # Attach cliff-flagged candidates from candidate_results.csv.
    # Same definition as exp4_candidates's _build_cliff_metrics: a candidate counts as a
    # cliff token if (a) the statistical test flagged it as cliff_stat, OR
    # (b) it was the originally sampled cliff token at this position.
    # Including (b) is critical for deterministic/uncertain cases where the selected greedy token
    # carries essentially all the probability mass.
    with open(cands_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            uid = row.get("cliff_uid", "")
            if uid not in cliff_meta:
                continue
            is_stat = _parse_bool(row.get("is_candidate_cliff_stat", ""))
            is_selected = _parse_bool(row.get("is_candidate_selected_cliff", ""))
            if not (is_stat or is_selected):
                continue
            try:
                tid = int(row.get("candidate_token_id", ""))
                rank = int(row.get("candidate_rank", ""))
                prob = float(row.get("candidate_prob", "0") or 0.0)
                lp = float(row.get("candidate_logprob", "0") or 0.0)
            except Exception as e:
                print(f"  WARN: bad candidate row for {uid}: {e}")
                continue
            cliff_meta[uid]["cliff_candidates"].append({
                "tid": tid,
                "token_str": row.get("candidate_token_str", ""),
                "rank": rank,
                "prob": prob,
                "logprob": lp,
            })

    # Compute cpm_source; drop cliffs with zero cliff candidates.
    cleaned: Dict[str, Dict] = {}
    for uid, entry in cliff_meta.items():
        cands = entry["cliff_candidates"]
        if not cands:
            continue
        entry["cliff_candidates"] = sorted(cands, key=lambda c: c["rank"])
        entry["cpm_source"] = sum(c["prob"] for c in cands)
        cleaned[uid] = entry
    return cleaned


def _load_rollout_paths(rollout_dir: str, source_model_alias: str, dataset: str) -> Dict[str, Dict]:
    model_dir = _model_dir_name(source_model_alias)
    path = os.path.join(rollout_dir, model_dir, f"{dataset}_all_paths.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing rollout file: {path}")
    with open(path) as f:
        data = json.load(f)
    return {p["id"]: p for p in data}


def _assert_tokenizer_compat(source_alias: str, eval_alias: str):
    """Abort if source and eval tokenizers differ."""
    from transformers import AutoTokenizer
    s_path = config.resolve_model_path(source_alias)
    e_path = config.resolve_model_path(eval_alias)
    s_tok = AutoTokenizer.from_pretrained(s_path, trust_remote_code=True)
    e_tok = AutoTokenizer.from_pretrained(e_path, trust_remote_code=True)
    mismatches = []
    if s_tok.vocab_size != e_tok.vocab_size:
        mismatches.append(f"vocab_size {s_tok.vocab_size} != {e_tok.vocab_size}")
    if getattr(s_tok, "eos_token_id", None) != getattr(e_tok, "eos_token_id", None):
        mismatches.append(
            f"eos_token_id {s_tok.eos_token_id} != {e_tok.eos_token_id}"
        )
    sample_text = "The quick brown fox jumps over the lazy dog. 12345 + 67890 = ?"
    s_ids = tuple(s_tok.encode(sample_text, add_special_tokens=False))
    e_ids = tuple(e_tok.encode(sample_text, add_special_tokens=False))
    if s_ids != e_ids:
        mismatches.append(f"sample token ids differ: {s_ids[:10]}... vs {e_ids[:10]}...")
    if mismatches:
        raise SystemExit(
            f"[tokenizer mismatch] source={source_alias} eval={eval_alias}: "
            + "; ".join(mismatches)
        )
    return s_tok


def _build_prefix_ids(tokenizer, path_record: Dict, cliff_position: int) -> List[int]:
    """prefix = encode(full_prompt) + response_token_ids[:cliff_position - 1]."""
    full_prompt = path_record.get("full_prompt", "")
    response_ids = path_record.get("response_token_ids", [])
    if cliff_position < 1 or cliff_position - 1 > len(response_ids):
        raise ValueError(f"cliff_position {cliff_position} out of range (len={len(response_ids)})")
    prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    return list(prompt_ids) + list(response_ids[:cliff_position - 1])


def _extract_top20(logprobs_dict) -> List[Tuple[int, int, float, float]]:
    """vLLM logprobs dict at one position → list of (rank, tid, logprob, prob).

    Uses tie-aware competition rank: tokens within EPS of the top logprob all
    share rank=1 (matches deterministic/uncertain/sampled_off taxonomy semantics elsewhere).
    """
    if logprobs_dict is None:
        return []
    tie_ranks = compute_tie_aware_ranks(logprobs_dict)
    out = []
    for tid, lp_entry in logprobs_dict.items():
        lp = _logprob_value(lp_entry)
        rank = tie_ranks.get(int(tid), 10 ** 9)
        prob = math.exp(lp) if math.isfinite(lp) else 0.0
        out.append((rank, int(tid), float(lp), float(prob)))
    out.sort(key=lambda x: x[0])
    return out[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source model alias")
    ap.add_argument("--eval", dest="eval_alias", required=True, help="eval model alias")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp4_candidates_source_dir", required=True,
                    help="output/07_candidate_replacement/<source>_<ts> directory")
    ap.add_argument("--rollout_dir", default="./output/03_rollout")
    ap.add_argument("--gpus", default="0", help="GPU IDs for vLLM (comma list)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.35,
                    help="vLLM gpu_memory_utilization (low by default — we only "
                         "do a handful of forward passes per combo, so a large "
                         "KV cache is unnecessary and would block sequential combos)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[exp5_cpm_shift] source={args.source} eval={args.eval_alias} dataset={args.dataset}")

    # 1. tokenizer compatibility
    tokenizer = _assert_tokenizer_compat(args.source, args.eval_alias)

    # 2. source cliff data
    source_cliffs = _load_exp4_candidates_source(args.exp4_candidates_source_dir, args.dataset)
    print(f"  loaded {len(source_cliffs)} cliff instances from {args.exp4_candidates_source_dir}")
    if not source_cliffs:
        print("  no cliffs after dataset filter; writing empty outputs and exiting.")
        _write_empty(args)
        return

    # 3. rollout paths
    path_map = _load_rollout_paths(args.rollout_dir, args.source, args.dataset)
    print(f"  loaded {len(path_map)} rollout paths")

    # 4. prefix build
    prefixes: List[Tuple[str, List[int]]] = []  # (cliff_uid, prefix_ids)
    skipped = 0
    for uid, entry in source_cliffs.items():
        path_rec = path_map.get(entry["path_id"])
        if path_rec is None:
            print(f"  WARN: path_id {entry['path_id']} missing in rollout; skip")
            skipped += 1
            continue
        try:
            pids = _build_prefix_ids(tokenizer, path_rec, entry["cliff_position"])
        except Exception as e:
            print(f"  WARN: prefix build failed for {uid}: {e}")
            skipped += 1
            continue
        prefixes.append((uid, pids))
    print(f"  built {len(prefixes)} prefixes (skipped {skipped})")

    if not prefixes:
        _write_empty(args)
        return

    # 5. vLLM eval model init
    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # After CUDA_VISIBLE_DEVICES, vLLM sees devices 0..N-1 regardless of physical IDs.
    visible = list(range(len(gpu_ids)))

    from src.cli import _init_heavy_imports, create_llm
    _init_heavy_imports()
    eval_path = config.resolve_model_path(args.eval_alias)
    print(f"  loading eval model {args.eval_alias} from {eval_path} on GPUs {gpu_ids}...")
    llm = create_llm(eval_path, visible, args.gpu_mem)

    # 6. batched forward pass
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    requests = [{"prompt_token_ids": pids} for _, pids in prefixes]
    print(f"  running vLLM forward pass on {len(requests)} prompts...")
    outputs = llm.generate(requests, sp, use_tqdm=True)

    # 7. extract + 8. compute CPM / per-token shift + 9. write
    per_cliff_rows = []
    per_token_rows = []
    top20_lines = []

    source_model_name = _model_dir_name(args.source)
    eval_model_name = _model_dir_name(args.eval_alias)

    for (uid, _pids), out in zip(prefixes, outputs):
        entry = source_cliffs[uid]
        cliff_cands = entry["cliff_candidates"]
        # vLLM: outputs[0].logprobs is list of per-step logprob dicts (len = max_tokens).
        step_lps = out.outputs[0].logprobs if out.outputs else None
        if not step_lps:
            continue
        top20 = _extract_top20(step_lps[0])
        top20_by_tid = {tid: (rank, lp, prob) for (rank, tid, lp, prob) in top20}

        cliff_tid_set = {c["tid"] for c in cliff_cands}
        cpm_eval = sum(
            prob for tid, (_, _, prob) in top20_by_tid.items() if tid in cliff_tid_set
        )
        cpm_source = entry["cpm_source"]

        per_cliff_rows.append({
            "model_source": source_model_name,
            "model_eval": eval_model_name,
            "dataset": entry["dataset"],
            "data_idx": entry["path_id"],
            "cliff_position": entry["cliff_position"],
            "taxonomy_type": entry["taxonomy"],
            "n_cliff_tokens": len(cliff_cands),
            "cpm_source": cpm_source,
            "cpm_eval": cpm_eval,
            "delta_cpm": cpm_eval - cpm_source,
        })

        for cand in cliff_cands:
            tid = cand["tid"]
            eval_entry = top20_by_tid.get(tid)
            if eval_entry is None:
                eval_rank, eval_lp, eval_prob = 21, -1e9, 0.0
            else:
                eval_rank, eval_lp, eval_prob = eval_entry
            per_token_rows.append({
                "model_source": source_model_name,
                "model_eval": eval_model_name,
                "dataset": entry["dataset"],
                "data_idx": entry["path_id"],
                "cliff_position": entry["cliff_position"],
                "taxonomy_type": entry["taxonomy"],
                "cliff_token_id": tid,
                "cliff_token_str": cand["token_str"],
                "source_rank": cand["rank"],
                "source_prob": cand["prob"],
                "source_logprob": cand["logprob"],
                "eval_rank": eval_rank,
                "eval_prob": eval_prob,
                "eval_logprob": eval_lp,
                # delta_rank: eval_rank - source_rank. Since tie-aware competition
                # ranking is applied, if the source side has a top tie, source rank
                # can jump from 1 to 3, 4, etc. (skipping rank 2). Therefore,
                # do not interpret the absolute delta directly as "number of tier
                # movements"; always view it together with the source's tie distribution.
                "delta_rank": eval_rank - cand["rank"],
                "delta_prob": eval_prob - cand["prob"],
            })

        top20_lines.append(json.dumps({
            "cliff_uid": uid,
            "top20": [
                {"rank": rank, "tid": tid, "logprob": lp, "prob": prob}
                for (rank, tid, lp, prob) in top20
            ],
        }))

    _write_outputs(args.output_dir, per_cliff_rows, per_token_rows, top20_lines)
    print(f"  done: {len(per_cliff_rows)} per_cliff rows, {len(per_token_rows)} per_token rows")

    # Release GPU memory cleanly so the next combo in the same batch can
    # reinitialize vLLM on the same GPUs without OOM. Skipping vLLM's
    # shutdown path (e.g. via os._exit) leaves spawn-workers orphaned,
    # holding the entire model + KV cache.
    import gc
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _per_cliff_fields():
    return [
        "model_source", "model_eval", "dataset", "data_idx", "cliff_position",
        "taxonomy_type", "n_cliff_tokens", "cpm_source", "cpm_eval", "delta_cpm",
    ]


def _per_token_fields():
    return [
        "model_source", "model_eval", "dataset", "data_idx", "cliff_position",
        "taxonomy_type", "cliff_token_id", "cliff_token_str",
        "source_rank", "source_prob", "source_logprob",
        "eval_rank", "eval_prob", "eval_logprob",
        "delta_rank", "delta_prob",
    ]


def _write_outputs(output_dir: str, per_cliff_rows, per_token_rows, top20_lines):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "per_cliff.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_per_cliff_fields())
        w.writeheader()
        for r in per_cliff_rows:
            w.writerow(r)
    with open(os.path.join(output_dir, "per_token.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_per_token_fields())
        w.writeheader()
        for r in per_token_rows:
            w.writerow(r)
    with open(os.path.join(output_dir, "eval_top20.jsonl"), "w") as f:
        for line in top20_lines:
            f.write(line + "\n")


def _write_empty(args):
    _write_outputs(args.output_dir, [], [], [])


if __name__ == "__main__":
    main()
