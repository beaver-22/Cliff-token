"""RQ2-2 runner: Qwen3-8B cliff top-k replacement experiment.

Pipeline:
1) Load rollout *_all_paths.json and detect statistical cliffs.
2) At each cliff position, query top-k logprobs (k=20) and classify cell.
3) For each (cliff, candidate token), run n=64 replacement rollout.
4) Save raw correctness arrays and aggregated metrics.

Output files (under --output_dir):
  - config.json
  - cliff_instances.jsonl / cliff_instances.csv
  - candidate_results.jsonl / candidate_results.csv
  - candidate_rollout_raw.jsonl
  - cliff_summary.csv
  - cell_summary.csv
  - priority_1_highH_non_greedy_greedy_baseline.csv
  - priority_2_highH_non_greedy_topk.csv
  - priority_3_highH_greedy_non_greedy_candidates.csv
  - validation_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

sys.path.insert(0, ".")

from src import config
from src.analysis.cliff_threshold import is_cliff_lookup, score_to_k
from src.analysis.detector import find_all_cliff_tokens_statistical
from src.analysis.entropy import (
    _logprob_value,
    compute_entropy_from_logprobs,
    compute_tie_aware_ranks,
)
from src.analysis.exp3_entropy_aggregator import GREEDY_99_BOUND_NATS
from src.cli import _init_heavy_imports, create_llm
from src.utils.grader import batch_grade_responses_mathverify


@dataclass
class CliffInstance:
    cliff_uid: str
    model: str
    dataset: str
    path_id: str
    problem_id: Optional[str]
    path_is_correct: bool
    cliff_position: int
    cliff_token_id: int
    cliff_token_str: str
    cliff_token_rank: int
    cliff_token_logprob: float
    cliff_token_prob: float
    entropy_at_t: float
    cell_label: str
    potential_t_minus_1: float
    potential_t_cliff: float
    cliff_drop: float
    greedy_token_id: int
    greedy_token_str: str
    greedy_token_rank: int
    greedy_token_logprob: float
    greedy_token_prob: float
    is_cliff_eq_greedy: bool
    topk_prob_mass: float
    tail_mass: float
    n_candidates: int
    is_target_cell: bool


@dataclass
class CandidateResult:
    cliff_uid: str
    model: str
    dataset: str
    path_id: str
    cliff_position: int
    cell_label: str
    cliff_token_id: int
    greedy_token_id: int
    candidate_token_id: int
    candidate_token_str: str
    candidate_rank: int
    candidate_logprob: float
    candidate_prob: float
    is_candidate_greedy: bool
    is_candidate_selected_cliff: bool
    potential_t_minus_1: float
    potential_t_cliff: float
    potential_t_candidate: float
    delta_vs_t_minus_1: float
    delta_vs_cliff: float
    delta_vs_greedy: float
    is_candidate_cliff_stat: bool
    candidate_num_correct: int
    num_samples: int


@dataclass
class CandidateRaw:
    cliff_uid: str
    model: str
    dataset: str
    path_id: str
    cliff_position: int
    candidate_token_id: int
    candidate_rank: int
    candidate_correctness: List[bool]


@dataclass
class CandidateMeta:
    token_id: int
    token_str: str
    rank: int
    logprob: float
    prob: float


@dataclass
class CliffWorkItem:
    cliff: CliffInstance
    prompt_ids: List[int]
    response_token_ids: List[int]
    golden_answers: List[str]
    candidates: List[CandidateMeta]


def _parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_chunks(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _num_chunks(total: int, size: int) -> int:
    if size <= 0:
        return 0
    return (total + size - 1) // size


def _cell_label(entropy_at_t: float, is_eq_greedy: bool) -> str:
    is_low = entropy_at_t <= GREEDY_99_BOUND_NATS
    if is_low and is_eq_greedy:
        return "low-H + greedy"
    if is_low and not is_eq_greedy:
        return "low-H + non-greedy"
    if (not is_low) and is_eq_greedy:
        return "high-H + greedy"
    return "high-H + non-greedy"


def _safe_problem_id(path_obj: Dict) -> Optional[str]:
    pid = path_obj.get("problem_id")
    if pid is None:
        return None
    return str(pid)


def _entry_rank(token_id: int, tie_ranks: Dict[int, int], fallback_rank: int) -> int:
    """Tie-aware rank: tokens within EPS of the top logprob share rank=1."""
    return tie_ranks.get(int(token_id), fallback_rank)


def _sort_candidate_entries(
    logprob_dict: Dict[int, object],
    tie_ranks: Dict[int, int],
) -> List[Tuple[int, object]]:
    items = list(logprob_dict.items())

    def _key(item: Tuple[int, object]):
        token_id, entry = item
        rank = tie_ranks.get(int(token_id), 10 ** 9)
        return (rank, -_logprob_value(entry), token_id)

    items.sort(key=_key)
    return items


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_rollout_paths(rollout_dir: Path, model_short: str, datasets: List[str]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for ds in datasets:
        fp = rollout_dir / model_short / f"{ds}_all_paths.json"
        if fp.exists():
            resolved[ds] = fp
    return resolved


def _count_target_cliffs_for_path(path_obj: Dict, target_cells: Optional[set]) -> int:
    scores = path_obj.get("all_position_scores", [])
    if not scores:
        return 0

    cliffs = find_all_cliff_tokens_statistical(
        scores,
        tokens=path_obj.get("response_tokens"),
        token_ids=path_obj.get("response_token_ids"),
    )
    if not cliffs:
        return 0

    # NOTE:
    # `find_all_cliff_tokens_statistical` returns `CliffTokenInfo`, which does
    # not include entropy/is_greedy metadata required for cell labeling.
    # Shard balancing only needs a stable cost proxy, so we use raw cliff
    # counts here and apply exact `target_cells` filtering later in
    # `_extract_cliffs_and_candidates` (where entropy + eq-greedy are computed).
    _ = target_cells
    return len(cliffs)


def _build_balanced_path_assignment(
    paths: List[Dict],
    num_shards: int,
    target_cells: Optional[set],
    top_k: int,
) -> Tuple[List[List[int]], Dict[str, List[int]]]:
    if num_shards <= 1:
        return [list(range(len(paths)))], {
            "estimated_loads": [len(paths)],
            "estimated_cliffs": [0],
            "path_counts": [len(paths)],
        }

    # Weight combines rollout cost (dominant) and a small per-path base cost.
    # rollout ~ (#cliffs * top_k * num_samples), so top_k-scaled cliff count is
    # a good proxy. Exact target-cell filtering is applied downstream.
    items: List[Tuple[int, int, int]] = []
    for idx, path_obj in enumerate(paths):
        n_cliffs = _count_target_cliffs_for_path(path_obj, target_cells)
        weight = n_cliffs * max(1, int(top_k)) + 1
        items.append((idx, weight, n_cliffs))

    items.sort(key=lambda x: (-x[1], x[0]))
    shard_indices: List[List[int]] = [[] for _ in range(num_shards)]
    shard_loads: List[int] = [0 for _ in range(num_shards)]
    shard_cliffs: List[int] = [0 for _ in range(num_shards)]

    for idx, weight, n_cliffs in items:
        shard = min(
            range(num_shards),
            key=lambda s: (shard_loads[s], len(shard_indices[s]), s),
        )
        shard_indices[shard].append(idx)
        shard_loads[shard] += weight
        shard_cliffs[shard] += n_cliffs

    for shard in range(num_shards):
        shard_indices[shard].sort()

    stats = {
        "estimated_loads": shard_loads,
        "estimated_cliffs": shard_cliffs,
        "path_counts": [len(v) for v in shard_indices],
    }
    return shard_indices, stats


def _filter_paths_for_shard(
    paths: List[Dict],
    num_shards: int,
    shard_index: int,
    target_cells: Optional[set],
    top_k: int,
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    if num_shards <= 1:
        return paths, {
            "estimated_loads": [len(paths)],
            "estimated_cliffs": [0],
            "path_counts": [len(paths)],
        }

    shard_indices, stats = _build_balanced_path_assignment(
        paths=paths,
        num_shards=num_shards,
        target_cells=target_cells,
        top_k=top_k,
    )
    keep = set(shard_indices[shard_index])
    filtered = [p for i, p in enumerate(paths) if i in keep]
    return filtered, stats


def _extract_cliffs_and_candidates(
    llm,
    tokenizer,
    paths: List[Dict],
    model_short: str,
    dataset: str,
    top_k: int,
    extract_batch_size: int,
    target_cells: Optional[set],
    max_cliffs_per_dataset: int,
) -> List[CliffWorkItem]:
    from vllm import SamplingParams

    print(f"[{dataset}] detecting statistical cliffs...")

    requests = []  # (path_obj, path_idx, cliff_obj, prefix_ids, prompt_ids)
    for path_idx, p in enumerate(paths):
        scores = p.get("all_position_scores", [])
        if not scores:
            continue
        cliffs = find_all_cliff_tokens_statistical(
            scores,
            tokens=p.get("response_tokens"),
            token_ids=p.get("response_token_ids"),
        )
        if not cliffs:
            continue

        response_token_ids = p.get("response_token_ids", [])
        if not response_token_ids:
            continue

        prompt_ids = tokenizer.encode(p["full_prompt"], add_special_tokens=False)
        for cliff in cliffs:
            cliff_idx = cliff.position - 1
            if cliff_idx < 0 or cliff_idx >= len(response_token_ids):
                continue
            prefix_ids = prompt_ids + response_token_ids[:cliff_idx]
            requests.append((p, path_idx, cliff, prefix_ids, prompt_ids))

    if max_cliffs_per_dataset > 0 and len(requests) > max_cliffs_per_dataset:
        requests = requests[:max_cliffs_per_dataset]
        print(f"[{dataset}] max_cliffs_per_dataset={max_cliffs_per_dataset} applied")

    if not requests:
        print(f"[{dataset}] no cliff requests")
        return []

    sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=top_k)

    work_items: List[CliffWorkItem] = []
    total = len(requests)
    print(f"[{dataset}] extracting top-{top_k} candidates for {total} cliffs")

    total_batches = _num_chunks(total, extract_batch_size)
    batch_iter = _iter_chunks(requests, extract_batch_size)
    batch_iter = tqdm(
        batch_iter,
        total=total_batches,
        desc=f"[{dataset}] top-k batches",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for batch_idx, batch in enumerate(batch_iter, start=1):
        prompts = [{"prompt_token_ids": req[3]} for req in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        if batch_idx % 25 == 0 or batch_idx == total_batches:
            pct = 100.0 * batch_idx / max(1, total_batches)
            print(
                f"[{dataset}] top-k extraction progress: {batch_idx}/{total_batches} batches ({pct:.1f}%)",
                flush=True,
            )

        for (p, _path_idx, cliff, _prefix_ids, prompt_ids), output in zip(batch, outputs):
            sample = output.outputs[0]
            logprob_dict = sample.logprobs[0] if sample.logprobs else {}
            if not logprob_dict:
                continue

            tie_ranks = compute_tie_aware_ranks(logprob_dict)
            sorted_entries = _sort_candidate_entries(logprob_dict, tie_ranks)
            candidate_entries = sorted_entries[:top_k]
            candidates: List[CandidateMeta] = []
            for token_id, entry in candidate_entries:
                rank = _entry_rank(token_id, tie_ranks, fallback_rank=top_k + 1)
                lp = float(_logprob_value(entry))
                candidates.append(CandidateMeta(
                    token_id=int(token_id),
                    token_str=tokenizer.decode([int(token_id)], skip_special_tokens=True),
                    rank=rank,
                    logprob=lp,
                    prob=float(math.exp(lp)) if lp != float("-inf") else 0.0,
                ))

            if not candidates:
                continue

            entropy_at_t = float(compute_entropy_from_logprobs(logprob_dict))

            greedy_token_id = int(sample.token_ids[0]) if sample.token_ids else int(cliff.token_id)
            greedy_token_str = tokenizer.decode([greedy_token_id], skip_special_tokens=True)

            cliff_token_id = int(cliff.token_id)
            if cliff_token_id in logprob_dict:
                cliff_entry = logprob_dict[cliff_token_id]
                cliff_rank = _entry_rank(cliff_token_id, tie_ranks, fallback_rank=top_k + 1)
                cliff_lp = float(_logprob_value(cliff_entry))
                cliff_prob = float(math.exp(cliff_lp)) if cliff_lp != float("-inf") else 0.0
            else:
                cliff_rank = top_k + 1
                cliff_lp = float("-inf")
                cliff_prob = 0.0

            if greedy_token_id in logprob_dict:
                greedy_entry = logprob_dict[greedy_token_id]
                greedy_rank = _entry_rank(greedy_token_id, tie_ranks, fallback_rank=top_k + 1)
                greedy_lp = float(_logprob_value(greedy_entry))
                greedy_prob = float(math.exp(greedy_lp)) if greedy_lp != float("-inf") else 0.0
            else:
                greedy_rank = top_k + 1
                greedy_lp = float("-inf")
                greedy_prob = 0.0

            scores = p.get("all_position_scores", [])
            cliff_idx = cliff.position - 1
            if cliff_idx > 0 and cliff_idx < len(scores):
                pot_prev = scores[cliff_idx - 1]
                pot_curr = scores[cliff_idx]
            else:
                pot_prev = cliff.prev_score
                pot_curr = cliff.curr_score
            pot_prev = float(pot_prev) if pot_prev is not None else float(cliff.prev_score)
            pot_curr = float(pot_curr) if pot_curr is not None else float(cliff.curr_score)
            drop = pot_prev - pot_curr

            # Tie-aware: if the cliff token is in the top probability tier (rank==1,
            # including ties), classify it in the same cell as greedy. Token ID
            # comparison can produce false-negatives due to non-deterministic
            # tie-breaking in vLLM at temperature=0.
            is_eq = (cliff_rank == 1)
            cell = _cell_label(entropy_at_t, is_eq)
            is_target = True if target_cells is None else (cell in target_cells)

            topk_prob_mass = sum(c.prob for c in candidates)
            tail_mass = max(0.0, 1.0 - topk_prob_mass)

            cliff_uid = f"{dataset}::{p['id']}::{cliff.position}"
            cliff_instance = CliffInstance(
                cliff_uid=cliff_uid,
                model=model_short,
                dataset=dataset,
                path_id=str(p["id"]),
                problem_id=_safe_problem_id(p),
                path_is_correct=bool(p.get("is_correct", False)),
                cliff_position=int(cliff.position),
                cliff_token_id=cliff_token_id,
                cliff_token_str=str(cliff.token_str) if cliff.token_str is not None else "",
                cliff_token_rank=int(cliff_rank),
                cliff_token_logprob=cliff_lp,
                cliff_token_prob=float(cliff_prob),
                entropy_at_t=entropy_at_t,
                cell_label=cell,
                potential_t_minus_1=pot_prev,
                potential_t_cliff=pot_curr,
                cliff_drop=drop,
                greedy_token_id=greedy_token_id,
                greedy_token_str=greedy_token_str,
                greedy_token_rank=int(greedy_rank),
                greedy_token_logprob=greedy_lp,
                greedy_token_prob=float(greedy_prob),
                is_cliff_eq_greedy=is_eq,
                topk_prob_mass=float(topk_prob_mass),
                tail_mass=float(tail_mass),
                n_candidates=len(candidates),
                is_target_cell=is_target,
            )

            work_items.append(CliffWorkItem(
                cliff=cliff_instance,
                prompt_ids=prompt_ids,
                response_token_ids=list(p["response_token_ids"]),
                golden_answers=list(p.get("golden_answer", [])),
                candidates=candidates,
            ))

    print(f"[{dataset}] extracted cliff instances: {len(work_items)}")
    return work_items


def _run_candidate_rollouts(
    llm,
    tokenizer,
    model_path: str,
    mode: str,
    temperature: Optional[float],
    dataset: str,
    dataset_source_name: str,
    work_items: List[CliffWorkItem],
    num_samples: int,
    top_k: int,
    rollout_batch_size: int,
    max_candidates_per_cliff: int,
) -> Tuple[List[CandidateResult], List[CandidateRaw]]:
    from vllm import SamplingParams

    sampling_cfg = config.get_sampling_config_with_temperature(
        mode=mode,
        model_path=model_path,
        temperature=temperature,
    )
    max_new_tokens = config.get_rollout_max_tokens(dataset, mode)

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=sampling_cfg.temperature,
        top_p=sampling_cfg.top_p,
        top_k=sampling_cfg.top_k,
        presence_penalty=sampling_cfg.presence_penalty,
        repetition_penalty=sampling_cfg.repetition_penalty,
        max_tokens=max_new_tokens,
        stop=config.STOP_TOKENS,
    )

    requests = []
    for wi in work_items:
        cliff = wi.cliff
        cliff_idx = cliff.cliff_position - 1
        prefix_base = wi.prompt_ids + wi.response_token_ids[:cliff_idx]
        cands = wi.candidates[:max_candidates_per_cliff] if max_candidates_per_cliff > 0 else wi.candidates
        for cand in cands:
            prefix_ids = prefix_base + [cand.token_id]
            prefix_response_ids = prefix_ids[len(wi.prompt_ids):]
            prefix_text = tokenizer.decode(prefix_response_ids, skip_special_tokens=True)
            requests.append((wi, cand, prefix_ids, prefix_text))

    if not requests:
        return [], []

    candidate_results: List[CandidateResult] = []
    candidate_raws: List[CandidateRaw] = []

    total_candidates = len(requests)
    total_batches = _num_chunks(total_candidates, rollout_batch_size)
    print(
        f"[{dataset}] rollout requests: {total_candidates} candidates x {num_samples} samples "
        f"({total_batches} batches)",
        flush=True,
    )

    chunk_iter = _iter_chunks(requests, rollout_batch_size)
    chunk_iter = tqdm(
        chunk_iter,
        total=total_batches,
        desc=f"[{dataset}] rollout batches",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    processed_candidates = 0
    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        prompts = [{"prompt_token_ids": req[2]} for req in chunk]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        all_responses: List[str] = []
        all_golds: List[List[str]] = []
        response_map: List[int] = []

        for req_idx, output in enumerate(outputs):
            _wi, _cand, _prefix_ids, prefix_text = chunk[req_idx]
            for sample in output.outputs:
                all_responses.append(prefix_text + sample.text)
                all_golds.append(chunk[req_idx][0].golden_answers)
                response_map.append(req_idx)

        correctness = batch_grade_responses_mathverify(
            all_responses,
            all_golds,
            dataset_source_name,
            wall_timeout=2.0,
        )

        grouped: Dict[int, List[bool]] = defaultdict(list)
        for ridx, ok in enumerate(correctness):
            grouped[response_map[ridx]].append(bool(ok))

        greedy_potential_by_cliff: Dict[str, float] = {}
        for req_idx, (wi, cand, _prefix_ids, _prefix_text) in enumerate(chunk):
            corr = grouped.get(req_idx, [])
            if len(corr) < num_samples:
                corr = corr + [False] * (num_samples - len(corr))
            elif len(corr) > num_samples:
                corr = corr[:num_samples]

            n_correct = sum(1 for x in corr if x)
            pot_cand = n_correct / num_samples if num_samples else 0.0

            if cand.token_id == wi.cliff.greedy_token_id:
                greedy_potential_by_cliff[wi.cliff.cliff_uid] = pot_cand

            k_prev = score_to_k(wi.cliff.potential_t_minus_1, N=num_samples)
            k_curr = score_to_k(pot_cand, N=num_samples)
            is_cand_cliff = bool(is_cliff_lookup(k_prev, k_curr))

            candidate_raws.append(CandidateRaw(
                cliff_uid=wi.cliff.cliff_uid,
                model=wi.cliff.model,
                dataset=wi.cliff.dataset,
                path_id=wi.cliff.path_id,
                cliff_position=wi.cliff.cliff_position,
                candidate_token_id=cand.token_id,
                candidate_rank=cand.rank,
                candidate_correctness=corr,
            ))

            candidate_results.append(CandidateResult(
                cliff_uid=wi.cliff.cliff_uid,
                model=wi.cliff.model,
                dataset=wi.cliff.dataset,
                path_id=wi.cliff.path_id,
                cliff_position=wi.cliff.cliff_position,
                cell_label=wi.cliff.cell_label,
                cliff_token_id=wi.cliff.cliff_token_id,
                greedy_token_id=wi.cliff.greedy_token_id,
                candidate_token_id=cand.token_id,
                candidate_token_str=cand.token_str,
                candidate_rank=cand.rank,
                candidate_logprob=cand.logprob,
                candidate_prob=cand.prob,
                # Tie-aware: treat all rank==1 candidates as greedy
                # (including tied-top tokens). Token ID comparison is sensitive to tie-breaking.
                is_candidate_greedy=(cand.rank == 1),
                is_candidate_selected_cliff=(cand.token_id == wi.cliff.cliff_token_id),
                potential_t_minus_1=wi.cliff.potential_t_minus_1,
                potential_t_cliff=wi.cliff.potential_t_cliff,
                potential_t_candidate=pot_cand,
                delta_vs_t_minus_1=wi.cliff.potential_t_minus_1 - pot_cand,
                delta_vs_cliff=pot_cand - wi.cliff.potential_t_cliff,
                delta_vs_greedy=0.0,
                is_candidate_cliff_stat=is_cand_cliff,
                candidate_num_correct=n_correct,
                num_samples=num_samples,
            ))

        if greedy_potential_by_cliff:
            for row in candidate_results:
                gp = greedy_potential_by_cliff.get(row.cliff_uid)
                if gp is not None:
                    row.delta_vs_greedy = row.potential_t_candidate - gp

        processed_candidates += len(chunk)
        if chunk_idx % 25 == 0 or chunk_idx == total_batches:
            pct = 100.0 * processed_candidates / max(1, total_candidates)
            print(
                f"[{dataset}] rollout progress: {chunk_idx}/{total_batches} batches, "
                f"{processed_candidates}/{total_candidates} candidates ({pct:.1f}%)",
                flush=True,
            )

    return candidate_results, candidate_raws


def _build_cliff_summary(
    cliff_rows: List[CliffInstance],
    candidate_rows: List[CandidateResult],
) -> List[Dict]:
    by_cliff: Dict[str, List[CandidateResult]] = defaultdict(list)
    for r in candidate_rows:
        by_cliff[r.cliff_uid].append(r)

    out = []
    for c in cliff_rows:
        rows = by_cliff.get(c.cliff_uid, [])
        n_cand_cliff = sum(1 for r in rows if r.is_candidate_cliff_stat)
        cum_prob = sum(r.candidate_prob for r in rows if r.is_candidate_cliff_stat)
        cliff_ranks = [r.candidate_rank for r in rows if r.is_candidate_cliff_stat]
        min_rank = min(cliff_ranks) if cliff_ranks else None

        out.append({
            "cliff_uid": c.cliff_uid,
            "model": c.model,
            "dataset": c.dataset,
            "path_id": c.path_id,
            "cliff_position": c.cliff_position,
            "cell_label": c.cell_label,
            "is_target_cell": c.is_target_cell,
            "n_candidates": len(rows),
            "n_candidate_cliffs": n_cand_cliff,
            "cumulative_prob_candidate_cliffs": round(cum_prob, 8),
            "min_rank_candidate_cliff": min_rank,
            "tail_mass": round(c.tail_mass, 8),
            "potential_t_minus_1": c.potential_t_minus_1,
            "potential_t_cliff": c.potential_t_cliff,
            "cliff_drop": c.cliff_drop,
        })
    return out


def _build_cell_summary(
    cliff_rows: List[CliffInstance],
    cliff_summary_rows: List[Dict],
    candidate_rows: List[CandidateResult],
) -> List[Dict]:
    cliffs_by_cell: Dict[str, List[CliffInstance]] = defaultdict(list)
    for c in cliff_rows:
        cliffs_by_cell[c.cell_label].append(c)

    cliff_summary_by_cell: Dict[str, List[Dict]] = defaultdict(list)
    for r in cliff_summary_rows:
        cliff_summary_by_cell[r["cell_label"]].append(r)

    cand_by_cell: Dict[str, List[CandidateResult]] = defaultdict(list)
    for c in candidate_rows:
        cand_by_cell[c.cell_label].append(c)

    cells = [
        "low-H + greedy",
        "low-H + non-greedy",
        "high-H + greedy",
        "high-H + non-greedy",
    ]

    out = []
    for cell in cells:
        crows = cliffs_by_cell.get(cell, [])
        srows = cliff_summary_by_cell.get(cell, [])
        grows = cand_by_cell.get(cell, [])

        n_cliff = len(crows)
        n_cand = len(grows)
        avg_n_cliffs = (
            sum(r["n_candidate_cliffs"] for r in srows) / len(srows)
            if srows else 0.0
        )
        avg_cum_prob = (
            sum(r["cumulative_prob_candidate_cliffs"] for r in srows) / len(srows)
            if srows else 0.0
        )
        mean_pot = (
            sum(r.potential_t_candidate for r in grows) / len(grows)
            if grows else 0.0
        )

        out.append({
            "cell_label": cell,
            "n_cliff_instances": n_cliff,
            "n_candidate_rows": n_cand,
            "avg_n_candidate_cliffs_per_cliff": round(avg_n_cliffs, 6),
            "avg_cumulative_prob_candidate_cliffs": round(avg_cum_prob, 8),
            "mean_potential_t_candidate": round(mean_pot, 8),
        })
    return out


def _build_priority_views(candidate_rows: List[CandidateResult]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}

    p1 = [
        asdict(r) for r in candidate_rows
        if r.cell_label == "high-H + non-greedy" and r.is_candidate_greedy
    ]
    p2 = [
        asdict(r) for r in candidate_rows
        if r.cell_label == "high-H + non-greedy"
    ]
    p3 = [
        asdict(r) for r in candidate_rows
        if r.cell_label == "high-H + greedy" and (not r.is_candidate_greedy)
    ]

    out["priority_1_highH_non_greedy_greedy_baseline"] = p1
    out["priority_2_highH_non_greedy_topk"] = p2
    out["priority_3_highH_greedy_non_greedy_candidates"] = p3
    return out


def _validate_results(
    cliff_rows: List[CliffInstance],
    candidate_rows: List[CandidateResult],
    top_k: int,
    num_samples: int,
    exp3_runs_dir: Optional[Path],
) -> Dict:
    report: Dict[str, object] = {
        "n_cliffs": len(cliff_rows),
        "n_candidate_rows": len(candidate_rows),
        "top_k": top_k,
        "num_samples": num_samples,
        "checks": {},
    }

    by_cliff_expected = {c.cliff_uid: c.n_candidates for c in cliff_rows if c.is_target_cell}
    by_cliff_observed: Dict[str, int] = defaultdict(int)
    rank_violations = 0
    grid_violations = 0

    for r in candidate_rows:
        by_cliff_observed[r.cliff_uid] += 1
        if not (1 <= int(r.candidate_rank) <= top_k):
            rank_violations += 1
        n_ok = r.potential_t_candidate * num_samples
        if abs(n_ok - round(n_ok)) > 1e-9:
            grid_violations += 1

    count_mismatch = 0
    for cliff_uid, exp_n in by_cliff_expected.items():
        if by_cliff_observed.get(cliff_uid, 0) != exp_n:
            count_mismatch += 1

    report["checks"]["candidate_count_match"] = {
        "ok": count_mismatch == 0,
        "mismatched_cliffs": count_mismatch,
    }
    report["checks"]["candidate_rank_range_1_to_topk"] = {
        "ok": rank_violations == 0,
        "violations": rank_violations,
    }
    report["checks"]["potential_on_k_over_n_grid"] = {
        "ok": grid_violations == 0,
        "violations": grid_violations,
    }

    # Optional consistency check against existing exp3_entropy cliff_logprobs
    consistency = {
        "enabled": bool(exp3_runs_dir and exp3_runs_dir.exists()),
        "matched": 0,
        "entropy_mismatch": 0,
        "eq_mismatch": 0,
        "position_mismatch": 0,
    }
    if consistency["enabled"]:
        index: Dict[Tuple[str, str, int], Dict] = {}
        for c in cliff_rows:
            index[(c.dataset, c.path_id, c.cliff_position)] = {
                "entropy_at_t": c.entropy_at_t,
                "is_cliff_eq_greedy": c.is_cliff_eq_greedy,
                "cliff_position": c.cliff_position,
            }

        for ds in sorted({c.dataset for c in cliff_rows}):
            run_path = exp3_runs_dir / f"Qwen3-8B_{ds}" / "cliff_logprobs.json"
            if not run_path.exists():
                continue
            try:
                old_rows = json.load(run_path.open("r", encoding="utf-8"))
            except Exception:
                continue
            for old in old_rows:
                key = (ds, str(old.get("path_id")), int(old.get("cliff_position", -1)))
                cur = index.get(key)
                if not cur:
                    continue
                consistency["matched"] += 1
                if abs(float(old.get("entropy_at_t", 0.0)) - cur["entropy_at_t"]) > 1e-6:
                    consistency["entropy_mismatch"] += 1
                if bool(old.get("is_cliff_eq_greedy")) != bool(cur["is_cliff_eq_greedy"]):
                    consistency["eq_mismatch"] += 1
                if int(old.get("cliff_position", -1)) != int(cur["cliff_position"]):
                    consistency["position_mismatch"] += 1

    report["checks"]["consistency_vs_exp3_entropy"] = consistency
    report["ok"] = all(v.get("ok", True) for v in report["checks"].values() if isinstance(v, dict))
    return report


def _save_outputs(
    output_dir: Path,
    config_payload: Dict,
    all_cliffs: List[CliffInstance],
    all_candidate_results: List[CandidateResult],
    all_candidate_raws: List[CandidateRaw],
    top_k: int,
    num_samples: int,
    exp3_runs_dir: Optional[Path],
) -> Dict:
    cliff_rows = [asdict(c) for c in all_cliffs]
    candidate_rows = [asdict(c) for c in all_candidate_results]
    raw_rows = [asdict(c) for c in all_candidate_raws]

    cliff_summary = _build_cliff_summary(all_cliffs, all_candidate_results)
    cell_summary = _build_cell_summary(all_cliffs, cliff_summary, all_candidate_results)
    priority_views = _build_priority_views(all_candidate_results)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    _write_jsonl(output_dir / "cliff_instances.jsonl", cliff_rows)
    _write_jsonl(output_dir / "candidate_results.jsonl", candidate_rows)
    _write_jsonl(output_dir / "candidate_rollout_raw.jsonl", raw_rows)

    if cliff_rows:
        _write_csv(output_dir / "cliff_instances.csv", cliff_rows, fieldnames=list(cliff_rows[0].keys()))
    else:
        _write_csv(output_dir / "cliff_instances.csv", [], fieldnames=[
            "cliff_uid", "model", "dataset", "path_id", "problem_id", "path_is_correct",
            "cliff_position", "cliff_token_id", "cliff_token_str", "cliff_token_rank",
            "cliff_token_logprob", "cliff_token_prob", "entropy_at_t", "cell_label",
            "potential_t_minus_1", "potential_t_cliff", "cliff_drop", "greedy_token_id",
            "greedy_token_str", "greedy_token_rank", "greedy_token_logprob", "greedy_token_prob",
            "is_cliff_eq_greedy", "topk_prob_mass", "tail_mass", "n_candidates", "is_target_cell",
        ])

    if candidate_rows:
        _write_csv(output_dir / "candidate_results.csv", candidate_rows, fieldnames=list(candidate_rows[0].keys()))
    else:
        _write_csv(output_dir / "candidate_results.csv", [], fieldnames=[
            "cliff_uid", "model", "dataset", "path_id", "cliff_position", "cell_label",
            "cliff_token_id", "greedy_token_id", "candidate_token_id", "candidate_token_str",
            "candidate_rank", "candidate_logprob", "candidate_prob", "is_candidate_greedy",
            "is_candidate_selected_cliff", "potential_t_minus_1", "potential_t_cliff",
            "potential_t_candidate", "delta_vs_t_minus_1", "delta_vs_cliff", "delta_vs_greedy",
            "is_candidate_cliff_stat", "candidate_num_correct", "num_samples",
        ])

    if cliff_summary:
        _write_csv(output_dir / "cliff_summary.csv", cliff_summary, fieldnames=list(cliff_summary[0].keys()))
    else:
        _write_csv(output_dir / "cliff_summary.csv", [], fieldnames=[
            "cliff_uid", "model", "dataset", "path_id", "cliff_position", "cell_label", "is_target_cell",
            "n_candidates", "n_candidate_cliffs", "cumulative_prob_candidate_cliffs",
            "min_rank_candidate_cliff", "tail_mass", "potential_t_minus_1", "potential_t_cliff", "cliff_drop",
        ])

    if cell_summary:
        _write_csv(output_dir / "cell_summary.csv", cell_summary, fieldnames=list(cell_summary[0].keys()))
    else:
        _write_csv(output_dir / "cell_summary.csv", [], fieldnames=[
            "cell_label", "n_cliff_instances", "n_candidate_rows", "avg_n_candidate_cliffs_per_cliff",
            "avg_cumulative_prob_candidate_cliffs", "mean_potential_t_candidate",
        ])

    pv_map = {
        "priority_1_highH_non_greedy_greedy_baseline": output_dir / "priority_1_highH_non_greedy_greedy_baseline.csv",
        "priority_2_highH_non_greedy_topk": output_dir / "priority_2_highH_non_greedy_topk.csv",
        "priority_3_highH_greedy_non_greedy_candidates": output_dir / "priority_3_highH_greedy_non_greedy_candidates.csv",
    }
    for key, rows in priority_views.items():
        dest = pv_map[key]
        if rows:
            _write_csv(dest, rows, fieldnames=list(rows[0].keys()))
        else:
            _write_csv(dest, [], fieldnames=[
                "cliff_uid", "model", "dataset", "path_id", "cliff_position", "cell_label",
                "cliff_token_id", "greedy_token_id", "candidate_token_id", "candidate_token_str",
                "candidate_rank", "candidate_logprob", "candidate_prob", "is_candidate_greedy",
                "is_candidate_selected_cliff", "potential_t_minus_1", "potential_t_cliff",
                "potential_t_candidate", "delta_vs_t_minus_1", "delta_vs_cliff", "delta_vs_greedy",
                "is_candidate_cliff_stat", "candidate_num_correct", "num_samples",
            ])

    validation = _validate_results(
        cliff_rows=all_cliffs,
        candidate_rows=all_candidate_results,
        top_k=top_k,
        num_samples=num_samples,
        exp3_runs_dir=exp3_runs_dir,
    )
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    return validation


def _merge_shard_outputs(
    output_dir: Path,
    shard_dirs: List[Path],
    top_k: int,
    num_samples: int,
    exp3_runs_dir: Optional[Path],
) -> Dict:
    print(f"Merging shard outputs: {[str(d) for d in shard_dirs]}")

    cliff_map: Dict[str, Dict] = {}
    cand_map: Dict[Tuple[str, int], Dict] = {}
    raw_map: Dict[Tuple[str, int], Dict] = {}
    shard_configs: List[Dict] = []

    for shard_dir in shard_dirs:
        cfg_path = shard_dir / "config.json"
        if cfg_path.exists():
            try:
                shard_configs.append(json.load(cfg_path.open("r", encoding="utf-8")))
            except Exception:
                pass

        for row in _read_jsonl(shard_dir / "cliff_instances.jsonl"):
            cliff_map[str(row["cliff_uid"])] = row

        for row in _read_jsonl(shard_dir / "candidate_results.jsonl"):
            key = (str(row["cliff_uid"]), int(row["candidate_token_id"]))
            cand_map[key] = row

        for row in _read_jsonl(shard_dir / "candidate_rollout_raw.jsonl"):
            key = (str(row["cliff_uid"]), int(row["candidate_token_id"]))
            raw_map[key] = row

    all_cliffs = [CliffInstance(**row) for row in cliff_map.values()]
    all_candidate_results = [CandidateResult(**row) for row in cand_map.values()]
    all_candidate_raws = [CandidateRaw(**row) for row in raw_map.values()]

    all_cliffs.sort(key=lambda x: (x.dataset, x.path_id, x.cliff_position))
    all_candidate_results.sort(key=lambda x: (x.dataset, x.path_id, x.cliff_position, x.candidate_rank))
    all_candidate_raws.sort(key=lambda x: (x.dataset, x.path_id, x.cliff_position, x.candidate_rank))

    base_cfg = shard_configs[0] if shard_configs else {}
    config_payload = {
        "model": base_cfg.get("model", "qwen3-8b"),
        "model_short": base_cfg.get("model_short", "Qwen3-8B"),
        "rollout_dir": base_cfg.get("rollout_dir", ""),
        "datasets_requested": base_cfg.get("datasets_requested", []),
        "datasets_found": base_cfg.get("datasets_found", sorted({c.dataset for c in all_cliffs})),
        "datasets_skipped": base_cfg.get("datasets_skipped", []),
        "top_k": int(top_k),
        "num_samples": int(num_samples),
        "mode": base_cfg.get("mode", ""),
        "temperature": base_cfg.get("temperature", None),
        "gpus": base_cfg.get("gpus", []),
        "extract_batch_size": base_cfg.get("extract_batch_size", 128),
        "rollout_batch_size": base_cfg.get("rollout_batch_size", 16),
        "rollout_cells": base_cfg.get("rollout_cells", "all"),
        "max_cliffs_per_dataset": base_cfg.get("max_cliffs_per_dataset", 0),
        "max_candidates_per_cliff": base_cfg.get("max_candidates_per_cliff", 0),
        "greedy_99_bound_nats": GREEDY_99_BOUND_NATS,
        "exp3_runs_dir": str(exp3_runs_dir) if exp3_runs_dir else "",
        "num_shards": len(shard_dirs),
        "shard_index": None,
        "merge_mode": True,
        "merged_from_shards": [str(d) for d in shard_dirs],
    }

    validation = _save_outputs(
        output_dir=output_dir,
        config_payload=config_payload,
        all_cliffs=all_cliffs,
        all_candidate_results=all_candidate_results,
        all_candidate_raws=all_candidate_raws,
        top_k=top_k,
        num_samples=num_samples,
        exp3_runs_dir=exp3_runs_dir,
    )

    print("\n" + "=" * 60)
    print("RQ2-2 MERGE DONE")
    print("=" * 60)
    print(f"Merged cliffs:        {len(all_cliffs)}")
    print(f"Merged candidates:    {len(all_candidate_results)}")
    print(f"Merged raw rows:      {len(all_candidate_raws)}")
    print(f"Validation ok:        {validation.get('ok')}")
    print(f"Output dir:           {output_dir}")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp4_candidates top-k candidate replacement experiment")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--datasets", default="math500,gsm1k,aime25")
    parser.add_argument("--rollout_dir", default="./output/03_rollout")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--extract_batch_size", type=int, default=128)
    parser.add_argument("--rollout_batch_size", type=int, default=16,
                        help="Number of candidate prompts per vLLM generate call")
    parser.add_argument("--max_cliffs_per_dataset", type=int, default=0,
                        help="0 means no limit")
    parser.add_argument("--max_candidates_per_cliff", type=int, default=0,
                        help="0 means use top_k")
    parser.add_argument("--rollout_cells", default="all",
                        help="all OR comma-separated cell labels")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--mode", default="",
                        help="Reasoning mode override: thinking|non_thinking (default: model default)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature override (matches rollout semantics)")
    parser.add_argument("--gpu_mem", type=float, default=0.65)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Data-parallel shard count")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="0-based shard index")
    parser.add_argument("--merge_shard_dirs", default="",
                        help="Comma-separated shard output dirs to merge (merge-only mode)")
    parser.add_argument("--exp3_runs_dir", default="",
                        help="Optional path like ./output/06_entropy_rank/0408_full/runs for consistency check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    exp3_runs_dir = Path(args.exp3_runs_dir) if args.exp3_runs_dir else None

    if args.merge_shard_dirs:
        shard_dirs = [Path(p) for p in _parse_csv_arg(args.merge_shard_dirs)]
        if not shard_dirs:
            raise RuntimeError("--merge_shard_dirs was provided but no shard dirs were parsed")
        _merge_shard_outputs(
            output_dir=output_dir,
            shard_dirs=shard_dirs,
            top_k=args.top_k,
            num_samples=args.num_samples,
            exp3_runs_dir=exp3_runs_dir,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    model_path = config.resolve_model_path(args.model)
    model_short = config.get_model_short_name(model_path)
    mode = args.mode.strip() if args.mode else config.get_default_mode(model_path)
    if mode not in {"thinking", "non_thinking"}:
        raise ValueError(f"Invalid --mode '{mode}'. Expected thinking|non_thinking")

    datasets = _parse_csv_arg(args.datasets)
    rollout_dir = Path(args.rollout_dir)
    gpu_ids = [int(g) for g in _parse_csv_arg(args.gpus)]
    if not gpu_ids:
        gpu_ids = [0]

    if args.rollout_cells.strip().lower() == "all":
        target_cells: Optional[set] = None
    else:
        target_cells = set(_parse_csv_arg(args.rollout_cells))

    resolved_paths = _resolve_rollout_paths(rollout_dir, model_short, datasets)
    skipped = [d for d in datasets if d not in resolved_paths]

    print("=" * 60)
    print("RQ2-2 Top-k Cliff Replacement")
    print("=" * 60)
    print(f"Model:            {args.model} ({model_short})")
    print(f"Rollout root:     {rollout_dir}")
    print(f"Datasets req:     {datasets}")
    print(f"Datasets found:   {list(resolved_paths.keys())}")
    print(f"Datasets skipped: {skipped}")
    print(f"top_k:            {args.top_k}")
    print(f"num_samples:      {args.num_samples}")
    print(f"mode:             {mode}")
    if args.temperature is not None:
        print(f"temperature:      {args.temperature} (override)")
    print(f"gpus:             {gpu_ids}")
    print(f"rollout_cells:    {'all' if target_cells is None else sorted(target_cells)}")
    print(f"shard:            {args.shard_index + 1}/{args.num_shards}")
    print(f"Output:           {output_dir}")

    if not resolved_paths:
        raise RuntimeError("No dataset rollout files found")

    _init_heavy_imports()
    from transformers import AutoTokenizer

    llm = create_llm(model_path, gpu_ids, memory_utilization=args.gpu_mem)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_cliffs: List[CliffInstance] = []
    all_candidate_results: List[CandidateResult] = []
    all_candidate_raws: List[CandidateRaw] = []

    for ds, path in resolved_paths.items():
        print(f"\n[{ds}] loading {path}")
        with path.open("r", encoding="utf-8") as f:
            paths = json.load(f)

        orig_n = len(paths)
        paths, shard_stats = _filter_paths_for_shard(
            paths=paths,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            target_cells=target_cells,
            top_k=args.top_k,
        )
        print(f"[{ds}] shard-filtered paths: {len(paths)} / {orig_n} (shard {args.shard_index + 1}/{args.num_shards})")
        if args.num_shards > 1:
            est_loads = shard_stats.get("estimated_loads", [])
            est_cliffs = shard_stats.get("estimated_cliffs", [])
            path_counts = shard_stats.get("path_counts", [])
            if est_loads and est_cliffs and path_counts:
                print(
                    f"[{ds}] shard balance (estimated) loads={est_loads}, cliffs={est_cliffs}, paths={path_counts}",
                    flush=True,
                )
        if not paths:
            continue

        work_items = _extract_cliffs_and_candidates(
            llm=llm,
            tokenizer=tokenizer,
            paths=paths,
            model_short=model_short,
            dataset=ds,
            top_k=args.top_k,
            extract_batch_size=args.extract_batch_size,
            target_cells=target_cells,
            max_cliffs_per_dataset=args.max_cliffs_per_dataset,
        )

        all_cliffs.extend([w.cliff for w in work_items])

        rollout_items = [w for w in work_items if w.cliff.is_target_cell]
        print(f"[{ds}] target-cell cliffs for rollout: {len(rollout_items)} / {len(work_items)}")
        if not rollout_items:
            continue

        source_name = config.get_dataset_source_name(ds)
        cand_results, cand_raws = _run_candidate_rollouts(
            llm=llm,
            tokenizer=tokenizer,
            model_path=model_path,
            mode=mode,
            temperature=args.temperature,
            dataset=ds,
            dataset_source_name=source_name,
            work_items=rollout_items,
            num_samples=args.num_samples,
            top_k=args.top_k,
            rollout_batch_size=args.rollout_batch_size,
            max_candidates_per_cliff=args.max_candidates_per_cliff,
        )
        all_candidate_results.extend(cand_results)
        all_candidate_raws.extend(cand_raws)

    config_payload = {
        "model": args.model,
        "model_short": model_short,
        "rollout_dir": str(rollout_dir),
        "datasets_requested": datasets,
        "datasets_found": list(resolved_paths.keys()),
        "datasets_skipped": skipped,
        "top_k": args.top_k,
        "num_samples": args.num_samples,
        "mode": mode,
        "temperature": args.temperature,
        "gpus": gpu_ids,
        "extract_batch_size": args.extract_batch_size,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_cells": "all" if target_cells is None else sorted(target_cells),
        "max_cliffs_per_dataset": args.max_cliffs_per_dataset,
        "max_candidates_per_cliff": args.max_candidates_per_cliff,
        "greedy_99_bound_nats": GREEDY_99_BOUND_NATS,
        "exp3_runs_dir": str(exp3_runs_dir) if exp3_runs_dir else "",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "merge_mode": False,
    }

    validation = _save_outputs(
        output_dir=output_dir,
        config_payload=config_payload,
        all_cliffs=all_cliffs,
        all_candidate_results=all_candidate_results,
        all_candidate_raws=all_candidate_raws,
        top_k=args.top_k,
        num_samples=args.num_samples,
        exp3_runs_dir=exp3_runs_dir,
    )

    print("\n" + "=" * 60)
    print("RQ2-2 DONE")
    print("=" * 60)
    print(f"Cliff instances:      {len(all_cliffs)}")
    print(f"Candidate rows:       {len(all_candidate_results)}")
    print(f"Raw rollout rows:     {len(all_candidate_raws)}")
    print(f"Validation ok:        {validation.get('ok')}")
    print(f"Output dir:           {output_dir}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
