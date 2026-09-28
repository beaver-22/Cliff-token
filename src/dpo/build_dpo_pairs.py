"""
Cliff-DPO pair construction from candidate rollout results.

At each cliff position:
    rejected = the cliff token
    chosen   = each top-10 candidate whose estimated success probability does
               NOT satisfy the cliff criterion against prev_score
               (i.e. the candidate would not itself be a cliff token)

Pairs are split by the cliff type of the rejected token.

Usage:
    python -m src.dpo.build_dpo_pairs \
        --candidates_path ./output/07_cliff_dpo/01_candidates/Qwen3-0.6B/gsm8k_train_cliff_candidates.json \
        --output_dir ./output/07_cliff_dpo/02_pairs/Qwen3-0.6B/

Outputs (suffix = dataset name, e.g. gsm8k_train):
    cliff_deterministic_only_<dataset>.json
    cliff_uncertainty_only_<dataset>.json
    cliff_sampled_off_only_<dataset>.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

from src import config
from src.analysis.cliff_threshold import is_cliff_lookup, score_to_k
from src.dpo.logging_utils import parse_log_level, setup_logger

logger = logging.getLogger("dpo.build_pairs")


# ============================================================
# DPO Pair data structure
# ============================================================

def make_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    prompt_token_ids: List[int],
    chosen_token_id: int,
    rejected_token_id: int,
    path_id: str,
    cliff_position: int,
    category: str,
    chosen_potential: float,
    rejected_potential: float,
    prev_score: float,
) -> Dict:
    """Construct a DPO pair record.

    Both text and token-id fields are stored:
    - text fields (prompt/chosen/rejected): for inspection
    - token-id fields (prompt_token_ids/chosen_token_id/rejected_token_id):
      for token-level DPO that bypasses BPE re-tokenization at the boundary.
      Loss will flow on EXACTLY one token (cliff token vs candidate).
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_token_ids": prompt_token_ids,
        "chosen_token_id": chosen_token_id,
        "rejected_token_id": rejected_token_id,
        "path_id": path_id,
        "cliff_position": cliff_position,
        "category": category,
        "chosen_potential": chosen_potential,
        "rejected_potential": rejected_potential,
        "prev_score": prev_score,
    }


# ============================================================
# Cliff z-test check for candidate potential
# ============================================================

def _is_candidate_cliff(prev_score: float, candidate_potential: float, N: int = 64) -> bool:
    """Check if a candidate token would be classified as a cliff.

    A candidate is a cliff if the drop from prev_score to candidate_potential
    passes the statistical z-test threshold.
    """
    k_prev = score_to_k(prev_score, N)
    k_cand = score_to_k(candidate_potential, N)
    return is_cliff_lookup(k_prev, k_cand)


# ============================================================
# Pair construction
# ============================================================

def build_pairs(
    analyses: List[Dict],
) -> List[Dict]:
    """Cliff token (rejected) vs each non-cliff candidate (chosen).

    chosen condition: candidate's potential does NOT trigger cliff z-test
    against prev_score. i.e., the candidate would not be classified as a cliff.
    """
    pairs = []
    for analysis in analyses:
        cliff_token = None
        non_cliff_candidates = []

        for cand in analysis["candidates"]:
            if cand["is_cliff_token"]:
                cliff_token = cand
            else:
                non_cliff_candidates.append(cand)

        if cliff_token is None:
            continue

        prev_score = analysis["prev_score"]
        prompt = analysis["prefix_text"]
        prompt_ids = analysis.get("prefix_token_ids", [])
        rejected_text = cliff_token["token_str"]
        rejected_potential = cliff_token["potential"]
        rejected_token_id = cliff_token["token_id"]

        for cand in non_cliff_candidates:
            if cand["potential"] < 0:
                continue  # Not yet evaluated

            # chosen condition: candidate is NOT a cliff (z-test)
            if _is_candidate_cliff(prev_score, cand["potential"]):
                continue

            pairs.append(make_pair(
                prompt=prompt,
                chosen=cand["token_str"],
                rejected=rejected_text,
                prompt_token_ids=prompt_ids,
                chosen_token_id=cand["token_id"],
                rejected_token_id=rejected_token_id,
                path_id=analysis["path_id"],
                cliff_position=analysis["cliff_position"],
                category=analysis["category"],
                chosen_potential=cand["potential"],
                rejected_potential=rejected_potential,
                prev_score=prev_score,
            ))

    return pairs


# ============================================================
# Category filtering
# ============================================================

def filter_by_category(pairs: List[Dict], categories: List[str]) -> List[Dict]:
    return [p for p in pairs if p["category"] in categories]


# ============================================================
# Save to HuggingFace DPO format
# ============================================================

def save_dpo_dataset(pairs: List[Dict], output_path: str):
    """Save DPO pairs as JSON (HuggingFace datasets compatible)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {len(pairs)} pairs to {output_path}")


def _resolve_existing_path_case(path: str) -> str:
    """Resolve an existing path case-insensitively.

    Useful when users pass `Qwen3-0.6B` vs `Qwen3-0.6b` and only one exists.
    Returns the original input if no unique case-insensitive match is found.
    """
    if os.path.exists(path):
        return path

    abs_path = os.path.abspath(path)
    parts = Path(abs_path).parts
    if not parts:
        return path

    current = parts[0]
    for part in parts[1:]:
        if not os.path.isdir(current):
            return path
        try:
            entries = os.listdir(current)
        except OSError:
            return path

        matches = [name for name in entries if name.lower() == part.lower()]
        if len(matches) != 1:
            return path
        current = os.path.join(current, matches[0])

    return current if os.path.exists(current) else path


# ============================================================
# Full pipeline
# ============================================================

CATEGORY_OUTPUTS = [
    ("deterministic", "cliff_deterministic_only"),
    ("uncertain", "cliff_uncertainty_only"),
    ("sampled_off", "cliff_sampled_off_only"),
]


def build_all_dpo_datasets(
    candidates_path: str,
    output_dir: str,
    dataset_hint: str = "",
) -> Dict[str, str]:
    """Build per-cliff-type DPO datasets from candidate rollout results.

    Output files are suffixed with `dataset_hint` so running on multiple
    datasets won't overwrite each other.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load candidates (supports both final .json and partial .jsonl)
    resolved_candidates_path = _resolve_existing_path_case(candidates_path)
    if resolved_candidates_path != candidates_path:
        logger.warning(
            "Candidates path case mismatch detected; using existing path: "
            f"{resolved_candidates_path}"
        )
    if not os.path.exists(resolved_candidates_path):
        raise FileNotFoundError(
            f"Candidates file not found: {candidates_path} "
            f"(also checked case-insensitive path: {resolved_candidates_path})"
        )

    logger.info(f"Loading candidates from {resolved_candidates_path}")
    if resolved_candidates_path.endswith(".jsonl"):
        analyses = []
        with open(resolved_candidates_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    analyses.append(json.loads(line))
    else:
        with open(resolved_candidates_path) as f:
            analyses = json.load(f)
    logger.info(f"  {len(analyses)} cliff analyses loaded")

    suffix = f"_{dataset_hint}" if dataset_hint else ""

    pairs = build_pairs(analyses)
    logger.info(f"Total pairs: {len(pairs)}")

    outputs = {}
    for category, stem in CATEGORY_OUTPUTS:
        filtered = filter_by_category(pairs, [category])
        if not filtered:
            logger.warning(f"  No {category} pairs")
            continue
        path = os.path.join(output_dir, f"{stem}{suffix}.json")
        save_dpo_dataset(filtered, path)
        outputs[stem] = path

    logger.info("=" * 60)
    logger.info("DPO Dataset Summary:")
    for name, path in outputs.items():
        with open(path) as f:
            n = len(json.load(f))
        logger.info(f"  {name}: {n} pairs")

    return outputs


# ============================================================
# CLI
# ============================================================

def _infer_output_dir_from_candidates(candidates_path: str) -> str:
    """Infer canonical model short name from candidates path.

    Example:
    ./output/07_cliff_dpo/01_candidates/Qwen3-0.6b/gsm8k_train_cliff_candidates.json
    -> ./output/07_cliff_dpo/02_pairs/Qwen3-0.6B/
    """
    parts = os.path.normpath(candidates_path).split(os.sep)
    # Find candidate-stage segment and grab next segment as model name.
    for i, part in enumerate(parts):
        if part == "01_candidates" and i + 1 < len(parts):
            model_raw = parts[i + 1]
            model_short = config.get_model_short_name(config.resolve_model_path(model_raw))
            return f"./output/07_cliff_dpo/02_pairs/{model_short}"

    # Fallback: use parent dir basename, then canonicalize if possible.
    parent = os.path.basename(os.path.dirname(candidates_path))
    model_short = config.get_model_short_name(config.resolve_model_path(parent))
    return os.path.join("./output/07_cliff_dpo/02_pairs", model_short)


def main():
    parser = argparse.ArgumentParser(description="Build DPO pairs from candidate rollouts")
    parser.add_argument("--candidates_path", required=True, help="Path to cliff_candidates.json (or .jsonl)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir. Default: inferred ./output/07_cliff_dpo/02_pairs/{model_short}/")
    parser.add_argument("--log_dir", default="./output/07_cliff_dpo/logs")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    output_dir = args.output_dir or _infer_output_dir_from_candidates(args.candidates_path)

    dataset_hint = os.path.basename(args.candidates_path).split("_cliff")[0]
    global logger
    logger = setup_logger(
        name=f"build_pairs_{dataset_hint}",
        log_dir=args.log_dir,
        level=parse_log_level(args.log_level),
    )
    logger.info(f"Config: candidates={args.candidates_path}, output={output_dir}")

    build_all_dpo_datasets(
        candidates_path=args.candidates_path,
        output_dir=output_dir,
        dataset_hint=dataset_hint,
    )


if __name__ == "__main__":
    main()
