#!/usr/bin/env python3
"""
Dataset Downloader for Cliff Token Analysis

Downloads directly via the HuggingFace Datasets Server REST API.
(No load_dataset or HF authentication required)

Sources:
  gsm1k   → ScaleAI/gsm1k          (test,  ~1205 problems) → data/input/GSM1K_test.jsonl
  math500 → HuggingFaceH4/MATH-500  (test,   500 problems)  → data/input/MATH_test.jsonl
  aime24  → HuggingFaceH4/aime_2024 (train,   30 problems)  → data/input/aime24.jsonl
  aime25  → MathArena/aime_2025     (train,   30 problems)  → data/input/aime25.jsonl

Output JSONL format:
  {"id": "GSM1K_0", "question": "...", "answer": ["42"]}

Usage:
  python download_datasets.py                      # download all
  python download_datasets.py --dataset gsm1k      # specific dataset
  python download_datasets.py --dataset math500 aime24
  python download_datasets.py --output_dir ./data/input
"""

import json
import argparse
from pathlib import Path


# =============================================================================
# HuggingFace Datasets Server API helper
# =============================================================================

HF_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # max rows per API call


def _fetch_all_rows(dataset: str, config: str, split: str) -> list:
    """Paginate through all rows via HF Datasets Server REST API.

    Handles transient 429 (rate limit) with exponential backoff.
    """
    import time
    import urllib.request
    import urllib.error

    rows = []
    offset = 0
    while True:
        url = (
            f"{HF_API}"
            f"?dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={urllib.parse.quote(config, safe='')}"
            f"&split={urllib.parse.quote(split, safe='')}"
            f"&offset={offset}&length={PAGE_SIZE}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        data = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    wait = 2 ** attempt
                    print(f"    429 rate limit, backing off {wait}s (attempt {attempt+1}/6)")
                    time.sleep(wait)
                    continue
                raise
        if data is None:
            raise RuntimeError(f"failed to fetch offset={offset} after retries")

        page = [item["row"] for item in data.get("rows", [])]
        rows.extend(page)

        if len(page) < PAGE_SIZE:
            break  # last page
        offset += PAGE_SIZE
        time.sleep(0.1)  # gentle pacing to avoid 429

    return rows


# Need urllib.parse for quoting
import urllib.parse


# =============================================================================
# Download Functions
# =============================================================================

def download_gsm1k(output_dir: Path) -> Path:
    """
    Download GSM1K from ScaleAI/gsm1k via HF Datasets Server.
    Fields: question (str), answer (str, integer answer)
    """
    print("Downloading GSM1K (ScaleAI/gsm1k)...")
    rows = _fetch_all_rows("ScaleAI/gsm1k", "default", "test")

    output_path = output_dir / "GSM1K_test.jsonl"
    records = []
    for i, row in enumerate(rows):
        records.append({
            "id": f"GSM1K_{i}",
            "question": row["question"].strip(),
            "answer": [str(row["answer"]).strip()],
        })

    _save_jsonl(records, output_path)
    print(f"  Saved {len(records)} problems → {output_path}")
    return output_path


def download_math500(output_dir: Path) -> Path:
    """
    Download MATH-500 from HuggingFaceH4/MATH-500 via HF Datasets Server.
    Fields: problem (str), answer (str, LaTeX), unique_id, subject, level
    """
    print("Downloading MATH-500 (HuggingFaceH4/MATH-500)...")
    rows = _fetch_all_rows("HuggingFaceH4/MATH-500", "default", "test")

    output_path = output_dir / "MATH_test.jsonl"
    records = []
    for i, row in enumerate(rows):
        uid = row.get("unique_id") or f"MATH_{i}"
        records.append({
            "id": str(uid),
            "question": row["problem"].strip(),
            "answer": [str(row["answer"]).strip()],
            "level": row.get("level", ""),
            "subject": row.get("subject", ""),
        })

    _save_jsonl(records, output_path)
    print(f"  Saved {len(records)} problems → {output_path}")
    return output_path


def download_aime24(output_dir: Path) -> Path:
    """
    Download AIME 2024 from HuggingFaceH4/aime_2024 via HF Datasets Server.
    30 problems from AIME 2024 I & II.
    Fields: id, problem, answer (integer 0-999), solution, url, year
    """
    print("Downloading AIME 2024 (HuggingFaceH4/aime_2024)...")
    rows = _fetch_all_rows("HuggingFaceH4/aime_2024", "default", "train")

    output_path = output_dir / "aime24.jsonl"
    records = []
    for i, row in enumerate(rows):
        uid = row.get("id") or f"AIME24_{i}"
        answer = str(int(row["answer"])) if str(row["answer"]).strip().isdigit() else str(row["answer"]).strip()
        records.append({
            "id": str(uid),
            "question": row["problem"].strip(),
            "answer": [answer],
        })

    _save_jsonl(records, output_path)
    print(f"  Saved {len(records)} problems → {output_path}")
    return output_path


def download_gsm8k(output_dir: Path) -> list:
    """
    Download GSM8K (openai/gsm8k) via HuggingFace `datasets` library.

    Saves:
      - Train split (~7473 problems) into 3 shards of 2500/2500/2473 with
        globally-unique ids (GSM8K_0 ... GSM8K_7472) so multi-server rollout
        results can be concatenated later.
      - Full test split (~1319 problems) as GSM8K_test.jsonl.

    Fields: question (str), answer (str, CoT ending in "#### <final>")
    """
    print("Downloading GSM8K (openai/gsm8k)...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")

    def _to_records(split_rows, id_prefix: str) -> list:
        out = []
        for i, row in enumerate(split_rows):
            raw_answer = str(row["answer"])
            if "####" in raw_answer:
                final = raw_answer.split("####")[-1].strip()
            else:
                final = raw_answer.strip()
            out.append({
                "id": f"{id_prefix}_{i}",
                "question": row["question"].strip(),
                "answer": [final],
            })
        return out

    train_records = _to_records(ds["train"], "GSM8K")
    test_records = _to_records(ds["test"], "GSM8K_test")

    paths = []
    # Train: 2500 / 2500 / 2473 shards
    shards = [train_records[0:2500], train_records[2500:5000], train_records[5000:]]
    for idx, shard in enumerate(shards, start=1):
        out_path = output_dir / f"GSM8K_train_{idx}.jsonl"
        _save_jsonl(shard, out_path)
        print(f"  Saved {len(shard)} problems → {out_path}")
        paths.append(out_path)

    # Test split (kept for later use)
    test_path = output_dir / "GSM8K_test.jsonl"
    _save_jsonl(test_records, test_path)
    print(f"  Saved {len(test_records)} problems → {test_path}")
    paths.append(test_path)

    return paths


def download_aime25(output_dir: Path) -> Path:
    """
    Download AIME 2025 from MathArena/aime_2025 via HF Datasets Server.
    30 problems from AIME 2025 I & II.
    Fields: problem_idx, problem, answer (integer), problem_type
    """
    print("Downloading AIME 2025 (MathArena/aime_2025)...")
    rows = _fetch_all_rows("MathArena/aime_2025", "default", "train")

    output_path = output_dir / "aime25.jsonl"
    records = []
    for i, row in enumerate(rows):
        uid = row.get("problem_idx") or f"AIME25_{i}"
        records.append({
            "id": str(uid),
            "question": row["problem"].strip(),
            "answer": [str(row["answer"]).strip()],
        })

    _save_jsonl(records, output_path)
    print(f"  Saved {len(records)} problems → {output_path}")
    return output_path


# =============================================================================
# Utility
# =============================================================================

def _save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


DATASET_FUNCS = {
    "gsm1k":          download_gsm1k,
    "gsm8k":          download_gsm8k,
    "math500":        download_math500,
    "aime24":         download_aime24,
    "aime25":         download_aime25,
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Cliff Token Analysis (via HF Datasets Server API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  gsm1k          → GSM1K_test.jsonl           (~1205 problems, ScaleAI/gsm1k)
  gsm8k          → GSM8K_train_{1,2,3}.jsonl  (~7473 problems, openai/gsm8k, 3-shard split)
  math500        → MATH_test.jsonl            (  500 problems, HuggingFaceH4/MATH-500)
  aime24         → aime24.jsonl               (   30 problems, HuggingFaceH4/aime_2024)
  aime25         → aime25.jsonl               (   30 problems, MathArena/aime_2025)

Examples:
  python download_datasets.py
  python download_datasets.py --dataset gsm1k math500
  python download_datasets.py --dataset aime24 --output_dir ./data/input
        """
    )
    parser.add_argument(
        "--dataset", nargs="+",
        choices=list(DATASET_FUNCS.keys()),
        default=list(DATASET_FUNCS.keys()),
        help="Dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/input",
        help="Output directory (default: ./data/input)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print()

    failed = []
    for name in args.dataset:
        print(f"{'=' * 50}")
        print(f"Dataset: {name.upper()}")
        print(f"{'=' * 50}")
        try:
            DATASET_FUNCS[name](output_dir)
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(name)
            print()

    print("=" * 50)
    print("DONE")
    print("=" * 50)
    succeeded = [d for d in args.dataset if d not in failed]
    if succeeded:
        print(f"Downloaded: {', '.join(succeeded)}")
    if failed:
        print(f"Failed:     {', '.join(failed)}")
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.glob("*.jsonl")):
        lines = sum(1 for _ in open(f))
        print(f"  {f.name:<30} {lines} problems")


if __name__ == "__main__":
    main()
