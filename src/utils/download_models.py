#!/usr/bin/env python3
"""
Model Downloader for Cliff Token Analysis

Downloads the paper reproduction model set from HuggingFace to ./model/.
After downloading, use the local path or alias with the --model argument:
  python -m src.cli inference --model qwen3-4b --dataset math500 ...

Paper model set (7 models):
  qwen3-0.6b    → Qwen/Qwen3-0.6B                       → model/Qwen3-0.6B/
  qwen3-4b      → Qwen/Qwen3-4B                         → model/Qwen3-4B/
  qwen3-8b      → Qwen/Qwen3-8B                         → model/Qwen3-8B/
  llama-3.2-1b  → meta-llama/Llama-3.2-1B-Instruct      → model/Llama-3.2-1B-Instruct/
  llama-3.2-3b  → meta-llama/Llama-3.2-3B-Instruct      → model/Llama-3.2-3B-Instruct/
  llama-3.1-8b  → meta-llama/Llama-3.1-8B-Instruct      → model/Llama-3.1-8B-Instruct/
  gemma-3-4b    → google/gemma-3-4b-it                  → model/gemma-3-4b-it/

Usage:
  python download_models.py                                  # download paper set
  python download_models.py --model qwen3-4b                 # specific model only
  python download_models.py --model qwen3-4b llama-3.2-1b
  python download_models.py --output_dir ./model             # default
  python download_models.py --hf_token YOUR_TOKEN            # gated Llama/Gemma
"""

import argparse
import os
import sys
from pathlib import Path


# =============================================================================
# Default Paths & Token
# =============================================================================

DEFAULT_OUTPUT_DIR = "./model"


# =============================================================================
# Model Registry (alias → HuggingFace repo ID)
# =============================================================================

MODEL_REGISTRY = {
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",
    "qwen3-4b":     "Qwen/Qwen3-4B",
    "qwen3-8b":     "Qwen/Qwen3-8B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-3-4b":   "google/gemma-3-4b-it",
}
# Local directory name (last part of repo ID)
def _local_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


# =============================================================================
# Download Function
# =============================================================================

def download_model(
    alias: str,
    output_dir: Path,
    hf_token: str = None,
    force: bool = False,
) -> Path:
    """Download a single model to output_dir/<ModelName>/."""
    from huggingface_hub import snapshot_download

    repo_id = MODEL_REGISTRY[alias]
    local_name = _local_name(repo_id)
    local_dir = output_dir / local_name

    # Skip if already downloaded (non-empty directory)
    if not force and local_dir.exists() and any(local_dir.iterdir()):
        size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e6
        print(f"  [{alias}] Already downloaded → {local_dir} ({size_mb:.0f} MB). Use --force to re-download.")
        return local_dir

    print(f"  [{alias}] Downloading {repo_id} → {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "ignore_patterns": ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    }
    if hf_token:
        kwargs["token"] = hf_token

    snapshot_download(**kwargs)

    size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e6
    print(f"  [{alias}] Done → {local_dir} ({size_mb:.0f} MB)")
    return local_dir


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download models for Cliff Token Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Paper model set:
  qwen3-0.6b    → Qwen/Qwen3-0.6B
  qwen3-4b      → Qwen/Qwen3-4B
  qwen3-8b      → Qwen/Qwen3-8B
  llama-3.2-1b  → meta-llama/Llama-3.2-1B-Instruct
  llama-3.2-3b  → meta-llama/Llama-3.2-3B-Instruct
  llama-3.1-8b  → meta-llama/Llama-3.1-8B-Instruct
  gemma-3-4b    → google/gemma-3-4b-it

After downloading, use aliases or local paths:
  python -m src.cli inference --model qwen3-4b --dataset math500 ...
  python -m src.cli inference --model ./model/Llama-3.2-1B-Instruct --dataset gsm1k ...

Examples:
  python download_models.py
  python download_models.py --model qwen3-4b
  python download_models.py --model qwen3-4b qwen3-8b llama-3.2-1b
  python download_models.py --hf_token hf_xxx  # for gated Llama/Gemma
        """
    )
    parser.add_argument(
        "--model", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Model(s) to download (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token (required for gated models: Llama, Gemma). "
             "Can also be set via the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if already present",
    )
    args = parser.parse_args()

    # HF token: CLI flag > HF_TOKEN env var > HUGGING_FACE_HUB_TOKEN env var
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Llama and Gemma are gated models — require HF token + license acceptance.
    gated = [m for m in args.model if m.startswith("gemma") or m.startswith("llama")]
    if gated and not hf_token:
        print("WARNING: The following models are gated and require a HuggingFace token:")
        for m in gated:
            print(f"  {m}")
        print("  Set --hf_token YOUR_TOKEN  or  export HF_TOKEN=YOUR_TOKEN")
        print("  Also accept the license on HuggingFace Hub before downloading:")
        print("    Llama 3.2 1B: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("    Llama 3.2 3B: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        print("    Llama 3.1 8B: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("    Gemma: https://huggingface.co/google/gemma-3-4b-it")
        print()

    failed = []
    downloaded = []

    for alias in args.model:
        repo_id = MODEL_REGISTRY[alias]
        print(f"{'=' * 55}")
        print(f"Model: {alias} ({repo_id})")
        print(f"{'=' * 55}")
        try:
            local_dir = download_model(alias, output_dir, hf_token=hf_token, force=args.force)
            downloaded.append((alias, local_dir))
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(alias)
            print()

    print("=" * 55)
    print("DONE")
    print("=" * 55)
    if downloaded:
        print("Downloaded models:")
        for alias, local_dir in downloaded:
            print(f"  {alias:<14} → {local_dir}")
    if failed:
        print(f"\nFailed: {', '.join(failed)}")

    if downloaded:
        print("\nUsage with local paths:")
        for alias, local_dir in downloaded:
            print(f"  python -m src.cli inference --model {local_dir} --dataset math500 ...")
        print()
        print("Aliases are already configured in src/config.py for the paper model set.")


if __name__ == "__main__":
    main()
