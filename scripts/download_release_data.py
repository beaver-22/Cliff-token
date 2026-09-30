"""Download the public dataset and place its JSON files in the pipeline directories."""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("CLIFF_TOKEN_DATASET", "Beaverdam/cliff-token-data"),
                        help="Hugging Face dataset ID (or set CLIFF_TOKEN_DATASET)")
    args = parser.parse_args()
    dataset = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["stem_traces/*/*.json", "rollouts/*/*.json"],
    ))
    copied = 0
    for source_dir, target_dir in (("stem_traces", "01_stem_traces"),
                                   ("rollouts", "03_rollouts")):
        for source in sorted((dataset / source_dir).glob("*/*.json")):
            name = source.stem
            if name in {"gsm1k", "math500"}:
                name += "_100"
            target = ROOT / "output" / target_dir / source.parent.name / f"{name}_all_paths.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            print(target.relative_to(ROOT))
            copied += 1
    if copied == 0:
        parser.error("no JSON data files found in the dataset repository")


if __name__ == "__main__":
    main()
