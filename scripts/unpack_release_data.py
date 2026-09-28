"""Expand the released rollout files into the pipeline's two input directories."""

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    for source in sorted((ROOT / "release_data" / "rollouts").glob("*/*.json.gz")):
        model = source.parent.name
        dataset = source.name.removesuffix(".json.gz")
        if dataset in {"gsm1k", "math500"}:
            dataset += "_100"
        filename = f"{dataset}_all_paths.json"

        with gzip.open(source, "rt", encoding="utf-8") as handle:
            rollouts = json.load(handle)
        stems = [
            {key: value for key, value in trace.items() if key != "all_position_scores"}
            for trace in rollouts
        ]

        for directory, traces in (("01_stem_traces", stems), ("03_rollouts", rollouts)):
            target = ROOT / "output" / directory / model / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(traces, handle, ensure_ascii=False)
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
