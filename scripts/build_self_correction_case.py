#!/usr/bin/env python3
"""Figure 4: success probability across a self-correcting trace.

Reads the Qwen3-4B AIME 2025 Problem 1 rollout, runs the statistical cliff
detector, and writes a self-contained snapshot to
``figure/data/fig4_self_correction/trace.json``. The drawing lives in
``figure/_fig_self_correction.py`` so that ``figure/figure.ipynb`` can
regenerate the figure from the snapshot alone.

Outputs:
  figure/data/fig4_self_correction/trace.json
  figure/output/fig4_self_correction.{png,pdf}
  figure/output/fig4_self_correction_legend.{png,pdf}

Usage:
  python3 scripts/build_self_correction_case.py [--rollout_dir output/03_rollouts]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'figure'))

import matplotlib
matplotlib.use('Agg')

from _fig_self_correction import render  # noqa: E402  (figure/ is on sys.path)
from src.analysis.detector import find_all_cliff_tokens_statistical  # noqa: E402

MODEL, DATASET, TRACE_ID, N_ROLLOUTS = 'Qwen3-4B', 'aime25', '1_path_0', 64
DATA_DIR = REPO / 'figure' / 'data' / 'fig4_self_correction'
OUT_DIR = REPO / 'figure' / 'output'
# the uniformly high-probability algebra before the first cliff is replaced by a
# single marker; boundaries are token positions, the span itself stays verbatim.
# ELIDE_BEFORE = 584 resumes at the '### Step 4' heading, so the shown text starts
# exactly where the probability curve leaves its collapsed span.
ELIDE_AFTER, ELIDE_BEFORE = 50, 584


def _error_span(tokens: list[str]) -> tuple[int, int]:
    """Token span of the first occurrence of the miscomputed value 443."""
    start = next(
        i for i in range(len(tokens) - 2)
        if (tokens[i].strip(), tokens[i + 1].strip(), tokens[i + 2].strip()) == ('4', '4', '3')
    )
    return start + 1, start + 3


def build_snapshot(rollout_dir: Path) -> dict:
    source = rollout_dir / MODEL / f'{DATASET}_all_paths.json'
    trace = next(
        row for row in json.loads(source.read_text())
        if row['id'] == TRACE_ID
    )
    tokens = trace['response_tokens']
    scores = trace['all_position_scores']
    cliffs = find_all_cliff_tokens_statistical(
        scores, tokens=tokens, token_ids=trace['response_token_ids'], N=N_ROLLOUTS
    )

    def potential(position: int) -> float:
        return scores[min(position, len(scores)) - 1]

    cue = next(i for i in range(cliffs[-1].position, len(tokens))
               if 'wait' in tokens[i].lower()) + 1
    # the rebound spans "But wait", not the single "wait" token
    cue_start = cue - 1 if tokens[cue - 2].strip() == 'But' else cue
    error = _error_span(tokens)

    marks = [
        {'span': [cliff.position, cliff.position], 'role': 'cliff',
         'label': f'cliff token {index}',
         'value': f'{cliff.prev_score:.2f}→{cliff.curr_score:.2f}'}
        for index, cliff in enumerate(cliffs, start=1)
    ]
    marks.insert(1, {
        'span': list(error), 'role': 'error',
        'label': 'wrong value 443 (should be 448)', 'value': '',
    })
    marks.append({
        'span': [cue_start, cue], 'role': 'cue', 'label': 'self-correction',
        'value': f'{potential(cue_start - 1):.2f}→{potential(cue):.2f}',
    })

    return {
        'source': f'output/03_rollouts/{MODEL}/{DATASET}_all_paths.json',
        'model': MODEL,
        'dataset': DATASET,
        'trace_id': TRACE_ID,
        'problem_id': trace['problem_id'],
        'question': trace['question'],
        'golden_answer': trace['golden_answer'],
        'is_correct': trace['is_correct'],
        'n_rollouts_per_position': N_ROLLOUTS,
        'tokens': tokens,
        'scores': scores,
        'marks': marks,
        'elide': [ELIDE_AFTER, ELIDE_BEFORE],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollout_dir', type=Path, default=REPO / 'output' / '03_rollouts')
    args = ap.parse_args()
    snapshot = build_snapshot(args.rollout_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / 'trace.json'
    path.write_text(json.dumps(snapshot, ensure_ascii=False) + '\n')
    print(f'wrote {path} ({len(snapshot["tokens"])} tokens, '
          f'{len(snapshot["marks"])} marks)')

    # same callout wording the notebook uses, built from the detected numbers
    labels = [f"{m['label']}  {m['value']}".rstrip() for m in snapshot['marks']]
    info = render(snapshot, OUT_DIR, labels=labels)
    print(f'wrote {OUT_DIR}/fig4_self_correction{{,_legend}}.png/.pdf | '
          f'{info["atoms"]} atoms ({info["formulas"]} typeset formulas), '
          f'{info["lines"]} lines, font {info["font"]}, '
          f'figure {info["figure_inches"][0]}x{info["figure_inches"][1]} in')
    print(f'elided span: {info["elided_tokens"]} scored tokens, '
          f'p(success) {info["elided_probability"][0]}-{info["elided_probability"][1]}')


if __name__ == '__main__':
    main()
