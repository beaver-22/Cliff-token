# Figure Regeneration

This folder is the portable figure package for the Cliff Token Analysis paper. It contains the notebook, reduced plotting data, curated examples, and generated PNG/PDF outputs.

## Layout

```text
figure/
├── figure_revision.ipynb      # current paper figure notebook
├── figure.ipynb               # older working notebook
├── figure_yukyung.ipynb       # older working notebook
├── _build_data.py             # optional extractor from full output/ runs
├── curated_examples/          # example images and source JSON/CSV snippets
├── data/                      # reduced data read by the notebook
├── output/                    # current generated paper PNG/PDF files
└── output_recovery/           # preserved recovery/legacy PNG/PDF files
```

## Regenerate Figures

From `cliff_token_code/`:

```bash
jupyter nbconvert --to notebook --execute figure/figure_revision.ipynb \
  --output figure_revision.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The notebook writes regenerated assets to `figure/output/`.

## Paper Images

The exact PDFs used by the paper LaTeX source are stored separately in `../paper_images/`. Use those when the goal is to inspect the submitted/pre-rendered paper image; use this folder when the goal is to regenerate figures from code and reduced data.

## Data Lineage

`figure/data/` is a reduced, portable snapshot derived from experiment outputs under `output/`. Rebuilding it from full raw outputs is optional:

```bash
python figure/_build_data.py
```

Run this only when the upstream full experiment outputs have changed; otherwise the bundled `figure/data/` and `figure/output/` are already sufficient for paper-figure reproduction.
