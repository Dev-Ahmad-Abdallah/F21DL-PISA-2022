# Use of AI in the pre-project stage

This folder was produced with an AI coding assistant during idea selection. It is research and data preparation to help the group pick and understand a dataset. It is not part of the assessed project work, and it is kept separate from the stage folders (`01_topic/` onwards).

## Tool

Claude Code (Anthropic), running the Claude Opus model, used on 27-29 September 2026.

## What was asked

1. Download all PISA 2022 data files from the OECD site, extract them in a local Jupyter notebook, and show what data points exist and what is going on in the data.
2. Re-run the notebooks with saved outputs, and prepare the data and notebooks as a repository the group can clone and run, with a README.

## What the AI produced

- `notebooks/01_extract.ipynb`: download, conversion to Parquet, data dictionary.
- `notebooks/02_explore.ipynb` and `notebooks/_build_02.py`: the descriptive tour, including the OECD weighting and standard-error method.
- `outputs/`: tables, charts and `key_findings.md`, all computed by notebook 02.
- The README files and this repo's folder layout.

## What was not done with AI

No research questions, hypotheses, objectives, feature choices or models. Those belong to the group and start in `01_topic/`.

## How the output was checked

- Section 0 of `02_explore.ipynb` compares the notebook's estimates with the OECD's published PISA 2022 results (OECD averages 472 / 476 / 485, Singapore 575 in math). The notebook stops if they do not match.
- After the data was split into parts for GitHub, every output file was compared byte for byte with the run before the split, and all were identical.
- Treat `key_findings.md` as a starting point for discussion, not as results to cite. They are descriptive associations, not causes.
