# F21DL · PISA 2022

Group project repository for F21DL Data Mining and Machine Learning (Heriot-Watt University Dubai, 2026/27). Topic: **education**, using the OECD **PISA 2022** database.

The data is already in the repo, extracted and cleaned into a fast format. Clone it and you can start working straight away.

## Repository structure

```
data/                    shared by every stage (see "The data" below)
  parquet/               all ten PISA 2022 files, one folder each
  data_dictionary.csv    all 12,759 variables: label, type, answer codes, % missing
  inventory.csv          rows, columns and countries per file
00_pre-project/          idea selection: getting the data and a first look at it (AI-assisted, see its AI_USE.md)
01_topic/                R1  research questions, hypotheses, objectives, pitch (D1)
02_data-analysis/        R2  cleaning, preprocessing, EDA, feature selection, clustering
03_baselines/            R3  classic ML models and evaluation
04_neural-networks/      R4  MLP / neural networks, tuning, comparison with baselines
report/                  D2  6-page report, figures, references
requirements.txt
```

Each folder has its own README saying what belongs there.

## Quick start

Needs Python 3.11 or newer and about 2 GB of disk space.

```bash
git clone https://github.com/Dev-Ahmad-Abdallah/F21DL-PISA-2022.git
cd F21DL-PISA-2022
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Start with `00_pre-project/notebooks/02_explore.ipynb`, a tour of every file. It is saved with its outputs, so you can read it without running anything.

Load data in your own notebook. This finds the repo root from any folder:

```python
from pathlib import Path
import pandas as pd

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'requirements.txt').exists())
stu = pd.read_parquet(ROOT/'data/parquet/CY08MSP_STU_QQQ',
                      columns=['CNT', 'CNTSCHID', 'W_FSTUWT', 'ESCS', 'PV1MATH'])
DD = pd.read_csv(ROOT/'data/data_dictionary.csv')      # look up what any variable means
```

Pass `columns=` whenever you can. The student files have 1,278 to 5,023 columns, and loading all of them takes several GB of memory.

## The data

| Folder in `data/parquet/` | Rows | Columns | Countries | What it is |
|---|---:|---:|---:|---|
| `CY08MSP_STU_QQQ` | 613,744 | 1,278 | 80 | Student questionnaire: scores (plausible values), weights, ~130 background and attitude indices |
| `CY08MSP_STU_COG` | 613,744 | 5,023 | 80 | Answers, scores, time taken and clicks for every test question |
| `CY08MSP_STU_TIM` | 613,744 | 172 | 80 | Time spent on each questionnaire screen |
| `CY08MSP_SCH_QQQ` | 21,629 | 431 | 80 | School questionnaire: type, resources, class size, policies |
| `CY08MSP_TCH_QQQ` | 68,054 | 543 | 18 | Teacher questionnaire (18 systems only) |
| `CY08MSP_CRT_COG` | 499,843 | 301 | 63 | Creative Thinking, new in 2022 (scale 0-60) |
| `CY08MSP_FLT_COG` / `_QQQ` / `_TIM` | 97,983 | 3,637 / 1,192 / 172 | 20 | Financial Literacy: a separate sample with its own scores and weights |
| `escs_trend` | 1,447,787 | 10 | 71 | Socio-economic index (ESCS) for 2012-2018, rescaled to match 2022 |

Join keys: `CNT` (3-letter country code), `CNTSCHID` (school) and `CNTSTUID` (student).

**How it differs from the OECD download.** The OECD ships uncompressed SPSS files (~9 GB). Here they are stored as compressed Parquet (1.8 GB), with every row, column and value kept. Two things changed:

1. **Split into parts.** GitHub rejects files over 100 MB, so each file is a folder of parts under 90 MB. `pd.read_parquet(folder)` reads the folder as one table, with the same rows in the same order.
2. **Missing codes became `NaN`.** The SPSS codes 95 (valid skip), 97 (not applicable), 98 (invalid) and 99 (no response) are now `NaN`, so the data no longer says *why* an answer is missing. What each code meant is in the dictionary's `value_labels` column. If you need the distinction, rebuild from the source (see `00_pre-project/README.md`).

## Things you need to know before using PISA

These are how the OECD says the data must be analysed. Getting them wrong gives wrong numbers.

- **Weights.** PISA samples some countries far more heavily than their size. Always weight by `W_FSTUWT`. Unweighted counts and averages can be badly off, especially when you combine several countries.
- **Scores are 10 plausible values.** There is no single "math score". `PV1MATH` to `PV10MATH` are ten draws per student. Run an analysis once per plausible value and combine the results (Rubin's rules). Using only `PV1MATH` gives the right average but understates uncertainty.
- **Standard errors** come from the 80 replicate weights `W_FSTURWT1` to `W_FSTURWT80` (Fay's BRR method, k = 0.5). The "Estimation engine" section of `02_explore.ipynb` has a ready-made function for this.
- **Categorical answers are numbers.** Look up what `1`, `2`, `3` mean in the `value_labels` column of `data/data_dictionary.csv`.
- **Leakage.** Reading and science scores are almost the same thing as math scores. Don't use them as features to predict math.
- **Schools cluster.** Students in the same school are similar, so split train and test sets by school (`CNTSCHID`), not by student.

The OECD's guide: [How to prepare and analyse the PISA database](https://www.oecd.org/en/about/programmes/pisa/how-to-prepare-and-analyse-the-pisa-database.html).

## Source and licence

OECD (2023), *PISA 2022 Database*, <https://www.oecd.org/en/data/datasets/pisa-2022-database.html>. The data is published by the OECD and is free to reuse with attribution under the [OECD Terms and Conditions](https://www.oecd.org/en/about/terms-conditions.html). Cite the OECD when you use it. The student data is anonymised: no names, only generated IDs.
