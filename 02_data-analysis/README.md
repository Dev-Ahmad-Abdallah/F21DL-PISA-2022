# 02 · Data analysis and exploration (R2)

What goes here: cleaning and preprocessing, EDA with visualisations, feature selection, and clustering (optional, with suitable metrics).

Read the data from `data/parquet/`; don't copy it into this folder. The main README explains how to load it.

Marked in D2 and D3.

## Part A: the clean table

`notebooks/01_clean_table.ipynb` builds one clean table for the six systems (UAE, Saudi Arabia, Qatar, Jordan, Morocco, Palestine). Start every later notebook from it. The notebook is saved with its outputs, and each step explains what it does and why.

| File | What it is |
|---|---|
| `data/clean/pisa6_clean.parquet` | 61,775 students x 151 columns: IDs, 10 maths plausible values, 10 below-Level-2 targets, 17 features, missing indicators, weights, side-question variables |
| `data/clean/columns.json` | Which column is an ID, target, feature (by factor group), missing indicator, weight, side-question, EDA-only or audit-only column |
| `outputs/part_a/cleaning_log.csv` | Rows and columns after each step |
| `outputs/part_a/missing_by_system.csv` | % missing per predictor and system |
| `outputs/part_a/implausible_schools.csv` | The 35 schools whose student-teacher ratio or size was set to NaN, and why |

Part A only applies fixed rules. Imputation, clipping, scaling, encoding and the choice of weight are **Part B** and must be fitted inside each training fold.

Load it with:

```python
import json, pandas as pd
df = pd.read_parquet(ROOT / "data/clean/pisa6_clean.parquet")
column_roles = json.loads((ROOT / "data/clean/columns.json").read_text())
feature_cols = [col for group_cols in column_roles["feature_groups"].values() for col in group_cols]
```

To change a cleaning rule (Level 2 cut-off, plausible student-teacher range), edit the constants at the top of the notebook and re-run it. Don't edit the Parquet file by hand.

## EDA

`notebooks/02_eda.ipynb` explores the clean table in ten sections and ends with a summary of what each finding means for modelling. It is weighted throughout (`W_FSTUWT` within a system, `SENWT` when pooling) and uses all 10 plausible values.

1. Sample vs population (the UAE is 40% of the sample, 6% of the population)
2. Maths score distributions and the Level 2 line
3. Missing values by predictor and system
4. Whether being missing goes with lower scores
5. Shape of each predictor
6. Correlations between predictors
7. Correlation of each predictor with maths, within each system
8. % below Level 2 by socio-economic quarter
9. Share of score variation between schools
10. First look at the side questions (distraction, resilience)

Figures for the report are in `outputs/eda/figures/`, with the numbers behind each figure in `outputs/eda/*.csv`.

## Part B: folds and the preprocessing pipeline

`pisa_prep.py` is the shared code every model uses (plan tasks T07 and T08): the school-grouped folds, the model inputs, the preprocessing pipeline, the targets and the weights. `notebooks/03_folds_and_pipeline.ipynb` builds the fold file and checks that every step only learns from the training schools.

| File | What it is |
|---|---|
| `pisa_prep.py` | The shared code. Import it, don't copy it |
| `data/clean/folds.csv` | Which of the 5 folds each of the 1,973 schools is in. Frozen: everyone uses this file |
| `outputs/part_b/fold_*.csv` | How alike the folds are (schools, students, % below Level 2, per system) |
| `outputs/part_b/learned_vs_whole_table.csv` | Medians and clipping limits learned from the training schools, next to the whole-table values |
| `outputs/part_b/smoke_test.csv` | A first end-to-end run (logistic and ridge regression, first plausible value only). Not results |

Use it from any notebook in the repo:

```python
import sys
sys.path.insert(0, str(ROOT / "02_data-analysis"))
import pisa_prep as prep

df = prep.load_clean_table()
fold_by_student = prep.get_fold_by_student(df)     # the frozen folds
feature_cols = prep.get_feature_cols()             # 17 features + 17 missing flags
for fold in range(prep.N_FOLDS):
    is_test = fold_by_student == fold
    train_df, test_df = df[~is_test], df[is_test]
    model = prep.make_model_pipeline(LogisticRegression(max_iter=2000), feature_cols)
    model.fit(train_df[feature_cols], train_df["BELOW_L2_PV1"],
              model__sample_weight=prep.get_sample_weights(train_df))
```

Section 9 of the notebook lists every function and three rules: fit only on training rows, tune inside the training folds, and don't make your own folds.

Four team decisions are still open. Each has a default, set in one line at the top of `pisa_prep.py`: the pooled weight (`SENWT`), the system as a predictor (no), the family group as ESCS alone (no), and clipping at the 1st and 99th percentiles (yes). Section 10 of the notebook explains each one.

## Variable names

The notebooks and `pisa_prep.py` name variables so you can tell what kind of thing each one holds. Please follow the same rules in new notebooks:

| The name... | means it holds | Examples |
|---|---|---|
| is a singular word | one value | `system` = `"ARE"`, `feature` = `"ESCS"` |
| is a plural word | a list of values | `SYSTEMS` = `["ARE", "SAU", ...]` |
| ends in `_col` / `_cols` | one column name / a list of column names | `pv_col` = `"PV1MATH"`, `PV_COLS`, `FEATURE_COLS` |
| ends in `_df` (or is just `df`, the main table) | a table (pandas DataFrame) | `students_df`, `system_df` |
| looks like `<value>_by_<key>` | a lookup (dict or pandas Series) | `FEATURES_BY_GROUP["family"]` |
| starts with `is_` / `has_` | True/False per student, used as a filter | `df[is_complete]` |
| starts with `n_` / `pct_` / `share_` | a count / a percentage (0-100) / a fraction (0-1) | `n_flagged_students`, `pct_below` |
| ends in `_idx` | a position (0, 1, 2, ...) | `row_idx` |
| is in `UPPER_CASE` | a fixed setting, set once and never changed | `LEVEL2_CUTOFF`, `CLEAN_DIR` |

Loops use the singular of the list (`for system in SYSTEMS`), and helper functions have type hints (`cols: list[str]`, `-> float`). PISA's own column names (`ESCS`, `W_FSTUWT`) are never renamed. The full table is at the top of each notebook.
