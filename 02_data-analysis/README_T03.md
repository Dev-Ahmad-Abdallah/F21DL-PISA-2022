# 02 · Data analysis and exploration (R2)

What goes here: cleaning and preprocessing, EDA with visualisations, feature selection, and clustering (optional, with suitable metrics).

Read the data from `data/parquet/`; don't copy it into this folder. The main README explains how to load it.

Marked in D2 and D3.

## T03: load and recode data

Run from the repository root with the project environment active:

```bash
python 02_data-analysis/scripts/t03_prepare_data.py
```

Alternatively, open `notebooks/t03_prepare_data.ipynb` and run all cells. It runs
the same script and displays the audit tables, so both entry points stay consistent.
Only pandas, NumPy and PyArrow are needed for the script.

### What the script does

1. Loads only the required columns for ARE, SAU, QAT, JOR, MAR and PSE.
2. Checks missing identifiers and duplicate students/schools before joining.
3. Left-joins school data on `CNT` and `CNTSCHID`, using `validate="many_to_one"`.
4. Uses each predictor's `value_labels` in `data/data_dictionary.csv` to replace
   codes labelled Valid Skip, Not Applicable, Invalid or No Response with `NaN`.
   Existing NaNs remain missing. Codes are variable-specific: for example,
   SCHLTYPE uses 5/7/8/9, whereas STRATIO uses 995/997/998/999.
5. Adds 17 `<predictor>_missing` columns using `isna().astype("uint8")`:
   1 means missing and 0 means observed.
6. Saves prepared data and audit files to `outputs/t03/`.

The 11 student predictors are ESCS, PAREDINT, HISEI, HOMEPOS, ANXMAT, MATHEFF,
MATHEF21, BELONG, BULLIED, DISCLIM and TEACHSUP. The six school predictors are
EDUSHORT, STAFFSHORT, MCLSIZE, STRATIO, SCHLTYPE and SCHSIZE.

All ten mathematics plausible values, W_FSTUWT and 80 replicate weights are
retained. Identifiers and country codes are retained for subsequent splitting
and auditing. They are not automatically designated model predictors.

### Verified result

The current source data produce **61,775 students, 1,973 schools and 128 columns**.
No students are duplicated or unmatched after the join. All selected documented
missing codes were already NaN, so this run recoded zero additional values.

| System | Students | Schools |
| --- | ---: | ---: |
| ARE | 24,600 | 840 |
| SAU | 6,928 | 193 |
| QAT | 7,676 | 229 |
| JOR | 7,799 | 260 |
| MAR | 6,867 | 178 |
| PSE | 7,905 | 273 |

These are unweighted sample counts. Missingness percentages are also unweighted
data-quality summaries, not population estimates.

### Output files

| File in `outputs/t03/` | Contents |
| --- | --- |
| `pisa_six_systems_prepared.parquet` | One row per student with predictors, indicators, IDs, mathematics PVs and weights |
| `audit.json` | Join checks, row counts, retained variables and weight checks |
| `system_summary.csv` | Student/school counts and invalid final-weight counts by system |
| `missingness.csv` | Dictionary missing codes and missing counts for each predictor |
| `missingness_by_system.csv` | Percentage missing for each predictor within each system |
| `unmatched_schools.csv` | School keys without a match; header-only for the current data |

Load the result in later tasks:

```python
prepared = pd.read_parquet(ROOT / "02_data-analysis/outputs/t03/pisa_six_systems_prepared.parquet")
```

Next, use the retained IDs for school-grouped splits and the missingness tables
for EDA. Fit any imputation, scaling or encoding inside training folds only.
T03 does not construct targets, impute values, scale data, encode categories or
train models. The original student and school Parquet files are unchanged.

