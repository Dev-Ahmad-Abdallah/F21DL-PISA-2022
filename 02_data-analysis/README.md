# 02 · Data analysis and exploration (R2)

Cleaning, EDA, clustering and the school folds for the six systems (UAE, Saudi Arabia, Qatar, Jordan, Morocco, Palestine). This folder brings together the R2 work from three team branches in one run order, plus the steps none of them covered.

## Run order

Run the notebooks top to bottom, in this order, from this folder. They read the data with relative paths (`../data/...`). Each one is saved with its outputs, so you can read them without running anything.

| Notebook | What it does | Writes | Run time |
|---|---|---|---|
| `01_cleaning.ipynb` | Loads the six systems, joins schools, fixes impossible values, builds the 10 plausible-value targets, adds missing and quality flags | `../data/clean/pisa6_clean.parquet`, `../data/clean/columns.json` | ~1 min |
| `02_eda.ipynb` | Weights, target, missing data, response quality, distributions, correlations, factor groups, schools vs systems, subgroups, side questions | `outputs/figures/01–10`, `outputs/target_by_system.csv` | ~2 min |
| `03_clustering.ipynb` | PCA, k-means and a Gaussian mixture (EM) with silhouette, Davies-Bouldin and BIC | `outputs/figures/11–13` | ~2 min |
| `04_folds.ipynb` | Splits the 1,973 schools into 5 folds every model uses | `../data/clean/folds.csv` | <1 min |

Needs `scikit-learn` (now in `requirements.txt`).

## Using the clean table

```python
import json
import pandas as pd

df = pd.read_parquet("../data/clean/pisa6_clean.parquet")
roles = json.loads(open("../data/clean/columns.json").read())
features = [c for cols in roles["feature_groups"].values() for c in cols]   # the 17 predictors
folds = pd.read_csv("../data/clean/folds.csv")
df["fold"] = df["CNTSCHID"].map(folds.set_index("CNTSCHID")["fold"])
```

`columns.json` says what every column is for: IDs, targets (`PV1MATH`–`PV10MATH` and `BELOW_L2_PV1`–`BELOW_L2_PV10`), the 5 feature groups, missing flags, weights, side-question columns, quality flags and audit columns.

## What the notebooks found

**Data**
- 61,775 students in 1,973 schools, none dropped. The UAE is 40% of the sample but 6% of the population, so everything is weighted (`W_FSTUWT` within a system, `SENWT` when pooled).
- 70% of students are below Level 2 (76% with population weights). The class we flag is the majority, so accuracy alone is misleading.
- Maths means with standard errors: UAE 431 (±0.9), Qatar 414 (±1.1), Saudi Arabia 389 (±1.8), Palestine 366 (±1.8), Morocco 365 (±3.4), Jordan 361 (±2.0). The bottom three overlap.
- A third of students change below-Level-2 label across the 10 plausible values.

**Missing data and quality**
- Only 56% of students answered all 17 predictors (36% in Morocco). Students who leave a question blank score 17–45 points lower, so we keep everyone, add missing flags and impute inside the folds.
- Those who skip the attitude block are mostly boys (33% girls vs 54%) and poorer. 22% of them never reached those screens.
- `ICTDISTR` is missing in the UAE, Qatar and Palestine because they didn't run the ICT questionnaire.
- Students who speed through the questionnaire (under 1 second per item) score 60–78 points lower in Qatar and the UAE.

**Predictors**
- ESCS is an exact weighted combination of `PAREDINT`, `HISEI` and `HOMEPOS` (R² = 1.000). Use ESCS alone in logistic regression and naive Bayes.
- Strongest single links with maths (r ≈ 0.3): self-efficacy, private school, parents' occupation, ESCS. Some flip sign between systems (student-teacher ratio: +0.26 Qatar, −0.36 Morocco).
- Several predictors aren't linear (parents' education, school size, bullying), so trees and the MLP should beat linear models.
- Factor groups (weighted R², pooled): all 17 explain 0.365. School organisation explains most alone (0.22) and loses most when dropped (0.07). Classroom adds almost nothing (0.002).
- 16–47% of maths variance lies between schools, so folds hold out whole schools.
- Immigrant students score 90 points higher in the UAE and 71 in Qatar. `IMMIG` isn't one of the 17. The team needs to decide whether to add it.

**Clustering**
- No strong clusters (best silhouette 0.17 at k = 2; BIC keeps falling). The split is advantaged (mostly UAE and Qatar) vs disadvantaged (mostly Jordan, Morocco, Palestine), 57% vs 86% below Level 2. It's description, not a model feature.

**Folds**
- 5 folds of about 395 schools and 20% of students each, with 69–71% below Level 2 in every fold. This is the same split as Mohamed's original `folds.csv`: every school lands in the same fold.

## Where the work comes from

| Part | Based on |
|---|---|
| Clean table, targets, leakage check, missing flags, most of the EDA, folds | Mohamed (`preprocessing-and-EDA`) |
| Dictionary check for missing codes, join checks, source checks on school size and ratio, missing patterns, pooled vs within-system correlations | Annie (`annie-eda`) |
| Candidate predictors left out (`ICTDISTR`, `FAMSUP`, `FEELSAFE`), weighted averages per system, points per unit | Fathah (`fathah-eda-cleaning`) |
| Standard errors, who skips and not-reached, response quality, outliers, VIF, non-linearity, factor-group R², between-system check, subgroups, clustering | New in this consolidation (Ahmed) |

Each notebook section names its source at the top.

## Not here yet

- The in-fold preprocessing pipeline (fill, clip, scale learned on training folds only) is Mohamed's `pisa_prep.py`. It comes in with the models in `03_baselines/`.
- Open team decisions: add `IMMIG` as an 18th predictor or not, and whether to use the system (`CNT`) as a feature.
