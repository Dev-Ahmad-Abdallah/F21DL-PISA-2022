# Next steps: from EDA to models

Report and code are due **Wed 25 Nov, 15:30**. R2 (cleaning, EDA, clustering, folds) is done and lives in `02_data-analysis/`. This page says what's ready, what to build next and what's still missing.

## What we're trying to find

How well do student, family and school factors predict maths, and flag students below Level 2 (420 points), across six systems (UAE, Saudi Arabia, Qatar, Jordan, Morocco, Palestine)? Which of the 5 factor groups matter most?

Two tasks, same features, same folds:
- **Regression:** predict the maths score (`PV1MATH`–`PV10MATH`)
- **Classification:** below Level 2 or not (`BELOW_L2_PV1`–`BELOW_L2_PV10`)

## What's ready

| File | What it is |
|---|---|
| `data/clean/pisa6_clean.parquet` | 61,775 students, 17 predictors + missing flags, 10 targets of each kind, weights |
| `data/clean/columns.json` | What each column is: features by group, targets, weights, flags |
| `data/clean/folds.csv` | 5 school folds. Everyone uses these, nobody makes their own |
| `02_data-analysis/README.md` | The EDA results in one page |

## What the data told us, and what it means for models

| We found | So |
|---|---|
| 70% of students are below Level 2 | Don't use accuracy. Use ROC-AUC, PR-AUC, balanced accuracy and recall for both classes |
| Blank answers go with 17–45 points lower | Keep the `_missing` flags as features |
| ESCS = exact mix of `PAREDINT`, `HISEI`, `HOMEPOS` | Logistic regression and naive Bayes use ESCS alone; trees and MLP can keep all four |
| Several predictors aren't straight lines | Expect trees and the MLP to beat linear models |
| School organisation is the strongest group, classroom the weakest | Ablations: drop school organisation, attitudes and family first |
| Up to 47% of variance between schools | Use `folds.csv` (whole schools held out) |
| Systems differ a lot | Report results per system as well as pooled |
| The UAE is 40% of the sample but 6% of the population | Train and score with `SENWT` weights |

## Suggested models

All of these are in scikit-learn and all accept `sample_weight`. Owners are from the project plan.

| Req. | Model | scikit-learn | Owner |
|---|---|---|---|
| R3 | Logistic regression (classification), Ridge (regression) | `LogisticRegression`, `Ridge` | Annie |
| R3 | Decision tree, with readable rules | `DecisionTreeClassifier` / `Regressor`, `export_text` | Fathah |
| R3 | Random forest | `RandomForestClassifier` / `Regressor` | Fathah |
| R3 | Naive Bayes, compared with the GMM from `03_clustering` | `GaussianNB` | Mohamed |
| R3 | Pooled results table, ablations by factor group | — | Ahmed |
| R4 | MLP, compared with the best baseline | `MLPClassifier` / `MLPRegressor` | Annie |
| R4 | CNN on AHCD | — | Anshita |

## Don't rebuild what scikit-learn already has

| Need | Use |
|---|---|
| Fill blanks inside the training fold | `SimpleImputer(strategy="median")` |
| Scaling that isn't thrown by extreme values | `RobustScaler()` |
| Chain the steps so they only learn from training rows | `make_pipeline(...)` |
| Weighted metrics | `roc_auc_score`, `average_precision_score`, `balanced_accuracy_score`, `recall_score`, `mean_squared_error`, `r2_score`, all with `sample_weight=` |
| Decision-tree rules | `export_text` |

The only loop we write ourselves runs each model on all 10 plausible values × 5 folds and combines the results. Copy this, swap the model, and you're done. It runs in about 10 seconds and gives ROC-AUC 0.799 ± 0.004 for logistic regression.

```python
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

df = pd.read_parquet("../data/clean/pisa6_clean.parquet")
roles = json.loads(open("../data/clean/columns.json").read())
folds = pd.read_csv("../data/clean/folds.csv")
df["fold"] = df["CNTSCHID"].map(folds.set_index("CNTSCHID")["fold"])

# logistic regression and naive Bayes: ESCS alone, not its three parts
drop = ["PAREDINT", "HISEI", "HOMEPOS"]
features = [c for g in roles["feature_groups"].values() for c in g if c not in drop]
features += [f"{c}_missing" for c in features]
X = df[features]
w = df["SENWT"] / df["SENWT"].mean()

rows = []
for pv in range(1, 11):
    y = df[f"BELOW_L2_PV{pv}"]
    for k in range(5):
        train, test = df["fold"] != k, df["fold"] == k
        model = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=1000))
        model.fit(X[train], y[train], logisticregression__sample_weight=w[train])
        p = model.predict_proba(X[test])[:, 1]
        rows.append({"pv": pv, "fold": k,
                     "roc_auc": roc_auc_score(y[test], p, sample_weight=w[test]),
                     "pr_auc": average_precision_score(y[test], p, sample_weight=w[test]),
                     "balanced_acc": balanced_accuracy_score(y[test], p > 0.5, sample_weight=w[test])})

res = pd.DataFrame(rows)
per_pv = res.groupby("pv").mean().drop(columns="fold")
within = res.groupby("pv").var().drop(columns="fold").mean() / 5   # fold-to-fold noise
between = per_pv.var()                                               # PV-to-PV noise
print(pd.DataFrame({"score": per_pv.mean(), "se": np.sqrt(within + 1.1 * between)}).round(3))
```

For trees and forests, drop the `RobustScaler` (they don't need it). For regression, use `PV{pv}MATH` as `y` and score with RMSE, MAE and R². Tune on `PV1` only, then run all 10.

## Rules

- Never use reading or science scores as features.
- Always use `folds.csv` and the `SENWT` weights.
- Run all 10 plausible values before reporting a number.
- Put model notebooks in `03_baselines/` and `04_neural-networks/`, and load data with `../data/...`.

## Still missing

| What | Owner | Status |
|---|---|---|
| Baselines: logistic/ridge, tree + rules, forest, naive Bayes | Annie, Fathah, Mohamed | Not started |
| MLP and MLP vs best baseline | Annie | Not started |
| Pooled results table and ablations | Ahmed | Not started |
| Per-system results for the best model | Ahmed | Not started |
| AHCD CNN, or a decision to drop it | Anshita | Status unknown, nothing in the repo |
| Side questions: distraction, resilience (first numbers in `02_eda` 2.11) | Mohamed, Fathah | Optional, cut first if late |
| Decide: add `IMMIG` as an 18th predictor? Use country as a feature? | Team | Open |
| Report sections and the contribution appendix | Everyone | Not started |
| AI-use note for any AI-assisted notebook (coursework rule) | Everyone | Check your own notebooks |
