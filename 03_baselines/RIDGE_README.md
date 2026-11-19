# Ridge regression for PISA 2022 mathematics

This README documents the implementation choices in `Ridge_Regression.ipynb` and provides a guide for reviewing the experiment. It is technical documentation, not a final assessment discussion or an independent evaluation of the findings.

## Run the notebook

Use a checkout based on Ahmed-EDA. Open the notebook with `03_baselines/` as the working directory, then run cells in order. The saved outputs can also be viewed without running it.

The notebook reads the existing files in `../data/clean/`:

- `pisa6_clean.parquet`
- `columns.json`
- `folds.csv`

The original cleaning and EDA do not need rerunning.

Dependencies are pandas, NumPy, PyArrow, Matplotlib, scikit-learn 1.6 or later, and IPython/Jupyter. The notebook was tested using scikit-learn 1.9.0 and pandas 3.0.3.

## Task and model

The task is continuous mathematics-score prediction across six education systems. Each of the ten mathematics plausible values is a separate target. Native multioutput Ridge fits those ten independent targets in one call with the same preprocessing and selected alpha. It does not average students' PV scores or force the targets to share regression coefficients.

Ridge combines squared prediction error with an L2 coefficient penalty. It provides a regularised linear baseline for comparing more flexible models later. All 17 agreed predictors remain in the primary model, together with their missing indicators. Ridge can accommodate correlated predictors, but regularisation does not make coefficients causal effects or independent importance measurements.

The intercept is fitted by the estimator. The direct linear-system solver does not require a learning rate or an epoch count. There are no custom estimator classes or custom metric functions.

## Why these metrics?

All three metrics use SENWT. Within each PV, they evaluate predictions for students whose schools were excluded from that fitted model. The headline results average the ten PV-specific metrics, rather than evaluating against an averaged-PV target.

| Metric | What it measures | Why include it | How to read it and its limits |
| --- | --- | --- | --- |
| RMSE | Square root of weighted mean squared prediction error | Matches the squared-error objective and gives larger errors more influence, while returning to mathematics-score units | Lower is better. Sensitive to large errors. Compare systems alongside their outcome spread |
| MAE | Weighted mean absolute prediction error | Provides an easily understood average error in mathematics points, with less emphasis on large errors than RMSE | Lower is better. It is not a confidence interval or a guarantee about an individual prediction |
| R² | One minus weighted squared error divided by weighted outcome variation | Shows performance relative to the weighted mean of the evaluated outcomes and gives context beyond raw score units | Higher is better. Zero corresponds to that evaluation-set mean reference, negative values are possible, and differing outcome variation complicates system comparisons |

The separately fitted DummyRegressor is a deployable training-mean baseline: it uses only training outcomes. It differs from the evaluation-set mean reference in the mathematical definition of R².

Classification accuracy, precision, recall and ROC-AUC are not used because this notebook predicts numerical scores, not classes. SENWT gives each system equal total contribution to pooled evaluation. A constant rescaling within a system does not change these metrics.

For inner tuning, GridSearchCV uses negative weighted MSE so its standard maximisation selects the smallest MSE. The displayed inner RMSE is the square root of the mean inner-fold MSE. This is not the arithmetic mean of the fold RMSEs.

## Reasoning at each step and what follows

| Notebook step | Purpose and decision | What to do next and why |
| --- | --- | --- |
| 1. Load shared files | Reuse the agreed clean table, column roles and school assignments. Avoid introducing a different extraction or cleaning version | Check identifiers, targets and weights before fitting, because successful file loading alone does not establish usable inputs |
| 2. Audit data and folds | Check unique student identifiers, fold coverage, whole-school separation, finite targets and positive weights | Resolve failed checks before modelling. If checks pass, define predictors so identifiers and outcomes cannot accidentally become features |
| 3. Specify features and pipeline | Use all 17 predictors as the main specification and three predefined family-based comparisons. Include each set's own missing flags. Median imputation retains incomplete records, and RobustScaler uses median/IQR scaling | Fit transformations inside training folds. This avoids estimating medians or scaling parameters from excluded schools |
| 4. Tune and train | Use the five saved outer school folds. Within each outer training partition, use the remaining saved folds for PV1-only alpha tuning. Fix that alpha for all ten PV targets | Evaluate the untouched outer schools. Keep their outcomes out of alpha selection so evaluation is separate from tuning |
| 5. Calculate weighted performance | Calculate RMSE, MAE and R² separately for every PV and system. Headline pooled metrics use all out-of-fold records, rather than an unweighted average of fold scores | Examine uncertainty before treating small metric differences as convincing. Also compare with the weighted training-mean baseline |
| 6. Estimate conditional uncertainty | Use the 80 Fay BRR replicate weights and combine within-PV variance with between-PV variance. Keep fitted predictions fixed during this calculation | Interpret these standard errors as conditional evaluation uncertainty. Do not describe them as repeated-training uncertainty or use them alone to claim a significant difference between models |
| 7. Compare predictor sets | Compare family alone, family plus attitudes, family plus school organisation, and all 17. Use identical outer records for a fair descriptive comparison | Check whether improvements hold across PVs and systems. Added groups may overlap, so an improvement is not a causal or unique contribution |
| 8. Check baseline and group improvements | Quantify error reductions and count PVs with improvements. PV consistency is a descriptive robustness check, not ten independent samples | Retain per-system context. If little improvement is seen, investigate the model assumptions or preprocessing in a separately documented training-only experiment |
| 9. Put system errors in context | Compare absolute error with weighted outcome SD, R² and the training-mean baseline. Narrower score distributions can produce smaller absolute errors | Identify systems needing closer investigation without interpreting raw error rankings as evidence of fairness or causal differences |
| 10. Inspect alpha boundaries | Count selections at the smallest and largest candidate alpha. A single boundary selection does not establish a failure | Repeated boundary selections can motivate a wider search in a separate training-only experiment. Do not change the grid because of a preferred outer-test result |
| 11. Check uncertainty and handover | Keep conditional SE separate from descriptive fold SD. Verify that weights and school assignments are preserved and every student receives one excluded-school prediction per PV | Share the notebook with the unchanged agreed inputs. A separate resampling-and-refitting study would be needed to estimate retraining variability |
| 12. Inspect alpha tables and plots | Display GridSearchCV's existing PV1 training and validation scores across alpha settings. No additional model selection uses the outer results | Review the error gap and sensitivity to regularisation. If planning further experiments, specify them and tune inside training partitions rather than treating the existing outer results as a new validation set |

## Why these predictor-set comparisons?

Family alone establishes a background benchmark. Adding attitudes asks whether self-reported mathematics attitudes offer additional predictive information. Adding school organisation instead asks whether those school characteristics offer additional information beyond family. The all-17 specification includes all five agreed groups and remains the primary model.

These additions are conditional on the family starting set. They are not a complete leave-one-group-out study and do not replace the team's planned shared ablations. Missing indicators accompany only the predictors included in each configuration.

## Further experiments

A sensible next model comparison uses the same data, folds, weights, targets and metrics. A nonlinear model can test whether interactions and curved relationships improve prediction over Ridge. Improvement is not guaranteed.

Any new feature representation, interaction, hyperparameter search or system-indicator experiment should be documented and selected inside training folds. The existing school-fold evaluation concerns new schools within represented systems. It does not establish transportability to an unseen system.

Training and validation error curves, replicate SEs and PV variation answer different questions. The conditional standard errors keep predictions fixed. Fold SD describes the five school partitions. Neither estimates the full variability from collecting new data, retuning and retraining the model.

## 
