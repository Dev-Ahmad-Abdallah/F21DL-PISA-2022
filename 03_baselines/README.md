# 03 · Baseline training and evaluation (R3)

What goes here: at least 3 classic ML algorithms (e.g. decision trees, naive Bayes, linear models, perceptron, k-NN, ensembles), with justified metrics and a discussion of decision trees' practical use for our topic.

Marked in D2 and D3.

## Notebooks

| Notebook | Who | What | Run time |
|---|---|---|---|
| `ahmed_gradient_boosting.ipynb` | Ahmed | Gradient boosting for both tasks, tuned on PV1, all 10 PVs, results per system | ~4 min |
| `ahmed_ablations.ipynb` | Ahmed | Gradient boosting with each factor group dropped. Run it after the one above, since it reads `outputs/ahmed_hgb_params.json` | ~18 min |

Results are saved in `outputs/` as `model, task, metric, score, se`. Use the same shape for your model so we can stack everything into one table.
