# 03 · Baseline training and evaluation (R3)

What goes here: at least 3 classic ML algorithms (e.g. decision trees, naive Bayes, linear models, perceptron, k-NN, ensembles), with justified metrics and a discussion of decision trees' practical use for our topic.

Marked in D2 and D3.

## Mohamed's Naive Bayes comparison

Open `mohamed_naive_bayes.ipynb` and run all cells using the repository's Python environment. It works when Jupyter starts from the repository root or `03_baselines/`. The notebook is saved with verified outputs and needs only the existing packages in `requirements.txt`.

It evaluates weighted GaussianNB on all 10 plausible values and the shared five school folds, compares a refitted two-component GMM with training-only risk mapping, and includes matched-input and prior-only checks. Results, predictions, a figure and a reproducibility manifest are saved under `outputs/mohamed_naive_bayes/`.

The GMM EM fit is unweighted, matching `03_clustering`; its risk mapping and scores are weighted. The Naive Bayes standard errors use BRR and plausible-value pooling conditional on fixed out-of-fold predictions. See the notebook for these limits, the probability-quality finding, and AI-use attribution.
