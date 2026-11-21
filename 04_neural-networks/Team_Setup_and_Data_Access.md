# Keras CNN baseline: setup and shared data

## Included work

- [AHCD_EDA.ipynb](AHCD_EDA.ipynb): executed data audit with observations and preprocessing decisions.
- [Keras_CNN_Baseline.ipynb](keras/Keras_CNN_Baseline.ipynb): executed two-block CNN with training curves, validation metrics, retained-model training/validation comparison, per-class recall, annotated confusion matrices and fixed diagnostic examples.
- `keras/results/`: trained model, settings, predictions, metrics and figures.
- `data/ahcd/`: original archive, preparation script and prepared shared data at the repository root.

This package does not depend on another model notebook or framework. Reading the saved outputs does not require retraining.

## Install and open

Use Python 3.12 in a separate environment. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .cnn-tf-venv
source .cnn-tf-venv/bin/activate
python -m pip install -r 04_neural-networks/keras/requirements-cnn.txt
python -m ipykernel install --user --name ahcd-tensorflow --display-name "Python (AHCD TensorFlow)"
python -m jupyterlab
```

On Windows Command Prompt, create the environment with `py -3.12 -m venv .cnn-tf-venv` and activate it using `.cnn-tf-venv\Scripts\activate`, then run the remaining installation commands.

Open `keras/Keras_CNN_Baseline.ipynb`, choose the TensorFlow kernel and run from `04_neural-networks/keras/`. Open the EDA from `04_neural-networks/`. Relative paths depend on the kernel working folder, which can be checked with `print(Path.cwd())`.

## Load the shared dataset

The model's Section 1 contains its complete loader. It reads `../../data/ahcd/shared/training_data.npz` and `split_indices.npz`. No Kaggle account, separate download or preparation rerun is needed when both repository folders are present.

Stored images have shape `(13440, 1, 32, 32)`. The Keras loader moves the channel axis to the end, converts to float and divides by 255 once. Images are already upright. Do not rotate them or scale twice.

The loader excludes the single blank fitting image, original row 6050. It then produces 10,751 training images of shape `(10751, 32, 32, 1)` and 2,688 validation images of shape `(2688, 32, 32, 1)`. Targets are integer labels 0 through 27. Tables and plots show original labels 1 through 28.

The shared arrays retain all original rows and the stored split retains 10,752 fitting indices. Filtering occurs only in the notebook. Validation membership is unchanged. The EDA deliberately includes the blank image to demonstrate the finding.

## Saved model and new experiments

From the model notebook folder, load `results/CNN.keras` with `tf.keras.models.load_model`. The notebook verifies that the reloaded model reproduces its predictions. `validation_predictions.npz` stores original row IDs, true labels, predicted labels and class probabilities. JSON files record settings and numerical results.

Running all cells retrains and overwrites this notebook's own results. For another experiment, copy the notebook and choose a different `output_folder` before training. Preserve the shared split and blank-image rule so comparisons are consistent.

## Results and limits

Validation accuracy is 87.09%, macro F1 is 0.8710 and loss is 0.4381. Training accuracy is 94.22%, with a gap of 7.13 percentage points. Final subset comparisons use the retained best weights. Learning-curve training scores are collected during updates and can differ.

The shared inner split is image-level, not verified as writer-independent. No exact image duplicates were found, but that does not rule out similar images or writer overlap. Fixed examples are diagnostic rather than representative. The model does not use the separate original test portion. Earlier project exploration means it should not be called never previously inspected. Coordinate final testing after model selection.

See [Dataset and preparation details](../data/ahcd/README.md)
