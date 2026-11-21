# Keras CNN baseline

The baseline has two convolutional blocks with 64 filters, ReLU and 2 by 2 max pooling, followed by Flatten, a Dense layer with 128 ReLU units and 28 softmax outputs. It has 336,220 trainable parameters and starts from random weights.

Training uses Keras compile and fit, Adam with learning rate 0.001, batch size 64, up to 30 epochs and built-in early stopping on validation loss with patience 5 and best-weight restoration. There is no dropout, augmentation or pretrained initialization.

The EDA found one all-zero fitting image. The notebook excludes it using a general rule, retaining all validation rows and leaving shared files unchanged. This is a data-quality choice rather than a claim of improved predictive performance.

Accuracy measures overall correctness. Macro F1 weights classes equally. Per-class recall, annotated confusion matrices and fixed image examples show where recognition is difficult. Cross-entropy is the training objective and epoch-selection metric. Training and validation are evaluated using the retained best weights.

Validation accuracy is 87.09%, macro F1 is 0.8710 and loss is 0.4381. The retained epoch is 5, after 10 attempted epochs. These are single-run validation results, not final test results. Training accuracy is 94.22%, giving a 7.13 percentage-point gap.

See [Team_Setup_and_Data_Access.md](Team_Setup_and_Data_Access.md) for environment setup and dataset access.
