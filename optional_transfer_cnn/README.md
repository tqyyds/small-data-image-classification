# Transfer-learning implementation

This directory preserves the project's supervised CNN extension. `cross_validate_transfer_cnn.py` runs stratified CV; `train_transfer_cnn.py` provides the shared training functions and a holdout runner. `train_cnn.py` retains a from-scratch baseline, `ensemble_transfer_cnn.py` handles checkpoint-logit averaging, and `verify_cuda.py` checks the local device.

Install optional dependencies with `python -m pip install -r requirements-deep.txt` from the repository root. Use the commands in the [reproduction guide](../docs/reproduction.md). The directory's historical name does not imply that CNN results are omitted from the portfolio.

Historical OOF macro-F1 is 0.9492 for animals and 0.9378 for birds, with horizontal-flip TTA. These values were checked from archived predictions, not from a new GPU training run. Validation folds also select checkpoints; see the [evaluation boundaries](../docs/methodology.md). Weights and raw course data are not distributed.
