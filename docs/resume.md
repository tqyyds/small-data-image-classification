# Resume and interview notes

## Resume entry

**Small-data Image Classification — Python, scikit-learn, PyTorch, torchvision**

[GitHub portfolio](https://github.com/tqyyds/small-data-image-classification)

- Independently built image-classification experiments across two 10-class datasets, comparing engineered descriptors, frozen CNN embeddings, SVMs, tree ensembles and ResNet50 transfer learning.
- Recorded historical OOF macro-F1 of **0.9492** on 3,750 animal images and **0.9378** on 417 bird images using stratified 3-fold and 5-fold validation respectively; documented model/checkpoint-selection limitations.
- Created reproducible CPU checks and an auditable evaluation trail that recomputes metrics for **13 saved prediction sets**, with class-level reports and confusion matrices.

## Short version

Independent image-classification project comparing classical ML and pretrained CNN representations on two 10-class tasks; achieved historical validation macro-F1 of 0.9492/0.9378 and published reproducible evaluation checks with explicit selection-bias limitations.

## Interview discussion points

**Why compare frozen features with fine-tuning?** Frozen ImageNet representations test whether a simpler classifier can perform well with scarce labels. Fine-tuning allows adaptation but adds training cost and checkpoint selection. The bird results illustrate why a small aggregate gain is not enough to dismiss the simpler approach.

**Why macro-F1?** It averages class-level F1 equally. Accuracy can hide uneven performance; the bird warbler and gull results make class-level reporting useful even when class counts are roughly balanced.

**How did you prevent implementation leakage?** Metadata checks reject overlapping train/test IDs, feature joins use identifiers, and imputation/scaling/searched PCA live inside sklearn pipelines. Distinguish those safeguards from the remaining selection bias: classical hyperparameters reuse CV folds, CNN checkpoints reuse validation folds, and supplied precomputed-feature provenance is not fully auditable.

**What would you change next?** Add nested CV for classical hyperparameters and an inner checkpoint-selection split for CNNs; reserve untouched outer folds or an independent evaluation set. Compare model cost and class-level errors as well as aggregate F1. These are proposed improvements, not completed work.

**What can a reviewer run?** Installation, twelve focused tests, verification of archived predictions/reports, a synthetic end-to-end CPU experiment and prediction CSV generation. Full course-data experiments and GPU retraining require separately obtained inputs and were not rerun for the portfolio edition.

Use “historical OOF macro-F1” or “validation macro-F1” when quoting these figures. Do not relabel them accuracy, independent test performance or leaderboard rank. No inference API, production deployment or GPU rerun is claimed.
