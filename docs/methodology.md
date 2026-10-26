# Methods and evaluation boundaries

## Study design

This independent course project compares representations and learning algorithms on animal and bird image classification. Animal training data contain 3,750 observations with 375 per class; bird training data contain 417 observations with 39–42 per class. Counts are corroborated by the saved predictions and classification reports, rather than a new inspection of the undistributed images.

Macro-F1 is the primary selection metric because it weights classes equally. Accuracy and balanced accuracy are retained in the historical tables. The default seed is 27. Classical stratified folds use shuffling with that seed; the CNN runner uses fold-specific training seeds derived from it. Historical fold summaries record the actual seeds and selected epochs.

## Representations and algorithms

Provided feature tables combine additional image descriptors, colour histograms and precomputed HOG/PCA features. The implementation adds RGB moments, HSV histograms, edge statistics, texture-difference summaries and aspect ratio. Feature tables join on `image_id`, never row position.

Frozen EfficientNet-B0, ResNet50 and MobileNetV3-Large use torchvision's supervised ImageNet weights. Combined CNN features concatenate the three backbones; `cnn_plus_provided` also includes the provided tables. Different backbones are representation choices, not separate classical learning algorithms.

The core registry includes a majority baseline, Logistic Regression, linear SVM, RBF SVM and ExtraTrees. Targeted RandomForest and SVM/RandomForest voting candidates are also retained. Median imputation, scaling where applicable, and searched PCA sit inside sklearn pipelines. The optional registry includes additional historical candidates; their presence is not evidence that they outperform the selected models.

The ResNet50 extension first trains a classifier head and then fine-tunes the backbone. The historical configuration uses a two-epoch head stage, up to 25 epochs for animals and 45 for birds, and horizontal-flip TTA. Fold summaries record validation-based early stopping/checkpoint selection. TTA averages predictions for the original and horizontally flipped image; it does not use test labels.

## What OOF does and does not establish

For each saved OOF prediction, fitting of that fold's estimator excludes that observation. However, the full selection procedure is not independent of the validation observations:

- **Classical models:** `GridSearchCV` selects hyperparameters on the training set's stratified folds. `cross_val_predict` then evaluates those selected hyperparameters with the same fold configuration. Hyperparameter selection has already seen information from those validation scores. OOF macro-F1 and mean fold macro-F1 are different summaries and are not interchangeable.
- **CNNs:** each fold's validation labels choose a best checkpoint, and the selected checkpoint is evaluated on that validation fold. That score is subject to checkpoint-selection optimism.
- **Provided features:** the archive does not establish how externally supplied precomputed PCA features were fitted. The pipeline's own PCA is fold-local; that does not prove the provenance of every supplied feature.
- **Pretraining:** ImageNet initialization supplies external supervised information. This project does not establish the absence of image overlap with pretraining sources.

Consequently, the portfolio describes historical validation results, not an unbiased estimate of the complete selection process, a new test benchmark, or statistically significant model superiority. A stronger future study would use nested CV for classical selection and an inner checkpoint-selection split inside each outer CNN training fold, followed by evaluation on untouched outer folds or a separate test set. These changes have not been run in this edition.

## Test data and public feedback

The implemented experiment entry point requests training-only features for selection. Prediction entry points refit on labelled training data and write `image_id,class_id` CSVs for unlabelled images. Structural checks reject overlapping train/test IDs. Rechecking the saved evidence does not independently prove the provenance of the original input data.

Earlier README prose recorded Kaggle public scores, but the supplied archive did not contain independently verifiable leaderboard evidence. Those numbers are omitted from the public results and resume claims. No leaderboard rank is claimed.
