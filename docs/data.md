# Data contract

Course datasets are not included. Obtain inputs only through a source you are entitled to use. This repository does not provide an unauthenticated course-data download or redistribute source images. Readers can verify saved results and run synthetic checks without those inputs.

## Directory layout

```text
data/
  task1_data/
    train_metadata.csv
    test_metadata.csv
    class_mapping.csv       # optional; inferred from training labels if absent
    additional_features.csv
    color_histogram.csv
    hog_pca.csv
    <image files referenced by metadata>
  task2_data/
    <same layout>
```

Use `--data-root /path/to/data` to keep inputs elsewhere. Transfer scripts instead accept `--data-dir /path/to/data/task1_data` or `task2_data`.

| File | Required columns / interpretation |
|---|---|
| `train_metadata.csv` | `image_id,image_path,class_id,class_name` |
| `test_metadata.csv` | `image_id,image_path`; no labels required |
| `class_mapping.csv` | `class_id,class_name`, consistent with training metadata |
| Each feature CSV | `image_id` plus numeric feature columns; names must be distinct across tables |

Image paths are relative to the task directory. IDs must be non-null and unique within each table. Training and test IDs must not overlap. Missing referenced images, duplicated feature keys and missing features for requested observations are errors. Row counts are warnings rather than hard constraints so synthetic fixtures and legitimate subsets can run.

The original tasks use class IDs 0–9. Animals are bird, butterfly, cat, deer, dog, elephant, frog, horse, sheep and spider. Birds are American Goldfinch, Blue Jay, Cardinal, Herring Gull, House Sparrow, Red-winged Blackbird, Ring-billed Gull, Song Sparrow, Wilson Warbler and Yellow Warbler. The archive's exact class-name spelling is preserved in the [evidence manifest](../results/manifest.json).

The synthetic smoke command creates 100 training and 10 test observations in an OS temporary directory, with class names prefixed `synthetic_class_`. It shuffles feature-table rows to check ID-based alignment and removes its files on exit. It is an integration fixture, not a model-quality benchmark.
