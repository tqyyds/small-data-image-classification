# Reproduction guide

## 1. Verify the public evidence on CPU

Use Python 3.13 and a fresh virtual environment. Commands are run from the repository root.

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux alternative: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python -m src.verify_results
python -m src.synthetic_smoke
python -m src.check_portfolio
```

PowerShell users can use `.\.venv\Scripts\python.exe` instead of activation. CPU requirements are pinned to the portfolio verification environment; the original training archive did not preserve a package lock. Passing these checks does not demonstrate bit-for-bit reproduction of historical GPU training.

`src.verify_results` reads the manifest, verifies historical file SHA-256 values, checks prediction IDs/labels, recomputes accuracy/balanced accuracy/macro-F1, and compares all available per-class reports, OOF result-table rows and CNN JSON/fold summaries. It does not need raw images or weights.

```bash
python -m src.verify_results --write results/verification.json
python -m src.portfolio_figures
```

Only the second command regenerates the presentation figures and bird class-level table. Historical evidence under `results/historical/` remains untouched. New experiment outputs default to ignored `outputs/`, separately from the archived evidence.

## 2. Run classical baselines with your authorised course data

Follow the [data contract](data.md). These commands use provided features and do not need PyTorch:

```bash
python -m src.run_experiments --task task1 --mode core --feature-type provided --model-key logistic_regression --cv 3 --data-root ./data
python -m src.run_experiments --task task2 --mode core --feature-type provided_engineered --model-key logistic_regression --cv 5 --force-engineered --data-root ./data
```

For a small check on real data, add `--quick --max-rows 120 --cv 3` and use a separate `--output-dir outputs/smoke`. Quick/subsampled runs are marked `is_smoke=True` and cannot be automatically selected as the best production of a full experiment. They are not comparable to full historical results.

## 3. Install optional deep-learning dependencies

```bash
python -m pip install -r requirements-deep.txt
python optional_transfer_cnn/verify_cuda.py
```

`requirements-deep.txt` declares compatible package ranges, not a validated GPU lock. Select matching PyTorch and torchvision CPU/CUDA builds for your hardware. The archive used CUDA, but this edition does not claim a newly tested CUDA stack. Frozen-backbone extraction accepts `--device cpu`, `--device cuda`, or `--device auto`; CPU extraction can be slow. The first deep run may download ImageNet weights.

## 4. Re-run the selected frozen-feature configurations

```bash
python -m src.run_experiments --task task1 --mode core --feature-type efficientnet_b0 --model-key logistic_regression --cv 3 --data-root ./data
python -m src.run_experiments --task task2 --mode kaggle --feature-type cnn_plus_provided --include-random-forest --model-key random_forest --cv 5 --data-root ./data
```

For fixed-classifier representation ablations, use `--mode core --model-key logistic_regression` with `provided`, `provided_engineered`, `efficientnet_b0`, and respectively `efficientnet_b0_plus_provided` (animals) or `cnn_plus_provided` (birds). Use 3 and 5 folds respectively. Full grids and feature extraction may be expensive.

## 5. Re-run the historical transfer-learning protocol

```bash
python optional_transfer_cnn/cross_validate_transfer_cnn.py --task task1 --data-dir ./data/task1_data --arch resnet50 --cv 3 --epochs 25 --batch-size 32 --num-workers 0 --seed 27 --tta --output-dir outputs/transfer_cnn_cv --model-dir outputs/models_cv
python optional_transfer_cnn/cross_validate_transfer_cnn.py --task task2 --data-dir ./data/task2_data --arch resnet50 --cv 5 --epochs 45 --batch-size 16 --num-workers 0 --seed 27 --tta --output-dir outputs/transfer_cnn_cv --model-dir outputs/models_cv
```

These commands preserve the original validation/checkpoint-selection approach, including its [limitations](methodology.md). They do not implement nested evaluation. Use `--make-final-submission` for full-training refit and unlabelled test predictions, or `--save-checkpoints` to save weights locally. GPU training was not rerun for the portfolio.

## 6. Generate selected prediction CSVs

After a full experiment, fit and generate predictions with explicit configurations:

```bash
python -m src.make_submissions --task task1 --mode core --feature-type efficientnet_b0 --model-key logistic_regression --cv 3 --data-root ./data
python -m src.make_submissions --task task2 --mode kaggle --feature-type cnn_plus_provided --model-key random_forest --cv 5 --data-root ./data
```

The output has exactly `image_id,class_id`. Automatic `best` selection reads newly generated `outputs/results/model_comparison.csv`, excludes smoke rows, and does not silently use historical evidence. The original report-submission helper is retained for users who regenerate the complete report experiment set. The image-error helper requires private local data; its image montage is deliberately not distributed.

## 7. Package the public portfolio

```bash
python -m src.package_results
```

This command packages only the reviewed paths in `public-files.json`. It excludes local datasets, credentials, environments, checkpoints and regenerated outputs even when those exist alongside the source.
