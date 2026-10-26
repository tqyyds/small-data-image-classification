"""Exercise the CPU experiment and submission paths on temporary synthetic data."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root: Path) -> Path:
    """Create explicitly synthetic task2 data with ten classes and distinct IDs."""
    task_dir = root / "task2_data"
    task_dir.mkdir(parents=True)
    rows = []
    for index in range(110):
        label = index % 10
        image_id = f"synthetic_{index:03}"
        path = f"{image_id}.png"
        Image.new("RGB", (16, 16), color=(label * 20, index % 255, 90)).save(task_dir / path)
        row = {"image_id": image_id, "image_path": path}
        if index < 100:
            row.update(class_id=label, class_name=f"synthetic_class_{label}")
        rows.append(row)
    pd.DataFrame(rows[:100]).to_csv(task_dir / "train_metadata.csv", index=False)
    pd.DataFrame(rows[100:]).to_csv(task_dir / "test_metadata.csv", index=False)
    rng = np.random.default_rng(27)
    for filename, prefix in (("additional_features.csv", "extra"),
                             ("color_histogram.csv", "color"), ("hog_pca.csv", "hog")):
        features = pd.DataFrame(rng.normal(size=(110, 3)), columns=[f"{prefix}_{i}" for i in range(3)])
        features.insert(0, "image_id", [row["image_id"] for row in rows])
        # Deliberately shuffle IDs to exercise alignment rather than row-position joins.
        features.sample(frac=1, random_state=27).to_csv(task_dir / filename, index=False)
    return task_dir


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="image_classification_synthetic_") as temporary:
        root = Path(temporary)
        make_fixture(root / "data")
        common = ["--task", "task2", "--data-root", str(root / "data"),
                  "--output-dir", str(root / "outputs")]
        subprocess.run([sys.executable, "-m", "src.run_experiments", *common,
                        "--mode", "core", "--feature-type", "provided_engineered",
                        "--model-key", "logistic_regression", "--quick", "--cv", "3"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        table = pd.read_csv(root / "outputs/results/model_comparison.csv")
        if len(table) != 1 or not table.is_smoke.all() or table.iloc[0].n_train != 100:
            raise ValueError("Synthetic run was not correctly marked as a smoke experiment")
        predictions = next((root / "outputs/tables/oof_predictions").glob("*.csv"))
        oof = pd.read_csv(predictions)
        if len(oof) != 100 or oof.image_id.duplicated().any():
            raise ValueError("Synthetic OOF coverage failed")
        subprocess.run([sys.executable, "-m", "src.make_submissions", *common,
                        "--mode", "core", "--quick",
                        "--feature-type", "provided_engineered", "--model-key", "logistic_regression",
                        "--cv", "3"], cwd=ROOT, check=True, capture_output=True, text=True)
        submission = pd.read_csv(next((root / "outputs/submissions").glob("*.csv")))
        if list(submission.columns) != ["image_id", "class_id"] or len(submission) != 10:
            raise ValueError("Synthetic prediction CSV format/coverage failed")
    print("PASS: synthetic CPU feature extraction, shuffled-ID joins, CV, OOF and prediction CSV; temporary files removed.")


if __name__ == "__main__":
    main()
