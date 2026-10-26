"""Create the representative OOF misclassification figure used in the report.

The script reads saved out-of-fold prediction CSVs for the selected frozen-feature
models, selects deterministic examples from the main off-diagonal error pairs,
and writes both a figure and a small table documenting the chosen image IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR
from .data import load_task


@dataclass(frozen=True)
class ErrorExampleSpec:
    task: str
    display_task: str
    oof_file: str
    true_class: str
    predicted_class: str
    title: str


EXAMPLES = [
    ErrorExampleSpec(
        task="task1",
        display_task="Task 1",
        oof_file="task1_core_efficientnet_b0_official_logistic_regression.csv",
        true_class="cat",
        predicted_class="dog",
        title="cat -> dog",
    ),
    ErrorExampleSpec(
        task="task1",
        display_task="Task 1",
        oof_file="task1_core_efficientnet_b0_official_logistic_regression.csv",
        true_class="dog",
        predicted_class="cat",
        title="dog -> cat",
    ),
    ErrorExampleSpec(
        task="task2",
        display_task="Task 2",
        oof_file="task2_kaggle_cnn_plus_provided_official_random_forest.csv",
        true_class="Herring_Gull",
        predicted_class="Ring_billed_Gull",
        title="Herring Gull -> Ring-billed Gull",
    ),
    ErrorExampleSpec(
        task="task2",
        display_task="Task 2",
        oof_file="task2_kaggle_cnn_plus_provided_official_random_forest.csv",
        true_class="Ring_billed_Gull",
        predicted_class="Herring_Gull",
        title="Ring-billed Gull -> Herring Gull",
    ),
    ErrorExampleSpec(
        task="task2",
        display_task="Task 2",
        oof_file="task2_kaggle_cnn_plus_provided_official_random_forest.csv",
        true_class="Yellow_Warbler",
        predicted_class="Wilson_Warbler",
        title="Yellow Warbler -> Wilson Warbler",
    ),
    ErrorExampleSpec(
        task="task2",
        display_task="Task 2",
        oof_file="task2_kaggle_cnn_plus_provided_official_random_forest.csv",
        true_class="Wilson_Warbler",
        predicted_class="Yellow_Warbler",
        title="Wilson Warbler -> Yellow Warbler",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate representative OOF misclassification figure.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=None,
        help="Directory containing OOF prediction CSVs. Defaults to outputs/tables/oof_predictions.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Destination PNG path. Defaults to outputs/figures/representative_oof_misclassifications.png.",
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=None,
        help="Destination CSV path. Defaults to outputs/tables/representative_misclassifications.csv.",
    )
    return parser.parse_args()


def select_error(oof_dir: Path, spec: ErrorExampleSpec) -> pd.Series:
    path = oof_dir / spec.oof_file
    if not path.exists():
        raise FileNotFoundError(f"OOF prediction file not found: {path}")
    df = pd.read_csv(path)
    required = {"image_id", "true_class_name", "predicted_class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    matches = df[
        (df["true_class_name"].astype(str) == spec.true_class)
        & (df["predicted_class_name"].astype(str) == spec.predicted_class)
    ].sort_values("image_id")
    if matches.empty:
        raise ValueError(f"No OOF error found for {spec.display_task}: {spec.title}")
    return matches.iloc[len(matches) // 2]


def image_lookup(task: str, data_root: Path) -> dict[str, Path]:
    loaded = load_task(task, data_root)
    return {
        str(row.image_id): Path(row.absolute_image_path)
        for row in loaded.train_metadata.itertuples(index=False)
    }


def relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    oof_dir = args.oof_dir or (output_dir / "tables" / "oof_predictions")
    figure_path = args.figure_path or (output_dir / "figures" / "representative_oof_misclassifications.png")
    table_path = args.table_path or (output_dir / "tables" / "representative_misclassifications.csv")

    lookups = {
        "task1": image_lookup("task1", args.data_root),
        "task2": image_lookup("task2", args.data_root),
    }

    records: list[dict[str, str]] = []
    selected_images: list[tuple[ErrorExampleSpec, Path, str]] = []
    for spec in EXAMPLES:
        row = select_error(oof_dir, spec)
        image_id = str(row["image_id"])
        image_path = lookups[spec.task].get(image_id)
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image for {image_id} not found in {spec.task} metadata.")
        selected_images.append((spec, image_path, image_id))
        records.append(
            {
                "task": spec.display_task,
                "image_id": image_id,
                "true_class_name": spec.true_class,
                "predicted_class_name": spec.predicted_class,
                "image_path": relative_to_data_root(image_path, args.data_root),
            }
        )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    for ax, (spec, image_path, image_id) in zip(axes.flat, selected_images, strict=True):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{spec.display_task}: {spec.title}", fontsize=9)
        ax.set_xlabel(image_id, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(records).to_csv(table_path, index=False)
    print(f"Wrote {figure_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
