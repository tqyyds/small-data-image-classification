"""Data loading and validation utilities.

The checks in this module deliberately distinguish between soft count
warnings and hard structural failures. The project specification gives
approximate dataset sizes, so row counts are observed expectations rather
than brittle assertions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_ML_ROOT, TASKS, TaskConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedTask:
    config: TaskConfig
    data_dir: Path
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    class_mapping: pd.DataFrame


def get_task_config(task: str) -> TaskConfig:
    if task not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task}'. Expected one of: {valid}")
    return TASKS[task]


def task_data_dir(task: str, data_root: Path | str = DEFAULT_ML_ROOT) -> Path:
    config = get_task_config(task)
    return Path(data_root) / config.data_dir_name


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _warn_if_count_differs(name: str, observed: int, expected: int, tolerance: float = 0.10) -> None:
    if expected <= 0:
        return
    rel_diff = abs(observed - expected) / expected
    if rel_diff > tolerance:
        LOGGER.warning(
            "%s has %s rows; observed local expectation is approximately %s rows.",
            name,
            observed,
            expected,
        )


def _validate_required_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _validate_unique_ids(df: pd.DataFrame, source_name: str) -> None:
    if df["image_id"].isna().any():
        raise ValueError(f"{source_name} contains null image_id values")
    duplicates = df.loc[df["image_id"].duplicated(), "image_id"].head(10).tolist()
    if duplicates:
        raise ValueError(f"{source_name} contains duplicated image_id values, e.g. {duplicates}")


def _resolve_image_paths(df: pd.DataFrame, data_dir: Path, source_name: str) -> pd.DataFrame:
    resolved = df.copy()
    resolved["image_path"] = resolved["image_path"].astype(str)
    resolved["absolute_image_path"] = resolved["image_path"].map(lambda p: str((data_dir / p).resolve()))
    missing = [p for p in resolved["absolute_image_path"].head(len(resolved)) if not Path(p).exists()]
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"{source_name} references missing image files, e.g. {preview}")
    return resolved


def load_task(task: str, data_root: Path | str = DEFAULT_ML_ROOT) -> LoadedTask:
    config = get_task_config(task)
    data_dir = task_data_dir(task, data_root)
    if not data_dir.exists():
        raise FileNotFoundError(f"Task data directory not found: {data_dir}")

    train_path = data_dir / "train_metadata.csv"
    test_path = data_dir / "test_metadata.csv"
    train_df = _read_csv(train_path)
    test_df = _read_csv(test_path)

    _validate_required_columns(train_df, {"image_id", "image_path", "class_id", "class_name"}, train_path)
    _validate_required_columns(test_df, {"image_id", "image_path"}, test_path)
    _validate_unique_ids(train_df, f"{task} train metadata")
    _validate_unique_ids(test_df, f"{task} test metadata")

    overlap = sorted(set(train_df["image_id"]).intersection(set(test_df["image_id"])))
    if overlap:
        raise ValueError(f"{task} train/test image_id overlap detected, e.g. {overlap[:10]}")

    train_df = _resolve_image_paths(train_df, data_dir, f"{task} train metadata")
    test_df = _resolve_image_paths(test_df, data_dir, f"{task} test metadata")

    _warn_if_count_differs(f"{task} train metadata", len(train_df), config.expected_train_rows)
    _warn_if_count_differs(f"{task} test metadata", len(test_df), config.expected_test_rows)

    mapping_path = data_dir / "class_mapping.csv"
    if mapping_path.exists():
        class_mapping = _read_csv(mapping_path)
        _validate_required_columns(class_mapping, {"class_id", "class_name"}, mapping_path)
    else:
        class_mapping = (
            train_df[["class_id", "class_name"]]
            .drop_duplicates()
            .sort_values("class_id")
            .reset_index(drop=True)
        )

    train_mapping = (
        train_df[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    if not train_mapping.reset_index(drop=True).equals(
        class_mapping[["class_id", "class_name"]].sort_values("class_id").reset_index(drop=True)
    ):
        raise ValueError(f"{task} class mapping does not match train labels.")

    return LoadedTask(
        config=config,
        data_dir=data_dir,
        train_metadata=train_df,
        test_metadata=test_df,
        class_mapping=class_mapping,
    )


def load_provided_features(task: str, data_root: Path | str = DEFAULT_ML_ROOT) -> pd.DataFrame:
    config = get_task_config(task)
    data_dir = task_data_dir(task, data_root)
    feature_files = ["additional_features.csv", "color_histogram.csv", "hog_pca.csv"]
    merged: pd.DataFrame | None = None
    for file_name in feature_files:
        path = data_dir / file_name
        df = _read_csv(path)
        _validate_required_columns(df, {"image_id"}, path)
        _validate_unique_ids(df, f"{task} {file_name}")
        merged = df if merged is None else merged.merge(df, on="image_id", how="inner", validate="one_to_one")

    assert merged is not None
    _warn_if_count_differs(f"{task} provided features", len(merged), config.expected_feature_rows)
    return merged


def get_label_vector(loaded: LoadedTask) -> pd.Series:
    return loaded.train_metadata.set_index("image_id")["class_id"].astype(int)
