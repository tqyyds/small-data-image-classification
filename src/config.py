"""Project-wide paths and constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ML_ROOT = DEFAULT_PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "outputs"
RANDOM_SEED = 27


@dataclass(frozen=True)
class TaskConfig:
    task: str
    data_dir_name: str
    expected_train_rows: int
    expected_test_rows: int
    expected_feature_rows: int
    image_size: int


TASKS: dict[str, TaskConfig] = {
    "task1": TaskConfig(
        task="task1",
        data_dir_name="task1_data",
        expected_train_rows=3750,
        expected_test_rows=1250,
        expected_feature_rows=5000,
        image_size=64,
    ),
    "task2": TaskConfig(
        task="task2",
        data_dir_name="task2_data",
        expected_train_rows=417,
        expected_test_rows=180,
        expected_feature_rows=597,
        image_size=128,
    ),
}


BACKBONES = ("efficientnet_b0", "resnet50", "mobilenet_v3_large")
TRANSFORM_MODES = ("official", "resize_only_224")
