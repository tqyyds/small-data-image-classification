"""Create Kaggle-format test predictions for every reported model.

This script is intended for the final code/results archive. It refits each
non-smoke experiment recorded in outputs/results/model_comparison.csv on the
full labelled training set using its stored best hyperparameters, then writes
one Kaggle-format CSV per model/feature/protocol combination. It also writes
majority-class DummyClassifier baselines and copies the cross-validated
ResNet50 transfer-learning submissions into the same directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR, RANDOM_SEED
from .data import load_task
from .features import load_feature_frame, make_train_test_matrices
from .models import get_additional_kaggle_registry, get_model_registry

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "submissions_report_models",
        help="Directory for the generated Kaggle-format CSV files.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _decode_param(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "PCA":
        return PCA(n_components=int(value["n_components"]), random_state=RANDOM_SEED)
    if isinstance(value, dict):
        return {key: _decode_param(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_param(item) for item in value]
    return value


def _parse_best_params(raw: Any) -> dict[str, Any]:
    if pd.isna(raw) or raw in ("", "{}"):
        return {}
    decoded = json.loads(str(raw))
    return {key: _decode_param(value) for key, value in decoded.items()}


def _safe_name(*parts: object) -> str:
    return "_".join(str(part).replace(" ", "_").replace("/", "_") for part in parts if str(part))


def _manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _registry_entry(mode: str, feature_type: str, model_key: str) -> dict[str, Any]:
    registry = get_model_registry(mode, feature_type)
    if mode == "kaggle":
        registry.update(get_additional_kaggle_registry(feature_type))
    if model_key not in registry:
        raise KeyError(f"Model {model_key!r} is unavailable for {mode}/{feature_type}.")
    return registry[model_key]


def _fit_and_write_submission(
    *,
    task: str,
    mode: str,
    feature_type: str,
    transform_mode: str,
    model_key: str,
    best_params: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    submission_dir: Path,
    device: str | None,
) -> Path:
    loaded = load_task(task, data_root)
    features = load_feature_frame(
        task=task,
        feature_type=feature_type,
        data_root=data_root,
        output_dir=output_dir,
        transform_mode=transform_mode,
        device=device,
        include_test=True,
    )
    x_train, y_train, x_test, test_ids = make_train_test_matrices(loaded, features, require_test=True)
    entry = _registry_entry(mode, feature_type, model_key)
    model = clone(entry["pipeline"])
    if best_params:
        model.set_params(**best_params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).astype(int)
    submission = pd.DataFrame({"image_id": test_ids.to_numpy(), "class_id": predictions})
    path = submission_dir / f"{_safe_name(task, mode, feature_type, transform_mode, model_key)}_submission.csv"
    submission.to_csv(path, index=False)
    return path


def _write_dummy_submission(task: str, data_root: Path, submission_dir: Path) -> Path:
    loaded = load_task(task, data_root)
    majority_class = int(loaded.train_metadata["class_id"].mode().iloc[0])
    submission = pd.DataFrame(
        {
            "image_id": loaded.test_metadata["image_id"].to_numpy(),
            "class_id": majority_class,
        }
    )
    path = submission_dir / f"{task}_dummy_majority_submission.csv"
    submission.to_csv(path, index=False)
    return path


def _copy_transfer_submissions(output_dir: Path, submission_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for task in ("task1", "task2"):
        source = output_dir / "transfer_cnn_cv" / f"{task}_resnet50_transfer_cv_submission.csv"
        if not source.exists():
            LOGGER.warning("Transfer CNN submission not found: %s", source)
            continue
        destination = submission_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _validate_submission(path: Path, task: str, data_root: Path) -> None:
    loaded = load_task(task, data_root)
    expected_ids = loaded.test_metadata["image_id"].astype(str).tolist()
    df = pd.read_csv(path)
    if list(df.columns) != ["image_id", "class_id"]:
        raise ValueError(f"{path} has columns {list(df.columns)}, expected ['image_id', 'class_id'].")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate image_id values.")
    if df["image_id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"{path} image_id values do not match the {task} test metadata order.")
    if not df["class_id"].between(0, 9).all():
        raise ValueError(f"{path} contains class_id values outside 0-9.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    output_dir = args.output_dir
    submission_dir = args.submission_dir
    project_root = Path.cwd()
    submission_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in submission_dir.glob("*.csv"):
        old_csv.unlink()
    device = None if args.device == "auto" else args.device

    result_path = output_dir / "results" / "model_comparison.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing model comparison table: {result_path}")
    results = pd.read_csv(result_path)
    if "is_smoke" in results.columns:
        results = results[results["is_smoke"] == False].copy()
    results = results.drop_duplicates(["task", "mode", "feature_type", "transform_mode", "model_key"])

    manifest_rows: list[dict[str, object]] = []
    for task in ("task1", "task2"):
        path = _write_dummy_submission(task, args.data_root, submission_dir)
        _validate_submission(path, task, args.data_root)
        manifest_rows.append(
            {
                "task": task,
                "model_key": "dummy",
                "feature_type": "majority_class",
                "path": _manifest_path(path, project_root),
            }
        )
        print(path)

    for row in results.itertuples(index=False):
        task = str(row.task)
        mode = str(row.mode)
        feature_type = str(row.feature_type)
        transform_mode = str(row.transform_mode)
        model_key = str(row.model_key)
        best_params = _parse_best_params(row.best_params)
        LOGGER.info("Creating %s | %s | %s | %s | %s", task, mode, feature_type, transform_mode, model_key)
        path = _fit_and_write_submission(
            task=task,
            mode=mode,
            feature_type=feature_type,
            transform_mode=transform_mode,
            model_key=model_key,
            best_params=best_params,
            data_root=args.data_root,
            output_dir=output_dir,
            submission_dir=submission_dir,
            device=device,
        )
        _validate_submission(path, task, args.data_root)
        manifest_rows.append(
            {
                "task": task,
                "mode": mode,
                "feature_type": feature_type,
                "transform_mode": transform_mode,
                "model_key": model_key,
                "path": _manifest_path(path, project_root),
            }
        )
        print(path)

    for path in _copy_transfer_submissions(output_dir, submission_dir):
        task = path.name.split("_")[0]
        _validate_submission(path, task, args.data_root)
        manifest_rows.append(
            {
                "task": task,
                "mode": "transfer_cnn_cv",
                "feature_type": "trainable_resnet50",
                "transform_mode": "tta",
                "model_key": "resnet50_transfer_tta",
                "path": _manifest_path(path, project_root),
            }
        )
        print(path)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = submission_dir / "submission_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(manifest_path)


if __name__ == "__main__":
    main()
