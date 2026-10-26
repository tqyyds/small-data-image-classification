"""Fit selected models on labelled training data and create Kaggle CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR, RANDOM_SEED
from .data import load_task
from .features import load_feature_frame, make_train_test_matrices
from .models import (
    build_safe_param_grid,
    build_svm_random_forest_voting_classifier,
    build_voting_classifier,
    get_additional_kaggle_registry,
    get_model_registry,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["task1", "task2", "all"], default="all")
    parser.add_argument("--mode", choices=["core", "extended", "kaggle"], default="kaggle")
    parser.add_argument("--feature-type", default="best")
    parser.add_argument("--model-key", default="best")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transform-mode", choices=["official", "resize_only_224"], default="official")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _tasks_from_arg(task_arg: str) -> list[str]:
    return ["task1", "task2"] if task_arg == "all" else [task_arg]


def _select_best_result(task: str, output_dir: Path) -> tuple[str, str, str, str]:
    result_path = output_dir / "results" / "model_comparison.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"No result table found at {result_path}. Run experiments first.")
    df = pd.read_csv(result_path)
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        raise ValueError(f"No experiment results found for {task}.")
    if "is_smoke" in task_df.columns:
        task_df = task_df[task_df["is_smoke"].astype(str).str.lower() == "false"].copy()
        if task_df.empty:
            raise ValueError(f"Only smoke results found for {task}; select a model explicitly or run full experiments.")
    task_df = task_df.sort_values(["mean_cv_macro_f1", "mean_cv_accuracy"], ascending=False)
    row = task_df.iloc[0]
    transform_mode = str(row["transform_mode"]) if "transform_mode" in row and pd.notna(row["transform_mode"]) else "official"
    return str(row["mode"]), str(row["feature_type"]), transform_mode, str(row["model_key"])


def create_submission_for_task(args: argparse.Namespace, task: str) -> Path:
    output_dir = Path(args.output_dir)
    if args.feature_type == "best" or args.model_key == "best":
        mode, feature_type, transform_mode, model_key = _select_best_result(task, output_dir)
        if args.feature_type != "best":
            feature_type = args.feature_type
        if args.model_key != "best":
            model_key = args.model_key
        if args.transform_mode != "official":
            transform_mode = args.transform_mode
    else:
        mode, feature_type, transform_mode, model_key = args.mode, args.feature_type, args.transform_mode, args.model_key

    LOGGER.info("Creating submission for %s with %s | %s | %s | %s", task, mode, feature_type, transform_mode, model_key)
    loaded = load_task(task, args.data_root)
    device = None if args.device == "auto" else args.device
    features = load_feature_frame(
        task,
        feature_type,
        data_root=args.data_root,
        output_dir=output_dir,
        transform_mode=transform_mode,
        device=device,
        include_test=True,
    )
    x_train, y_train, x_test, test_ids = make_train_test_matrices(loaded, features, require_test=True)
    registry = get_model_registry(mode, feature_type)
    if mode == "kaggle":
        registry.update(get_additional_kaggle_registry(feature_type))
    if model_key == "voting_lr_rbf_extra":
        entry = {
            "model_name": "voting_lr_rbf_extra",
            "pipeline": build_voting_classifier(),
            "param_grid": {},
            "requires_scaling": True,
            "pca_allowed": True,
            "pca_required": False,
            "report_note": "Voting ensemble combining Logistic Regression, RBF SVM, and ExtraTrees.",
        }
    elif model_key in {"svm_random_forest", "svm_random_forest_vote"}:
        entry = {
            "model_name": "svm_random_forest",
            "pipeline": build_svm_random_forest_voting_classifier(),
            "param_grid": {
                "classifier__svm__classifier__C": [0.1, 1, 10],
                "classifier__random_forest__classifier__n_estimators": [300, 500],
                "classifier__random_forest__classifier__max_depth": [None, 20],
                "classifier__random_forest__classifier__min_samples_leaf": [1, 2],
            },
            "requires_scaling": True,
            "pca_allowed": False,
            "pca_required": False,
            "report_note": "Hard-voting ensemble combining a margin-based SVM with a RandomForest classifier.",
        }
    elif model_key not in registry:
        raise ValueError(f"Model '{model_key}' is not available in registry for {mode}/{feature_type}.")
    else:
        entry = registry[model_key]
    fold_train_size = max(1, len(y_train) * (args.cv - 1) // args.cv)
    param_grid = build_safe_param_grid(entry, fold_train_size, x_train.shape[1], quick=args.quick)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        entry["pipeline"],
        param_grid=param_grid,
        scoring=make_scorer(f1_score, average="macro", zero_division=0),
        refit=True,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    predictions = search.best_estimator_.predict(x_test).astype(int)
    submission = pd.DataFrame({"image_id": test_ids.to_numpy(), "class_id": predictions})
    output_path = output_dir / "submissions" / f"{task}_{feature_type}_{transform_mode}_{model_key}_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    for task in _tasks_from_arg(args.task):
        path = create_submission_for_task(args, task)
        print(path)


if __name__ == "__main__":
    main()
