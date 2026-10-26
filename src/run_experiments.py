"""Run train-only model selection experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR, RANDOM_SEED
from .data import load_task
from .evaluation import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    serialise_params,
)
from .features import load_feature_frame, make_train_test_matrices
from .models import (
    build_safe_param_grid,
    build_voting_classifier,
    get_additional_kaggle_registry,
    get_model_registry,
)

LOGGER = logging.getLogger(__name__)


DEFAULT_FEATURES_BY_MODE = {
    "core": ["provided", "provided_engineered", "efficientnet_b0", "resnet50", "cnn_plus_provided"],
    "extended": ["provided", "provided_engineered", "efficientnet_b0", "resnet50", "cnn_plus_provided"],
    "kaggle": ["efficientnet_b0", "resnet50", "mobilenet_v3_large", "cnn_plus_provided"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["task1", "task2", "all"], default="all")
    parser.add_argument("--mode", choices=["core", "extended", "kaggle"], default="core")
    parser.add_argument("--feature-type", action="append", help="Feature type to run. Can be repeated.")
    parser.add_argument("--model-key", action="append", help="Run only selected registry model keys. Can be repeated.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transform-mode", choices=["official", "resize_only_224"], default="official")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use the first value from each grid for smoke tests.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional train-row cap for smoke tests only.")
    parser.add_argument("--include-voting", action="store_true", help="Run optional voting candidate in Kaggle mode.")
    parser.add_argument("--include-random-forest", action="store_true", help="Run RandomForest and SVM+RandomForest candidates in Kaggle mode.")
    parser.add_argument("--force-engineered", action="store_true")
    parser.add_argument("--force-deep", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _tasks_from_arg(task_arg: str) -> list[str]:
    return ["task1", "task2"] if task_arg == "all" else [task_arg]


def _feature_types_from_args(mode: str, feature_type_args: list[str] | None) -> list[str]:
    return feature_type_args or DEFAULT_FEATURES_BY_MODE[mode]


def _maybe_cap_rows(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(x_train) <= max_rows:
        return x_train, y_train
    per_class = max(1, int(max_rows / y_train.nunique()))
    sampled_parts = []
    row_frame = pd.DataFrame({"row_id": np.arange(len(y_train)), "y": y_train.to_numpy()})
    for _, group in row_frame.groupby("y"):
        sampled_parts.append(group.sample(min(per_class, len(group)), random_state=RANDOM_SEED))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    row_ids = sampled["row_id"].sort_values().to_numpy()
    return x_train.iloc[row_ids], y_train.iloc[row_ids]


def run_single_model(
    task: str,
    mode: str,
    feature_type: str,
    model_key: str,
    entry: dict,
    transform_mode: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    class_mapping: pd.DataFrame,
    output_dir: Path,
    cv_folds: int,
    quick: bool,
    is_smoke: bool,
) -> dict:
    LOGGER.info("Running %s | %s | %s | %s", task, mode, feature_type, model_key)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
    }
    fold_train_size = max(1, len(y_train) * (cv_folds - 1) // cv_folds)
    param_grid = build_safe_param_grid(
        entry,
        n_train=fold_train_size,
        n_features=x_train.shape[1],
        quick=quick,
    )
    search = GridSearchCV(
        estimator=entry["pipeline"],
        param_grid=param_grid,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    best_estimator = search.best_estimator_
    y_pred = cross_val_predict(clone(best_estimator), x_train, y_train, cv=cv, n_jobs=-1)
    metrics = compute_metrics(y_train.to_numpy(), y_pred)

    stem = f"{task}_{mode}_{feature_type}_{transform_mode}_{model_key}"
    class_name_by_id = class_mapping.set_index("class_id")["class_name"].to_dict()
    oof_predictions = pd.DataFrame(
        {
            "image_id": y_train.index.to_numpy(),
            "true_class_id": y_train.to_numpy(),
            "predicted_class_id": y_pred,
        }
    )
    oof_predictions["true_class_name"] = oof_predictions["true_class_id"].map(class_name_by_id)
    oof_predictions["predicted_class_name"] = oof_predictions["predicted_class_id"].map(class_name_by_id)
    oof_path = output_dir / "tables" / "oof_predictions" / f"{stem}.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_csv(oof_path, index=False)
    save_classification_report(
        y_train.to_numpy(),
        y_pred,
        class_mapping,
        output_dir / "tables" / "classification_reports" / f"{stem}.csv",
    )
    save_confusion_matrix(
        y_train.to_numpy(),
        y_pred,
        class_mapping,
        output_dir / "figures" / f"confusion_matrix_{stem}.png",
        title=f"{task} {model_key} ({feature_type})",
    )

    best_index = int(search.best_index_)
    result = {
        "task": task,
        "mode": mode,
        "feature_type": feature_type,
        "transform_mode": transform_mode,
        "model_key": model_key,
        "model_name": entry["model_name"],
        "n_train": int(len(y_train)),
        "n_features": int(x_train.shape[1]),
        "cv_folds": int(cv_folds),
        "is_smoke": bool(is_smoke),
        "best_params": serialise_params(search.best_params_),
        "mean_cv_accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
        "mean_cv_macro_f1": float(search.cv_results_["mean_test_macro_f1"][best_index]),
        "mean_cv_balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][best_index]),
        "oof_accuracy": metrics["accuracy"],
        "oof_macro_f1": metrics["macro_f1"],
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "report_note": entry["report_note"],
    }
    return result


def run_experiments(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    device = None if args.device == "auto" else args.device
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for task in _tasks_from_arg(args.task):
        loaded = load_task(task, args.data_root)
        for feature_type in _feature_types_from_args(args.mode, args.feature_type):
            LOGGER.info("Loading features for %s | %s", task, feature_type)
            feature_frame = load_feature_frame(
                task,
                feature_type,
                data_root=args.data_root,
                output_dir=output_dir,
                transform_mode=args.transform_mode,
                device=device,
                include_test=False,
                force_engineered=args.force_engineered,
                force_deep=args.force_deep,
            )
            x_train, y_train, _, _ = make_train_test_matrices(loaded, feature_frame, require_test=False)
            x_train, y_train = _maybe_cap_rows(x_train, y_train, args.max_rows)
            registry = get_model_registry(args.mode, feature_type)
            if args.mode == "kaggle" and args.include_random_forest:
                registry.update(get_additional_kaggle_registry(feature_type))
            for model_key, entry in registry.items():
                if args.model_key and model_key not in set(args.model_key):
                    continue
                results.append(
                    run_single_model(
                        task=task,
                        mode=args.mode,
                        feature_type=feature_type,
                        model_key=model_key,
                        entry=entry,
                        transform_mode=args.transform_mode,
                        x_train=x_train,
                        y_train=y_train,
                        class_mapping=loaded.class_mapping,
                        output_dir=output_dir,
                        cv_folds=args.cv,
                        quick=args.quick,
                        is_smoke=bool(args.quick or args.max_rows is not None),
                    )
                )
            if args.mode == "kaggle" and args.include_voting:
                entry = {
                    "model_name": "voting_lr_rbf_extra",
                    "pipeline": build_voting_classifier(),
                    "param_grid": {},
                    "requires_scaling": True,
                    "pca_allowed": True,
                    "pca_required": False,
                    "report_note": "Voting ensemble combining Logistic Regression, RBF SVM, and ExtraTrees.",
                }
                results.append(
                    run_single_model(
                        task=task,
                        mode=args.mode,
                        feature_type=feature_type,
                        model_key="voting_lr_rbf_extra",
                        entry=entry,
                        transform_mode=args.transform_mode,
                        x_train=x_train,
                        y_train=y_train,
                        class_mapping=loaded.class_mapping,
                        output_dir=output_dir,
                        cv_folds=args.cv,
                        quick=args.quick,
                        is_smoke=bool(args.quick or args.max_rows is not None),
                    )
                )

    result_df = pd.DataFrame(results)
    output_path = output_dir / "results" / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        result_df = pd.concat([existing, result_df], ignore_index=True)
    result_df.to_csv(output_path, index=False)
    return result_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    results = run_experiments(args)
    print(results.sort_values(["task", "mean_cv_macro_f1", "mean_cv_accuracy"], ascending=[True, False, False]).tail(20))


if __name__ == "__main__":
    main()
