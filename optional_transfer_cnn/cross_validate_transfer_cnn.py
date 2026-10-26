from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from train_transfer_cnn import (
    ClassInfo,
    ImageMetadataDataset,
    averaged_logits,
    build_transforms,
    create_model,
    evaluate,
    load_metadata,
    make_optimizer,
    make_scheduler,
    prediction_frame,
    predict_test,
    save_confusion_matrix,
    seed_everything,
    set_backbone_trainable,
    train_one_epoch,
)


TASK_DEFAULT_CV = {"task1": 3, "task2": 5}
TASK_DEFAULT_EPOCHS = {"task1": 25, "task2": 45}
TASK_DEFAULT_BATCH_SIZE = {"task1": 32, "task2": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stratified cross-validation for ImageNet-pretrained transfer CNNs."
    )
    parser.add_argument("--task", choices=["task1", "task2"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_transfer_cv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models_transfer_cv"))
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--final-refit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--make-final-submission",
        action="store_true",
        help="Train once on all labelled training data after CV and write the final test submission.",
    )
    parser.add_argument(
        "--refit-epochs",
        type=int,
        default=None,
        help="Epochs for final full-training refit. Defaults to median best CV epoch.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: cap epochs to 1 and fold count to at most 2.",
    )
    return parser.parse_args()


def validate_metadata(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    data_dir: Path,
    class_info: list[ClassInfo],
) -> dict[str, object]:
    required_train = {"image_id", "image_path", "class_id", "class_name"}
    required_test = {"image_id", "image_path"}
    missing_train = sorted(required_train - set(train_metadata.columns))
    missing_test = sorted(required_test - set(test_metadata.columns))
    if missing_train:
        raise ValueError(f"train_metadata.csv missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test_metadata.csv missing columns: {missing_test}")
    if train_metadata["image_id"].duplicated().any():
        raise ValueError("train_metadata.csv contains duplicate image_id values")
    if test_metadata["image_id"].duplicated().any():
        raise ValueError("test_metadata.csv contains duplicate image_id values")
    overlap = set(train_metadata["image_id"]).intersection(set(test_metadata["image_id"]))
    if overlap:
        raise ValueError(f"Train/test image_id overlap detected, e.g. {sorted(overlap)[:5]}")

    for source_name, frame in (("train", train_metadata), ("test", test_metadata)):
        missing_paths = [
            str(data_dir / path)
            for path in frame["image_path"].astype(str).tolist()
            if not (data_dir / path).exists()
        ]
        if missing_paths:
            raise FileNotFoundError(f"{source_name} metadata references missing images: {missing_paths[:5]}")

    class_ids = sorted(train_metadata["class_id"].astype(int).unique().tolist())
    expected_ids = [info.class_id for info in class_info]
    if class_ids != expected_ids:
        raise ValueError(f"Class IDs {class_ids} do not match class mapping {expected_ids}")
    if class_ids != list(range(10)):
        raise ValueError(f"Expected class_id range 0-9, found {class_ids}")

    mapping = (
        train_metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    if len(mapping) != len(class_info):
        raise ValueError("class_id/class_name mapping is inconsistent in train metadata")

    return {
        "train_rows": int(len(train_metadata)),
        "test_rows": int(len(test_metadata)),
        "class_ids": class_ids,
        "train_test_overlap": 0,
    }


def make_loader(
    metadata: pd.DataFrame,
    data_dir: Path,
    transform,
    has_labels: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = ImageMetadataDataset(metadata, data_dir, transform, has_labels=has_labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def compute_metric_dict(targets: list[int] | np.ndarray, preds: list[int] | np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
    }


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_fold(
    *,
    fold_index: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_info: list[ClassInfo],
    data_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    fold_seed = int(args.seed + fold_index)
    seed_everything(fold_seed)
    fold_dir = run_dir / f"fold_{fold_index:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_tfms, eval_tfms = build_transforms(args.image_size, aggressive=args.task == "task2")
    train_loader = make_loader(
        train_df,
        data_dir,
        train_tfms,
        has_labels=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = make_loader(
        val_df,
        data_dir,
        eval_tfms,
        has_labels=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    test_loader = make_loader(
        test_df,
        data_dir,
        eval_tfms,
        has_labels=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )

    model = create_model(args.arch, len(class_info)).to(device)
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, trainable=False)
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print(f"Fold {fold_index}: train={len(train_df)} val={len(val_df)}")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            print(f"Fold {fold_index}: backbone unfrozen.")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_targets, val_preds, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            use_tta=args.tta,
        )
        val_metrics = compute_metric_dict(val_targets, val_preds)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"Fold {fold_index} epoch {epoch:03d}/{args.epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Fold {fold_index}: early stopping after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    val_loss, val_acc, val_targets, val_preds, val_image_ids = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        use_tta=args.tta,
    )
    metrics = compute_metric_dict(val_targets, val_preds)
    fold_predictions = prediction_frame(val_image_ids, val_preds, class_info)
    fold_predictions.insert(0, "fold", fold_index)
    fold_predictions.insert(2, "true_class_id", val_targets)
    fold_predictions.insert(3, "correct", fold_predictions["true_class_id"] == fold_predictions["class_id"])
    fold_predictions.to_csv(fold_dir / "val_predictions.csv", index=False)

    report = classification_report(
        val_targets,
        val_preds,
        labels=[info.class_id for info in class_info],
        target_names=[info.class_name for info in class_info],
        zero_division=0,
    )
    (fold_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(fold_dir / "confusion_matrix.png", val_targets, val_preds, class_info)
    write_history(fold_dir / "history.csv", history)

    if args.save_checkpoints:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_info": [asdict(info) for info in class_info],
                "task": args.task,
                "arch": args.arch,
                "fold": fold_index,
                "best_epoch": best_epoch,
                "metrics": metrics,
                "args": vars(args),
            },
            args.model_dir / f"{args.task}_{args.arch}_transfer_cv_fold_{fold_index:02d}.pt",
        )

    # Producing fold test predictions is useful for diagnostics, but these are
    # not used for CV metrics or model selection.
    test_image_ids, test_preds = predict_test(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        use_tta=args.tta,
    )
    prediction_frame(test_image_ids, test_preds, class_info)[["image_id", "class_id"]].to_csv(
        fold_dir / "fold_test_submission.csv",
        index=False,
    )

    summary = {
        "task": args.task,
        "model": f"{args.arch}_transfer_tta" if args.tta else f"{args.arch}_transfer_no_tta",
        "arch": args.arch,
        "fold": fold_index,
        "cv_folds": args.cv,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "best_epoch": int(best_epoch),
        "seed": int(fold_seed),
        "device": str(device),
        "tta": bool(args.tta),
        "val_loss": float(val_loss),
    }
    return summary, fold_predictions


def save_oof_outputs(
    *,
    oof_predictions: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    class_info: list[ClassInfo],
    run_dir: Path,
    args: argparse.Namespace,
    fold_summaries: list[dict[str, object]],
    metadata_checks: dict[str, object],
) -> dict[str, object]:
    if len(oof_predictions) != len(train_metadata):
        raise ValueError("OOF row count does not match train metadata row count")
    if oof_predictions["image_id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate image_id values")
    if set(oof_predictions["image_id"]) != set(train_metadata["image_id"]):
        raise ValueError("OOF predictions do not exactly cover train image IDs")
    overlap = set(oof_predictions["image_id"]).intersection(set(test_metadata["image_id"]))
    if overlap:
        raise ValueError(f"OOF/test image_id overlap detected, e.g. {sorted(overlap)[:5]}")

    id_order = {image_id: idx for idx, image_id in enumerate(train_metadata["image_id"].tolist())}
    oof_predictions = oof_predictions.copy()
    oof_predictions["_order"] = oof_predictions["image_id"].map(id_order)
    oof_predictions = oof_predictions.sort_values("_order").drop(columns=["_order"])
    oof_predictions.to_csv(run_dir / "transfer_cnn_oof_predictions.csv", index=False)

    y_true = oof_predictions["true_class_id"].astype(int).to_numpy()
    y_pred = oof_predictions["class_id"].astype(int).to_numpy()
    oof_metrics = compute_metric_dict(y_true, y_pred)
    save_confusion_matrix(run_dir / "confusion_matrix_oof.png", y_true.tolist(), y_pred.tolist(), class_info)
    report = classification_report(
        y_true,
        y_pred,
        labels=[info.class_id for info in class_info],
        target_names=[info.class_name for info in class_info],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(run_dir / "classification_report_oof.csv")

    fold_summary_df = pd.DataFrame(fold_summaries)
    aggregate_row = {
        "task": args.task,
        "model": f"{args.arch}_transfer_tta" if args.tta else f"{args.arch}_transfer_no_tta",
        "arch": args.arch,
        "fold": "aggregate_oof",
        "cv_folds": args.cv,
        "n_train": int(len(train_metadata)),
        "n_val": int(len(oof_predictions)),
        "accuracy": oof_metrics["accuracy"],
        "macro_f1": oof_metrics["macro_f1"],
        "balanced_accuracy": oof_metrics["balanced_accuracy"],
        "best_epoch": "",
        "seed": int(args.seed),
        "device": fold_summaries[0]["device"] if fold_summaries else "",
        "tta": bool(args.tta),
        "val_loss": "",
    }
    summary_df = pd.concat([fold_summary_df, pd.DataFrame([aggregate_row])], ignore_index=True)
    summary_df.to_csv(run_dir / "transfer_cnn_cv_summary.csv", index=False)

    metrics_payload = {
        "oof_metrics": oof_metrics,
        "mean_fold_metrics": {
            "accuracy": float(fold_summary_df["accuracy"].mean()),
            "macro_f1": float(fold_summary_df["macro_f1"].mean()),
            "balanced_accuracy": float(fold_summary_df["balanced_accuracy"].mean()),
        },
        "metadata_checks": metadata_checks,
        "leakage_checks": {
            "oof_rows_equal_train_rows": True,
            "oof_unique_image_ids": True,
            "oof_test_overlap": 0,
        },
        "config": {
            "task": args.task,
            "arch": args.arch,
            "cv": args.cv,
            "epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "seed": args.seed,
            "tta": bool(args.tta),
            "image_size": args.image_size,
        },
    }
    (run_dir / "transfer_cnn_oof_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics_payload


def train_final_refit(
    *,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    class_info: list[ClassInfo],
    data_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    refit_epochs: int,
) -> Path:
    seed_everything(args.seed + 1000)
    refit_dir = run_dir / "final_submission"
    refit_dir.mkdir(parents=True, exist_ok=True)
    train_tfms, eval_tfms = build_transforms(args.image_size, aggressive=args.task == "task2")
    train_loader = make_loader(
        train_metadata,
        data_dir,
        train_tfms,
        has_labels=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
    )
    test_loader = make_loader(
        test_metadata,
        data_dir,
        eval_tfms,
        has_labels=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    model = create_model(args.arch, len(class_info)).to(device)
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, trainable=False)
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    total_steps = max(1, len(train_loader) * refit_epochs)
    warmup_steps = max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict[str, float]] = []

    for epoch in range(1, refit_epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            print("Final submission training: backbone unfrozen.")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
            }
        )
        print(f"Final submission epoch {epoch:03d}/{refit_epochs} train_acc={train_acc:.4f}")

    write_history(refit_dir / "history.csv", history)
    test_image_ids, test_preds = predict_test(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        use_tta=args.tta,
    )
    submission = prediction_frame(test_image_ids, test_preds, class_info)[["image_id", "class_id"]]
    submission_path = refit_dir / f"{args.task}_{args.arch}_transfer_cv_submission.csv"
    submission.to_csv(submission_path, index=False)
    latest_path = run_dir / f"{args.task}_{args.arch}_transfer_cv_submission.csv"
    submission.to_csv(latest_path, index=False)
    if args.save_checkpoints:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_info": [asdict(info) for info in class_info],
                "task": args.task,
                "arch": args.arch,
                "final_epochs": refit_epochs,
                "args": vars(args),
            },
            args.model_dir / f"{args.task}_{args.arch}_transfer_cv.pt",
        )
    return latest_path


def main() -> None:
    args = parse_args()
    args.cv = int(args.cv or TASK_DEFAULT_CV[args.task])
    args.epochs = int(args.epochs or TASK_DEFAULT_EPOCHS[args.task])
    args.batch_size = int(args.batch_size or TASK_DEFAULT_BATCH_SIZE[args.task])
    if args.quick:
        args.cv = min(args.cv, 2)
        args.epochs = 1
        args.patience = 0

    seed_everything(args.seed)
    train_metadata, test_metadata, class_info = load_metadata(args.data_dir)
    metadata_checks = validate_metadata(train_metadata, test_metadata, args.data_dir, class_info)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.task}_{args.arch}_transfer_cv_{args.cv}fold_{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {args.task}")
    print(f"Architecture: {args.arch}")
    print(f"CV folds: {args.cv}")
    print(f"Epochs: {args.epochs}")
    print(f"Data dir: {args.data_dir}")
    print(f"Train images: {len(train_metadata)}")
    print(f"Test images: {len(test_metadata)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TTA: {args.tta}")
    print(f"Run outputs: {run_dir}")
    print()

    splitter = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    y = train_metadata["class_id"].astype(int).to_numpy()
    fold_summaries: list[dict[str, object]] = []
    fold_predictions: list[pd.DataFrame] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(train_metadata, y), start=1):
        train_ids = set(train_metadata.iloc[train_idx]["image_id"])
        val_ids = set(train_metadata.iloc[val_idx]["image_id"])
        if train_ids.intersection(val_ids):
            raise ValueError(f"Fold {fold_index} train/validation overlap detected")
        train_df = train_metadata.iloc[train_idx].reset_index(drop=True)
        val_df = train_metadata.iloc[val_idx].reset_index(drop=True)
        fold_class_counts = val_df["class_id"].value_counts().sort_index()
        if len(fold_class_counts) != len(class_info):
            raise ValueError(f"Fold {fold_index} validation split does not contain all classes")
        summary, predictions = train_fold(
            fold_index=fold_index,
            train_df=train_df,
            val_df=val_df,
            test_df=test_metadata,
            class_info=class_info,
            data_dir=args.data_dir,
            args=args,
            run_dir=run_dir,
            device=device,
            use_amp=use_amp,
        )
        summary["val_class_counts"] = json.dumps(
            {str(int(k)): int(v) for k, v in fold_class_counts.to_dict().items()},
            sort_keys=True,
        )
        fold_summaries.append(summary)
        fold_predictions.append(predictions)

    oof_predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics_payload = save_oof_outputs(
        oof_predictions=oof_predictions,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        class_info=class_info,
        run_dir=run_dir,
        args=args,
        fold_summaries=fold_summaries,
        metadata_checks=metadata_checks,
    )

    if args.final_refit or args.make_final_submission:
        best_epochs = [int(row["best_epoch"]) for row in fold_summaries if int(row["best_epoch"]) > 0]
        refit_epochs = int(args.refit_epochs or max(1, round(float(np.median(best_epochs)))))
        submission_path = train_final_refit(
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            class_info=class_info,
            data_dir=args.data_dir,
            args=args,
            run_dir=run_dir,
            device=device,
            use_amp=use_amp,
            refit_epochs=refit_epochs,
        )
        print(f"Final submission: {submission_path}")

    print()
    print("OOF metrics:")
    for key, value in metrics_payload["oof_metrics"].items():
        print(f"  {key}: {value:.6f}")
    print(f"Summary CSV: {run_dir / 'transfer_cnn_cv_summary.csv'}")
    print(f"OOF predictions: {run_dir / 'transfer_cnn_oof_predictions.csv'}")


if __name__ == "__main__":
    main()
