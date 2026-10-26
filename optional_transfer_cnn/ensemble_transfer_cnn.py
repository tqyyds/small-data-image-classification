from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from train_transfer_cnn import ClassInfo, ImageMetadataDataset, create_model, load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average predictions from transfer CNN checkpoints.")
    parser.add_argument("--task", choices=["task1", "task2"], required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
    has_labels: bool,
) -> tuple[list[str], np.ndarray, list[int] | None]:
    model.eval()
    image_ids_all: list[str] = []
    targets_all: list[int] = []
    logits_all: list[np.ndarray] = []

    for batch in loader:
        if has_labels:
            images, targets, image_ids = batch
            targets_all.extend([int(t) for t in targets])
        else:
            images, image_ids = batch

        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if use_tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2

        image_ids_all.extend(list(image_ids))
        logits_all.append(logits.float().cpu().numpy())

    targets = targets_all if has_labels else None
    return image_ids_all, np.concatenate(logits_all, axis=0), targets


def prediction_frame(image_ids: list[str], preds: np.ndarray, class_info: list[ClassInfo]) -> pd.DataFrame:
    id_to_name = {c.class_id: c.class_name for c in class_info}
    class_ids = preds.astype(int).tolist()
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "class_id": class_ids,
            "class_name": [id_to_name[class_id] for class_id in class_ids],
        }
    )


def save_confusion_matrix(path: Path, targets: list[int], preds: np.ndarray, class_info: list[ClassInfo]) -> None:
    labels = [c.class_id for c in class_info]
    names = [c.class_name for c in class_info]
    cm = confusion_matrix(targets, preds, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    first_ckpt = load_checkpoint(args.checkpoints[0], device)
    first_args = first_ckpt.get("args", {})
    data_dir = args.data_dir or Path(first_args.get("data_dir") or f"{args.task}_data")
    val_split = args.val_split if args.val_split is not None else float(first_args.get("val_split", 0.2))
    seed = args.seed if args.seed is not None else int(first_args.get("seed", 27))

    train_metadata, test_metadata, class_info = load_metadata(data_dir)
    _, val_df = train_test_split(
        train_metadata,
        test_size=val_split,
        random_state=seed,
        stratify=train_metadata["class_id"],
    )

    run_name = f"{args.task}_transfer_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    val_logits_sum: np.ndarray | None = None
    test_logits_sum: np.ndarray | None = None
    val_image_ids_ref: list[str] | None = None
    test_image_ids_ref: list[str] | None = None
    val_targets_ref: list[int] | None = None

    print(f"Task: {args.task}")
    print(f"Data dir: {data_dir}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TTA: {args.tta}")
    print()

    for checkpoint_path in args.checkpoints:
        ckpt = load_checkpoint(checkpoint_path, device)
        arch = ckpt["arch"]
        image_size = int(ckpt["image_size"])
        model = create_model(arch, len(class_info)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        transform = eval_transform(image_size)
        val_loader = DataLoader(
            ImageMetadataDataset(val_df, data_dir, transform, has_labels=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            ImageMetadataDataset(test_metadata, data_dir, transform, has_labels=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        val_image_ids, val_logits, val_targets = collect_logits(model, val_loader, device, use_amp, args.tta, True)
        test_image_ids, test_logits, _ = collect_logits(model, test_loader, device, use_amp, args.tta, False)

        if val_image_ids_ref is None:
            val_image_ids_ref = val_image_ids
            test_image_ids_ref = test_image_ids
            val_targets_ref = val_targets
            val_logits_sum = val_logits
            test_logits_sum = test_logits
        else:
            if val_image_ids != val_image_ids_ref or test_image_ids != test_image_ids_ref:
                raise RuntimeError("Checkpoint predictions are not aligned by image_id.")
            val_logits_sum += val_logits
            test_logits_sum += test_logits

        single_preds = val_logits.argmax(axis=1)
        print(f"{checkpoint_path}: val_acc={accuracy_score(val_targets, single_preds):.4f}")

    assert val_logits_sum is not None
    assert test_logits_sum is not None
    assert val_image_ids_ref is not None
    assert test_image_ids_ref is not None
    assert val_targets_ref is not None

    val_logits_avg = val_logits_sum / len(args.checkpoints)
    test_logits_avg = test_logits_sum / len(args.checkpoints)
    val_preds = val_logits_avg.argmax(axis=1)
    test_preds = test_logits_avg.argmax(axis=1)
    val_acc = accuracy_score(val_targets_ref, val_preds)

    val_predictions = prediction_frame(val_image_ids_ref, val_preds, class_info)
    val_predictions.insert(1, "true_class_id", val_targets_ref)
    val_predictions.insert(2, "correct", val_predictions["true_class_id"] == val_predictions["class_id"])
    val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    test_predictions = prediction_frame(test_image_ids_ref, test_preds, class_info)
    test_predictions.to_csv(run_dir / "test_predictions_with_names.csv", index=False)
    test_predictions[["image_id", "class_id"]].to_csv(run_dir / "submission.csv", index=False)
    latest_submission = args.output_dir / f"{args.task}_transfer_ensemble_submission.csv"
    test_predictions[["image_id", "class_id"]].to_csv(latest_submission, index=False)

    report = classification_report(
        val_targets_ref,
        val_preds,
        labels=[c.class_id for c in class_info],
        target_names=[c.class_name for c in class_info],
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(run_dir / "confusion_matrix.png", val_targets_ref, val_preds, class_info)

    print()
    print(f"Ensemble validation accuracy: {val_acc:.4f}")
    print(f"Run outputs: {run_dir}")
    print(f"Latest submission CSV: {latest_submission}")


if __name__ == "__main__":
    main()
