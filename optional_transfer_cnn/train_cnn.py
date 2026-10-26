from __future__ import annotations

import argparse
import copy
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TASK_DEFAULTS = {
    "task1": {
        "data_dir": "task1_data",
        "image_size": 64,
    },
    "task2": {
        "data_dir": "task2_data",
        "image_size": 128,
    },
}


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    class_name: str


class ImageMetadataDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: Path,
        transform: transforms.Compose,
        has_labels: bool,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.data_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.has_labels:
            return image, int(row["class_id"]), row["image_id"]
        return image, row["image_id"]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            nn.MaxPool2d(2),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CNN for COMP30027 Project 2.")
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="task1")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[ClassInfo]]:
    train_metadata = pd.read_csv(data_dir / "train_metadata.csv")
    test_metadata = pd.read_csv(data_dir / "test_metadata.csv")
    classes = (
        train_metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .itertuples(index=False)
    )
    class_info = [ClassInfo(int(row.class_id), str(row.class_name)) for row in classes]
    return train_metadata, test_metadata, class_info


def make_loaders(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    data_dir: Path,
    image_size: int,
    val_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        train_metadata,
        test_size=val_split,
        random_state=seed,
        stratify=train_metadata["class_id"],
    )
    train_tfms, eval_tfms = build_transforms(image_size)
    train_ds = ImageMetadataDataset(train_df, data_dir, train_tfms, has_labels=True)
    val_ds = ImageMetadataDataset(val_df, data_dir, eval_tfms, has_labels=True)
    test_ds = ImageMetadataDataset(test_metadata, data_dir, eval_tfms, has_labels=False)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, train_df, val_df


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_image_ids: list[str] = []

    for images, targets, image_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_image_ids.extend(list(image_ids))

    return total_loss / len(loader.dataset), accuracy_score(all_targets, all_preds), all_targets, all_preds, all_image_ids


@torch.no_grad()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[list[str], list[int]]:
    model.eval()
    image_ids: list[str] = []
    preds: list[int] = []

    for images, batch_image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        image_ids.extend(list(batch_image_ids))
        preds.extend(logits.argmax(dim=1).cpu().tolist())

    return image_ids, preds


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion_matrix(
    path: Path,
    targets: list[int],
    preds: list[int],
    class_info: list[ClassInfo],
) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def prediction_frame(image_ids: list[str], preds: list[int], class_info: list[ClassInfo]) -> pd.DataFrame:
    id_to_name = {c.class_id: c.class_name for c in class_info}
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "class_id": preds,
            "class_name": [id_to_name[int(pred)] for pred in preds],
        }
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    task_defaults = TASK_DEFAULTS[args.task]
    data_dir = args.data_dir or Path(task_defaults["data_dir"])
    image_size = args.image_size or int(task_defaults["image_size"])
    train_metadata, test_metadata, class_info = load_metadata(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    run_name = f"{args.task}_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    args.model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, _, _ = make_loaders(
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        data_dir=data_dir,
        image_size=image_size,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    model = SmallCNN(num_classes=len(class_info), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Task: {args.task}")
    print(f"Data dir: {data_dir}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
    print(f"Classes: {len(class_info)}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_loss, val_acc, val_targets, val_preds, val_image_ids = evaluate(model, val_loader, criterion, device, use_amp)
    test_image_ids, test_preds = predict_test(model, test_loader, device, use_amp)

    model_path = args.model_dir / f"{args.task}_cnn_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_info": [c.__dict__ for c in class_info],
            "task": args.task,
            "image_size": image_size,
            "best_val_acc": best_acc,
            "args": vars(args),
        },
        model_path,
    )

    history_path = run_dir / "history.csv"
    write_history(history_path, history)

    val_predictions = prediction_frame(val_image_ids, val_preds, class_info)
    val_predictions.insert(1, "true_class_id", val_targets)
    val_predictions.insert(2, "correct", val_predictions["true_class_id"] == val_predictions["class_id"])
    val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    test_predictions = prediction_frame(test_image_ids, test_preds, class_info)
    test_predictions.to_csv(run_dir / "test_predictions_with_names.csv", index=False)
    test_predictions[["image_id", "class_id"]].to_csv(run_dir / "submission.csv", index=False)
    test_predictions[["image_id", "class_id"]].to_csv(args.output_dir / f"{args.task}_cnn_submission.csv", index=False)

    report = classification_report(
        val_targets,
        val_preds,
        labels=[c.class_id for c in class_info],
        target_names=[c.class_name for c in class_info],
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(run_dir / "confusion_matrix.png", val_targets, val_preds, class_info)

    print()
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Run outputs: {run_dir}")
    print(f"Latest submission CSV: {args.output_dir / f'{args.task}_cnn_submission.csv'}")


if __name__ == "__main__":
    main()
