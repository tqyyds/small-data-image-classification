from __future__ import annotations

import argparse
import copy
import csv
import math
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
from torchvision import models, transforms


TASK_DEFAULTS = {
    "task1": {
        "data_dir": "task1_data",
        "image_size": 224,
    },
    "task2": {
        "data_dir": "task2_data",
        "image_size": 224,
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
        image = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        image = self.transform(image)
        if self.has_labels:
            return image, int(row["class_id"]), row["image_id"]
        return image, row["image_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an ImageNet-pretrained CNN.")
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="task1")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--arch",
        choices=["efficientnet_b0", "efficientnet_b2", "resnet18", "resnet50", "mobilenet_v3_large"],
        default="efficientnet_b0",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip TTA for evaluation and test prediction.")
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


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[ClassInfo]]:
    train_metadata = pd.read_csv(data_dir / "train_metadata.csv")
    test_metadata = pd.read_csv(data_dir / "test_metadata.csv")
    class_rows = (
        train_metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .itertuples(index=False)
    )
    class_info = [ClassInfo(int(row.class_id), str(row.class_name)) for row in class_rows]
    return train_metadata, test_metadata, class_info


def build_transforms(image_size: int, aggressive: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_ops: list[object] = [
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
    ]
    if aggressive:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
    train_ops.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if aggressive:
        train_ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)))

    train_tfms = transforms.Compose(train_ops)
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def make_loaders(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    data_dir: Path,
    image_size: int,
    val_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    task: str,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df = train_test_split(
        train_metadata,
        test_size=val_split,
        random_state=seed,
        stratify=train_metadata["class_id"],
    )
    train_tfms, eval_tfms = build_transforms(image_size, aggressive=task == "task2")
    train_ds = ImageMetadataDataset(train_df, data_dir, train_tfms, has_labels=True)
    val_ds = ImageMetadataDataset(val_df, data_dir, eval_tfms, has_labels=True)
    test_ds = ImageMetadataDataset(test_metadata, data_dir, eval_tfms, has_labels=False)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def replace_classifier(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def create_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return replace_classifier(model, arch, num_classes)


def is_head_parameter(name: str) -> bool:
    return name.startswith("fc.") or name.startswith("classifier.")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not is_head_parameter(name):
            param.requires_grad = trainable


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if is_head_parameter(name):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def averaged_logits(model: nn.Module, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    logits = model(images)
    if use_tta:
        logits = (logits + model(torch.flip(images, dims=[3]))) / 2
    return logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    targets_all: list[int] = []
    preds_all: list[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.item()) * images.size(0)
        targets_all.extend(targets.detach().cpu().tolist())
        preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(targets_all, preds_all)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
) -> tuple[float, float, list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    targets_all: list[int] = []
    preds_all: list[int] = []
    image_ids_all: list[str] = []

    for images, targets, image_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = averaged_logits(model, images, use_tta)
            loss = criterion(logits, targets)

        total_loss += float(loss.item()) * images.size(0)
        targets_all.extend(targets.cpu().tolist())
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        image_ids_all.extend(list(image_ids))

    return total_loss / len(loader.dataset), accuracy_score(targets_all, preds_all), targets_all, preds_all, image_ids_all


@torch.no_grad()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
) -> tuple[list[str], list[int]]:
    model.eval()
    image_ids_all: list[str] = []
    preds_all: list[int] = []

    for images, image_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = averaged_logits(model, images, use_tta)
        image_ids_all.extend(list(image_ids))
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())

    return image_ids_all, preds_all


def prediction_frame(image_ids: list[str], preds: list[int], class_info: list[ClassInfo]) -> pd.DataFrame:
    id_to_name = {c.class_id: c.class_name for c in class_info}
    return pd.DataFrame(
        {
            "image_id": image_ids,
            "class_id": preds,
            "class_name": [id_to_name[int(pred)] for pred in preds],
        }
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_confusion_matrix(path: Path, targets: list[int], preds: list[int], class_info: list[ClassInfo]) -> None:
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    defaults = TASK_DEFAULTS[args.task]
    data_dir = args.data_dir or Path(defaults["data_dir"])
    image_size = args.image_size or int(defaults["image_size"])
    train_metadata, test_metadata, class_info = load_metadata(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    run_name = f"{args.task}_{args.arch}_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        data_dir=data_dir,
        image_size=image_size,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        task=args.task,
        device=device,
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

    print(f"Task: {args.task}")
    print(f"Architecture: {args.arch}")
    print(f"Data dir: {data_dir}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
    print(f"Classes: {len(class_info)}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TTA: {args.tta}")
    print()

    best_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            print("Backbone unfrozen.")

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
        val_loss, val_acc, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            use_tta=args.tta,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
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
    test_image_ids, test_preds = predict_test(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        use_tta=args.tta,
    )

    model_path = args.model_dir / f"{args.task}_{args.arch}_transfer_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_info": [c.__dict__ for c in class_info],
            "task": args.task,
            "arch": args.arch,
            "image_size": image_size,
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        model_path,
    )

    write_history(run_dir / "history.csv", history)

    val_predictions = prediction_frame(val_image_ids, val_preds, class_info)
    val_predictions.insert(1, "true_class_id", val_targets)
    val_predictions.insert(2, "correct", val_predictions["true_class_id"] == val_predictions["class_id"])
    val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    test_predictions = prediction_frame(test_image_ids, test_preds, class_info)
    test_predictions.to_csv(run_dir / "test_predictions_with_names.csv", index=False)
    test_predictions[["image_id", "class_id"]].to_csv(run_dir / "submission.csv", index=False)
    latest_submission = args.output_dir / f"{args.task}_{args.arch}_transfer_submission.csv"
    test_predictions[["image_id", "class_id"]].to_csv(latest_submission, index=False)

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
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Run outputs: {run_dir}")
    print(f"Latest submission CSV: {latest_submission}")


if __name__ == "__main__":
    main()
