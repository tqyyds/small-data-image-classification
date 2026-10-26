"""Frozen ImageNet CNN feature extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR
from .data import load_task

LOGGER = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose):
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        row = self.metadata.iloc[index]
        image = Image.open(row["absolute_image_path"]).convert("RGB")
        return row["image_id"], self.transform(image)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = backbone
        if backbone == "resnet50":
            self.weights = get_weights(backbone)
            self.model = models.resnet50(weights=self.weights)
            self.feature_model = nn.Sequential(*list(self.model.children())[:-1])
        elif backbone == "efficientnet_b0":
            self.weights = get_weights(backbone)
            self.model = models.efficientnet_b0(weights=self.weights)
        elif backbone == "mobilenet_v3_large":
            self.weights = get_weights(backbone)
            self.model = models.mobilenet_v3_large(weights=self.weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone == "resnet50":
            out = self.feature_model(x)
            return torch.flatten(out, 1)
        if self.backbone in {"efficientnet_b0", "mobilenet_v3_large"}:
            out = self.model.features(x)
            out = self.model.avgpool(out)
            return torch.flatten(out, 1)
        raise RuntimeError(f"Unsupported backbone: {self.backbone}")


def get_weights(backbone: str):
    if backbone == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if backbone == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    if backbone == "mobilenet_v3_large":
        return models.MobileNet_V3_Large_Weights.DEFAULT
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_transform(backbone: str, transform_mode: str = "official") -> transforms.Compose:
    weights = get_weights(backbone)
    if transform_mode == "official":
        return weights.transforms()
    if transform_mode == "resize_only_224":
        official_transform = weights.transforms()
        mean = getattr(official_transform, "mean", [0.485, 0.456, 0.406])
        std = getattr(official_transform, "std", [0.229, 0.224, 0.225])
        return transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    raise ValueError("transform_mode must be 'official' or 'resize_only_224'")


def _cache_path(output_dir: Path, task: str, backbone: str, transform_mode: str, include_test: bool) -> Path:
    scope = "all" if include_test else "train"
    return output_dir / "cached_features" / f"{task}_{scope}_{backbone}_{transform_mode}_embeddings.csv"


def _extract_deep_features_from_metadata(
    metadata: pd.DataFrame,
    task: str,
    backbone: str,
    transform_mode: str,
    batch_size: int,
    runtime_device: torch.device,
) -> pd.DataFrame:
    extractor = FeatureExtractor(backbone).to(runtime_device)
    transform = build_transform(backbone, transform_mode)
    dataset = ImagePathDataset(metadata, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=runtime_device.type == "cuda",
    )
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch_idx, (image_ids, images) in enumerate(loader):
            LOGGER.info(
                "Extracting %s %s features for %s: batch %s/%s",
                backbone,
                transform_mode,
                task,
                batch_idx + 1,
                len(loader),
            )
            images = images.to(runtime_device, non_blocking=runtime_device.type == "cuda")
            embeddings = extractor(images).cpu().numpy()
            for image_id, embedding in zip(image_ids, embeddings, strict=True):
                row: dict[str, float | str] = {"image_id": str(image_id)}
                for idx, value in enumerate(embedding):
                    row[f"{backbone}_{idx}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def resolve_runtime_device(device: str | None = None) -> torch.device:
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but the installed PyTorch build cannot access a CUDA device. "
            "Use --device cpu, or install a CUDA-enabled PyTorch build on a machine with a supported GPU."
        )
    LOGGER.info("Using %s for frozen CNN feature extraction.", resolved)
    return resolved


def get_or_create_deep_features(
    task: str,
    backbone: str,
    data_root: Path | str = DEFAULT_ML_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    transform_mode: str = "official",
    include_test: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    output_root = Path(output_dir)
    path = _cache_path(output_root, task, backbone, transform_mode, include_test)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return pd.read_csv(path)

    loaded = load_task(task, data_root)
    runtime_device = resolve_runtime_device(device)

    if include_test:
        train_path = _cache_path(output_root, task, backbone, transform_mode, include_test=False)
        if train_path.exists() and not force:
            train_df = pd.read_csv(train_path)
            test_metadata = loaded.test_metadata[["image_id", "absolute_image_path"]]
            test_df = _extract_deep_features_from_metadata(
                metadata=test_metadata,
                task=task,
                backbone=backbone,
                transform_mode=transform_mode,
                batch_size=batch_size,
                runtime_device=runtime_device,
            )
            df = pd.concat([train_df, test_df], ignore_index=True)
            df.to_csv(path, index=False)
            return df

    frames = [loaded.train_metadata[["image_id", "absolute_image_path"]]]
    if include_test:
        frames.append(loaded.test_metadata[["image_id", "absolute_image_path"]])
    metadata = pd.concat(frames, ignore_index=True)
    df = _extract_deep_features_from_metadata(
        metadata=metadata,
        task=task,
        backbone=backbone,
        transform_mode=transform_mode,
        batch_size=batch_size,
        runtime_device=runtime_device,
    )
    df.to_csv(path, index=False)
    return df
