"""Feature assembly for provided, engineered, and frozen-CNN features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import DEFAULT_ML_ROOT, DEFAULT_OUTPUT_DIR
from .data import LoadedTask, load_provided_features, load_task

LOGGER = logging.getLogger(__name__)


def _image_color_moments(rgb: np.ndarray) -> dict[str, float]:
    arr = rgb.astype(np.float32) / 255.0
    out: dict[str, float] = {}
    for channel_idx, channel_name in enumerate(("r", "g", "b")):
        channel = arr[:, :, channel_idx]
        out[f"{channel_name}_mean"] = float(np.mean(channel))
        out[f"{channel_name}_std"] = float(np.std(channel))
        centered = channel - out[f"{channel_name}_mean"]
        out[f"{channel_name}_skew"] = float(np.mean(centered**3))
    return out


def _hsv_histogram(image: Image.Image, bins: int = 8) -> dict[str, float]:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    features: dict[str, float] = {}
    for channel_idx, prefix in enumerate(("h", "s", "v")):
        hist, _ = np.histogram(hsv[:, :, channel_idx], bins=bins, range=(0, 255), density=True)
        for bin_idx, value in enumerate(hist):
            features[f"hsv_{prefix}_{bin_idx}"] = float(value)
    return features


def _edge_features(gray: np.ndarray) -> dict[str, float]:
    arr = gray.astype(np.float32) / 255.0
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    return {
        "edge_mean": float(np.mean(mag)),
        "edge_std": float(np.std(mag)),
        "edge_density_025": float(np.mean(mag > 0.25)),
        "edge_density_050": float(np.mean(mag > 0.50)),
    }


def _texture_features(gray: np.ndarray) -> dict[str, float]:
    arr = gray.astype(np.float32) / 255.0
    quantized = np.floor(arr * 8).astype(np.int16)
    right_diff = np.abs(quantized[:, 1:] - quantized[:, :-1])
    down_diff = np.abs(quantized[1:, :] - quantized[:-1, :])
    return {
        "texture_right_mean_abs_diff": float(np.mean(right_diff)),
        "texture_down_mean_abs_diff": float(np.mean(down_diff)),
        "texture_gray_var": float(np.var(arr)),
    }


def compute_engineered_features_for_image(image_path: str | Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    features: dict[str, float] = {}
    features.update(_image_color_moments(rgb))
    features.update(_hsv_histogram(image))
    features.update(_edge_features(gray))
    features.update(_texture_features(gray))
    features["aspect_ratio"] = float(image.width / image.height)
    return features


def get_or_create_engineered_features(
    task: str,
    data_root: Path | str = DEFAULT_ML_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    include_test: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    scope = "all" if include_test else "train"
    output_path = Path(output_dir) / "cached_features" / f"{task}_{scope}_engineered_image_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    loaded = load_task(task, data_root)
    frames = [loaded.train_metadata[["image_id", "absolute_image_path"]]]
    if include_test:
        frames.append(loaded.test_metadata[["image_id", "absolute_image_path"]])
    all_images = pd.concat(frames, ignore_index=True)
    rows = []
    for idx, row in all_images.iterrows():
        if idx % 250 == 0:
            LOGGER.info("Computing engineered image features for %s: %s/%s", task, idx, len(all_images))
        features = compute_engineered_features_for_image(row["absolute_image_path"])
        features["image_id"] = row["image_id"]
        rows.append(features)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def _merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No feature frames were provided.")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="image_id", how="inner", validate="one_to_one")
    return merged


def load_feature_frame(
    task: str,
    feature_type: str,
    data_root: Path | str = DEFAULT_ML_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    transform_mode: str = "official",
    device: str | None = None,
    include_test: bool = True,
    force_engineered: bool = False,
    force_deep: bool = False,
) -> pd.DataFrame:
    """Load a feature frame containing image_id plus numeric feature columns."""
    frames: list[pd.DataFrame] = []
    if feature_type in {"provided", "provided_engineered"} or feature_type.endswith("_plus_provided"):
        frames.append(load_provided_features(task, data_root))

    if feature_type in {"engineered", "provided_engineered"}:
        frames.append(get_or_create_engineered_features(task, data_root, output_dir, include_test, force_engineered))

    backbone = None
    for candidate in ("efficientnet_b0", "resnet50", "mobilenet_v3_large"):
        if feature_type == candidate or feature_type.startswith(f"{candidate}_"):
            backbone = candidate
            break
    if backbone is not None:
        from .deep_features import get_or_create_deep_features

        frames.append(
            get_or_create_deep_features(
                task=task,
                backbone=backbone,
                data_root=data_root,
                output_dir=output_dir,
                transform_mode=transform_mode,
                device=device,
                include_test=include_test,
                force=force_deep,
            )
        )

    if feature_type in {"cnn_combined", "cnn_plus_provided"}:
        from .deep_features import get_or_create_deep_features

        for backbone_name in ("efficientnet_b0", "resnet50", "mobilenet_v3_large"):
            frames.append(
                get_or_create_deep_features(
                    task=task,
                    backbone=backbone_name,
                    data_root=data_root,
                    output_dir=output_dir,
                    transform_mode=transform_mode,
                    device=device,
                    include_test=include_test,
                    force=force_deep,
                )
            )

    if not frames:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    return _merge_feature_frames(frames)


def make_train_test_matrices(
    loaded: LoadedTask,
    features: pd.DataFrame,
    require_test: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if features["image_id"].isna().any() or features["image_id"].duplicated().any():
        raise ValueError("Feature frame must contain unique, non-null image_id values")
    train_ids = loaded.train_metadata["image_id"]
    test_ids = loaded.test_metadata["image_id"]
    label_by_id = loaded.train_metadata.set_index("image_id")["class_id"].astype(int)

    missing_train = sorted(set(train_ids) - set(features["image_id"]))
    missing_test = sorted(set(test_ids) - set(features["image_id"])) if require_test else []
    if missing_train or missing_test:
        raise ValueError(
            f"Feature frame is missing ids. train examples: {missing_train[:5]}, "
            f"test examples: {missing_test[:5]}"
        )

    feature_cols = [col for col in features.columns if col != "image_id"]
    feature_by_id = features.set_index("image_id")
    x_train = feature_by_id.loc[train_ids, feature_cols]
    x_test = feature_by_id.loc[test_ids, feature_cols] if require_test else pd.DataFrame(columns=feature_cols)
    y_train = label_by_id.loc[train_ids]
    return x_train, y_train, x_test, test_ids
