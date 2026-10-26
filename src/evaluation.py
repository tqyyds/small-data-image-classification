"""Evaluation helpers for COMP30027 Project 2 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def serialise_param_value(value: Any) -> Any:
    if value == "passthrough" or value is None:
        return value
    if hasattr(value, "n_components"):
        return {"type": value.__class__.__name__, "n_components": int(value.n_components)}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def serialise_params(params: dict[str, Any]) -> str:
    clean = {key: serialise_param_value(value) for key, value in params.items()}
    return json.dumps(clean, sort_keys=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = class_mapping["class_id"].astype(int).tolist()
    target_names = class_mapping["class_name"].astype(str).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_path)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    labels = class_mapping["class_id"].astype(int).tolist()
    names = class_mapping["class_name"].astype(str).tolist()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig_width = max(8, 0.55 * len(names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
