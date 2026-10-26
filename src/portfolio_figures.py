"""Regenerate public figures from verified saved OOF predictions, without images."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .verify_results import ROOT, load_predictions, verify

BLUE, GOLD, GREY = "#225d8a", "#ba791c", "#d8e0e7"


def main() -> None:
    verified = verify()
    metrics = {row["id"]: row for row in verified["experiments"]}
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    entries = {entry["id"]: entry for entry in manifest["experiments"]}
    output = ROOT / "results/figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#253746", "axes.labelcolor": "#253746"})
    selections = {
        "task1": ["task1_core_provided_official_logistic_regression",
                  "task1_core_provided_engineered_official_logistic_regression",
                  "task1_core_efficientnet_b0_official_logistic_regression",
                  "task1_resnet50_transfer_tta"],
        "task2": ["task2_core_provided_official_logistic_regression",
                  "task2_core_provided_engineered_official_logistic_regression",
                  "task2_kaggle_cnn_plus_provided_official_random_forest",
                  "task2_resnet50_transfer_tta"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for ax, task, title in zip(axes, selections,
                               ["Animals · 3,750 images · 3-fold", "Birds · 417 images · 5-fold"]):
        ids = selections[task]
        labels = ["Provided features + LR", "Provided + engineered + LR",
                  "EfficientNet-B0 + LR" if task == "task1" else "CNN + provided + RF",
                  "ResNet50 fine-tuning + TTA"]
        scores = [metrics[key]["macro_f1"] for key in ids]
        ax.barh(range(4), scores, color=[GREY, GREY, BLUE, GOLD], height=.6)
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Macro-F1 (0–1)")
        ax.set_title(title, loc="left", fontsize=13, pad=14)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        for y, score in enumerate(scores):
            ax.text(score + .014, y, f"{score:.4f}", va="center", fontsize=10)
    fig.suptitle("Historical out-of-fold results", fontsize=19, weight="bold", x=.02, ha="left")
    fig.supxlabel("LR = Logistic Regression · RF = RandomForest · TTA = horizontal-flip test-time augmentation\n"
                  "Model/checkpoint selection used validation data; these are not independent test scores.",
                  fontsize=9, ha="center")
    fig.savefig(output / "model_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)

    for task, experiment_id in (("task1", selections["task1"][2]), ("task2", selections["task2"][2])):
        entry = entries[experiment_id]
        frame = load_predictions(ROOT, entry)
        labels = sorted(map(int, entry["classes"]))
        names = [entry["classes"][str(label)].replace("_", " ") for label in labels]
        matrix = confusion_matrix(frame.true_class_id, frame.prediction, labels=labels)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(10.8, 9), layout="constrained")
        heat = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(10), names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(10), names)
        for row in range(10):
            for column in range(10):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                        color="white" if normalized[row, column] > .55 else "#253746", fontsize=10)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.colorbar(heat, ax=ax, shrink=.7, label="Fraction of true-class observations")
        title = "Animals · EfficientNet-B0 + Logistic Regression" if task == "task1" else "Birds · CNN + provided features + RandomForest"
        ax.set_title(title + f"\nHistorical OOF · n={len(frame):,} · cell labels are counts", loc="left", pad=14)
        fig.savefig(output / f"{task}_confusion.png", dpi=160, facecolor="white")
        plt.close(fig)

    bird_ids = [selections["task2"][2], selections["task2"][3]]
    class_reports = []
    per_class_rows = []
    for experiment_id in bird_ids:
        entry = entries[experiment_id]
        frame = load_predictions(ROOT, entry)
        report = classification_report(frame.true_class_id, frame.prediction, output_dict=True, zero_division=0)
        class_reports.append(report)
        for label, name in entry["classes"].items():
            per_class_rows.append(dict(experiment=experiment_id, class_id=int(label), class_name=name,
                                       **report[label]))
    pd.DataFrame(per_class_rows).to_csv(ROOT / "results/bird_per_class.csv", index=False)
    names = [entries[bird_ids[0]]["classes"][str(i)].replace("_", " ") for i in range(10)]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.subplots_adjust(left=.29, right=.95, top=.83, bottom=.16)
    y = np.arange(10)
    for offset, report, color, label in zip([-.18, .18], class_reports, [BLUE, GOLD],
                                            ["Frozen features + RandomForest", "ResNet50 fine-tuning + TTA"]):
        scores = [report[str(i)]["f1-score"] for i in range(10)]
        ax.barh(y + offset, scores, height=.32, color=color, label=label)
        for position, score in zip(y + offset, scores):
            ax.text(score + .01, position, f"{score:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("Per-class F1 (0–1)")
    fig.suptitle("Bird classification · per-class historical OOF F1", x=.025, y=.98,
                 ha="left", fontsize=16)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), frameon=False, ncol=1, fontsize=10)
    fig.supxlabel("417 labelled images · 39–42 observations per class · 5-fold validation\n"
                  "Different selection protocols; small score differences are descriptive, not significance claims.", fontsize=9)
    fig.savefig(output / "bird_per_class.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("PASS: four source-backed PNG figures and bird per-class table regenerated.")


if __name__ == "__main__":
    main()
