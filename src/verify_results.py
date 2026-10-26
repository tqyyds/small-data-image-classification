"""Recompute archived OOF metrics without course data, checkpoints, or PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .evaluation import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10


def assert_close(actual: float, expected: float, context: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=TOLERANCE):
        raise ValueError(f"{context}: recomputed {actual}, archived {expected}")


def load_predictions(root: Path, entry: dict) -> pd.DataFrame:
    frame = pd.read_csv(root / entry["predictions"])
    prediction_column = entry["prediction_column"]
    required = {"image_id", "true_class_id", prediction_column}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"{entry['id']}: missing prediction fields or values")
    if len(frame) != entry["n_train"] or frame.image_id.duplicated().any():
        raise ValueError(f"{entry['id']}: wrong OOF coverage or duplicated IDs")
    labels = set(map(int, entry["classes"]))
    for column in ("true_class_id", prediction_column):
        if not set(frame[column]).issubset(labels):
            raise ValueError(f"{entry['id']}: invalid class IDs in {column}")
    if set(frame.true_class_id) != labels:
        raise ValueError(f"{entry['id']}: missing true classes")
    if "fold" in frame:
        if set(frame.fold) != set(range(1, entry["cv_folds"] + 1)):
            raise ValueError(f"{entry['id']}: invalid fold coverage")
    if "correct" in frame:
        actual = frame.true_class_id.eq(frame[prediction_column])
        if not np.array_equal(frame.correct.astype(str).str.lower().eq("true"), actual):
            raise ValueError(f"{entry['id']}: incorrect saved correctness flags")
    for column, id_column in (("true_class_name", "true_class_id"),
                              ("predicted_class_name", prediction_column),
                              ("class_name", prediction_column)):
        if column in frame:
            expected = frame[id_column].map(lambda value: entry["classes"][str(value)])
            if not frame[column].equals(expected):
                raise ValueError(f"{entry['id']}: inconsistent class names")
    return frame.rename(columns={prediction_column: "prediction"})


def verify_report(root: Path, entry: dict, frame: pd.DataFrame) -> None:
    labels = sorted(map(int, entry["classes"]))
    names = [entry["classes"][str(label)] for label in labels]
    report = classification_report(frame.true_class_id, frame.prediction,
                                   labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    saved = pd.read_csv(root / entry["report"], index_col=0)
    for name in names + ["macro avg", "weighted avg"]:
        for field in ("precision", "recall", "f1-score", "support"):
            assert_close(report[name][field], float(saved.loc[name, field]),
                         f"{entry['id']} report {name}/{field}")
    assert_close(report["accuracy"], float(saved.loc["accuracy", "precision"]),
                 f"{entry['id']} report accuracy")


def verify(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "results/manifest.json").read_text(encoding="utf-8"))
    for source in manifest["source_files"]:
        digest = hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Historical file changed: {source['path']}")
    table = pd.read_csv(root / "results/historical/model_comparison.csv")
    verified = []
    task_labels = {}
    for entry in manifest["experiments"]:
        frame = load_predictions(root, entry)
        truth = frame.set_index("image_id").true_class_id.sort_index()
        if entry["task"] in task_labels and not truth.equals(task_labels[entry["task"]]):
            raise ValueError(f"{entry['id']}: OOF IDs/labels disagree across methods")
        task_labels[entry["task"]] = truth
        metrics = compute_metrics(frame.true_class_id.to_numpy(), frame.prediction.to_numpy())
        verify_report(root, entry, frame)
        for row_index in entry.get("table_rows", []):
            row = table.iloc[row_index]
            for field in ("task", "model_key", "feature_type", "cv_folds", "n_train"):
                if str(row[field]) != str(entry[field]):
                    raise ValueError(f"{entry['id']}: table configuration mismatch: {field}")
            if str(row.is_smoke).lower() != "false":
                raise ValueError(f"{entry['id']}: synthetic/smoke entry in historical evidence")
            for name, value in metrics.items():
                assert_close(value, float(row[f"oof_{name}"]), f"{entry['id']} table {name}")
        if entry.get("metrics_json"):
            saved = json.loads((root / entry["metrics_json"]).read_text())
            if saved["config"]["cv"] != entry["cv_folds"] or saved["config"]["seed"] != 27:
                raise ValueError(f"{entry['id']}: inconsistent CV configuration")
            for name, value in metrics.items():
                assert_close(value, saved["oof_metrics"][name], f"{entry['id']} JSON {name}")
            folds = pd.read_csv(root / entry["fold_summary"])
            for _, fold in folds.iterrows():
                subset = frame if fold.fold == "aggregate_oof" else frame[frame.fold == int(fold.fold)]
                if len(subset) != int(fold.n_val):
                    raise ValueError(f"{entry['id']}: fold row count mismatch")
                for name, value in compute_metrics(subset.true_class_id, subset.prediction).items():
                    assert_close(value, float(fold[name]), f"{entry['id']} fold {fold.fold}/{name}")
            for name in metrics:
                mean = pd.to_numeric(folds[folds.fold != "aggregate_oof"][name]).mean()
                assert_close(float(mean), saved["mean_fold_metrics"][name], f"{entry['id']} fold mean")
        verified.append({"id": entry["id"], "task": entry["task"],
                         "n_train": len(frame), "cv_folds": entry["cv_folds"], **metrics})
    return {"status": "passed", "historical_files_checked": len(manifest["source_files"]),
            "prediction_sets_checked": len(verified), "experiments": verified,
            "limits": ["Checks saved predictions and reported metrics; does not retrain models.",
                       "Raw course data and independent test labels are not distributed.",
                       "CV model/checkpoint selection limits remain; see docs/methodology.md."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = verify()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {result['prediction_sets_checked']} OOF sets, "
          f"{result['historical_files_checked']} historical file hashes, reports and metrics.")


if __name__ == "__main__":
    main()
