"""Meaningful failure-mode checks for metadata, joins, and result selection."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import load_task, load_provided_features
from src.features import make_train_test_matrices
from src.make_submissions import _select_best_result
from src.synthetic_smoke import make_fixture
from src.verify_results import assert_close, verify


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task_dir = make_fixture(self.root)

    def edit_csv(self, name, edit):
        path = self.task_dir / name
        frame = pd.read_csv(path)
        edit(frame).to_csv(path, index=False)

    def test_duplicate_metadata_ids_rejected(self):
        self.edit_csv("train_metadata.csv", lambda df: pd.concat([df, df.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            load_task("task2", self.root)

    def test_null_metadata_ids_rejected(self):
        self.edit_csv("train_metadata.csv", lambda df: df.assign(image_id=[None] + df.image_id.tolist()[1:]))
        with self.assertRaisesRegex(ValueError, "null"):
            load_task("task2", self.root)

    def test_train_test_overlap_rejected(self):
        self.edit_csv("test_metadata.csv", lambda df: df.assign(image_id=["synthetic_000"] + df.image_id.tolist()[1:]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_task("task2", self.root)

    def test_incorrect_class_mapping_rejected(self):
        pd.DataFrame({"class_id": range(10), "class_name": ["wrong"] * 10}).to_csv(
            self.task_dir / "class_mapping.csv", index=False)
        with self.assertRaisesRegex(ValueError, "class mapping"):
            load_task("task2", self.root)

    def test_missing_image_rejected(self):
        self.edit_csv("train_metadata.csv", lambda df: df.assign(image_path="missing.png"))
        with self.assertRaisesRegex(FileNotFoundError, "missing image"):
            load_task("task2", self.root)

    def test_duplicate_feature_ids_rejected_before_merge(self):
        self.edit_csv("additional_features.csv", lambda df: pd.concat([df, df.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            load_provided_features("task2", self.root)

    def test_feature_merge_aligns_by_id(self):
        loaded = load_task("task2", self.root)
        features = load_provided_features("task2", self.root)
        ordered = make_train_test_matrices(loaded, features)
        shuffled = make_train_test_matrices(loaded, features.sample(frac=1, random_state=4))
        for left, right in zip(ordered[:3], shuffled[:3]):
            self.assertTrue(left.equals(right))
        self.assertEqual(len(ordered[0]), 100)
        self.assertEqual(len(ordered[2]), 10)

    def test_missing_feature_ids_rejected(self):
        loaded = load_task("task2", self.root)
        features = load_provided_features("task2", self.root)
        with self.assertRaisesRegex(ValueError, "missing ids"):
            make_train_test_matrices(loaded, features[features.image_id != "synthetic_000"])

    def test_duplicate_matrix_ids_rejected(self):
        loaded = load_task("task2", self.root)
        features = load_provided_features("task2", self.root)
        with self.assertRaisesRegex(ValueError, "unique"):
            make_train_test_matrices(loaded, pd.concat([features, features.iloc[:1]]))

    def test_smoke_scores_cannot_be_selected_as_best(self):
        (self.root / "results").mkdir()
        pd.DataFrame([{"task": "task2", "is_smoke": True, "mean_cv_macro_f1": 1.0}]).to_csv(
            self.root / "results/model_comparison.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Only smoke"):
            _select_best_result("task2", self.root)

    def test_changed_metric_is_detected(self):
        with self.assertRaisesRegex(ValueError, "recomputed"):
            assert_close(0.8, 0.9, "tampered metric")

    def test_archived_results_are_consistent(self):
        result = verify()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["prediction_sets_checked"], 13)


if __name__ == "__main__":
    unittest.main()
