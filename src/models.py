"""Model registry for core, extended, and Kaggle experiment modes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED


REPORT_NOTES = {
    "dummy": "Majority-class baseline used to calibrate task difficulty.",
    "logistic_regression": "Linear classifier with regularisation.",
    "linear_svc": "Margin-based linear classifier suitable for high-dimensional embeddings.",
    "rbf_svm": "Margin-based classifier; the RBF kernel allows nonlinear decision boundaries.",
    "knn": "Distance-based classifier sensitive to scaling and high dimensionality.",
    "gaussian_nb": "Probabilistic model with strong independence assumptions.",
    "decision_tree": "Interpretable but high variance.",
    "random_forest": "Ensemble trees reduce variance by averaging bootstrapped trees.",
    "extra_trees": "Ensemble trees reduce variance with highly randomised split selection.",
    "mlp": "Shallow neural network trained on frozen embeddings, with early stopping to reduce overfitting.",
    "voting": "Voting ensemble combining complementary linear, margin-based, and tree-based classifiers.",
    "svm_random_forest": "Hard-voting ensemble combining a margin-based SVM with a RandomForest classifier.",
}


def _is_cnn_feature(feature_type: str) -> bool:
    return any(name in feature_type for name in ("resnet50", "efficientnet_b0", "mobilenet_v3_large", "cnn"))


def _allows_pca_search(feature_type: str) -> bool:
    return _is_cnn_feature(feature_type) or feature_type in {
        "provided",
        "provided_engineered",
        "cnn_plus_provided",
    }


def _pipeline(needs_scaling: bool, pca_allowed: bool, classifier: Any) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    if pca_allowed:
        steps.append(("pca", "passthrough"))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _entry(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    requires_scaling: bool,
    pca_allowed: bool,
    report_note: str,
    pca_required: bool = False,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "param_grid": param_grid,
        "requires_scaling": requires_scaling,
        "pca_allowed": pca_allowed,
        "pca_required": pca_required,
        "report_note": report_note,
    }


def _core_entries(feature_type: str) -> dict[str, dict[str, Any]]:
    high_dimensional = _allows_pca_search(feature_type)
    return {
        "dummy": _entry(
            "dummy",
            _pipeline(False, False, DummyClassifier(strategy="most_frequent")),
            {},
            False,
            False,
            REPORT_NOTES["dummy"],
        ),
        "logistic_regression": _entry(
            "logistic_regression",
            _pipeline(True, high_dimensional, LogisticRegression(max_iter=5000, solver="lbfgs")),
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
            high_dimensional,
            REPORT_NOTES["logistic_regression"],
        ),
        "linear_svc": _entry(
            "linear_svc",
            _pipeline(True, high_dimensional, LinearSVC(max_iter=50000, dual="auto")),
            {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
            high_dimensional,
            REPORT_NOTES["linear_svc"],
        ),
        "rbf_svm": _entry(
            "rbf_svm",
            _pipeline(True, high_dimensional, SVC(kernel="rbf")),
            {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": ["scale", 0.001, 0.01, 0.1],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
            high_dimensional,
            REPORT_NOTES["rbf_svm"],
        ),
        "extra_trees": _entry(
            "extra_trees",
            _pipeline(False, False, ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
            {
                "classifier__n_estimators": [300, 500, 800],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
            False,
            REPORT_NOTES["extra_trees"],
        ),
    }


def _extended_entries(feature_type: str) -> dict[str, dict[str, Any]]:
    high_dimensional = _allows_pca_search(feature_type)
    is_cnn = _is_cnn_feature(feature_type)
    return {
        "knn": _entry(
            "knn",
            _pipeline(True, high_dimensional, KNeighborsClassifier()),
            {
                "classifier__n_neighbors": [1, 3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
            True,
            high_dimensional,
            REPORT_NOTES["knn"],
        ),
        "gaussian_nb": _entry(
            "gaussian_nb",
            _pipeline(False, high_dimensional, GaussianNB()),
            {
                "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
            False,
            high_dimensional,
            REPORT_NOTES["gaussian_nb"],
        ),
        "decision_tree": _entry(
            "decision_tree",
            _pipeline(False, False, DecisionTreeClassifier(random_state=RANDOM_SEED)),
            {
                "classifier__max_depth": [None, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 5],
                "classifier__criterion": ["gini", "entropy"],
            },
            False,
            False,
            REPORT_NOTES["decision_tree"],
        ),
        "random_forest": _entry(
            "random_forest",
            _pipeline(False, False, RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
            {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
            False,
            REPORT_NOTES["random_forest"],
        ),
        "mlp": _entry(
            "mlp",
            _pipeline(
                True,
                True,
                MLPClassifier(early_stopping=True, max_iter=300, random_state=RANDOM_SEED),
            ),
            {
                "classifier__hidden_layer_sizes": [(128,), (256,), (256, 64)],
                "classifier__alpha": [1e-4, 1e-3, 1e-2],
                "classifier__learning_rate_init": [1e-3, 3e-4],
            },
            True,
            True,
            REPORT_NOTES["mlp"],
            pca_required=not is_cnn,
        ),
    }


def get_model_registry(mode: str, feature_type: str) -> dict[str, dict[str, Any]]:
    """Return model definitions for a requested experiment mode.

    The registry treats CNN backbones as feature representations rather than
    distinct learning algorithms.
    """
    if mode not in {"core", "extended", "kaggle"}:
        raise ValueError("mode must be one of: core, extended, kaggle")

    if mode == "core":
        return _core_entries(feature_type)
    if mode == "extended":
        entries = _core_entries(feature_type)
        entries.update(_extended_entries(feature_type))
        return entries

    entries: dict[str, dict[str, Any]] = {}
    if _is_cnn_feature(feature_type) and "plus_provided" not in feature_type:
        core = _core_entries(feature_type)
        entries["logistic_regression"] = core["logistic_regression"]
        entries["linear_svc"] = core["linear_svc"]
        entries["rbf_svm"] = core["rbf_svm"]
    if "plus_provided" in feature_type or "cnn_combined" in feature_type or "provided" in feature_type:
        entries["extra_trees"] = _core_entries(feature_type)["extra_trees"]
    if not entries:
        entries = _core_entries(feature_type)
    return entries


def build_safe_param_grid(
    entry: dict[str, Any],
    n_train: int,
    n_features: int,
    quick: bool = False,
) -> list[dict[str, list[Any]]]:
    """Expand a registry entry with fold-safe PCA options."""
    base_grid = deepcopy(entry["param_grid"])
    if quick:
        base_grid = {key: [values[0]] for key, values in base_grid.items()}

    if not entry.get("pca_allowed", False):
        return [base_grid]

    pca_options: list[Any] = []
    if not entry.get("pca_required", False):
        pca_options.append("passthrough")
    for n_components in (50, 100, 200):
        if n_components < n_train and n_components < n_features:
            pca_options.append(PCA(n_components=n_components, random_state=RANDOM_SEED))
    if not pca_options:
        pca_options = ["passthrough"]

    base_grid["pca"] = pca_options
    return [base_grid]


def build_voting_classifier() -> Pipeline:
    """Build an optional soft voting classifier for strong feature sets."""
    classifier = VotingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(C=1.0, class_weight=None, max_iter=5000, solver="lbfgs"),
            ),
            (
                "rbf",
                SVC(C=10, gamma="scale", class_weight=None, kernel="rbf", probability=True),
            ),
            (
                "extra",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight=None,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return _pipeline(True, True, classifier)


def get_additional_kaggle_registry(feature_type: str) -> dict[str, dict[str, Any]]:
    """Extra legal candidates for targeted Kaggle experiments."""
    entries: dict[str, dict[str, Any]] = {}
    if "plus_provided" in feature_type or feature_type in {"cnn_plus_provided", "provided_engineered", "provided"}:
        entries["random_forest"] = _entry(
            "random_forest",
            _pipeline(False, False, RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
            {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 20],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
            False,
            REPORT_NOTES["random_forest"],
        )
        entries["svm_random_forest"] = _entry(
            "svm_random_forest",
            build_svm_random_forest_voting_classifier(),
            {
                "classifier__svm__classifier__C": [0.1, 1, 10],
                "classifier__random_forest__classifier__n_estimators": [300, 500],
                "classifier__random_forest__classifier__max_depth": [None, 20],
                "classifier__random_forest__classifier__min_samples_leaf": [1, 2],
            },
            True,
            False,
            REPORT_NOTES["svm_random_forest"],
        )
    return entries


def build_svm_random_forest_voting_classifier() -> Pipeline:
    """Build a hard-voting ensemble of LinearSVC and RandomForest pipelines."""
    svm_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=1.0, class_weight=None, max_iter=50000, dual="auto")),
        ]
    )
    random_forest_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight=None,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier = VotingClassifier(
        estimators=[
            ("svm", svm_pipeline),
            ("random_forest", random_forest_pipeline),
        ],
        voting="hard",
        n_jobs=-1,
    )
    return Pipeline([("classifier", classifier)])
