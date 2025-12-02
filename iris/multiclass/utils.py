from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .dataclasses import ModelResult
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
    _predict_binary,
)

RANDOM_STATE = 42


def select_method(estimator: BaseEstimator) -> str:
    """Выбирает метод получения вероятностей для заданного оценщика."""
    return (
        "decision_function"
        if hasattr(estimator, "decision_function")
        else "predict_proba"
    )


def ecoc_scores(
    estimator: OutputCodeClassifier, X: pd.DataFrame, y: pd.Series, cv
) -> np.ndarray:
    """Вычисляет ECOC-оценки для мультиклассовой классификации."""
    n_classes = len(np.unique(y))
    scores = np.zeros((len(y), n_classes))
    for train_idx, test_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        Y = np.array(
            [_predict_binary(e, X.iloc[test_idx]) for e in est.estimators_], order="F"
        ).T
        distances = pairwise_distances(Y, est.code_book_, metric="euclidean")
        scores[test_idx] = -distances
    return scores


def ecoc_auc_score(
    estimator: OutputCodeClassifier, X: pd.DataFrame, y: pd.Series
) -> float:
    """Вычисляет ROC AUC для ECOC-классификатора."""
    scores = ecoc_scores(
        estimator,
        X,
        y,
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    return roc_auc_score(y, scores, multi_class="ovr", average="macro")


def to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Преобразует оценки в вероятности с помощью softmax, если это необходимо."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    row_sums = scores.sum(axis=1, keepdims=True)
    if np.allclose(row_sums, 1.0):
        return scores
    return softmax(scores, axis=1)


def build_base_estimators() -> Dict[str, Tuple[Pipeline, Dict]]:
    """Создает словарь базовых классификаторов."""
    return {
        "logreg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "estimator__model__C": [0.1, 1.0, 10.0],
                "estimator__model__penalty": ["l2"],
                "estimator__model__solver": ["lbfgs"],
            },
        ),
        "svm": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=False, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "estimator__model__C": [0.1, 1, 10],
                "estimator__model__gamma": ["scale", "auto"],
                "estimator__model__kernel": ["rbf", "linear"],
            },
        ),
        "knn": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "estimator__model__n_neighbors": [3, 5, 7, 9],
                "estimator__model__weights": ["uniform", "distance"],
                "estimator__model__p": [1, 2],
            },
        ),
        "naive_bayes": (
            Pipeline(
                [
                    ("scaler", "passthrough"),
                    ("model", GaussianNB()),
                ]
            ),
            {"estimator__model__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "decision_tree": (
            Pipeline(
                [
                    ("scaler", "passthrough"),
                    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "estimator__model__max_depth": [None, 3, 5, 8],
                "estimator__model__min_samples_leaf": [1, 2, 4],
                "estimator__model__criterion": ["gini", "entropy"],
            },
        ),
    }


def results_to_frame(results: List[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "base_model": r.base_model,
                "strategy": r.strategy,
                "cv_mean_auc": r.cv_mean_auc,
                "cv_predict_auc": r.holdout_auc,
                "train_time_sec": r.train_time_sec,
                "best_params": r.best_params,
            }
            for r in results
        ]
    )


def make_strategy(strategy: str, base_estimator: Pipeline) -> BaseEstimator:
    """Создает мультиклассовый классификатор на основе заданной стратегии."""
    if strategy == "OvR":
        return OneVsRestClassifier(base_estimator)
    if strategy == "OvO":
        return OneVsOneClassifier(base_estimator)
    if strategy == "ECOC":
        return OutputCodeClassifier(
            base_estimator, code_size=1.5, random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown strategy {strategy}")
