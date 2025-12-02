import numpy as np
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from typing import Dict, Tuple


RANDOM_STATE = 42

def build_base_estimators() -> Dict[str, Tuple[Pipeline, Dict]]:
    """Создает базовые классификаторы и сетки гиперпараметров."""
    return {
        "logreg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=400, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "model__C": [0.5, 1.0, 2.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        ),
        "svm": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__C": [0.5, 1.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf", "linear"],
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
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "naive_bayes": (
            Pipeline(
                [
                    ("scaler", "passthrough"),
                    ("model", GaussianNB()),
                ]
            ),
            {"model__var_smoothing": [1e-9, 1e-8]},
        ),
        "decision_tree": (
            Pipeline(
                [
                    ("scaler", "passthrough"),
                    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__max_depth": [None, 5, 8], "model__min_samples_leaf": [1, 2, 4]},
        ),
    }


def make_strategy(strategy: str, base_estimator: Pipeline):
    """Оборачивает базовый алгоритм в стратегию мультилейбл-классификации."""
    if strategy == "MultiOutput":
        return MultiOutputClassifier(base_estimator)
    if strategy == "ClassifierChain":
        return ClassifierChain(
            estimator=base_estimator, order="random", random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown strategy {strategy}")


def prefix_params(grid: Dict, prefix: str) -> Dict:
    """Добавляет префикс параметрам для грид-сёрча с обёртками."""
    return {f"{prefix}{k}": v for k, v in grid.items()}


def predict_proba_matrix(estimator, X) -> np.ndarray:
    """Возвращает матрицу вероятностей положительного класса для каждой метки."""
    proba = estimator.predict_proba(X)
    if isinstance(proba, list):
        return np.column_stack([p[:, 1] for p in proba])
    proba = np.asarray(proba)
    if proba.ndim == 3 and proba.shape[2] == 2:
        return proba[:, :, 1]
    if proba.ndim == 2:
        return proba
    raise ValueError("Неизвестный формат predict_proba для мультилейбл модели.")
