from __future__ import annotations

import os
from .utils import (
    build_base_estimators,
    make_strategy,
    results_to_frame,
    select_method,
    to_probabilities,
    ecoc_scores,
)

FIG_DIR = "figures"
# Ensure matplotlib cache is writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(FIG_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
from .dataclasses import ModelResult
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.multiclass import (
    _predict_binary,
)


RANDOM_STATE = 42


class DatasetLoaderMixin:
    """Класс-миксин для загрузки и предобработки набора данных ирисов."""

    def load_iris_df(self) -> Tuple[pd.DataFrame, pd.Series]:
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df.rename(
            columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
            },
            inplace=True,
        )
        df["target_name"] = df["target"].map(dict(enumerate(data.target_names)))
        X = df.drop(columns=["target", "target_name"])
        y = df["target"]
        return X, y


class EDARunnerMixin:
    """Класс для проведения разведочного анализа данных (EDA) на наборе ирисов."""

    def run_eda(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Выполняет EDA и сохраняет графики в каталог figures/."""
        os.makedirs(FIG_DIR, exist_ok=True)
        df = X.copy()
        df["target"] = y
        df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

        sns.set_theme(style="whitegrid", context="notebook")

        pairplot_path = os.path.join(FIG_DIR, "pairplot.png")
        sns.pairplot(
            df,
            vars=X.columns,
            hue="species",
            diag_kind="hist",
            corner=True,
            plot_kws={"alpha": 0.8, "s": 45},
        )
        plt.suptitle("Iris Feature Pairplot", y=1.02)
        plt.savefig(pairplot_path, dpi=200, bbox_inches="tight")
        plt.close()

        box_path = os.path.join(FIG_DIR, "boxplots.png")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col in zip(axes.flat, X.columns):
            sns.boxplot(data=df, x="species", y=col, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel(col)
        fig.suptitle("Feature Distributions by Species")
        plt.tight_layout()
        plt.savefig(box_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        corr_path = os.path.join(FIG_DIR, "correlation.png")
        corr = df[X.columns].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="crest", fmt=".2f")
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(corr_path, dpi=200, bbox_inches="tight")
        plt.close()

        scatter_path = os.path.join(FIG_DIR, "petal_scatter.png")
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", s=70)
        plt.title("Petal Length vs Width by Species")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(
            "EDA visuals saved to:",
            pairplot_path,
            box_path,
            corr_path,
            scatter_path,
            sep="\n- ",
        )


class Evaluator:
    """Класс для оценки моделей с разными стратегиями мультиклассовой классификации."""

    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> List[ModelResult]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        results: List[ModelResult] = []
        base_estimators = build_base_estimators()
        strategies = ["OvR", "OvO", "ECOC"]

        for base_name, (base_pipe, grid) in base_estimators.items():
            for strategy in strategies:
                estimator = make_strategy(strategy, clone(base_pipe))
                param_grid = grid

                def standard_auc_scorer(est, X_val, y_val):
                    method = select_method(est)
                    scores = getattr(est, method)(X_val)
                    probas_local = to_probabilities(scores)
                    return roc_auc_score(
                        y_val, probas_local, multi_class="ovr", average="macro"
                    )

                scoring = standard_auc_scorer
                if strategy == "ECOC":

                    def scorer(est, X_val, y_val):
                        Y_val = np.array(
                            [_predict_binary(e, X_val) for e in est.estimators_],
                            order="F",
                        ).T
                        distances = pairwise_distances(
                            Y_val, est.code_book_, metric="euclidean"
                        )
                        scores = to_probabilities(-distances)
                        return roc_auc_score(
                            y_val, scores, multi_class="ovr", average="macro"
                        )

                    scoring = scorer

                search = GridSearchCV(
                    estimator,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=None,
                )

                start = time.perf_counter()
                search.fit(X, y)
                train_time = time.perf_counter() - start
                best_estimator = search.best_estimator_

                if strategy == "ECOC":
                    scores = ecoc_scores(best_estimator, X, y, cv)
                else:
                    method = select_method(best_estimator)
                    scores = cross_val_predict(
                        best_estimator,
                        X,
                        y,
                        cv=cv,
                        method=method,
                    )
                probas = to_probabilities(scores)
                holdout_auc = roc_auc_score(
                    y, probas, multi_class="ovr", average="macro"
                )

                results.append(
                    ModelResult(
                        base_model=base_name,
                        strategy=strategy,
                        best_params=search.best_params_,
                        cv_mean_auc=search.best_score_,
                        holdout_auc=holdout_auc,
                        train_time_sec=train_time,
                    )
                )
                print(
                    f"[{base_name} | {strategy}] AUC (CV best): {search.best_score_:.4f} | "
                    f"AUC (CV predict): {holdout_auc:.4f} | train_time: {train_time:.3f}s"
                )
        return results


class EntryPoint(DatasetLoaderMixin, EDARunnerMixin, Evaluator):
    """Класс-энтрипоинт для запуска всех этапов анализа датасета ирисов."""

    def main(self):
        X, y = self.load_iris_df()
        self.run_eda(X, y)
        results = self.evaluate_models(X, y)
        df_results = results_to_frame(results).sort_values(
            by="cv_predict_auc", ascending=False
        )
        out_csv = "model_results.csv"
        df_results.to_csv(out_csv, index=False)
        print("\nTop models by cross-val ROC-AUC:")
        print(
            df_results[
                [
                    "base_model",
                    "strategy",
                    "cv_mean_auc",
                    "cv_predict_auc",
                    "train_time_sec",
                ]
            ].head(10)
        )


if __name__ == "__main__":
    ep = EntryPoint()
    ep.main()
