from __future__ import annotations

import os
import time
from typing import List, Tuple

from .utils import (
    build_base_estimators,
    make_strategy,
    predict_proba_matrix,
    prefix_params,
)

FIG_DIR = "figures_multilabel"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(FIG_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold

from ..multiclass.dataclasses import ModelResult

RANDOM_STATE = 42


class DatasetLoaderMixin:
    def generate_multilabel_dataset(
        self,
        n_samples: int = 800,
        n_features: int = 15,
        n_classes: int = 5,
        n_labels: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Создает синтетический датасет для мультилейбл-классификации."""
        X, Y = make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_labels=n_labels,
            length=50,
            allow_unlabeled=False,
            sparse=False,
            random_state=RANDOM_STATE,
        )
        feature_cols = [f"feat_{i}" for i in range(n_features)]
        label_cols = [f"label_{i}" for i in range(n_classes)]
        X_df = pd.DataFrame(X, columns=feature_cols)
        Y_df = pd.DataFrame(Y, columns=label_cols)
        return X_df, Y_df


class EDARunnerMixin:
    def run_multilabel_eda(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """Генерирует базовые EDA-графики для мультилейбл-датасета."""
        os.makedirs(FIG_DIR, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

        # Распределение количества активных меток на объект.
        label_count = Y.sum(axis=1)
        plt.figure(figsize=(6, 4))
        sns.histplot(
            label_count, bins=range(0, Y.shape[1] + 2), discrete=True, color="#4C72B0"
        )
        plt.title("Распределение числа меток на объект")
        plt.xlabel("Кол-во меток")
        plt.ylabel("Частота")
        plt.tight_layout()
        count_path = os.path.join(FIG_DIR, "multilabel_count_hist.png")
        plt.savefig(count_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Частоты по каждой метке.
        plt.figure(figsize=(6, 4))
        sns.barplot(x=Y.columns, y=Y.sum(axis=0), color="#55A868")
        plt.title("Частоты меток")
        plt.ylabel("Количество положительных примеров")
        plt.tight_layout()
        freq_path = os.path.join(FIG_DIR, "multilabel_label_freq.png")
        plt.savefig(freq_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Корреляции меток.
        plt.figure(figsize=(5, 4))
        sns.heatmap(Y.corr(), annot=True, fmt=".2f", cmap="rocket_r")
        plt.title("Корреляция меток")
        plt.tight_layout()
        corr_path = os.path.join(FIG_DIR, "multilabel_label_corr.png")
        plt.savefig(corr_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Pairplot по первым четырём признакам, закодировано число меток.
        pair_df = X.iloc[:, :4].copy()
        pair_df["label_count"] = label_count
        pairplot_path = os.path.join(FIG_DIR, "multilabel_pairplot.png")
        sns.pairplot(
            pair_df,
            vars=pair_df.columns[:-1],
            hue="label_count",
            diag_kind="hist",
            corner=True,
            palette="viridis",
            plot_kws={"alpha": 0.7, "s": 35},
        )
        plt.suptitle("Признаки (первые 4) и число меток", y=1.02)
        plt.savefig(pairplot_path, dpi=180, bbox_inches="tight")
        plt.close()

        print(
            "EDA visuals saved to:",
            count_path,
            freq_path,
            corr_path,
            pairplot_path,
            sep="\n- ",
        )


def cross_val_proba(estimator, X: np.ndarray, Y: np.ndarray, cv) -> np.ndarray:
    """Возвращает out-of-fold вероятности для оценки ROC AUC."""
    proba = np.zeros_like(Y, dtype=float)
    for train_idx, test_idx in cv.split(X):
        est = clone(estimator)
        est.fit(X[train_idx], Y[train_idx])
        proba[test_idx] = predict_proba_matrix(est, X[test_idx])
    return proba

class Evaluator:
    def evaluate_models(self, X: pd.DataFrame, Y: pd.DataFrame) -> List[ModelResult]:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        results: List[ModelResult] = []
        base_estimators = build_base_estimators()
        strategies = ["MultiOutput", "ClassifierChain"]
        X_arr, Y_arr = X.values, Y.values

        for base_name, (base_pipe, grid) in base_estimators.items():
            for strategy in strategies:
                estimator = make_strategy(strategy, clone(base_pipe))
                prefix = "estimator__"
                param_grid = prefix_params(grid, prefix)

                def scorer(est, X_val, y_val):
                    probs = predict_proba_matrix(est, X_val)
                    return roc_auc_score(y_val, probs, average="macro")

                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=cv,
                    n_jobs=None,
                )

                start = time.perf_counter()
                search.fit(X_arr, Y_arr)
                train_time = time.perf_counter() - start
                best_estimator = search.best_estimator_

                oof_probs = cross_val_proba(best_estimator, X_arr, Y_arr, cv)
                holdout_auc = roc_auc_score(Y_arr, oof_probs, average="macro")

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
                    f"AUC (OOF): {holdout_auc:.4f} | train_time: {train_time:.3f}s"
                )

        return results


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

class EntryPoint(DatasetLoaderMixin, EDARunnerMixin, Evaluator):
    """Класс-энтрипоинт для запуска всех этапов анализа датасета ирисов."""
    def main(self):
        X, Y = self.generate_multilabel_dataset()
        self.run_multilabel_eda(X, Y)
        results = self.evaluate_models(X, Y)
        df_results = results_to_frame(results).sort_values(
            by="cv_predict_auc", ascending=False
        )
        out_csv = "multilabel_model_results.csv"
        df_results.to_csv(out_csv, index=False)
        print("\nTop models by cross-val ROC-AUC (macro):")
        print(
            df_results[
                [
                    "base_model",
                    "strategy",
                    "cv_mean_auc",
                    "cv_predict_auc",
                    "train_time_sec",
                ]
            ]
        )
        print(f"\nДетальные результаты сохранены в {out_csv}")


if __name__ == "__main__":
    ep = EntryPoint()
    ep.main()
