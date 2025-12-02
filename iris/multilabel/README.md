# Мультилейбл классификация (MultiOutputClassifier, ClassifierChain)

## Что внутри
`iris_multilabel_analysis.py` — генерация синтетического датасета, EDA, подбор моделей и сравнение стратегий MultiOutput/ClassifierChain.
`figures_multilabel/` — графики EDA: `multilabel_count_hist.png`, `multilabel_label_freq.png`, `multilabel_label_corr.png`, `multilabel_pairplot.png`.
`multilabel_model_results.csv` — таблица с итоговыми метриками и лучшими гиперпараметрами.

## Запуск
Из корня репозитория:
```bash
python3 -m iris.iris_multilabel_analysis
```
После выполнения появятся `multilabel_model_results.csv` и визуализации в `figures_multilabel/`. Возможны предупреждения fontconfig в ограниченных средах — на расчёты не влияют.

## Датасет и EDA
Данные: `make_multilabel_classification` (800 объектов, 15 признаков, 5 меток, в среднем 2 метки на объект).
EDA: распределение числа меток, частоты меток, корреляции меток, pairplot первых 4 признаков с подсветкой числа меток.
-Наблюдения: у объектов чаще 1–3 активных метки, наблюдается умеренный дисбаланс меток
Существуют корреляции между метками, что делает цепочки (ClassifierChain) полезными при решении текущей задачи

## Эксперименты и результаты
Метрика: макро AUC-ROC по 5-fold CV; время — полное время грид-подбора и обучения.

| База + стратегия             | CV AUC | AUC (OOF) | Время, с |
|------------------------------|--------|-----------|----------|
| SVM + ClassifierChain        | 0.9250 | 0.9234    | 3.579    |
| SVM + MultiOutput            | 0.9230 | 0.9213    | 3.573    |
| Logistic Regression + Chain  | 0.8954 | 0.8947    | 0.118    |
| Logistic Regression + MultiOutput | 0.8923 | 0.8910 | 0.101    |
| Naive Bayes + MultiOutput    | 0.8911 | 0.8900    | 0.031    |
| Naive Bayes + ClassifierChain| 0.8904 | 0.8896    | 0.039    |
| KNN + MultiOutput            | 0.8903 | 0.8892    | 0.471    |
| KNN + ClassifierChain        | 0.8898 | 0.8890    | 1.646    |
| Decision Tree + ClassifierChain | 0.8356 | 0.8364 | 0.387    |
| Decision Tree + MultiOutput  | 0.8318 | 0.8323    | 0.359    |

Замечания:
- Для AUC использованы вероятности `predict_proba`; у цепочек вероятности собираются по каждой метке.
- Время включает поиск по сетке базовой модели и 5-fold CV.

## Итоговые выводы
Лучшее качество даёт SVM + ClassifierChain; MultiOutput на SVM практически не уступает.
Самый быстрый и при этом точный компромисс — Logistic Regression + ClassifierChain (~0.12 с, AUC ~0.895).
Наивный Баес почти не уступает логрегрессии по AUC, но работает мгновенно; полезен как лёгкий бенчмарк.
Цепочки выигрывают за счёт учёта корреляций меток, эффект особенно заметен на SVM и логрегрессии.
