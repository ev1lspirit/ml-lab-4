from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelResult:
    base_model: str
    strategy: str
    best_params: Dict
    cv_mean_auc: float
    holdout_auc: float
    train_time_sec: float
