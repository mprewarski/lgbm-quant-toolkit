from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OptunaConfig:
    data_path: str
    target_col: str
    output_dir: str = "optuna_output"
    date_col: Optional[str] = None
    exclude_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    n_trials: int = 50
    timeout: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42
    metric: Optional[str] = None
    direction: Optional[str] = None
    study_name: str = "lgbm_optuna"
    storage_path: Optional[str] = "optuna_output/study.db"
    log_level: str = "INFO"
    log_path: Optional[str] = "optuna_output/run.log"
