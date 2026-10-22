from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ShapConfig:
    data_path: str
    target_col: str
    date_col: Optional[str] = None
    model_path: Optional[str] = None
    output_dir: str = "shap_output"
    sample_rows: Optional[int] = 200000
    shap_sample_rows: int = 10000
    random_state: int = 42
    exclude_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    lgbm_params: Optional[dict] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_date_range: Optional[List[str]] = None
    val_date_range: Optional[List[str]] = None
    test_date_range: Optional[List[str]] = None
    gap_days: int = 0
    shap_class_index: Optional[int] = None
    interaction_sample_rows: int = 2000
    top_features: int = 20
    dependence_plots: int = 6
    low_importance_quantile: float = 0.1
    low_importance_threshold: Optional[float] = None
    force_index: int = 0
    top_interactions: int = 30
    log_level: str = "INFO"
    log_path: Optional[str] = None
