from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnalysisConfig:
    data_path: str
    target_col: str
    date_col: Optional[str] = None
    model_path: Optional[str] = None
    output_dir: str = "feature_analysis_output"
    sample_rows: Optional[int] = 50000
    time_bucket: str = "M"
    random_state: int = 42
    leakage_corr_threshold: float = 0.9
    leakage_mi_quantile: float = 0.99
    vif_threshold: float = 10.0
    stability_psi_threshold: float = 0.2
    shap_sample_rows: int = 2000
    exclude_columns: Optional[List[str]] = None
    log_level: str = "INFO"
    log_path: Optional[str] = None
    lgbm_params: Optional[dict] = None
