from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StatsConfig:
    data_path: str
    target_col: Optional[str] = "TARGET"
    date_col: Optional[str] = "Date"
    output_dir: str = "feature_stats_output"
    exclude_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    include_target: bool = True
    sample_rows: Optional[int] = None
    random_state: int = 42
    time_bucket: str = "M"
    hist_bins: int = 50
    hist_clip_quantile: float = 0.0
    top_categories: int = 30
    missing_threshold: float = 0.2
    pileup_threshold: float = 0.10
    kurtosis_threshold: float = 10.0
    low_unique_threshold: int = 20
    date_constant_threshold: float = 0.9
    drift_threshold: float = 0.5
    log_level: str = "INFO"
    log_path: Optional[str] = None
