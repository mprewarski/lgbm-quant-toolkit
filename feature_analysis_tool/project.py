from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .utils import dump_json, ensure_dir, load_json


def save_project(output_dir: str, results: Dict[str, Any], model: Optional[Any]) -> Dict[str, str]:
    ensure_dir(output_dir)
    paths = {}

    results_path = os.path.join(output_dir, "results.json")
    dump_json(results_path, results)
    paths["results"] = results_path

    if model is not None and hasattr(model, "save_model"):
        model_path = os.path.join(output_dir, "model.txt")
        model.save_model(model_path)
        paths["model"] = model_path

    return paths


def load_project(output_dir: str) -> Dict[str, Any]:
    results_path = os.path.join(output_dir, "results.json")
    return load_json(results_path)
