"""
AeroReady AI - Inference Module
Loads trained model artifacts to predict Remaining Useful Life (RUL) and assign health/risk categories.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import yaml

# Add project root and src to sys.path
import sys
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.preprocessing import compute_tree_window_features, extract_windowed_sequences
except ImportError:
    from preprocessing import compute_tree_window_features, extract_windowed_sequences


def _load_config() -> Dict[str, Any]:
    with (_project_root / "config.yaml").open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_default_subset() -> str:
    """Return the dashboard default subset configured for this project."""
    config = _load_config()
    subset = config.get("dataset", {}).get("default_subset")
    if not subset:
        raise ValueError("Set dataset.default_subset in config.yaml.")
    return str(subset)


def get_available_subsets() -> List[str]:
    """Return subsets with model, preprocessor, metrics, and comparison artifacts."""
    models_dir = _project_root / "models"
    results_dir = _project_root / "results"
    benchmark_file = results_dir / "benchmark_all.csv"
    if not benchmark_file.is_file():
        return []
    available = []
    if not models_dir.is_dir():
        return available
    for model_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        subset = model_dir.name
        required_files = (
            model_dir / "randomforest_model.joblib",
            model_dir / "regime_normalizer.joblib",
            model_dir / "metrics.json",
            benchmark_file,
        )
        if all(path.is_file() for path in required_files):
            available.append(subset)
    return available


def get_default_model_paths(subset: str) -> Tuple[Path, Path]:
    """Resolve the selected subset's model and fitted preprocessor."""
    model_dir = _project_root / "models" / subset
    return model_dir / "randomforest_model.joblib", model_dir / "regime_normalizer.joblib"


def categorize_rul(rul: float) -> Tuple[str, str, str]:
    """
    Categorizes predicted RUL into an operational risk bucket.

    Returns:
        (category_name, hex_color, operational_explanation)
    """
    if rul <= 25.0:
        return (
            "Critical",
            "#EF4444",
            "Imminent failure risk (<= 25 cycles). Flight prohibited; immediate maintenance required.",
        )
    elif rul <= 60.0:
        return (
            "Warning",
            "#F59E0B",
            "Moderate degradation (26-60 cycles). Requires scheduled maintenance within next window.",
        )
    else:
        return (
            "Healthy",
            "#10B981",
            "Engine within normal operating envelope (> 60 cycles). Cleared for operational missions.",
        )


class RULPredictor:
    """
    Inference class for Remaining Useful Life predictions.
    """

    def __init__(
        self,
        subset: Optional[str] = None,
        model_path: Optional[Path] = None,
        prep_path: Optional[Path] = None,
    ):
        self.subset = subset or get_default_subset()
        if model_path is None or prep_path is None:
            default_model, default_prep = get_default_model_paths(self.subset)
            self.model_path = Path(model_path) if model_path else default_model
            self.prep_path = Path(prep_path) if prep_path else default_prep
        else:
            self.model_path = Path(model_path)
            self.prep_path = Path(prep_path)

        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> bool:
        """Load the selected subset model, preprocessor, and feature metadata."""
        if not self.model_path.is_file() or not self.prep_path.is_file():
            self.is_loaded = False
            return False
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.prep_path)
        with (self.model_path.parent / "metrics.json").open("r", encoding="utf-8") as metrics_file:
            parameters = json.load(metrics_file)["parameters"]
        self.active_sensors = parameters["active_sensors"]
        self.window_length = int(parameters["window_length"])
        self.is_loaded = True
        return True

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts from per-engine telemetry histories using trained window features.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Model artifacts for {self.subset} were not found. "
                "Run 'python src/train_model.py' to train the model first."
            )
        normalized = self.preprocessor.transform(df)
        windows, _, _ = extract_windowed_sequences(
            normalized,
            feature_cols=self.active_sensors,
            window_length=self.window_length,
            is_test_last=True,
            target_col=None,
        )
        X = compute_tree_window_features(windows, feature_names=self.active_sensors)
        preds = self.model.predict(X)
        return np.maximum(0.0, preds)  # RUL cannot be negative

    def predict_snapshot(self, cycle: int, sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts RUL for an engine given its current cycle and sensor readings.
        """
        row = {"engine_id": 1, "cycle": cycle}
        row.update(sensor_readings)
        single_df = pd.DataFrame([row])
        pred_rul = float(self.predict_dataframe(single_df)[0])
        category, color, explanation = categorize_rul(pred_rul)

        return {
            "predicted_rul": round(pred_rul, 1),
            "category": category,
            "badge_color": color,
            "explanation": explanation,
            "cycle": cycle,
        }
