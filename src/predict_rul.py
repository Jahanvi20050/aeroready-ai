"""
AeroReady AI - Inference Module
Loads trained model artifacts to predict Remaining Useful Life (RUL) and assign health/risk categories.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd

# Add project root and src to sys.path
import sys
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.preprocessing import CMapssPreprocessor, extract_test_last_cycles
    from src.data_loader import load_raw_data, get_default_data_dir
except ImportError:
    from preprocessing import CMapssPreprocessor, extract_test_last_cycles
    from data_loader import load_raw_data, get_default_data_dir


def get_default_model_paths() -> Tuple[Path, Path]:
    """Resolves standard model artifact paths."""
    candidates = [
        _project_root / "models",
        Path.cwd() / "aeroready-ai" / "models",
        Path.cwd() / "models",
    ]
    for d in candidates:
        model_file = d / "rul_rf_model.joblib"
        prep_file = d / "preprocessor.joblib"
        if model_file.is_file() and prep_file.is_file():
            return model_file, prep_file

    # Fallback to project root / models
    default_dir = _project_root / "models"
    return default_dir / "rul_rf_model.joblib", default_dir / "preprocessor.joblib"


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

    def __init__(self, model_path: Optional[Path] = None, prep_path: Optional[Path] = None):
        if model_path is None or prep_path is None:
            default_model, default_prep = get_default_model_paths()
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
        """Attempts to load saved model and preprocessor."""
        if self.model_path.is_file() and self.prep_path.is_file():
            try:
                self.model = joblib.load(self.model_path)
                self.preprocessor = joblib.load(self.prep_path)
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[WARN] Error loading model artifacts: {e}")
                self.is_loaded = False
                return False
        else:
            self.is_loaded = False
            return False

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input dataframe using preprocessor and predicts RUL.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model artifacts not found. Please run 'python src/train_model.py' to train the model first."
            )
        X = self.preprocessor.transform(df)
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
