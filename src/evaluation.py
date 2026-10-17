"""
AeroReady AI - Evaluation Module
Computes regression metrics (RMSE, MAE, R2) and NASA PHM asymmetric scoring.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics and the NASA asymmetric score.

    Args:
        y_true: Array of ground-truth RUL values
        y_pred: Array of predicted RUL values

    Returns:
        Dictionary containing RMSE, MAE, R2, and NASA Score
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # NASA PHM 2008 Asymmetric Scoring Function
    # Late predictions (predicted > actual, d > 0) are penalized much more heavily
    # than early predictions (predicted < actual, d < 0).
    diff = y_pred - y_true
    nasa_score = 0.0
    for d in diff:
        if d < 0:
            nasa_score += np.exp(-d / 13.0) - 1.0
        else:
            nasa_score += np.exp(d / 10.0) - 1.0

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "nasa_score": round(float(nasa_score), 2),
        "sample_count": int(len(y_true)),
    }


def get_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes residual statistics for diagnostic plots.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    return {
        "residuals": residuals.tolist(),
        "mean_error": round(float(np.mean(residuals)), 2),
        "std_error": round(float(np.std(residuals)), 2),
        "median_absolute_error": round(float(np.median(np.abs(residuals))), 2),
        "max_overprediction": round(float(np.max(residuals)), 2),
        "max_underprediction": round(float(np.min(residuals)), 2),
    }
