"""
AeroReady AI - Metrics Module
Computes RMSE, MAE, and the official NASA PHM 2008 Asymmetric Scoring Function.
"""
from typing import Dict, Union, Sequence
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: Union[np.ndarray, Sequence[float]], y_pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Computes Root Mean Squared Error.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(mean_squared_error(y_t, y_p)))


def mae(y_true: Union[np.ndarray, Sequence[float]], y_pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Computes Mean Absolute Error.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    return float(mean_absolute_error(y_t, y_p))


def nasa_score(y_true: Union[np.ndarray, Sequence[float]], y_pred: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Computes the NASA PHM 2008 Asymmetric Scoring Function:
        d = pred - true
        score_i = exp(-d / 13) - 1   if d < 0 (early prediction)
        score_i = exp(d / 10) - 1    if d >= 0 (late prediction)

    Late predictions (predicting longer life than actual) are penalized exponentially
    more severely than early predictions.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    d = y_p - y_t

    # Vectorized computation for high efficiency
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_all_metrics(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    round_digits: int = 4
) -> Dict[str, float]:
    """
    Computes RMSE, MAE, and NASA Score in a single dictionary.
    """
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    s = nasa_score(y_true, y_pred)

    return {
        "rmse": round(r, round_digits),
        "mae": round(m, round_digits),
        "nasa_score": round(s, round_digits),
    }
