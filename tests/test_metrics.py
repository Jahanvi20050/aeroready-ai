"""
Unit tests for evaluation metrics, including known-value tests for the NASA asymmetric score.
"""
import pytest
import numpy as np

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.metrics import rmse, mae, nasa_score, compute_all_metrics


def test_nasa_score_perfect_predictions():
    """When predictions exactly match ground truth, NASA score must be 0.0."""
    y_true = [50.0, 100.0, 150.0]
    y_pred = [50.0, 100.0, 150.0]
    score = nasa_score(y_true, y_pred)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_nasa_score_known_values():
    """
    Known-value analytical test:
    - Sample 1: d = pred - true = 87 - 100 = -13 (early)
      penalty = exp(-(-13)/13) - 1 = exp(1) - 1 = e - 1
    - Sample 2: d = pred - true = 110 - 100 = +10 (late)
      penalty = exp(10/10) - 1 = exp(1) - 1 = e - 1
    Total score = 2 * (e - 1) ≈ 3.4365636569...
    """
    y_true = [100.0, 100.0]
    y_pred = [87.0, 110.0]
    expected_score = 2.0 * (np.e - 1.0)

    actual_score = nasa_score(y_true, y_pred)
    assert actual_score == pytest.approx(expected_score, rel=1e-6)


def test_nasa_score_asymmetry_penalty():
    """
    Verifies the asymmetry property:
    Late predictions (predicting longer life than actual) are penalized more
    severely than early predictions of the same absolute error magnitude.
    """
    y_true = [100.0]
    # Prediction early by 15 cycles: d = -15
    score_early = nasa_score(y_true, [85.0])
    # Prediction late by 15 cycles: d = +15
    score_late = nasa_score(y_true, [115.0])

    expected_early = np.exp(15.0 / 13.0) - 1.0  # ≈ 2.170
    expected_late = np.exp(15.0 / 10.0) - 1.0   # ≈ 3.481

    assert score_early == pytest.approx(expected_early, rel=1e-6)
    assert score_late == pytest.approx(expected_late, rel=1e-6)
    assert score_late > score_early


def test_rmse_and_mae_known_values():
    """Tests standard regression metrics against known hand-calculated values."""
    y_true = [10.0, 20.0]
    y_pred = [14.0, 17.0]  # errors: +4, -3

    expected_mae = 3.5  # (4 + 3) / 2
    expected_rmse = np.sqrt(12.5)  # sqrt((16 + 9) / 2) ≈ 3.5355339

    assert mae(y_true, y_pred) == pytest.approx(expected_mae, rel=1e-6)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse, rel=1e-6)


def test_compute_all_metrics_dict():
    """Tests compute_all_metrics bundle function."""
    y_true = np.array([50.0, 80.0])
    y_pred = np.array([50.0, 80.0])
    res = compute_all_metrics(y_true, y_pred)

    assert "rmse" in res
    assert "mae" in res
    assert "nasa_score" in res
    assert res["rmse"] == 0.0
    assert res["mae"] == 0.0
    assert res["nasa_score"] == 0.0
