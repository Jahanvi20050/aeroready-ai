import numpy as np
import pandas as pd
import pytest

from src.uncertainty import (
    compute_interval_metrics,
    finite_sample_quantile,
    order_interval,
    sample_calibration_cut_points,
    split_calibration_engines,
)


def test_interval_metrics_have_known_coverage_width_and_late_risk():
    metrics = compute_interval_metrics(
        y_true=[1, 5, 9, 12],
        lower=[2, 4, 8, 10],
        upper=[4, 8, 10, 14],
    )

    assert metrics == {
        "empirical_coverage": 0.75,
        "mean_interval_width": 3.0,
        "late_risk_fraction": 0.25,
    }


def test_calibration_engines_are_reproducibly_disjoint():
    fitting, calibration = split_calibration_engines(
        range(1, 21),
        calibration_fraction=0.2,
        seed=42,
    )

    assert not set(fitting).intersection(calibration)
    assert set(fitting).union(calibration) == set(range(1, 21))
    assert calibration == split_calibration_engines(
        range(1, 21),
        calibration_fraction=0.2,
        seed=42,
    )[1]


def test_calibration_cut_points_are_seeded_and_use_end_observation_cycles():
    frame = pd.DataFrame({
        "unit": np.repeat([1, 2], 100),
        "cycle": list(range(1, 101)) * 2,
    })

    cuts = sample_calibration_cut_points(
        frame,
        [1, 2],
        minimum_observed_cycles=30,
        seed=7,
    )

    assert cuts == sample_calibration_cut_points(frame, [1, 2], 30, 7)
    assert all(30 <= cycle <= 100 for cycle in cuts.values())


def test_calibration_cut_points_sample_from_supplied_endpoint_distribution():
    frame = pd.DataFrame({
        "unit": np.repeat([1, 2], 100),
        "cycle": list(range(1, 101)) * 2,
    })

    cuts = sample_calibration_cut_points(
        frame,
        [1, 2],
        minimum_observed_cycles=30,
        seed=11,
        endpoint_cycles=[35, 70, 90],
    )

    assert set(cuts.values()).issubset({35, 70, 90})


def test_finite_sample_quantile_uses_corrected_order_statistic():
    assert finite_sample_quantile([1, 2, 3, 4], coverage=0.8) == 4.0


def test_quantile_crossing_is_repaired_into_ordered_nonnegative_intervals():
    lower, median, upper = order_interval(
        lower=[5, -3, 20],
        median=[2, 1, 10],
        upper=[4, 0, 15],
    )

    assert np.all(lower >= 0)
    assert np.all(lower <= median)
    assert np.all(median <= upper)


@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_uncertainty_predictions_exist_for_every_engine_and_method(subset):
    from pathlib import Path

    predictions_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / f"predictions_{subset}.csv"
    )
    assert predictions_path.is_file()
    predictions = pd.read_csv(predictions_path)
    assert set(predictions["method"]) == {
        "LightGBMQuantile",
        "SplitConformalRandomForest",
    }
    assert predictions.groupby("unit")["method"].nunique().eq(2).all()
    assert predictions["unit"].nunique() > 0
    assert (predictions["rul_lower"] <= predictions["rul_median"]).all()
    assert (predictions["rul_median"] <= predictions["rul_upper"]).all()
    assert np.isfinite(
        predictions[["rul_lower", "rul_median", "rul_upper", "true_rul"]]
    ).all().all()
