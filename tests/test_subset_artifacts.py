"""Smoke tests for the trained model bundle of every dashboard subset."""
import numpy as np
import pytest

from src.data_loader import get_default_data_dir, load_raw_data
from src.fleet_simulator import FleetSimulator
from src.predict_rul import RULPredictor, get_available_subsets


@pytest.mark.parametrize("subset", get_available_subsets())
def test_subset_artifacts_make_prediction(subset):
    predictor = RULPredictor(subset=subset)
    assert predictor.is_loaded

    _, test_df, _ = load_raw_data(get_default_data_dir(), subset=subset)
    engine_id = int(test_df["engine_id"].min())
    engine_history = test_df[test_df["engine_id"] == engine_id]

    prediction = predictor.predict_dataframe(engine_history)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()

    simulator = FleetSimulator(predictor=predictor, subset=subset)
    fleet = simulator.generate_fleet()
    test_engine_ids = set(test_df["engine_id"].unique())
    assert all(aircraft["assigned_engine_id"] in test_engine_ids for aircraft in fleet)
    assert all(subset in aircraft["data_source"] for aircraft in fleet)
