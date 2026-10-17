"""
Unit tests for data preprocessing and RUL target calculation.
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.preprocessing import compute_train_rul, split_train_val_by_engine, CMapssPreprocessor


def test_compute_train_rul_exact():
    """Validates RUL = max_cycle - current_cycle label generation."""
    data = {
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "sensor_1": [500.0, 501.0, 502.0, 499.0, 498.0],
    }
    df = pd.DataFrame(data)
    result = compute_train_rul(df)

    assert "RUL" in result.columns
    # Engine 1 has max_cycle = 3 -> RULs: 3-1=2, 3-2=1, 3-3=0
    # Engine 2 has max_cycle = 2 -> RULs: 2-1=1, 2-2=0
    expected_rul = [2, 1, 0, 1, 0]
    np.testing.assert_array_equal(result["RUL"].values, expected_rul)


def test_split_train_val_by_engine_no_leakage():
    """Ensures engines are strictly partitioned between train and val without overlap."""
    rows = []
    for eng in range(1, 11):  # 10 engines
        for c in range(1, 6):  # 5 cycles each
            rows.append({"engine_id": eng, "cycle": c, "val": eng * 10 + c})
    df = pd.DataFrame(rows)

    train_df, val_df, train_ids, val_ids = split_train_val_by_engine(df, val_ratio=0.3, random_state=42)

    # Intersection of engine IDs must be empty
    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert len(train_ids) + len(val_ids) == 10
    assert len(val_ids) == 3  # 30% of 10

    # Ensure no rows leaked
    assert not any(train_df["engine_id"].isin(val_ids))
    assert not any(val_df["engine_id"].isin(train_ids))


def test_cmapss_preprocessor_variance_filter():
    """Tests that constant sensors are dropped while dynamic sensors are retained."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 1],
        "cycle": [1, 2, 3, 4, 5],
        "op_setting_1": [0.0, 0.0, 0.0, 0.0, 0.0],  # Constant -> should be dropped
        "op_setting_2": [10.0, 20.0, 30.0, 40.0, 50.0],  # Dynamic
        "op_setting_3": [100.0, 100.0, 100.0, 100.0, 100.0],  # Constant -> dropped
        "sensor_1": [518.67, 518.67, 518.67, 518.67, 518.67],  # Constant -> dropped
        "sensor_2": [640.0, 641.0, 642.5, 643.0, 644.0],  # Dynamic -> kept
    })

    prep = CMapssPreprocessor(rolling_window=2)
    prep.fit(df)

    assert "sensor_1" not in prep.active_sensors_
    assert "op_setting_1" not in prep.active_settings_
    assert "sensor_2" in prep.active_sensors_
    assert "op_setting_2" in prep.active_settings_

    transformed = prep.transform(df)
    assert "sensor_2" in transformed.columns
    assert "sensor_2_roll_mean_2" in transformed.columns
    assert transformed.shape[0] == 5
