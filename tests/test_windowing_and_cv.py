"""
Unit tests for 30-cycle windowing, sequence padding, and GroupKFold engine partitioning.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.preprocessing import (
    pad_engine_sequence,
    extract_windowed_sequences,
    compute_tree_window_features,
    iter_group_kfold_engine_ids,
)


def test_window_shape_normal_engine():
    """
    Tests that an engine with T >= 30 cycles produces:
    - (T - W + 1, W, F) sliding windows
    - (1, W, F) for last observation window
    - (T - W + 1, 5 * F) tabular tree features
    """
    n_cycles = 45
    window_length = 30
    sensor_cols = ["sensor_2", "sensor_3", "sensor_4"]

    df = pd.DataFrame({
        "unit": [1] * n_cycles,
        "cycle": list(range(1, n_cycles + 1)),
        "RUL": list(range(n_cycles - 1, -1, -1)),
        "sensor_2": np.linspace(600, 650, n_cycles),
        "sensor_3": np.linspace(1500, 1600, n_cycles),
        "sensor_4": np.linspace(1400, 1450, n_cycles),
    })

    # 1. Sliding windows
    windows, targets, engine_ids = extract_windowed_sequences(
        df, feature_cols=sensor_cols, window_length=window_length, is_test_last=False, target_col="RUL"
    )
    expected_count = n_cycles - window_length + 1  # 45 - 30 + 1 = 16
    assert windows.shape == (expected_count, window_length, len(sensor_cols))
    assert targets.shape == (expected_count,)
    assert len(engine_ids) == expected_count

    # 2. Last window only (as used for test engines)
    last_window, _, last_ids = extract_windowed_sequences(
        df, feature_cols=sensor_cols, window_length=window_length, is_test_last=True
    )
    assert last_window.shape == (1, window_length, len(sensor_cols))
    assert last_ids == [1]

    # 3. Tree features shape
    tree_features = compute_tree_window_features(windows, feature_names=sensor_cols)
    assert tree_features.shape == (expected_count, 5 * len(sensor_cols))
    assert "sensor_2_last" in tree_features.columns
    assert "sensor_2_mean" in tree_features.columns
    assert "sensor_2_std" in tree_features.columns
    assert "sensor_2_slope" in tree_features.columns
    assert "sensor_2_ewm" in tree_features.columns


def test_window_padding_short_engine():
    """
    Tests that an engine shorter than window_length (e.g. 12 cycles < 30):
    - is padded to exactly window_length by repeating the first row
    - first (30 - 12) = 18 rows are identical to cycle 1
    - remaining 12 rows equal original trajectory
    """
    n_cycles = 12
    window_length = 30
    sensor_cols = ["sensor_2", "sensor_3"]

    df = pd.DataFrame({
        "unit": [5] * n_cycles,
        "cycle": list(range(1, n_cycles + 1)),
        "sensor_2": [640.0 + i * 2.0 for i in range(n_cycles)],
        "sensor_3": [1580.0 + i * 1.5 for i in range(n_cycles)],
    })

    padded_df = pad_engine_sequence(df, window_length=window_length)
    assert len(padded_df) == window_length

    # First 18 rows must equal row 0 of original df
    pad_count = window_length - n_cycles  # 18
    for col in sensor_cols:
        expected_val = df[col].iloc[0]
        actual_padded = padded_df[col].iloc[:pad_count].values
        np.testing.assert_array_equal(actual_padded, [expected_val] * pad_count)

        # Remaining 12 rows must match original df
        np.testing.assert_array_equal(padded_df[col].iloc[pad_count:].values, df[col].values)

    # Test extract_windowed_sequences with is_test_last=True on short engine
    last_window, _, _ = extract_windowed_sequences(
        df, feature_cols=sensor_cols, window_length=window_length, is_test_last=True
    )
    assert last_window.shape == (1, window_length, len(sensor_cols))
    # First 18 time steps must equal original first step
    for j in range(len(sensor_cols)):
        np.testing.assert_array_equal(last_window[0, :pad_count, j], [df[sensor_cols[j]].iloc[0]] * pad_count)


@pytest.mark.parametrize("subset,lengths", [("FD002", [7, 30, 47]), ("FD004", [11, 33, 61])])
def test_variable_length_subset_windows_and_padding(subset, lengths):
    window_length = 30
    sensor_cols = ["sensor_2", "sensor_3"]
    frames = []
    for engine_id, length in enumerate(lengths, start=1):
        frames.append(pd.DataFrame({
            "unit": [engine_id] * length,
            "cycle": np.arange(1, length + 1),
            "sensor_2": np.arange(length, dtype=float) + engine_id * 100,
            "sensor_3": np.arange(length, dtype=float) * 2 + engine_id * 10,
        }))
    frame = pd.concat(frames, ignore_index=True)

    last_windows, _, engine_ids = extract_windowed_sequences(
        frame, feature_cols=sensor_cols, window_length=window_length, is_test_last=True
    )

    assert last_windows.shape == (len(lengths), window_length, len(sensor_cols))
    assert engine_ids == [1, 2, 3]
    for index, length in enumerate(lengths):
        pad_count = max(0, window_length - length)
        for feature_index, column in enumerate(sensor_cols):
            if pad_count:
                np.testing.assert_array_equal(
                    last_windows[index, :pad_count, feature_index],
                    np.repeat(frame.loc[frame["unit"] == index + 1, column].iloc[0], pad_count),
                )


def test_no_engine_leakage_group_kfold():
    """
    Validates that in 5-fold cross-validation across 3 seeds:
    - Train and validation engine ID sets are completely disjoint in every single fold.
    - No data rows from validation engines appear in training folds.
    """
    n_engines = 100
    n_splits = 5
    seeds = [42, 123, 999]
    engine_ids = np.arange(1, n_engines + 1)

    # Synthetic dataset with multiple cycles per engine
    rows = []
    for eng in engine_ids:
        for c in range(1, 35):
            rows.append({"unit": eng, "cycle": c, "val": eng * 100 + c})
    full_df = pd.DataFrame(rows)

    fold_memberships = {}
    for seed in seeds:
        for fold, (train_engine_ids, val_engine_ids) in enumerate(
            iter_group_kfold_engine_ids(engine_ids, n_splits=n_splits, random_state=seed)
        ):
            val_ids = set(val_engine_ids)
            train_ids = set(train_engine_ids)
            fold_memberships[(seed, fold)] = frozenset(val_ids)

            # 1. Disjoint engine sets
            assert len(val_ids) == 20
            assert len(train_ids) == 80
            assert len(train_ids.intersection(val_ids)) == 0, f"Engine leakage detected in seed {seed}, fold {fold}!"

            # 2. Row-level verification
            train_rows = full_df[full_df["unit"].isin(train_ids)]
            val_rows = full_df[full_df["unit"].isin(val_ids)]

            assert not any(train_rows["unit"].isin(val_ids)), "Validation engine leaked into train rows!"
            assert not any(val_rows["unit"].isin(train_ids)), "Train engine leaked into validation rows!"

    assert len({fold_memberships[(seed, 0)] for seed in seeds}) == len(seeds)
