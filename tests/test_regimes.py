import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.regimes import RegimeNormalizer


def _regime_frame(settings, sensors):
    frame = pd.DataFrame(settings, columns=["op_setting_1", "op_setting_2", "op_setting_3"])
    frame["sensor_2"] = np.asarray(sensors, dtype=float).reshape(-1)
    return frame


def test_single_regime_matches_standard_scaler():
    train = _regime_frame(np.zeros((6, 3)), [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    expected = StandardScaler().fit_transform(train[["sensor_2"]])

    actual = RegimeNormalizer(sensor_columns=["sensor_2"]).fit_transform(train)

    np.testing.assert_allclose(actual["sensor_2"], expected[:, 0])


def test_transform_does_not_change_training_regime_statistics():
    train = _regime_frame(
        [[0.0, 0.0, 0.0]] * 4 + [[10.0, 0.0, 0.0]] * 4,
        [[1.0], [2.0], [3.0], [4.0], [101.0], [102.0], [103.0], [104.0]],
    )
    normalizer = RegimeNormalizer(n_clusters=2, sensor_columns=["sensor_2"]).fit(train)
    before = {
        label: (scaler.mean_.copy(), scaler.scale_.copy())
        for label, scaler in normalizer.scalers_.items()
    }
    unseen_test = _regime_frame([[20.0, 0.0, 0.0]], [[10000.0]])

    assert len(normalizer.transform(unseen_test)) == 1
    for label, scaler in normalizer.scalers_.items():
        np.testing.assert_array_equal(scaler.mean_, before[label][0])
        np.testing.assert_array_equal(scaler.scale_, before[label][1])


def test_zero_variance_regime_is_finite():
    train = _regime_frame(np.zeros((4, 3)), [[5.0], [5.0], [5.0], [5.0]])
    normalized = RegimeNormalizer(sensor_columns=["sensor_2"]).fit_transform(train)

    assert np.isfinite(normalized["sensor_2"]).all()
    np.testing.assert_array_equal(normalized["sensor_2"], np.zeros(4))
