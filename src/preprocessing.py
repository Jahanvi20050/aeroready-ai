"""
AeroReady AI - Preprocessing and Feature Engineering Module
Handles RUL target generation, engine-wise validation splitting,
and consistent feature extraction for training and inference.
Supports both 'unit' and 'engine_id' column notations.
"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .data_loader import SENSOR_COLUMNS, SETTING_COLUMNS
except (ImportError, ValueError):
    from data_loader import SENSOR_COLUMNS, SETTING_COLUMNS


def _get_unit_col(df: pd.DataFrame) -> str:
    """Helper to detect unit/engine identifier column."""
    if "unit" in df.columns:
        return "unit"
    elif "engine_id" in df.columns:
        return "engine_id"
    return "unit"


def compute_train_rul(df: pd.DataFrame, max_rul_clip: Optional[float] = None) -> pd.DataFrame:
    """
    Computes the Remaining Useful Life (RUL) for training engines.
    Formula: RUL = maximum cycle for that engine - current cycle.

    If max_rul_clip is provided (e.g. 125), values above the clip threshold
    are capped for training labels only (standard practice in prognostics to reflect healthy plateau phase).
    """
    df = df.copy()
    unit_col = _get_unit_col(df)
    max_cycles = df.groupby(unit_col)["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]

    if max_rul_clip is not None:
        df["RUL"] = df["RUL"].clip(upper=float(max_rul_clip))

    return df


def split_train_val_by_engine(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Splits data by engine ID/unit to prevent data leakage across flight cycles.
    
    Returns:
        train_df, val_df, train_engine_ids, val_engine_ids
    """
    unit_col = _get_unit_col(df)
    unique_engines = np.sort(df[unit_col].unique())
    rng = np.random.RandomState(random_state)
    shuffled_engines = rng.permutation(unique_engines)
    
    val_size = int(len(unique_engines) * val_ratio)
    val_engine_ids = sorted(shuffled_engines[:val_size].tolist())
    train_engine_ids = sorted(shuffled_engines[val_size:].tolist())

    train_df = df[df[unit_col].isin(train_engine_ids)].copy()
    val_df = df[df[unit_col].isin(val_engine_ids)].copy()

    return train_df, val_df, train_engine_ids, val_engine_ids


class CMapssPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that:
    1. Identifies and removes constant/zero-variance sensor columns during fit.
    2. Computes rolling mean and standard deviation features per engine.
    3. Handles initial boundary conditions (backfills within engine).
    4. Guarantees consistent feature columns at inference time.
    """

    def __init__(self, rolling_window: int = 5, variance_threshold: float = 1e-5):
        self.rolling_window = rolling_window
        self.variance_threshold = variance_threshold
        self.active_sensors_: List[str] = []
        self.active_settings_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns which sensors and settings have non-zero variance.
        """
        # Detect active settings
        self.active_settings_ = [
            col for col in SETTING_COLUMNS
            if col in X.columns and X[col].std() > self.variance_threshold
        ]

        # Detect active sensors
        self.active_sensors_ = [
            col for col in SENSOR_COLUMNS
            if col in X.columns and X[col].std() > self.variance_threshold
        ]

        # Pre-calculate feature names
        feature_names = ["cycle"] + self.active_settings_ + self.active_sensors_
        if self.rolling_window > 1:
            for col in self.active_sensors_:
                feature_names.append(f"{col}_roll_mean_{self.rolling_window}")
                feature_names.append(f"{col}_roll_std_{self.rolling_window}")

        self.feature_names_ = feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame into the model feature space.
        """
        if not self.feature_names_:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        df = X.copy()
        unit_col = _get_unit_col(df)
        features = pd.DataFrame(index=df.index)
        features["cycle"] = df["cycle"]

        for col in self.active_settings_:
            features[col] = df[col] if col in df.columns else 0.0

        for col in self.active_sensors_:
            features[col] = df[col] if col in df.columns else 0.0

        # Compute rolling statistics grouped by unit if multiple cycles exist
        if self.rolling_window > 1 and unit_col in df.columns and len(df) > 1:
            grouped = df.groupby(unit_col)
            for col in self.active_sensors_:
                roll = grouped[col].rolling(window=self.rolling_window, min_periods=1)
                roll_mean = roll.mean().reset_index(level=0, drop=True)
                roll_std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
                features[f"{col}_roll_mean_{self.rolling_window}"] = roll_mean
                features[f"{col}_roll_std_{self.rolling_window}"] = roll_std
        elif self.rolling_window > 1:
            # Fallback for single snapshot inference without engine history
            for col in self.active_sensors_:
                features[f"{col}_roll_mean_{self.rolling_window}"] = features[col]
                features[f"{col}_roll_std_{self.rolling_window}"] = 0.0

        # Fill any remaining NaNs safely
        features = features.fillna(0.0)

        # Enforce exact column order learned during fit
        return features[self.feature_names_]


def extract_test_last_cycles(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the last recorded cycle for each engine in the test set.
    According to the NASA C-MAPSS competition, the ground truth RUL vector
    corresponds to the remaining life after this exact last observation.
    """
    unit_col = _get_unit_col(test_df)
    last_indices = test_df.groupby(unit_col)["cycle"].idxmax()
    return test_df.loc[last_indices].sort_values(unit_col).reset_index(drop=True)
