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
from sklearn.model_selection import GroupKFold

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


def iter_group_kfold_engine_ids(engine_ids, n_splits: int = 5, random_state: int = 42):
    """Yield seeded, disjoint engine ID folds using GroupKFold."""
    unique_engines = np.asarray(sorted(set(engine_ids)))
    if len(unique_engines) < n_splits:
        raise ValueError(
            f"GroupKFold with {n_splits} splits requires at least {n_splits} engines; "
            f"got {len(unique_engines)}."
        )

    shuffled_group_labels = np.random.RandomState(random_state).permutation(len(unique_engines))
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(unique_engines, groups=shuffled_group_labels):
        yield unique_engines[train_idx].tolist(), unique_engines[val_idx].tolist()


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


def pad_engine_sequence(engine_df: pd.DataFrame, window_length: int = 30) -> pd.DataFrame:
    """
    If engine sequence length is shorter than window_length, pad by repeating the first row
    at the beginning until length equals window_length.
    """
    length = len(engine_df)
    if length >= window_length:
        return engine_df
    pad_count = window_length - length
    first_row = engine_df.iloc[[0]]
    pad_df = pd.concat([first_row] * pad_count, ignore_index=True)
    return pd.concat([pad_df, engine_df], ignore_index=True)


def extract_windowed_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_length: int = 30,
    is_test_last: bool = False,
    target_col: Optional[str] = "RUL",
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Extracts sliding or last-observation windows of shape (N, window_length, len(feature_cols)).

    If is_test_last is True:
        For each engine, extracts exactly one window of length window_length ending at its last cycle.
        If engine has fewer than window_length cycles, pads by repeating the first row.
        Returns:
            windows: np.ndarray of shape (num_engines, window_length, num_features)
            targets: None or np.ndarray
            engine_ids: list of engine IDs in order

    If is_test_last is False (training/validation):
        Slides a window of length window_length across each engine trajectory.
        For each engine:
            Pads if shorter than window_length.
            Takes slices [t - window_length : t] for t in range(window_length, len(engine) + 1).
        Returns:
            windows: np.ndarray of shape (N, window_length, num_features)
            targets: np.ndarray of shape (N,) with corresponding target values
            engine_ids: list of engine IDs corresponding to each window
    """
    unit_col = _get_unit_col(df)
    unique_units = sorted(df[unit_col].unique())

    windows_list = []
    targets_list = []
    engine_ids = []

    for unit_id in unique_units:
        eng_df = df[df[unit_col] == unit_id].sort_values("cycle").reset_index(drop=True)
        eng_df = pad_engine_sequence(eng_df, window_length=window_length)
        eng_vals = eng_df[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(eng_vals)

        if is_test_last:
            # Last window only
            window = eng_vals[-window_length:]
            windows_list.append(window)
            engine_ids.append(unit_id)
            if target_col and target_col in eng_df.columns:
                targets_list.append(eng_df[target_col].iloc[-1])
        else:
            # Sliding windows
            target_vals = eng_df[target_col].to_numpy(dtype=np.float32) if (target_col and target_col in eng_df.columns) else None
            for t in range(window_length, n_rows + 1):
                window = eng_vals[t - window_length : t]
                windows_list.append(window)
                engine_ids.append(unit_id)
                if target_vals is not None:
                    targets_list.append(target_vals[t - 1])

    windows = np.array(windows_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32) if targets_list else None
    return windows, targets, engine_ids


def compute_tree_window_features(
    windows: np.ndarray,
    feature_names: List[str],
    span: Optional[float] = None,
) -> pd.DataFrame:
    """
    Transforms 3D windows of shape (N, W, F) into 2D tabular features for tree-based models:
    For each sensor feature:
    - {feature}_last: last value in the window
    - {feature}_mean: arithmetic mean over the window
    - {feature}_std: standard deviation over the window
    - {feature}_slope: linear regression slope over the window
    - {feature}_ewm: exponential weighted moving average over the window

    Returns a DataFrame with N rows and 5 * F columns.
    """
    N, W, F = windows.shape
    if F != len(feature_names):
        raise ValueError(f"Feature count mismatch: windows has {F} features, but {len(feature_names)} names provided.")

    span_val = float(span) if span is not None else float(W)
    alpha = 2.0 / (span_val + 1.0)

    # 1. Slope weights: denom = sum((tau - bar(tau))^2)
    x_diff = np.arange(W, dtype=np.float64) - (W - 1) / 2.0
    denom = np.sum(x_diff ** 2)

    # 2. EWM weights: w_tau = (1 - alpha)^(W - 1 - tau), normalized
    w = (1.0 - alpha) ** np.arange(W - 1, -1, -1, dtype=np.float64)
    norm_w = w / np.sum(w)

    # Vectorized computation across all N windows and F features
    last_vals = windows[:, -1, :]                                    # (N, F)
    mean_vals = np.mean(windows, axis=1)                              # (N, F)
    std_vals = np.std(windows, axis=1)                                # (N, F)
    slope_vals = np.einsum('w, nwf -> nf', x_diff, windows) / denom   # (N, F)
    ewm_vals = np.einsum('w, nwf -> nf', norm_w, windows)           # (N, F)

    # Construct DataFrame with clear column names
    col_names = []
    feature_arrays = []
    for j, fname in enumerate(feature_names):
        col_names.extend([
            f"{fname}_last",
            f"{fname}_mean",
            f"{fname}_std",
            f"{fname}_slope",
            f"{fname}_ewm",
        ])
        feature_arrays.extend([
            last_vals[:, j],
            mean_vals[:, j],
            std_vals[:, j],
            slope_vals[:, j],
            ewm_vals[:, j],
        ])

    df_tree = pd.DataFrame(np.column_stack(feature_arrays), columns=col_names)
    return df_tree
