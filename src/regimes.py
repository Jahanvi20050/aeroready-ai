"""Operating-regime assignment and regime-local sensor normalization."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_loader import SENSOR_COLUMNS, SETTING_COLUMNS


class RegimeNormalizer:
    """Cluster operating settings and standardize sensors within each cluster."""

    def __init__(
        self,
        n_clusters: int = 1,
        random_state: int = 42,
        setting_columns: List[str] = None,
        sensor_columns: List[str] = None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.setting_columns = setting_columns or SETTING_COLUMNS
        self.sensor_columns = sensor_columns or SENSOR_COLUMNS
        self.clusterer_ = None
        self.scalers_: Dict[int, StandardScaler] = {}
        self.cluster_sizes_: Dict[int, int] = {}

    def fit(self, frame: pd.DataFrame):
        columns = self.setting_columns + self.sensor_columns
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing regime normalization columns: {missing}")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        self.clusterer_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.clusterer_.fit_predict(frame[self.setting_columns])
        self.scalers_ = {}
        self.cluster_sizes_ = {}
        for label in range(self.n_clusters):
            mask = labels == label
            if not np.any(mask):
                continue
            scaler = StandardScaler()
            scaler.fit(frame.loc[mask, self.sensor_columns])
            self.scalers_[label] = scaler
            self.cluster_sizes_[label] = int(mask.sum())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.clusterer_ is None:
            raise RuntimeError("RegimeNormalizer has not been fitted.")
        columns = self.setting_columns + self.sensor_columns
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing regime normalization columns: {missing}")

        labels = self.clusterer_.predict(frame[self.setting_columns])
        transformed = frame.copy()
        normalized_sensors = frame[self.sensor_columns].to_numpy(dtype=float, copy=True)
        for label, scaler in self.scalers_.items():
            mask = labels == label
            if np.any(mask):
                normalized_sensors[mask] = scaler.transform(frame.loc[mask, self.sensor_columns])
        transformed[self.sensor_columns] = normalized_sensors
        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
