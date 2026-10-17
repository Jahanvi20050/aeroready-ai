"""
AeroReady AI - Data Loader Module
Loads and validates NASA C-MAPSS Turbofan Engine Degradation Simulation datasets
from data/raw/ for any subset (FD001 - FD004).
"""
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd

# Standard NASA C-MAPSS Column Definitions
SETTING_COLUMNS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS: List[str] = SETTING_COLUMNS + SENSOR_COLUMNS
RAW_COLUMNS: List[str] = ["unit", "cycle"] + FEATURE_COLUMNS
ALL_COLUMNS: List[str] = RAW_COLUMNS


def get_default_data_dir() -> Path:
    """
    Locates the raw data directory flexibly regardless of current working directory.
    Prioritizes data/raw/ location.
    """
    current_file_dir = Path(__file__).resolve().parent
    candidates = [
        current_file_dir.parent / "data" / "raw",
        Path.cwd() / "aeroready-ai" / "data" / "raw",
        Path.cwd() / "data" / "raw",
        current_file_dir.parent / "data",
        Path.cwd() / "aeroready-ai" / "data",
        Path.cwd() / "data",
    ]
    for cand in candidates:
        if cand.exists() and any(cand.glob("train_*.txt")):
            return cand
    # Fallback to standard data/raw
    return current_file_dir.parent / "data" / "raw"


def check_dataset_exists(data_dir: Optional[Path] = None, subset: str = "FD001") -> Tuple[bool, str]:
    """
    Checks if all required C-MAPSS dataset files for the given subset exist.
    Supports FD001, FD002, FD003, FD004.
    """
    dir_path = Path(data_dir) if data_dir else get_default_data_dir()
    train_file = dir_path / f"train_{subset}.txt"
    test_file = dir_path / f"test_{subset}.txt"
    rul_file = dir_path / f"RUL_{subset}.txt"

    missing = []
    for f in [train_file, test_file, rul_file]:
        if not f.is_file():
            missing.append(f.name)

    if missing:
        msg = (
            f"Dataset files missing in '{dir_path}' for subset '{subset}': {', '.join(missing)}.\n"
            f"Please verify raw data files in data/raw/."
        )
        return False, msg

    return True, f"Dataset files for {subset} found in '{dir_path}'."


def load_raw_data(data_dir: Optional[Path] = None, subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, test, and ground-truth test RUL files for the given C-MAPSS subset.
    Columns: unit, cycle, 3 op settings, 21 sensors.

    Returns:
        train_df: DataFrame with columns: unit, cycle, op_setting_1..3, sensor_1..21 (and engine_id alias)
        test_df: DataFrame with matching schema
        rul_df: DataFrame with true_rul, unit, and engine_id
    """
    dir_path = Path(data_dir) if data_dir else get_default_data_dir()
    exists, status_msg = check_dataset_exists(dir_path, subset)
    if not exists:
        raise FileNotFoundError(status_msg)

    train_file = dir_path / f"train_{subset}.txt"
    test_file = dir_path / f"test_{subset}.txt"
    rul_file = dir_path / f"RUL_{subset}.txt"

    # NASA dataset files are whitespace-delimited without headers
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=RAW_COLUMNS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=RAW_COLUMNS)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])

    # Cast integer identifiers
    train_df["unit"] = train_df["unit"].astype(int)
    test_df["unit"] = test_df["unit"].astype(int)
    train_df["cycle"] = train_df["cycle"].astype(int)
    test_df["cycle"] = test_df["cycle"].astype(int)

    # Provide engine_id alias for full backward compatibility
    train_df["engine_id"] = train_df["unit"]
    test_df["engine_id"] = test_df["unit"]

    rul_df["unit"] = range(1, len(rul_df) + 1)
    rul_df["engine_id"] = rul_df["unit"]

    return train_df, test_df, rul_df
