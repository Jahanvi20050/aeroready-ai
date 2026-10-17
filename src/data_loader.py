"""
AeroReady AI - Data Loader Module
Loads and validates NASA C-MAPSS Turbofan Engine Degradation Simulation datasets.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

# Standard NASA C-MAPSS Column Definitions
SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
ALL_COLUMNS = ["engine_id", "cycle"] + FEATURE_COLUMNS


def get_default_data_dir() -> Path:
    """
    Locates the data directory flexibly regardless of current working directory.
    """
    # Check relative to this file
    current_file_dir = Path(__file__).resolve().parent
    candidates = [
        current_file_dir.parent / "data",
        Path.cwd() / "aeroready-ai" / "data",
        Path.cwd() / "data",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # Fallback to sibling of src
    return current_file_dir.parent / "data"


def check_dataset_exists(data_dir: Optional[Path] = None, subset: str = "FD001") -> Tuple[bool, str]:
    """
    Checks if all required C-MAPSS dataset files are present.
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
            f"Dataset files missing in '{dir_path}': {', '.join(missing)}.\n"
            f"Please review 'data/README.md' for download instructions."
        )
        return False, msg

    return True, f"Dataset files for {subset} found in '{dir_path}'."


def load_raw_data(data_dir: Optional[Path] = None, subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, test, and true test RUL files for the given C-MAPSS subset.

    Returns:
        train_df: DataFrame with 26 columns (engine_id, cycle, op_settings 1-3, sensor 1-21)
        test_df: DataFrame with the same 26 columns
        rul_df: DataFrame containing the remaining cycles for test engines at test cutoff
    """
    dir_path = Path(data_dir) if data_dir else get_default_data_dir()
    exists, status_msg = check_dataset_exists(dir_path, subset)
    if not exists:
        raise FileNotFoundError(status_msg)

    train_file = dir_path / f"train_{subset}.txt"
    test_file = dir_path / f"test_{subset}.txt"
    rul_file = dir_path / f"RUL_{subset}.txt"

    # NASA dataset files are whitespace-delimited without header
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])

    # Ensure engine_id is 1-indexed and integer
    train_df["engine_id"] = train_df["engine_id"].astype(int)
    test_df["engine_id"] = test_df["engine_id"].astype(int)
    train_df["cycle"] = train_df["cycle"].astype(int)
    test_df["cycle"] = test_df["cycle"].astype(int)
    rul_df["engine_id"] = range(1, len(rul_df) + 1)

    return train_df, test_df, rul_df
