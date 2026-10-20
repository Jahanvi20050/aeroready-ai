"""
AeroReady AI - Task 1 Execution Script: Baseline and RUL Capping
1. Loads raw NASA dataset from data/raw/ (FD001).
2. Reproduces existing baseline: RandomForestRegressor on current features with linear RUL.
   Evaluated on the LAST cycle of each test engine vs RUL_FD001.txt.
   Saves to results/baseline_fd001.json.
3. Applies RUL capping at 125 (read from config.yaml) for training labels only.
   Re-runs the model with the exact same architecture.
   Saves to results/capped_fd001.json.
4. Prints a before/after comparison table (RMSE, MAE, NASA Score).
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Setup sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_raw_data, get_default_data_dir
from src.preprocessing import (
    compute_train_rul,
    split_train_val_by_engine,
    CMapssPreprocessor,
    extract_test_last_cycles,
)
from src.metrics import compute_all_metrics


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    cfg_file = config_path or (_project_root / "config.yaml")
    if not cfg_file.is_file():
        # Fallback default configuration
        return {
            "dataset": {"default_subset": "FD001", "data_dir": "data/raw"},
            "preprocessing": {"rolling_window": 5, "variance_threshold": 1e-5},
            "model": {
                "type": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
            "training": {"val_ratio": 0.2, "random_state": 42, "rul_cap": 125},
        }
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def run_experiment(rul_cap: Optional[float], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains RandomForestRegressor and evaluates against official NASA test set.
    """
    subset = config.get("dataset", {}).get("default_subset", "FD001")
    raw_dir = _project_root / config.get("dataset", {}).get("data_dir", "data/raw")
    if not raw_dir.exists():
        raw_dir = get_default_data_dir()

    # Load raw data with columns: unit, cycle, 3 op settings, 21 sensors
    train_df, test_df, rul_df = load_raw_data(data_dir=raw_dir, subset=subset)

    # 1. Compute training RUL (linear if rul_cap is None, capped if rul_cap is set)
    train_with_rul = compute_train_rul(train_df, max_rul_clip=rul_cap)

    # 2. Engine-wise split to prevent data leakage
    val_ratio = config.get("training", {}).get("val_ratio", 0.2)
    seed = config.get("training", {}).get("random_state", 42)
    train_sub, val_sub, train_ids, val_ids = split_train_val_by_engine(
        train_with_rul, val_ratio=val_ratio, random_state=seed
    )

    # 3. Fit preprocessor on training partition only
    roll_win = config.get("preprocessing", {}).get("rolling_window", 5)
    var_thresh = float(config.get("preprocessing", {}).get("variance_threshold", 1e-5))
    preprocessor = CMapssPreprocessor(rolling_window=roll_win, variance_threshold=var_thresh)
    preprocessor.fit(train_sub)

    X_train = preprocessor.transform(train_sub)
    y_train = train_sub["RUL"].values

    # 4. Train RandomForestRegressor
    m_cfg = config.get("model", {})
    rf = RandomForestRegressor(
        n_estimators=m_cfg.get("n_estimators", 100),
        max_depth=m_cfg.get("max_depth", 15),
        min_samples_split=m_cfg.get("min_samples_split", 5),
        min_samples_leaf=m_cfg.get("min_samples_leaf", 2),
        random_state=m_cfg.get("random_state", 42),
        n_jobs=m_cfg.get("n_jobs", -1),
    )
    rf.fit(X_train, y_train)

    # 5. Evaluate on LAST cycle of each test engine vs RUL_FD001.txt
    test_last_snapshots = extract_test_last_cycles(test_df)
    X_test_last = preprocessor.transform(test_last_snapshots)
    y_test_pred = rf.predict(X_test_last)
    y_test_true = rul_df["true_rul"].values  # Uncapped ground truth

    # Compute metrics using src/metrics.py
    eval_metrics = compute_all_metrics(y_test_true, y_test_pred)

    unit_col = "unit" if "unit" in test_last_snapshots.columns else "engine_id"
    predictions_record = [
        {
            "unit": int(test_last_snapshots[unit_col].iloc[i]),
            "true_rul": float(y_test_true[i]),
            "predicted_rul": round(float(y_test_pred[i]), 2),
            "error": round(float(y_test_pred[i] - y_test_true[i]), 2),
        }
        for i in range(len(y_test_true))
    ]

    return {
        "dataset_subset": subset,
        "model_type": m_cfg.get("type", "RandomForestRegressor"),
        "rul_cap_training": rul_cap,
        "metrics": eval_metrics,
        "hyperparameters": m_cfg,
        "feature_count": len(preprocessor.feature_names_),
        "active_sensors": preprocessor.active_sensors_,
        "training_engines_count": len(train_ids),
        "validation_engines_count": len(val_ids),
        "test_engines_count": len(y_test_true),
        "predictions": predictions_record,
    }


def main():
    config = load_config()
    results_dir = _project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" TASK 1: BASELINE vs RUL CAPPING (NASA C-MAPSS FD001)")
    print("=" * 65)

    # Step 3: Reproduce existing baseline (linear RUL)
    print("\n[*] Running Experiment 1: Baseline (Linear RUL, cap=None)...")
    baseline_result = run_experiment(rul_cap=None, config=config)
    baseline_path = results_dir / "baseline_fd001.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_result, f, indent=2)
    print(f"    Saved baseline results to: {baseline_path}")

    # Step 4: Run Capped RUL model (cap from config.yaml, default 125)
    cap_value = config.get("training", {}).get("rul_cap", 125)
    print(f"\n[*] Running Experiment 2: Capped RUL (cap={cap_value} on training labels only)...")
    capped_result = run_experiment(rul_cap=cap_value, config=config)
    capped_path = results_dir / "capped_fd001.json"
    with open(capped_path, "w") as f:
        json.dump(capped_result, f, indent=2)
    print(f"    Saved capped results to: {capped_path}")

    # Step 5: Print Before / After Table
    b_m = baseline_result["metrics"]
    c_m = capped_result["metrics"]

    rmse_change = c_m["rmse"] - b_m["rmse"]
    mae_change = c_m["mae"] - b_m["mae"]
    nasa_change = c_m["nasa_score"] - b_m["nasa_score"]

    rmse_pct = (rmse_change / b_m["rmse"]) * 100.0
    mae_pct = (mae_change / b_m["mae"]) * 100.0
    nasa_pct = (nasa_change / b_m["nasa_score"]) * 100.0

    print("\n" + "=" * 65)
    print(f"{'Metric':<18} | {'Baseline (Linear)':<18} | {'Capped (RUL=125)':<18} | {'Delta (%)':<10}")
    print("-" * 65)
    print(f"{'RMSE (cycles)':<18} | {b_m['rmse']:<18.4f} | {c_m['rmse']:<18.4f} | {rmse_pct:+.2f}%")
    print(f"{'MAE (cycles)':<18} | {b_m['mae']:<18.4f} | {c_m['mae']:<18.4f} | {mae_pct:+.2f}%")
    print(f"{'NASA Score':<18} | {b_m['nasa_score']:<18.2f} | {c_m['nasa_score']:<18.2f} | {nasa_pct:+.2f}%")
    print("=" * 65)

    return baseline_result, capped_result


if __name__ == "__main__":
    main()
