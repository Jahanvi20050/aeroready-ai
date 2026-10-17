"""
AeroReady AI - Training Script
Trains a RandomForestRegressor on NASA C-MAPSS FD001 dataset to predict Remaining Useful Life (RUL).
Saves model, preprocessor, and honest performance metrics to disk.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Add project root and src to sys.path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.data_loader import load_raw_data, check_dataset_exists, get_default_data_dir
    from src.preprocessing import (
        compute_train_rul,
        split_train_val_by_engine,
        CMapssPreprocessor,
        extract_test_last_cycles,
    )
    from src.evaluation import compute_metrics, get_error_distribution
except ImportError:
    from data_loader import load_raw_data, check_dataset_exists, get_default_data_dir
    from preprocessing import (
        compute_train_rul,
        split_train_val_by_engine,
        CMapssPreprocessor,
        extract_test_last_cycles,
    )
    from evaluation import compute_metrics, get_error_distribution


def get_default_model_dir() -> Path:
    """Returns path to the models artifact directory."""
    current_file_dir = Path(__file__).resolve().parent
    candidates = [
        current_file_dir.parent / "models",
        Path.cwd() / "aeroready-ai" / "models",
        Path.cwd() / "models",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    cand = current_file_dir.parent / "models"
    cand.mkdir(parents=True, exist_ok=True)
    return cand


def train_pipeline(
    data_dir: Path,
    model_dir: Path,
    subset: str = "FD001",
    n_estimators: int = 100,
    max_depth: int = 15,
    val_ratio: float = 0.2,
    random_state: int = 42,
    max_rul_clip: float = None,
) -> dict:
    """
    Executes the full training and evaluation workflow.
    """
    print("=" * 60)
    print("AeroReady AI - Model Training Pipeline")
    print("=" * 60)

    exists, msg = check_dataset_exists(data_dir, subset)
    if not exists:
        print(f"[ERROR] {msg}")
        return {"status": "failed", "error": msg}

    print(f"[*] Loading {subset} data from: {data_dir}")
    train_df, test_df, rul_df = load_raw_data(data_dir, subset)
    print(f"    Train raw records: {len(train_df)} across {train_df['engine_id'].nunique()} engines")
    print(f"    Test raw records:  {len(test_df)} across {test_df['engine_id'].nunique()} engines")
    print(f"    Test ground truth: {len(rul_df)} engines")

    # 1. Compute ground-truth RUL target for training
    print("[*] Computing target labels: RUL = max_cycle - current_cycle ...")
    train_with_rul = compute_train_rul(train_df, max_rul_clip=max_rul_clip)

    # 2. Engine-wise split to prevent data leakage
    print(f"[*] Performing engine-wise validation split (val_ratio={val_ratio}) ...")
    train_subset, val_subset, train_ids, val_ids = split_train_val_by_engine(
        train_with_rul, val_ratio=val_ratio, random_state=random_state
    )
    print(f"    Training engines ({len(train_ids)}):   IDs {train_ids[:5]}...{train_ids[-3:]}")
    print(f"    Validation engines ({len(val_ids)}): IDs {val_ids[:5]}...{val_ids[-3:]}")

    # 3. Fit preprocessor strictly on training engines
    print("[*] Fitting CMapssPreprocessor on training engines ...")
    preprocessor = CMapssPreprocessor(rolling_window=5)
    preprocessor.fit(train_subset)
    print(f"    Selected {len(preprocessor.feature_names_)} informative features:")
    print(f"    Active sensors: {preprocessor.active_sensors_}")

    X_train = preprocessor.transform(train_subset)
    y_train = train_subset["RUL"].values

    X_val = preprocessor.transform(val_subset)
    y_val = val_subset["RUL"].values

    # 4. Train RandomForestRegressor
    print(f"[*] Training RandomForestRegressor (n_estimators={n_estimators}, max_depth={max_depth}) ...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_duration = round(time.time() - t0, 2)
    print(f"    Model fitted in {train_duration}s.")

    # 5. Evaluate on Validation Engine Cycles
    print("[*] Evaluating on validation engine cycles ...")
    val_preds = rf.predict(X_val)
    val_metrics = compute_metrics(y_val, val_preds)
    print(f"    Validation RMSE: {val_metrics['rmse']} cycles")
    print(f"    Validation MAE:  {val_metrics['mae']} cycles")
    print(f"    Validation R2:   {val_metrics['r2']}")

    # 6. Evaluate on Official NASA Test Set (last cycle of each test engine)
    print("[*] Evaluating on official NASA Test set ...")
    test_last_snapshots = extract_test_last_cycles(test_df)
    X_test_last = preprocessor.transform(test_last_snapshots)
    y_test_pred = rf.predict(X_test_last)
    y_test_true = rul_df["true_rul"].values

    test_metrics = compute_metrics(y_test_true, y_test_pred)
    error_stats = get_error_distribution(y_test_true, y_test_pred)

    print("-" * 60)
    print(f"  OFFICIAL NASA TEST RESULTS ({subset}):")
    print(f"  * Test RMSE:       {test_metrics['rmse']} cycles")
    print(f"  * Test MAE:        {test_metrics['mae']} cycles")
    print(f"  * Test R2:         {test_metrics['r2']}")
    print(f"  * NASA Score:      {test_metrics['nasa_score']}")
    print(f"  * Mean Error:      {error_stats['mean_error']} cycles")
    print("-" * 60)

    # 7. Extract Feature Importances
    importances = rf.feature_importances_
    feat_names = preprocessor.feature_names_
    feat_importance_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = feat_importance_df.head(10).to_dict(orient="records")

    # 8. Save artifacts to models/
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "rul_rf_model.joblib"
    prep_path = model_dir / "preprocessor.joblib"
    metrics_path = model_dir / "metrics.json"
    preds_path = model_dir / "test_predictions.csv"

    joblib.dump(rf, model_path)
    joblib.dump(preprocessor, prep_path)

    # Save test predictions CSV
    test_pred_df = pd.DataFrame({
        "engine_id": rul_df["engine_id"],
        "actual_rul": y_test_true,
        "predicted_rul": np.round(y_test_pred, 1),
        "error": np.round(y_test_pred - y_test_true, 1),
    })
    test_pred_df.to_csv(preds_path, index=False)

    summary = {
        "status": "success",
        "dataset_subset": subset,
        "model_type": "RandomForestRegressor",
        "parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "val_ratio": val_ratio,
            "random_state": random_state,
            "max_rul_clip": max_rul_clip,
        },
        "training_time_seconds": train_duration,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "error_distribution": error_stats,
        "top_features": top_features,
        "artifacts": {
            "model": str(model_path.name),
            "preprocessor": str(prep_path.name),
            "metrics": str(metrics_path.name),
            "test_predictions": str(preds_path.name),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[*] Artifacts successfully saved to '{model_dir}'")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train AeroReady AI RUL Model")
    parser.add_argument("--subset", type=str, default="FD001", help="C-MAPSS subset (default: FD001)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in RandomForest")
    parser.add_argument("--max-depth", type=int, default=15, help="Maximum tree depth")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation engine split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--model-dir", type=str, default=None, help="Custom model output directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_default_data_dir()
    model_dir = Path(args.model_dir) if args.model_dir else get_default_model_dir()

    train_pipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        subset=args.subset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
