"""
AeroReady AI - Model Benchmark on NASA C-MAPSS FD001
Compares:
1. RandomForest on 30-cycle windowed features
2. LightGBM on 30-cycle windowed features
3. PyTorch Small GRU on raw 30-cycle sequence windows with early stopping

Validation:
GroupKFold by engine ID (5 folds) x 3 seeds. All selectors & scalers fit on training folds only.

Final evaluation:
Last cycle of each test engine vs RUL_FD001.txt.
Outputs results/model_comparison_fd001.csv and prints comparison table.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import yaml

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_raw_data, get_default_data_dir, SENSOR_COLUMNS
from src.regimes import RegimeNormalizer
from src.regimes import RegimeNormalizer
from src.preprocessing import (
    compute_train_rul,
    extract_windowed_sequences,
    compute_tree_window_features,
    iter_group_kfold_engine_ids,
)
from src.metrics import compute_all_metrics, rmse, mae, nasa_score


def load_config() -> Dict[str, Any]:
    cfg_file = _project_root / "config.yaml"
    if cfg_file.is_file():
        with open(cfg_file, "r") as f:
            return yaml.safe_load(f)
    return {
        "dataset": {"subset": "FD001", "data_dir": "data/raw"},
        "preprocessing": {"window_length": 30, "variance_threshold": 1e-5},
        "training": {"rul_cap": 125},
    }


class SmallGRU(nn.Module):
    """Lightweight 1-layer GRU with early stopping for sequence prognostics."""
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_gru_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_epochs: int = 35,
    patience: int = 5,
    batch_size: int = 256,
    lr: float = 0.003,
) -> SmallGRU:
    """Trains SmallGRU on CPU with early stopping on validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_size = X_train.shape[2]
    model = SmallGRU(input_size=input_size, hidden_size=32)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            v_preds = model(val_x)
            v_loss = float(criterion(v_preds, val_y).item())

        if v_loss < best_loss - 1e-4:
            best_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model


def predict_gru(model: SmallGRU, X: np.ndarray) -> np.ndarray:
    """Inference helper for SmallGRU."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(x_tensor).numpy()
    return preds


def split_engines_for_early_stopping(engine_ids, seed: int, validation_ratio: float = 0.15):
    """Create an engine-disjoint fit/early-stopping partition."""
    unique_engines = np.asarray(sorted(set(engine_ids)))
    if len(unique_engines) < 2:
        raise ValueError("At least two engines are required for GRU early stopping.")
    shuffled = np.random.RandomState(seed).permutation(unique_engines)
    validation_count = max(1, int(round(len(unique_engines) * validation_ratio)))
    validation_count = min(validation_count, len(unique_engines) - 1)
    stop_engines = set(shuffled[:validation_count].tolist())
    fit_engines = set(shuffled[validation_count:].tolist())
    if fit_engines & stop_engines:
        raise AssertionError("GRU fit and early-stopping engines must be disjoint.")
    return fit_engines, stop_engines


def run_benchmark():
    config = load_config()
    subset = config.get("dataset", {}).get("subset", "FD001")
    window_length = int(config.get("preprocessing", {}).get("window_length", 30))
    variance_threshold = float(config.get("preprocessing", {}).get("variance_threshold", 1e-5))
    rul_cap = float(config.get("training", {}).get("rul_cap", 125))

    print("=" * 75)
    print(f" AEROREADY AI - MODEL BENCHMARK ON NASA C-MAPSS {subset}")
    print(f" Settings: Window Length = {window_length} cycles | RUL Cap = {rul_cap}")
    print(" Validation: GroupKFold by engine ID (5 folds x 3 seeds = 15 runs)")
    print("=" * 75)

    data_dir = _project_root / config.get("dataset", {}).get("data_dir", "data/raw")
    if not data_dir.exists():
        data_dir = get_default_data_dir()

    train_df, test_df, rul_df = load_raw_data(data_dir=data_dir, subset=subset)
    train_with_rul = compute_train_rul(train_df, max_rul_clip=rul_cap)

    n_splits = 5
    seeds = [42, 123, 999]

    # Metrics storage across folds for each model
    cv_records: Dict[str, List[Dict[str, float]]] = {
        "RandomForest": [],
        "LightGBM": [],
        "PyTorch_GRU": [],
    }

    t0 = time.time()
    print("\n[*] Running 5-fold cross-validation across 3 seeds (15 folds total)...")

    fold_idx = 0
    for seed in seeds:
        for fold, (train_engines, val_engines) in enumerate(
            iter_group_kfold_engine_ids(
                train_df["unit"].unique(), n_splits=n_splits, random_state=seed
            )
        ):
            fold_idx += 1

            # Strict verification of zero data leakage
            assert len(set(train_engines).intersection(set(val_engines))) == 0

            df_tr = train_with_rul[train_with_rul["unit"].isin(train_engines)].copy()
            df_val = train_with_rul[train_with_rul["unit"].isin(val_engines)].copy()

            # 1. Feature selection on training engines ONLY
            active_sensors = [
                col for col in SENSOR_COLUMNS
                if col in df_tr.columns and df_tr[col].std() > variance_threshold
            ]

            # 2. Scaler fitted on training engines ONLY
            scaler = StandardScaler()
            df_tr[active_sensors] = scaler.fit_transform(df_tr[active_sensors])
            df_val[active_sensors] = scaler.transform(df_val[active_sensors])

            # 3. Extract 30-cycle sequence windows
            X_val_seq, y_val, _ = extract_windowed_sequences(
                df_val, feature_cols=active_sensors, window_length=window_length, is_test_last=False, target_col="RUL"
            )
            X_tr_seq, y_tr, _ = extract_windowed_sequences(
                df_tr, feature_cols=active_sensors, window_length=window_length,
                is_test_last=False, target_col="RUL"
            )
            gru_fit_engines, gru_stop_engines = split_engines_for_early_stopping(
                train_engines, seed=seed + fold
            )
            df_gru_fit = df_tr[df_tr["unit"].isin(gru_fit_engines)]
            df_gru_stop = df_tr[df_tr["unit"].isin(gru_stop_engines)]
            X_gru_fit, y_gru_fit, _ = extract_windowed_sequences(
                df_gru_fit, feature_cols=active_sensors, window_length=window_length,
                is_test_last=False, target_col="RUL"
            )
            X_gru_stop, y_gru_stop, _ = extract_windowed_sequences(
                df_gru_stop, feature_cols=active_sensors, window_length=window_length,
                is_test_last=False, target_col="RUL"
            )

            # 4. Extract tree features (last, mean, std, slope, ewm)
            X_tr_tree = compute_tree_window_features(X_tr_seq, feature_names=active_sensors)
            X_val_tree = compute_tree_window_features(X_val_seq, feature_names=active_sensors)

            # --- Model A: RandomForest ---
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_tr_tree, y_tr)
            rf_val_preds = rf.predict(X_val_tree)
            cv_records["RandomForest"].append(compute_all_metrics(y_val, rf_val_preds))

            # --- Model B: LightGBM ---
            lgb = LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            lgb.fit(X_tr_tree, y_tr)
            lgb_val_preds = lgb.predict(X_val_tree)
            cv_records["LightGBM"].append(compute_all_metrics(y_val, lgb_val_preds))

            # --- Model C: PyTorch Small GRU ---
            gru_model = train_gru_with_early_stopping(
                X_gru_fit, y_gru_fit, X_gru_stop, y_gru_stop,
                seed=seed, max_epochs=35, patience=5
            )
            gru_val_preds = predict_gru(gru_model, X_val_seq)
            cv_records["PyTorch_GRU"].append(compute_all_metrics(y_val, gru_val_preds))

            print(f"    Completed Fold {fold_idx}/15 (Seed {seed}, Split {fold + 1})...", end="\r")

    cv_duration = round(time.time() - t0, 1)
    print(f"\n[+] 15-fold cross-validation completed in {cv_duration}s.")

    # =========================================================================
    # FINAL EVALUATION ON OFFICIAL NASA TEST SET (100 ENGINES)
    # =========================================================================
    print("\n[*] Training final models on full FD001 training set (100 engines)...")
    active_sensors_full = [
        col for col in SENSOR_COLUMNS
        if col in train_with_rul.columns and train_with_rul[col].std() > variance_threshold
    ]
    scaler_full = StandardScaler()
    train_full_scaled = train_with_rul.copy()
    test_full_scaled = test_df.copy()

    train_full_scaled[active_sensors_full] = scaler_full.fit_transform(train_full_scaled[active_sensors_full])
    test_full_scaled[active_sensors_full] = scaler_full.transform(test_full_scaled[active_sensors_full])

    # Extract full training tree windows; GRU stopping uses a separate engine holdout.
    X_train_full_seq, y_train_full, _ = extract_windowed_sequences(
        train_full_scaled, feature_cols=active_sensors_full, window_length=window_length,
        is_test_last=False, target_col="RUL"
    )
    X_train_full_tree = compute_tree_window_features(X_train_full_seq, feature_names=active_sensors_full)

    # Extract test windows (last cycle of each of the 100 test engines, padded if < 30 cycles)
    X_test_seq, _, test_engine_ids = extract_windowed_sequences(
        test_full_scaled, feature_cols=active_sensors_full, window_length=window_length, is_test_last=True
    )
    X_test_tree = compute_tree_window_features(X_test_seq, feature_names=active_sensors_full)
    y_test_true = rul_df["true_rul"].values

    assert X_test_seq.shape == (100, window_length, len(active_sensors_full))
    assert len(y_test_true) == 100

    test_results = {}

    # Final RandomForest
    print("    Fitting final RandomForest...")
    final_rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    final_rf.fit(X_train_full_tree, y_train_full)
    rf_test_preds = final_rf.predict(X_test_tree)
    test_results["RandomForest"] = compute_all_metrics(y_test_true, rf_test_preds)

    # Final LightGBM
    print("    Fitting final LightGBM...")
    final_lgb = LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    final_lgb.fit(X_train_full_tree, y_train_full)
    lgb_test_preds = final_lgb.predict(X_test_tree)
    test_results["LightGBM"] = compute_all_metrics(y_test_true, lgb_test_preds)

    # Final PyTorch Small GRU (train with 10% held-out validation for early stopping)
    print("    Fitting final PyTorch Small GRU...")
    fit_engines, stop_engines = split_engines_for_early_stopping(
        train_full_scaled["unit"].unique(), seed=42
    )
    X_gru_fit, y_gru_fit, _ = extract_windowed_sequences(
        train_full_scaled[train_full_scaled["unit"].isin(fit_engines)],
        feature_cols=active_sensors_full, window_length=window_length,
        is_test_last=False, target_col="RUL"
    )
    X_gru_stop, y_gru_stop, _ = extract_windowed_sequences(
        train_full_scaled[train_full_scaled["unit"].isin(stop_engines)],
        feature_cols=active_sensors_full, window_length=window_length,
        is_test_last=False, target_col="RUL"
    )
    final_gru = train_gru_with_early_stopping(
        X_gru_fit,
        y_gru_fit,
        X_gru_stop,
        y_gru_stop,
        seed=42,
        max_epochs=40,
        patience=6,
    )
    gru_test_preds = predict_gru(final_gru, X_test_seq)
    test_results["PyTorch_GRU"] = compute_all_metrics(y_test_true, gru_test_preds)

    # =========================================================================
    # COMPILE COMPARISON TABLE & SAVE TO CSV
    # =========================================================================
    summary_rows = []
    for model_name in ["RandomForest", "LightGBM", "PyTorch_GRU"]:
        rec = cv_records[model_name]
        cv_rmse_vals = [r["rmse"] for r in rec]
        cv_mae_vals = [r["mae"] for r in rec]
        cv_nasa_vals = [r["nasa_score"] for r in rec]

        t_m = test_results[model_name]

        summary_rows.append({
            "model": model_name,
            "cv_rmse_mean": round(float(np.mean(cv_rmse_vals)), 2),
            "cv_rmse_std": round(float(np.std(cv_rmse_vals)), 2),
            "cv_mae_mean": round(float(np.mean(cv_mae_vals)), 2),
            "cv_mae_std": round(float(np.std(cv_mae_vals)), 2),
            "cv_nasa_mean": round(float(np.mean(cv_nasa_vals)), 1),
            "cv_nasa_std": round(float(np.std(cv_nasa_vals)), 1),
            "test_rmse": round(float(t_m["rmse"]), 2),
            "test_mae": round(float(t_m["mae"]), 2),
            "test_nasa": round(float(t_m["nasa_score"]), 1),
        })

    results_df = pd.DataFrame(summary_rows)
    results_dir = _project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "model_comparison_fd001.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved benchmark results to: {out_csv}")

    # Print Clean Console Table
    print("\n" + "=" * 90)
    print(f"{'MODEL COMPARISON RESULTS (NASA C-MAPSS FD001)':^90}")
    print("=" * 90)
    header = (
        f"{'Model':<16} | {'CV RMSE (5-Fold, 3-Seed)':<24} | {'CV MAE':<14} | "
        f"{'Test RMSE':<10} | {'Test MAE':<9} | {'Test NASA Score':<15}"
    )
    print(header)
    print("-" * 90)
    for r in summary_rows:
        cv_rmse_str = f"{r['cv_rmse_mean']:.2f} +/- {r['cv_rmse_std']:.2f}"
        cv_mae_str = f"{r['cv_mae_mean']:.2f} +/- {r['cv_mae_std']:.2f}"
        line = (
            f"{r['model']:<16} | {cv_rmse_str:<24} | {cv_mae_str:<14} | "
            f"{r['test_rmse']:<10.2f} | {r['test_mae']:<9.2f} | {r['test_nasa']:<15.1f}"
        )
        print(line)
    print("=" * 90)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NASA C-MAPSS subsets.")
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--all", action="store_true", help="Run FD001 through FD004.")
    arguments = parser.parse_args()
    if arguments.all and arguments.subset:
        parser.error("Use either --all or --subset, not both.")
    from src.benchmark_subsets import load_config as load_benchmark_config
    from src.benchmark_subsets import run_benchmarks

    config = load_benchmark_config()
    selected = (
        ["FD001", "FD002", "FD003", "FD004"]
        if arguments.all
        else [arguments.subset or config.get("dataset", {}).get("subset", "FD001")]
    )
    run_benchmarks(selected)
