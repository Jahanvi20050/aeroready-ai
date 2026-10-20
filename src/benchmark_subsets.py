"""Leakage-safe RandomForest and LightGBM benchmarks for FD001-FD004."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import SENSOR_COLUMNS, get_default_data_dir, load_raw_data
from src.metrics import compute_all_metrics
from src.preprocessing import (
    compute_train_rul,
    compute_tree_window_features,
    extract_windowed_sequences,
    iter_group_kfold_engine_ids,
)
from src.regimes import RegimeNormalizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> Dict[str, Any]:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def run_subset(subset: str, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    preprocessing = config.get("preprocessing", {})
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    window_length = int(preprocessing.get("window_length", 30))
    variance_threshold = float(preprocessing.get("variance_threshold", 1e-5))
    rul_cap = float(training.get("rul_cap", 125))
    seeds = [int(seed) for seed in training.get("seeds", [42, 123, 999])]
    cluster_map = preprocessing.get("regime_clusters", {})
    n_clusters = int(cluster_map.get(subset, 1 if subset in {"FD001", "FD003"} else 6))
    data_dir = PROJECT_ROOT / dataset.get("data_dir", "data/raw")
    if not data_dir.exists():
        data_dir = get_default_data_dir()

    train_df, test_df, rul_df = load_raw_data(data_dir=data_dir, subset=subset)
    train_labeled = compute_train_rul(train_df, max_rul_clip=rul_cap)
    cv_metrics = {"RandomForest": [], "LightGBM": []}
    started = time.perf_counter()
    total_folds = len(seeds) * 5
    current_fold = 0
    print(f"\n{subset}: {n_clusters} regimes; five engine-grouped folds x {len(seeds)} seeds")

    for seed in seeds:
        for fold, (train_ids, validation_ids) in enumerate(
            iter_group_kfold_engine_ids(train_df["unit"].unique(), n_splits=5, random_state=seed),
            start=1,
        ):
            if set(train_ids).intersection(validation_ids):
                raise AssertionError(f"Engine overlap in {subset}, seed {seed}, fold {fold}.")
            fold_train = train_labeled[train_labeled["unit"].isin(train_ids)].copy()
            fold_validation = train_labeled[train_labeled["unit"].isin(validation_ids)].copy()

            normalizer = RegimeNormalizer(n_clusters=n_clusters, random_state=seed)
            fold_train = normalizer.fit_transform(fold_train)
            fold_validation = normalizer.transform(fold_validation)
            sensors = [
                column for column in SENSOR_COLUMNS
                if fold_train[column].std() > variance_threshold
            ]
            x_train, y_train, _ = extract_windowed_sequences(
                fold_train, sensors, window_length=window_length,
                is_test_last=False, target_col="RUL",
            )
            x_validation, y_validation, _ = extract_windowed_sequences(
                fold_validation, sensors, window_length=window_length,
                is_test_last=False, target_col="RUL",
            )
            x_train = compute_tree_window_features(x_train, sensors)
            x_validation = compute_tree_window_features(x_validation, sensors)

            models = (
                ("RandomForest", RandomForestRegressor(
                    n_estimators=100, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, random_state=seed, n_jobs=-1,
                )),
                ("LightGBM", LGBMRegressor(
                    n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    n_jobs=-1, verbose=-1,
                )),
            )
            for name, model in models:
                model.fit(x_train, y_train)
                cv_metrics[name].append(compute_all_metrics(y_validation, model.predict(x_validation)))

            current_fold += 1
            print(f"  fold {current_fold}/{total_folds} complete", end="\r")

    print(f"\n  CV completed in {time.perf_counter() - started:.1f}s")

    final_normalizer = RegimeNormalizer(n_clusters=n_clusters, random_state=seeds[0])
    normalized_train = final_normalizer.fit_transform(train_labeled)
    normalized_test = final_normalizer.transform(test_df)
    active_sensors = [
        column for column in SENSOR_COLUMNS
        if normalized_train[column].std() > variance_threshold
    ]
    dropped_sensors = [column for column in SENSOR_COLUMNS if column not in active_sensors]
    print(f"  Dropped sensors: {dropped_sensors}")

    x_train_seq, y_train, _ = extract_windowed_sequences(
        normalized_train, active_sensors, window_length=window_length,
        is_test_last=False, target_col="RUL",
    )
    x_test_seq, _, test_engine_ids = extract_windowed_sequences(
        normalized_test, active_sensors, window_length=window_length,
        is_test_last=True, target_col=None,
    )
    expected_test_engine_ids = rul_df["unit"].tolist()
    if test_engine_ids != expected_test_engine_ids:
        raise ValueError(
            f"{subset} test-window engine IDs do not match the ordered RUL file: "
            f"{test_engine_ids[:5]} != {expected_test_engine_ids[:5]}"
        )
    x_train = compute_tree_window_features(x_train_seq, active_sensors)
    x_test = compute_tree_window_features(x_test_seq, active_sensors)
    y_test = rul_df["true_rul"].to_numpy()

    models = (
        ("RandomForest", RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=seeds[0], n_jobs=-1,
        )),
        ("LightGBM", LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seeds[0],
            n_jobs=-1, verbose=-1,
        )),
    )
    rows = []
    fitted_models = {}
    for name, model in models:
        model.fit(x_train, y_train)
        test_predictions = model.predict(x_test)
        test_metrics = compute_all_metrics(y_test, test_predictions)
        fold_rmse = [record["rmse"] for record in cv_metrics[name]]
        rows.append({
            "subset": subset,
            "model": name,
            "cv_rmse_mean": round(float(np.mean(fold_rmse)), 2),
            "cv_rmse_std": round(float(np.std(fold_rmse)), 2),
            "test_rmse": round(float(test_metrics["rmse"]), 2),
            "test_mae": round(float(test_metrics["mae"]), 2),
            "test_nasa": round(float(test_metrics["nasa_score"]), 1),
        })
        fitted_models[name] = (model, test_metrics, test_predictions)

    info = {
        "subset": subset,
        "window_length": window_length,
        "variance_threshold": variance_threshold,
        "rul_cap": rul_cap,
        "active_sensors": active_sensors,
        "n_regimes": n_clusters,
        "cluster_sizes": {
            str(label): size for label, size in final_normalizer.cluster_sizes_.items()
        },
        "dropped_sensors": dropped_sensors,
    }

    model_dir = PROJECT_ROOT / "models" / subset
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_normalizer, model_dir / "regime_normalizer.joblib")
    for name, (model, _, _) in fitted_models.items():
        joblib.dump(model, model_dir / f"{name.lower()}_model.joblib")

    _, rf_metrics, rf_predictions = fitted_models["RandomForest"]
    pd.DataFrame({
        "engine_id": test_engine_ids,
        "actual_rul": y_test,
        "predicted_rul": np.maximum(0.0, rf_predictions),
        "error": rf_predictions - y_test,
    }).to_csv(model_dir / "test_predictions.csv", index=False)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump({
            "status": "success",
            "dataset_subset": subset,
            "model_type": "RandomForestRegressor",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": seeds[0],
                "window_length": window_length,
                "rul_cap": rul_cap,
                "active_sensors": active_sensors,
            },
            "test_metrics": rf_metrics,
            "artifacts": {
                "model": "randomforest_model.joblib",
                "preprocessor": "regime_normalizer.joblib",
                "test_predictions": "test_predictions.csv",
            },
        }, metrics_file, indent=2)

    return rows, info


def run_benchmarks(subsets: List[str]) -> pd.DataFrame:
    config = load_config()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benchmark_all.csv"
    info_path = results_dir / "regime_info.json"
    previous_results = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()
    all_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    rows: List[Dict[str, Any]] = []
    elapsed = time.perf_counter()

    for subset in subsets:
        subset_rows, subset_info = run_subset(subset, config)
        rows.extend(subset_rows)
        all_info[subset] = subset_info
        print(
            f"  Final-training regimes: {subset_info['cluster_sizes']}; "
            f"dropped sensors: {subset_info['dropped_sensors']}"
        )
        pd.DataFrame(subset_rows).to_csv(
            results_dir / f"model_comparison_{subset.lower()}.csv", index=False
        )

        for row in subset_rows:
            if row["test_rmse"] < 8:
                raise RuntimeError(
                    f"Suspicious {subset}/{row['model']} test RMSE {row['test_rmse']:.2f} (< 8). "
                    "Stopped; diagnose leakage before continuing."
                )
            if subset in {"FD002", "FD004"}:
                baseline = next((
                    item for item in rows
                    if item["subset"] == "FD001" and item["model"] == row["model"]
                ), None)
                if row["test_rmse"] > 35:
                    print(
                        f"  DIAGNOSTIC: {subset}/{row['model']} RMSE is above 35; "
                        f"inspect regime cluster sizes {subset_info['cluster_sizes']} "
                        f"and dropped sensors {subset_info['dropped_sensors']}."
                    )
                    sizes = list(subset_info["cluster_sizes"].values())
                    if len(sizes) != subset_info["n_regimes"] or min(sizes, default=0) == 0:
                        raise RuntimeError(
                            f"{subset} has empty/missing regime clusters; stop to diagnose "
                            f"before benchmarking further subsets: {subset_info['cluster_sizes']}"
                        )
                if baseline and row["test_rmse"] < 0.8 * baseline["test_rmse"]:
                    raise RuntimeError(
                        f"Suspicious {subset}/{row['model']} is over 20% better than FD001. "
                        "Stopped; diagnose leakage before continuing."
                    )
                if baseline and row["test_rmse"] < baseline["test_rmse"]:
                    raise RuntimeError(
                        f"Suspicious {subset}/{row['model']} test RMSE "
                        f"{row['test_rmse']:.2f} is better than FD001 "
                        f"({baseline['test_rmse']:.2f}). Stop and diagnose leakage."
                    )

    if not previous_results.empty:
        previous_results = previous_results[~previous_results["subset"].isin(subsets)]
        rows = previous_results.to_dict(orient="records") + rows
    result_frame = pd.DataFrame(rows).sort_values(["subset", "model"]).reset_index(drop=True)
    result_frame = result_frame[[
        "subset", "model", "cv_rmse_mean", "cv_rmse_std",
        "test_rmse", "test_mae", "test_nasa",
    ]]
    result_frame.to_csv(csv_path, index=False)
    for subset in subsets:
        result_frame[result_frame["subset"] == subset].to_csv(
            results_dir / f"model_comparison_{subset.lower()}.csv", index=False
        )
    info_path.write_text(json.dumps(all_info, indent=2) + "\n", encoding="utf-8")

    subsets_in_table = list(dict.fromkeys(result_frame["subset"].tolist()))
    model_names = list(dict.fromkeys(result_frame["model"].tolist()))
    positions = np.arange(len(subsets_in_table))
    width = 0.8 / max(1, len(model_names))
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, model_name in enumerate(model_names):
        model_rows = result_frame[result_frame["model"] == model_name].set_index("subset")
        values = [
            model_rows.loc[subset, "test_rmse"] if subset in model_rows.index else np.nan
            for subset in subsets_in_table
        ]
        axis.bar(positions + index * width, values, width, label=model_name)
    axis.set_xticks(positions + width * (len(model_names) - 1) / 2, subsets_in_table)
    axis.set_ylabel("Test RMSE (cycles)")
    axis.set_xlabel("C-MAPSS subset")
    axis.set_title("Test RMSE by subset and model")
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "benchmark_all.png", dpi=160)
    plt.close(figure)

    print("\nRF + LightGBM results (GRU skipped to keep the four-subset run bounded):")
    print(result_frame.to_string(index=False))
    print(f"Total runtime: {time.perf_counter() - elapsed:.1f}s")
    print(f"Saved {csv_path}, {info_path}, and {results_dir / 'benchmark_all.png'}")
    return result_frame
