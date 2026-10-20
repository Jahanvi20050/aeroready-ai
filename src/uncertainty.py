"""Train-engine-calibrated RUL prediction intervals for the C-MAPSS subsets."""
import argparse
import json
from importlib.metadata import version
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import SENSOR_COLUMNS, get_default_data_dir, load_raw_data
from src.preprocessing import (
    compute_train_rul,
    compute_tree_window_features,
    extract_windowed_sequences,
)
from src.regimes import RegimeNormalizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUANTILE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95)
NOMINAL_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90)
QUANTILE_METHOD = "LightGBMQuantile"
CONFORMAL_METHOD = "SplitConformalRandomForest"
PREDICTION_COLUMNS = ["unit", "method", "rul_lower", "rul_median", "rul_upper", "true_rul"]
METRIC_COLUMNS = [
    "subset",
    "method",
    "empirical_coverage",
    "mean_interval_width",
    "late_risk_fraction",
]


def load_config() -> Dict[str, Any]:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def split_calibration_engines(
    engine_ids: Iterable[int],
    calibration_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return deterministic, disjoint fitting and calibration engine IDs."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")
    unique_ids = np.asarray(sorted(set(engine_ids)), dtype=int)
    if len(unique_ids) < 2:
        raise ValueError("At least two engines are required for split calibration.")
    calibration_count = min(
        len(unique_ids) - 1,
        max(1, int(ceil(len(unique_ids) * calibration_fraction))),
    )
    shuffled_ids = np.random.default_rng(seed).permutation(unique_ids)
    calibration_ids = sorted(shuffled_ids[:calibration_count].tolist())
    fitting_ids = sorted(shuffled_ids[calibration_count:].tolist())
    if set(fitting_ids).intersection(calibration_ids):
        raise AssertionError("Fitting and calibration engines must be disjoint.")
    return fitting_ids, calibration_ids


def sample_calibration_cut_points(
    frame: pd.DataFrame,
    engine_ids: Iterable[int],
    minimum_observed_cycles: int,
    seed: int,
    endpoint_cycles: Sequence[int] = None,
) -> Dict[int, int]:
    """Sample one seeded observed-cycle endpoint per engine."""
    if minimum_observed_cycles < 1:
        raise ValueError("minimum_observed_cycles must be at least one.")
    candidates = (
        np.asarray(endpoint_cycles, dtype=int).reshape(-1)
        if endpoint_cycles is not None
        else None
    )
    if candidates is not None and len(candidates) == 0:
        raise ValueError("endpoint_cycles must not be empty.")
    rng = np.random.default_rng(seed)
    cut_points: Dict[int, int] = {}
    for engine_id in sorted(set(engine_ids)):
        engine = frame.loc[frame["unit"] == engine_id, "cycle"]
        if engine.empty:
            raise ValueError(f"No training cycles found for calibration engine {engine_id}.")
        minimum_cycle = int(engine.min()) + minimum_observed_cycles - 1
        maximum_cycle = int(engine.max())
        if minimum_cycle > maximum_cycle:
            raise ValueError(
                f"Engine {engine_id} has fewer than {minimum_observed_cycles} cycles."
            )
        if candidates is None:
            cut_cycle = rng.integers(minimum_cycle, maximum_cycle + 1)
        else:
            sampled_endpoint = int(rng.choice(candidates))
            cut_cycle = min(maximum_cycle, max(minimum_cycle, sampled_endpoint))
        cut_points[int(engine_id)] = int(cut_cycle)
    return cut_points


def compute_interval_metrics(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> Dict[str, float]:
    """Calculate coverage, average width, and the late-risk fraction."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    lower_bound = np.asarray(lower, dtype=float).reshape(-1)
    upper_bound = np.asarray(upper, dtype=float).reshape(-1)
    if len(truth) == 0 or len(truth) != len(lower_bound) or len(truth) != len(upper_bound):
        raise ValueError("Truth and interval bounds must have equal, nonzero lengths.")
    if not (
        np.isfinite(truth).all()
        and np.isfinite(lower_bound).all()
        and np.isfinite(upper_bound).all()
    ):
        raise ValueError("Truth and interval bounds must contain only finite values.")
    if np.any(lower_bound > upper_bound):
        raise ValueError("Every lower interval bound must be <= its upper bound.")
    return {
        "empirical_coverage": float(np.mean((truth >= lower_bound) & (truth <= upper_bound))),
        "mean_interval_width": float(np.mean(upper_bound - lower_bound)),
        "late_risk_fraction": float(np.mean(truth < lower_bound)),
    }


def finite_sample_quantile(scores: Sequence[float], coverage: float) -> float:
    """Return the split-conformal order statistic with its finite-sample correction."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be a nonempty finite sequence.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1.")
    rank = min(len(values), int(ceil((len(values) + 1) * coverage)))
    return float(np.sort(values)[rank - 1])


def order_interval(
    lower: Sequence[float],
    median: Sequence[float],
    upper: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repair quantile crossing and return nonnegative ordered RUL bounds."""
    values = np.column_stack((lower, median, upper)).astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Predicted quantiles must contain only finite values.")
    values = np.maximum(values, 0.0)
    values.sort(axis=1)
    return values[:, 0], values[:, 1], values[:, 2]


def _prepare_last_observation_features(
    frame: pd.DataFrame,
    normalizer: RegimeNormalizer,
    sensors: List[str],
    window_length: int,
) -> Tuple[pd.DataFrame, List[int]]:
    normalized = normalizer.transform(frame)
    windows, _, engine_ids = extract_windowed_sequences(
        normalized,
        feature_cols=sensors,
        window_length=window_length,
        is_test_last=True,
        target_col=None,
    )
    return compute_tree_window_features(windows, feature_names=sensors), engine_ids


def _quantile_interval(
    predictions: np.ndarray,
    median_predictions: np.ndarray,
    levels: Sequence[float],
    nominal_coverage: float,
    correction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_to_index = {float(level): index for index, level in enumerate(levels)}
    tail = (1.0 - nominal_coverage) / 2.0
    lower_level = round(tail, 8)
    upper_level = round(1.0 - tail, 8)
    lower_index = level_to_index[lower_level]
    upper_index = level_to_index[upper_level]
    median = np.maximum(0.0, median_predictions)
    lower = np.maximum(0.0, predictions[:, lower_index] - correction)
    upper = np.maximum(median, predictions[:, upper_index] + correction)
    lower = np.minimum(lower, median)
    return order_interval(lower, median, upper)


def _plot_calibration_curve(
    subset: str,
    calibration_results: Dict[str, Dict[float, float]],
) -> None:
    results_dir = PROJECT_ROOT / "results"
    figure, axis = plt.subplots(figsize=(7, 6))
    nominal = np.asarray(NOMINAL_COVERAGES)
    axis.plot(nominal, nominal, linestyle="--", color="gray", label="Ideal")
    for method, empirical in calibration_results.items():
        axis.plot(
            nominal,
            [empirical[level] for level in NOMINAL_COVERAGES],
            marker="o",
            label=method,
        )
    axis.set(
        xlabel="Nominal coverage",
        ylabel="Empirical test coverage",
        title=f"RUL interval calibration — {subset}",
        xlim=(0.45, 0.95),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / f"calibration_{subset}.png", dpi=160)
    plt.close(figure)


def run_subset_uncertainty(
    subset: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    dataset = config.get("dataset", {})
    preprocessing = config.get("preprocessing", {})
    training = config.get("training", {})
    uncertainty = config.get("uncertainty", {})
    window_length = int(preprocessing.get("window_length", 30))
    variance_threshold = float(preprocessing.get("variance_threshold", 1e-5))
    rul_cap = float(training.get("rul_cap", 125))
    model_seed = int(training.get("seeds", [42])[0])
    calibration_fraction = float(uncertainty.get("calibration_fraction", 0.2))
    split_seed = int(uncertainty.get("split_seed", 2026))
    cut_seed = int(uncertainty.get("cut_seed", 2027)) + int(subset[-3:])
    minimum_observed_cycles = int(
        uncertainty.get("minimum_observed_cycles", window_length)
    )
    cluster_map = preprocessing.get("regime_clusters", {})
    n_clusters = int(cluster_map.get(subset, 1 if subset in {"FD001", "FD003"} else 6))
    data_dir = PROJECT_ROOT / dataset.get("data_dir", "data/raw")
    if not data_dir.exists():
        data_dir = get_default_data_dir()

    train_df, test_df, rul_df = load_raw_data(data_dir=data_dir, subset=subset)
    train_labeled = compute_train_rul(train_df, max_rul_clip=rul_cap)
    fitting_ids, calibration_ids = split_calibration_engines(
        train_df["unit"].unique(),
        calibration_fraction=calibration_fraction,
        seed=split_seed + int(subset[-3:]),
    )
    fitting_labeled = train_labeled[train_labeled["unit"].isin(fitting_ids)].copy()
    active_sensors = [
        column for column in SENSOR_COLUMNS
        if fitting_labeled[column].std() > variance_threshold
    ]
    if not active_sensors:
        raise ValueError(f"{subset} fitting-engine data has no active sensors.")

    normalizer = RegimeNormalizer(n_clusters=n_clusters, random_state=model_seed)
    normalized_fit = normalizer.fit_transform(fitting_labeled)
    train_windows, train_targets, _ = extract_windowed_sequences(
        normalized_fit,
        feature_cols=active_sensors,
        window_length=window_length,
        is_test_last=False,
        target_col="RUL",
    )
    train_features = compute_tree_window_features(train_windows, active_sensors)
    if train_targets is None:
        raise ValueError(f"{subset} fitting data did not produce RUL training targets.")

    cut_points = sample_calibration_cut_points(
        train_df,
        calibration_ids,
        minimum_observed_cycles=minimum_observed_cycles,
        seed=cut_seed,
        endpoint_cycles=test_df.groupby("unit")["cycle"].max().to_numpy(dtype=int),
    )
    calibration_parts = [
        train_df.loc[
            (train_df["unit"] == engine_id)
            & (train_df["cycle"] <= cut_cycle)
        ]
        for engine_id, cut_cycle in cut_points.items()
    ]
    calibration_history = pd.concat(calibration_parts, ignore_index=True)
    calibration_features, calibration_feature_ids = _prepare_last_observation_features(
        calibration_history,
        normalizer,
        active_sensors,
        window_length,
    )
    if calibration_feature_ids != list(cut_points):
        raise ValueError(
            f"{subset} calibration feature engine order does not match sampled cuts."
        )
    max_cycles = train_df.groupby("unit")["cycle"].max()
    calibration_truth = np.asarray(
        [int(max_cycles[engine_id]) - cut_points[engine_id] for engine_id in cut_points],
        dtype=float,
    )

    test_features, test_engine_ids = _prepare_last_observation_features(
        test_df,
        normalizer,
        active_sensors,
        window_length,
    )
    expected_test_ids = rul_df["unit"].tolist()
    if test_engine_ids != expected_test_ids:
        raise ValueError(
            f"{subset} test engine order does not match the ordered RUL file."
        )
    test_truth = rul_df["true_rul"].to_numpy(dtype=float)

    levels = tuple(float(value) for value in uncertainty.get("quantile_levels", QUANTILE_LEVELS))
    if (
        not all(0.0 < level < 1.0 for level in levels)
        or tuple(sorted(set(levels))) != levels
        or not set(QUANTILE_LEVELS).issubset(levels)
    ):
        raise ValueError(
            "uncertainty.quantile_levels must be strictly increasing, within (0, 1), "
            f"and include {list(QUANTILE_LEVELS)}."
        )
    quantile_dir = PROJECT_ROOT / "models" / subset / "uncertainty"
    quantile_dir.mkdir(parents=True, exist_ok=True)
    quantile_params = {
        "objective": "quantile",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": model_seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    quantile_calibration_predictions = []
    quantile_test_predictions = []
    for level in levels:
        model = LGBMRegressor(alpha=level, **quantile_params)
        model.fit(train_features, train_targets)
        filename = f"lightgbm_quantile_{level:.2f}.joblib".replace(".", "_", 1)
        joblib.dump(model, quantile_dir / filename)
        quantile_calibration_predictions.append(
            np.maximum(0.0, model.predict(calibration_features))
        )
        quantile_test_predictions.append(np.maximum(0.0, model.predict(test_features)))
    quantile_calibration_matrix = np.maximum.accumulate(
        np.column_stack(quantile_calibration_predictions), axis=1
    )
    quantile_test_matrix = np.maximum.accumulate(
        np.column_stack(quantile_test_predictions), axis=1
    )
    median_index = levels.index(0.50)
    quantile_calibration_median = quantile_calibration_matrix[:, median_index]
    quantile_test_median = quantile_test_matrix[:, median_index]

    def quantile_correction(nominal_coverage: float) -> float:
        lower, _, upper = _quantile_interval(
            quantile_calibration_matrix,
            quantile_calibration_median,
            levels,
            nominal_coverage,
        )
        scores = np.maximum(lower - calibration_truth, calibration_truth - upper)
        return finite_sample_quantile(scores, nominal_coverage)

    quantile_raw_interval = _quantile_interval(
        quantile_test_matrix,
        quantile_test_median,
        levels,
        0.80,
    )
    quantile_raw_metrics = compute_interval_metrics(
        test_truth, quantile_raw_interval[0], quantile_raw_interval[2]
    )
    quantile_80_correction = quantile_correction(0.80)
    quantile_80_interval = _quantile_interval(
        quantile_test_matrix,
        quantile_test_median,
        levels,
        0.80,
        correction=quantile_80_correction,
    )
    quantile_metrics = compute_interval_metrics(
        test_truth, quantile_80_interval[0], quantile_80_interval[2]
    )

    point_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=model_seed,
        n_jobs=-1,
    )
    point_model.fit(train_features, train_targets)
    calibration_point_predictions = np.maximum(
        0.0, point_model.predict(calibration_features)
    )
    calibration_residuals = np.abs(calibration_truth - calibration_point_predictions)
    test_point_predictions = np.maximum(0.0, point_model.predict(test_features))
    naive_conformal_radius = float(
        np.quantile(calibration_residuals, 0.80, method="higher")
    )
    finite_conformal_radius = finite_sample_quantile(calibration_residuals, 0.80)
    conformal_raw_lower = np.maximum(0.0, test_point_predictions - naive_conformal_radius)
    conformal_raw_upper = test_point_predictions + naive_conformal_radius
    conformal_raw_metrics = compute_interval_metrics(
        test_truth, conformal_raw_lower, conformal_raw_upper
    )
    conformal_lower = np.maximum(0.0, test_point_predictions - finite_conformal_radius)
    conformal_upper = test_point_predictions + finite_conformal_radius
    conformal_metrics = compute_interval_metrics(
        test_truth, conformal_lower, conformal_upper
    )

    prediction_rows = []
    quantile_lower, quantile_median, quantile_upper = quantile_80_interval
    for index, engine_id in enumerate(test_engine_ids):
        prediction_rows.extend([
            {
                "unit": int(engine_id),
                "method": QUANTILE_METHOD,
                "rul_lower": float(quantile_lower[index]),
                "rul_median": float(quantile_median[index]),
                "rul_upper": float(quantile_upper[index]),
                "true_rul": float(test_truth[index]),
            },
            {
                "unit": int(engine_id),
                "method": CONFORMAL_METHOD,
                "rul_lower": float(conformal_lower[index]),
                "rul_median": float(test_point_predictions[index]),
                "rul_upper": float(conformal_upper[index]),
                "true_rul": float(test_truth[index]),
            },
        ])
    prediction_frame = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)

    metric_rows = [
        {"subset": subset, "method": QUANTILE_METHOD, **quantile_metrics},
        {"subset": subset, "method": CONFORMAL_METHOD, **conformal_metrics},
    ]
    calibration_results: Dict[str, Dict[float, float]] = {
        QUANTILE_METHOD: {},
        CONFORMAL_METHOD: {},
    }
    for nominal_coverage in NOMINAL_COVERAGES:
        correction = quantile_correction(nominal_coverage)
        lower, _, upper = _quantile_interval(
            quantile_test_matrix,
            quantile_test_median,
            levels,
            nominal_coverage,
            correction=correction,
        )
        calibration_results[QUANTILE_METHOD][nominal_coverage] = compute_interval_metrics(
            test_truth, lower, upper
        )["empirical_coverage"]

        radius = finite_sample_quantile(calibration_residuals, nominal_coverage)
        lower = np.maximum(0.0, test_point_predictions - radius)
        upper = test_point_predictions + radius
        calibration_results[CONFORMAL_METHOD][nominal_coverage] = compute_interval_metrics(
            test_truth, lower, upper
        )["empirical_coverage"]

    model_dir = PROJECT_ROOT / "models" / subset / "uncertainty"
    joblib.dump(normalizer, model_dir / "conformal_regime_normalizer.joblib")
    joblib.dump(point_model, model_dir / "conformal_randomforest_model.joblib")
    calibration_info = {
        "subset": subset,
        "fit_engine_ids": fitting_ids,
        "calibration_engine_ids": calibration_ids,
        "calibration_cut_points": {
            str(engine_id): cycle for engine_id, cycle in cut_points.items()
        },
        "calibration_fraction": calibration_fraction,
        "split_seed": split_seed + int(subset[-3:]),
        "cut_seed": cut_seed,
        "minimum_observed_cycles": minimum_observed_cycles,
        "n_clusters": n_clusters,
        "active_sensors": active_sensors,
        "cluster_sizes": {
            str(label): int(size) for label, size in normalizer.cluster_sizes_.items()
        },
        "conformal_quantile_80": finite_conformal_radius,
        "quantile_conformal_correction_80": quantile_80_correction,
        "quantile_levels": list(levels),
        "dependencies": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "lightgbm": version("lightgbm"),
        },
    }
    (model_dir / "calibration.json").write_text(
        json.dumps(calibration_info, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics = {
        "subset": subset,
        "nominal_coverage": 0.80,
        "quantile_regression": {
            "before_calibration_coverage": quantile_raw_metrics["empirical_coverage"],
            "after_calibration_coverage": quantile_metrics["empirical_coverage"],
            "calibration_adjustment": quantile_80_correction,
        },
        "split_conformal_random_forest": {
            "before_finite_sample_correction_coverage": conformal_raw_metrics[
                "empirical_coverage"
            ],
            "after_finite_sample_correction_coverage": conformal_metrics[
                "empirical_coverage"
            ],
            "naive_radius": naive_conformal_radius,
            "finite_sample_radius": finite_conformal_radius,
        },
        "calibration_engine_count": len(calibration_ids),
        "fitting_engine_count": len(fitting_ids),
    }
    _plot_calibration_curve(subset, calibration_results)
    print(
        f"{subset}: quantile coverage {quantile_raw_metrics['empirical_coverage']:.3f} "
        f"-> {quantile_metrics['empirical_coverage']:.3f}; "
        f"split-conformal coverage {conformal_raw_metrics['empirical_coverage']:.3f} "
        f"-> {conformal_metrics['empirical_coverage']:.3f}; "
        f"calibration engines={len(calibration_ids)}"
    )
    for row in metric_rows:
        coverage = row["empirical_coverage"]
        if coverage < 0.65 or coverage > 0.92:
            print(
                f"DIAGNOSTIC {subset}/{row['method']}: final 80% coverage is "
                f"{coverage:.3f}; the split uses {len(calibration_ids)} held-out "
                "training engines and one random cut per engine. See "
                f"results/uncertainty_diagnostics.json and "
                f"models/{subset}/uncertainty/calibration.json."
            )
    (PROJECT_ROOT / "results" / f"predictions_{subset}.csv").write_text(
        prediction_frame.to_csv(index=False),
        encoding="utf-8",
    )
    return prediction_frame, metric_rows, diagnostics


def run_uncertainty_benchmark(subsets: Sequence[str] = None) -> pd.DataFrame:
    config = load_config()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = results_dir / "benchmark_all.csv"
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"Existing benchmark results not found: {benchmark_file}")
    available_subsets = sorted(pd.read_csv(benchmark_file, usecols=["subset"])["subset"].unique())
    selected_subsets = list(subsets) if subsets else available_subsets
    unknown_subsets = sorted(set(selected_subsets) - set(available_subsets))
    if unknown_subsets:
        raise ValueError(f"Subsets are missing from benchmark_all.csv: {unknown_subsets}")

    metrics_path = results_dir / "uncertainty.csv"
    diagnostics_path = results_dir / "uncertainty_diagnostics.json"
    initial_metrics_path = results_dir / "uncertainty_initial.csv"
    initial_diagnostics_path = results_dir / "uncertainty_diagnostics_initial.json"
    if metrics_path.is_file() and not initial_metrics_path.is_file():
        initial_metrics_path.write_bytes(metrics_path.read_bytes())
    if diagnostics_path.is_file() and not initial_diagnostics_path.is_file():
        initial_diagnostics_path.write_bytes(diagnostics_path.read_bytes())
    existing_metrics = (
        pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame(columns=METRIC_COLUMNS)
    )
    existing_diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.is_file()
        else {}
    )
    all_metric_rows = []
    for subset in selected_subsets:
        predictions, metric_rows, diagnostics = run_subset_uncertainty(subset, config)
        all_metric_rows.extend(metric_rows)
        existing_diagnostics[subset] = diagnostics
        existing_metrics = existing_metrics[
            existing_metrics["subset"] != subset
        ]
        metric_updates = pd.DataFrame(all_metric_rows)
        metric_parts = (
            [existing_metrics, metric_updates]
            if not existing_metrics.empty
            else [metric_updates]
        )
        metrics_frame = pd.concat(
            metric_parts,
            ignore_index=True,
        )[METRIC_COLUMNS].sort_values(["subset", "method"]).reset_index(drop=True)
        metrics_frame.to_csv(metrics_path, index=False)
        diagnostics_path.write_text(
            json.dumps(existing_diagnostics, indent=2) + "\n",
            encoding="utf-8",
        )
    output = pd.read_csv(metrics_path)
    print(output.to_string(index=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate engine-disjoint, training-calibrated RUL intervals."
    )
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"])
    arguments = parser.parse_args()
    run_uncertainty_benchmark([arguments.subset] if arguments.subset else None)


if __name__ == "__main__":
    main()
