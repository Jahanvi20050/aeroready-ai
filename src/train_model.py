"""Train and compare the leakage-safe C-MAPSS RUL models for one subset."""
import argparse
import sys
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.benchmark_subsets import load_config, run_benchmarks


def train_pipeline(subset: Optional[str] = None):
    """Run the configured RandomForest and LightGBM pipeline for a subset."""
    config = load_config()
    selected_subset = subset or config.get("dataset", {}).get("subset", "FD001")
    return run_benchmarks([selected_subset])


def main():
    config = load_config()
    default_subset = config.get("dataset", {}).get("subset", "FD001")
    parser = argparse.ArgumentParser(description="Train AeroReady AI C-MAPSS RUL models.")
    parser.add_argument(
        "--subset",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default=default_subset,
        help=f"C-MAPSS subset to train (default: {default_subset})",
    )
    args = parser.parse_args()
    results = train_pipeline(args.subset)
    random_forest = results[results["model"] == "RandomForest"]
    if not random_forest.empty and float(random_forest.iloc[0]["test_rmse"]) > 25:
        raise RuntimeError(
            f"{args.subset} test RMSE is above 25; stop and investigate before deployment."
        )


if __name__ == "__main__":
    main()
