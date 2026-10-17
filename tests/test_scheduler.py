"""
Unit tests for maintenance scheduler constraints and simulation.
"""
import pytest
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.scheduler import MaintenanceScheduler
from src.fleet_simulator import FleetSimulator


def test_scheduler_bay_capacity_constraint():
    """Validates that hangar bay capacity is strictly respected every single day."""
    num_bays = 2
    scheduler = MaintenanceScheduler(planning_horizon_days=14, num_hangar_bays=num_bays)
    simulator = FleetSimulator()
    fleet = simulator.generate_fleet()

    # Test baseline schedule
    baseline_result = scheduler.run_baseline_schedule(fleet)
    for day in range(1, 15):
        day_bay_count = sum(
            1 for item in baseline_result["schedule"]
            if item["start_day"] <= day <= item["end_day"]
        )
        assert day_bay_count <= num_bays, f"Baseline exceeded hangar capacity on day {day}"

    # Test optimized schedule
    optim_result = scheduler.run_optimized_schedule(fleet)
    for day in range(1, 15):
        day_bay_count = sum(
            1 for item in optim_result["schedule"]
            if item["start_day"] <= day <= item["end_day"]
        )
        assert day_bay_count <= num_bays, f"Optimizer exceeded hangar capacity on day {day}"


def test_scheduler_spare_parts_lead_time_respected():
    """Ensures optimized schedule never dispatches aircraft before spare parts arrive."""
    scheduler = MaintenanceScheduler(planning_horizon_days=14, num_hangar_bays=2)
    simulator = FleetSimulator()
    fleet = simulator.generate_fleet()

    optim_result = scheduler.run_optimized_schedule(fleet)
    for item in optim_result["schedule"]:
        aircraft = next(a for a in fleet if a["aircraft_id"] == item["aircraft_id"])
        lead_days = aircraft.get("spare_part_lead_days", 0)
        # Optimized scheduler must not begin maintenance in bay until day >= lead_days + 1
        assert item["start_day"] >= lead_days + 1, (
            f"Aircraft {item['aircraft_id']} scheduled on day {item['start_day']}, "
            f"before parts lead time of {lead_days} days."
        )


def test_schedule_comparison_structure():
    """Checks that schedule comparison outputs valid numerical metrics."""
    scheduler = MaintenanceScheduler(planning_horizon_days=14, num_hangar_bays=2)
    simulator = FleetSimulator()
    fleet = simulator.generate_fleet()

    comparison = scheduler.compare_schedules(fleet)

    assert "baseline" in comparison
    assert "optimized" in comparison
    assert "metrics" in comparison
    assert "assumptions" in comparison

    m = comparison["metrics"]
    assert "baseline_avg_readiness_pct" in m
    assert "optimized_avg_readiness_pct" in m
    assert "delta_readiness_pct" in m
    assert isinstance(m["delta_readiness_pct"], (int, float))
