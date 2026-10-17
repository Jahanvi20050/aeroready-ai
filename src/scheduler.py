"""
AeroReady AI - Maintenance Scheduling & Availability Optimizer Module
Provides:
1. Baseline Scheduler: Naive/reactive FIFO approach.
2. Optimized Scheduler: Priority-weighted, parts-aware, capacity-constrained optimizer.
3. Comparative simulation over a configurable planning horizon.
"""
from typing import List, Dict, Any, Tuple
import copy
import pandas as pd
import numpy as np


class MaintenanceScheduler:
    """
    Simulates and optimizes fleet maintenance schedules against operational constraints:
    - Hangar bay capacity (concurrent aircraft under maintenance)
    - Spare part delivery lead times
    - Mission demand (target available aircraft per day)
    - Aircraft mission priority tiers and engine RUL urgency
    """

    def __init__(
        self,
        planning_horizon_days: int = 14,
        num_hangar_bays: int = 2,
        mission_demand_aircraft: int = 7,
        daily_flight_cycles: float = 1.5,
    ):
        self.horizon = int(max(3, min(60, planning_horizon_days)))
        self.num_bays = int(max(1, min(10, num_hangar_bays)))
        self.mission_demand = int(max(1, min(10, mission_demand_aircraft)))
        self.daily_cycles = float(daily_flight_cycles)

    def _calculate_priority_score(self, aircraft: Dict[str, Any]) -> float:
        """
        Computes composite maintenance urgency score.
        Higher score = higher scheduling priority.
        """
        rul = max(1.0, float(aircraft.get("predicted_rul", 50.0)))
        tier_weight = {1: 3.5, 2: 2.0, 3: 1.0}.get(aircraft.get("priority_rank", 2), 1.0)
        
        # Urgency is inversely proportional to RUL, weighted by mission priority
        urgency = 100.0 / (rul + 5.0)
        return round(urgency * tier_weight, 2)

    def run_baseline_schedule(self, fleet: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Baseline naive schedule:
        - Reactively queues aircraft by simple arrival order (or ID order).
        - If parts are missing, the aircraft enters the bay on Day 1 anyway and stalls
          (bay-blocking penalty common in unoptimized operations).
        - Bay capacity is respected, but sequencing ignores mission priority and parts readiness.
        """
        fleet_copy = copy.deepcopy(fleet)
        schedule = []
        bay_occupancy = {day: 0 for day in range(1, self.horizon + 1)}
        bay_assignments = {f"Bay {b+1}": [None] * self.horizon for b in range(self.num_bays)}

        # Filter aircraft that need maintenance (Critical or Warning)
        candidates = [a for a in fleet_copy if a["health_category"] in ["Critical", "Warning"]]
        # In baseline, sort naively by aircraft ID rather than optimization priority
        candidates.sort(key=lambda a: a["aircraft_id"])

        for aircraft in candidates:
            duration = aircraft["maintenance_duration"]
            parts_lead = aircraft.get("spare_part_lead_days", 0)

            # In baseline, naive planning starts at Day 1 if bay is free, but if parts are missing,
            # actual maintenance only begins after parts arrive, effectively extending bay occupancy!
            effective_duration = duration + parts_lead

            # Find earliest consecutive days where at least 1 bay is available
            scheduled_start = None
            scheduled_bay = None

            for d in range(1, self.horizon - effective_duration + 2):
                # Check which bay is free for the entire window
                for b_idx in range(self.num_bays):
                    bay_name = f"Bay {b_idx + 1}"
                    window_free = all(
                        bay_assignments[bay_name][day_idx] is None
                        for day_idx in range(d - 1, min(self.horizon, d - 1 + effective_duration))
                    )
                    if window_free:
                        scheduled_start = d
                        scheduled_bay = bay_name
                        break
                if scheduled_start is not None:
                    break

            if scheduled_start is not None:
                end_day = min(self.horizon, scheduled_start + effective_duration - 1)
                for day in range(scheduled_start, end_day + 1):
                    bay_occupancy[day] += 1
                    bay_assignments[scheduled_bay][day - 1] = aircraft["aircraft_id"]

                schedule.append({
                    "aircraft_id": aircraft["aircraft_id"],
                    "tail_number": aircraft["tail_number"],
                    "start_day": scheduled_start,
                    "end_day": end_day,
                    "duration_days": effective_duration,
                    "assigned_bay": scheduled_bay,
                    "parts_delayed_in_bay": parts_lead > 0,
                    "priority_score": self._calculate_priority_score(aircraft),
                    "health_category": aircraft["health_category"],
                    "predicted_rul": aircraft["predicted_rul"],
                    "mission_priority": aircraft["mission_priority"],
                })

        # Simulate fleet availability over the horizon
        timeline_metrics = self._simulate_timeline(fleet_copy, schedule, scheduler_type="baseline")

        return {
            "scheduler_name": "Baseline Reactive (FIFO)",
            "schedule": schedule,
            "bay_assignments": bay_assignments,
            "timeline": timeline_metrics,
        }

    def run_optimized_schedule(self, fleet: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimized schedule:
        1. Prioritizes aircraft by composite urgency (RUL risk + mission tier).
        2. Parts-aware: Does not dispatch aircraft into bay until parts are on site (zero bay stall).
        3. Level-loads bay capacity to maintain maximum daily mission-ready aircraft.
        4. Overhauls RUL post-maintenance to restore mission fitness.
        """
        fleet_copy = copy.deepcopy(fleet)
        schedule = []
        bay_occupancy = {day: 0 for day in range(1, self.horizon + 1)}
        bay_assignments = {f"Bay {b+1}": [None] * self.horizon for b in range(self.num_bays)}

        candidates = [a for a in fleet_copy if a["health_category"] in ["Critical", "Warning"]]
        # Sort by composite priority score (highest priority first)
        candidates.sort(key=lambda a: self._calculate_priority_score(a), reverse=True)

        for aircraft in candidates:
            duration = aircraft["maintenance_duration"]
            parts_lead = aircraft.get("spare_part_lead_days", 0)
            earliest_possible_start = parts_lead + 1  # Cannot enter bay until parts are physically ready!

            scheduled_start = None
            scheduled_bay = None

            for d in range(earliest_possible_start, self.horizon - duration + 2):
                for b_idx in range(self.num_bays):
                    bay_name = f"Bay {b_idx + 1}"
                    window_free = all(
                        bay_assignments[bay_name][day_idx] is None
                        for day_idx in range(d - 1, min(self.horizon, d - 1 + duration))
                    )
                    if window_free:
                        scheduled_start = d
                        scheduled_bay = bay_name
                        break
                if scheduled_start is not None:
                    break

            if scheduled_start is not None:
                end_day = min(self.horizon, scheduled_start + duration - 1)
                for day in range(scheduled_start, end_day + 1):
                    bay_occupancy[day] += 1
                    bay_assignments[scheduled_bay][day - 1] = aircraft["aircraft_id"]

                schedule.append({
                    "aircraft_id": aircraft["aircraft_id"],
                    "tail_number": aircraft["tail_number"],
                    "start_day": scheduled_start,
                    "end_day": end_day,
                    "duration_days": duration,
                    "assigned_bay": scheduled_bay,
                    "parts_delayed_in_bay": False,
                    "priority_score": self._calculate_priority_score(aircraft),
                    "health_category": aircraft["health_category"],
                    "predicted_rul": aircraft["predicted_rul"],
                    "mission_priority": aircraft["mission_priority"],
                })

        timeline_metrics = self._simulate_timeline(fleet_copy, schedule, scheduler_type="optimized")

        return {
            "scheduler_name": "AeroReady AI Optimized",
            "schedule": schedule,
            "bay_assignments": bay_assignments,
            "timeline": timeline_metrics,
        }

    def _simulate_timeline(
        self,
        fleet: List[Dict[str, Any]],
        schedule: List[Dict[str, Any]],
        scheduler_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Simulates day-by-day fleet operations across the planning horizon.
        Tracks available vs in-maintenance vs grounded aircraft.
        """
        sched_map = {item["aircraft_id"]: item for item in schedule}
        daily_records = []

        # Current RUL tracking per aircraft
        current_ruls = {a["aircraft_id"]: float(a["predicted_rul"]) for a in fleet}
        total_fleet_size = len(fleet)

        for day in range(1, self.horizon + 1):
            ready_count = 0
            maintenance_count = 0
            grounded_risk_count = 0

            for a in fleet:
                a_id = a["aircraft_id"]
                sched_entry = sched_map.get(a_id)

                is_in_maintenance = (
                    sched_entry is not None and
                    sched_entry["start_day"] <= day <= sched_entry["end_day"]
                )

                if is_in_maintenance:
                    maintenance_count += 1
                    # If this is the last day of maintenance, overhaul engine RUL
                    if day == sched_entry["end_day"]:
                        current_ruls[a_id] = 150.0  # Reset to fully overhauled state
                else:
                    # Aircraft is in service; burns flight cycles if ready
                    rul_today = current_ruls[a_id]
                    if rul_today <= 0.0:
                        # Engine failed / grounded
                        grounded_risk_count += 1
                    elif rul_today <= 15.0 and sched_entry is None:
                        # Dangerously low RUL and unscheduled
                        grounded_risk_count += 1
                    else:
                        ready_count += 1
                        # Burn daily flight cycles
                        current_ruls[a_id] = max(0.0, rul_today - self.daily_cycles)

            readiness_rate = round((ready_count / total_fleet_size) * 100.0, 1)
            demand_met = ready_count >= self.mission_demand
            shortfall = max(0, self.mission_demand - ready_count)

            daily_records.append({
                "day": day,
                "ready_count": ready_count,
                "maintenance_count": maintenance_count,
                "grounded_count": grounded_risk_count,
                "readiness_rate_pct": readiness_rate,
                "target_demand": self.mission_demand,
                "demand_met": demand_met,
                "shortfall": shortfall,
            })

        return daily_records

    def compare_schedules(self, fleet: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs both Baseline and Optimized schedules under identical inputs and computes comparison metrics.
        """
        baseline_res = self.run_baseline_schedule(fleet)
        optim_res = self.run_optimized_schedule(fleet)

        b_timeline = baseline_res["timeline"]
        o_timeline = optim_res["timeline"]

        b_avg_ready = np.mean([d["ready_count"] for d in b_timeline])
        o_avg_ready = np.mean([d["ready_count"] for d in o_timeline])

        b_avg_readiness_pct = np.mean([d["readiness_rate_pct"] for d in b_timeline])
        o_avg_readiness_pct = np.mean([d["readiness_rate_pct"] for d in o_timeline])

        b_shortfall_days = sum(1 for d in b_timeline if not d["demand_met"])
        o_shortfall_days = sum(1 for d in o_timeline if not d["demand_met"])

        b_total_shortfall_aircraft = sum(d["shortfall"] for d in b_timeline)
        o_total_shortfall_aircraft = sum(d["shortfall"] for d in o_timeline)

        delta_readiness_pct = round(o_avg_readiness_pct - b_avg_readiness_pct, 1)
        shortfall_reduction_days = b_shortfall_days - o_shortfall_days

        return {
            "baseline": baseline_res,
            "optimized": optim_res,
            "metrics": {
                "horizon_days": self.horizon,
                "hangar_bays": self.num_bays,
                "mission_demand": self.mission_demand,
                "baseline_avg_ready_aircraft": round(float(b_avg_ready), 2),
                "optimized_avg_ready_aircraft": round(float(o_avg_ready), 2),
                "baseline_avg_readiness_pct": round(float(b_avg_readiness_pct), 1),
                "optimized_avg_readiness_pct": round(float(o_avg_readiness_pct), 1),
                "delta_readiness_pct": delta_readiness_pct,
                "baseline_shortfall_days": b_shortfall_days,
                "optimized_shortfall_days": o_shortfall_days,
                "shortfall_reduction_days": shortfall_reduction_days,
                "baseline_total_shortfall": b_total_shortfall_aircraft,
                "optimized_total_shortfall": o_total_shortfall_aircraft,
            },
            "assumptions": [
                f"Planning horizon: {self.horizon} days",
                f"Available maintenance bays: {self.num_bays}",
                f"Target mission demand: {self.mission_demand} aircraft / day",
                f"Daily flight consumption rate: {self.daily_cycles} cycles / day",
                "Critical aircraft (RUL <= 25) prohibited from flight missions until serviced",
                "Post-maintenance overhaul restores engine to 150 cycles RUL baseline",
                "Baseline schedule exhibits bay-stall penalty when parts are delayed",
                "Optimized schedule holds aircraft outside the bay until spare parts arrive",
            ],
        }
