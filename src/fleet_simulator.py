"""Simulate a fictional fleet using predictions on NASA test-engine telemetry."""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

import sys
from pathlib import Path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
for p in [str(_project_root), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.predict_rul import RULPredictor, categorize_rul, get_default_subset
    from src.data_loader import load_raw_data, get_default_data_dir
    from src.preprocessing import extract_test_last_cycles
except ImportError:
    from predict_rul import RULPredictor, categorize_rul, get_default_subset
    from data_loader import load_raw_data, get_default_data_dir
    from preprocessing import extract_test_last_cycles

# Standard synthetic aircraft profiles for the 10 fictional aircraft
FLEET_PROFILES = [
    {
        "aircraft_id": "AERO-101",
        "tail_number": "N101AR",
        "model_type": "F-22B Raptor Sim",
        "mission_priority": "Tier 1 - Quick Reaction",
        "priority_rank": 1,
        "default_duration_days": 3,
        "spare_part_name": "HPC Stage-3 Rotor Blisk",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-102",
        "tail_number": "N102AR",
        "model_type": "F-22B Raptor Sim",
        "mission_priority": "Tier 1 - Quick Reaction",
        "priority_rank": 1,
        "default_duration_days": 4,
        "spare_part_name": "Turbine Shroud Seal Kit",
        "spare_parts_available": False,
        "spare_part_lead_days": 2,
    },
    {
        "aircraft_id": "AERO-103",
        "tail_number": "N103AR",
        "model_type": "C-130J Hercules Sim",
        "mission_priority": "Tier 2 - Tactical Transport",
        "priority_rank": 2,
        "default_duration_days": 3,
        "spare_part_name": "Fuel Metering Valve Assy",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-104",
        "tail_number": "N104AR",
        "model_type": "C-130J Hercules Sim",
        "mission_priority": "Tier 2 - Tactical Transport",
        "priority_rank": 2,
        "default_duration_days": 2,
        "spare_part_name": "LPC Stator Vane Bushing",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-105",
        "tail_number": "N105AR",
        "model_type": "P-8A Poseidon Sim",
        "mission_priority": "Tier 2 - Maritime Recon",
        "priority_rank": 2,
        "default_duration_days": 3,
        "spare_part_name": "Bleed Enthalpy Sensor Pack",
        "spare_parts_available": False,
        "spare_part_lead_days": 3,
    },
    {
        "aircraft_id": "AERO-106",
        "tail_number": "N106AR",
        "model_type": "P-8A Poseidon Sim",
        "mission_priority": "Tier 3 - Routine Patrol",
        "priority_rank": 3,
        "default_duration_days": 2,
        "spare_part_name": "Exhaust Gas Thermocouple",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-107",
        "tail_number": "N107AR",
        "model_type": "T-38 Talon Sim",
        "mission_priority": "Tier 3 - Advanced Training",
        "priority_rank": 3,
        "default_duration_days": 2,
        "spare_part_name": "Oil Scavenge Pump Filter",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-108",
        "tail_number": "N108AR",
        "model_type": "T-38 Talon Sim",
        "mission_priority": "Tier 3 - Advanced Training",
        "priority_rank": 3,
        "default_duration_days": 2,
        "spare_part_name": "Ignition Exciter Lead",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-109",
        "tail_number": "N109AR",
        "model_type": "KC-46A Pegasus Sim",
        "mission_priority": "Tier 1 - Strategic Tanker",
        "priority_rank": 1,
        "default_duration_days": 3,
        "spare_part_name": "Fan Blade Containment Seal",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
    {
        "aircraft_id": "AERO-110",
        "tail_number": "N110AR",
        "model_type": "KC-46A Pegasus Sim",
        "mission_priority": "Tier 1 - Strategic Tanker",
        "priority_rank": 1,
        "default_duration_days": 2,
        "spare_part_name": "Starter Generator Brushes",
        "spare_parts_available": True,
        "spare_part_lead_days": 0,
    },
]


class FleetSimulator:
    """
    Simulates a fictional fleet using selected-subset test-engine telemetry.
    """

    def __init__(
        self,
        predictor: Optional[RULPredictor] = None,
        subset: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        self.subset = subset or getattr(predictor, "subset", None) or get_default_subset()
        self.predictor = predictor or RULPredictor(subset=self.subset)
        if self.predictor.subset != self.subset:
            raise ValueError("Fleet subset must match the predictor's trained subset.")
        self.data_dir = data_dir or get_default_data_dir()
        self.test_data_df = None
        self.test_snapshots_df = None
        self._load_test_snapshots()

    def _load_test_snapshots(self):
        """Load test histories and their final observations for the selected subset."""
        _, self.test_data_df, _ = load_raw_data(self.data_dir, subset=self.subset)
        self.test_snapshots_df = extract_test_last_cycles(self.test_data_df).set_index("engine_id")

    def generate_fleet(
        self,
        duration_multiplier: float = 1.0,
        override_parts_all_available: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generates current state for the 10 fictional aircraft.

        Args:
            duration_multiplier: Scales standard maintenance duration
            override_parts_all_available: If True, forces all parts to be immediately available

        Returns:
            List of 10 aircraft state dictionaries
        """
        if not self.predictor.is_loaded:
            raise RuntimeError(f"Model artifacts for {self.subset} are not loaded.")
        if self.test_snapshots_df is None or self.test_snapshots_df.empty:
            raise RuntimeError(f"No test-engine telemetry is available for {self.subset}.")

        fleet = []
        test_engine_ids = self.test_snapshots_df.index.tolist()
        for profile_index, profile in enumerate(FLEET_PROFILES):
            aircraft_id = profile["aircraft_id"]
            eng_id = test_engine_ids[profile_index % len(test_engine_ids)]
            eng_history = self.test_data_df[
                self.test_data_df["engine_id"] == eng_id
            ].sort_values("cycle")
            pred = self.predictor.predict_dataframe(eng_history)[0]
            predicted_rul = float(round(pred, 1))
            data_source = f"Model-Derived ({self.subset} Engine #{eng_id})"

            category, badge_color, explanation = categorize_rul(predicted_rul)

            # Maintenance duration
            base_duration = profile["default_duration_days"]
            maintenance_duration = max(1, int(round(base_duration * duration_multiplier)))

            # Parts availability
            parts_avail = True if override_parts_all_available else profile["spare_parts_available"]
            parts_lead = 0 if parts_avail else profile["spare_part_lead_days"]

            # Operational status logic:
            # - Critical (RUL <= 25): Grounded for safety; urgent maintenance needed
            # - Warning (25 < RUL <= 60): Currently in service, but flagged for scheduling
            # - Healthy (RUL > 60): In service, full availability
            if category == "Critical":
                is_available = False
                status = "Grounded - Urgent Maintenance"
            elif category == "Warning":
                is_available = True
                status = "In Service (Maintenance Required Soon)"
            else:
                is_available = True
                status = "In Service (Mission Ready)"

            aircraft_record = {
                "aircraft_id": aircraft_id,
                "tail_number": profile["tail_number"],
                "model_type": profile["model_type"],
                "assigned_engine_id": eng_id,
                "predicted_rul": predicted_rul,
                "health_category": category,
                "badge_color": badge_color,
                "explanation": explanation,
                "maintenance_duration": maintenance_duration,
                "maintenance_status": status,
                "spare_part_name": profile["spare_part_name"],
                "spare_parts_available": parts_avail,
                "spare_part_lead_days": parts_lead,
                "mission_priority": profile["mission_priority"],
                "priority_rank": profile["priority_rank"],
                "is_currently_available": is_available,
                "data_source": data_source,
                "is_simulated": True,
            }
            fleet.append(aircraft_record)

        return fleet

    def get_fleet_summary(self, fleet: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates executive KPI metrics for the fleet."""
        total = len(fleet)
        available = sum(1 for a in fleet if a["is_currently_available"])
        critical = sum(1 for a in fleet if a["health_category"] == "Critical")
        warning = sum(1 for a in fleet if a["health_category"] == "Warning")
        healthy = sum(1 for a in fleet if a["health_category"] == "Healthy")
        availability_pct = round((available / total) * 100.0, 1) if total > 0 else 0.0

        model_derived_count = sum(1 for a in fleet if "Model-Derived" in a["data_source"])

        return {
            "total_aircraft": total,
            "available_aircraft": available,
            "grounded_or_maintenance": total - available,
            "critical_count": critical,
            "warning_count": warning,
            "healthy_count": healthy,
            "availability_pct": availability_pct,
            "model_derived_count": model_derived_count,
            "is_fully_model_derived": (model_derived_count == total),
        }
