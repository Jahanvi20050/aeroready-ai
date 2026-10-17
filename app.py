"""
AeroReady AI — Predictive Maintenance & Fleet Availability Optimizer
Streamlit Interactive Dashboard
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup sys.path for local imports
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from src.data_loader import check_dataset_exists, get_default_data_dir, load_raw_data
from src.predict_rul import RULPredictor, categorize_rul
from src.fleet_simulator import FleetSimulator
from src.scheduler import MaintenanceScheduler

# Page Configuration
st.set_page_config(
    page_title="AeroReady AI | Fleet Maintenance Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Aviation Dark-Navy Theme Styling
CUSTOM_CSS = """
<style>
    /* Dark Navy Theme Foundations */
    .stApp {
        background-color: #070D1E;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .aero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .aero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .aero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.18);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-healthy {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Disclaimer Banner */
    .disclaimer-box {
        background-color: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0284C7;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0 24px 0;
        font-size: 0.85rem;
        color: #BAE6FD;
    }

    /* Section Cards */
    .content-box {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background-color: #090E1F;
        border-right: 1px solid #1E293B;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Shared Dark Theme Helper for Plotly Charts
def apply_dark_theme(fig, height=300):
    fig.update_layout(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0", family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=40, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        showline=True,
        linecolor="#334155",
    )
    fig.update_yaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        showline=True,
        linecolor="#334155",
    )
    return fig


@st.cache_resource
def get_cached_predictor():
    return RULPredictor()


@st.cache_data
def load_model_metrics():
    model_dir = _current_dir / "models"
    metrics_file = model_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data
def load_test_predictions_table():
    model_dir = _current_dir / "models"
    preds_file = model_dir / "test_predictions.csv"
    if preds_file.is_file():
        try:
            return pd.read_csv(preds_file)
        except Exception:
            return None
    return None


@st.cache_data
def get_raw_dataset_samples():
    try:
        train_df, test_df, rul_df = load_raw_data()
        return train_df, test_df, rul_df
    except Exception:
        return None, None, None


# Main Application Entry
def main():
    predictor = get_cached_predictor()
    simulator = FleetSimulator(predictor=predictor)
    metrics_data = load_model_metrics()

    # Sidebar Navigation & Global Controls
    with st.sidebar:
        st.markdown("### ✈️ AeroReady AI")
        st.caption("Predictive Maintenance & Fleet Availability Optimizer")

        nav_choice = st.radio(
            "Navigate Modules",
            [
                "Fleet Overview",
                "Engine Health",
                "Predictive Maintenance",
                "Maintenance Planner",
                "What-if Simulator",
                "Model Evaluation",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("#### System Status")
        if predictor.is_loaded:
            st.success("ML Engine: Online (FD001 Trained)")
            if metrics_data:
                st.caption(f"Test RMSE: **{metrics_data['test_metrics']['rmse']} cycles**")
        else:
            st.warning("ML Engine: Untrained (Synthetic Fallback)")
            st.caption("Run `python src/train_model.py` to train model on NASA FD001.")

        data_present, _ = check_dataset_exists()
        if data_present:
            st.info("NASA Dataset: Verified in `data/`")
        else:
            st.error("NASA Dataset: Missing from `data/`")

        st.markdown("---")
        st.caption("Student Innovation Project Prototype\n© 2026 AeroReady AI")

    # Top Header Banner
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">
                <span>✈️</span> AeroReady AI — Fleet Availability Optimizer
            </h1>
            <p class="aero-subtitle">
                Turbofan engine prognostic modeling & capacity-constrained maintenance scheduling
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # PAGE ROUTING
    if nav_choice == "Fleet Overview":
        render_fleet_overview(simulator, predictor)
    elif nav_choice == "Engine Health":
        render_engine_health(simulator, predictor)
    elif nav_choice == "Predictive Maintenance":
        render_predictive_maintenance(simulator)
    elif nav_choice == "Maintenance Planner":
        render_maintenance_planner(simulator)
    elif nav_choice == "What-if Simulator":
        render_what_if_simulator(simulator)
    elif nav_choice == "Model Evaluation":
        render_model_evaluation(metrics_data)


# -----------------------------------------------------------------------------
# 1. FLEET OVERVIEW
# -----------------------------------------------------------------------------
def render_fleet_overview(simulator: FleetSimulator, predictor: RULPredictor):
    st.markdown("### 1. Fleet Readiness & Status Overview")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Demonstration Notice:</strong> The simulated fleet contains 10 fictional aircraft. 
            When the ML model is trained, engine degradation data is sampled from the official NASA C-MAPSS FD001 test engines.
            This prototype is designed for engineering education and is not an operational flight safety system.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fleet = simulator.generate_fleet()
    summary = simulator.get_fleet_summary(fleet)

    # Executive KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Simulated Fleet</div>
                <div class="metric-value">{summary['total_aircraft']}</div>
                <div class="metric-sub">Fictional Aircraft Units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mission Available</div>
                <div class="metric-value" style="color: #10B981;">{summary['available_aircraft']}</div>
                <div class="metric-sub">{summary['availability_pct']}% of Fleet Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Requiring Attention</div>
                <div class="metric-value" style="color: #F59E0B;">{summary['grounded_or_maintenance']}</div>
                <div class="metric-sub">{summary['critical_count']} Critical | {summary['warning_count']} Warning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        source_label = "Trained ML Model" if summary["is_fully_model_derived"] else "Synthetic Fallback"
        source_color = "#38BDF8" if summary["is_fully_model_derived"] else "#94A3B8"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Origin</div>
                <div class="metric-value" style="color: {source_color}; font-size: 1.35rem; padding-top: 6px;">
                    {source_label}
                </div>
                <div class="metric-sub">{summary['model_derived_count']}/10 Engines ML-Derived</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Fleet Distributions
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Health Category Distribution")
        health_counts = {
            "Healthy (>60 cycles)": summary["healthy_count"],
            "Warning (26-60 cycles)": summary["warning_count"],
            "Critical (<=25 cycles)": summary["critical_count"],
        }
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=list(health_counts.keys()),
                    values=list(health_counts.values()),
                    hole=0.6,
                    marker=dict(colors=["#10B981", "#F59E0B", "#EF4444"]),
                    textinfo="label+value",
                )
            ]
        )
        apply_dark_theme(fig_donut, height=300)
        fig_donut.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("#### Predicted RUL by Aircraft")
        df_fleet = pd.DataFrame(fleet)
        color_map = {"Critical": "#EF4444", "Warning": "#F59E0B", "Healthy": "#10B981"}
        df_fleet["color"] = df_fleet["health_category"].map(color_map)

        fig_bar = px.bar(
            df_fleet,
            x="aircraft_id",
            y="predicted_rul",
            color="health_category",
            color_discrete_map=color_map,
            labels={"aircraft_id": "Aircraft ID", "predicted_rul": "Predicted RUL (cycles)"},
        )
        fig_bar.add_hline(y=25, line_dash="dash", line_color="#EF4444", annotation_text="Critical Limit (25)")
        fig_bar.add_hline(y=60, line_dash="dash", line_color="#F59E0B", annotation_text="Warning Limit (60)")
        apply_dark_theme(fig_bar, height=300)
        fig_bar.update_layout(showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Fleet Status Table
    st.markdown("#### Current Fleet Status Roster")
    table_display = df_fleet[
        [
            "aircraft_id",
            "tail_number",
            "model_type",
            "predicted_rul",
            "health_category",
            "maintenance_duration",
            "spare_parts_available",
            "mission_priority",
            "maintenance_status",
            "data_source",
        ]
    ].copy()
    table_display.columns = [
        "Aircraft ID",
        "Tail #",
        "Airframe",
        "RUL (Cycles)",
        "Risk Category",
        "Maint. Duration",
        "Parts In Stock",
        "Mission Tier",
        "Current Operational Status",
        "Source",
    ]
    st.dataframe(table_display, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 2. ENGINE HEALTH
# -----------------------------------------------------------------------------
def render_engine_health(simulator: FleetSimulator, predictor: RULPredictor):
    st.markdown("### 2. Engine Sensor Telemetry & Degradation Profiling")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Prognostic Evaluation:</strong> Sensor telemetry is extracted from the NASA C-MAPSS simulation dataset. 
            The ML model monitors thermomechanical degradation across compressor and turbine stages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fleet = simulator.generate_fleet()
    aircraft_options = {a["aircraft_id"]: a for a in fleet}

    selected_id = st.selectbox(
        "Select Aircraft to Inspect:",
        options=list(aircraft_options.keys()),
        format_func=lambda x: f"{x} ({aircraft_options[x]['tail_number']} - {aircraft_options[x]['model_type']}) - RUL: {aircraft_options[x]['predicted_rul']} cycles",
    )

    selected_ac = aircraft_options[selected_id]
    eng_id = selected_ac["assigned_engine_id"]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted RUL</div>
                <div class="metric-value" style="color: {selected_ac['badge_color']};">
                    {selected_ac['predicted_rul']} <span style="font-size: 1.1rem; color: #94A3B8;">cycles</span>
                </div>
                <div class="metric-sub">Category: <strong>{selected_ac['health_category']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Assigned NASA Engine</div>
                <div class="metric-value">Unit #{eng_id}</div>
                <div class="metric-sub">{selected_ac['data_source']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    parts_status_text = "In Stock" if selected_ac["spare_parts_available"] else f"Ordered ({selected_ac['spare_part_lead_days']}d lead)"
    parts_status_color = "#10B981" if selected_ac["spare_parts_available"] else "#EF4444"
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Spare Part Readiness</div>
                <div class="metric-value" style="font-size: 1.3rem; color: {parts_status_color};">
                    {parts_status_text}
                </div>
                <div class="metric-sub">{selected_ac['spare_part_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Technical Rationale Box
    st.info(f"**Operational Assessment:** {selected_ac['explanation']}")

    # Plot Engine Sensor History from NASA Dataset if available
    train_df, test_df, _ = get_raw_dataset_samples()
    if test_df is not None and eng_id in test_df["engine_id"].values:
        eng_history = test_df[test_df["engine_id"] == eng_id].sort_values("cycle")
        st.markdown(f"#### Sensor Trajectory for Engine #{eng_id} ({len(eng_history)} Flight Cycles Recorded)")

        # Key Informative C-MAPSS Sensors
        # sensor_2: LPC Outlet Temp (Ranked high in feature importance)
        # sensor_3: HPC Outlet Temp
        # sensor_7: HPC Outlet Pressure
        # sensor_15: Bypass Ratio
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            fig1 = px.line(
                eng_history,
                x="cycle",
                y="sensor_2",
                title="LPC Outlet Temperature (Sensor 2, °R)",
                labels={"cycle": "Operating Flight Cycle", "sensor_2": "Temp (°R)"},
            )
            fig1.update_traces(line_color="#38BDF8", line_width=2)
            apply_dark_theme(fig1, height=280)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.line(
                eng_history,
                x="cycle",
                y="sensor_7",
                title="HPC Outlet Static Pressure (Sensor 7, psia)",
                labels={"cycle": "Operating Flight Cycle", "sensor_7": "Pressure (psia)"},
            )
            fig2.update_traces(line_color="#F59E0B", line_width=2)
            apply_dark_theme(fig2, height=280)
            st.plotly_chart(fig2, use_container_width=True)

        with s_col2:
            fig3 = px.line(
                eng_history,
                x="cycle",
                y="sensor_3",
                title="HPC Outlet Total Temperature (Sensor 3, °R)",
                labels={"cycle": "Operating Flight Cycle", "sensor_3": "Temp (°R)"},
            )
            fig3.update_traces(line_color="#EF4444", line_width=2)
            apply_dark_theme(fig3, height=280)
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.line(
                eng_history,
                x="cycle",
                y="sensor_15",
                title="Bypass Ratio (Sensor 15)",
                labels={"cycle": "Operating Flight Cycle", "sensor_15": "Bypass Ratio"},
            )
            fig4.update_traces(line_color="#10B981", line_width=2)
            apply_dark_theme(fig4, height=280)
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning(
            "Detailed cycle trajectory not available for this synthetic engine profile. "
            "Please ensure NASA dataset files are present in `data/`."
        )


# -----------------------------------------------------------------------------
# 3. PREDICTIVE MAINTENANCE
# -----------------------------------------------------------------------------
def render_predictive_maintenance(simulator: FleetSimulator):
    st.markdown("### 3. Predictive Maintenance Urgency Ranking")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Urgency Prioritization:</strong> Aircraft are ranked using a multi-criteria scoring function 
            combining Remaining Useful Life (RUL), mission criticality tier, and spare part logistics readiness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fleet = simulator.generate_fleet()
    scheduler = MaintenanceScheduler()

    # Enrich with urgency scores
    ranked_fleet = []
    for a in fleet:
        score = scheduler._calculate_priority_score(a)
        a_copy = dict(a)
        a_copy["priority_score"] = score
        ranked_fleet.append(a_copy)

    # Sort descending by priority score
    ranked_fleet.sort(key=lambda x: x["priority_score"], reverse=True)

    # Display Top 3 Urgent Actions
    top_actions = [a for a in ranked_fleet if a["health_category"] in ["Critical", "Warning"]][:3]
    if top_actions:
        st.markdown("#### Immediate Maintenance Action Items")
        act_cols = st.columns(len(top_actions))
        for idx, ac in enumerate(top_actions):
            with act_cols[idx]:
                parts_lead_txt = "Ready in Hangar" if ac["spare_parts_available"] else f"Delayed ({ac['spare_part_lead_days']}d lead)"
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 4px solid {ac['badge_color']};">
                        <div class="metric-label">{ac['aircraft_id']} • {ac['mission_priority']}</div>
                        <div class="metric-value" style="color: {ac['badge_color']}; font-size: 1.5rem;">
                            RUL: {ac['predicted_rul']} cycles
                        </div>
                        <div style="font-size: 0.85rem; color: #CBD5E1; margin: 6px 0;">
                            <strong>Part:</strong> {ac['spare_part_name']}<br>
                            <strong>Status:</strong> {parts_lead_txt}
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">
                            Urgency Score: <strong>{ac['priority_score']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Comprehensive Prioritized Maintenance Roster")

    df_ranked = pd.DataFrame(ranked_fleet)
    display_df = df_ranked[
        [
            "aircraft_id",
            "tail_number",
            "priority_score",
            "predicted_rul",
            "health_category",
            "mission_priority",
            "maintenance_duration",
            "spare_part_name",
            "spare_parts_available",
            "maintenance_status",
        ]
    ].copy()
    display_df.columns = [
        "Aircraft ID",
        "Tail #",
        "Urgency Score",
        "RUL (Cycles)",
        "Risk Category",
        "Mission Tier",
        "Est. Days",
        "Required Spare Part",
        "Parts In Stock",
        "Current Status",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 4. MAINTENANCE PLANNER
# -----------------------------------------------------------------------------
def render_maintenance_planner(simulator: FleetSimulator):
    st.markdown("### 4. Maintenance Schedule Optimizer & Capacity Planner")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Head-to-Head Comparison:</strong> Measures the performance of a naive reactive (FIFO) 
            schedule against the AeroReady AI priority & capacity-leveled schedule under identical operational conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        bays = st.slider("Hangar Bay Capacity:", min_value=1, max_value=4, value=2, step=1)
    with col_ctrl2:
        horizon = st.slider("Planning Horizon (Days):", min_value=7, max_value=28, value=14, step=1)
    with col_ctrl3:
        demand = st.slider("Target Daily Mission Demand:", min_value=4, max_value=9, value=7, step=1)

    scheduler = MaintenanceScheduler(
        planning_horizon_days=horizon,
        num_hangar_bays=bays,
        mission_demand_aircraft=demand,
    )

    fleet = simulator.generate_fleet()
    comp = scheduler.compare_schedules(fleet)
    metrics = comp["metrics"]

    # Comparative KPI Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Baseline Avg Readiness</div>
                <div class="metric-value" style="color: #94A3B8;">{metrics['baseline_avg_readiness_pct']}%</div>
                <div class="metric-sub">{metrics['baseline_avg_ready_aircraft']} aircraft / day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Optimized Avg Readiness</div>
                <div class="metric-value" style="color: #10B981;">{metrics['optimized_avg_readiness_pct']}%</div>
                <div class="metric-sub">{metrics['optimized_avg_ready_aircraft']} aircraft / day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        delta_color = "#10B981" if metrics["delta_readiness_pct"] >= 0 else "#EF4444"
        delta_sign = "+" if metrics["delta_readiness_pct"] > 0 else ""
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Readiness Improvement</div>
                <div class="metric-value" style="color: {delta_color};">
                    {delta_sign}{metrics['delta_readiness_pct']}%
                </div>
                <div class="metric-sub">Net Fleet Availability Gain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mission Shortfall Days</div>
                <div class="metric-value" style="color: #38BDF8;">
                    {metrics['optimized_shortfall_days']} <span style="font-size: 1rem; color: #94A3B8;">vs {metrics['baseline_shortfall_days']} base</span>
                </div>
                <div class="metric-sub">Reduced by {metrics['shortfall_reduction_days']} days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Availability Curve Comparison
    st.markdown("#### Simulated Fleet Availability Over Time (Daily Ready Aircraft)")
    b_timeline = pd.DataFrame(comp["baseline"]["timeline"])
    o_timeline = pd.DataFrame(comp["optimized"]["timeline"])

    fig_timeline = go.Figure()
    fig_timeline.add_trace(
        go.Scatter(
            x=b_timeline["day"],
            y=b_timeline["ready_count"],
            name="Baseline Schedule (Reactive)",
            line=dict(color="#EF4444", width=2, dash="dot"),
            mode="lines+markers",
        )
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=o_timeline["day"],
            y=o_timeline["ready_count"],
            name="AeroReady Optimized Schedule",
            line=dict(color="#10B981", width=3),
            mode="lines+markers",
        )
    )
    fig_timeline.add_hline(
        y=demand,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Mission Demand ({demand} Aircraft)",
    )
    apply_dark_theme(fig_timeline, height=320)
    fig_timeline.update_layout(
        xaxis_title="Simulation Day",
        yaxis_title="Ready Aircraft Count",
    )
    fig_timeline.update_yaxes(range=[0, 11])
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Gantt Timeline of Hangar Bay Schedule
    st.markdown("#### Optimized Hangar Bay Gantt Timeline")
    optim_sched = comp["optimized"]["schedule"]
    if optim_sched:
        gantt_rows = []
        for item in optim_sched:
            gantt_rows.append(
                dict(
                    Aircraft=item["aircraft_id"],
                    Bay=item["assigned_bay"],
                    Start=item["start_day"],
                    End=item["end_day"],
                    Duration=item["duration_days"],
                    Priority=item["mission_priority"],
                )
            )
        gantt_df = pd.DataFrame(gantt_rows)

        fig_gantt = px.bar(
            gantt_df,
            base="Start",
            x="Duration",
            y="Bay",
            color="Aircraft",
            text="Aircraft",
            orientation="h",
            title=f"Bay Allocations across {horizon}-Day Horizon (Capacity: {bays} Bays)",
            labels={"Duration": "Maintenance Days", "Bay": "Hangar Bay"},
        )
        apply_dark_theme(fig_gantt, height=280)
        fig_gantt.update_xaxes(
            range=[1, horizon + 1],
            tickmode="linear",
            tick0=1,
            dtick=1,
            title="Horizon Day",
        )
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.info("No aircraft required maintenance scheduling within this window.")

    # Assumptions & Constraints
    st.markdown("#### Mathematical Assumptions & Operational Constraints")
    for asm in comp["assumptions"]:
        st.markdown(f"- {asm}")


# -----------------------------------------------------------------------------
# 5. WHAT-IF SIMULATOR
# -----------------------------------------------------------------------------
def render_what_if_simulator(simulator: FleetSimulator):
    st.markdown("### 5. Interactive What-If Scenario Simulator")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Interactive Sensitivity Analysis:</strong> Adjust operational resources and constraints in real-time. 
            The optimizer recalculates bay assignments, resolves spare-part bottlenecks, and re-evaluates mission readiness instantly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_bays = st.slider("Hangar Bays Available:", min_value=1, max_value=5, value=2, step=1, key="sim_bays")
        sim_duration_mult = st.select_slider(
            "Maintenance Duration Multiplier:",
            options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            value=1.0,
            key="sim_dur",
        )
    with c2:
        sim_demand = st.slider("Mission Aircraft Demand:", min_value=3, max_value=10, value=7, step=1, key="sim_dem")
        sim_horizon = st.slider("Planning Horizon (Days):", min_value=7, max_value=30, value=14, step=1, key="sim_hor")
    with c3:
        parts_override = st.checkbox(
            "Expedite All Spare Parts (Zero Lead Time)",
            value=False,
            help="Simulates an emergency logistics surge where all parts are immediately in stock.",
        )
        daily_flight_rate = st.slider(
            "Flight Operations Tempo (Cycles / Day):",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5,
        )

    fleet = simulator.generate_fleet(
        duration_multiplier=sim_duration_mult,
        override_parts_all_available=parts_override,
    )
    scheduler = MaintenanceScheduler(
        planning_horizon_days=sim_horizon,
        num_hangar_bays=sim_bays,
        mission_demand_aircraft=sim_demand,
        daily_flight_cycles=daily_flight_rate,
    )

    sim_comp = scheduler.compare_schedules(fleet)
    m = sim_comp["metrics"]

    # Dynamic KPI Readouts
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Optimized Readiness Rate",
            value=f"{m['optimized_avg_readiness_pct']}%",
            delta=f"{m['delta_readiness_pct']}% vs Base",
        )
    with k2:
        st.metric(
            label="Avg Ready Aircraft",
            value=f"{m['optimized_avg_ready_aircraft']} / 10",
            delta=f"{round(m['optimized_avg_ready_aircraft'] - m['baseline_avg_ready_aircraft'], 1)} vs Base",
        )
    with k3:
        st.metric(
            label="Shortfall Days",
            value=f"{m['optimized_shortfall_days']} days",
            delta=f"-{m['shortfall_reduction_days']} days vs Base",
            delta_color="inverse",
        )
    with k4:
        demand_status = "Achieved" if m["optimized_shortfall_days"] == 0 else f"{m['optimized_shortfall_days']}d Deficit"
        st.metric(label="Demand Compliance", value=demand_status)

    # Availability Timeline
    df_opt_tl = pd.DataFrame(sim_comp["optimized"]["timeline"])
    df_base_tl = pd.DataFrame(sim_comp["baseline"]["timeline"])

    fig_sim = go.Figure()
    fig_sim.add_trace(
        go.Scatter(
            x=df_base_tl["day"],
            y=df_base_tl["ready_count"],
            name="Baseline Schedule",
            line=dict(color="#EF4444", dash="dash"),
        )
    )
    fig_sim.add_trace(
        go.Scatter(
            x=df_opt_tl["day"],
            y=df_opt_tl["ready_count"],
            name="AeroReady Optimized",
            line=dict(color="#38BDF8", width=3),
        )
    )
    fig_sim.add_hline(
        y=sim_demand,
        line_color="#F59E0B",
        line_dash="dot",
        annotation_text=f"Target: {sim_demand} Aircraft",
    )
    apply_dark_theme(fig_sim, height=320)
    fig_sim.update_layout(
        title="Simulated Fleet Availability Curve",
        xaxis_title="Day",
        yaxis_title="Ready Aircraft",
    )
    fig_sim.update_yaxes(range=[0, 11])
    st.plotly_chart(fig_sim, use_container_width=True)


# -----------------------------------------------------------------------------
# 6. MODEL EVALUATION
# -----------------------------------------------------------------------------
def render_model_evaluation(metrics_data: dict):
    st.markdown("### 6. Machine Learning Model Performance & Diagnostics")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Honest Evaluation Policy:</strong> All performance figures shown are computed strictly from the 
            RandomForestRegressor model trained on the NASA C-MAPSS FD001 dataset without synthetic extrapolation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not metrics_data:
        st.warning(
            "⚠️ No trained model metrics found. Please train the baseline model first by running:\n\n"
            "```powershell\npython src/train_model.py\n```"
        )
        return

    test_m = metrics_data["test_metrics"]
    val_m = metrics_data["validation_metrics"]
    params = metrics_data.get("parameters", {})

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">NASA Test Set RMSE</div>
                <div class="metric-value" style="color: #38BDF8;">{test_m['rmse']}</div>
                <div class="metric-sub">Operating Cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">NASA Test Set MAE</div>
                <div class="metric-value" style="color: #10B981;">{test_m['mae']}</div>
                <div class="metric-sub">Average Absolute Error (Cycles)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Validation Set R²</div>
                <div class="metric-value" style="color: #F59E0B;">{val_m['r2']}</div>
                <div class="metric-sub">Engine-wise Split (No Leakage)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">NASA PHM Score</div>
                <div class="metric-value" style="color: #F8FAFC; font-size: 1.4rem; padding-top: 5px;">
                    {test_m['nasa_score']:,.0f}
                </div>
                <div class="metric-sub">Asymmetric Penalty Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted Plot & Residual Distribution
    preds_df = load_test_predictions_table()
    if preds_df is not None:
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.markdown("#### Actual vs. Predicted RUL (100 NASA Test Engines)")
            fig_scatter = px.scatter(
                preds_df,
                x="actual_rul",
                y="predicted_rul",
                hover_data=["engine_id", "error"],
                labels={"actual_rul": "Ground Truth RUL (Cycles)", "predicted_rul": "Model Predicted RUL (Cycles)"},
            )
            # Add ideal 45-degree reference line
            max_val = max(preds_df["actual_rul"].max(), preds_df["predicted_rul"].max()) + 10
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name="Ideal Perfect Fit (y=x)",
                    line=dict(color="#EF4444", dash="dash"),
                )
            )
            fig_scatter.update_traces(marker=dict(size=7, color="#38BDF8", opacity=0.85))
            apply_dark_theme(fig_scatter, height=340)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with plot_col2:
            st.markdown("#### Prediction Error Distribution (Residuals)")
            fig_hist = px.histogram(
                preds_df,
                x="error",
                nbins=25,
                labels={"error": "Prediction Error (Predicted - Actual Cycles)"},
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#10B981", annotation_text="Zero Error")
            fig_hist.update_traces(marker_color="#38BDF8")
            apply_dark_theme(fig_hist, height=340)
            fig_hist.update_layout(yaxis_title="Engine Count")
            st.plotly_chart(fig_hist, use_container_width=True)

    # Feature Importance
    top_features = metrics_data.get("top_features", [])
    if top_features:
        st.markdown("#### Top Sensor Features Driving RUL Predictions")
        feat_df = pd.DataFrame(top_features)
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Gini Feature Importance", "feature": "Sensor Feature"},
        )
        fig_feat.update_traces(marker_color="#38BDF8")
        apply_dark_theme(fig_feat, height=320)
        fig_feat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_feat, use_container_width=True)

    # Limitations & Academic Context
    st.markdown("#### Dataset & Model Limitations")
    st.markdown(
        """
        - **Synthetic Simulation Bias:** The C-MAPSS dataset was synthetically generated using a thermodynamic engine model, which may exhibit less noise than real-world flight operations.
        - **Single Operating Condition (FD001):** FD001 tests sea-level conditions with a single failure mode (HPC degradation). Higher subsets (FD002/FD004) incorporate 6 flight conditions and multiple fault modes.
        - **Piecewise Degradation Plateau:** Aircraft engines operate nominally for a prolonged initial period before observable physical degradation commences; advanced piece-wise clipping (e.g. at 125 cycles) can further refine early-life predictions.
        - **Student Educational Scope:** This software is an engineering student demonstrator and is not certified for aviation airworthiness or FAA/EASA flight compliance.
        """
    )


if __name__ == "__main__":
    main()
