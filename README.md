# AeroReady AI — Predictive Maintenance & Fleet Availability Optimizer

> **Student Innovation Project Prototype**  
> An end-to-end AI/ML and operations research prototype combining Remaining Useful Life (RUL) estimation from NASA turbofan sensor telemetry with capacity-constrained fleet maintenance scheduling.

---

## 📌 Executive Summary

**AeroReady AI** addresses two interdependent challenges in aerospace operations:
1. **Predictive Maintenance (ML Prognostics):** Predicting how many flight cycles remain before an engine reaches failure threshold using multivariate sensor telemetry from the **NASA C-MAPSS FD001** turbofan degradation dataset.
2. **Fleet Availability Optimization (Operations Scheduling):** Using predicted RUL, maintenance durations, limited hangar bay capacity, spare-part logistics lead times, and mission criticality tiers to generate an optimal maintenance schedule that maximizes mission-ready aircraft.

> ⚠️ **Educational & Prototype Disclaimer:**  
> This software is an engineering student demonstrator built for educational purposes. It is **not** certified for real-world aircraft airworthiness, FAA/EASA compliance, or operational flight safety.

---

## 🏗️ Architecture & System Design

```
aeroready-ai/
│
├── app.py                      # Interactive Streamlit Web Application (Plotly visualizations)
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive setup and user guide
│
├── data/
│   ├── README.md               # Dataset documentation, column schema, and download sources
│   ├── train_FD001.txt         # 100 run-to-failure engine trajectories (20,631 records)
│   ├── test_FD001.txt          # 100 test trajectories ending prior to failure (13,096 records)
│   ├── RUL_FD001.txt           # True remaining cycles for the 100 test engines
│   └── readme.txt              # Official NASA C-MAPSS dataset documentation
│
├── models/
│   ├── rul_rf_model.joblib     # Trained RandomForestRegressor model
│   ├── preprocessor.joblib     # Fitted CMapssPreprocessor (rolling features & zero-variance filter)
│   ├── metrics.json            # Honest performance metrics (RMSE, MAE, R², NASA score)
│   └── test_predictions.csv    # Per-engine test set predictions & residuals
│
├── src/
│   ├── __init__.py             # Module initialization
│   ├── data_loader.py          # NASA C-MAPSS data ingestion & schema validation
│   ├── preprocessing.py       # Engine-wise train/val split, RUL labeling, feature engineering
│   ├── evaluation.py           # RMSE, MAE, R², and NASA asymmetric scoring functions
│   ├── train_model.py          # Full model training and evaluation pipeline
│   ├── predict_rul.py          # Inference class & risk categorization (Critical / Warning / Healthy)
│   ├── fleet_simulator.py      # 10-aircraft fictional fleet with model-derived telemetry
│   └── scheduler.py            # Baseline (FIFO) vs Optimized capacity-constrained scheduler
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # RUL label tests, leakage validation, variance filter
    └── test_scheduler.py       # Hangar capacity tests, spare-part constraints, comparison tests
```

---

## 🔬 Machine Learning Pipeline

### 1. Dataset & Ground-Truth RUL Formula
- **Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) subset **FD001**.
- **Conditions:** Sea Level, Single Fault Mode (High-Pressure Compressor Degradation).
- **Target Label Formulation:** For each training engine $i$ at operating flight cycle $t$:
  $$\text{RUL}_{i,t} = \max_{k}(\text{cycle}_{i,k}) - t$$
- **Engine-Wise Partitioning:** Engines are split by engine unit ID (80 training engines, 20 validation engines). Individual flight cycles from the same engine are never randomly shuffled into both splits, preventing data leakage.

### 2. Feature Engineering
- **Active Channel Selection:** Drops constant channels (`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) that have zero variance in FD001.
- **Temporal Dynamics:** Computes 5-cycle rolling means and rolling standard deviations per engine, capturing the rate of thermomechanical degradation.

### 3. Model Architecture & Evaluated Performance
- **Estimator:** Scikit-Learn `RandomForestRegressor` ($100$ estimators, `max_depth=15`, `min_samples_split=5`).
- **Honest Test Results (Official NASA Test Set — 100 Test Engines):**
  - **Test Root Mean Squared Error (RMSE):** **35.53 flight cycles**
  - **Test Mean Absolute Error (MAE):** **25.62 flight cycles**
  - **Validation Engine Set $R^2$:** **0.764**
  - **NASA PHM 2008 Asymmetric Penalty Score:** **124,561**
- **Top Driving Sensor Features:**
  1. `sensor_2_roll_mean_5` (LPC Outlet Temperature)
  2. `sensor_4_roll_mean_5` (LPT Outlet Temperature)
  3. `sensor_15_roll_mean_5` (Bypass Ratio)
  4. `sensor_11_roll_mean_5` (HPC Static Pressure)
  5. `cycle` (Accumulated operating flight hours)

---

## ⚙️ Maintenance Scheduling Optimizer

The scheduling engine compares a **Baseline Reactive Schedule** against the **AeroReady Priority & Capacity Optimizer** over a configurable planning horizon (default 14 days):

### Mathematical Formulation
$$\text{Maximize} \quad \frac{1}{H} \sum_{t=1}^{H} \text{ReadyAircraft}(t)$$
Subject to:
1. **Hangar Bay Capacity:** Concurrent aircraft in maintenance on day $t \le B$ (e.g. $B = 2$ bays).
2. **Spare Parts Lead Time:** Aircraft $i$ cannot enter maintenance until $t \ge \text{LeadDays}_i + 1$.
3. **Flight Restriction:** Aircraft with Critical RUL ($\le 25$ cycles) are grounded until serviced.
4. **Post-Maintenance Restoration:** Once maintenance is completed, engine RUL is restored to nominal baseline ($150$ cycles).

### Comparison Results (Simulated 10-Aircraft Fleet, 14-Day Horizon)
- **Baseline Reactive Schedule:** Prone to bay-stall penalties when spare parts are delayed, resulting in average fleet readiness of **~65.7%** and multiple mission deficit days.
- **AeroReady Optimized Schedule:** Levels hangar load, resolves part arrival bottlenecks, and delivers **~75.0%** average readiness (+**9.3% net readiness gain**), cutting mission shortfall days by **50%**.

---

## 🚀 Step-by-Step Windows PowerShell Setup Guide

Follow these exact commands to run the project on Windows:

### Step 1: Open PowerShell and Navigate to Project Directory
```powershell
cd "c:\Users\dell\OneDrive\Documents\JN\all syllabussss\ps12\aeroready-ai"
```

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```powershell
# Create virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify the NASA Dataset
The dataset files are located in `aeroready-ai/data/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

*(If missing, follow instructions in `data/README.md`)*

### Step 5: Train the ML Model
Train the Random Forest model and save artifacts:
```powershell
python src/train_model.py
```
*Output will display engine split details, training time, validation metrics, and official NASA test set results.*

### Step 6: Run Unit Tests
Execute the automated test suite with pytest:
```powershell
pytest -v
```
*Expected: 6 passed tests validating RUL labels, zero data leakage, and scheduler capacity constraints.*

### Step 7: Launch the Interactive Dashboard
Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🖥️ Dashboard Tour & Features

| Page | Key Capabilities |
| :--- | :--- |
| **1. Fleet Overview** | Executive readiness KPIs, donut health breakdown, RUL comparison bar chart, and current fleet status table. |
| **2. Engine Health** | Aircraft selector, real-time RUL gauge, technical explanation, and 4 sensor degradation line plots (LPC Temp, HPC Pressure, HPC Temp, Bypass Ratio). |
| **3. Predictive Maintenance** | Ranked urgency roster, immediate action cards for high-risk aircraft, and spare-part supply chain status. |
| **4. Maintenance Planner** | Side-by-side Baseline vs Optimized schedule, hangar bay Gantt chart, simulated availability curves, and demand compliance metrics. |
| **5. What-If Simulator** | Interactive sliders for available bays (1-5), duration multipliers (0.5x-2x), planning horizon, flight tempo, and spare parts surge override. |
| **6. Model Evaluation** | Honest test set metrics (RMSE, MAE, R², NASA score), Actual vs Predicted scatter plot, residual error histogram, and Gini feature importances. |

---

## 📜 Citations

- A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
