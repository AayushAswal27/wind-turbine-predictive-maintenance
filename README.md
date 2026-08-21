# Wind Turbine Predictive Maintenance & Failure Intelligence System

Predicting wind turbine failures early enough to prioritize maintenance, using
classical machine learning on real SCADA sensor data.

> **Status:** In development. Results marked `[X]` / `[TBD]` are placeholders
> until experiments are run.

---

## Problem

Wind turbines fail expensively. An unplanned gearbox or generator failure means
emergency crews, crane rentals, and days of lost generation. Fixed-schedule
maintenance wastes money servicing healthy turbines. The industry goal is
*condition-based maintenance* — service what is actually degrading, before it fails.

**ML framing:** Given time-series SCADA sensor readings (temperatures, wind speed,
power output, rotational variables), predict whether a turbine will experience a
failure within a future window of N hours/days — using only information available
at prediction time.

This is a **temporal** predictive-maintenance problem, so the project treats
leakage prevention and chronological validation as first-class concerns, not
afterthoughts.

---

## Dataset

**Source:** [EDP Open Data platform](https://www.edp.com/en/innovation/data) — Wind Farm 1
(onshore, Portugal). Real SCADA signals + a manually recorded failure logbook
(gearbox, generator, transformer, hydraulic failures).

The data is **not included** in this repo (size + licensing). To reproduce:

1. Go to the [EDP Open Data page](https://www.edp.com/en/innovation/data).
2. Download for **Wind Farm 1**:
   - Wind Turbine SCADA signals — 2016
   - Wind Turbine SCADA signals — 2017
   - Historical Failure Logbook — 2016
   - Historical Failure Logbook — 2017
3. Place the `.xlsx` files in `data/raw/`.

> **Known data-quality note:** The Fraunhofer *CARE to Compare* repackaging of
> this data ([Zenodo](https://zenodo.org/records/10958775)) documents that per-timestamp
> Min/Max/Std statistics can be implausible and that pitch-angle values suffer from
> angle-wrapping (0° ≡ 360°). Prefer Avg signals; validate Min/Max/Std before use.
> This is addressed in the cleaning notebook.

---

## Approach

```
Raw SCADA + Failure Logbook
        |
   Data Understanding  (join SCADA to failure events by timestamp)
        |
       EDA             (sensor behavior, pre-failure patterns)
        |
     Cleaning          (leakage-safe; Avg signals, quality filtering)
        |
 Feature Engineering   (lag, backward-only rolling, ratios, temp deltas)
        |
 Target + Temporal     (build "failure within N", chronological split)
   Validation
        |
   +----------+-------------+----------------+
   |          |             |                |
Classification Clustering  Anomaly Detection SHAP
(LogReg/RF/    (K-Means/    (IsoForest/LOF)  (why is this
 XGBoost)       PCA)                          turbine risky?)
   |          |             |                |
   +----------+------+------+----------------+
                     |
               Risk Scoring  (heuristic; failure prob + anomaly score)
                     |
             Business Insights & Maintenance Priority
```

---

## Repository Structure

```
wind-turbine-predictive-maintenance/
├── data/
│   ├── raw/            # EDP xlsx files go here (gitignored)
│   └── processed/      # cleaned + engineered outputs (gitignored)
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_target_and_temporal_validation.ipynb
│   ├── 06_classification.ipynb
│   ├── 07_clustering_pca.ipynb
│   ├── 08_anomaly_detection.ipynb
│   ├── 09_shap_explainability.ipynb
│   ├── 10_risk_scoring.ipynb
│   └── 11_business_insights.ipynb
├── reports/
│   ├── figures/        # exported plots
│   └── model_results/  # saved metrics & comparison tables
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Key Results

_To be filled in as notebooks are completed._

| Metric | Value |
|---|---|
| Failure-prediction window (N) | `[TBD]` |
| Best model | `[TBD]` |
| PR-AUC | `[X]` |
| Recall @ chosen threshold | `[X]` |
| Turbines flagged high/critical risk | `[X]` |

---

## Setup

```bash
# Use Python 3.11 (SHAP compatibility)
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks in order (01 → 11). Each is designed to be **Restart & Run All**
clean before commit.

---

## Tech Stack

pandas · numpy · scipy · scikit-learn · xgboost · shap · matplotlib · seaborn

Classical ML only — no deep learning, no deployment layer. The focus is
disciplined applied ML: leakage-safe temporal validation, honest evaluation
under class imbalance, and translating model output into maintenance decisions.
