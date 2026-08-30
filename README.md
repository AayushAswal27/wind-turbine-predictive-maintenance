# Wind Turbine Predictive Maintenance & Failure Intelligence System

Detecting wind turbine failures **before they happen** using classical machine
learning on real industrial SCADA sensor data — with rigorous, leakage-safe
validation on turbines the models have never seen.

> **Headline result:** No single method predicts every fault. But **supervised
> learning and anomaly detection are complementary** — combined, they flag rising
> risk before **all 12 real faults** on held-out turbines, each traced to the
> correct failing component via SHAP.

---

## The problem

Unplanned wind turbine failures are expensive: emergency crews, crane rentals, and
days of lost generation. Fixed-schedule maintenance wastes money servicing healthy
turbines. The goal is **condition-based maintenance** — predict which turbines are
degrading, early enough to act.

**ML framing:** given 10-minute SCADA sensor readings, predict whether a turbine is
inside the run-up window to a real fault — using only information available at
prediction time, and generalising to turbines not seen during training.

---

## Dataset

**[CARE to Compare](https://zenodo.org/records/15846963)** (Fraunhofer IEE), derived
from the **EDP open wind-farm data** — 5 onshore turbines in Portugal.

| Property | Value |
|---|---|
| Turbines | 5 (assets 0, 10, 11, 13, 21) |
| Datasets | 22 (12 fault runs, 10 normal runs) |
| Rows | ~1.2 million (10-min sampling) |
| Sensors | 83 raw SCADA channels |
| Real faults | 12 events: 6 hydraulic, 3 gearbox, 2 generator-bearing, 1 transformer |

The core difficulty — and what makes this realistic — is that **12 fault events is
very few**. This scarcity drives every design decision below.

---

## Key results

### 1. A real, subtle, fault-specific signal exists

EDA showed sensor signatures before faults are real but **not** simple thresholds —
they are fault-specific (a hot generator bearing predicts a bearing failure, not a
hydraulic one) and buried in noise. This ruled out naive rules and motivated
engineered temporal features.

### 2. Honest validation: leave-turbines-out

With 5 turbines and 12 events, a random train/test split leaks massively (adjacent
10-minute rows are near-identical). Every model here is validated with
**GroupKFold on turbine ID** — trained on 4 turbines, tested on the 5th it has never
seen. This is the real deployment scenario, and it is far harder than a random split.

### 3. Supervised classification — strongest for mechanical faults

Model progression on leave-turbines-out folds (metric: PR-AUC, baseline 0.017):

| Model | Mean PR-AUC | vs baseline |
|---|---|---|
| Logistic Regression | 0.035 | 2× |
| Random Forest | 0.081 | 5× |
| XGBoost | 0.088 | 5× |
| **LightGBM** | **0.099** | **6×** |

Per-fault detectability revealed the real story — **gearbox and mechanical faults are
predictable (11–21× lift over baseline); generator-bearing faults are not** (the two
examples behave oppositely, so no learnable pattern exists).

### 4. Anomaly detection catches what supervision misses

Trained on normal operation only, evaluated leave-turbines-out:

| Method | Faults detected (event-level) |
|---|---|
| Supervised (LightGBM) | 9 / 12 |
| Isolation Forest (global) | 6 / 12 |
| Local Outlier Factor (local) | 2 / 12 |
| **Combined (any method)** | **12 / 12** |

The methods are **complementary**: supervised learns consistent signatures (gearbox);
anomaly detection flags deviations without needing fault examples (the
generator-bearing fault supervised missed entirely). **Different fault mechanisms
require different detection strategies** — this is the project's central finding.

### 5. Explainability — the model reasons about the right components

SHAP confirms the model learned real physics: top features are 24-hour rolling
temperatures of the components that fail, and engineered features dominate raw
sensors. For the **transformer failure**, the #1 risk driver was the **transformer's
own temperature** — the model localises to the correct component, turning a risk
score into a maintenance instruction.

### 6. Deliverable — a fused risk score

The three signals are fused (heuristic, documented weights) into one risk score,
banded Low → Critical. Validated: pre-fault rate rises monotonically across bands
(**1.2% → 18.4%**, an ~11× concentration in the Critical band). Output is a
per-turbine maintenance priority ranking.

---

## Pipeline

```
Raw SCADA + fault logbook
        │
   01  Data understanding   (found: status_type_id is NOT the label — it leaks)
   02  EDA                  (signal exists, subtle, fault-specific)
   03  Cleaning             (leakage-safe; pitch-angle wrap fix; Avg signals only)
   04  Feature engineering  (lag / rolling / rate-of-change / domain deltas; 184 features)
   05  Target + validation  (binary target from event windows; leave-turbines-out folds)
        │
   06  Classification       (LogReg → RF → XGBoost → LightGBM)
   07  Anomaly detection    (Isolation Forest + LOF — complementary coverage)
   08  SHAP explainability  (per-component attribution)
   09  Risk scoring         (fused score → maintenance priority ranking)
```

---

## What makes this rigorous

- **Caught a target-leakage trap.** `status_type_id` looked like a fault label but is
  an operating-state code present in healthy turbines too — using it would have
  produced fake results. Verified against the raw event windows instead.
- **Leakage-safe features.** Every temporal feature is backward-looking only, verified
  with inline sanity checks; computed per-file so windows never cross run boundaries.
- **Honest metrics.** PR-AUC and recall (not accuracy) at ~1.6% positives; results
  reported per-fold and per-fault-type, never hidden behind a single average.
- **Documented limitations** (see below) rather than overclaimed performance.

---

## Honest limitations

- **12 fault events** is a small sample; single-event fault types (transformer,
  gearbox-bearings) are suggestive, not statistically robust.
- **Not a hard alarm.** Row-level precision is low; the system is decision-support for
  *prioritisation* (which turbines to inspect first), not automated alarming.
- **No healthy-turbine baseline** — every turbine in the benchmark eventually faulted,
  so the ranking reflects *degree* of risk among at-risk turbines.
- **No separate final holdout** — with only 5 turbines, leave-turbines-out
  cross-validation is used throughout rather than sacrificing turbines to a held-out
  set. Correlation-based feature pruning saw all folds (a minor, noted leak; pruning
  removed redundancy only, not target-based selection).

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `shap` ·
`matplotlib` · `seaborn`

Classical ML only — no deep learning (deliberate: 12 events would overfit a deep
model), no deployment layer. The focus is disciplined applied ML: leakage-safe
temporal validation, honest evaluation under extreme class imbalance, and translating
model output into maintenance decisions.

---

## Reproducing

```bash
# Python 3.11 (SHAP compatibility)
conda create -n windturbine python=3.11 -y
conda activate windturbine
pip install -r requirements.txt
```

1. Download the [CARE to Compare dataset](https://zenodo.org/records/15846963) and
   place **Wind Farm A** in `data/raw/Wind Farm A/`.
2. Run the notebooks in order (`01` → `09`).

Data is not included in this repo (size + CC-BY-SA licensing) — see the download step
above.

---

## Repository structure

```
├── notebooks/        # 01–09, the full analytical pipeline
├── data/             # raw + processed (gitignored)
├── reports/
│   ├── figures/      # EDA plots, SHAP summary
│   └── model_results/# saved model, predictions, risk scores
├── requirements.txt
└── LICENSE           # MIT (code); data under EDP/CARE terms
```