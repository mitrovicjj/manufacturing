# Predictive Maintenance and Throughput Optimization for Manufacturing Lines

Tailored for domestic industry with limited sensors, real-world noise, and a focus on practical, measurable impact.

---

## Project Overview

This project combines machine learning (XGBoost), fuzzy logic, and discrete-event simulation (Tecnomatix Plant Simulation).

The core idea is to:
- predict slowdowns and failures,
- generate interpretable machine-health scores using fuzzy systems,
- simulate how different maintenance strategies affect throughput.

**Production status:**  
Feature Store, MLflow experiment tracking and a production model.  
The full pipeline processes 43 features, achieving ROC-AUC up to 0.97.

Fuzzy logic is a central component, complementing ML by providing transparent reasoning in environments where data is limited, noisy, or incomplete, a common case in domestic industrial settings.

---

## Objectives

- Predict short-term and mid-term slowdown/failure risk  
- Generate an interpretable fuzzy-based machine health index  
- Compare fuzzy reasoning vs. ML predictions for decision support  
- Run simulation experiments for:
  - maintenance interval optimization  
  - buffer sizing  
  - line balancing  
  - operator allocation  
- Evaluate scenarios using KPIs:
  - throughput  
  - avoided downtime hours  
  - cost impact  

---

## Data Requirements

### Machine Logs
- cycle time
- micro-downtimes
- alarms

### Maintenance
- type
- date
- cause

### Production
- material
- shift
- operator

### Optional Sensors
- vibration
- temperature
- pressure

---

## Production Data Pipeline

- **Raw data:**  
  `data/raw/logs_all_machines_v2.csv`  
  (15k cycles, 18 columns)

- **Features:**  
  49 generated columns (rolling statistics, lags, encoded categoricals)

- **Cache:**  
  `data/feature_cache/`  
  FeatureStore with ~70% cache hit rate

---

## ML Component

### Preprocessing
- Rolling features (window size 15–40)
- Lag features
- Utilization patterns
- Categorical encoding (operator, maintenance_type)

### Model
- XGBoost production pipeline  
- 42 numeric + 6 encoded categorical features

### Production Model
- `models/production/PRODUCTION_MODEL.pkl`
- Configuration: `rw15_tw5_lr008`

### Metrics
- ROC-AUC: **0.8382**
- Recall @ 0.35 threshold: **99.2%**
- False alarms: **76 / 3000** test cycles

### Experiment Tracking
- MLflow (local UI)
- 12 hyperparameter combinations tested

---

## Fuzzy Logic Component

Fuzzy logic is used to model machine health and slowdown risk in a transparent and interpretable way.  
This is especially relevant when sensor coverage is minimal or maintenance teams prefer rule-based reasoning.

### Fuzzy System Design

**Membership functions for:**
- cycle time deviation
- vibration level
- temperature trend
- recent micro-downtimes

**Example rules:**
- IF cycle_time is high AND vibration is increasing - health is low  
- IF micro-downtimes are frequent - slowdown risk is medium-to-high  

**Outputs:**
- Fuzzy machine health score (0–1)
- Fuzzy slowdown-risk classification (low / medium / high)

---

## ANFIS (Adaptive Neuro-Fuzzy Inference System)

Hybrid model combining neural learning and fuzzy inference.

- Trained on the same feature set as XGBoost
- Learns membership functions and rule weights
- Enables comparison between learned fuzzy reasoning and purely statistical ML prediction

**Purpose:**
- interpretability
- robustness to missing or noisy inputs
- practical applicability in low-sensor environments

---

## Simulation Component (Tecnomatix Plant Simulation)

- Production-line model with key machines and buffers
- ML predictions and fuzzy scores feed simulation parameters

### Scenario Experiments
- maintenance interval changes
- buffer adjustments
- line balancing
- operator allocation

### Outputs
- throughput
- downtime
- lost units
- maintenance cost

---

## Experiment Results

12 hyperparameter combinations tested via MLflow:

| Model           | Rolling Window | Target Window | ROC-AUC | Recall @ 0.35 | False Alarms |
|-----------------|----------------|---------------|---------|---------------|--------------|
| rw15_tw5_lr008  | 15             | 5             | 0.8382  | 99.2%         | 76           |
| rw40_tw10_lr010 | 40             | 10            | 0.9674  | 99.6%         | 241          |
| long_rw20_tw12  | 20             | 12            | 0.9746  | 99.8%         | 98           |

**Production model selection criteria:**  
Lowest false alarms (76 / 3000) while maintaining high recall (>99%) for short-term prediction.

**MLflow UI:**  
`http://localhost:5000`

---

## Experiment Plan Status

- In progress:
  - Fuzzy system and ANFIS development
  - ML vs. fuzzy comparison
  - Simulation integration
  - Scenario experiments and KPI evaluation

---

## Production Pipeline Overview

```text
data/raw/logs_all_machines_v2.csv
    ↓ Feature Store (70% cache hit)
data/feature_cache/features_*.csv (43 features)
    ↓ Full pipeline (numeric + categorical)
models/production/PRODUCTION_MODEL.pkl
    ↓ MLflow tracking
http://localhost:5000 (12 experiments)