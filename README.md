# Predictive Maintenance and Throughput Optimization for Manufacturing Lines

ML-driven predictive maintenance combining XGBoost and ANFIS (Adaptive Neuro-Fuzzy Inference System) for interpretable failure prediction. Tailored for industrial environments with limited sensors, real-world noise and focus on practical impact.

---

## Overview

This project explores the balance between predictive accuracy and interpretability in manufacturing maintenance by comparing:

- **XGBoost**: High-accuracy black-box predictions
- **ANFIS**: Interpretable fuzzy logic with learned membership functions
- **Pure Fuzzy** (planned): Expert-driven rule-based baseline

**Objectives:**
- Predict equipment failures 5-12 cycles ahead
- Generate interpretable machine health scores using fuzzy reasoning
- Quantify accuracy-interpretability tradeoffs
- Provide actionable maintenance scheduling insights

**Current status:**  
Production XGBoost model deployed. ANFIS implementation complete with hybrid PyTorch-based training, threshold optimization and MLflow experiment tracking

---

## Architecture

### Data pipeline

```
Raw Data
    ↓
Feature Engineering (49 features: rolling stats, lags, utilization)
    ↓
Feature Store
    ↓
Model Training (XGBoost + ANFIS)
    ↓
MLflow Tracking & Evaluation
```

**Dataset:**
- Volume: 15,000 production cycles
- Features: Cycle times, vibration, temperature, pressure, maintenance records

**Feature Store:**
- Configurable rolling windows (15-40 cycles) and target horizons (5-12 cycles)
- Automated feature versioning by pipeline configuration

---

## Model Comparison

### XGBoost (Production Model)

**Performance:**
- ROC-AUC: 0.8382
- Recall @ 0.35 threshold: 99.2%
- False alarms: 76 / 3000 test cycles

**Configuration:**
- Rolling window: 15 cycles
- Prediction horizon: 5 cycles ahead
- Hyperparameters: lr=0.08, depth=6, scale_pos_weight=13

### ANFIS

**Architecture:**
- 5-layer fuzzy inference system (fuzzification, rule firing, normalization, consequent, output)
- Gaussian membership functions with adaptive parameters
- Rule explosion mitigation: 4-8 input features create 16-256 rules

**Training:**
- Hybrid learning algorithm (PyTorch backend):
  - Premise parameters (membership functions): gradient descent with lr=1e-3
  - Consequent parameters (linear functions): gradient descent with lr=1e-2
- Binary classification with BCEWithLogitsLoss + Focal Loss
- Automatic threshold optimization (0.35-0.45)
- Batch training with gradient tracking via MLflow

**Current Performance (6 features, 64 rules):**
- F1 Score: **0.48**
- Precision: **0.34** | Recall: **0.85**
- Optimal threshold: **0.375**

**Implementation:**
- `src/anfis/config.py`: Domain-aware configuration
- `src/anfis/membership.py`: Gaussian MF initialization
- `src/anfis/layers.py`: Layer-by-layer forward pass
- `src/anfis/core.py`: Main ANFIS class
- `src/anfis/train.py`: Hybrid learning + evaluation
- `src/anfis/utils.py`: Rule generation utilities

---

## MLflow Experiment Tracking

**Tracked Metrics:**
- Classification: F1, Precision, Recall, ROC-AUC, Optimal Threshold
- Regression: MSE, RMSE, MAE, R²
- Loss: BCE, Focal Loss
- Training: Premise/Consequent gradient norms

**Experiment Modes:**
```bash
# XGBoost grid search
python run_mlflow.py --mode xgboost_grid

# ANFIS single run
python run_mlflow.py --mode anfis_single --n_anfis_features 6 --epochs 50

# ANFIS ablation study (16, 32, 64 rules)
python run_mlflow.py --mode rule_ablation --experiment_name "ablation_v1"
```

**MLflow UI:** `http://localhost:5000`

---

## Key Features

### Feature Store
- Hash-based caching with configuration versioning
- Supports multiple rolling windows and target horizons
- Automatic invalidation on data/config changes

### Threshold Optimization
- Automatic F1-optimal threshold search (0.2-0.6 range)
- Prioritizes recall for failure-critical applications
- Logs optimal threshold per experiment

### Feature Selection for ANFIS
- Random Forest importance ranking
- Diversity enforcement across sensor groups (vibration, temperature, pressure, cycle time)
- Prevents rule explosion: 4-8 features → manageable rule counts

---

## Planned Extensions

### 1. Interpretability Analysis
- Extract top-k ANFIS rules by activation frequency
- SHAP values for XGBoost feature importance
- Reasoning comparison: "Why did models disagree?"

### 2. Ablation studies
- Feature group contribution: rolling vs. lag vs. sensor-only
- Membership function count
- Rule count scaling

### 3. Real-world Robustness Testing
- Edge cases: sensor dropouts, sudden failures, gradual degradation
- Synthetic data generator for out-of-distribution scenarios
- Cost-benefit analysis: downtime cost vs. false alarm cost

### 4. Production Deployment
- Real-time monitoring dashboard (PowerBI integration planned)
- Model agreement/disagreement tracking
- Feature drift detection

---

## Setup

**Key dependencies:**
- `xgboost`, `scikit-learn`, `imbalanced-learn`
- `torch` (ANFIS backend)
- `mlflow` (experiment tracking)
- `pandas`, `numpy`

**Quick start:**
```bash
# Train XGBoost
python run_mlflow.py --mode xgboost_grid

# Train ANFIS (6 features, 64 rules)
python run_mlflow.py --mode anfis_single --n_anfis_features 6 --epochs 50

# View experiments
mlflow ui
```

---

## Project structure

```
manufacturing/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Feature-engineered data
│   └── feature_cache/          # Cached feature sets
├── src/
│   ├── ml/                     # XGBoost pipeline
│   │   ├── train.py
│   │   ├── features.py
│   │   └── feature_store.py
│   └── anfis/                  # ANFIS implementation
│       ├── config.py
│       ├── membership.py
│       ├── layers.py
│       ├── core.py
│       ├── train.py
│       └── utils.py
├── models/
│   └── production/             # Deployed models
├── run_mlflow.py               # Main experiment runner
└── README.md
```

---