# Predictive Maintenance and Throughput Optimization for Manufacturing Lines

Tailored for domestic industry with limited sensors, real-world noise, and a focus on practical, measurable impact.

---

## Project Overview

This project combines machine learning (XGBoost) and adaptive neuro-fuzzy systems (ANFIS) to predict equipment failures and generate interpretable machine health scores.

The core objectives are to:
- predict short-term and mid-term slowdown/failure risk
- generate interpretable machine-health scores using fuzzy systems
- compare learned fuzzy reasoning (ANFIS) against purely statistical ML predictions (XGBoost)
- provide actionable insights for maintenance scheduling

**Current status:**  
Feature Store implementation, MLflow experiment tracking, and production XGBoost model deployed.  
ANFIS implementation complete with training pipeline.  

Fuzzy logic is a central component, complementing ML by providing transparent reasoning in environments where data is limited, noisy, or incomplete - common characteristics of domestic industrial settings.

---

## Data pipeline

### Raw data
- Source: `data/raw/logs_all_machines_v2.csv`
- Volume: 15k production cycles, 18 columns
- Includes: cycle times, sensor readings, maintenance records, operator assignments

### Feature engineering
- 49 generated features including:
  - Rolling statistics (window sizes 15-40)
  - Lag features (periods 1, 3, 5)
  - Utilization patterns
  - Categorical encoding (operator, maintenance_type)

### Feature store
- Cache directory: `data/feature_cache/`
- Cache hit rate: ~70%

---

## Machine Learning Component

### XGBoost Model

**Architecture:**
- 42 numeric features, 6 encoded categorical features
- Binary classification (failure/no-failure prediction)
- Windowed target variable (configurable prediction horizon)

**Production Model:**
- Path: `models/production/PRODUCTION_MODEL.pkl`
- Configuration: `rw15_tw5_lr008`
- Hyperparameters:
  - Rolling window: 15 cycles
  - Target window: 5 cycles ahead
  - Learning rate: 0.08
  - Max depth: 6
  - Scale pos weight: 13

**Performance Metrics:**
- ROC-AUC: 0.8382
- Recall @ 0.35 threshold: 99.2%
- False alarms: 76 / 3000 test cycles
- Precision-recall trade-off optimized for business cost minimization

**Experiment Tracking:**
- MLflow integration for 12 hyperparameter configurations
- Tracked metrics: ROC-AUC, F1, recall, precision, false alarms, downtime caught

### Hyperparameter Search Results

| Model           | Rolling Window | Target Window | ROC-AUC | Recall @ 0.35 | False Alarms |
|-----------------|----------------|---------------|---------|---------------|--------------|
| rw15_tw5_lr008  | 15             | 5             | 0.8382  | 99.2%         | 76           |
| rw40_tw10_lr010 | 40             | 10            | 0.9674  | 99.6%         | 241          |
| long_rw20_tw12  | 20             | 12            | 0.9746  | 99.8%         | 98           |

Production model selected based on lowest false alarm rate while maintaining >99% recall - TBD further analysis 

---

## Fuzzy Logic Component

### ANFIS (Adaptive Neuro-Fuzzy Inference System)

**Architecture:**
- 5-layer structure: fuzzification, rule firing, normalization, consequent, output
- Domain-aware initialization
- Gaussian membership functions with adaptive centers and widths
- Linguistic terms: Low, Medium-Low, Medium, Medium-High, High

**Implementation:**
- Modular design across 6 modules:
  - `config.py`: Configuration and domain knowledge
  - `membership.py`: Membership function initialization
  - `layers.py`: Layer-by-layer forward pass
  - `core.py`: Main ANFIS class
  - `train.py`: Hybrid learning algorithm (PyTorch-based)
  - `utils.py`: Rule generation and utilities

**Training:**
- Hybrid learning approach:
  - Premise parameters (membership functions): gradient descent with lower learning rate
  - Consequent parameters (linear functions): gradient descent with higher learning rate
- PyTorch backend for automatic differentiation
- Batch training with configurable epochs and learning rates

**Purpose:**
- Interpretability: extract human-readable fuzzy rules
- Robustness: handle missing or noisy sensor data
- Transparency: provide reasoning for maintenance teams

### Pure Fuzzy System (Planned)

Rule-based system with manually defined membership functions and expert-crafted rules for baseline comparison.

---

## Comparison Framework

### Approaches Under Evaluation

1. **XGBoost (Black Box ML)**
   - Highest predictive accuracy
   - Minimal interpretability
   - Sensitive to data distribution shifts

2. **ANFIS (Learned Fuzzy)**
   - Balanced accuracy and interpretability
   - Learns membership functions from data
   - Maintains rule-based structure

3. **Pure Fuzzy (Domain Expert)**
   - Full interpretability
   - No learning required
   - Performance depends on expert knowledge quality

---

## Production Pipeline

```text
data/raw/logs_all_machines_v2.csv (15k cycles, 18 cols)
    ↓
Feature Engineering (rolling, lag, categorical encoding)
    ↓
data/feature_cache/features_*.csv (49 features, 70% cache hit)
    ↓
Model Training (XGBoost + ANFIS)
    ↓
models/production/PRODUCTION_MODEL.pkl
    ↓
MLflow Tracking (http://localhost:5000)
    ↓
Evaluation & Comparison
```

## Planned Extensions

### Interpretability Analysis
- SHAP values for XGBoost feature importance
- Fuzzy rule extraction from ANFIS (identify most activated rules)
- Side-by-side comparison of reasoning paths between approaches
- Disagreement analysis: when and why models produce different predictions

### Synthetic Data Generator
- Generate edge-case scenarios:
  - Gradual degradation patterns
  - Sudden failure events
  - Sensor noise and missing data
  - Out-of-distribution conditions
- Evaluate model robustness and failure modes
- Test ANFIS performance with incomplete sensor coverage

### Ablation Study
- Train models with different feature subsets:
  - Rolling features only
  - Lag features only
  - Sensor data only
  - Full feature set
- Quantify contribution of each feature group
- Identify minimal feature set for acceptable performance

### Cost-Benefit Analysis
- Define business costs:
  - Unplanned downtime cost
  - Preventive maintenance cost
  - False alarm cost
- Calculate net savings
- Optimize decision threshold based on cost function
- Compare business value across different models

### Real-time Monitoring Dashboard
- PowerBI integration for visualization
- Live prediction streaming simulation
- Model agreement/disagreement tracking
- Feature drift detection
- Fuzzy rule activation heatmap