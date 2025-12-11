# Predictive maintenance and throughput optimization for manufacturing lines

Tailored for domestic industry with limited sensors, real-world noise and with a focus on practical, measurable impact.

---

## Project overview
This project combines ML (XGBoost), fuzzy logic, and discrete-event simulation (Tecnomatix Plant Simulation).  
The core idea is to predict slowdowns and failures, generate interpretable machine-health scores using fuzzy systems and simulate how different maintenance strategies affect throughput.

Fuzzy logic is a central component which complements ML by providing transparent reasoning in situations where data is limited, noisy, or incomplete,which is typical for domestic industrial environments.

---

## Objectives
- Predict short-term and mid-term slowdown/failure risk
- Generate an interpretable fuzzy-based machine health index
- Compare fuzzy reasoning vs ML predictions for decision support
- Run simulation experiments for maintenance interval optimization, buffer sizing, line balancing, operator allocation
- Evaluate scenarios using KPIs such as throughput, OEE, avoided downtime hours, cost impact

---

## Data requirements
- **Machine logs:** cycle_time, micro-downtimes, alarms
- **Maintenance:** type, date, cause  
- **Production:** material, shift, operator  
- **Optional sensors:** vibration, temperature, pressure  
- **Note:** includes cleaning, imputation and dealing with typical data gaps  

---

## ML component
- **Preprocessing:** rolling features, utilization patterns, downtime history
- **Model:** XGBoost as main predictive engine
- **Output:** downtime risk score for next few hours
- **Focus:** stability with limited/noisy instrumentation

---

## Fuzzy logic component
Fuzzy logic is used to model machine health and slowdown risk in a transparent way.  
This is relevant when sensor coverage is minimal/maintenance staff prefer interpretable rules.

### Fuzzy system design:
- Membership functions for:
  - cycle_time deviation  
  - vibration level  
  - temperature trend  
  - recent micro-downtimes  
- Expert-style rules such as:
  - IF cycle_time is high AND vibration is increasing → health is low  
  - IF micro-downtimes are frequent → slowdown risk is medium-to-high  
- Output: Fuzzy machine health score (0–1), Fuzzy slowdown-risk classification (low / medium / high)

### ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Hybrid model which learns membership functions and rule weights
- Trained on the same feature set as XGBoost
- Allows comparison between learned fuzzy reasoning and purely statistical ML prediction
- Purpose: interpretability, robustness to missing/noisy inputs, practical applicability in low-sensor environments

---

## Simulation component (Tecnomatix Plant Simulation)
- Production-line model with key machines and buffers  
- Both ML predictions and fuzzy scores feed simulation parameters  
- Run scenario experiments include different maintenance intervals ,buffer adjustments, line balancing, operator allocation  
- Outputs include throughput, downtime, lost units, maintenance cost  

---

## Experiment plan
1. EDA of downtime and slowdown patterns  
2. Build feature set (rolling windows, history, utilization)  
3. Train ML model (XGBoost)  
4. Develop fuzzy system + ANFIS  
5. Compare ML vs fuzzy outcomes  
6. Integrate both into simulation environment  
7. Run scenario experiments and evaluate KPIs  
8. Generate business-focused recommendations  

---

## Deliverables
- Trained XGBoost predictive model with evaluation metrics  
- Full fuzzy logic system + ANFIS implementation  
- Simplified Tecnomatix simulation with scenario runner  
- Scenario KPI reports (throughput, OEE, cost impact)  
- Documented codebase  

---