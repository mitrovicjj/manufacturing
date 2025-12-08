# Predictive maintenance and throughput optimization for manufacturing lines

Tailored for domestic industry with limited sensor setups and real-world noisy datasets.

---

## Project overview
This project combines machine learning and discrete-event simulation to predict equipment slowdowns and failures and simulate the effect of maintenance strategies on throughput.  
The focus is on domestic manufacturing lines with limited sensor instrumentation, aiming for practical, measurable impact.

---

## Objectives
- Predict short-term and mid-term probabilities of machine slowdown or failure  
- Identify key drivers of downtime  
- Run discrete-event simulations for:  
  - Maintenance interval optimization  
  - Buffer sizing  
  - Shift scheduling  
  - Operator allocation  
- Evaluate scenarios using clear KPIs: throughput, OEE, avoided downtime hours, estimated cost savings  

---

## Data requirements
- **Machine logs:** cycle times, stoppages, alarms, micro-downtimes  
- **Maintenance records:** type, date, failure cause  
- **Production metadata:** material, shift, operator  
- **Optional sensors:** temperature, vibration, pressure  
- **Notes:** Cleaning, imputation, handling of incomplete domestic data is included  

---

## ML component
- **Preprocessing:** rolling statistics, failure history, utilization patterns
- **Model:** Gradient Boosting (XGBoost / LightGBM)
- **Output:** downtime risk score for the next 2–24 hours
- **Focus:** Practical applicability with noisy or limited data

---

## Simulation component (Tecnomatix Plant Simulation)
- Simplified model of the production line with key machines and buffers  
- ML predictions feed into simulation parameters  
- Scenario experiments include varying maintenance intervals, buffer adjustment, line balancing, operator allocation  
- Metrics tracked: throughput, downtime, lost units, maintenance cost  

---

## Experiment plan
1. Exploratory data analysis of downtime patterns and key factors  
2. Train predictive ML model
3. Integrate predictions into the simulation environment  
4. Run scenario simulations and evaluate KPIs  
5. Generate business-focused recommendations with quantified benefits

---

## Deliverables
- Trained predictive ML model with evaluation metrics  
- Simplified discrete-event simulation model with scenario runner  
- Dashboard/report for scenario performance visualization  
- KPI improvement and cost-saving estimates  
- Modular, documented codebase

---

## Usage
```bash
# Clone repository
git clone https://github.com/mitrovicjj/manufacturing.git
cd manufacturing

# Install dependencies
pip install -r requirements.txt

# Run experiment
python run.py --experiment_name "your_experiment_name"

# Evaluate model
python src/ml/eval.py --model "path/to/model.pkl" --test_csv "data/test_logs.csv" --output_dir "experiments/eval"