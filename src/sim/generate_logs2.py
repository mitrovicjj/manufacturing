import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler

def simulate_machine_logs_v2(
    machine_id,
    start_time,
    n_cycles,
    base_cycle_time=60.0,
    sigma_cycle=5.0,
    micro_downtime_prob=0.02,
    preventive_interval=100,
    operators=("Op1", "Op2"),
    shift_len_hours=8,
    temp_baseline=22.0,
    vibration_baseline=0.02,
    pressure_baseline=6.0,
    failure_state="healthy"
):
    """ICS ANFIS generator - EU manufacturing standards"""
    
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    
    rows = []
    t = start_time
    cycle_counter = 0
    current_failure_state = failure_state
    
    state_thresholds = {
        "vib_warning": 0.028,
        "vib_critical": 0.045,
        "temp_warning": 26.0
    }
    failure_probs = {
        "healthy": 0.001,
        "warning": 0.12,
        "degraded": 0.35,
        "critical": 0.75
    }
    
    tool_wear_factor = 1.0
    wear_rate = 0.00015
    current_batch_id = f"PO-{np.random.randint(1000,9999)}"
    batch_complexity = 1.0
    
    shift_fatigue = {1: 1.0, 2: 1.03, 3: 1.05}
    temp_drift = np.random.uniform(-0.005, 0.005)
    vib_drift = np.random.uniform(-2e-5, 2e-5)
    press_drift = np.random.uniform(-0.002, 0.002)
    
    for i in range(n_cycles):
        cycle_counter += 1
        
        # SHIFT + OPERATOR
        current_shift = ((i * base_cycle_time) // (shift_len_hours * 3600)) % 3 + 1
        shift_effect = shift_fatigue[int(current_shift)]
        operator = np.random.choice(operators)
        op_effect = 1.0 + np.random.uniform(-0.07, 0.07)
        
        # TOOL WEAR
        tool_wear_factor = min(1.0 + i * wear_rate, 2.0)
        
        # BATCH
        if i % np.random.randint(20, 50) == 0:
            current_batch_id = f"PO-{np.random.randint(1000,9999)}"
            batch_complexity = np.random.choice([0.9, 1.0, 1.25], p=[0.7, 0.2, 0.1])
        
        # BASE CYCLE
        cycle_time = (max(1.0, np.random.normal(base_cycle_time, sigma_cycle)) * 
                     op_effect * shift_effect * tool_wear_factor * batch_complexity)
        
        # SEASONAL
        month = t.month
        seasonal_offset = 0.0
        if month in [6,7,8]:    seasonal_offset = 6.0
        elif month in [12,1,2]: seasonal_offset = -3.0
        elif month in [3,4,10,11]: seasonal_offset = np.random.choice([-1.0, 1.0])
        
        # BASE SENSORS
        temperature = np.random.normal(temp_baseline + i*temp_drift + seasonal_offset, 1.2)
        temperature = np.clip(temperature, 15.0, 35.0)  # ✅ 15-35°C REAL FACTORY
        
        vibration = np.random.normal(vibration_baseline + i*vib_drift, 0.012)
        pressure = np.random.normal(pressure_baseline + i*press_drift, 0.25)
        
        # SENSOR CORRELATION
        if temperature > 26.0:
            vibration *= 1.5
            pressure *= 0.85
        if cycle_time > 70:
            vibration += 0.030
            temperature += np.random.uniform(1.0, 2.0)
        if current_failure_state == "critical":
            vibration += np.random.uniform(0.05, 0.10)
            temperature = min(temperature + np.random.uniform(4, 6), 35.0)  # CAP
        
        # Lakse tranzicije
        if current_failure_state == "healthy":
            if (vibration > state_thresholds["vib_warning"] or 
                temperature > state_thresholds["temp_warning"]):
                current_failure_state = "warning"
        elif current_failure_state == "warning":
            if vibration > state_thresholds["vib_critical"]:
                current_failure_state = "degraded"
            elif (vibration < state_thresholds["vib_warning"] * 0.75 and 
                  temperature < state_thresholds["temp_warning"] - 1.5):
                current_failure_state = "healthy"
        elif current_failure_state == "degraded":
            if np.random.rand() < 0.25:  # Brze u critical
                current_failure_state = "critical"
        
        # DOWNTIME
        downtime = 0.0
        downtime_flag = 0
        maintenance_type = ""
        
        if np.random.rand() < failure_probs[current_failure_state]:
            downtime = np.random.uniform(60, 900)
            downtime_flag = 1
            maintenance_type = "corrective"
            current_failure_state = "healthy" if np.random.rand() < 0.75 else "warning"
        elif np.random.rand() < micro_downtime_prob:
            downtime = np.random.uniform(1, 10)
            downtime_flag = 1
        if cycle_counter % preventive_interval == 0:
            downtime = max(downtime, np.random.uniform(60, 300))
            downtime_flag = 1
            maintenance_type = "preventive"
            if np.random.rand() < 0.6:
                tool_wear_factor = 1.0
        
        uptime = max(0.0, cycle_time - downtime)
        
        timestamp_str = t.replace(microsecond=0).isoformat()
        
        rows.append({
            "timestamp": timestamp_str,
            "machine_id": machine_id,
            "cycle_id": f"{machine_id}_{i}",
            "cycle_time": round(cycle_time, 3),
            "uptime": round(uptime, 3),
            "downtime": round(downtime, 3),
            "downtime_flag": int(downtime_flag),
            "maintenance_type": maintenance_type,
            "operator": operator,
            "shift": int(current_shift),
            "production_order_id": current_batch_id,
            "temperature": round(temperature, 2),
            "vibration": round(vibration, 4),
            "pressure": round(pressure, 2),
            "failure_state": current_failure_state,
            "tool_wear_factor": round(tool_wear_factor, 3),
            "batch_complexity": round(batch_complexity, 3),
            "seasonal_offset": round(seasonal_offset, 2)
        })
        
        t += timedelta(seconds=cycle_time)
    
    return pd.DataFrame(rows)

def prepare_anfis_features(df):
    df = df.copy()
    state_map = {"healthy": 0, "warning": 1, "degraded": 2, "critical": 3}
    df['failure_state_encoded'] = df['failure_state'].map(state_map)
    
    df['cycle_dev_trend_5'] = (
        df.groupby('machine_id')['cycle_time']
        .rolling(5, min_periods=1).mean().reset_index(0, drop=True) / df['cycle_time'] - 1
    )
    
    df['vib_temp_ratio'] = df['vibration'] / (df['temperature'] + 1e-6)
    
    df['maint_group'] = (df['maintenance_type'] != "").cumsum()
    df['cum_uptime_since_maint'] = (
        df.groupby(['machine_id', 'maint_group'])['uptime'].cumsum()
    )
    
    scaler = MinMaxScaler()
    num_cols = ['temperature', 'vibration', 'pressure', 'cycle_time', 
                'tool_wear_factor', 'batch_complexity']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    return df

# MAIN
if __name__ == "__main__":
    machine_params = [
        {"base_cycle_time": 60.0, "sigma_cycle": 5.0},
        {"base_cycle_time": 65.0, "sigma_cycle": 6.0},
        {"base_cycle_time": 55.0, "sigma_cycle": 4.0},
    ]
    
    dfs = []
    for idx, mp in enumerate(machine_params, 1):
        
        df_i = simulate_machine_logs_v2(
            machine_id=idx,
            start_time="2025-01-01 06:00:00",
            n_cycles=5000,
            base_cycle_time=mp["base_cycle_time"],
            sigma_cycle=mp["sigma_cycle"],
            micro_downtime_prob=0.02,
            preventive_interval=100
        )
        dfs.append(df_i)
    
    df_all = pd.concat(dfs, ignore_index=True)
        
    DATA_RAW = Path("data/raw")
    DATA_PROCESSED = Path("data/processed")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(DATA_RAW / "logs_all_machines_v2.csv", index=False)
    df_anfis = prepare_anfis_features(df_all)
    df_anfis.to_csv(DATA_PROCESSED / "logs_anfis_ready.csv", index=False)

    # VALIDATION
    print("1. Temperature range (°C):", 
          f"{df_all['temperature'].min():.1f} - {df_all['temperature'].max():.1f}")

    df_all['month'] = pd.to_datetime(df_all['timestamp'], format='mixed').dt.month
    print("2. Seasonal effect (Jan, Jul, Dec):")
    seasonal_temp = df_all.groupby('month')['temperature'].mean()
    print(seasonal_temp.get([1,7,12], "N/A"))
    
    print("3. Failure progression:")
    print(df_all.groupby('failure_state')['downtime_flag'].mean())
    print("4. Sensor correlation:")
    print(df_all[['temperature','vibration','pressure']].corr())
    print(f"\nGenerated {len(df_all):,} cycles")