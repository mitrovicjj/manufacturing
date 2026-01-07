import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
print("🚀 Starting generate_logs2.py...")
print(f"Working directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).parent}")

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
    df = df.sort_values(['machine_id', 'timestamp']).copy()  # Ensure sorted for lags/diffs
    state_map = {"healthy": 0, "warning": 1, "degraded": 2, "critical": 3}
    df['failure_state_encoded'] = df['failure_state'].map(state_map)
    
    # Existing features
    df['cycle_dev_trend_5'] = (
        df.groupby('machine_id')['cycle_time']
        .rolling(5, min_periods=1).mean().reset_index(0, drop=True) / df['cycle_time'] - 1
    )
    df['vib_temp_ratio'] = df['vibration'] / (df['temperature'] + 1e-6)
    df['maint_group'] = (df['maintenance_type'] != "").cumsum()
    df['cum_uptime_since_maint'] = (
        df.groupby(['machine_id', 'maint_group'])['uptime'].cumsum()
    )
    
    # NEW: Differentials (anomaly rates)
    df['vibration_diff'] = df.groupby('machine_id')['vibration'].diff()
    df['cycle_time_diff'] = df.groupby('machine_id')['cycle_time'].diff()
    df['cycle_time_accel'] = df.groupby('machine_id')['cycle_time_diff'].diff()
    
    # NEW: Expanded lags/rolling (volatility)
    df['vibration_lag1'] = df.groupby('machine_id')['vibration'].shift(1)
    df['vibration_lag3'] = df.groupby('machine_id')['vibration'].shift(3)
    df['cycle_time_lag7'] = df.groupby('machine_id')['cycle_time'].shift(7)
    df['vib_rolling_std_7'] = df.groupby('machine_id')['vibration'].rolling(7, min_periods=1).std().reset_index(0, drop=True)
    df['cycle_q90_14'] = df.groupby('machine_id')['cycle_time'].rolling(14, min_periods=1).quantile(0.9).reset_index(0, drop=True)
    
    # NEW: Trends (EMA decay)
    df['vib_ema'] = df.groupby('machine_id')['vibration'].ewm(span=14).mean().reset_index(0, drop=True)
    
    # NEW: Simplified Fourier (low-freq on rolling; vectorized to avoid apply errors)
    for mach in df['machine_id'].unique():
        mask = df['machine_id'] == mach
        rolled = df.loc[mask, 'vibration'].rolling(30, min_periods=1).mean()
        fft_vals = np.fft.rfft(rolled.dropna()).real[:3]  # Top 3 components
        fft_series = pd.Series(np.resize(fft_vals, len(rolled)), index=rolled.index)
        df.loc[mask, 'fft_vib_1'] = fft_series.fillna(0)
        df.loc[mask, 'fft_vib_2'] = fft_series.shift(1).fillna(0)  # Phase shift proxy
    
    # NEW: Interactions
    df['vib_cycle_ratio'] = df['vibration'] / (df['cycle_time'] + 1e-6)
    
    # NEW: Cyclic maintenance (sin/cos for type periodicity)
    maint_map = {'': 0, 'preventive': 1, 'corrective': 2}
    df['maint_encoded'] = df['maintenance_type'].map(maint_map).fillna(0)
    df['maint_sin'] = np.sin(2 * np.pi * df['maint_encoded'] / 3)
    df['maint_cos'] = np.cos(2 * np.pi * df['maint_encoded'] / 3)
    
    # Fill NaNs forward/back (realistic for TS)
    num_feats = df.select_dtypes(include=[np.number]).columns
    df[num_feats] = df[num_feats].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Scale all numerics (now richer set)
    scaler = MinMaxScaler()
    num_cols = ['temperature', 'vibration', 'pressure', 'cycle_time', 'tool_wear_factor', 
                'batch_complexity'] + [col for col in df.columns if any(k in col for k in 
                ['diff', 'lag', 'std', 'q90', 'ema', 'fft', 'ratio', 'accel'])]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    print("New TS features added:", [col for col in df.columns if any(k in col.lower() for k in ['diff','lag','std','q90','ema','fft','ratio','accel','sin','cos'])])
    
    return df

if __name__ == "__main__":
    print("Generating logs...")
    
    # Generate 3 machines x 5000 cycles
    all_logs = []
    start_date = "2025-01-01T08:00:00"
    
    for machine_id in [1, 2, 3]:
        logs = simulate_machine_logs_v2(
            machine_id=machine_id,
            start_time=start_date,
            n_cycles=5000
        )
        all_logs.append(logs)
    
    df_full = pd.concat(all_logs, ignore_index=True)
    
    print("Raw shape:", df_full.shape)
    df_anfis = prepare_anfis_features(df_full)
    print("Saving logs_anfis_ready.csv...")
    
    df_anfis.to_csv('data/processed/logs_anfis_ready.csv', index=False)
    print("COMPLETE! Generated:", len(df_anfis), "rows")
    print("Check: data/processed/logs_anfis_ready.csv")