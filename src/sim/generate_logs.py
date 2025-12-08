import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def simulate_machine_logs(
    machine_id,
    start_time,
    n_cycles,
    base_cycle_time=60.0,
    sigma_cycle=5.0,
    micro_downtime_prob=0.02,
    failure_prob=0.005,
    preventive_interval=100,
    operators=("Op1", "Op2"),
    shift_len_hours=8,
    temp_baseline=60.0,
    vibration_baseline=0.02,
    pressure_baseline=5.0
):
    """Simulate realistic machine log data for predictive maintenance + simulation purposes."""
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    
    rows = []
    t = start_time
    cycle_counter = 0
    
    # Shift baseline fatigue effect (later shift can be slightly slower)
    shift_fatigue = {1: 1.0, 2: 1.03, 3: 1.05}
    
    # Sensor drift over time
    temp_drift = np.random.uniform(-0.01, 0.01)  # per cycle
    vib_drift = np.random.uniform(-1e-5, 1e-5)
    press_drift = np.random.uniform(-0.001, 0.001)
    
    for i in range(n_cycles):
        cycle_counter += 1
        
        # Rotate operator (could also add random)
        operator = np.random.choice(operators)
        op_effect = 1.0 + np.random.uniform(-0.07, 0.07)
        
        # Shift calculation
        current_shift = ((i * base_cycle_time) // (shift_len_hours * 3600)) % 3 + 1
        shift_effect = shift_fatigue[int(current_shift)]
        
        # Base cycle_time with noise, operator and shift effect
        cycle_time = max(1.0, np.random.normal(base_cycle_time, sigma_cycle)) * op_effect * shift_effect
        
        downtime = 0.0
        downtime_flag = 0
        maintenance_type = ""
        
        # Micro-stoppage?
        if np.random.rand() < micro_downtime_prob:
            downtime = np.random.uniform(1, 10)
            downtime_flag = 1
        
        # Failure?
        if np.random.rand() < failure_prob:
            downtime = max(downtime, np.random.uniform(30, 600))
            downtime_flag = 1
            maintenance_type = "corrective"
        
        # Preventive maintenance at intervals
        if cycle_counter % preventive_interval == 0:
            downtime = max(downtime, np.random.uniform(60, 300))
            downtime_flag = 1
            maintenance_type = "preventive"
        
        uptime = max(0.0, cycle_time - downtime)
        
        # Production order with batch effect
        production_order_id = f"PO-{np.random.randint(1000, 9999)}"
        
        # Sensors with drift + small noise + anomalies on downtime
        temperature = np.random.normal(temp_baseline + i*temp_drift, 1.0)
        vibration = np.random.normal(vibration_baseline + i*vib_drift, 0.005)
        pressure = np.random.normal(pressure_baseline + i*press_drift, 0.2)
        
        # spike sensors if downtime occurs
        if downtime_flag:
            temperature += np.random.uniform(0, 3)
            vibration += np.random.uniform(0, 0.01)
            pressure += np.random.uniform(0, 0.5)
        
        rows.append({
            "timestamp": t.isoformat(),
            "machine_id": machine_id,
            "cycle_id": f"{machine_id}_{i}",
            "cycle_time": round(cycle_time, 3),
            "uptime": round(uptime, 3),
            "downtime": round(downtime, 3),
            "downtime_flag": int(downtime_flag),
            "maintenance_type": maintenance_type,
            "operator": operator,
            "shift": int(current_shift),
            "production_order_id": production_order_id,
            "temperature": round(temperature, 3),
            "vibration": round(vibration, 5),
            "pressure": round(pressure, 3)
        })
        
        # increment time
        t += timedelta(seconds=cycle_time)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Multiple machines with different params
    machine_params = [
        {"machine_id": 1, "base_cycle_time": 60.0, "sigma_cycle": 5.0, "failure_prob": 0.005},
        {"machine_id": 2, "base_cycle_time": 65.0, "sigma_cycle": 6.0, "failure_prob": 0.007},
        {"machine_id": 3, "base_cycle_time": 55.0, "sigma_cycle": 4.0, "failure_prob": 0.004},
    ]
    
    dfs = []
    for mp in machine_params:
        df_i = simulate_machine_logs(
            machine_id=mp["machine_id"],
            start_time="2025-01-01 06:00:00",
            n_cycles=5000,
            base_cycle_time=mp.get("base_cycle_time",60),
            sigma_cycle=mp.get("sigma_cycle",5),
            micro_downtime_prob=0.02,
            failure_prob=mp.get("failure_prob",0.005),
            preventive_interval=100,
            operators=("Op1","Op2"),
            shift_len_hours=8
        )
        dfs.append(df_i)
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # CSV ready for Tecnomatix , ML
    out_path = Path("../../data/raw/logs_all_machines.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    
    print(f"Saved: {out_path.absolute()}")