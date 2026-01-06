import requests
import json

BASE_URL = "http://localhost:5001"

print("="*70)
print("TEST: Buffer-based prediction")
print("="*70)

requests.post(f"{BASE_URL}/clear_history")
print("\nBuffer cleared")

status = requests.get(f"{BASE_URL}/buffer_status").json()
print(f"Buffer: {status['buffer_size']}/{status['max_size']}")


print("\nPopunjavanje buffer-a sa normalnim ciklusima...")
for i in range(20):
    cycle = {
        "cycle_id": 1000 + i,
        "cycle_time": 60 + (i % 5 - 2),
        "temperature": 30 + (i % 3),
        "vibration": 0.02 + (i % 4) * 0.001,
        "pressure": 5.0 + (i % 3) * 0.05,
        "operator": "op1",
        "maintenance_type": "preventive"
    }
    
    result = requests.post(f"{BASE_URL}/predict", json=cycle).json()
    
    if i == 0 or i == 9 or i == 19:
        print(f"  Cycle {cycle['cycle_id']}: "
              f"Risk={result['risk_score']:.4f}, "
              f"Used History={result.get('used_history', False)}, "
              f"Buffer={result.get('history_length', 0)}")


status = requests.get(f"{BASE_URL}/buffer_status").json()
print(f"\nBuffer popunjen: {status['buffer_size']}/{status['max_size']}")


print("\nSlanje kritičnih ciklusa...")
for i in range(3):
    critical = {
        "cycle_id": 2000 + i,
        "cycle_time": 75 + i * 5,
        "temperature": 45 + i * 5,
        "vibration": 0.08 + i * 0.03,
        "pressure": 4.2 - i * 0.3,
        "operator": "op1",
        "maintenance_type": "emergency"
    }
    
    result = requests.post(f"{BASE_URL}/predict", json=critical).json()
    
    icon = "🚨" if result['maintenance_trigger'] else "✅"
    print(f"{icon} Cycle {critical['cycle_id']}: "
          f"Risk={result['risk_score']:.4f}, "
          f"Priority={result['priority']}")

print("\n" + "="*70)
print("Test završen!")
print("="*70)