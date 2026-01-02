def select_anfis_features(df):
    """best 6 features za ANFIS (fuzzy rules)"""
    anfis_features = [
        'temperature',           # Senzor 1
        'vibration',             # Senzor 2  
        'cycle_time',            # Performanse
        'tool_wear_factor',      # Degradacija
        'failure_state_encoded', # State
        'vib_temp_ratio'         # Korelacija
    ]
    
    X = df[anfis_features]
    y = df['downtime_flag']
    
    print("ANFIS INPUT SHAPE:", X.shape)
    print("\nFeature correlations sa target:")
    print(X.corrwith(y).sort_values(ascending=False))
    
    return X, y