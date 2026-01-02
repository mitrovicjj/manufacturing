import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.anfis.anfis_features import select_anfis_features
from src.anfis.train_anfis import build_anfis_model

def anfis_pipeline():
    # Load data
    df = pd.read_csv("data/processed/logs_anfis_ready.csv")
    X, y = select_anfis_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ANFIS
    anfis_model = build_anfis_model(X_train[['temperature','vibration','cycle_time','tool_wear_factor']], y_train)
    
    # Test predictions
    y_pred = []
    for idx, row in X_test.iterrows():
        anfis_model.input['temperature'] = row['temperature']
        anfis_model.input['vibration'] = row['vibration']
        anfis_model.input['cycle_time'] = row['cycle_time']
        anfis_model.input['tool_wear_factor'] = row['tool_wear_factor']
        anfis_model.compute()
        y_pred.append(1 if anfis_model.output['failure_risk'] > 0.5 else 0)
    
    print("ANFIS PERFORMANCE:")
    print(classification_report(y_test, y_pred))
    
    return anfis_model

# RUN
model = anfis_pipeline()