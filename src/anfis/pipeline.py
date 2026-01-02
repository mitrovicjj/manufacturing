# src/anfis/complete_pipeline.py  (KOMPLETAN, SAMOSTALAN)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib

print("=== ICS ANFIS PIPELINE START ===")

# 1. LOAD & INSPECT DATA
DATA_PROCESSED = Path("data/processed/logs_anfis_ready.csv")
df = pd.read_csv(DATA_PROCESSED)
print(f"LOADED: {len(df):,} cycles")
print("\nSHAPE:", df.shape)
print("\nTOP FEATURES:")
print(df.columns.tolist())

# 2. FEATURE SELECTION (6 najboljih za ANFIS)
def select_anfis_features(df):
    anfis_features = [
        'temperature', 'vibration', 'pressure', 'cycle_time',
        'tool_wear_factor', 'failure_state_encoded', 'vib_temp_ratio'
    ]
    X = df[anfis_features].fillna(0)
    y = df['downtime_flag']
    return X, y

X, y = select_anfis_features(df)
print(f"\nANFIS INPUT: {X.shape} → TARGET: {y.value_counts(normalize=True)}")

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTRAIN: {len(X_train)}, TEST: {len(X_test)}")

# 4. SIMPLIFIED ANFIS (Fuzzy Logic + Logistic Regression HYBRID)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("\n=== TRAINING MODELS ===")

# Baseline: Random Forest (za poređenje)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("RANDOM FOREST (BASELINE):")
print(classification_report(y_test, rf_pred))

# 5. FEATURE IMPORTANCE
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTOP 6 FEATURES (RF Importance):")
print(importances.head(6))

# 6. FUZZY-INSPIRED FEATURES (za pravi ANFIS)
def create_fuzzy_features(X):
    Xf = X.copy()
    
    # Fuzzy rules kao features
    Xf['fuzzy_temp_vib'] = np.minimum(Xf['temperature'], Xf['vibration'])  # MIN operator
    Xf['fuzzy_wear_cycle'] = np.minimum(Xf['tool_wear_factor'], Xf['cycle_time'])
    Xf['fuzzy_overload'] = np.maximum(Xf['temperature'] + Xf['vibration'] - 1, 0)  # MAX operator
    
    return Xf

X_train_fuzzy = create_fuzzy_features(X_train)
X_test_fuzzy = create_fuzzy_features(X_test)

# Fuzzy + Logistic (ANFIS-like)
fuzzy_model = LogisticRegression(random_state=42, max_iter=1000)
fuzzy_model.fit(X_train_fuzzy, y_train)
fuzzy_pred = fuzzy_model.predict(X_test_fuzzy)

print("\n=== FUZZY-LOGIC MODEL (ANFIS-like) ===")
print(classification_report(y_test, fuzzy_pred))

# 7. SAVE MODELS
Path("models").mkdir(exist_ok=True)
joblib.dump(rf_model, 'models/rf_baseline.pkl')
joblib.dump(fuzzy_model, 'models/fuzzy_anfis.pkl')
joblib.dump(X.columns, 'models/feature_names.pkl')

print("\n✅ MODELS SAVED: rf_baseline.pkl + fuzzy_anfis.pkl")

# 8. VISUALIZATION
plt.figure(figsize=(15, 10))

# Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, fuzzy_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Fuzzy ANFIS Confusion Matrix')

# Feature Importance
plt.subplot(2, 3, 2)
top_features = importances.head(6)
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('Top ANFIS Features')

# Failure progression
plt.subplot(2, 3, 3)
df['datetime'] = pd.to_datetime(df['timestamp'], format='mixed')
failure_by_hour = df.groupby(df['datetime'].dt.hour)['downtime_flag'].mean()
failure_by_hour.plot(kind='bar')
plt.title('Failure Rate by Hour (Shift Effect)')
plt.xticks(rotation=0)

# Sensor correlations
plt.subplot(2, 3, 4)
corr = df[['temperature','vibration','pressure','cycle_time']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Sensor Correlations')

plt.tight_layout()
plt.savefig('models/anfis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. REAL-TIME INFERENCE TEST
print("\n=== REAL-TIME TEST ===")
def predict_realtime(model, scaler=None, data_dict=None):
    if data_dict is None:
        # Simulate critical failure
        data_dict = {
            'temperature': 0.85, 'vibration': 0.75, 'pressure': 0.2,
            'cycle_time': 0.8, 'tool_wear_factor': 0.9, 
            'failure_state_encoded': 2, 'vib_temp_ratio': 0.88
        }
    
    test_df = pd.DataFrame([data_dict])[X.columns]
    fuzzy_test = create_fuzzy_features(test_df)
    
    pred = model.predict(fuzzy_test)[0]
    prob = model.predict_proba(fuzzy_test)[0].max()
    
    status = "CRITICAL" if pred == 1 else "OK"
    print(f"INPUT: temp={data_dict['temperature']:.2f}, vib={data_dict['vibration']:.2f}")
    print(f"PREDICTION: {status} (confidence: {prob:.1%})")

predict_realtime(fuzzy_model)

print("\n ANFIS PIPELINE COMPLETE!")
print("Data inspected, models trained, ready for ICS deployment")