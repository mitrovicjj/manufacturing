import pandas as pd
import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz

def build_anfis_model(X_train, y_train):
    """ANFIS-like Fuzzy Inference System"""
    
    # define membership functions (fuzzy sets)
    temp = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'temperature')
    vib = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'vibration')
    cycle = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'cycle_time')
    wear = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'tool_wear_factor')
    
    failure = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'failure_risk')
    
    # Fuzzy membership functions
    temp['normal'] = fuzz.trimf(temp.universe, [0, 0.3, 0.5])
    temp['high'] = fuzz.trimf(temp.universe, [0.4, 0.7, 1.0])
    
    vib['low'] = fuzz.trimf(vib.universe, [0, 0.2, 0.4])
    vib['high'] = fuzz.trimf(vib.universe, [0.3, 0.6, 1.0])

    
    # FUZZY RULES (ICS manufacturing knowledge)
    rule1 = ctrl.Rule(temp['high'] & vib['high'], failure['high'])
    rule2 = ctrl.Rule(wear['high'] & cycle['high'], failure['medium'])
    
    # Control system
    failure_ctrl = ctrl.ControlSystem([rule1, rule2])
    failure_sim = ctrl.ControlSystemSimulation(failure_ctrl)
    
    return failure_sim