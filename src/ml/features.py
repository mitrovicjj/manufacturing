import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# TARGET CREATION
def create_downtime_next(df, shift_periods=1):
    """
    Create target label: downtime in next cycle(s).
    
    Args:
        df: DataFrame with 'downtime_flag' column
        shift_periods: How many cycles ahead to look (default=1)
    
    Returns:
        DataFrame with 'downtime_next' column added
    """
    df = df.copy()
    df['downtime_next'] = df['downtime_flag'].shift(-shift_periods)
    # Remove rows where target is NaN (last N rows)
    df = df.dropna(subset=['downtime_next'])
    return df

def create_windowed_target(df, window=5):
    """
    Create windowed target: downtime in next N cycles.
    More signal for rare events - if ANY downtime in next N cycles, label=1.
    
    Args:
        df: DataFrame with 'downtime_flag' column
        window: Number of future cycles to look ahead
    
    Returns:
        DataFrame with 'downtime_next_N' column added
    """
    df = df.copy()
    col_name = f'downtime_next_{window}'
    
    # Rolling max over next N cycles (reverse direction)
    # Shift -1 to -window and take max
    future_downtime = pd.concat(
        [df['downtime_flag'].shift(-i) for i in range(1, window+1)],
        axis=1
    ).max(axis=1)
    
    df[col_name] = future_downtime
    df = df.dropna(subset=[col_name])
    
    return df

# CATEGORICAL ENCODING
def encode_categoricals(df, cat_columns=['operator', 'maintenance_type']):
    """
    Encode categorical columns as integer codes.
    
    Args:
        df: DataFrame
        cat_columns: List of categorical column names
    
    Returns:
        DataFrame with '{col}_encoded' columns added
    """
    df = df.copy()
    
    for col in cat_columns:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        else:
            print(f"Warning: Column '{col}' not found, skipping encoding.")
    
    return df

# ROLLING FEATURES
def create_rolling_features(df, columns, window=20):
    """
    Create rolling statistics for given columns.
    
    Features created:
    - mean_last_N
    - std_last_N
    - diff (current - previous)
    - trend (current - mean_last_N)
    
    Args:
        df: DataFrame
        columns: List of column names to create rolling features for
        window: Rolling window size (default=20)
    
    Returns:
        DataFrame with rolling feature columns added
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping rolling features.")
            continue
        
        # Rolling mean
        df[f'{col}_mean_last_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Rolling std
        df[f'{col}_std_last_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        # Diff: current - previous
        df[f'{col}_diff'] = df[col].diff()
        
        # Trend: current - rolling mean
        df[f'{col}_trend'] = df[col] - df[f'{col}_mean_last_{window}']
    
    # Fill NaN values from diff/std with 0
    df = df.fillna(0)
    
    return df

# LAG FEATURES
def create_lag_features(df, columns, lags=[1, 2, 3]):
    """
    Create lag features (previous cycle values).
    
    Args:
        df: DataFrame
        columns: List of column names to create lag features for
        lags: List of lag periods (default=[1,2,3])
    
    Returns:
        DataFrame with lag feature columns added
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping lag features.")
            continue
        
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Fill NaN values from lagging with 0
    df = df.fillna(0)
    
    return df

# NEW: ANFIS-SPECIFIC TIME-SERIES FEATURES (from prepare_anfis_features)
def create_anfis_features(df):
    """
    Create ANFIS-optimized time-series features with proper groupby alignment.
    Adds ~20 dynamic features post-encoding, pre-scaling.
    """
    df = df.sort_values(['machine_id', 'timestamp']).copy()
    
    # State encoding (if exists)
    if 'failure_state' in df.columns:
        state_map = {"healthy": 0, "warning": 1, "degraded": 2, "critical": 3}
        df['failure_state_encoded'] = df['failure_state'].map(state_map)
    
    # Existing core features
    df['cycle_dev_trend_5'] = (
        df.groupby('machine_id')['cycle_time']
        .rolling(5, min_periods=1).mean().reset_index(0, drop=True) / df['cycle_time'] - 1
    )
    df['vib_temp_ratio'] = df['vibration'] / (df['temperature'] + 1e-6)
    
    if 'maintenance_type' in df.columns:
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
    
    # NEW: Simplified Fourier (low-freq components - vectorized)
    for mach in df['machine_id'].unique():
        mask = df['machine_id'] == mach
        if mask.sum() > 0:
            rolled = df.loc[mask, 'vibration'].rolling(30, min_periods=1).mean()
            if len(rolled.dropna()) > 0:
                fft_vals = np.fft.rfft(rolled.dropna()).real[:3]  # Top 3 components
                fft_series = pd.Series(np.resize(fft_vals, len(rolled)), index=rolled.index)
                df.loc[mask, 'fft_vib_1'] = fft_series.fillna(0)
                df.loc[mask, 'fft_vib_2'] = fft_series.shift(1).fillna(0)
    
    # NEW: Interactions
    df['vib_cycle_ratio'] = df['vibration'] / (df['cycle_time'] + 1e-6)
    
    # NEW: Cyclic maintenance encoding (sin/cos)
    if 'maintenance_type' in df.columns:
        maint_map = {'': 0, 'preventive': 1, 'corrective': 2}
        df['maint_encoded'] = df['maintenance_type'].map(maint_map).fillna(0)
        df['maint_sin'] = np.sin(2 * np.pi * df['maint_encoded'] / 3)
        df['maint_cos'] = np.cos(2 * np.pi * df['maint_encoded'] / 3)
    
    # Fill NaNs forward/backward (realistic for TS)
    num_feats = df.select_dtypes(include=[np.number]).columns
    df[num_feats] = df[num_feats].clip(lower=-1e6, upper=1e6)  # Clip inf
    df[num_feats] = df[num_feats].ffill().bfill().fillna(0).clip(-10, 10)


    print("✓ ANFIS TS features added:", [col for col in df.columns 
           if any(k in col.lower() for k in ['diff','lag','std','q90','ema','fft','ratio','accel','sin','cos'])])
    
    return df

# FULL FEATURE PIPELINE (ENHANCED)
def build_features(df, 
                   target_type='next',  # 'next' or 'windowed'
                   target_window=5,
                   rolling_window=20,
                   rolling_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                   lag_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                   lag_periods=[1, 2, 3],
                   cat_cols=['operator', 'maintenance_type'],
                   include_anfis_features=True):
    """
    Complete feature engineering pipeline with ANFIS enhancements.
    
    Args:
        df: Raw DataFrame
        target_type: 'next' (single cycle) or 'windowed' (next N cycles)
        target_window: If windowed, how many cycles ahead
        rolling_window: Size of rolling window for statistics
        rolling_cols: Columns for rolling features
        lag_cols: Columns for lag features
        lag_periods: List of lag periods
        cat_cols: Categorical columns to encode
        include_anfis_features: Add ANFIS-optimized TS features (~20 extra)
    
    Returns:
        DataFrame with all features engineered
    """
    df = df.copy()
    df = df.sort_values(['machine_id', 'timestamp'])  # ✅ Added per comments
    print(f"Starting feature engineering on {len(df)} rows...")
    
    # 1. Create target
    if target_type == 'next':
        df = create_downtime_next(df, shift_periods=1)
        print(f"✓ Created 'downtime_next' target")
    elif target_type == 'windowed':
        df = create_windowed_target(df, window=target_window)
        print(f"✓ Created 'downtime_next_{target_window}' windowed target")
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    # 2. Encode categoricals
    df = encode_categoricals(df, cat_columns=cat_cols)
    print(f"✓ Encoded categorical columns: {cat_cols}")
    
    # 3. ANFIS Time-Series Features (post-encoding, pre-scaling)
    if include_anfis_features:
        df = create_anfis_features(df)
    
    # 4. Rolling features
    df = create_rolling_features(df, columns=rolling_cols, window=rolling_window)
    print(f"✓ Created rolling features (window={rolling_window}) for: {rolling_cols}")
    
    # 5. Lag features
    df = create_lag_features(df, columns=lag_cols, lags=lag_periods)
    print(f"✓ Created lag features (lags={lag_periods}) for: {lag_cols}")
    
    print(f"Feature engineering complete! Final shape: {df.shape}")
    
    return df

# FEATURE SELECTION HELPER (UPDATED)
def get_feature_columns(df, target_col='downtime_next', numeric_only=False):
    exclude = [
        'timestamp', 'cycle_id', 'production_order_id', 'machine_id', 'shift',
        'downtime_next', 'downtime_next_5', 'downtime_next_8', 'downtime_next_10',
        'operator', 'maintenance_type', 'failure_state', 'maint_encoded',  # ✅ Exclude encodings
        'failure_state_encoded',  # NEW: ANFIS state encoding
        target_col, 'downtime_flag', 'downtime',  # Targets/leaks
        'maint_group'  # NEW: Maintenance grouping (derived)
    ]
    
    features = [c for c in df.columns if c not in exclude]
    
    if numeric_only:
        numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        print(f"🔢 Filtered {len(numeric_features)}/{len(features)} numeric features")
        return numeric_features  # ✅ Fixed .tolist() issue
    
    return features

# BUILD INFERENCE FEATURES (ENHANCED)
def build_inference_features(df, 
                             rolling_windows=[15, 20],
                             rolling_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                             lag_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                             lag_periods=[1, 2, 3],
                             cat_cols=['operator', 'maintenance_type'],
                             include_anfis_features=True):
    """
    Feature engineering for inference - no target column.
    Supports single-cycle prediction with ANFIS features.
    """
    required_input_cols = ['cycle_time', 'temperature', 'vibration', 'pressure']
    missing = [col for col in required_input_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    
    df = df.copy()
    df = df.sort_values(['machine_id', 'timestamp']) if 'machine_id' in df.columns else df  # ✅ Added sorting
    print(f"Starting inference feature engineering on {len(df)} rows...")
    
    # 1. Encode categoricals
    df = encode_categoricals(df, cat_columns=cat_cols)
    print(f"✓ Encoded categorical columns: {cat_cols}")
    
    # 2. ANFIS features for inference (✅ Added per comments)
    if include_anfis_features:
        df = create_anfis_features(df)
        print("✓ Added ANFIS inference features (maint_sin, maint_cos, vib_cycle_ratio, etc.)")
    
    # 3. Rolling features for each window
    for window in rolling_windows:
        for col in rolling_cols:
            if col not in df.columns:
                continue
            df[f'{col}_mean_last_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_std_last_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        print(f"✓ Created rolling features (window={window})")
    
    # 4. Diff and trend features
    for col in rolling_cols:
        if col not in df.columns:
            continue
        df[f'{col}_diff'] = df[col].diff()
        df[f'{col}_trend'] = df[col] - df[f'{col}_mean_last_20']
    
    # 5. Lag features
    df = create_lag_features(df, columns=lag_cols, lags=lag_periods)
    print(f"✓ Created lag features (lags={lag_periods})")
    
    # 6. Missing model columns (✅ Extended with ANFIS defaults)
    required_cols = {
        'machine_id': 1,
        'shift': 1,
        'seasonal_offset': 0,
        'tool_wear_factor': 1.0,
        'batch_complexity': 1.0,
        'downtime': 0,
        'uptime': 100,
        'failure_state_encoded': 0  # ✅ ANFIS default
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
    
    print(f"✓ Added required model columns")
    
    # 7. Fill NaN values
    df = df.fillna(0)
    
    print(f"Inference feature engineering complete! Final shape: {df.shape}")
    return df

# VALIDATION HELPER
def validate_features(df, target_col='downtime_next'):
    """Quick feature validation for ANFIS/RF selection."""
    feats = get_feature_columns(df, target_col, numeric_only=True)
    print(f"\n📊 Feature Summary:")
    print(f"Total safe numeric features: {len(feats)}")
    print(f"Sample: {feats[:10]}")
    
    # Check multicollinearity for top ANFIS candidates
    ts_feats = [f for f in feats if any(k in f.lower() for k in ['diff', 'lag', 'std', 'fft', 'accel'])]
    if len(ts_feats) >= 3:
        corr_sample = df[ts_feats[:5]].corr()
        print(f"TS feature correlations:\n{corr_sample.abs().max().max() < 0.8 and '✅ Low multicollinearity' or '⚠️ Check correlations'}")
    
    return feats
