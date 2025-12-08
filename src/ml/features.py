import pandas as pd
import numpy as np

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

# FULL FEATURE PIPELINE

def build_features(df, 
                   target_type='next',  # 'next' or 'windowed'
                   target_window=5,
                   rolling_window=20,
                   rolling_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                   lag_cols=['cycle_time', 'temperature', 'vibration', 'pressure'],
                   lag_periods=[1, 2, 3],
                   cat_cols=['operator', 'maintenance_type']):
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw DataFrame
        target_type: 'next' (single cycle) or 'windowed' (next N cycles)
        target_window: If windowed, how many cycles ahead
        rolling_window: Size of rolling window for statistics
        rolling_cols: Columns for rolling features
        lag_cols: Columns for lag features
        lag_periods: List of lag periods
        cat_cols: Categorical columns to encode
    
    Returns:
        DataFrame with all features engineered
    """
    df = df.copy()
    
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
    
    # 3. Rolling features
    df = create_rolling_features(df, columns=rolling_cols, window=rolling_window)
    print(f"✓ Created rolling features (window={rolling_window}) for: {rolling_cols}")
    
    # 4. Lag features
    df = create_lag_features(df, columns=lag_cols, lags=lag_periods)
    print(f"✓ Created lag features (lags={lag_periods}) for: {lag_cols}")
    
    print(f"Feature engineering complete! Final shape: {df.shape}")
    
    return df


# FEATURE SELECTION HELPER

def get_feature_columns(df, target_col='downtime_next'):
    """
    Auto-detect feature columns by excluding non-feature columns.
    
    Args:
        df: DataFrame with all columns
        target_col: Name of target column
    
    Returns:
        List of feature column names
    """
    exclude = [
        'timestamp', 'cycle_id', 'production_order_id',
        'downtime_next', 'downtime_next_5', 'downtime_next_10',  # all potential targets
        'operator', 'maintenance_type',  # original categorical (we use _encoded versions)
        target_col  # make sure target is excluded
    ]
    
    features = [c for c in df.columns if c not in exclude]
    
    return features