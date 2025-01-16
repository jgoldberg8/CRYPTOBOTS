
import numpy as np
import pandas as pd


def clean_dataset(df):
    """
    Specialized cleaning function for hit peak before 30 seconds model
    Focuses on early window features and avoids using time_to_peak for feature generation
    """
    # Create a copy at the start to avoid fragmentation
    df = df.copy()
    df = df.dropna(subset=critical_cols)
    df = df[df['time_to_peak'] < 180]
    
    # Drop records with missing critical initial window features
    critical_cols = [
        'initial_market_cap',
        'hit_peak_before_30',
        'peak_market_cap'
    ]
    
    # Drop rows with missing critical columns
    df = df.dropna(subset=critical_cols)
    
    # Create new volume and pressure features based on early windows
    df['volume_pressure_early'] = df['volume_0to5s'] / (df['initial_market_cap'] + 1)
    df['buy_sell_ratio_early'] = df['buy_pressure_0to5s'].clip(0, 1)
    
    # Calculate early transaction features
    df['transaction_density_5s'] = df['transaction_count_0to5s'] / 5
    df['transaction_density_10s'] = df['transaction_count_0to10s'] / 10
    
    # Log transform heavily skewed early window features
    early_skewed_features = [
        'volume_0to5s', 
        'volume_0to10s', 
        'trade_amount_variance_0to5s', 
        'trade_amount_variance_0to10s'
    ]
    
    for feature in early_skewed_features:
        # Only log transform if the feature exists
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
    
    # Calculate temporal decay features for early windows
    windows = ['5s', '10s']
    for window in windows:
        volume_cols = [col for col in df.columns if f'volume_0to{window}' in col]
        if volume_cols:
            df[f'volume_decay_{window}'] = df[volume_cols].pct_change(axis=1).mean(axis=1)
        
        transaction_cols = [col for col in df.columns if f'transaction_count_0to{window}' in col]
        if transaction_cols:
            df[f'transaction_decay_{window}'] = df[transaction_cols].pct_change(axis=1).mean(axis=1)
    
    # Add creation time cyclic features
    df['creation_time_numeric'] = pd.to_datetime(df['creation_time']).dt.hour + pd.to_datetime(df['creation_time']).dt.minute / 60
    df['creation_time_sin'] = np.sin(2 * np.pi * df['creation_time_numeric'] / 24.0)
    df['creation_time_cos'] = np.cos(2 * np.pi * df['creation_time_numeric'] / 24.0)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('rsi'):
            # Neutral RSI value
            df[col] = df[col].fillna(50)
        elif col.endswith('_decay'):
            # No decay for missing values
            df[col] = df[col].fillna(0)
        elif col in ['volume_pressure_early', 'buy_sell_ratio_early']:
            # Default to 0 for these new features
            df[col] = df[col].fillna(0)
        else:
            # Use median for other features
            df[col] = df[col].fillna(df[col].median())
    
    # If hit_peak_before_30 doesn't exist, create it
    if 'hit_peak_before_30' not in df.columns:
        # This should be added AFTER all other processing
        # Let whoever uses this function decide how to create the column
        pass
    
    # Convert all numeric columns to float32
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    return df