
import numpy as np
import pandas as pd


def clean_dataset(df):
    """
    Clean and preprocess the dataset
    
    Args:
        df (pd.DataFrame): Raw dataframe with token trading data
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()
    
    # Remove tokens that peak before 30 seconds
    df = df[df['time_to_peak'] > 30]
    
    # Remove peaks after 17 minutes (1020 seconds)
    df = df[df['time_to_peak'] <= 1020]
    
    # Fill missing values appropriately
    for col in df.columns:
        if col.startswith('rsi'):
            df[col] = df[col].fillna(50)  # Neutral RSI
        elif 'volume' in col:
            df[col] = df[col].fillna(0)
        elif 'price' in col:
            df[col] = df[col].fillna(method='ffill').fillna(0)
        else:
            df[col] = df[col].fillna(0)
    
    return df