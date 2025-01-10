
import numpy as np
import pandas as pd


def clean_dataset(df):
    """Clean and preprocess the dataset"""
    df = df.copy()
    
    # Remove extreme values
    df = df[df['time_to_peak'] > 30]  # Remove peaks before 30s
    df = df[df['time_to_peak'] <= 1020]  # Only consider first 17 minutes
    
    # Convert time features
    df['creation_time'] = pd.to_datetime(df['creation_time'])
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('rsi'):
            df[col] = df[col].fillna(50)  # Neutral RSI
        else:
            df[col] = df[col].fillna(df[col].median())
    
    return df