import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext



    



def clean_dataset(df):
    # Create a copy at the start to avoid fragmentation
    df = df.copy()
    
    # Drop records where no peak was recorded (time_to_peak is 0)
    # df = df[df['time_to_peak'] > 30]
    
    # Drop records where critical initial windows are missing
    critical_cols = [
        'transaction_count_0to10s',
        'initial_market_cap',
        'peak_market_cap',
        'time_to_peak'
    ]
    df = df.dropna(subset=critical_cols)
    df = df[df['time_to_peak'] > 30]
    
    # Create new columns
    df['volume_pressure'] = df['volume_0to30s'] / (df['initial_market_cap'] + 1)
    df['buy_sell_ratio'] = df['buy_pressure_0to30s'].clip(0, 1)
    
    # Log transform heavily skewed numerical features
    skewed_features = ['volume', 'trade_amount_variance', 'initial_market_cap', 'peak_market_cap']
    for feature in skewed_features:
        cols = [col for col in df.columns if feature in col]
        for col in cols:
            df[col] = np.log1p(df[col])
    
    # Add temporal decay features
    windows = ['5s', '10s', '20s', '30s']
    for window in windows:
        volume_cols = [col for col in df.columns if f'volume_{window}' in col]
        if volume_cols:
            df[f'volume_decay_{window}'] = df[volume_cols].pct_change(axis=1).mean(axis=1)
        
        transaction_cols = [col for col in df.columns if f'transaction_count_{window}' in col]
        if transaction_cols:
            df[f'transaction_decay_{window}'] = df[transaction_cols].pct_change(axis=1).mean(axis=1)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('rsi'):
            df[col] = df[col].fillna(50)
        else:
            df[col] = df[col].fillna(0)
    
    # Log transform time_to_peak
    df['time_to_peak'] = np.log1p(df['time_to_peak'])
    
    # Convert all numeric columns to float32
    # Get updated list of numeric columns after all transformations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    return df





def custom_market_cap_loss(pred, target, underprediction_penalty=3.5):
    """
    Asymmetric loss function with tunable penalty for underprediction
    
    Args:
        pred (torch.Tensor): Model predictions
        target (torch.Tensor): True target values
        underprediction_penalty (float): Multiplier for underprediction penalty
    """
    diff = pred - target
    loss = torch.where(diff < 0,
                      torch.abs(diff) * underprediction_penalty,  # Tunable penalty
                      torch.abs(diff))
    return torch.mean(loss)