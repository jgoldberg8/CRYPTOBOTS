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
    
    # Initialize dictionary to store all new features
    new_features = {}
    
    # Drop records where critical initial windows are missing
    critical_cols = [
        'transaction_count_0to10s',
        'initial_market_cap',
        'peak_market_cap',
        'time_to_peak'
    ]
    df = df.dropna(subset=critical_cols)
    df = df[df['time_to_peak'] > 30]
    
    # Identify different feature types
    momentum_cols = [col for col in df.columns if 'momentum' in col]
    volume_cols = [col for col in df.columns if 'volume' in col]
    market_cap_cols = ['initial_market_cap', 'peak_market_cap']
    variance_cols = [col for col in df.columns if 'trade_amount_variance' in col]
    
    # Calculate all new features first
    new_features['volume_pressure'] = df['volume_0to30s'] / (df['initial_market_cap'] + 1)
    new_features['buy_sell_ratio'] = df['buy_pressure_0to30s'].clip(0, 1)
    new_features['volume_acceleration'] = df['volume_0to30s'].diff().diff()
    
    # Momentum-specific features
    new_features['momentum_consistency'] = df[momentum_cols].std(axis=1)
    new_features['momentum_direction'] = np.sign(df[momentum_cols].mean(axis=1))
    new_features['momentum_magnitude'] = np.abs(df[momentum_cols]).mean(axis=1)
    
    # Volume-based features
    new_features['volume_growth_rate'] = (df['volume_20to30s'] - df['volume_0to10s']) / (df['volume_0to10s'] + 1e-8)
    new_features['market_pressure'] = df['volume_0to30s'] * df['buy_pressure_0to30s'] / (df['initial_market_cap'] + 1)
    
    # Add temporal decay features
    windows = ['5s', '10s', '20s', '30s']
    for window in windows:
        # Volume decay
        volume_window_cols = [col for col in df.columns if f'volume_{window}' in col]
        if volume_window_cols:
            new_features[f'volume_decay_{window}'] = df[volume_window_cols].pct_change(axis=1).mean(axis=1)
        
        # Transaction decay
        transaction_cols = [col for col in df.columns if f'transaction_count_{window}' in col]
        if transaction_cols:
            new_features[f'transaction_decay_{window}'] = df[transaction_cols].pct_change(axis=1).mean(axis=1)
            
        # Momentum decay
        momentum_window_cols = [col for col in df.columns if f'momentum_{window}' in col]
        if momentum_window_cols:
            new_features[f'momentum_decay_{window}'] = df[momentum_window_cols].pct_change(axis=1).mean(axis=1)
    
    # Add interaction features
    new_features['momentum_volume_interaction'] = new_features['momentum_magnitude'] * new_features['volume_pressure']
    new_features['momentum_pressure_ratio'] = new_features['momentum_magnitude'] / (new_features['market_pressure'] + 1e-10)
    
    # Add all new features at once
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    # Log transform positive-only features
    for col in volume_cols:
        df[col] = df[col].clip(lower=1e-10)
        df[col] = np.log1p(df[col])
        
    for col in market_cap_cols:
        df[col] = df[col].clip(lower=1e-10)
        df[col] = np.log1p(df[col])
        
    for col in variance_cols:
        df[col] = df[col].clip(lower=1e-10)
        df[col] = np.log1p(df[col])
    
    # Fill missing values with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('rsi'):
            df[col] = df[col].fillna(50)
        elif col in momentum_cols or 'momentum_decay' in col:
            df[col] = df[col].fillna(0)
        elif 'decay' in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Log transform time_to_peak
    df['time_to_peak'] = np.log1p(df['time_to_peak'])
    
    # Convert all numeric columns to float32
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    return df




def custom_market_cap_loss(pred, target, underprediction_penalty=3.5, scale_factor=100):
    """Enhanced loss function with dynamic scaling and proper handling of transformed values"""
    # Since we're working with log-transformed values, we need to scale appropriately
    diff = pred - target
    
    # Calculate base loss with higher penalty for larger values
    base_loss = torch.where(diff < 0,
                         torch.abs(diff) * underprediction_penalty,
                         torch.abs(diff))
    
    # Add scale-dependent penalty (larger values get proportionally higher weight)
    # We divide by scale_factor to keep the scaling reasonable
    scale_weight = torch.log1p(torch.abs(target)) / scale_factor
    weighted_loss = base_loss * (1 + scale_weight)
    
    return torch.mean(weighted_loss)



class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Add skip connection
        self.skip_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        skip_connection = self.skip_projection(x.mean(dim=1))
        return attended + skip_connection