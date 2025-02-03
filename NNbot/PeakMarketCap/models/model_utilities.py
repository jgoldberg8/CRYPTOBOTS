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
    # Filter for only tokens that actually increased
    df = df[(df['hit_peak_before_30'] == False) & (df['percent_increase'] > 0.5)]

    
    
    # Identify different feature types
    momentum_cols = [col for col in df.columns if 'momentum' in col]
    volume_cols = [col for col in df.columns if 'volume' in col]
    market_cap_cols = ['initial_market_cap', 'peak_market_cap']
    variance_cols = [col for col in df.columns if 'trade_amount_variance' in col]
    
    # Calculate base features first
    new_features['volume_pressure'] = df['volume_0to30s'] / (df['initial_market_cap'] + 1)
    new_features['buy_sell_ratio'] = df['buy_pressure_0to30s'].clip(0, 1)
    new_features['volume_acceleration'] = df['volume_0to30s'].diff().diff()
    
    # Calculate momentum-specific features first since they're used later
    new_features['momentum_consistency'] = df[momentum_cols].std(axis=1)
    new_features['momentum_direction'] = np.sign(df[momentum_cols].mean(axis=1))
    new_features['momentum_magnitude'] = np.abs(df[momentum_cols]).mean(axis=1)
    
    # Create temporary DataFrame for early features calculation
    temp_df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    # Now add early-stage specific features using temp_df
    new_features['early_stage_volume'] = np.where(df['peak_market_cap'] < df['peak_market_cap'].median() * 0.5,
                                                df['volume_0to10s'], 0)
    new_features['early_momentum_intensity'] = np.where(df['peak_market_cap'] < df['peak_market_cap'].median() * 0.5,
                                                      temp_df['momentum_magnitude'] * 1.5, temp_df['momentum_magnitude'])
    
    # Add initial growth rate features
    new_features['initial_growth_rate'] = (df['volume_5to10s'] - df['volume_0to5s']) / (df['volume_0to5s'] + 1e-8)
    new_features['early_pressure_indicator'] = temp_df['volume_pressure'] * temp_df['momentum_magnitude'] * \
                                             (df['peak_market_cap'] < df['peak_market_cap'].median())
    
    # Rest of your feature calculations...
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
    
    # Create a temporary DataFrame with the new features calculated so far
    temp_df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    # Add range-based features using the temporary DataFrame
    quantiles = df['peak_market_cap'].quantile([0.25, 0.5, 0.75])
    new_features['high_value_momentum'] = (temp_df['momentum_magnitude'] * 
                                         (df['peak_market_cap'] > quantiles[0.75]).astype(float))
    new_features['mid_value_momentum'] = (temp_df['momentum_magnitude'] * 
                                        ((df['peak_market_cap'] > quantiles[0.25]) & 
                                         (df['peak_market_cap'] <= quantiles[0.75])).astype(float))
    
    # Add interaction features using the temporary DataFrame
    new_features['momentum_volume_interaction'] = temp_df['momentum_magnitude'] * temp_df['volume_pressure']
    new_features['momentum_pressure_ratio'] = temp_df['momentum_magnitude'] / (temp_df['market_pressure'] + 1e-10)
    
    # Add volume-momentum interactions by range
    new_features['high_value_volume_momentum'] = new_features['high_value_momentum'] * temp_df['volume_pressure']
    new_features['mid_value_volume_momentum'] = new_features['mid_value_momentum'] * temp_df['volume_pressure']
    # Add to clean_dataset
    new_features['high_value_indicator'] = (df['peak_market_cap'] > df['peak_market_cap'].median() * 1.5).astype(float)
    new_features['volume_high_value_interaction'] = new_features['volume_pressure'] * new_features['high_value_indicator']
    new_features['momentum_high_value_interaction'] = new_features['momentum_magnitude'] * new_features['high_value_indicator']
    
    # Add all new features to original dataframe
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

def custom_market_cap_loss(pred, target):
    # Convert from log space
    pred_orig = torch.expm1(pred)
    target_orig = torch.expm1(target)
    
    # Calculate relative error
    relative_error = torch.abs(pred_orig - target_orig) / (target_orig + 1)
    
    # Add range-specific weighting
    weights = torch.where(
        target_orig > 500,
        torch.ones_like(target_orig) * 3.0,  # High range
        torch.where(
            target_orig > 100,
            torch.ones_like(target_orig) * 2.0,  # Medium range
            torch.ones_like(target_orig) * 1.0  # Low range
        )
    )
    
    loss = relative_error * weights
    return torch.mean(loss)

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
        # x shape: [batch_size, seq_len, hidden_size]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # [batch_size, hidden_size]
        skip_connection = self.skip_projection(torch.mean(x, dim=1))  # [batch_size, hidden_size]
        attended = weighted_sum + skip_connection  # [batch_size, hidden_size]
        # Reshape back to 3D for consistency
        return attended.unsqueeze(1)  # [batch_size, 1, hidden_size]
    


class RangeAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, value_range):
        # x shape: [batch_size, seq_len, hidden_size]
        # value_range shape: [batch_size, hidden_size]
        
        # First make value_range match x's sequence dimension
        value_range = value_range.unsqueeze(1)  # [batch_size, 1, hidden_size]
        value_range = value_range.expand(-1, x.size(1), -1)  # [batch_size, seq_len, hidden_size]
        
        # Concatenate along feature dimension
        attention_input = torch.cat([x, value_range], dim=-1)
        attention_weights = self.attention(attention_input)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended