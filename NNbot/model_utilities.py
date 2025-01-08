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




class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Calculate attention weights
        weights = self.attention(x)
        # Apply attention weights
        return torch.sum(weights * x, dim=1)
    


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'EarlyStopping: Initializing best loss to {val_loss:.6f}')
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: Loss did not improve from {self.best_loss:.6f}. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'EarlyStopping: Early stopping triggered after {self.patience} epochs without improvement')
                return True
        else:
            if self.verbose:
                print(f'EarlyStopping: Loss improved from {self.best_loss:.6f} to {val_loss:.6f}. Resetting counter.')
            self.best_loss = val_loss
            self.counter = 0
            
        return False


class TimePredictionLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred_mean, pred_log_var, target):
        # Clamp pred_log_var for numerical stability
        pred_log_var = torch.clamp(pred_log_var, min=-20, max=20)
        
        # Gaussian negative log likelihood with stability fixes
        precision = torch.exp(-pred_log_var) + 1e-6  # Add small epsilon
        se = (pred_mean - target)**2
        nll_loss = 0.5 * (pred_log_var + se * precision)
        
        # MSE loss (always non-negative)
        mse_loss = self.mse(pred_mean, target)
        
        # Relative error with improved numerical stability
        abs_diff = torch.abs(pred_mean - target)
        relative_error = abs_diff / (torch.abs(target) + 1e-6)
        relative_loss = torch.mean(relative_error)
        
        # Combine all terms
        total_loss = (
            mse_loss + 
            self.alpha * torch.mean(torch.clamp(nll_loss, min=0.0)) + 
            (1 - self.alpha) * relative_loss
        )
        
        return total_loss
    


def clean_dataset(df):
    # Create a copy at the start to avoid fragmentation
    df = df.copy()
    
    # Drop records where no peak was recorded (time_to_peak is 0)
    df = df[df['time_to_peak'] > 0]
    
    # Drop records where critical initial windows are missing
    critical_cols = [
        'transaction_count_0to5s',
        'transaction_count_0to10s',
        'initial_market_cap',
        'peak_market_cap'
    ]
    df = df.dropna(subset=critical_cols)
    
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

def add_data_quality_features(df):
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Calculate basic quality metrics
    df['data_completeness'] = df.notna().mean(axis=1)
    df['active_intervals'] = df[[col for col in df.columns 
                              if col.startswith('transaction_count_')]].gt(0).sum(axis=1)
    
    # Add volatility quality metrics
    price_vol_cols = [col for col in df.columns if 'price_volatility' in col]
    df['price_stability'] = 1 / (df[price_vol_cols].mean(axis=1) + 1)
    
    # Add trading consistency metrics
    volume_cols = [col for col in df.columns if 'volume_' in col and col.endswith('s')]
    df['trading_consistency'] = 1 - df[volume_cols].std(axis=1) / (df[volume_cols].mean(axis=1) + 1e-8)
    
    return df

def train_val_split(df):
    # Create time buckets for stratification
    df['time_bucket'] = pd.qcut(df['time_to_peak'], q=5, 
                               labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    # Split preserving the distribution of time buckets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['time_bucket'], 
        random_state=42
    )
    
    # Drop the temporary bucketing column
    train_df = train_df.drop('time_bucket', axis=1)
    val_df = val_df.drop('time_bucket', axis=1)
    
    return train_df, val_df




def custom_market_cap_loss(pred, target):
    # Asymmetric loss function that penalizes underprediction more heavily
    diff = pred - target
    loss = torch.where(diff < 0, 
                      torch.abs(diff) * 1.5,  # Higher penalty for underprediction
                      torch.abs(diff))
    return torch.mean(loss)