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


class TokenDataset(Dataset):
    def __init__(self, df, scaler=None, train=True):
        df = df.copy()
        # Convert all numeric columns to float32
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Features for each window
        self.base_features = [
            'transaction_count',
            'buy_pressure',
            'volume',
            'rsi',
            'price_volatility',
            'volume_volatility',
            'momentum',
            'trade_amount_variance',
            'transaction_rate',
            'trade_concentration',
            'unique_wallets'
        ]
        
        # Time windows
        self.time_windows = {
            '5s': ['0to5', '5to10', '10to15', '15to20', '20to25', '25to30'],
            '10s': ['0to10', '10to20', '20to30'],
            '20s': ['0to20'],
            '30s': ['0to30']
        }
        
        # Global features including new derived features
        self.global_features = [
            'initial_investment_ratio', 
            'initial_market_cap',
            'volume_pressure',
            'buy_sell_ratio'
        ] 
        
        self.targets = ['peak_market_cap', 'time_to_peak']
        
        # Scale the data
        if train:
            if scaler is None:
                self.global_scaler = StandardScaler()  # Separate scaler for global features
                self.target_scaler = StandardScaler()  # Separate scaler for targets
                self.scaled_data = self._preprocess_data(df, fit=True)
            else:
                self.global_scaler = scaler['global']
                self.target_scaler = scaler['target']
                self.scaled_data = self._preprocess_data(df, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for test data")
            self.global_scaler = scaler['global']
            self.target_scaler = scaler['target']
            self.scaled_data = self._preprocess_data(df, fit=False)
            
        self.quality_features = self._calculate_quality_features(df)


            


    def _preprocess_data(self, df, fit=False):
        # Process each time window
        processed_data = {}
        
        # Process 5s intervals
        for window_type, windows in self.time_windows.items():
            window_data = []
            for window in windows:
                features = []
                for feature in self.base_features:
                    col_name = f"{feature}_{window}s"
                    features.append(df[col_name].values)
                window_data.append(np.stack(features, axis=1))
            processed_data[window_type] = np.stack(window_data, axis=1)
            
        # Process global features
        global_data = df[self.global_features].values
        if fit:
            global_data = self.global_scaler.fit_transform(global_data)
        else:
            global_data = self.global_scaler.transform(global_data)
            
        # Process targets
        target_data = df[self.targets].values
        if fit:
            target_data = self.target_scaler.fit_transform(target_data)
        else:
            target_data = self.target_scaler.transform(target_data)
            
        return {
            'data': processed_data,
            'global': global_data,
            'targets': target_data
        }
        


    def _calculate_quality_features(self, df):
        """Calculate data quality features"""
        # Calculate completeness ratio
        completeness = df.notna().mean(axis=1).values
        
        # Calculate active intervals
        active_intervals = df[[f"transaction_count_{window}s" 
                             for windows in self.time_windows.values() 
                             for window in windows]].gt(0).sum(axis=1).values
        
        return np.stack([completeness, active_intervals], axis=1)
        


    def __len__(self):
        return len(self.scaled_data['targets'])
        
    def __getitem__(self, idx):
        # Get time window data
        x_5s = torch.FloatTensor(self.scaled_data['data']['5s'][idx])
        x_10s = torch.FloatTensor(self.scaled_data['data']['10s'][idx])
        x_20s = torch.FloatTensor(self.scaled_data['data']['20s'][idx])
        x_30s = torch.FloatTensor(self.scaled_data['data']['30s'][idx])

        # Get global features
        global_features = torch.FloatTensor(self.scaled_data['global'][idx])

        # Get quality features
        quality_features = torch.FloatTensor(self.quality_features[idx])

        # Get targets
        targets = torch.FloatTensor(self.scaled_data['targets'][idx])

        return {
            'x_5s': x_5s,
            'x_10s': x_10s,
            'x_20s': x_20s,
            'x_30s': x_30s,
            'global_features': global_features,
            'quality_features': quality_features,
            'targets': targets
        }
    
        

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
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return False    
    


class TimePredictionLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred_mean, pred_log_var, target):
        # Gaussian negative log likelihood
        precision = torch.exp(-pred_log_var)
        nll_loss = 0.5 * (pred_log_var + (pred_mean - target)**2 * precision)
        
        # Combine with MSE loss
        mse_loss = self.mse(pred_mean, target)
        
        # Add relative error penalty for longer time predictions
        relative_error = torch.abs(pred_mean - target) / (target + 1e-6)
        relative_loss = torch.mean(relative_error)
        
        return mse_loss + self.alpha * nll_loss.mean() + (1 - self.alpha) * relative_loss
    


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