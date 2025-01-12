import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import wandb
from tqdm import tqdm

class MultiGranularTimeDataset(Dataset):
    def __init__(self, df, scaler=None, train=True, max_sequence_length=1020):
        df = df.copy()
        self.max_sequence_length = max_sequence_length
        
        # Convert time features with proper handling
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        df['creation_time_numeric'] = (
            df['creation_time'].dt.hour * 3600 + 
            df['creation_time'].dt.minute * 60 + 
            df['creation_time'].dt.second
        ) / 86400.0  # Normalize to [0, 1]
        
        df['creation_time_sin'] = np.sin(2 * np.pi * df['creation_time_numeric'])
        df['creation_time_cos'] = np.cos(2 * np.pi * df['creation_time_numeric'])
        
        # Base features for each time window
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
        
        # Define time granularities and their windows
        self.time_granularities = {
            '5s': self._generate_windows(5, max_sequence_length),
            '10s': self._generate_windows(10, max_sequence_length),
            '20s': self._generate_windows(20, max_sequence_length),
            '30s': self._generate_windows(30, max_sequence_length),
            '60s': self._generate_windows(60, max_sequence_length)
        }
        
        # Global features
        self.global_features = [
            'initial_investment_ratio',
            'initial_market_cap',
            'peak_market_cap',
            'creation_time_sin',
            'creation_time_cos'
        ]
        
        # Initialize scalers with robust scaling for better outlier handling
        if train:
            if scaler is None:
                self.scalers = {
                    'global': RobustScaler(),
                    'target': RobustScaler(quantile_range=(10, 90)),  # More robust to outliers
                }
                for granularity in self.time_granularities.keys():
                    self.scalers[granularity] = RobustScaler()
                self.scaled_data = self._preprocess_data(df, fit=True)
            else:
                self.scalers = scaler
                self.scaled_data = self._preprocess_data(df, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test data")
            self.scalers = scaler
            self.scaled_data = self._preprocess_data(df, fit=False)

    def _generate_windows(self, step_size, max_length):
        """Generate time windows for a specific granularity"""
        return [f"{start}to{min(start + step_size, max_length)}" 
                for start in range(0, max_length, step_size)]

    def _extract_window_features(self, row, windows):
        """Extract features for time windows with proper handling of missing data"""
        features = []
        valid_count = 0
        
        for window in windows:
            window_features = []
            has_data = False
            
            for feature in self.base_features:
                col_name = f"{feature}_{window}s"
                if col_name in row.index and pd.notnull(row[col_name]):
                    window_features.append(float(row[col_name]))
                    has_data = True
                else:
                    window_features.append(0.0)
            
            features.append(window_features)
            if has_data:
                valid_count += 1
                
        return np.array(features, dtype=np.float32), valid_count

    def _preprocess_data(self, df, fit=False):
        """Preprocess data with improved error handling and scaling"""
        processed_data = {
            'granular_features': {},
            'sequence_lengths': {},
            'global_features': None,
            'targets': None,
            'peak_targets': None
        }
        
        # Process each granularity
        for granularity, windows in self.time_granularities.items():
            all_sequences = []
            lengths = []
            
            for idx in range(len(df)):
                sequence, valid_count = self._extract_window_features(
                    df.iloc[idx], windows
                )
                all_sequences.append(sequence)
                lengths.append(valid_count)
            
            # Scale features with proper reshaping
            all_sequences = np.array(all_sequences)
            original_shape = all_sequences.shape
            reshaped = all_sequences.reshape(-1, len(self.base_features))
            
            if fit:
                scaled = self.scalers[granularity].fit_transform(reshaped)
            else:
                scaled = self.scalers[granularity].transform(reshaped)
            
            scaled = scaled.reshape(original_shape)
            
            processed_data['granular_features'][granularity] = scaled
            processed_data['sequence_lengths'][granularity] = np.array(lengths)
        
        # Process global features
        global_features = df[self.global_features].fillna(0).values
        if fit:
            processed_data['global_features'] = self.scalers['global'].fit_transform(global_features)
        else:
            processed_data['global_features'] = self.scalers['global'].transform(global_features)
        
        # Process target with outlier handling
        target_data = df[['time_to_peak']].values
        if fit:
            processed_data['targets'] = self.scalers['target'].fit_transform(target_data)
        else:
            processed_data['targets'] = self.scalers['target'].transform(target_data)
        
        # Add peak detection targets
        if 'is_peak' in df.columns:
            processed_data['peak_targets'] = df['is_peak'].values
        
        return processed_data

    def __len__(self):
        return len(self.scaled_data['targets'])

    def __getitem__(self, idx):
        sample = {
            'global_features': torch.FloatTensor(self.scaled_data['global_features'][idx]),
            'targets': torch.FloatTensor(self.scaled_data['targets'][idx])
        }
        
        for granularity in self.time_granularities.keys():
            sample[f'features_{granularity}'] = torch.FloatTensor(
                self.scaled_data['granular_features'][granularity][idx]
            )
            sample[f'length_{granularity}'] = torch.LongTensor(
                [self.scaled_data['sequence_lengths'][granularity][idx]]
            )
        
        if self.scaled_data['peak_targets'] is not None:
            sample['peak_target'] = torch.FloatTensor([self.scaled_data['peak_targets'][idx]])
        
        return sample


    def get_scalers(self):
        return self.scalers

def create_multi_granular_loaders(df_train, df_val, batch_size=32):
    """Helper function to create train and validation data loaders"""
    # Create training dataset
    train_dataset = MultiGranularTimeDataset(df_train, train=True)
    
    # Create validation dataset with scalers from training
    val_dataset = MultiGranularTimeDataset(
        df_val,
        scaler=train_dataset.get_scalers(),
        train=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader