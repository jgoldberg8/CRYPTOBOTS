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
        df['creation_time_numeric'] = pd.to_datetime(df['creation_time']).dt.hour + pd.to_datetime(df['creation_time']).dt.minute / 60

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
            'buy_sell_ratio',
            'creation_time_numeric',
            'percent_increase_at_30'
        ] 
        
        self.targets = ['percent_increase']


        
        # Scale the data
        if train:
            if scaler is None:
                self.global_scaler = StandardScaler()  # Separate scaler for global features
                self.target_scaler = StandardScaler()  # Separate scaler for targets
                self.scaled_data = self._preprocess_data(df, fit=True)
                self.scaler = scaler
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
        target_data = np.log1p(df[self.targets].values)
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
    


    def _calculate_sample_weights(self, df):
        """Calculate sample weights to handle class imbalance"""
        percent_increases = df['percent_increase'].values
        
        # Initialize weights array
        weights = np.ones_like(percent_increases, dtype=np.float32)
        
        # Define ranges and target proportions
        ranges = [
            (0, 100),      # Low range
            (100, 500),    # Medium range
            (500, np.inf)  # High range
        ]
        
        # Count samples in each range
        range_counts = {}
        for low, high in ranges:
            mask = (percent_increases >= low) & (percent_increases < high)
            range_counts[(low, high)] = np.sum(mask)
        
        # Calculate inverse frequency weights
        total_samples = len(percent_increases)
        for (low, high), count in range_counts.items():
            if count > 0:
                mask = (percent_increases >= low) & (percent_increases < high)
                # Weight = total_samples / (num_ranges * count)
                weights[mask] = total_samples / (3 * count)
                
                # Additional scaling within each range
                range_values = percent_increases[mask]
                if len(range_values) > 0:
                    min_val = np.min(range_values)
                    max_val = np.max(range_values)
                    if max_val > min_val:
                        # Scale weights within range based on value
                        relative_pos = (range_values - min_val) / (max_val - min_val)
                        # More weight to higher values within each range
                        weights[mask] *= (1 + relative_pos)
        
        # Add small constant to avoid zero weights
        weights += 1e-5
        
        # Normalize weights to sum to 1
        weights /= weights.sum()
        
        return weights


    def _calculate_feature_importance(self, df):
        """Calculate feature importance scores"""
        correlations = df.corr()['peak_market_cap'].abs()
        return correlations / correlations.sum()



