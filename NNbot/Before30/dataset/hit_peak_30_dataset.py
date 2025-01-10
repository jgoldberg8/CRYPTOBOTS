import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class HitPeakBefore30Dataset(Dataset):
    def __init__(self, df, scaler=None, train=True):
        df = df.copy()
        
        
        # Convert time features
        df['creation_time_numeric'] = pd.to_datetime(df['creation_time']).dt.hour + pd.to_datetime(df['creation_time']).dt.minute / 60
        df['creation_time_sin'] = np.sin(2 * np.pi * df['creation_time_numeric'] / 24.0)
        df['creation_time_cos'] = np.cos(2 * np.pi * df['creation_time_numeric'] / 24.0)
        
        # Add volume pressure features
        for window in ['5s', '10s', '20s', '30s']:
            vol_col = f'volume_0to{window}'
            if vol_col in df.columns:
                df[f'volume_pressure_{window}'] = df[vol_col] / (df['initial_market_cap'] + 1e-8)
        
        # Convert numeric columns to float32
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Base features
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
        
        # Global features for hit peak prediction
        self.global_features = [
            'initial_investment_ratio', 
            'initial_market_cap',
            'volume_pressure_5s',
            'volume_pressure_10s',
            'creation_time_sin',
            'creation_time_cos',
            'creation_time_numeric'
        ]
        
        # Target is only hit_peak_before_30
        self.targets = ['hit_peak_before_30']
        
        # Scaling
        if train:
            if scaler is None:
                self.global_scaler = StandardScaler()
                self.scaled_data = self._preprocess_data(df, fit=True)
            else:
                self.global_scaler = scaler['global']
                self.scaled_data = self._preprocess_data(df, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for test data")
            self.global_scaler = scaler['global']
            self.scaled_data = self._preprocess_data(df, fit=False)
        
        # Calculate quality features
        self.quality_features = self._calculate_quality_features(df)

    def _preprocess_data(self, df, fit=False):
        # Process time windows
        processed_data = {}
        for window_type, windows in self.time_windows.items():
            window_data = []
            for window in windows:
                features = []
                for feature in self.base_features:
                    col_name = f"{feature}_{window}s"
                    if col_name in df.columns:
                        features.append(df[col_name].values)
                    else:
                        features.append(np.zeros(len(df)))
                window_data.append(np.stack(features, axis=1))
            processed_data[window_type] = np.stack(window_data, axis=1)
        
        # Process global features
        global_data = df[self.global_features].values
        if fit:
            global_data = self.global_scaler.fit_transform(global_data)
        else:
            global_data = self.global_scaler.transform(global_data)
        
        # Process target (binary hit_peak_before_30)
        target_data = df[self.targets].values.astype(float)
        
        return {
            'data': processed_data,
            'global': global_data,
            'targets': target_data
        }

    def _calculate_quality_features(self, df):
        # Basic completeness
        completeness = df.notna().mean(axis=1).values
        
        # Weighted active intervals
        active_intervals = 0
        total_weight = 0
        
        for window_type, windows in self.time_windows.items():
            weight = 1.0 / float(window_type[:-1])
            for window in windows:
                col_name = f"transaction_count_{window}s"
                if col_name in df.columns:
                    active_intervals += (df[col_name] > 0).astype(float) * weight
                    total_weight += weight
        
        active_intervals = active_intervals / (total_weight + 1e-8)
        
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

        # Get target (binary hit_peak_before_30)
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