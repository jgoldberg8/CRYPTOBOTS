import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class MultiGranularTimeDataset(Dataset):
    def __init__(self, df, scaler=None, train=True, max_sequence_length=1020):
        df = df.copy()
        self.max_sequence_length = max_sequence_length
        
        # Convert time features
        df['creation_time_numeric'] = pd.to_datetime(df['creation_time']).dt.hour + pd.to_datetime(df['creation_time']).dt.minute / 60
        df['creation_time_sin'] = np.sin(2 * np.pi * df['creation_time_numeric'] / 24.0)
        df['creation_time_cos'] = np.cos(2 * np.pi * df['creation_time_numeric'] / 24.0)
        
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
        
        # Global features that don't vary with time
        self.global_features = [
            'initial_investment_ratio',
            'initial_market_cap',
            'creation_time_sin',
            'creation_time_cos',
            'creation_time_numeric'
        ]
        
        # Initialize scalers
        if train:
            if scaler is None:
                self.scalers = {
                    'global': StandardScaler(),
                    'target': StandardScaler()
                }
                for granularity in self.time_granularities.keys():
                    self.scalers[granularity] = StandardScaler()
                self.scaled_data = self._preprocess_data(df, fit=True)
            else:
                self.scalers = scaler
                self.scaled_data = self._preprocess_data(df, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for test data")
            self.scalers = scaler
            self.scaled_data = self._preprocess_data(df, fit=False)

    def _generate_windows(self, step_size, max_length):
        """Generate time windows for a specific granularity"""
        windows = []
        for start in range(0, max_length, step_size):
            end = start + step_size
            if end <= max_length:
                windows.append(f"{start}to{end}")
        return windows

    def _extract_window_features(self, row, granularity, windows):
        """Extract features for a specific granularity's time windows"""
        features = []
        valid_count = 0
        
        for window in windows:
            window_features = []
            has_data = False
            
            for feature in self.base_features:
                col_name = f"{feature}_{window}s"
                if col_name in row.index and not pd.isna(row[col_name]):
                    window_features.append(row[col_name])
                    has_data = True
                else:
                    window_features.append(0.0)
            
            features.append(window_features)
            if has_data:
                valid_count += 1
                
        return np.array(features, dtype=np.float32), valid_count

    def _preprocess_data(self, df, fit=False):
        processed_data = {
            'granular_features': {},
            'sequence_lengths': {},
            'global_features': None,
            'targets': None
        }
        
        # Process each granularity separately
        for granularity, windows in self.time_granularities.items():
            all_sequences = []
            lengths = []
            
            for idx in range(len(df)):
                sequence, valid_count = self._extract_window_features(
                    df.iloc[idx], granularity, windows
                )
                all_sequences.append(sequence)
                lengths.append(valid_count)
            
            # Scale features
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
        global_features = df[self.global_features].values
        if fit:
            processed_data['global_features'] = self.scalers['global'].fit_transform(global_features)
        else:
            processed_data['global_features'] = self.scalers['global'].transform(global_features)
        
        # Process target
        target_data = df[['time_to_peak']].values
        if fit:
            processed_data['targets'] = self.scalers['target'].fit_transform(target_data)
        else:
            processed_data['targets'] = self.scalers['target'].transform(target_data)
        
        return processed_data

    def __len__(self):
        return len(self.scaled_data['targets'])

    def __getitem__(self, idx):
        sample = {
            'global_features': torch.FloatTensor(self.scaled_data['global_features'][idx]),
            'targets': torch.FloatTensor(self.scaled_data['targets'][idx])
        }
        
        # Add features for each granularity
        for granularity in self.time_granularities.keys():
            sample[f'features_{granularity}'] = torch.FloatTensor(
                self.scaled_data['granular_features'][granularity][idx]
            )
            sample[f'length_{granularity}'] = torch.LongTensor(
                [self.scaled_data['sequence_lengths'][granularity][idx]]
            )
        
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