import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

class HitPeakBefore30Dataset(Dataset):
    """Dataset class for peak prediction within 30 seconds.
    
    This class handles both training and live prediction scenarios, with robust
    feature processing and error handling for missing data.
    """
    
    def __init__(self, df, scaler=None, train=True):
        """Initialize the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe containing token data
            scaler (dict, optional): Dictionary containing pre-fit scalers. Defaults to None.
            train (bool, optional): Whether this is for training. Defaults to True.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        try:
            # Convert time features with error handling
            try:
                df['creation_time_numeric'] = pd.to_datetime(df['creation_time']).dt.hour + \
                                            pd.to_datetime(df['creation_time']).dt.minute / 60
            except Exception as e:
                self.logger.warning(f"Error converting creation_time: {e}")
                df['creation_time_numeric'] = 0
                
            # Create cyclic time features
            df['creation_time_sin'] = np.sin(2 * np.pi * df['creation_time_numeric'] / 24.0)
            df['creation_time_cos'] = np.cos(2 * np.pi * df['creation_time_numeric'] / 24.0)
            
            # Add volume pressure features
            for window in ['5s', '10s', '20s', '30s']:
                vol_col = f'volume_0to{window}'
                if vol_col in df.columns:
                    df[f'volume_pressure_{window}'] = df[vol_col] / (df['initial_market_cap'] + 1e-8)
                else:
                    df[f'volume_pressure_{window}'] = 0
                    self.logger.debug(f"Missing volume column: {vol_col}")
            
            # Convert numeric columns to float32
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            # Define base features
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
            
            # Define time windows
            self.time_windows = {
                '5s': ['0to5', '5to10', '10to15', '15to20', '20to25', '25to30'],
                '10s': ['0to10', '10to20', '20to30'],
                '20s': ['0to20'],
                '30s': ['0to30']
            }
            
            # Define global features
            self.global_features = [
                'initial_investment_ratio', 
                'initial_market_cap',
                'volume_pressure_5s',
                'volume_pressure_10s',
                'creation_time_sin',
                'creation_time_cos',
                'creation_time_numeric'
            ]
            
            # Handle targets differently for training vs. prediction
            if train and 'hit_peak_before_30' in df.columns:
                self.targets = ['hit_peak_before_30']
            else:
                # For prediction, create a dummy target
                df['hit_peak_before_30'] = 0
                self.targets = ['hit_peak_before_30']
                if train:
                    self.logger.warning("Training mode but 'hit_peak_before_30' not found. Using dummy targets.")
            
            # Initialize scaling
            if train:
                if scaler is None:
                    self.global_scaler = StandardScaler()
                    self.scaled_data = self._preprocess_data(df, fit=True)
                else:
                    self.global_scaler = scaler['global']
                    self.scaled_data = self._preprocess_data(df, fit=False)
            else:
                if scaler is None:
                    raise ValueError("Scaler must be provided for test/prediction data")
                self.global_scaler = scaler['global']
                self.scaled_data = self._preprocess_data(df, fit=False)
            
            # Calculate quality features
            self.quality_features = self._calculate_quality_features(df)
            
        except Exception as e:
            self.logger.error(f"Error initializing dataset: {e}")
            raise

    def _preprocess_data(self, df, fit=False):
        """Preprocess the data including scaling and feature engineering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool, optional): Whether to fit the scaler. Defaults to False.
            
        Returns:
            dict: Processed data including temporal, global features and targets
        """
        try:
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
                            self.logger.debug(f"Missing feature column: {col_name}")
                            features.append(np.zeros(len(df)))
                    window_data.append(np.stack(features, axis=1))
                processed_data[window_type] = np.stack(window_data, axis=1)
            
            # Process global features
            global_features = []
            for feature in self.global_features:
                if feature in df.columns:
                    global_features.append(df[feature].values)
                else:
                    self.logger.debug(f"Missing global feature: {feature}")
                    global_features.append(np.zeros(len(df)))
            global_data = np.stack(global_features, axis=1)
            
            # Scale global features
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
            
        except Exception as e:
            self.logger.error(f"Error in _preprocess_data: {e}")
            raise

    def _calculate_quality_features(self, df):
        """Calculate data quality features for the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            np.ndarray: Array of quality features
        """
        try:
            # Calculate basic completeness
            completeness = df.notna().mean(axis=1).values
            
            # Calculate weighted active intervals
            active_intervals = 0
            total_weight = 0
            
            for window_type, windows in self.time_windows.items():
                # Weight by inverse window size (e.g., 5s windows weighted more than 30s)
                weight = 1.0 / float(window_type[:-1])
                for window in windows:
                    col_name = f"transaction_count_{window}s"
                    if col_name in df.columns:
                        active_intervals += (df[col_name] > 0).astype(float) * weight
                        total_weight += weight
            
            # Normalize active intervals
            active_intervals = active_intervals / (total_weight + 1e-8)
            
            return np.stack([completeness, active_intervals], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality features: {e}")
            # Return default quality features in case of error
            return np.zeros((len(df), 2))

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.scaled_data['targets'])

    def __getitem__(self, idx):
        """Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary containing all features for the specified index
        """
        try:
            return {
                'x_5s': torch.FloatTensor(self.scaled_data['data']['5s'][idx]),
                'x_10s': torch.FloatTensor(self.scaled_data['data']['10s'][idx]),
                'x_20s': torch.FloatTensor(self.scaled_data['data']['20s'][idx]),
                'x_30s': torch.FloatTensor(self.scaled_data['data']['30s'][idx]),
                'global_features': torch.FloatTensor(self.scaled_data['global'][idx]),
                'quality_features': torch.FloatTensor(self.quality_features[idx]),
                'targets': torch.FloatTensor(self.scaled_data['targets'][idx])
            }
        except Exception as e:
            self.logger.error(f"Error in __getitem__ at index {idx}: {e}")
            raise