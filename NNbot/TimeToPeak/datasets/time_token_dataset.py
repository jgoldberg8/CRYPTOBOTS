import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset

class TokenPeakDataset(Dataset):
    def __init__(self, df, scaler=None, train=True, initial_window=30):
        """
        Dataset for peak prediction using historical token data.
        Each token has complete data from 0-1020s, but predictions simulate
        real-time conditions by only using data available up to each timestamp.
        
        Args:
            df: DataFrame with complete token history (0-1020s)
            scaler: Optional pre-fit scaler for validation data
            train: Whether this is training data
            initial_window: Initial data collection period (no predictions)
        """
        self.df = df.copy()
        self.initial_window = initial_window
        
        # Basic features that get updated each timestamp
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
        
        # All available time windows for features
        self.time_windows = [5, 10, 20, 30]  # Short time windows for quick reactions
        
        # Initialize or load scalers
        if train:
            if scaler is None:
                self.scalers = self._init_scalers()
                self.data = self._preprocess_data(fit=True)
            else:
                self.scalers = scaler
                self.data = self._preprocess_data(fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test data")
            self.scalers = scaler
            self.data = self._preprocess_data(fit=False)
    
    def _init_scalers(self):
        """Initialize scalers for each feature type"""
        return {
            'features': RobustScaler(quantile_range=(5, 95)),
            'global': RobustScaler(quantile_range=(5, 95))
        }
    
    def _preprocess_data(self, fit=False):
        """
        Preprocess each token's complete data into a format suitable for training.
        Creates samples that simulate real-time prediction scenarios.
        """
        processed_data = []
        
        # Process each token with progress bar
        from tqdm import tqdm
        for _, token_data in tqdm(self.df.iterrows(), desc="Processing tokens", total=len(self.df)):
            time_to_peak = token_data['time_to_peak']
            
            # Generate prediction timestamps with adaptive sampling
            timestamps = []
            
            # Early phase: Sample every 5s (critical early peaks)
            timestamps.extend(range(self.initial_window, min(200, int(time_to_peak) + 40), 5))
            
            # Medium phase: Sample every 10s
            timestamps.extend(range(200, min(500, int(time_to_peak) + 40), 10))
            
            # Late phase: Sample every 20s
            timestamps.extend(range(500, min(1020, int(time_to_peak) + 40), 20))
            
            # Always include points around the actual peak
            peak_vicinity = [
                max(self.initial_window, time_to_peak - 15),
                max(self.initial_window, time_to_peak - 10),
                max(self.initial_window, time_to_peak - 5),
                time_to_peak,
                min(1020, time_to_peak + 5),
                min(1020, time_to_peak + 10),
                min(1020, time_to_peak + 15)
            ]
            
            timestamps.extend(peak_vicinity)
            timestamps = sorted(list(set(timestamps)))  # Remove duplicates and sort
            
            for t in timestamps:
                # Get features available up to this timestamp
                features = self._extract_features_until_time(token_data, t)
                
                if fit:
                    features = self.scalers['features'].fit_transform(features)
                else:
                    features = self.scalers['features'].transform(features)
                
                # Global token features (available from start)
                global_features = np.array([[
                    token_data['initial_investment_ratio'],
                    token_data['initial_market_cap']
                ]])
                
                if fit:
                    global_features = self.scalers['global'].fit_transform(global_features)
                else:
                    global_features = self.scalers['global'].transform(global_features)
                
                # Label: 1 if this is the peak time (within a small window), 0 otherwise
                # Add some tolerance around the peak time
                is_peak = abs(t - time_to_peak) <= 5  # 5 second tolerance
                
                processed_data.append({
                    'features': features,
                    'global_features': global_features,
                    'is_peak': is_peak,
                    'timestamp': t,
                    'time_to_peak': time_to_peak,  # Store for analysis
                    'mask': True  # All points after initial_window are valid
                })
        
        return processed_data
    
    def _extract_features_until_time(self, token_data, current_time):
        """
        Extract all features that would be available at the current timestamp.
        Only uses data from time windows that have completed by current_time.
        """
        features = []
        
        # Extract features for each time window
        for window in self.time_windows:
            # Only include windows that have completed
            if current_time >= window:
                for feature in self.base_features:
                    col_name = f"{feature}_0to{window}s"
                    if col_name in token_data:
                        features.append(token_data[col_name])
                    else:
                        features.append(0)
                        
                # Add calculated features using only past data
                try:
                    window_features = self._calculate_window_features(token_data, window, current_time)
                    features.extend(window_features)
                except Exception as e:
                    print(f"Error calculating window features: {e}")
                    features.extend([0] * 4)  # Placeholder for missing features
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_window_features(self, token_data, window, current_time):
        """Calculate additional features for a time window using only past data"""
        features = []
        
        # Price acceleration (rate of change of price changes)
        price_cols = [f"price_volatility_0to{w}s" for w in range(window, current_time, window)]
        if len(price_cols) >= 2:
            price_changes = [token_data[col] for col in price_cols if col in token_data]
            price_acceleration = [price_changes[i] - price_changes[i-1] for i in range(1, len(price_changes))]
            avg_acceleration = sum(price_acceleration) / len(price_acceleration) if price_acceleration else 0
        else:
            avg_acceleration = 0
        features.append(avg_acceleration)
        
        # Volume concentration (how much volume in recent windows vs overall)
        volume_cols = [f"volume_0to{w}s" for w in range(window, current_time, window)]
        if volume_cols:
            recent_volume = sum([token_data[col] for col in volume_cols[-3:] if col in token_data])  # Last 3 windows
            total_volume = sum([token_data[col] for col in volume_cols if col in token_data])
            volume_concentration = recent_volume / (total_volume + 1e-8)
        else:
            volume_concentration = 0
        features.append(volume_concentration)
        
        # Buy pressure trend and acceleration
        pressure_cols = [f"buy_pressure_0to{w}s" for w in range(window, current_time, window)]
        if len(pressure_cols) >= 2:
            pressures = [token_data[col] for col in pressure_cols if col in token_data]
            pressure_changes = [pressures[i] - pressures[i-1] for i in range(1, len(pressures))]
            pressure_acceleration = sum(pressure_changes) / len(pressure_changes)
        else:
            pressure_acceleration = 0
        features.append(pressure_acceleration)
        
        # Wallet concentration (unique wallets vs transaction count)
        tx_cols = [f"transaction_count_0to{w}s" for w in range(window, current_time, window)]
        wallet_cols = [f"unique_wallets_0to{w}s" for w in range(window, current_time, window)]
        if tx_cols and wallet_cols:
            recent_tx = sum([token_data[col] for col in tx_cols[-3:] if col in token_data])
            recent_wallets = sum([token_data[col] for col in wallet_cols[-3:] if col in token_data]) + 1
            wallet_concentration = recent_tx / recent_wallets
        else:
            wallet_concentration = 0
        features.append(wallet_concentration)
        
        # Momentum indicators
        momentum_cols = [f"momentum_0to{w}s" for w in range(window, current_time, window)]
        if momentum_cols:
            recent_momentum = sum([token_data[col] for col in momentum_cols[-3:] if col in token_data]) / 3
        else:
            recent_momentum = 0
        features.append(recent_momentum)
        
        return features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single training sample"""
        sample = self.data[idx]
        
        return {
            'features': torch.FloatTensor(sample['features']),
            'global_features': torch.FloatTensor(sample['global_features']),
            'is_peak': torch.FloatTensor([sample['is_peak']]),
            'timestamp': torch.FloatTensor([sample['timestamp']]),
            'time_to_peak': torch.FloatTensor([sample['time_to_peak']]),
            'mask': torch.FloatTensor([sample['mask']])
        }