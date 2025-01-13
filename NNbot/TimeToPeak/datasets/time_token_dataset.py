import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset

from utils.add_data_quality_features import add_data_quality_features

class MultiGranularTokenDataset(Dataset):
    def __init__(self, df, scaler=None, train=True, initial_window=30):
        """
        Dataset for token peak prediction with initial data collection period.
        """
        df = df.copy()
        self.initial_window = initial_window
        
        # Group the data by mint to process each token's timeline
        self.token_groups = [group for _, group in df.groupby('mint')]
        print(f"Found {len(self.token_groups)} token groups")
        
        # Base features per window
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
        
        # Define time granularities
        self.granularities = ['5s', '10s', '20s', '30s']
        
        # Initialize or load scalers
        if train:
            if scaler is None:
                self.scalers = self._init_scalers()
                self.data = self._preprocess_data(self.token_groups, fit=True)
            else:
                self.scalers = scaler
                self.data = self._preprocess_data(self.token_groups, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test data")
            self.scalers = scaler
            self.data = self._preprocess_data(self.token_groups, fit=False)
    
    def _add_momentum_features(self, df):
        """Add momentum and market dynamics features"""
        # Calculate for each granularity
        for gran in self.granularities:
            window = int(gran.replace('s', ''))
            
            # Prevent NaN calculations
            for feature in ['price_volatility', 'volume', 'transaction_count', 'buy_pressure', 'unique_wallets']:
                col_name = f'{feature}_0to{window}s'
                
                # Replace NaNs with 0 or forward/backward fill
                if col_name in df.columns:
                    df[col_name] = df[col_name].fillna(0)  # or .fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            try:
                # Price dynamics with NaN handling
                df[f'price_delta_{gran}'] = df.groupby('mint')[f'price_volatility_0to{window}s'].diff().fillna(0)
                df[f'price_acceleration_{gran}'] = df[f'price_delta_{gran}'].diff().fillna(0)
                
                # Volume dynamics with NaN handling
                df[f'volume_delta_{gran}'] = df.groupby('mint')[f'volume_0to{window}s'].diff().fillna(0)
                df[f'volume_acceleration_{gran}'] = df[f'volume_delta_{gran}'].diff().fillna(0)
                
                # Trading intensity with NaN and zero handling
                transaction_col = f'transaction_count_0to{window}s'
                wallets_col = f'unique_wallets_0to{window}s'
                
                # Prevent division by zero or NaN
                df[f'trade_intensity_{gran}'] = (
                    df[transaction_col].fillna(0) / 
                    (df[wallets_col].fillna(1) + 1)
                ).fillna(0).rolling(window=min(5, len(df))).mean().fillna(0)
                
                # Buy pressure momentum
                pressure_col = f'buy_pressure_0to{window}s'
                df[f'buy_pressure_momentum_{gran}'] = df.groupby('mint')[pressure_col].diff().fillna(0)
            
            except KeyError as e:
                print(f"Warning: Could not create features for {gran} due to missing columns: {e}")
                # Create placeholder columns with zeros
                df[f'price_delta_{gran}'] = 0
                df[f'price_acceleration_{gran}'] = 0
                df[f'volume_delta_{gran}'] = 0
                df[f'volume_acceleration_{gran}'] = 0
                df[f'trade_intensity_{gran}'] = 0
                df[f'buy_pressure_momentum_{gran}'] = 0

        # Add data quality features
        # try:
        #     df = add_data_quality_features(df)
        # except Exception as e:
        #     print(f"Warning: Could not add data quality features: {e}")
        
        return df

    
    def _init_scalers(self):
        """Initialize scalers for all feature types"""
        return {
            'granular': {gran: RobustScaler(quantile_range=(5, 95)) 
                        for gran in self.granularities},
            'global': RobustScaler(quantile_range=(5, 95)),
            'target': RobustScaler(quantile_range=(5, 95))
        }
    
    def _preprocess_data(self, token_groups, fit=False):
            """Preprocess data and create sequences"""
            processed_data = []
            
            for token_data in token_groups:
                # Sort by time sequence (time_to_peak descending means forward in time)
                token_data = token_data.sort_values('time_to_peak', ascending=False)
                
                # Process features for each granularity
                gran_features = {}
                for gran in self.granularities:
                    features = self._extract_granularity_features(token_data, gran)
                    
                    if fit:
                        gran_features[gran] = self.scalers['granular'][gran].fit_transform(features)
                    else:
                        gran_features[gran] = self.scalers['granular'][gran].transform(features)
                
                # Process global features
                global_features = self._extract_global_features(token_data)
                if fit:
                    global_features = self.scalers['global'].fit_transform(global_features)
                else:
                    global_features = self.scalers['global'].transform(global_features)
                
                # Get target information
                target_info = self._process_target_info(token_data)
                
                # Store all timepoints for this token
                for t in range(len(token_data)):
                    processed_data.append({
                        'features': {gran: features[t:t+1] for gran, features in gran_features.items()},
                        'global_features': global_features[t:t+1],
                        'target_info': {
                            k: v[t:t+1] for k, v in target_info.items()
                        },
                        'token': token_data['mint'].iloc[0]
                    })
            
            return processed_data
    def _extract_granularity_features(self, df, granularity):
        """Extract features for specific time granularity"""
        window = int(granularity.replace('s', ''))
        features = []
        
        # Base trading features with NaN handling
        for feature in self.base_features:
            col_name = f"{feature}_0to{window}s"
            if col_name in df.columns:
                features.append(df[col_name].fillna(0).values)
            else:
                features.append(np.zeros(len(df)))
        
        # Momentum features with NaN handling
        momentum_features = [
            f'price_delta_{granularity}',
            f'price_acceleration_{granularity}',
            f'volume_delta_{granularity}',
            f'volume_acceleration_{granularity}',
            f'trade_intensity_{granularity}',
            f'buy_pressure_momentum_{granularity}'
        ]
        
        for feature in momentum_features:
            if feature in df.columns:
                features.append(df[feature].fillna(0).values)
            else:
                features.append(np.zeros(len(df)))
        
        return np.column_stack(features)
        
    def _extract_global_features(self, df):
        """Extract global token features"""
        return np.column_stack([
            df['initial_investment_ratio'].values,
            df['initial_market_cap'].values,
            df['peak_market_cap'].values,
            df['time_to_peak'].values,
        ])
    
    def _process_target_info(self, df):
        """Process target variables and create prediction masks"""
        time_to_peak = df['time_to_peak'].values
        
        # Create prediction mask (True after initial window)
        mask = (time_to_peak >= self.initial_window)
        
        # Calculate peak proximity using exponential decay
        # Shorter decay for earlier peaks to increase sensitivity
        peak_proximity = np.exp(-time_to_peak / (max(20, time_to_peak.min() / 4)))
        
        # Calculate sample weights based on time distribution
        time_buckets = pd.cut(time_to_peak, 
                            bins=[30, 100, 200, 400, 600, 800, 1020],
                            labels=['30-100', '100-200', '200-400', 
                                '400-600', '600-800', '800-1020'])
        counts = time_buckets.value_counts()
        weights = 1 / (counts[time_buckets] + 1)  # Add 1 to avoid division by zero
        sample_weights = weights / weights.sum()
        
        # Add extra weight to early peaks (30-200s)
        early_peak_mask = time_to_peak <= 200
        sample_weights[early_peak_mask] *= 1.5
        sample_weights = sample_weights / sample_weights.sum()
        
        # Print debug info
        print(f"Valid predictions in batch: {mask.sum()}/{len(mask)}")
        print(f"Time to peak range: [{time_to_peak.min()}, {time_to_peak.max()}]")
        print(f"Mask shape: {mask.shape}")
        
        return {
            'time_to_peak': torch.tensor(time_to_peak, dtype=torch.float32).view(-1, 1),
            'peak_proximity': torch.tensor(peak_proximity, dtype=torch.float32).view(-1, 1),
            'mask': torch.tensor(mask.astype(float), dtype=torch.float32).view(-1, 1),
            'sample_weights': torch.tensor(sample_weights, dtype=torch.float32).view(-1, 1)
        }
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single token's data"""
        token_data = self.data[idx]
        
        sample = {
            'global_features': torch.FloatTensor(token_data['global_features']),
            'time_to_peak': torch.FloatTensor(token_data['target_info']['time_to_peak']),
            'peak_proximity': torch.FloatTensor(token_data['target_info']['peak_proximity']),
            'mask': torch.FloatTensor(token_data['target_info']['mask']),
            'sample_weights': torch.FloatTensor(token_data['target_info']['sample_weights'])
        }
        
        # Add features for each granularity
        for i, gran in enumerate(self.granularities):
            sample[f'features_{i}'] = torch.FloatTensor(token_data['features'][gran])
        
        return sample