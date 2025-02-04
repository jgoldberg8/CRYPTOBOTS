import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

class TokenPricePredictor:
    def __init__(self, xgb_params=None):
        self.default_xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.xgb_params = xgb_params if xgb_params else self.default_xgb_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def _engineer_features(self, df):
        """Create engineered features from the raw data"""
        features = pd.DataFrame()
        
        # Base features (using 0-30s data as it's the final window)
        base_metrics = [
            'volume_0to30s', 'rsi_0to30s', 'momentum_0to30s', 'buy_pressure_0to30s',
            'price_volatility_0to30s', 'volume_volatility_0to30s', 'momentum_0to30s',
            'trade_amount_variance_0to30s', 'transaction_rate_0to30s',
            'trade_concentration_0to30s', 'unique_wallets_0to30s',
            'initial_investment_ratio', 'initial_market_cap'
        ]
        features = pd.concat([features, df[base_metrics]], axis=1)
        
        # Trend features (comparing different time windows)
        windows = ['0to10s', '10to20s', '20to30s']
        metrics = ['volume', 'rsi', 'momentum', 'buy_pressure', 'price_volatility']
        
        for metric in metrics:
            # Calculate trend (slope) across windows
            for i in range(len(windows)-1):
                curr_window = windows[i]
                next_window = windows[i+1]
                features[f'{metric}_trend_{i}'] = (
                    df[f'{metric}_{next_window}'] - df[f'{metric}_{curr_window}']
                )
        
        # Volume concentration features
        total_volume = df['volume_0to30s']
        for window in ['0to10s', '10to20s', '20to30s']:
            features[f'volume_concentration_{window}'] = df[f'volume_{window}'] / total_volume
        
        # Wallet activity features
        features['wallet_activity_trend'] = (
            df['unique_wallets_20to30s'] - df['unique_wallets_0to10s']
        )
        
        # Market cap related features
        features['market_cap_change'] = (
            df['market_cap_at_30s'] - df['initial_market_cap']
        ) / df['initial_market_cap']
        
        # Volatility features
        for metric in ['price', 'volume']:
            vol_cols = [f'{metric}_volatility_{window}' for window in windows]
            features[f'{metric}_volatility_trend'] = np.polyfit(
                range(len(vol_cols)),
                df[vol_cols].values.T,
                1
            )[0]
        
        return features
    
    def prepare_data(self, df):
        """Prepare data for training or prediction"""
        # Convert hit_peak_before_30 to string type to ensure consistent comparison
        df = df.copy()
        df['hit_peak_before_30'] = df['hit_peak_before_30'].astype(str)
        
        # Filter for tokens that haven't peaked before 30s and have positive increase
        if 'percent_increase' in df.columns:  # Training mode
            df_filtered = df[
                (df['hit_peak_before_30'].str.lower() == "false") & 
                (df['percent_increase'] > 0)
            ].copy()
            
            if len(df_filtered) == 0:
                raise ValueError("No samples left after filtering. Check data types and values.")
                
            df = df_filtered
        
        # Engineer features
        features = self._engineer_features(df)
        
        # Store feature columns for prediction
        if self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        
        # Scale features
        if 'percent_increase' in df.columns:  # Training mode
            self.scaler.fit(features)
        
        scaled_features = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns
        )
        
        return scaled_features, df['percent_increase'] if 'percent_increase' in df.columns else None
            
    def train(self, df):
        """Train the model"""
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        val_predictions = self.model.predict(X_val)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
            'mae': mean_absolute_error(y_val, val_predictions),
            'r2': r2_score(y_val, val_predictions)
        }
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics, importance
    
    def predict(self, df):
        """Make predictions for new data"""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        # Prepare data
        X, _ = self.prepare_data(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        df_with_predictions = df.copy()
        df_with_predictions['predicted_percent_increase'] = predictions
        
        return df_with_predictions

# Usage example:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/new-token-data.csv')
    
    # Initialize and train model
    predictor = TokenPricePredictor()
    metrics, importance = predictor.train(df)
    
    # Print results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))