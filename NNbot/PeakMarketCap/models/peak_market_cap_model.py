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
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'tree_method': 'hist',  # Faster training
            'gamma': 1  # Minimum loss reduction for split
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
            'volume_0to30s', 'rsi_0to30s', 'buy_pressure_0to30s',
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
            values = df[[f'{metric}_{window}' for window in windows]].values
            slope = np.polyfit(range(len(windows)), values, 1)[0]
            features[f'{metric}_trend'] = slope
            
            # Add acceleration (change in slope)
            diffs = np.diff(values, axis=1)
            features[f'{metric}_acceleration'] = np.diff(diffs, axis=1).squeeze()
        
        # Volume concentration features
        total_volume = df['volume_0to30s'].clip(lower=1e-8)  # Avoid division by zero
        for window in ['0to10s', '10to20s', '20to30s']:
            features[f'volume_concentration_{window}'] = df[f'volume_{window}'] / total_volume
        
        # Market cap features
        initial_mcap = df['initial_market_cap'].clip(lower=1e-8)  # Avoid division by zero
        features['market_cap_change'] = (df['market_cap_at_30s'] - initial_mcap) / initial_mcap
        features['log_initial_mcap'] = np.log1p(initial_mcap)
        
        # Transaction features
        features['avg_trade_size'] = df['volume_0to30s'] / df['transaction_count_0to30s'].clip(lower=1)
        features['wallet_to_transaction_ratio'] = df['unique_wallets_0to30s'] / df['transaction_count_0to30s'].clip(lower=1)
        
        return features.fillna(0)  # Fill any NaN values that might have been created
    
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
        
        # Convert to numpy arrays for XGBoost
        X = self.scaler.transform(features)
        y = df['percent_increase'].values if 'percent_increase' in df.columns else None
        
        return X, y
            
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