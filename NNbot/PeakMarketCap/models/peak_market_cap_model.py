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
            'n_estimators': 1000,
            'max_depth': 5,
            'learning_rate': 0.005,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'objective': 'reg:squarederror',
            'gamma': 2,  # Increased to reduce overfitting
            'random_state': 42
        }
        self.xgb_params = xgb_params if xgb_params else self.default_xgb_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def _engineer_features(self, df):
        """Create engineered features from the raw data"""
        feature_dict = {}
        
        # Base features (using 0-30s data as it's the final window)
        base_metrics = [
            'volume_0to30s', 'rsi_0to30s', 'buy_pressure_0to30s',
            'price_volatility_0to30s', 'volume_volatility_0to30s', 'momentum_0to30s',
            'trade_amount_variance_0to30s', 'transaction_rate_0to30s',
            'trade_concentration_0to30s', 'unique_wallets_0to30s',
            'initial_investment_ratio', 'initial_market_cap'
        ]
        for metric in base_metrics:
            feature_dict[metric] = df[metric]
        
        # 5-second window features
        five_sec_windows = ['0to5s', '5to10s', '10to15s', '15to20s', '20to25s', '25to30s']
        metrics = ['volume', 'rsi', 'momentum', 'buy_pressure', 'price_volatility', 
                'volume_volatility', 'trade_amount_variance', 'transaction_rate']
        
        # Compute all window-based features at once
        for metric in metrics:
            # Get all values for this metric across windows
            window_data = np.column_stack([df[f'{metric}_{window}'].values for window in five_sec_windows])
            
            # Compute statistics
            feature_dict[f'{metric}_5s_max'] = np.max(window_data, axis=1)
            feature_dict[f'{metric}_5s_min'] = np.min(window_data, axis=1)
            feature_dict[f'{metric}_5s_std'] = np.std(window_data, axis=1)
            feature_dict[f'{metric}_5s_mean'] = np.mean(window_data, axis=1)
            
            # Compute trends (differences between consecutive windows)
            diffs = np.diff(window_data, axis=1)
            feature_dict[f'{metric}_5s_trend_mean'] = np.mean(diffs, axis=1)
            feature_dict[f'{metric}_5s_trend_std'] = np.std(diffs, axis=1)
            feature_dict[f'{metric}_5s_acceleration'] = np.diff(diffs, axis=1).mean(axis=1)
            
            # Early vs Late period comparison
            early_period = window_data[:, :2].mean(axis=1)
            late_period = window_data[:, -2:].mean(axis=1)
            feature_dict[f'{metric}_early_vs_late'] = (late_period - early_period) / np.clip(early_period, 1e-8, None)
        
        # Volume concentration features
        total_volume = np.clip(df['volume_0to30s'].values, 1e-8, None)
        for window in five_sec_windows:
            feature_dict[f'volume_concentration_{window}'] = df[f'volume_{window}'].values / total_volume
        
        # Market cap features
        initial_mcap = np.clip(df['initial_market_cap'].values, 1e-8, None)
        feature_dict['market_cap_change'] = (df['market_cap_at_30s'].values - initial_mcap) / initial_mcap
        feature_dict['log_initial_mcap'] = np.log1p(initial_mcap)
        
        # Transaction features
        tx_count = np.clip(df['transaction_count_0to30s'].values, 1, None)
        feature_dict['avg_trade_size'] = df['volume_0to30s'].values / tx_count
        feature_dict['wallet_to_transaction_ratio'] = df['unique_wallets_0to30s'].values / tx_count
        
        # Create final DataFrame all at once
        features = pd.DataFrame(feature_dict)
        
        # Fill any NaN values
        features = features.fillna(0)
        
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
        
        # Create a copy of the dataframe
        df_with_predictions = df.copy()
        
        # Initialize predictions column with zeros
        df_with_predictions['predicted_percent_increase'] = 0.0
        
        # Get mask for valid prediction rows (same as training filter)
        prediction_mask = (
            df_with_predictions['hit_peak_before_30'].astype(str).str.lower() == "false"
        )
        
        if prediction_mask.any():
            # Prepare data only for rows meeting the criteria
            X, _ = self.prepare_data(df_with_predictions[prediction_mask])
            
            # Make predictions for filtered rows
            predictions = self.model.predict(X)
            
            # Assign predictions only to the relevant rows
            df_with_predictions.loc[prediction_mask, 'predicted_percent_increase'] = predictions
        
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