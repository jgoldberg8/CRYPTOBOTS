import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TokenPricePredictor:
    def __init__(self, xgb_params=None):
        self.default_xgb_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'objective': 'multi:softprob',  # Changed for multiclass
            'num_class': 4,  # Number of classes
            'tree_method': 'hist',
            'random_state': 42
        }
        self.xgb_params = xgb_params if xgb_params else self.default_xgb_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Define increase ranges
        self.ranges = [
            (0, 50),      # Low increase
            (50, 150),    # Medium increase
            (150, 300),   # High increase
            (300, float('inf'))  # Very high increase
        ]
    
    def _get_increase_category(self, percent_increase):
        """Convert percent increase to category"""
        for i, (low, high) in enumerate(self.ranges):
            if low <= percent_increase < high:
                return i
        return len(self.ranges) - 1  # Return last category if above all ranges
    
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
        
        # Compute all window-based features
        for metric in metrics:
            # Get all values for this metric across windows
            window_data = np.column_stack([df[f'{metric}_{window}'].values for window in five_sec_windows])
            
            # Compute statistics
            feature_dict[f'{metric}_5s_max'] = np.max(window_data, axis=1)
            feature_dict[f'{metric}_5s_min'] = np.min(window_data, axis=1)
            feature_dict[f'{metric}_5s_std'] = np.std(window_data, axis=1)
            feature_dict[f'{metric}_5s_mean'] = np.mean(window_data, axis=1)
            
            # Early vs Late period comparison
            early_period = window_data[:, :2].mean(axis=1)
            late_period = window_data[:, -2:].mean(axis=1)
            feature_dict[f'{metric}_early_vs_late'] = (late_period - early_period) / np.clip(early_period, 1e-8, None)
        
        # Create final DataFrame all at once
        features = pd.DataFrame(feature_dict)
        return features.fillna(0)
    
    def prepare_data(self, df):
        """Prepare data for training or prediction"""
        # Convert hit_peak_before_30 to string type to ensure consistent comparison
        df = df.copy()
        df['hit_peak_before_30'] = df['hit_peak_before_30'].astype(str)
        
        # Filter for valid training samples
        if 'percent_increase' in df.columns:  # Training mode
            df = df[
                (df['hit_peak_before_30'].str.lower() == "false") & 
                (df['percent_increase'] > 0)
            ].copy()
            
            if len(df) == 0:
                raise ValueError("No samples left after filtering. Check data types and values.")
            
            # Convert percent_increase to categories
            y = np.array([self._get_increase_category(val) for val in df['percent_increase']])
        else:
            y = None
        
        # Engineer features
        features = self._engineer_features(df)
        
        # Store feature columns for prediction
        if self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        
        # Scale features
        if 'percent_increase' in df.columns:  # Training mode
            self.scaler.fit(features)
        
        X = self.scaler.transform(features)
        return X, y
    
    def train(self, df):
        """Train the model"""
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Initialize and train model
        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X, y)
        
        # Make predictions on training data
        train_pred = self.model.predict(X)
        train_pred_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, train_pred),
            'classification_report': classification_report(y, train_pred, output_dict=True)
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
        
        # Initialize predictions columns
        df_with_predictions['predicted_category'] = -1
        for i in range(len(self.ranges)):
            df_with_predictions[f'probability_range_{i}'] = 0.0
        
        # Only predict for tokens that haven't peaked before 30s
        pred_mask = (df_with_predictions['hit_peak_before_30'].astype(str).str.lower() == "false")
        
        if pred_mask.any():
            # Get indices where pred_mask is True
            pred_indices = df_with_predictions.index[pred_mask]
            
            # Prepare data only for valid prediction rows
            pred_df = df_with_predictions[pred_mask].copy()
            X, _ = self.prepare_data(pred_df)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Update predictions
            df_with_predictions.loc[pred_mask, 'predicted_category'] = predictions
            for i in range(len(self.ranges)):
                df_with_predictions.loc[pred_mask, f'probability_range_{i}'] = probabilities[:, i]
        
        return df_with_predictions