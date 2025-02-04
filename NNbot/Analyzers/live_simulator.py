import joblib
import websocket
import json
import os
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import torch
import pandas as pd
from Before30.dataset.hit_peak_30_dataset import HitPeakBefore30Dataset
from Before30.models.peak_before_30_model import HitPeakBefore30Predictor
from PeakMarketCap.models.token_dataset import TokenDataset

class ScalerManager:
    """Manages loading and saving of scalers for the trading simulator"""
    def __init__(self, scaler_dir='scalers'):
        self.scaler_dir = scaler_dir
        os.makedirs(scaler_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_scalers(self, model_type, global_scaler=None):
        """Save scalers based on model type"""
        try:
            if model_type == 'before30' and global_scaler:
                global_path = os.path.join(self.scaler_dir, 'hit_peak_global_scaler.joblib')
                joblib.dump(global_scaler, global_path)
                self.logger.info(f"Before30 global scaler saved to {global_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving scalers: {e}")
            raise
            
    def load_scalers(self, model_type):
        """Load scalers based on model type"""
        global_scaler = None
        
        try:
            if model_type == 'before30':
                global_path = os.path.join(self.scaler_dir, 'hit_peak_global_scaler.joblib')
                if os.path.exists(global_path):
                    global_scaler = joblib.load(global_path)
                    self.logger.info("Before30 global scaler loaded successfully")
                else:
                    self.logger.warning(f"Before30 global scaler not found at {global_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
            
        return global_scaler

class TradingSimulator:
    def __init__(self, config, peak_before_30_model_path, price_classifier_model_path):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_simulator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler management
        self.scaler_manager = ScalerManager()
        
        # Load ML models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Load before30 model and scaler
        try:
            model_data = self._load_peak_before_30_model(peak_before_30_model_path)
            self.peak_before_30_model = model_data['model']
            self.before30_global_scaler = self.scaler_manager.load_scalers('before30')
            if self.before30_global_scaler is None:
                self.before30_global_scaler = model_data['scaler']
                if self.before30_global_scaler is not None:
                    self.scaler_manager.save_scalers('before30', self.before30_global_scaler)
        except Exception as e:
            self.logger.error(f"Error loading peak_before_30 model: {e}")
            raise
          
        # Load price classification model
        self.price_classifier = joblib.load(price_classifier_model_path)
        self.class_ranges = [
            (0, 50),    # Range 0
            (50, 150),  # Range 1
            (150, 300), # Range 2
            (300, float('inf'))  # Range 3
        ]

        # Initialize WebSocket configuration
        self.config = config
        self.ws = None
        self.is_connecting = False
        self._subscribed_tokens = set()
        
        # Trading state management
        self.active_tokens = {}
        self.positions = {}
        
        # Time windows for data collection
        self.time_windows = {
            '5s': [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)],
            '10s': [(0, 10), (10, 20), (20, 30)],
            '20s': [(0, 20)],
            '30s': [(0, 30)]
        }
        
        # Trading metrics
        self.trading_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit_loss': 0,
            'positions': []
        }

    def _load_peak_before_30_model(self, model_path):
        """Load the peak before 30 prediction model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            model = HitPeakBefore30Predictor(
                input_size=11,
                hidden_size=256,
                num_layers=3,
                dropout_rate=0.5,
                global_feature_dim=7
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return {
                'model': model,
                'scaler': checkpoint.get('global_scaler')
            }
            
        except Exception as e:
            self.logger.error(f"Error in _load_peak_before_30_model: {e}")
            raise

    def _calculate_features(self, token_data):
        """Calculate features for model prediction"""
        if not token_data['transactions']:
            self.logger.warning("No transactions available for feature calculation")
            return None
            
        try:
            start_time = token_data['first_trade_time']
            transactions = token_data['transactions']
            
            features = {}
            
            # Calculate metrics for all timeframes
            for window_type, intervals in self.time_windows.items():
                for start, end in intervals:
                    window_key = f"{start}to{end}s"
                    metrics = self._calculate_timeframe_metrics(
                        transactions, start_time, start, end
                    )
                    for feature, value in metrics.items():
                        features[f'{feature}_{window_key}'] = value

            # Create base DataFrame for dataset processing
            base_data = {
                'initial_investment_ratio': token_data['initial_investment_ratio'],
                'initial_market_cap': token_data['initial_market_cap'],
                'buy_sell_ratio': token_data['buy_sell_ratio'],
                'volume_pressure': token_data['volume_pressure']
            }
            
            base_data.update(features)
            
            # Global features for Before30 model
            before30_global_features = np.array([[
                token_data['initial_investment_ratio'],
                token_data['initial_market_cap'],
                token_data['volume_pressure'],
                token_data['buy_sell_ratio'],
                features['price_volatility_0to30s'],
                features['volume_volatility_0to30s'],
                features['momentum_0to30s']
            ]])
            features['global'] = self.before30_global_scaler.transform(before30_global_features)

            # Process data for each time window
            processed_data = {'data': {}}
            
            for window_type in ['5s', '10s', '20s', '30s']:
                window_features = []
                intervals = self.time_windows[window_type]
                
                for start, end in intervals:
                    window_key = f"{start}to{end}s"
                    feature_array = []
                    
                    base_features = [
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
                    
                    for feature in base_features:
                        key = f"{feature}_{window_key}"
                        feature_array.append(features.get(key, 0))
                    
                    window_features.append(feature_array)
                
                processed_data['data'][window_type] = np.array([window_features])

            processed_data['global'] = features['global']
            
            return processed_data
                
        except Exception as e:
            self.logger.error(f"Error in feature calculation: {str(e)}")
            return None

    def _calculate_timeframe_metrics(self, transactions, start_time, start, end):
        """Calculate metrics for a specific timeframe (unchanged)"""
        # ... (keep existing implementation) ...

    def _get_target_price(self, predicted_range, current_price):
        """Calculate target price based on predicted range"""
        min_increase = self.class_ranges[predicted_range][0]
        return current_price * (1 + min_increase / 100)

    def _should_enter_trade(self, token_mint):
        """Determine if we should enter a trade based on model predictions"""
        try:
            token_data = self.active_tokens[token_mint]
            
            if len(token_data['transactions']) < 5:
                return False
                
            features = self._calculate_features(token_data)
            if features is None:
                return False

            with torch.no_grad():
                # First check if token has peaked using Before30 model
                before30_global_features = np.array([[
                    features['global'][0, 0],
                    features['global'][0, 1],
                    features['global'][0, 2],
                    features['global'][0, 3],
                    features['data']['30s'][0, -1, 4],
                    features['data']['30s'][0, -1, 5],
                    features['data']['30s'][0, -1, 6]
                ]])
                
                global_features_before30 = torch.FloatTensor(before30_global_features).to(self.device)
                quality_features = torch.FloatTensor(self._calculate_quality_features(features)).to(self.device)
                
                x_5s = torch.FloatTensor(features['data']['5s']).to(self.device)
                x_10s = torch.FloatTensor(features['data']['10s']).to(self.device)
                x_20s = torch.FloatTensor(features['data']['20s']).to(self.device)
                x_30s = torch.FloatTensor(features['data']['30s']).to(self.device)
                
                peak_before_30_pred = self.peak_before_30_model(
                    x_5s, x_10s, x_20s, x_30s,
                    global_features_before30, quality_features
                )
                
                prob_peaked = torch.sigmoid(peak_before_30_pred).item()
                
                if prob_peaked < 0.5:  # Only proceed if hasn't peaked
                    # Prepare features for classification model
                    feature_df = pd.DataFrame([self._prepare_classification_features(features)])
                    X = self.price_classifier.scaler.transform(feature_df)
                    
                    # Get predictions
                    pred_proba = self.price_classifier.model.predict_proba(X)[0]
                    predicted_class = np.argmax(pred_proba)
                    confidence = pred_proba[predicted_class]
                    
                    # Only trade on high confidence Range 2/3 predictions
                    if predicted_class >= 2 and confidence >= 0.85:
                        current_mcap = token_data['current_market_cap']
                        token_data['predicted_class'] = predicted_class
                        token_data['prediction_confidence'] = confidence
                        token_data['target_sell_price'] = self._get_target_price(predicted_class, current_mcap)
                        
                        self.logger.info(f"\n{'='*50}")
                        self.logger.info(f"PREDICTION - Token: {token_mint}")
                        self.logger.info(f"Predicted Range: {predicted_class} (Confidence: {confidence:.2%})")
                        self.logger.info(f"Current Market Cap: {current_mcap:.4f}")
                        self.logger.info(f"Target Sell Price: {token_data['target_sell_price']:.4f}")
                        self.logger.info(f"{'='*50}\n")
                        return True
                        
            return False
                    
        except Exception as e:
            self.logger.error(f"Error in _should_enter_trade for {token_mint}: {str(e)}")
            return False

    def _prepare_classification_features(self, features):
        """Prepare features for classification model"""
        feature_dict = {}
        
        # Base features
        base_metrics = [
            'initial_investment_ratio', 'initial_market_cap',
            'buy_sell_ratio', 'volume_pressure'
        ]
        for metric in base_metrics:
            feature_dict[metric] = features['global'][0][base_metrics.index(metric)]
        
        # Window-based features
        for window_type in ['5s', '10s', '20s', '30s']:
            window_data = features['data'][window_type][0]
            for i, metric in enumerate([
                'transaction_count', 'buy_pressure', 'volume', 'rsi',
                'price_volatility', 'volume_volatility', 'momentum',
                'trade_amount_variance', 'transaction_rate',
                'trade_concentration', 'unique_wallets'
            ]):
                feature_dict[f'{metric}_{window_type}'] = window_data[-1][i]
                
        return feature_dict

    # Keep all other methods unchanged (handle_transaction, _execute_trade, 
    # WebSocket handlers, etc.) from your original implementation

def main():
    """Main function to run the trading simulator"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_simulator.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        config = {
            'websocket_url': 'wss://pumpportal.fun/api/data',
        }
        
        peak_before_30_model_path = 'best_hit_peak_before_30_model.pth'
        price_classifier_model_path = 'token_price_classifier.pkl'

        simulator = TradingSimulator(
            config,
            peak_before_30_model_path,
            price_classifier_model_path
        )
        
        websocket_thread = threading.Thread(
            target=simulator.connect,
            daemon=True
        )
        websocket_thread.start()
        
        try:
            while True:
                time.sleep(300)
                simulator.print_trading_metrics()
                
                if not websocket_thread.is_alive():
                    logger.warning("WebSocket connection lost, restarting...")
                    websocket_thread = threading.Thread(
                        target=simulator.connect,
                        daemon=True
                    )
                    websocket_thread.start()
                
        except KeyboardInterrupt:
            logger.info("\n\nShutting down gracefully...")
            simulator.print_trading_metrics()
            if simulator.ws:
                simulator.ws.close()
            websocket_thread.join(timeout=5)
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()