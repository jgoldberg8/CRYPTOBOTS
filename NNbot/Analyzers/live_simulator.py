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
from Before30.models.peak_before_30_model import HitPeakBefore30Predictor
from PeakMarketCap.models.peak_market_cap_model import PeakMarketCapPredictor
from PeakMarketCap.models.token_dataset import TokenDataset

class ScalerManager:
    """Manages loading and saving of scalers for the trading simulator"""
    def __init__(self, scaler_dir='scalers'):
        self.scaler_dir = scaler_dir
        os.makedirs(scaler_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_scalers(self, model_type, global_scaler=None, target_scaler=None):
        """Save scalers based on model type"""
        try:
            if model_type == 'before30':
                if global_scaler:
                    global_path = os.path.join(self.scaler_dir, 'hit_peak_global_scaler.joblib')
                    joblib.dump(global_scaler, global_path)
                    self.logger.info(f"Before30 global scaler saved to {global_path}")
            
            elif model_type == 'market_cap':
                if global_scaler:
                    global_path = os.path.join(self.scaler_dir, 'global_scaler.joblib')
                    joblib.dump(global_scaler, global_path)
                    self.logger.info(f"Market cap global scaler saved to {global_path}")
                    
                if target_scaler:
                    target_path = os.path.join(self.scaler_dir, 'target_scaler.joblib')
                    joblib.dump(target_scaler, target_path)
                    self.logger.info(f"Market cap target scaler saved to {target_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving scalers: {e}")
            raise
            
    def load_scalers(self, model_type):
        """Load scalers based on model type"""
        global_scaler = None
        target_scaler = None
        
        try:
            if model_type == 'before30':
                global_path = os.path.join(self.scaler_dir, 'hit_peak_global_scaler.joblib')
                if os.path.exists(global_path):
                    global_scaler = joblib.load(global_path)
                    self.logger.info("Before30 global scaler loaded successfully")
                else:
                    self.logger.warning(f"Before30 global scaler not found at {global_path}")
                    
            elif model_type == 'market_cap':
                global_path = os.path.join(self.scaler_dir, 'global_scaler.joblib')
                target_path = os.path.join(self.scaler_dir, 'target_scaler.joblib')
                
                if os.path.exists(global_path):
                    global_scaler = joblib.load(global_path)
                    self.logger.info("Market cap global scaler loaded successfully")
                else:
                    self.logger.warning(f"Market cap global scaler not found at {global_path}")
                    
                if os.path.exists(target_path):
                    target_scaler = joblib.load(target_path)
                    self.logger.info("Market cap target scaler loaded successfully")
                else:
                    self.logger.warning(f"Market cap target scaler not found at {target_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
            
        return global_scaler, target_scaler
    

class TradingSimulator:
    def __init__(self, config, peak_before_30_model_path, peak_market_cap_model_path):
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
          self.before30_global_scaler, _ = self.scaler_manager.load_scalers('before30')
          if self.before30_global_scaler is None:
              self.before30_global_scaler = model_data['scaler']
              if self.before30_global_scaler is not None:
                  self.scaler_manager.save_scalers('before30', global_scaler=self.before30_global_scaler)
      except Exception as e:
          self.logger.error(f"Error loading peak_before_30 model: {e}")
          raise
          
      # Load market cap model and scalers
      try:
          market_cap_data = self._load_peak_market_cap_model(peak_market_cap_model_path)
          self.peak_market_cap_model = market_cap_data['model']
          self.market_cap_global_scaler, self.market_cap_target_scaler = self.scaler_manager.load_scalers('market_cap')
          
          # If scalers not found in files, use from model checkpoint
          if self.market_cap_global_scaler is None:
              self.market_cap_global_scaler = market_cap_data.get('global_scaler')
              if self.market_cap_global_scaler is not None:
                  self.scaler_manager.save_scalers('market_cap', global_scaler=self.market_cap_global_scaler)
                  
          if self.market_cap_target_scaler is None:
              self.market_cap_target_scaler = market_cap_data.get('target_scaler')
              if self.market_cap_target_scaler is not None:
                  self.scaler_manager.save_scalers('market_cap', target_scaler=self.market_cap_target_scaler)
      except Exception as e:
          self.logger.error(f"Error loading peak_market_cap model: {e}")
          raise
          
      # Verify required scalers are available
      if self.before30_global_scaler is None:
          self.logger.error("Failed to load Before30 global scaler")
          raise RuntimeError("Required Before30 scaler not available")
          
      if self.market_cap_global_scaler is None or self.market_cap_target_scaler is None:
          self.logger.error("Failed to load Market Cap scalers")
          raise RuntimeError("Required Market Cap scalers not available")
          
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
            
            # Initialize model
            model = HitPeakBefore30Predictor(
                input_size=11,
                hidden_size=256,
                num_layers=3,
                dropout_rate=0.5
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

    def _load_peak_market_cap_model(self, model_path):
        """Load the peak market cap prediction model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            model = PeakMarketCapPredictor(
                input_size=11,
                hidden_size=1024,
                num_layers=4,
                dropout_rate=0.4
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return {
                'model': model,
                'target_scaler': checkpoint.get('target_scaler')
            }
            
        except Exception as e:
            self.logger.error(f"Error in _load_peak_market_cap_model: {e}")
            raise

    def _calculate_features(self, token_data):
      """Calculate features for model prediction with robust error handling"""
      if not token_data['transactions']:
          self.logger.warning("No transactions available for feature calculation")
          return None
          
      try:
          start_time = token_data['first_trade_time']
          transactions = token_data['transactions']
          
          # Initialize features dictionary
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
          
          # Create DataFrame with features
          df = pd.DataFrame([features])
          
          # Add global features
          df['initial_investment_ratio'] = token_data['initial_investment_ratio']
          df['initial_market_cap'] = token_data['initial_market_cap']
          df['buy_sell_ratio'] = token_data['buy_sell_ratio']
          df['volume_pressure'] = token_data['volume_pressure']
          
          # Convert creation_time to numeric format
          creation_time = token_data['creation_time']
          if isinstance(creation_time, str):
              creation_time = pd.to_datetime(creation_time)
          df['creation_time'] = creation_time.strftime("%Y-%m-%d %H:%M:%S")
          df['creation_time_numeric'] = creation_time.timestamp()
          
          # Create correct global feature sets based on context
          is_peak_pred = 'predicted_peak' in token_data
          
          if is_peak_pred:
              # Global features for Market Cap model
              market_cap_global_features = pd.DataFrame([{
                  'initial_market_cap': token_data['initial_market_cap'],
                  'volume_pressure': token_data['volume_pressure'],
                  'buy_sell_ratio': token_data['buy_sell_ratio'],
                  'price_volatility': features['price_volatility_0to30s'],
                  'momentum': features['momentum_0to30s']
              }])
              df = pd.concat([df, market_cap_global_features], axis=1)
              df['peak_market_cap'] = token_data['current_market_cap']
          else:
              # Global features for Before30 model
              before30_global_features = pd.DataFrame([{
                  'initial_investment_ratio': token_data['initial_investment_ratio'],
                  'initial_market_cap': token_data['initial_market_cap'],
                  'volume_pressure': token_data['volume_pressure'],
                  'buy_sell_ratio': token_data['buy_sell_ratio'],
                  'price_volatility': features['price_volatility_0to30s'],
                  'volume_volatility': features['volume_volatility_0to30s'],
                  'momentum': features['momentum_0to30s']
              }])
              df = pd.concat([df, before30_global_features], axis=1)

          # Create dataset with correct scalers
          dataset = TokenDataset(
              df,
              scaler={
                  'global': self.market_cap_global_scaler if is_peak_pred else self.before30_global_scaler,
                  'target': self.market_cap_target_scaler if is_peak_pred else None
              },
              train=False
          )
          
          # Get preprocessed data
          processed_data = dataset._preprocess_data(df, fit=False)
          if processed_data is None:
              return None
          
          # Store the appropriate global features
          processed_data['global'] = (
              np.array(list(market_cap_global_features.iloc[0])) if is_peak_pred 
              else np.array(list(before30_global_features.iloc[0]))
          )
          
          # Reshape features for each time window
          for window_type in ['5s', '10s', '20s', '30s']:
              if window_type in processed_data['data']:
                  reshaped = self._reshape_features(processed_data['data'][window_type])
                  if reshaped is None:
                      self.logger.error(f"Failed to reshape {window_type} features")
                      return None
                  processed_data['data'][window_type] = reshaped
          
          return processed_data
          
      except Exception as e:
          self.logger.error(f"Error in feature calculation: {str(e)}")
          return None

    def _calculate_timeframe_metrics(self, transactions, start_time, start, end):
        """Calculate metrics for a specific timeframe"""
        default_metrics = {
            'transaction_count': 0,
            'buy_pressure': 0,
            'volume': 0,
            'rsi': 50,
            'price_volatility': 0,
            'volume_volatility': 0,
            'momentum': 0,
            'trade_amount_variance': 0,
            'transaction_rate': 0,
            'trade_concentration': 0,
            'unique_wallets': 0
        }

        try:
            interval_start = start_time + timedelta(seconds=start)
            interval_end = start_time + timedelta(seconds=end)
            
            window_txs = [tx for tx in transactions 
                         if interval_start <= tx['timestamp'] <= interval_end]

            if not window_txs:
                return default_metrics

            # Calculate metrics
            tx_count = len(window_txs)
            buy_count = sum(1 for tx in window_txs if tx['txType'] == 'buy')
            total_volume = sum(tx['solAmount'] for tx in window_txs)
            
            # Calculate RSI
            price_changes = [tx['marketCapSol'] - prev_tx['marketCapSol'] 
                            for prev_tx, tx in zip(window_txs[:-1], window_txs[1:])]
            
            if price_changes:
                gains = [change for change in price_changes if change > 0]
                losses = [-change for change in price_changes if change < 0]
                avg_gain = sum(gains) / len(price_changes) if gains else 0
                avg_loss = sum(losses) / len(price_changes) if losses else 0
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50

            # Calculate other metrics
            prices = [tx['marketCapSol'] for tx in window_txs]
            volumes = [tx['solAmount'] for tx in window_txs]
            
            price_volatility = np.std(prices) / np.mean(prices) if prices else 0
            volume_volatility = np.std(volumes) / np.mean(volumes) if volumes else 0
            momentum = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0

            # Calculate trade concentration
            wallets = defaultdict(float)
            for tx in window_txs:
                wallets[tx['wallet']] += tx['solAmount']
            
            if wallets:
                values = sorted(wallets.values())
                cumsum = np.cumsum(values)
                n = len(values)
                index = np.arange(1, n + 1)
                trade_concentration = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            else:
                trade_concentration = 0

            return {
                'transaction_count': tx_count,
                'buy_pressure': buy_count / tx_count if tx_count > 0 else 0,
                'volume': total_volume,
                'rsi': rsi,
                'price_volatility': price_volatility,
                'volume_volatility': volume_volatility,
                'momentum': momentum,
                'trade_amount_variance': np.var(volumes) if volumes else 0,
                'transaction_rate': tx_count / (end - start) if (end - start) > 0 else 0,
                'trade_concentration': trade_concentration,
                'unique_wallets': len(wallets)
            }
            
        except Exception as e:
            self.logger.error(f"Error in timeframe metrics calculation: {e}")
            return default_metrics
        
    def _scale_prediction(self, peak_pred, current_mcap):
      """Helper method to scale peak market cap predictions"""
      try:
          # Create dummy array for inverse transform
          dummy_pred = np.zeros((1, 2))
          dummy_pred[:, 0] = peak_pred.cpu().numpy()
          
          # Create temporary dataset for scaling
          df = pd.DataFrame([{'peak_market_cap': current_mcap}])
          temp_dataset = TokenDataset(df, train=False)
          
          transformed_pred = temp_dataset.target_scaler.inverse_transform(dummy_pred)
          return np.expm1(transformed_pred[0, 0])
      except Exception as e:
          self.logger.error(f"Error scaling prediction: {e}")
          return None



    def _reshape_features(self, features_array, expected_dim=11):
      """Helper method to ensure correct feature shape"""
      try:
          # Convert to numpy if tensor
          if torch.is_tensor(features_array):
              features_array = features_array.cpu().numpy()
              
          # Ensure we're working with numpy array
          features_array = np.array(features_array)
          
          # Reshape based on data structure
          if len(features_array.shape) == 3:
              if features_array.shape[-1] == expected_dim:
                  return features_array
              else:
                  self.logger.error(f"Feature dimension mismatch: expected {expected_dim}, got {features_array.shape[-1]}")
                  return None
          elif len(features_array.shape) == 2:
              # For 2D array, we need to add batch dimension
              if features_array.shape[-1] == expected_dim:
                  return features_array.reshape(1, -1, expected_dim)
              elif features_array.shape[0] == expected_dim:
                  return features_array.reshape(1, -1, expected_dim)
              else:
                  self.logger.error(f"Cannot determine correct reshaping for shape {features_array.shape}")
                  return None
          else:
              # Try to infer correct shape
              total_elements = features_array.size
              if total_elements % expected_dim == 0:
                  seq_len = total_elements // expected_dim
                  return features_array.reshape(1, seq_len, expected_dim)
              else:
                  self.logger.error(f"Cannot reshape features of size {features_array.shape} to match expected dimensions")
                  return None
                  
      except Exception as e:
          self.logger.error(f"Error reshaping features: {e}")
          return None  

    def _should_enter_trade(self, token_mint):
      """Determine if we should enter a trade based on model predictions"""
      try:
          token_data = self.active_tokens[token_mint]
          
          # Ensure minimum transactions
          if len(token_data['transactions']) < 5:
              return False
              
          # Calculate features
          features = self._calculate_features(token_data)
          if features is None:
              self.logger.warning(f"Could not calculate features for {token_mint}")
              return False

          # Debug log the input shapes
          self.logger.debug(f"Features before reshaping:")
          self.logger.debug(f"Global features shape: {features['global'].shape}")
          for window in ['5s', '10s', '20s', '30s']:
              self.logger.debug(f"{window} shape: {features['data'][window].shape}")

          # Prepare model inputs with correct shapes
          x_5s = torch.FloatTensor(features['data']['5s']).reshape(1, -1, 11).to(self.device)
          x_10s = torch.FloatTensor(features['data']['10s']).reshape(1, -1, 11).to(self.device)
          x_20s = torch.FloatTensor(features['data']['20s']).reshape(1, -1, 11).to(self.device)
          x_30s = torch.FloatTensor(features['data']['30s']).reshape(1, -1, 11).to(self.device)
          
          # Reshape global features to match expected dimensions (batch_size, feature_dim)
          global_features = torch.FloatTensor(features['global']).reshape(1, 7).to(self.device)  # 7 is the expected global feature dimension
          quality_features = torch.FloatTensor(self._calculate_quality_features(features)).to(self.device)

          # Debug log the reshaped tensors
          self.logger.debug(f"\nReshaped tensor shapes:")
          self.logger.debug(f"x_5s: {x_5s.shape}")
          self.logger.debug(f"x_10s: {x_10s.shape}")
          self.logger.debug(f"x_20s: {x_20s.shape}")
          self.logger.debug(f"x_30s: {x_30s.shape}")
          self.logger.debug(f"global_features: {global_features.shape}")
          self.logger.debug(f"quality_features: {quality_features.shape}")

          # Make predictions with error handling
          with torch.no_grad():
              self.logger.info(f"\n{'='*50}")
              self.logger.info(f"PREDICTION - Token: {token_mint}")
              
              try:
                  peak_before_30_pred = self.peak_before_30_model(
                      x_5s, x_10s, x_20s, x_30s,
                      global_features, quality_features
                  )
                  
                  # Convert to probability
                  prob_peaked = torch.sigmoid(peak_before_30_pred).item()
                  self.logger.info(f"Probability already peaked: {prob_peaked:.2%}")
                  
                  if prob_peaked < 0.5:
                      self.logger.info("Token hasn't peaked - Running peak_market_cap model...")
                      peak_pred = self.peak_market_cap_model(
                          x_5s, x_10s, x_20s, x_30s,
                          global_features, quality_features
                      )
                      
                      current_mcap = token_data['current_market_cap']
                      final_pred = self._scale_prediction(peak_pred, current_mcap)
                      
                      if final_pred is None:
                          return False
                          
                      potential_upside = ((final_pred/current_mcap - 1) * 100)
                      
                      self.logger.info(f"Current Market Cap: {current_mcap:.4f}")
                      self.logger.info(f"Predicted Peak: {final_pred:.4f}")
                      self.logger.info(f"Potential Upside: {potential_upside:.2f}%")
                      
                      if potential_upside > 20:
                          token_data['predicted_peak'] = final_pred
                          self.logger.info("Sufficient upside potential - Will enter trade")
                          self.logger.info(f"{'='*50}\n")
                          return True
                      else:
                          self.logger.info("Insufficient upside potential - Skipping trade")
                          self.logger.info(f"{'='*50}\n")
                          return False
                  else:
                      self.logger.info("Token predicted to have already peaked - Skipping trade")
                      self.logger.info(f"{'='*50}\n")
                      return False
                      
              except Exception as e:
                  self.logger.error(f"Error in model prediction: {e}")
                  return False
                  
      except Exception as e:
          self.logger.error(f"Error in _should_enter_trade for {token_mint}: {str(e)}")
          return False
    

    def handle_transaction(self, transaction):
      """Handle incoming transaction data"""
      try:
          mint = transaction['mint']
          current_time = datetime.now()
          
          # Initialize token data if new
          if mint not in self.active_tokens:
              self.active_tokens[mint] = self._initialize_token_data(transaction)
          
          token_data = self.active_tokens[mint]
          
          # Update token data
          if not token_data['first_trade_time']:
              token_data['first_trade_time'] = current_time
              token_data['initial_market_cap'] = transaction['marketCapSol']
          
          token_data['transactions'].append({
              'timestamp': current_time,
              **transaction
          })
          token_data['current_market_cap'] = transaction['marketCapSol']
          
          # Update the buy/sell ratio
          total_transactions = len(token_data['transactions'])
          buy_transactions = sum(1 for tx in token_data['transactions'] if tx['txType'] == 'buy')
          token_data['buy_sell_ratio'] = buy_transactions / total_transactions if total_transactions > 0 else 0.0
          
          # Update volume pressure
          total_volume = sum(tx['solAmount'] for tx in token_data['transactions'])
          token_data['volume_pressure'] = total_volume / (token_data['initial_market_cap'] + 1e-8)
          
          # Trading logic
          if token_data['trade_status'] == 'monitoring':
              time_since_first = (current_time - token_data['first_trade_time']).total_seconds()
              
              # Check if we should make a trading decision (after 30 seconds)
              if time_since_first >= 30:
                  if self._should_enter_trade(mint):
                      self._execute_trade(mint, 'buy')
                      
          elif token_data['trade_status'] == 'bought':
              # Check if we should sell based on predicted peak
              if token_data['current_market_cap'] >= token_data['predicted_peak']:
                  self._execute_trade(mint, 'sell')
                  
      except Exception as e:
          self.logger.error(f"Error in handle_transaction: {e}")

    def _write_trade_to_file(self, trade_data):
        """Write trade details to the PnL file with error handling"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = "trading_pnl.csv"
            
            # Create headers if file doesn't exist
            if not os.path.exists(filename):
                headers = [
                    "timestamp", "token", "action", "entry_price", "exit_price",
                    "position_size", "profit_loss", "hold_time", "predicted_peak",
                    "running_total_pnl", "win_rate"
                ]
                with open(filename, 'w') as f:
                    f.write(','.join(headers) + '\n')
            
            # Calculate running metrics
            win_rate = (self.trading_metrics['successful_trades'] / 
                       self.trading_metrics['total_trades'] * 100) \
                      if self.trading_metrics['total_trades'] > 0 else 0
                    
            # Write trade data
            with open(filename, 'a') as f:
                row = [
                    timestamp,
                    trade_data['token'],
                    trade_data['action'],
                    f"{trade_data.get('entry_price', '')}",
                    f"{trade_data.get('exit_price', '')}",
                    f"{trade_data.get('position_size', '')}",
                    f"{trade_data.get('profit_loss', '')}",
                    f"{trade_data.get('hold_time', '')}",
                    f"{trade_data.get('predicted_peak', '')}",
                    f"{self.trading_metrics['total_profit_loss']}",
                    f"{win_rate:.2f}"
                ]
                f.write(','.join(map(str, row)) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing trade to file: {e}")

    def _execute_trade(self, token_mint, action, amount=None):
        """Simulate executing a trade"""
        try:
            token_data = self.active_tokens[token_mint]
            current_price = token_data['current_market_cap']
            
            if action == 'buy':
                # Simulate buying
                token_data['entry_price'] = current_price
                token_data['position_size'] = amount or 1.0
                token_data['trade_status'] = 'bought'
                
                trade_data = {
                    'token': token_mint,
                    'action': 'buy',
                    'entry_price': current_price,
                    'position_size': token_data['position_size'],
                    'predicted_peak': token_data['predicted_peak']
                }
                
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"NEW TRADE - BUY")
                self.logger.info(f"Token: {token_mint}")
                self.logger.info(f"Entry Price: {current_price:.4f}")
                self.logger.info(f"Predicted Peak: {token_data['predicted_peak']:.4f}")
                self.logger.info(f"{'='*50}\n")
                
                self._write_trade_to_file(trade_data)
                self.positions[token_mint] = token_data
                
            elif action == 'sell':
                # Calculate profit/loss
                if token_mint in self.positions:
                    entry_price = token_data['entry_price']
                    position_size = token_data['position_size']
                    profit_loss = (current_price - entry_price) * position_size
                    hold_time = (datetime.now() - token_data['first_trade_time']).total_seconds()
                    
                    self.trading_metrics['total_trades'] += 1
                    if profit_loss > 0:
                        self.trading_metrics['successful_trades'] += 1
                    self.trading_metrics['total_profit_loss'] += profit_loss
                    
                    trade_data = {
                        'token': token_mint,
                        'action': 'sell',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position_size,
                        'profit_loss': profit_loss,
                        'hold_time': hold_time,
                        'predicted_peak': token_data['predicted_peak']
                    }
                    
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"TRADE CLOSED - SELL")
                    self.logger.info(f"Token: {token_mint}")
                    self.logger.info(f"Entry: {entry_price:.4f}")
                    self.logger.info(f"Exit: {current_price:.4f}")
                    self.logger.info(f"P/L: {profit_loss:.4f}")
                    self.logger.info(f"Hold Time: {hold_time:.2f}s")
                    self.logger.info(f"Running Total P/L: {self.trading_metrics['total_profit_loss']:.4f}")
                    self.logger.info(f"{'='*50}\n")
                    
                    self._write_trade_to_file(trade_data)
                    
                    # Record trade details
                    self.trading_metrics['positions'].append({
                        'token': token_mint,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'hold_time': hold_time
                    })
                    
                    # Clean up
                    del self.positions[token_mint]
                    token_data['trade_status'] = 'sold'
                    
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

    def _initialize_token_data(self, token_data):
      """Initialize data structure for tracking a new token"""
      return {
          'details': token_data,
          'creation_time': datetime.now(),
          'first_trade_time': None,
          'transactions': [],
          'metrics': defaultdict(lambda: {
              'transaction_count': 0,
              'buy_count': 0,
              'volume': 0,
              'prices': [],
              'volumes': [],
              'trade_amounts': [],
              'unique_wallets': set()
          }),
          'current_market_cap': None,
          'initial_market_cap': None,
          'peak_market_cap': 0,
          'predicted_peak': None,
          'entry_price': None,
          'position_size': 0,
          'trade_status': 'monitoring',
          # Add the missing fields
          'initial_investment_ratio': 1.0,  # Default value as seen in hit_peak_30_dataset.py
          'buy_sell_ratio': 0.0,  # Will be updated in handle_transaction
          'volume_pressure': 0.0  # Will be calculated as volume/initial_market_cap
      }

    def _calculate_quality_features(self, features):
        """Calculate data quality features"""
        try:
            # Basic completeness ratio
            completeness = np.mean([
                1 if features['data'][window_type].any() else 0
                for window_type in ['5s', '10s', '20s', '30s']
            ])
            
            # Active intervals calculation
            active_intervals = sum(
                features['data'][window_type].any()
                for window_type in ['5s', '10s', '20s', '30s']
            ) / 4.0
            
            return np.array([[completeness, active_intervals]])
            
        except Exception as e:
            self.logger.error(f"Error calculating quality features: {e}")
            return np.array([[0.0, 0.0]])  # Return default values on error

    def handle_token_creation(self, creation_data):
      """Handle new token creation event"""
      try:
          mint = creation_data['mint']
          token_entry = self._initialize_token_data(creation_data)
          self.active_tokens[mint] = token_entry
          
          # Subscribe to token trades
          if self.ws and self.ws.sock and self.ws.sock.connected:
              self._subscribed_tokens.add(mint)
              sub_message = {
                  "method": "subscribeTokenTrade",
                  "keys": [mint]
              }
              self.ws.send(json.dumps(sub_message))

      except Exception as e:
          self.logger.error(f"Error in handle_token_creation: {e}")


    def on_message(self, ws, message):
        """Handle WebSocket message"""
        try:
            data = json.loads(message)
            
            if isinstance(data, dict) and data.get('message'):
                return
            
            if data.get('txType') == 'create':
                self.handle_token_creation(data)
                
            elif data.get('txType') in ['buy', 'sell']:
                transaction = {
                    'mint': data['mint'],
                    'txType': data['txType'],
                    'tokenAmount': data['tokenAmount'],
                    'solAmount': data['solAmount'],
                    'marketCapSol': data['marketCapSol'],
                    'wallet': data.get('wallet', 'unknown')
                }
                self.handle_transaction(transaction)
        
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def connect(self):
        """Establish WebSocket connection"""
        if self.is_connecting:
            return
            
        self.is_connecting = True
        try:
            websocket.enableTrace(False)
            
            self.ws = websocket.WebSocketApp(
                self.config['websocket_url'],
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
                on_ping=self.on_ping
            )
            
            # Add websocket options for stability
            websocket_options = {
                'ping_interval': 30,
                'ping_timeout': 10,
                'skip_utf8_validation': True
            }
            
            self.logger.info(f"Attempting to connect to: {self.config['websocket_url']}")
            self.ws.run_forever(**websocket_options)
            
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            time.sleep(5)
            threading.Thread(target=self.connect, daemon=True).start()
        finally:
            self.is_connecting = False

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {str(error)}")
        if not self.is_connecting:
            time.sleep(5)
            self.connect()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure"""
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        if not self.is_connecting:
            time.sleep(5)
            threading.Thread(target=self.connect, daemon=True).start()

    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.logger.info("WebSocket connected")
        try:
            # Subscribe to new token creation events
            subscribe_msg = {"method": "subscribeNewToken"}
            ws.send(json.dumps(subscribe_msg))
            self.logger.info("Subscribed to new token events")
            
            # Subscribe to token trades events
            trade_msg = {"method": "subscribeTokenTrade"}
            ws.send(json.dumps(trade_msg))
            self.logger.info("Subscribed to token trade events")
            
        except Exception as e:
            self.logger.error(f"Error sending subscription messages: {e}")

    def on_ping(self, ws, message):
      """Handle ping messages"""
      try:
          # Correct way to send pong in websocket-client
          ws.send(message, websocket.ABNF.OPCODE_PONG)
      except Exception as e:
          self.logger.error(f"Error sending pong: {e}")

    def print_trading_metrics(self):
        """Print current trading metrics"""
        self.logger.info("\nTrading Metrics:")
        self.logger.info(f"Total Trades: {self.trading_metrics['total_trades']}")
        self.logger.info(f"Successful Trades: {self.trading_metrics['successful_trades']}")
        
        win_rate = (self.trading_metrics['successful_trades'] / self.trading_metrics['total_trades'] * 100) \
                   if self.trading_metrics['total_trades'] > 0 else 0
        self.logger.info(f"Win Rate: {win_rate:.2f}%")
        self.logger.info(f"Total P/L: {self.trading_metrics['total_profit_loss']:.4f}")
        
        # Print recent positions
        self.logger.info("\nRecent Positions:")
        for pos in self.trading_metrics['positions'][-5:]:  # Show last 5 trades
            self.logger.info(f"Token: {pos['token']}")
            self.logger.info(f"Entry: {pos['entry_price']:.4f}")
            self.logger.info(f"Exit: {pos['exit_price']:.4f}")
            self.logger.info(f"P/L: {pos['profit_loss']:.4f}")
            self.logger.info(f"Hold Time: {pos['hold_time']:.2f}s")
            self.logger.info("---")


def main():
    """Main function to run the trading simulator"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_simulator.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        # Configuration
        config = {
            'websocket_url': 'wss://pumpportal.fun/api/data',
        }
        
        # Model paths
        peak_before_30_model_path = 'best_hit_peak_before_30_model.pth'
        peak_market_cap_model_path = 'best_peak_market_cap_model.pth'

        # Verify model files exist
        for model_path in [peak_before_30_model_path, peak_market_cap_model_path]:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return

        logger.info("\n" + "="*50)
        logger.info("Starting Trading Simulator")
        logger.info("="*50 + "\n")

        # Initialize simulator
        simulator = TradingSimulator(
            config,
            peak_before_30_model_path,
            peak_market_cap_model_path
        )
        
        # Start WebSocket connection in a separate thread
        websocket_thread = threading.Thread(
            target=simulator.connect,
            daemon=True
        )
        websocket_thread.start()
        
        # Main loop with proper interrupt handling
        try:
            while True:
                time.sleep(300)  # Print metrics every 5 minutes
                simulator.print_trading_metrics()
                
                # Check if websocket is still alive
                if not websocket_thread.is_alive():
                    logger.warning("WebSocket connection lost, restarting...")
                    websocket_thread = threading.Thread(
                        target=simulator.connect,
                        daemon=True
                    )
                    websocket_thread.start()
                
        except KeyboardInterrupt:
            logger.info("\n\n" + "="*50)
            logger.info("Shutting down gracefully...")
            logger.info("="*50 + "\n")
            
            # Print final metrics
            simulator.print_trading_metrics()
            
            # Clean up
            if simulator.ws:
                simulator.ws.close()
            
            # Wait for websocket thread to finish
            websocket_thread.join(timeout=5)
            
            logger.info("\nTrading simulation stopped by user.")
            
    except Exception as e:
        logger.error(f"Critical error in trading simulation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure final metrics are saved
        try:
            if 'simulator' in locals():
                simulator.print_trading_metrics()
                
                # Save final state to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f'final_state_{timestamp}.json', 'w') as f:
                    json.dump({
                        'total_trades': simulator.trading_metrics['total_trades'],
                        'successful_trades': simulator.trading_metrics['successful_trades'],
                        'total_profit_loss': simulator.trading_metrics['total_profit_loss'],
                        'last_positions': simulator.trading_metrics['positions'][-5:] if simulator.trading_metrics['positions'] else []
                    }, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving final state: {e}")


if __name__ == "__main__":
    main()