from Before30.models.peak_before_30_model import HitPeakBefore30Predictor
from PeakMarketCap.models.peak_market_cap_model import PeakMarketCapPredictor
from PeakMarketCap.models.token_dataset import TokenDataset
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

class TradingSimulator:
    def __init__(self, config, peak_before_30_model_path, peak_market_cap_model_path):
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize WebSocket configuration
        self.config = config
        self.ws = None
        self.is_connecting = False
        self._subscribed_tokens = set()
        
        # Load ML models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.peak_before_30_model = self._load_peak_before_30_model(peak_before_30_model_path)
        self.peak_market_cap_model = self._load_peak_market_cap_model(peak_market_cap_model_path)
        
        # Trading state management
        self.active_tokens = {}  # Tokens we're currently tracking
        self.positions = {}      # Tokens we've bought and are holding
        
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
        checkpoint = torch.load(model_path, map_location=self.device)
        model = HitPeakBefore30Predictor(
            input_size=11,
            hidden_size=256,
            num_layers=3,
            dropout_rate=0.5
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _load_peak_market_cap_model(self, model_path):
        """Load the peak market cap prediction model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = PeakMarketCapPredictor(
            input_size=11,
            hidden_size=1024,
            num_layers=4,
            dropout_rate=0.4
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _calculate_timeframe_metrics(self, transactions, start_time, start, end):
        """Calculate metrics for a specific timeframe"""
        # Default metrics for empty windows
        default_metrics = {
            'transaction_count': 0,
            'buy_pressure': 0,
            'volume': 0,
            'rsi': 50,  # Neutral RSI when no data
            'price_volatility': 0,
            'volume_volatility': 0,
            'momentum': 0,
            'trade_amount_variance': 0,
            'transaction_rate': 0,
            'trade_concentration': 0,
            'unique_wallets': 0
        }

        # Calculate interval times
        interval_start = start_time + timedelta(seconds=start)
        interval_end = start_time + timedelta(seconds=end)
        
        # Filter transactions within window
        window_txs = [tx for tx in transactions 
                     if interval_start <= tx['timestamp'] <= interval_end]

        if not window_txs:
            return default_metrics

        # Calculate basic metrics
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

        # Volatility metrics
        prices = [tx['marketCapSol'] for tx in window_txs]
        volumes = [tx['solAmount'] for tx in window_txs]
        price_volatility = np.std(prices) / np.mean(prices) if prices else 0
        volume_volatility = np.std(volumes) / np.mean(volumes) if volumes else 0

        # Momentum
        momentum = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0

        # Trade concentration (Gini coefficient)
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

    def _calculate_features(self, token_data):
        """Calculate features for model prediction"""
        if not token_data['transactions']:
            return None
            
        start_time = token_data['first_trade_time']
        transactions = token_data['transactions']
        
        # Initialize features dictionary
        features = {}
        
        # Calculate metrics for all timeframes
        for window_type, intervals in self.time_windows.items():
            for start, end in intervals:
                window_key = f"{start}to{end}s"
                metrics = self._calculate_timeframe_metrics(transactions, start_time, start, end)
                for feature, value in metrics.items():
                    features[f'{feature}_{window_key}'] = value
        
        # Add global features
        features['initial_investment_ratio'] = 1.0  # Default value
        features['initial_market_cap'] = token_data['initial_market_cap']
        features['volume_pressure'] = features['volume_0to30s'] / (features['initial_market_cap'] + 1)
        features['buy_sell_ratio'] = features['buy_pressure_0to30s']
        features['creation_time_numeric'] = token_data['creation_time'].hour + token_data['creation_time'].minute / 60
        
        # Convert to DataFrame for TokenDataset
        df = pd.DataFrame([features])
        dataset = TokenDataset(df, train=False)
        
        return dataset._preprocess_data(df, fit=False)

    def _should_enter_trade(self, token_mint):
      """Determine if we should enter a trade based on model predictions"""
      try:
          token_data = self.active_tokens[token_mint]
          
          # Ensure we have 30 seconds of data
          if len(token_data['transactions']) < 5:  # Minimum transactions threshold
              print(f"Not enough transactions for {token_mint}: {len(token_data['transactions'])}")
              return False
              
          # Calculate features
          features = self._calculate_features(token_data)
          if features is None:
              print(f"Could not calculate features for {token_mint}")
              return False

          # Create dataset inputs
          x_5s = torch.FloatTensor(features['data']['5s']).unsqueeze(0).to(self.device)
          x_10s = torch.FloatTensor(features['data']['10s']).unsqueeze(0).to(self.device)
          x_20s = torch.FloatTensor(features['data']['20s']).unsqueeze(0).to(self.device)
          x_30s = torch.FloatTensor(features['data']['30s']).unsqueeze(0).to(self.device)
          global_features = torch.FloatTensor(features['global']).unsqueeze(0).to(self.device)
          quality_features = torch.FloatTensor(self._calculate_quality_features(features)).to(self.device)
          
          # Make prediction with peak_before_30 model
          with torch.no_grad():
              # First model prediction
              print(f"\n{'*'*50}")
              print(f"PREDICTION - Token: {token_mint}")
              print("Input tensor shapes:")
              print(f"x_5s: {x_5s.shape}")
              print(f"x_10s: {x_10s.shape}")
              print(f"x_20s: {x_20s.shape}")
              print(f"x_30s: {x_30s.shape}")
              print(f"global_features: {global_features.shape}")
              print(f"quality_features: {quality_features.shape}")
              
              print(f"Running peak_before_30 model...")
              
              peak_before_30_pred = self.peak_before_30_model(
                  x_5s, x_10s, x_20s, x_30s,
                  global_features, quality_features
              )
              
              # Convert to probability
              prob_peaked = torch.sigmoid(peak_before_30_pred).item()
              print(f"Probability already peaked: {prob_peaked:.2%}")
              
              # If probability of having peaked is low, predict final peak
              if prob_peaked < 0.5:
                  print("Token hasn't peaked - Running peak_market_cap model...")
                  peak_pred = self.peak_market_cap_model(
                      x_5s, x_10s, x_20s, x_30s,
                      global_features, quality_features
                  )
                  
                  # Convert prediction back to original scale
                  dummy_pred = np.zeros((1, 2))
                  dummy_pred[:, 0] = peak_pred.cpu().numpy()
                  # Create dummy dataset for scaling
                  df = pd.DataFrame([{
                      'peak_market_cap': token_data['current_market_cap']
                  }])
                  temp_dataset = TokenDataset(df, train=False)
                  
                  transformed_pred = temp_dataset.target_scaler.inverse_transform(dummy_pred)
                  final_pred = np.expm1(transformed_pred[0, 0])
                  
                  current_mcap = token_data['current_market_cap']
                  potential_upside = ((final_pred/current_mcap - 1) * 100)
                  
                  print(f"Current Market Cap: {current_mcap:.4f}")
                  print(f"Predicted Peak: {final_pred:.4f}")
                  print(f"Potential Upside: {potential_upside:.2f}%")
                  
                  if potential_upside > 20:  # Only trade if potential upside > 20%
                      token_data['predicted_peak'] = final_pred
                      print(f"Sufficient upside potential - Will enter trade")
                      print(f"{'*'*50}\n")
                      return True
                  else:
                      print(f"Insufficient upside potential - Skipping trade")
                      print(f"{'*'*50}\n")
                      return False
              else:
                  print("Token predicted to have already peaked - Skipping trade")
                  print(f"{'*'*50}\n")
                  return False
                  
      except Exception as e:
          print(f"Error in _should_enter_trade for {token_mint}: {str(e)}")
          import traceback
          traceback.print_exc()
          return False

    def _write_trade_to_file(self, trade_data):
      """Write trade details to the PnL file"""
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
      win_rate = (self.trading_metrics['successful_trades'] / self.trading_metrics['total_trades'] * 100) \
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

    def _execute_trade(self, token_mint, action, amount=None):
      """Simulate executing a trade"""
      token_data = self.active_tokens[token_mint]
      current_price = token_data['current_market_cap']
      
      if action == 'buy':
          # Simulate buying
          token_data['entry_price'] = current_price
          token_data['position_size'] = amount or 1.0  # Default to 1 unit if no amount specified
          token_data['trade_status'] = 'bought'
          
          trade_data = {
              'token': token_mint,
              'action': 'buy',
              'entry_price': current_price,
              'position_size': token_data['position_size'],
              'predicted_peak': token_data['predicted_peak']
          }
          
          # Print to console
          print(f"\n{'='*50}")
          print(f"NEW TRADE - BUY")
          print(f"Token: {token_mint}")
          print(f"Entry Price: {current_price:.4f}")
          print(f"Predicted Peak: {token_data['predicted_peak']:.4f}")
          print(f"{'='*50}\n")
          
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
              
              # Print to console
              print(f"\n{'='*50}")
              print(f"TRADE CLOSED - SELL")
              print(f"Token: {token_mint}")
              print(f"Entry: {entry_price:.4f}")
              print(f"Exit: {current_price:.4f}")
              print(f"P/L: {profit_loss:.4f}")
              print(f"Hold Time: {hold_time:.2f}s")
              print(f"Running Total P/L: {self.trading_metrics['total_profit_loss']:.4f}")
              print(f"{'='*50}\n")
              
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

    def handle_transaction(self, transaction):
      """Handle incoming transaction data"""
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

    def handle_token_creation(self, creation_data):
      """Handle new token creation event"""
      try:
          mint = creation_data['mint']
          # print(f"Initializing new token: {mint}")  # Debug log
          
          token_entry = self._initialize_token_data(creation_data)
          self.active_tokens[mint] = token_entry
          
          # print(f"Successfully initialized token {mint}")  # Debug log
          
          # Subscribe to token trades
          if self.ws and self.ws.sock and self.ws.sock.connected:
              self._subscribed_tokens.add(mint)
              sub_message = {
                  "method": "subscribeTokenTrade",
                  "keys": [mint]
              }
              self.ws.send(json.dumps(sub_message))
              # print(f"Subscribed to trades for {mint}")  # Debug log

      except Exception as e:
          print(f"Error in handle_token_creation: {str(e)}")  # Debug log
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
          websocket.enableTrace(False)  # Enable detailed websocket logging
          
          self.ws = websocket.WebSocketApp(
              self.config['websocket_url'],
              on_message=self.on_message,
              on_error=self.on_error,
              on_close=self.on_close,
              on_open=self.on_open
          )
          
          # Add websocket options for stability
          websocket_options = {
              'ping_interval': 30,
              'ping_timeout': 10,
              'skip_utf8_validation': True
          }
          
          print(f"Attempting to connect to: {self.config['websocket_url']}")
          self.ws.run_forever(**websocket_options)
          
      except Exception as e:
          self.logger.error(f"Connection error: {str(e)}")
          print(f"Detailed connection error: {str(e)}")
          time.sleep(5)
          threading.Thread(target=self.connect, daemon=True).start()
      finally:
          self.is_connecting = False

    def on_error(self, ws, error):
      """Handle WebSocket errors"""
      self.logger.error(f"WebSocket error: {str(error)}")
      print(f"Detailed WebSocket error: {str(error)}")
      if not self.is_connecting:
          time.sleep(5)
          self.connect()

    def on_close(self, ws, close_status_code, close_msg):
      """Handle WebSocket closure"""
      self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
      print(f"WebSocket closed with status {close_status_code}: {close_msg}")
      if not self.is_connecting:
          time.sleep(5)
          threading.Thread(target=self.connect, daemon=True).start()

    def on_open(self, ws):
      """Handle WebSocket connection open"""
      self.logger.info("WebSocket connected")
      print("WebSocket connection established")
      try:
          # Subscribe to new token creation events
          subscribe_msg = {"method": "subscribeNewToken"}
          ws.send(json.dumps(subscribe_msg))
          print("Subscribed to new token events")
          
          # Subscribe to token trades events
          trade_msg = {"method": "subscribeTokenTrade"}
          ws.send(json.dumps(trade_msg))
          print("Subscribed to token trade events")
          
      except Exception as e:
          print(f"Error sending subscription messages: {str(e)}")

    def on_ping(self, ws, message):
      """Handle ping messages"""
      # print("Received ping")
      try:
          ws.pong(message)
      except Exception as e:
          print(f"Error sending pong: {str(e)}")

    def print_trading_metrics(self):
        """Print current trading metrics"""
        print("\nTrading Metrics:")
        print(f"Total Trades: {self.trading_metrics['total_trades']}")
        print(f"Successful Trades: {self.trading_metrics['successful_trades']}")
        win_rate = (self.trading_metrics['successful_trades'] / self.trading_metrics['total_trades'] * 100) \
                   if self.trading_metrics['total_trades'] > 0 else 0
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P/L: {self.trading_metrics['total_profit_loss']:.4f}")
        
        # Print recent positions
        print("\nRecent Positions:")
        for pos in self.trading_metrics['positions'][-5:]:  # Show last 5 trades
            print(f"Token: {pos['token']}")
            print(f"Entry: {pos['entry_price']:.4f}")
            print(f"Exit: {pos['exit_price']:.4f}")
            print(f"P/L: {pos['profit_loss']:.4f}")
            print(f"Hold Time: {pos['hold_time']:.2f}s")
            print("---")

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
            'trade_status': 'monitoring'  # monitoring, bought, sold
        }

    def _calculate_quality_features(self, features):
        """Calculate data quality features"""
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


def main():
    """Main function to run the trading simulator"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Configuration
    config = {
        'websocket_url': 'wss://pumpportal.fun/api/data',
    }
    
    # Model paths (ensure these exist)
    peak_before_30_model_path = 'best_hit_peak_before_30_model.pth'
    peak_market_cap_model_path = 'best_peak_market_cap_model.pth'

    # Verify model files exist
    for model_path in [peak_before_30_model_path, peak_market_cap_model_path]:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return

    try:
        print("\n" + "="*50)
        print("Starting Trading Simulator")
        print("="*50 + "\n")

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
            print("\n\n" + "="*50)
            print("Shutting down gracefully...")
            print("="*50 + "\n")
            
            # Print final metrics
            simulator.print_trading_metrics()
            
            # Clean up
            if simulator.ws:
                simulator.ws.close()
            
            # Wait for websocket thread to finish
            websocket_thread.join(timeout=5)
            
            print("\nTrading simulation stopped by user.")
            
    except Exception as e:
        logger.error(f"Critical error in trading simulation: {str(e)}")
        print(f"\nDetailed error: {str(e)}")
        
        # Print stack trace for debugging
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
            logger.error(f"Error saving final state: {str(e)}")

if __name__ == "__main__":
    main()