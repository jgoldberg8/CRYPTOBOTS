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

class TokenDataCollector:
    def __init__(self, config):
        # Time windows for data collection (in seconds)
        self.time_windows = self.generate_time_windows()
        
        # Create log directory if it doesn't exist
        log_directory = config.get('log_directory', './data')
        os.makedirs(log_directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize data storage file
        # self.data_file = os.path.join(log_directory, f'token_data_{datetime.now().strftime("%Y-%m-%d")}.csv')
        self.data_file = os.path.join(log_directory, 'temp.csv')
        self._initialize_data_file()

        # WebSocket configuration
        self.config = config
        self.ws = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 5
        self.max_delay = 60
        self.is_connecting = False
        self._subscribed_tokens = set()

        # Token data storage
        self.token_data: Dict[str, dict] = {}

    def generate_time_windows(self):
        intervals = {
            '5s': [],
            '10s': [],
            '20s': [],
            '30s': [],
            '60s': []
        }
        
        # 5-second intervals
        for start in range(0, 1021, 5):
            intervals['5s'].append((start, start + 5))
        
        
        # 10-second intervals
        for start in range(0, 1021, 10):
            intervals['10s'].append((start, start + 10))
        
        
        # 20-second intervals
        for start in range(0, 1021, 20):
            intervals['20s'].append((start, start + 20))
        
        
        # 30-second intervals
        for start in range(0, 1021, 30):
            intervals['30s'].append((start, start + 30))

        
        
        return intervals  
        
    def _initialize_data_file(self):
        """Initialize the CSV file with headers"""
        if not os.path.exists(self.data_file):
            headers = self._generate_headers()
            with open(self.data_file, 'w') as f:
                f.write(','.join(headers) + '\n')

    def _generate_headers(self) -> List[str]:
        """Generate CSV headers for all features and timeframes"""
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

        headers = ['mint', 'creation_time']

        # Add timeframe-based features
        for interval_type, intervals in self.time_windows.items():
            for i, (start, end) in enumerate(intervals):
                interval_name = f"{start}to{end}"
                for feature in base_features:
                    headers.append(f'{feature}_{interval_name}s')

        # Add global features
        headers.extend([
            'initial_investment_ratio',
            'initial_market_cap',
            'peak_market_cap',
            'time_to_peak'
        ])

        return headers


    def _initialize_token_metrics(self, token_entry: dict) -> dict:
        """Initialize metrics structure for a token"""
        metrics = {
            'creation_time': datetime.now(),
            'first_trade_time': None,
            'initial_market_cap': None,
            'peak_market_cap': 0,
            'peak_time': None,
            'transactions': [],
            'timeframe_metrics': defaultdict(lambda: {
                'transaction_count': 0,
                'buy_count': 0,
                'volume': 0,
                'prices': [],
                'volumes': [],
                'trade_amounts': [],
                'unique_wallets': set(),
                'last_price': None
            })
        }
        return metrics

    def _calculate_timeframe_metrics(self, token: dict, start: int, end: int) -> dict:
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

        if not token.get('first_trade_time'):
            return default_metrics

        # Ensure first_trade_time is a datetime object
        first_trade_time = token['first_trade_time']
        if isinstance(first_trade_time, str):
            first_trade_time = datetime.fromisoformat(first_trade_time.replace('Z', '+00:00'))
        
        # Calculate interval duration and times
        interval_duration = end - start
        interval_start = first_trade_time + timedelta(seconds=start)
        interval_end = first_trade_time + timedelta(seconds=end)
        
        # Filter transactions within this specific interval
        window_txs = [tx for tx in token['transactions'] 
                    if isinstance(tx['timestamp'], datetime) and  # Ensure timestamp is datetime
                    interval_start <= tx['timestamp'] <= interval_end]

        if not window_txs:
            return default_metrics

        # Basic metrics
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
            rsi = 50  # Default value when not enough data

        # Volatility metrics
        prices = [tx['marketCapSol'] for tx in window_txs]
        volumes = [tx['solAmount'] for tx in window_txs]
        price_volatility = np.std(prices) / np.mean(prices) if prices else 0
        volume_volatility = np.std(volumes) / np.mean(volumes) if volumes else 0

        # Momentum (price change over period)
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
            'transaction_rate': tx_count / interval_duration if interval_duration > 0 else 0,
            'trade_concentration': trade_concentration,
            'unique_wallets': len(wallets)
        }

    def _check_record_completion(self, token: dict):
        """Check if token data should be recorded (after 3 minutes)"""
        if not token.get('first_trade_time'):
            return

        current_time = datetime.now()
        if isinstance(token['first_trade_time'], str):
            token['first_trade_time'] = datetime.fromisoformat(token['first_trade_time'].replace('Z', '+00:00'))
            
        time_since_first = (current_time - token['first_trade_time']).total_seconds()

        if time_since_first >= 1020 and not token.get('recorded', False):
            self._record_token_data(token)
            token['recorded'] = True

    def _record_token_data(self, token: dict):
        """Record token data to CSV file"""
        try:
            current_time = datetime.now()
            data = {
                'mint': token['details']['mint'],
                'creation_time': token['creation_time'].isoformat()
            }

            # Calculate metrics for each interval type and its intervals
            for interval_type, intervals in self.time_windows.items():
                for start, end in intervals:
                    metrics = self._calculate_timeframe_metrics(token, start, end)
                    interval_name = f"{start}to{end}"
                    for feature, value in metrics.items():
                        data[f'{feature}_{interval_name}s'] = value

            # Add global metrics with defaults if missing
            data.update({
                'initial_investment_ratio': token.get('initial_investment_ratio', 1.0),
                'initial_market_cap': token.get('initial_market_cap', 0),
                'peak_market_cap': token.get('peak_market_cap', 0),
                'time_to_peak': (token['peak_time'] - token['first_trade_time']).total_seconds()
                    if token.get('peak_time') and token.get('first_trade_time') else 0
            })

            # Write to CSV
            with open(self.data_file, 'a') as f:
                headers = self._generate_headers()
                values = [str(data.get(header, '')) for header in headers]
                f.write(','.join(values) + '\n')

        except Exception as e:
            self.logger.error(f"Error recording token data: {e}")

    def handle_token_creation(self, creation_data: dict):
        """Handle new token creation event"""
        try:
            mint = creation_data['mint']
            
            token_entry = {
                'details': creation_data,
                'creation_time': datetime.now(),
                'first_trade_time': None,
                'initial_market_cap': None,
                'initial_investment_ratio': None,
                'peak_market_cap': 0,
                'peak_time': None,
                'transactions': [],
                'recorded': False
            }

            self.token_data[mint] = token_entry

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

    def handle_transaction(self, transaction: dict):
        try:
            mint = transaction['mint']
            token = self.token_data.get(mint)
            if not token:
                return

            current_time = datetime.now()

            # Initialize first trade data
            if not token.get('first_trade_time'):
                token['first_trade_time'] = current_time
                token['initial_market_cap'] = transaction['marketCapSol']
                token['initial_investment_ratio'] = 1.0

            # Store transaction with datetime object
            token['transactions'].append({
                'timestamp': current_time,  # Ensure this is a datetime object
                **transaction
            })

            # Update peak market cap (only within first 3 minutes)
            first_trade_time = token['first_trade_time']
            if isinstance(first_trade_time, str):
                first_trade_time = datetime.fromisoformat(first_trade_time.replace('Z', '+00:00'))
                token['first_trade_time'] = first_trade_time
                
            time_since_first = (current_time - first_trade_time).total_seconds()
            if time_since_first <= 1020 and transaction['marketCapSol'] > token['peak_market_cap']:
                token['peak_market_cap'] = transaction['marketCapSol']
                token['peak_time'] = current_time

            # Check if we should record the data
            self._check_record_completion(token)

        except Exception as e:
            self.logger.error(f"Error handling transaction: {e}")

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

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        if not self.is_connecting:
            self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
            threading.Thread(target=self.connect, daemon=True).start()

    def on_open(self, ws):
        self.logger.info("WebSocket connected")
        subscribe_msg = {"method": "subscribeNewToken"}
        ws.send(json.dumps(subscribe_msg))

    def connect(self):
        """Establish WebSocket connection"""
        try:
            if self.is_connecting:
                return
                
            self.is_connecting = True
            websocket.enableTrace(False)
            
            self.ws = websocket.WebSocketApp(
                self.config['websocket_url'],
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            self.ws.run_forever(
                reconnect=5,
                ping_interval=30,
                ping_timeout=10
            )
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.is_connecting = False
            time.sleep(5)
            threading.Thread(target=self.connect, daemon=True).start()
        finally:
            self.is_connecting = False

def main():
    """Main function to run the token data collector"""
    config = {
        'websocket_url': 'wss://pumpportal.fun/api/data',
        'log_directory': './data'
    }

    try:
        collector = TokenDataCollector(config)
        collector.connect()
    except KeyboardInterrupt:
        print("\nData collection stopped by user.")
    except Exception as e:
        print(f"Critical error in data collection system: {e}")

if __name__ == "__main__":
    main()