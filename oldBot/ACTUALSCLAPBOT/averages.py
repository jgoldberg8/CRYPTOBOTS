import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

class TokenAnalyzer:
    def __init__(self):
        self.good_tokens = []
        self.bad_tokens = []
        self.neutral_tokens = []

    def process_log_line(self, line: str) -> None:
        """Process a single log line and categorize the token."""
        try:
            # Extract JSON data from the log line
            json_start = line.find('{')
            if json_start == -1:
                return
            json_str = line[json_start:]
            token_data = json.loads(json_str)

            # Categorize based on classification
            classification = token_data.get('buy_classification')
            
            if classification == 'good':
                self.good_tokens.append(token_data)
            elif classification == 'bad':
                self.bad_tokens.append(token_data)
            elif classification == 'neutral':
                self.neutral_tokens.append(token_data)

        except json.JSONDecodeError:
            print(f"Failed to parse JSON from line: {line[:100]}...")
        except Exception as e:
            print(f"Error processing line: {str(e)}")

    def calculate_stats(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a group of tokens."""
        if not tokens:
            return {}

        # Technical Indicators
        def extract_numeric_list(key, nested_key=None):
            if nested_key:
                return [t[key][nested_key] for t in tokens if key in t and nested_key in t[key]]
            return [t[key] for t in tokens if key in t and t[key] is not None]

        # Basic Metrics
        peak_returns = extract_numeric_list('peak_return')
        max_drawdowns = extract_numeric_list('max_drawdown')
        transaction_rates = extract_numeric_list('transaction_rate')
        buy_sell_ratios = extract_numeric_list('buy_sell_ratio')
        volumes = extract_numeric_list('volume')
        volume_changes = extract_numeric_list('volume_change')
        initial_market_caps = extract_numeric_list('initial_market_cap')
        peak_market_caps = extract_numeric_list('peak_market_cap')
        times_to_rug = extract_numeric_list('time_to_rug')

        # Wallet Statistics
        unique_wallets = [t['trade_patterns']['unique_wallets'] for t in tokens if 'trade_patterns' in t and 'unique_wallets' in t['trade_patterns']]
        trade_concentrations = [t['trade_patterns']['trade_concentration'] for t in tokens if 'trade_patterns' in t and 'trade_concentration' in t['trade_patterns']]

        # Social Signals
        total_tokens = len(tokens)
        twitter_percent = sum(1 for t in tokens if t.get('twitter')) / total_tokens * 100 if total_tokens > 0 else 0
        telegram_percent = sum(1 for t in tokens if t.get('telegram')) / total_tokens * 100 if total_tokens > 0 else 0
        website_percent = sum(1 for t in tokens if t.get('website')) / total_tokens * 100 if total_tokens > 0 else 0

        # Trade Window Analysis
        def calculate_trade_window_stats():
            avg_values = defaultdict(list)

            for token in tokens:
                for key, value in token.items():
                    if key.endswith('_total_trades') or key.endswith('_total_volume'):
                        avg_values[key].append(value)

            return {key: np.mean(values) for key, values in avg_values.items()}

        # Compile statistics
        stats = {
            "count": total_tokens,
            "avg_peak_return": np.mean(peak_returns) if peak_returns else None,
            "avg_max_drawdown": np.mean(max_drawdowns) if max_drawdowns else None,
            "avg_transaction_rate": np.mean(transaction_rates) if transaction_rates else None,
            "avg_buy_sell_ratio": np.mean(buy_sell_ratios) if buy_sell_ratios else None,
            "avg_volume": np.mean(volumes) if volumes else None,
            "avg_volume_change": np.mean(volume_changes) if volume_changes else None,
            "avg_initial_market_cap": np.mean(initial_market_caps) if initial_market_caps else None,
            "avg_peak_market_cap": np.mean(peak_market_caps) if peak_market_caps else None,
            "rug_percentage": sum(1 for t in tokens if t.get('time_to_rug') is not None) / total_tokens * 100 if total_tokens > 0 else 0,
            "avg_time_to_rug": np.mean(times_to_rug) if times_to_rug else None,

            # Wallet Statistics
            "avg_unique_wallets": np.mean(unique_wallets) if unique_wallets else None,
            "max_unique_wallets": max(unique_wallets) if unique_wallets else None,
            "min_unique_wallets": min(unique_wallets) if unique_wallets else None,
            "avg_trade_concentration": np.mean(trade_concentrations) if trade_concentrations else None,

            # Social Signals
            "twitter_percent": twitter_percent,
            "telegram_percent": telegram_percent,
            "website_percent": website_percent,
        }

        # Merge trade window stats
        stats.update(calculate_trade_window_stats())

        return stats



    def analyze_and_print_results(self):
        """Analyze and print statistics for all token categories."""
        good_stats = self.calculate_stats(self.good_tokens)
        bad_stats = self.calculate_stats(self.bad_tokens)
        neutral_stats = self.calculate_stats(self.neutral_tokens)

        print("\n=== GOOD TOKENS STATISTICS ===")
        self._print_stats(good_stats)
        
        print("\n=== BAD TOKENS STATISTICS ===")
        self._print_stats(bad_stats)
        
        print("\n=== NEUTRAL TOKENS STATISTICS ===")
        self._print_stats(neutral_stats)

    def _print_stats(self, stats: Dict[str, Any]) -> None:
        """Print statistics in a formatted way."""
        print(f"Total Count: {stats.get('count', 0)}")
        
        # Basic Metrics
        print(f"Average Peak Return: {stats.get('avg_peak_return', 0):.2%}")
        print(f"Average Max Drawdown: {stats.get('avg_max_drawdown', 0):.2%}")
        print(f"Average Transaction Rate: {stats.get('avg_transaction_rate', 0):.2f} tx/min")
        print(f"Average Buy/Sell Ratio: {stats.get('avg_buy_sell_ratio', 0):.2f}")
        
        # Volume Metrics
        print(f"Average Volume: {stats.get('avg_volume', 0):.2f}")
        print(f"Average Volume Change: {stats.get('avg_volume_change', 0):.2%}")
        
        # Market Cap Metrics
        print(f"Average Initial Market Cap: {stats.get('avg_initial_market_cap', 0):.2f}")
        print(f"Average Peak Market Cap: {stats.get('avg_peak_market_cap', 0):.2f}")
        
        # Wallet Metrics
        print("\nWallet Statistics:")
        print(f"Average Unique Wallets: {stats.get('avg_unique_wallets', 0):.2f}")
        print(f"Maximum Unique Wallets: {stats.get('max_unique_wallets', 0):.0f}")
        print(f"Minimum Unique Wallets: {stats.get('min_unique_wallets', 0):.0f}")
        print(f"Average Trade Concentration: {stats.get('avg_trade_concentration', 0):.4f}")
        
        # Social Signals
        print("\nSocial Statistics:")
        print(f"Twitter Percentage: {stats.get('twitter_percent', 0):.2f}%")
        print(f"Telegram Percentage: {stats.get('telegram_percent', 0):.2f}%")
        print(f"Website Percentage: {stats.get('website_percent', 0):.2f}%")

        # Rug Metrics
        print("\nRug Statistics:")
        print(f"Rug Percentage: {stats.get('rug_percentage', 0):.2f}%")
        avg_time_to_rug = stats.get('avg_time_to_rug')
        if avg_time_to_rug is not None:
            print(f"Average Time to Rug: {avg_time_to_rug:.2f} seconds")
        else:
            print("Average Time to Rug: N/A")

        # Trade Window Averages
        print("\nTrade Window Averages:")
        for key, value in stats.items():
            if key.endswith('_total_trades') or key.endswith('_total_volume'):
                print(f"{key}: {value:.2f}")

def main():
    analyzer = TokenAnalyzer()
    
    # Read the log file
    with open('logs/new-data.log', 'r', encoding='utf-8') as f:
        for line in f:
            analyzer.process_log_line(line)
    
    # Print results
    analyzer.analyze_and_print_results()

if __name__ == "__main__":
    main()