import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from TimeToPeak.datasets.time_token_dataset import MultiGranularTokenDataset
from TimeToPeak.models.time_to_peak_model import RealTimePeakPredictor
from TimeToPeak.utils.clean_dataset import clean_dataset

class RealTimeEvaluator:
    def __init__(self, model_path, initial_window=30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_window = initial_window
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._load_model()
        self.model.eval()
        
        if 'scaler' in self.checkpoint:
            self.scaler = self.checkpoint['scaler']
        else:
            raise ValueError("Scaler not found in model checkpoint")
            
        self.predictions = {}
        self.true_values = []
        self.predicted_values = []
        self.prediction_times = []
        
    def _load_model(self):
        model = RealTimePeakPredictor().to(self.device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        return model
        
    def prepare_features(self, features_dict, current_time, mint):
        """Prepare features in format expected by model for a given timestamp"""
        # Add mint back to the features
        features_dict['mint'] = mint
        
        # print("\n--- Preparing Features ---")
        # print("Full features dictionary:")
        # for k, v in features_dict.items():
        #     print(f"{k}: {v}")
        
        # Create a DataFrame with the features
        try:
            df = pd.DataFrame([features_dict])
        except Exception as e:
            print(f"DataFrame creation error: {e}")
            raise
        
        # print("\nDataFrame columns:")
        # print(df.columns)
        
        # Attempt to create dataset
        try:
            dataset = MultiGranularTokenDataset(
                df, 
                scaler=self.scaler,
                train=False
            )
        except Exception as e:
            print(f"Dataset creation error: {e}")
            print("Type of error:", type(e))
            # If possible, print the full traceback
            import traceback
            traceback.print_exc()
            raise
        
        # Get sample in tensor format
        sample = dataset[0]
        
        # Move to device and add batch dimension
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in sample.items()}
        return batch
        
    def evaluate_token(self, token_df):
        """Simulate real-time evaluation of a single token"""
        mint = token_df['mint'].iloc[0]
        true_peak = token_df['time_to_peak'].iloc[0]
        
        print(f"\n=== Evaluating Token: {mint} ===")
        print(f"True peak time: {true_peak}")
        
        # Global features to include
        global_features = {
            'initial_investment_ratio': token_df['initial_investment_ratio'].iloc[0],
            'initial_market_cap': token_df['initial_market_cap'].iloc[0],
            'peak_market_cap': token_df['peak_market_cap'].iloc[0],
            'time_to_peak': true_peak
        }
        
        # Track state
        current_time = 0
        final_prediction = None
        
        # Base features to collect
        base_features = [
            'transaction_count', 'buy_pressure', 'volume', 'rsi', 
            'price_volatility', 'volume_volatility', 'momentum', 
            'trade_amount_variance', 'transaction_rate', 
            'trade_concentration', 'unique_wallets'
        ]
        
        # Simulate time progression in 5-second intervals
        while current_time <= min(true_peak + 60, 1020):  # Add buffer after true peak
            current_time += 5
            
            # Skip prediction during initial data collection
            if current_time <= self.initial_window:
                continue
            
            # Dynamically collect features for the current time window
            features_dict = global_features.copy()
            
            # Update available features
            for feature in base_features:
                # Find all columns for this feature with 0to prefix
                matching_cols = [
                    col for col in token_df.columns 
                    if col.startswith(f'{feature}_0to') and 
                    int(col.split('0to')[1].replace('s','')) <= current_time
                ]
                
                # Sort to get the most recent time window
                matching_cols.sort(
                    key=lambda x: int(x.split('0to')[1].replace('s','')), 
                    reverse=True
                )
                
                # Add the most recent time window feature
                if matching_cols:
                    features_dict[matching_cols[0]] = token_df[matching_cols[0]].iloc[0]
            
            print(f"\nNumber of features collected at time {current_time}: {len(features_dict)}")
            print("Collected feature columns:")
            for col in features_dict.keys():
                print(col)
            
            # Need minimum number of features before making prediction
            time_window_features = [
                col for col in features_dict.keys() 
                if '0to' in col
            ]
            
            if len(time_window_features) < 17 * 2:  # 4 granularities
                print(f"Not enough time window features: {len(time_window_features)}")
                continue
            
            # Prepare features and make prediction
            try:
                batch = self.prepare_features(features_dict, current_time, mint)
            except Exception as e:
                print(f"Error preparing features for {mint} at time {current_time}: {str(e)}")
                continue

            with torch.no_grad():
                try:
                    hazard_prob, time_pred, confidence = self.model(batch)
                    
                    # Convert logits to probabilities
                    hazard_score = torch.sigmoid(hazard_prob).item()
                    confidence_score = torch.sigmoid(confidence).item()
                    predicted_time = time_pred.item()
                    
                    print(f"Prediction details at time {current_time}:")
                    print(f"Hazard score: {hazard_score}")
                    print(f"Confidence score: {confidence_score}")
                    print(f"Predicted time: {predicted_time}")
                    
                    # Make final prediction if confident enough
                    if confidence_score > 0.8 or hazard_score > 0.7:
                        final_prediction = {
                            'mint': mint,
                            'predicted_time': predicted_time,
                            'true_time': true_peak,
                            'confidence': confidence_score,
                            'hazard_prob': hazard_score,
                            'prediction_made_at': current_time
                        }
                        break
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
            
            # Break if prediction time exceeds true peak
            if current_time > true_peak:
                break
        
        # Add results to tracking lists if final_prediction exists
        if final_prediction:
            self.predictions[mint] = final_prediction
            self.true_values.append(final_prediction['true_time'])
            self.predicted_values.append(final_prediction['predicted_time'])
            self.prediction_times.append(final_prediction['prediction_made_at'])
        else:
            print(f"No final prediction made for token: {mint}")
    
    def evaluate_dataset(self, test_df):
        """Evaluate entire test dataset"""
        print(f"Evaluating {len(test_df)} tokens...")
        
        for _, token_df in tqdm(test_df.groupby('mint')):
            self.evaluate_token(token_df)
            
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not self.true_values or not self.predicted_values:
            print("No predictions made. Cannot calculate metrics.")
            return None

        true_array = np.array(self.true_values)
        pred_array = np.array(self.predicted_values)
        time_array = np.array(self.prediction_times)
        
        metrics = {
            'mae': np.mean(np.abs(true_array - pred_array)),
            'rmse': np.sqrt(np.mean((true_array - pred_array) ** 2)),
            'r2': np.corrcoef(true_array, pred_array)[0,1] ** 2,
            'avg_prediction_time': np.mean(time_array),
            'early_predictions': sum(pred_array < true_array),
            'late_predictions': sum(pred_array > true_array),
            'total_predictions': len(self.predictions)
        }
        
        return metrics
    
    def plot_results(self):
        """Create scatter plot of predicted vs true peak times and save to file"""
        # Ensure no plotting if no predictions
        if not self.true_values or not self.predicted_values:
            print("No predictions to plot.")
            return
        
        plt.figure(figsize=(10, 10))
        plt.scatter(self.true_values, self.predicted_values, alpha=0.5)
        
        # Add diagonal line
        min_val = min(min(self.true_values), min(self.predicted_values))
        max_val = max(max(self.true_values), max(self.predicted_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('True Time to Peak (seconds)')
        plt.ylabel('Predicted Time to Peak (seconds)')
        plt.title('Time to Peak: Predicted vs True')
        
        # Calculate R² and add to plot
        r2 = self.calculate_metrics()['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # Create Visualizations directory in parent directory
        visualizations_dir = Path(__file__).parent.parent / 'Visualizations'
        visualizations_dir.mkdir(exist_ok=True)
        
        # Save plot
        plot_path = visualizations_dir / 'time_to_peak_scatter.png'
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free up memory
        
        print(f"Plot saved to {plot_path}")

def main():
    # Load test data
    test_df = pd.read_csv('data/time-data.csv')
    test_df = clean_dataset(test_df)
    
    # Initialize evaluator
    evaluator = RealTimeEvaluator(
        model_path='checkpoints/best_model.pt',
        initial_window=30
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(test_df)
    
    # Print metrics
    if metrics:
        print("\nEvaluation Metrics:")
        print(f"Mean Absolute Error: {metrics['mae']:.2f} seconds")
        print(f"RMSE: {metrics['rmse']:.2f} seconds")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Average Prediction Time: {metrics['avg_prediction_time']:.2f} seconds")
        print(f"Early Predictions: {metrics['early_predictions']}")
        print(f"Late Predictions: {metrics['late_predictions']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        
        # Plot results
        evaluator.plot_results()
    else:
        print("No metrics could be calculated.")

if __name__ == "__main__":
    main()