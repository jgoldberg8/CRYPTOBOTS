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
        
    def prepare_features(self, features_dict, current_time):
        """Prepare features in format expected by model for a given timestamp"""
        # Create dataset with single sample
        dataset = MultiGranularTokenDataset(
            pd.DataFrame([features_dict]), 
            scaler=self.scaler,
            train=False
        )
        
        # Get sample in tensor format
        sample = dataset[0]
        
        # Move to device and add batch dimension
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in sample.items()}
        return batch
        
    def evaluate_token(self, token_df):
        """Simulate real-time evaluation of a single token"""
        mint = token_df['mint'].iloc[0]
        true_peak = token_df['time_to_peak'].iloc[0]
        
        # Track state
        current_time = 0
        final_prediction = None
        features_dict = {}
        
        # Get valid feature columns (excluding metadata columns)
        time_window_cols = [col for col in token_df.columns 
                        if '_to' in col and col.split('_')[0] in [
                            'transaction_count', 'buy_pressure', 'volume',
                            'rsi', 'price_volatility', 'volume_volatility',
                            'momentum', 'trade_amount_variance', 'transaction_rate',
                            'trade_concentration', 'unique_wallets'
                        ]]
        
        # Simulate time progression in 5-second intervals
        while current_time <= min(true_peak + 60, 1020):  # Add buffer after true peak
            current_time += 5
            
            # Skip prediction during initial data collection
            if current_time <= self.initial_window:
                continue
                    
            # Update available features
            for col in time_window_cols:
                feature_base, time_range = col.rsplit('_', 1)
                if not time_range.endswith('s'):
                    continue
                    
                try:
                    start_time, end_time = map(int, time_range.replace('s','').split('to'))
                    # Only include features for elapsed time windows
                    if end_time <= current_time:
                        features_dict[col] = token_df[col].iloc[0]
                except ValueError:
                    continue
            
            # Need minimum number of features before making prediction
            if len(features_dict) < 17 * 2:  # 4 granularities
                continue
                
            # Prepare features and make prediction
            try:
                batch = self.prepare_features(features_dict, current_time)
            except Exception as e:
                print(f"Error preparing features for {mint} at time {current_time}: {str(e)}")
                continue

            with torch.no_grad():
                hazard_prob, time_pred, confidence = self.model(batch)
                
                # Convert logits to probabilities
                hazard_score = torch.sigmoid(hazard_prob).item()
                confidence_score = torch.sigmoid(confidence).item()
                predicted_time = time_pred.item()
                
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
    
        # Add results to tracking lists if final_prediction exists
        if final_prediction:
            self.predictions[mint] = final_prediction
            self.true_values.append(final_prediction['true_time'])
            self.predicted_values.append(final_prediction['predicted_time'])
            self.prediction_times.append(final_prediction['prediction_made_at'])
    
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