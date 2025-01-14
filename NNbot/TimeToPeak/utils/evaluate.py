import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
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
        
        # Load scaler from checkpoint or create a temporary training dataset to get scaler
        if 'scaler' in self.checkpoint:
            self.scaler = self.checkpoint['scaler']
        else:
            # Create a temporary training dataset to get the scaler
            print("Scaler not found in checkpoint, creating temporary training dataset...")
            # Use a small subset of test data to fit scaler
            temp_train_dataset = MultiGranularTokenDataset(
                test_df.head(100), 
                scaler=None, 
                train=True
            )
            self.scaler = temp_train_dataset.scalers
        
        self.model.eval()
        
        # Statistics tracking
        self.predictions = defaultdict(list)
        self.ground_truth = defaultdict(dict)
        self.prediction_times = defaultdict(list)
    
    def _load_model(self):
        model = RealTimePeakPredictor().to(self.device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        return model
    
    def evaluate_token(self, token_df):
        """Evaluate a single token's data in simulated real-time"""
        dataset = MultiGranularTokenDataset(token_df, train=False, 
                                            scaler=self.scaler,  # Pass the scaler
                                            initial_window=self.initial_window)
        
        mint = token_df['mint'].iloc[0]
        self.ground_truth[mint] = {
            'time_to_peak': token_df['time_to_peak'].iloc[0],
            'peak_market_cap': token_df['peak_market_cap'].iloc[0]
        }
        
        final_prediction = None
        prediction_made_at = None
        
        # Simulate real-time data processing
        with torch.no_grad():
            for t in range(len(dataset)):
                sample = dataset[t]
                batch = {k: v.unsqueeze(0).to(self.device) 
                        for k, v in sample.items()}
                
                if batch['mask'].bool().item():
                    hazard_prob, time_pred, confidence = self.model(batch)
                    current_time = t * 5
                    
                    # Decision logic: Make final prediction when confidence exceeds threshold
                    # or hazard probability indicates imminent peak
                    confidence_score = confidence.sigmoid().item()  # Convert logit to probability
                    hazard_score = hazard_prob.sigmoid().item()    # Convert logit to probability
                    
                    if confidence_score > 0.8 or hazard_score > 0.7:
                        if final_prediction is None:  # Only store first confident prediction
                            final_prediction = {
                                'hazard_prob': hazard_score,
                                'time_pred': time_pred.item(),
                                'confidence': confidence_score,
                                'current_time': current_time
                            }
                            prediction_made_at = current_time
                            break  # Stop processing future data after making prediction
            
            # If we never reached confidence threshold, use last prediction
            if final_prediction is None and len(dataset) > 0:
                hazard_prob, time_pred, confidence = self.model(batch)
                final_prediction = {
                    'hazard_prob': hazard_prob.sigmoid().item(),
                    'time_pred': time_pred.item(),
                    'confidence': confidence.sigmoid().item(),
                    'current_time': (len(dataset) - 1) * 5
                }
                prediction_made_at = (len(dataset) - 1) * 5
        
        if final_prediction:
            self.predictions[mint] = [final_prediction]
            self.prediction_times[mint] = [prediction_made_at]
    
    def evaluate_dataset(self, test_df):
        """Evaluate entire test dataset token by token"""
        for mint, token_df in tqdm(test_df.groupby('mint')):
            self.evaluate_token(token_df)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        metrics = {
            'mae': [],  # Mean Absolute Error for time predictions
            'rmse': [], # Root Mean Squared Error for time predictions
            'early_predictions': 0,  # Count of predictions before peak
            'late_predictions': 0,   # Count of predictions after peak
            'prediction_times': [],  # When predictions were made
            'confidence_calibration': [],  # Confidence vs Error correlation
            'avg_prediction_time': None,  # Average time predictions were made
            'prediction_time_std': None   # Standard deviation of prediction times
        }
        
        for mint in self.ground_truth.keys():
            true_peak_time = self.ground_truth[mint]['time_to_peak']
            
            if not self.predictions[mint]:
                continue
            
            # Get the final prediction (there should only be one)
            pred = self.predictions[mint][0]
            pred_time = self.prediction_times[mint][0]
            
            # Calculate prediction timing metrics
            metrics['prediction_times'].append(pred_time)
            
            # Calculate errors
            pred_error = abs(pred['time_pred'] - true_peak_time)
            metrics['mae'].append(pred_error)
            metrics['rmse'].append(pred_error ** 2)
            
            # Timing classification
            if pred['time_pred'] < true_peak_time:
                metrics['early_predictions'] += 1
            else:
                metrics['late_predictions'] += 1
            
            # Confidence calibration
            metrics['confidence_calibration'].append(
                (pred['confidence'], pred_error)
            )
        
        # Finalize metrics
        metrics['mae'] = np.mean(metrics['mae']) if metrics['mae'] else float('nan')
        metrics['rmse'] = np.sqrt(np.mean(metrics['rmse'])) if metrics['rmse'] else float('nan')
        metrics['avg_prediction_time'] = np.mean(metrics['prediction_times']) if metrics['prediction_times'] else float('nan')
        metrics['prediction_time_std'] = np.std(metrics['prediction_times']) if metrics['prediction_times'] else float('nan')
        
        if len(metrics['confidence_calibration']) > 1:
            metrics['confidence_correlation'] = np.corrcoef(
                [x[0] for x in metrics['confidence_calibration']],
                [x[1] for x in metrics['confidence_calibration']]
            )[0,1]
        else:
            metrics['confidence_correlation'] = float('nan')
        
        return metrics
    
    def plot_predictions(self, mint):
        """Plot predictions vs ground truth for a specific token"""
        if mint not in self.predictions:
            print(f"No predictions found for token {mint}")
            return
            
        true_peak = self.ground_truth[mint]['time_to_peak']
        pred = self.predictions[mint][0]  # Get the single prediction
        pred_time = self.prediction_times[mint][0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time predictions
        ax1.axhline(y=true_peak, color='r', linestyle='--', 
                   label='True Peak Time')
        ax1.axvline(x=pred_time, color='g', linestyle=':', 
                   label='Prediction Made')
        ax1.scatter(pred_time, pred['time_pred'], color='b', s=100,
                   label='Prediction')
        
        # Add confidence interval
        confidence_interval = 100 * pred['confidence']  # Scale for visibility
        ax1.fill_between([pred_time], 
                        [pred['time_pred'] - confidence_interval],
                        [pred['time_pred'] + confidence_interval],
                        alpha=0.2, color='b')
        
        ax1.set_ylabel('Time to Peak (s)')
        ax1.set_title(f'Prediction for Token {mint}')
        ax1.legend()
        
        # Hazard probability and confidence
        ax2.scatter(pred_time, pred['hazard_prob'], color='g',
                   label='Peak Probability')
        ax2.scatter(pred_time, pred['confidence'], color='b',
                   label='Confidence')
        ax2.axvline(x=pred_time, color='g', linestyle=':')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Probability')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

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
    print("\nEvaluation Metrics:")
    print(f"Mean Absolute Error: {metrics['mae']:.2f} seconds")
    print(f"RMSE: {metrics['rmse']:.2f} seconds")
    print(f"Early Predictions: {metrics['early_predictions']}")
    print(f"Late Predictions: {metrics['late_predictions']}")
    print(f"Average Prediction Time: {metrics['avg_prediction_time']:.2f} seconds")
    print(f"Prediction Time Std: {metrics['prediction_time_std']:.2f} seconds")
    print(f"Confidence Correlation: {metrics['confidence_correlation']:.3f}")
    
    # Plot some example predictions
    for mint in list(evaluator.predictions.keys())[:3]:
        print(f"\nPlotting predictions for token {mint}")
        evaluator.plot_predictions(mint)

if __name__ == "__main__":
    main()