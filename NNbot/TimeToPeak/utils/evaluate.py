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
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Statistics tracking
        self.predictions = defaultdict(list)
        self.ground_truth = defaultdict(dict)
        self.prediction_times = defaultdict(list)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = RealTimePeakPredictor().to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def evaluate_token(self, token_df):
        """Evaluate a single token's data in simulated real-time"""
        dataset = MultiGranularTokenDataset(token_df, train=False, 
                                          initial_window=self.initial_window)
        
        # Store ground truth
        mint = token_df['mint'].iloc[0]
        self.ground_truth[mint] = {
            'time_to_peak': token_df['time_to_peak'].iloc[0],
            'peak_market_cap': token_df['peak_market_cap'].iloc[0]
        }
        
        # Simulate real-time data processing
        with torch.no_grad():
            for t in range(len(dataset)):
                sample = dataset[t]
                # Convert sample to batch
                batch = {k: v.unsqueeze(0).to(self.device) 
                        for k, v in sample.items()}
                
                # Only predict after initial window
                if batch['mask'].bool().item():
                    hazard_prob, time_pred, confidence = self.model(batch)
                    
                    self.predictions[mint].append({
                        'hazard_prob': hazard_prob.item(),
                        'time_pred': time_pred.item(),
                        'confidence': confidence.item(),
                        'current_time': t * 5  # Assuming 5s intervals
                    })
                    
                    # Store prediction timing
                    self.prediction_times[mint].append(t * 5)
    
    def evaluate_dataset(self, test_df):
        """Evaluate entire test dataset token by token"""
        for mint, token_df in tqdm(test_df.groupby('mint')):
            self.evaluate_token(token_df)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        metrics = {
            'mae': [],  # Mean Absolute Error
            'rmse': [], # Root Mean Squared Error
            'early_predictions': 0,  # Count of predictions before peak
            'late_predictions': 0,   # Count of predictions after peak
            'confidence_calibration': []  # Confidence vs Error correlation
        }
        
        for mint in self.ground_truth.keys():
            true_peak_time = self.ground_truth[mint]['time_to_peak']
            
            if not self.predictions[mint]:
                continue
                
            # Get the highest confidence prediction
            best_pred = max(self.predictions[mint], 
                          key=lambda x: x['confidence'])
            
            # Calculate errors
            pred_error = abs(best_pred['time_pred'] - true_peak_time)
            metrics['mae'].append(pred_error)
            metrics['rmse'].append(pred_error ** 2)
            
            # Timing classification
            if best_pred['time_pred'] < true_peak_time:
                metrics['early_predictions'] += 1
            else:
                metrics['late_predictions'] += 1
            
            # Confidence calibration
            metrics['confidence_calibration'].append(
                (best_pred['confidence'], pred_error)
            )
        
        # Finalize metrics
        metrics['mae'] = np.mean(metrics['mae'])
        metrics['rmse'] = np.sqrt(np.mean(metrics['rmse']))
        metrics['confidence_correlation'] = np.corrcoef(
            [x[0] for x in metrics['confidence_calibration']],
            [x[1] for x in metrics['confidence_calibration']]
        )[0,1]
        
        return metrics
    
    def plot_predictions(self, mint):
        """Plot predictions vs ground truth for a specific token"""
        if mint not in self.predictions:
            print(f"No predictions found for token {mint}")
            return
            
        true_peak = self.ground_truth[mint]['time_to_peak']
        preds = self.predictions[mint]
        
        times = [p['current_time'] for p in preds]
        time_preds = [p['time_pred'] for p in preds]
        confidences = [p['confidence'] for p in preds]
        hazards = [p['hazard_prob'] for p in preds]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Time predictions
        ax1.plot(times, time_preds, 'b-', label='Predicted Peak Time')
        ax1.axhline(y=true_peak, color='r', linestyle='--', 
                   label='True Peak Time')
        ax1.fill_between(times, 
                        [t-c*100 for t,c in zip(time_preds, confidences)],
                        [t+c*100 for t,c in zip(time_preds, confidences)],
                        alpha=0.2)
        ax1.set_ylabel('Time to Peak (s)')
        ax1.legend()
        
        # Hazard probabilities
        ax2.plot(times, hazards, 'g-', label='Peak Probability')
        ax2.plot(times, confidences, 'b--', label='Confidence')
        ax2.set_xlabel('Current Time (s)')
        ax2.set_ylabel('Probability')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Load test data
    test_df = pd.read_csv('data/test_data.csv')
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
    print(f"Confidence Correlation: {metrics['confidence_correlation']:.3f}")
    
    # Plot some example predictions
    for mint in list(evaluator.predictions.keys())[:3]:
        print(f"\nPlotting predictions for token {mint}")
        evaluator.plot_predictions(mint)

if __name__ == "__main__":
    main()