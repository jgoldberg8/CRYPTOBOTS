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
        
        if 'scaler' in self.checkpoint:
            self.scaler = self.checkpoint['scaler']
        else:
            print("Scaler not found in checkpoint, creating temporary training dataset...")
            temp_train_dataset = MultiGranularTokenDataset(
                test_df.head(100), 
                scaler=None, 
                train=True
            )
            self.scaler = temp_train_dataset.scalers
        
        self.model.eval()
        self.true_values = []
        self.predicted_values = []
    
    def _load_model(self):
        model = RealTimePeakPredictor().to(self.device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        return model
    
    def evaluate_token(self, token_df):
        """Evaluate a single token's data and make one prediction"""
        dataset = MultiGranularTokenDataset(token_df, train=False, 
                                          scaler=self.scaler,
                                          initial_window=self.initial_window)
        
        mint = token_df['mint'].iloc[0]
        true_time = token_df['time_to_peak'].iloc[0]
        
        # Make single prediction after initial window
        with torch.no_grad():
            sample = dataset[self.initial_window]
            batch = {k: v.unsqueeze(0).to(self.device) 
                    for k, v in sample.items()}
            
            hazard_prob, time_pred, confidence = self.model(batch)
            
            # Store true and predicted values for plotting
            self.true_values.append(true_time)
            self.predicted_values.append(time_pred.item())
    
    def evaluate_dataset(self, test_df):
        """Evaluate entire test dataset token by token"""
        self.true_values = []
        self.predicted_values = []
        
        for mint, token_df in tqdm(test_df.groupby('mint')):
            self.evaluate_token(token_df)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        true_array = np.array(self.true_values)
        pred_array = np.array(self.predicted_values)
        
        metrics = {
            'mae': np.mean(np.abs(true_array - pred_array)),
            'rmse': np.sqrt(np.mean((true_array - pred_array) ** 2)),
            'r2': np.corrcoef(true_array, pred_array)[0,1] ** 2
        }
        
        return metrics
    
    def plot_results(self):
        """Create scatter plot of predicted vs true values"""
        plt.figure(figsize=(12, 8))
        plt.scatter(self.true_values, self.predicted_values, alpha=0.5, color='lightblue')
        
        # Add diagonal line
        min_val = min(min(self.true_values), min(self.predicted_values))
        max_val = max(max(self.true_values), max(self.predicted_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('True Values (time in seconds)')
        plt.ylabel('Predicted Values (time in seconds)')
        plt.title('Time to Peak: Predicted vs True')
        plt.text(0.05, 0.95, f'R² = {self.calculate_metrics()["r2"]:.4f}', 
                transform=plt.gca().transAxes)
        
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
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Plot results
    evaluator.plot_results()

if __name__ == "__main__":
    main()