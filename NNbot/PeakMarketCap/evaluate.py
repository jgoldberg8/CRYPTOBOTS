import sys
import os

from PeakMarketCap.models.model_utilities import clean_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from PeakMarketCap.models.peak_market_cap_model import PeakMarketCapPredictor
from utils.add_data_quality_features import add_data_quality_features
from token_dataset import TokenDataset
from torch.utils.data import DataLoader


def evaluate_peak_market_cap_model(peak_market_cap_model_path, data_paths):
    """
    Evaluate peak market cap model.
    
    Args:
        peak_market_cap_model_path (str): Path to saved peak market cap model
        data_paths (list): List of paths to data CSV files
    
    Returns:
        dict: Evaluation metrics, predictions, and additional model information
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        df = clean_dataset(df)
        df = add_data_quality_features(df)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    test_dataset = TokenDataset(test_df, scaler={
        'global': train_dataset.global_scaler, 
        'target': train_dataset.target_scaler
    })

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(input_size=input_size,hidden_size=256,num_layers=3, dropout_rate=0.39683333144243493).to(device)

    # Load saved model
    peak_market_cap_checkpoint = torch.load(peak_market_cap_model_path, map_location=device, weights_only=True)
    peak_market_cap_model.load_state_dict(peak_market_cap_checkpoint['model_state_dict'])
    peak_market_cap_model.eval()

    # Perform predictions
    batch_size = 64
    all_predictions = []
    all_true_values = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            x_5s = batch['x_5s'].to(device)
            x_10s = batch['x_10s'].to(device)
            x_20s = batch['x_20s'].to(device)
            x_30s = batch['x_30s'].to(device)
            global_features = batch['global_features'].to(device)
            quality_features = batch['quality_features'].to(device)
            
            peak_market_cap_pred = peak_market_cap_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)
            
            all_predictions.append(peak_market_cap_pred.cpu())
            all_true_values.append(batch['targets'][:, 0].cpu().unsqueeze(1))

    predictions = torch.cat(all_predictions, dim=0).numpy()
    true_values = torch.cat(all_true_values, dim=0).numpy()

    # Create a dummy array with zeros for the second column to match scaler's expected shape
    dummy_predictions = np.zeros((predictions.shape[0], 2))
    dummy_predictions[:, 0] = predictions.squeeze()
    
    dummy_true_values = np.zeros((true_values.shape[0], 2))
    dummy_true_values[:, 0] = true_values.squeeze()

    # Inverse transform using the scaler
    target_scaler = test_dataset.target_scaler
    dummy_predictions = target_scaler.inverse_transform(dummy_predictions)
    dummy_true_values = target_scaler.inverse_transform(dummy_true_values)

    # Exponentiate because of log1p transformation in clean_dataset
    peak_market_cap_predictions = np.expm1(dummy_predictions[:, 0])
    true_peak_market_cap = np.expm1(dummy_true_values[:, 0])

    # Calculate metrics on actual values
    metrics = {
        'peak_market_cap': {
            'mae': mean_absolute_error(true_peak_market_cap, peak_market_cap_predictions),
            'mse': mean_squared_error(true_peak_market_cap, peak_market_cap_predictions),
            'rmse': np.sqrt(mean_squared_error(true_peak_market_cap, peak_market_cap_predictions)),
            'r2': r2_score(true_peak_market_cap, peak_market_cap_predictions)
        }
    }

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(true_peak_market_cap, peak_market_cap_predictions, alpha=0.5, s=20)

    max_cap = max(true_peak_market_cap.max(), peak_market_cap_predictions.max())
    min_cap = min(true_peak_market_cap.min(), peak_market_cap_predictions.min())
    padding_cap = (max_cap - min_cap) * 0.05
    lims_cap = [min_cap - padding_cap, max_cap + padding_cap]

    plt.plot(lims_cap, lims_cap, 'r--', lw=2)
    plt.title('Peak Market Cap: Predicted vs True')
    plt.xlabel('True Values (SOL)')
    plt.ylabel('Predicted Values (SOL)')
    plt.text(0.05, 0.95, f'R² = {metrics["peak_market_cap"]["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a 'Visualizations' subdirectory if it doesn't exist
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    print(visualizations_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'peak_market_cap_prediction_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Print evaluation results
    print("\n=== Peak Market Cap Model Performance Evaluation ===")
    
    print("\nPeak Market Cap Metrics:")
    print(f"Mean Absolute Error: {metrics['peak_market_cap']['mae']:.4f} SOL")
    print(f"Mean Squared Error: {metrics['peak_market_cap']['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['peak_market_cap']['rmse']:.4f} SOL")
    print(f"R² Score: {metrics['peak_market_cap']['r2']:.4f}")

    return {
        'metrics': metrics,
        'predictions': {
            'peak_market_cap_predictions': peak_market_cap_predictions,
            'true_peak_market_cap': true_peak_market_cap
        },
        'datasets': {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        },
        'dataframes': {
            'train_df': train_df,
            'test_df': test_df
        },
        'model': peak_market_cap_model
    }







# Usage
if __name__ == "__main__":
    evaluate_peak_market_cap_model(
         'best_peak_market_cap_model.pth',
       ['data/higher-peak-data.csv']
    )