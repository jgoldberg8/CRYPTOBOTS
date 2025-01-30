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
from PeakMarketCap.models.token_dataset import TokenDataset
from torch.utils.data import DataLoader


def evaluate_peak_market_cap_model(peak_market_cap_model_path, data_paths):
    """
    Evaluate percent increase prediction model.
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
    peak_market_cap_model = PeakMarketCapPredictor(input_size=input_size, hidden_size=1024, num_layers=4, dropout_rate=0.4).to(device)

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
            
            percent_increase_pred = peak_market_cap_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)
            print("Model output range before expm1:", 
                  percent_increase_pred.min().item(), 
                  percent_increase_pred.max().item())
            print("Target range before expm1:", 
                  batch['targets'].min().item(), 
                  batch['targets'].max().item())
            # Convert from log space back to original values
            percent_increase_pred = torch.expm1(percent_increase_pred)
            targets = torch.expm1(batch['targets'][:, 0].unsqueeze(1))
            print("Range after expm1:", 
                  percent_increase_pred.min().item(), 
                  percent_increase_pred.max().item())
            
            all_predictions.append(percent_increase_pred.cpu())
            all_true_values.append(targets.cpu())

    predictions = torch.cat(all_predictions, dim=0).numpy()
    true_values = torch.cat(all_true_values, dim=0).numpy()

    # Use the values directly
    percent_increase_predictions = predictions.squeeze()
    true_percent_increase = true_values.squeeze()

    # Calculate metrics on actual values
    metrics = {
        'percent_increase': {
            'mae': mean_absolute_error(true_percent_increase, percent_increase_predictions),
            'mse': mean_squared_error(true_percent_increase, percent_increase_predictions),
            'rmse': np.sqrt(mean_squared_error(true_percent_increase, percent_increase_predictions)),
            'r2': r2_score(true_percent_increase, percent_increase_predictions)
        }
    }

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(true_percent_increase, percent_increase_predictions, alpha=0.5, s=20)

    max_val = max(true_percent_increase.max(), percent_increase_predictions.max())
    min_val = min(true_percent_increase.min(), percent_increase_predictions.min())
    padding = (max_val - min_val) * 0.05
    lims = [min_val - padding, max_val + padding]

    plt.plot(lims, lims, 'r--', lw=2)
    plt.title('Percent Increase: Predicted vs True')
    plt.xlabel('True Values (%)')
    plt.ylabel('Predicted Values (%)')
    plt.text(0.05, 0.95, f'R² = {metrics["percent_increase"]["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'percent_increase_prediction_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print evaluation results
    print("\n=== Percent Increase Model Performance Evaluation ===")
    print(f"Mean Absolute Error: {metrics['percent_increase']['mae']:.4f}%")
    print(f"Mean Squared Error: {metrics['percent_increase']['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['percent_increase']['rmse']:.4f}%")
    print(f"R² Score: {metrics['percent_increase']['r2']:.4f}")

    return {
        'metrics': metrics,
        'predictions': {
            'percent_increase_predictions': percent_increase_predictions,
            'true_percent_increase': true_percent_increase
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

if __name__ == "__main__":
    evaluate_peak_market_cap_model(
        'best_peak_market_cap_model.pth',
        ['data/new-token-data.csv']
    )