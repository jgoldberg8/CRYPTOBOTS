import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler

from model_utilities import add_data_quality_features
from peak_market_cap_model import PeakMarketCapPredictor
from time_to_peak_model import TimeToPeakPredictor
from token_dataset import TokenDataset
from train_models import clean_dataset

def evaluate_model_both(peak_market_cap_model_path, time_to_peak_model_path, data_paths):
    """
    Evaluate both peak market cap and time to peak models.
    
    Args:
        peak_market_cap_model_path (str): Path to saved peak market cap model
        time_to_peak_model_path (str): Path to saved time to peak model
        data_paths (list): List of paths to data CSV files
    
    Returns:
        dict: Dictionary containing evaluation results and model artifacts
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        df = clean_dataset(df)
        df = add_data_quality_features(df)
        dfs.append(df)

    # Concatenate dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Split data with stratification
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    test_dataset = TokenDataset(test_df, scaler={
        'global': train_dataset.global_scaler, 
        'target': train_dataset.target_scaler
    })

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(
        input_size=input_size,
        hidden_size=256,
        num_layers=4,
        dropout_rate=0.5
    ).to(device)
    
    time_to_peak_model = TimeToPeakPredictor(
        input_size=input_size,
        hidden_size=256,
        num_layers=3,
        dropout_rate=0.5
    ).to(device)

    # Load saved models
    peak_market_cap_checkpoint = torch.load(peak_market_cap_model_path, map_location=device)
    time_to_peak_checkpoint = torch.load(time_to_peak_model_path, map_location=device)
    
    peak_market_cap_model.load_state_dict(peak_market_cap_checkpoint['model_state_dict'])
    time_to_peak_model.load_state_dict(time_to_peak_checkpoint['model_state_dict'])

    # Set models to evaluation mode
    peak_market_cap_model.eval()
    time_to_peak_model.eval()

    # Use larger batch size for evaluation
    batch_size = 64
    all_predictions = []
    all_true_values = []

    # Create DataLoader for test dataset
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Perform predictions
    with torch.no_grad():
        for batch in test_loader:
            # Move all inputs to device
            x_5s = batch['x_5s'].to(device)
            x_10s = batch['x_10s'].to(device)
            x_20s = batch['x_20s'].to(device)
            x_30s = batch['x_30s'].to(device)
            global_features = batch['global_features'].to(device)
            quality_features = batch['quality_features'].to(device)
            
            # Get predictions
            peak_market_cap_pred = peak_market_cap_model(
                x_5s, x_10s, x_20s, x_30s, global_features, quality_features
            )
            time_to_peak_pred = time_to_peak_model(
                x_5s, x_10s, x_20s, x_30s, global_features, quality_features
            )

            # Combine predictions
            combined_pred = torch.cat([peak_market_cap_pred, time_to_peak_pred], dim=1)
            
            # Store predictions and true values
            all_predictions.append(combined_pred.cpu())
            all_true_values.append(batch['targets'].cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0).numpy()
    true_values = torch.cat(all_true_values, dim=0).numpy()

    # Inverse transform using the original scaler
    target_scaler = test_dataset.target_scaler
    predictions = target_scaler.inverse_transform(predictions)
    true_values = target_scaler.inverse_transform(true_values)

    # Split predictions and true values
    peak_market_cap_predictions = predictions[:, 0]
    time_to_peak_predictions = predictions[:, 1]
    true_peak_market_cap = true_values[:, 0]
    true_time_to_peak = true_values[:, 1]

    # Calculate metrics
    metrics = {
        'peak_market_cap': {
            'mae': mean_absolute_error(true_peak_market_cap, peak_market_cap_predictions),
            'mse': mean_squared_error(true_peak_market_cap, peak_market_cap_predictions),
            'rmse': np.sqrt(mean_squared_error(true_peak_market_cap, peak_market_cap_predictions)),
            'r2': r2_score(true_peak_market_cap, peak_market_cap_predictions)
        },
        'time_to_peak': {
            'mae': mean_absolute_error(true_time_to_peak, time_to_peak_predictions),
            'mse': mean_squared_error(true_time_to_peak, time_to_peak_predictions),
            'rmse': np.sqrt(mean_squared_error(true_time_to_peak, time_to_peak_predictions)),
            'r2': r2_score(true_time_to_peak, time_to_peak_predictions)
        }
    }

    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Peak Market Cap plot
    plt.subplot(2, 2, 1)
    plt.scatter(true_peak_market_cap, peak_market_cap_predictions, alpha=0.5, s=20)
    
    # Calculate proper limits for peak market cap
    max_cap = max(true_peak_market_cap.max(), peak_market_cap_predictions.max())
    min_cap = min(true_peak_market_cap.min(), peak_market_cap_predictions.min())
    padding_cap = (max_cap - min_cap) * 0.05
    lims_cap = [min_cap - padding_cap, max_cap + padding_cap]
    
    plt.plot(lims_cap, lims_cap, 'r--', lw=2)
    plt.title('Peak Market Cap: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))  # Scientific notation
    
    # Add R² value to plot
    plt.text(0.05, 0.95, f'R² = {metrics["peak_market_cap"]["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    # Time to Peak plot
    plt.subplot(2, 2, 2)
    plt.scatter(true_time_to_peak, time_to_peak_predictions, alpha=0.5, s=20)
    
    # Calculate proper limits for time to peak
    max_time = max(true_time_to_peak.max(), time_to_peak_predictions.max())
    min_time = min(true_time_to_peak.min(), time_to_peak_predictions.min())
    padding_time = (max_time - min_time) * 0.05
    lims_time = [min_time - padding_time, max_time + padding_time]
    
    plt.plot(lims_time, lims_time, 'r--', lw=2)
    plt.title('Time to Peak: Predicted vs True')
    plt.xlabel('True Values (hours)')
    plt.ylabel('Predicted Values (hours)')
    
    # Add R² value to plot
    plt.text(0.05, 0.95, f'R² = {metrics["time_to_peak"]["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    # Combined plot showing error relationships
    plt.subplot(2, 1, 2)
    
    # Define error clipping function
    def clip_errors(errors, clip_value=100):
        return np.clip(errors, -clip_value, clip_value)

    # Calculate absolute errors
    peak_cap_abs_errors = peak_market_cap_predictions - true_peak_market_cap
    time_abs_errors = time_to_peak_predictions - true_time_to_peak
    
    # Scale the errors to make them comparable
    peak_cap_scaled = peak_cap_abs_errors / np.std(peak_cap_abs_errors)
    time_scaled = time_abs_errors / np.std(time_abs_errors)
    
    # Clip extreme values for better visualization
    peak_cap_scaled = clip_errors(peak_cap_scaled, 5)
    time_scaled = clip_errors(time_scaled, 5)
    
    # Create scatter plot of scaled errors
    scatter = plt.scatter(peak_cap_scaled, time_scaled,
                         alpha=0.5, s=20, c=true_peak_market_cap,
                         cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='True Peak Market Cap')
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0,0))
    cbar.update_ticks()
    
    # Add reference lines
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    # Add labels
    plt.title('Joint Model Error Analysis (Standardized Errors)')
    plt.xlabel('Peak Market Cap Error (σ units)')
    plt.ylabel('Time to Peak Error (σ units)')
    
    # Add quadrant labels
    plt.text(0.95, 0.95, 'Both Over-predicted', 
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, alpha=0.7)
    plt.text(0.95, 0.05, 'Over-predicted Cap\nUnder-predicted Time', 
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, alpha=0.7)
    plt.text(0.05, 0.95, 'Under-predicted Cap\nOver-predicted Time', 
             horizontalalignment='left', verticalalignment='top',
             transform=plt.gca().transAxes, alpha=0.7)
    plt.text(0.05, 0.05, 'Both Under-predicted', 
             horizontalalignment='left', verticalalignment='bottom',
             transform=plt.gca().transAxes, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print evaluation results
    print("\n=== Model Performance Evaluation ===")
    
    print("\nPeak Market Cap Metrics:")
    print(f"Mean Absolute Error: {metrics['peak_market_cap']['mae']:.4f}")
    print(f"Mean Squared Error: {metrics['peak_market_cap']['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['peak_market_cap']['rmse']:.4f}")
    print(f"R² Score: {metrics['peak_market_cap']['r2']:.4f}")

    print("\nTime to Peak Metrics:")
    print(f"Mean Absolute Error: {metrics['time_to_peak']['mae']:.4f}")
    print(f"Mean Squared Error: {metrics['time_to_peak']['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['time_to_peak']['rmse']:.4f}")
    print(f"R² Score: {metrics['time_to_peak']['r2']:.4f}")

    return {
        'metrics': metrics,
        'predictions': {
            'peak_market_cap': peak_market_cap_predictions,
            'time_to_peak': time_to_peak_predictions,
            'true_peak_market_cap': true_peak_market_cap,
            'true_time_to_peak': true_time_to_peak
        },
        'datasets': {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        },
        'dataframes': {
            'train_df': train_df,
            'test_df': test_df
        },
        'models': {
            'peak_market_cap': peak_market_cap_model,
            'time_to_peak': time_to_peak_model
        }
    }

# Usage
if __name__ == "__main__":
    evaluate_model_both(
        'best_peak_market_cap_model.pth', 
        'best_time_to_peak_model.pth', 
        ['data/testData.csv']#['data/token_data_2025-01-07.csv', 'data/token_data_2025-01-08.csv']
    )