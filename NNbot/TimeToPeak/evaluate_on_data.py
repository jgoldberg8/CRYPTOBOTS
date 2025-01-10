"""evaluate.py - Main evaluation script"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from TimeToPeak.utils.clean_dataset import clean_dataset
from TimeToPeak.datasets.time_token_dataset import create_multi_granular_loaders
from TimeToPeak.models.time_to_peak_model import MultiGranularPeakPredictor


def evaluate_model_on_data(model_path, data_paths, batch_size=64):
    """
    Evaluate model on given dataset(s).
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        df = clean_dataset(df)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create data loaders and get scalers
    train_loader, test_loader = create_multi_granular_loaders(
        train_df,
        test_df,
        batch_size=batch_size
    )
    
    # Get scalers from training dataset
    scalers = train_loader.dataset.get_scalers()
    target_scaler = scalers['target']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model first
    model = MultiGranularPeakPredictor(
        input_size=11,
        hidden_size=256
    ).to(device)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    predictions = []
    true_values = []
    
    print("\nPerforming evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mean, _ = model(batch)
            
            # Store predictions and targets
            predictions.append(mean.cpu().numpy())
            true_values.append(batch['targets'].cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)
    
    # Inverse transform predictions and true values
    predictions_time = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_values_time = target_scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    
    # Print ranges for debugging
    print("\nValue ranges:")
    print(f"Raw predictions: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Raw true values: [{true_values.min():.4f}, {true_values.max():.4f}]")
    print(f"Inverse transformed predictions: [{predictions_time.min():.4f}, {predictions_time.max():.4f}]")
    print(f"Inverse transformed true values: [{true_values_time.min():.4f}, {true_values_time.max():.4f}]")
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(true_values_time, predictions_time),
        'mse': mean_squared_error(true_values_time, predictions_time),
        'rmse': np.sqrt(mean_squared_error(true_values_time, predictions_time)),
        'r2': r2_score(true_values_time, predictions_time)
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values_time, predictions_time, alpha=0.5, s=20)
    
    max_val = max(true_values_time.max(), predictions_time.max())
    min_val = min(true_values_time.min(), predictions_time.min())
    padding = (max_val - min_val) * 0.05
    lims = [min_val - padding, max_val + padding]
    
    plt.plot(lims, lims, 'r--', lw=2)
    plt.title('Time to Peak: Predicted vs True')
    plt.xlabel('True Values (seconds)')
    plt.ylabel('Predicted Values (seconds)')
    plt.text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'time_to_peak_prediction_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print evaluation results
    print("\n=== Time to Peak Model Performance Evaluation ===")
    print(f"Mean Absolute Error: {metrics['mae']:.4f} seconds")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f} seconds")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': predictions_time,
        'true_values': true_values_time,
    }


if __name__ == "__main__":
    results = evaluate_model_on_data(
        model_path='checkpoints/best_model.pt',
        data_paths=['data/time-data.csv']
    )