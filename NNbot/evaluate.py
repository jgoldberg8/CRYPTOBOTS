import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from model_utilities import TokenDataset, add_data_quality_features
from peak_market_cap_model import PeakMarketCapPredictor
from time_to_peak_model import TimeToPeakPredictor
from train_models import clean_dataset

# Import the previously defined classes (TokenDataset and TokenPredictor)




def evaluate_model_both(peak_market_cap_model_path, time_to_peak_model_path, data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = clean_dataset(df)
    df = add_data_quality_features(df)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    test_dataset = TokenDataset(test_df, scaler=train_dataset.scaler)

    # Initialize models
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(input_size, hidden_size=256, num_layers=3, dropout_rate=0.5)
    time_to_peak_model = TimeToPeakPredictor(input_size, hidden_size=256, num_layers=3, dropout_rate=0.5)

    # Load best saved models
    peak_market_cap_checkpoint = torch.load(peak_market_cap_model_path)
    time_to_peak_checkpoint = torch.load(time_to_peak_model_path)
    
    # Extract model state dictionaries from checkpoints
    peak_market_cap_model.load_state_dict(peak_market_cap_checkpoint['model_state_dict'])
    time_to_peak_model.load_state_dict(time_to_peak_checkpoint['model_state_dict'])

    # Set to evaluation mode
    peak_market_cap_model.eval()
    time_to_peak_model.eval()

    # Prepare for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peak_market_cap_model = peak_market_cap_model.to(device)
    time_to_peak_model = time_to_peak_model.to(device)

    # Create a small batch for predictions
    batch_size = 4
    all_predictions = []
    all_true_values = []

    # Disable gradient calculation
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(test_dataset)))
            batch_samples = [test_dataset[j] for j in batch_indices]
            
            # Prepare batch data
            x_5s = torch.stack([s['x_5s'] for s in batch_samples]).to(device)
            x_10s = torch.stack([s['x_10s'] for s in batch_samples]).to(device)
            x_20s = torch.stack([s['x_20s'] for s in batch_samples]).to(device)
            x_30s = torch.stack([s['x_30s'] for s in batch_samples]).to(device)
            global_features = torch.stack([s['global_features'] for s in batch_samples]).to(device)
            quality_features = torch.stack([s['quality_features'] for s in batch_samples]).to(device)
            
            # Make predictions
            peak_market_cap_prediction = peak_market_cap_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)
            time_to_peak_prediction = time_to_peak_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)

            # Combine predictions for inverse transform
            combined_pred = np.column_stack([
                peak_market_cap_prediction.cpu().numpy(),
                time_to_peak_prediction.cpu().numpy()
            ])

            # Store predictions and true values
            all_predictions.extend(combined_pred)
            all_true_values.extend([s['targets'].numpy() for s in batch_samples])

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_values = np.array(all_true_values)

    # Inverse transform all at once
    predictions_original = test_dataset.scaler.inverse_transform(predictions)
    true_values_original = test_dataset.scaler.inverse_transform(true_values)

    # Split predictions back into individual components
    peak_market_cap_predictions = predictions_original[:, 0]
    time_to_peak_predictions = predictions_original[:, 1]
    true_peak_market_cap = true_values_original[:, 0]
    true_time_to_peak = true_values_original[:, 1]

    # Calculate performance metrics
    peak_market_cap_mae = mean_absolute_error(true_peak_market_cap, peak_market_cap_predictions)
    time_to_peak_mae = mean_absolute_error(true_time_to_peak, time_to_peak_predictions)

    peak_market_cap_mse = mean_squared_error(true_peak_market_cap, peak_market_cap_predictions)
    time_to_peak_mse = mean_squared_error(true_time_to_peak, time_to_peak_predictions)

    peak_market_cap_r2 = r2_score(true_peak_market_cap, peak_market_cap_predictions)
    time_to_peak_r2 = r2_score(true_time_to_peak, time_to_peak_predictions)

    # Visualize predictions vs true values
    plt.figure(figsize=(12, 5))

    # Peak Market Cap
    plt.subplot(1, 2, 1)
    plt.scatter(true_peak_market_cap, peak_market_cap_predictions, alpha=0.5)
    plt.plot([true_peak_market_cap.min(), true_peak_market_cap.max()],
             [true_peak_market_cap.min(), true_peak_market_cap.max()],
             'r--', lw=2)
    plt.title('Peak Market Cap: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    # Time to Peak
    plt.subplot(1, 2, 2)
    plt.scatter(true_time_to_peak, time_to_peak_predictions, alpha=0.5)
    plt.plot([true_time_to_peak.min(), true_time_to_peak.max()],
             [true_time_to_peak.min(), true_time_to_peak.max()],
             'r--', lw=2)
    plt.title('Time to Peak: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.savefig('both_prediction_performance.png')

    # Print detailed results
    print("\n--- Model Performance Evaluation ---")
    print("\nPeak Market Cap Metrics:")
    print(f"Mean Absolute Error: {peak_market_cap_mae:.4f}")
    print(f"Mean Squared Error: {peak_market_cap_mse:.4f}")
    print(f"R² Score: {peak_market_cap_r2:.4f}")

    print("\nTime to Peak Metrics:")
    print(f"Mean Absolute Error: {time_to_peak_mae:.4f}")
    print(f"Mean Squared Error: {time_to_peak_mse:.4f}")
    print(f"R² Score: {time_to_peak_r2:.4f}")

    return {
        'results': {
            'peak_market_cap': {
                'mae': peak_market_cap_mae,
                'mse': peak_market_cap_mse,
                'r2': peak_market_cap_r2,
                'predictions': peak_market_cap_predictions,
                'true_values': true_peak_market_cap
            },
            'time_to_peak': {
                'mae': time_to_peak_mae,
                'mse': time_to_peak_mse,
                'r2': time_to_peak_r2,
                'predictions': time_to_peak_predictions,
                'true_values': true_time_to_peak
            }
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
    evaluate_model_both('best_peak_market_cap_model.pth', 'best_time_to_peak_model.pth', 'data/token_data_2025-01-08.csv')
