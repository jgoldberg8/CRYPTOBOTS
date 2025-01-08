import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import the previously defined classes (TokenDataset and TokenPredictor)
from nn_model import TokenDataset, TokenPredictor, clean_dataset, add_data_quality_features
from peak_model import PeakMarketCapPredictor, TimeToPeakPredictor

def evaluate_model_single(model_path, data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = clean_dataset(df)
    df = add_data_quality_features(df)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    test_dataset = TokenDataset(test_df, scaler=train_dataset.scaler)

    # Initialize model
    input_size = 11
    model = TokenPredictor(input_size, hidden_size=256, num_layers=3, dropout_rate=0.5)

    # Load best saved model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    # Prepare for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Collect predictions and true values
    all_predictions = []
    all_true_values = []

    # Disable gradient calculation
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Get individual sample
            sample = test_dataset[idx]

            # Move tensors to device
            x_5s = sample['x_5s'].unsqueeze(0).to(device)
            x_10s = sample['x_10s'].unsqueeze(0).to(device)
            x_20s = sample['x_20s'].unsqueeze(0).to(device)
            x_30s = sample['x_30s'].unsqueeze(0).to(device)
            global_features = sample['global_features'].unsqueeze(0).to(device)
            quality_features = sample['quality_features'].unsqueeze(0).to(device)

            # Make prediction
            prediction = model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)

            # Inverse transform the prediction and true values
            pred_np = prediction.cpu().numpy()[0]
            true_np = sample['targets'].numpy()

            # Inverse transform using the scaler
            pred_original = test_dataset.scaler.inverse_transform(
                pred_np.reshape(1, -1)
            )[0]
            true_original = test_dataset.scaler.inverse_transform(
                true_np.reshape(1, -1)
            )[0]

            all_predictions.append(pred_original)
            all_true_values.append(true_original)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_values = np.array(all_true_values)

    # Calculate performance metrics
    peak_market_cap_mae = mean_absolute_error(true_values[:, 0], predictions[:, 0])
    time_to_peak_mae = mean_absolute_error(true_values[:, 1], predictions[:, 1])

    peak_market_cap_mse = mean_squared_error(true_values[:, 0], predictions[:, 0])
    time_to_peak_mse = mean_squared_error(true_values[:, 1], predictions[:, 1])

    peak_market_cap_r2 = r2_score(true_values[:, 0], predictions[:, 0])
    time_to_peak_r2 = r2_score(true_values[:, 1], predictions[:, 1])

    # Visualize predictions vs true values
    plt.figure(figsize=(12, 5))

    # Peak Market Cap
    plt.subplot(1, 2, 1)
    plt.scatter(true_values[:, 0], predictions[:, 0], alpha=0.5)
    plt.plot([true_values[:, 0].min(), true_values[:, 0].max()],
             [true_values[:, 0].min(), true_values[:, 0].max()],
             'r--', lw=2)
    plt.title('Peak Market Cap: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    # Time to Peak
    plt.subplot(1, 2, 2)
    plt.scatter(true_values[:, 1], predictions[:, 1], alpha=0.5)
    plt.plot([true_values[:, 1].min(), true_values[:, 1].max()],
             [true_values[:, 1].min(), true_values[:, 1].max()],
             'r--', lw=2)
    plt.title('Time to Peak: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.savefig('single_prediction_performance.png')

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
    peak_market_cap_model.load_state_dict(peak_market_cap_checkpoint['model_state_dict'])
    time_to_peak_model.load_state_dict(time_to_peak_checkpoint['model_state_dict'])

    # Set to evaluation mode
    peak_market_cap_model.eval()
    time_to_peak_model.eval()

    # Prepare for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peak_market_cap_model = peak_market_cap_model.to(device)
    time_to_peak_model = time_to_peak_model.to(device)

    # Collect predictions and true values
    all_peak_market_cap_predictions = []
    all_time_to_peak_predictions = []
    all_true_values = []

    # Disable gradient calculation
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Get individual sample
            sample = test_dataset[idx]

            # Move tensors to device
            x_5s = sample['x_5s'].unsqueeze(0).to(device)
            x_10s = sample['x_10s'].unsqueeze(0).to(device)
            x_20s = sample['x_20s'].unsqueeze(0).to(device)
            x_30s = sample['x_30s'].unsqueeze(0).to(device)
            global_features = sample['global_features'].unsqueeze(0).to(device)
            quality_features = sample['quality_features'].unsqueeze(0).to(device)

            # Make predictions
            peak_market_cap_prediction = peak_market_cap_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)
            time_to_peak_prediction = time_to_peak_model(x_5s, x_10s, x_20s, x_30s, global_features, quality_features)

            # Inverse transform the predictions and true values
            peak_market_cap_pred_np = peak_market_cap_prediction.cpu().numpy()[0]
            time_to_peak_pred_np = time_to_peak_prediction.cpu().numpy()[0]
            true_np = sample['targets'].numpy()

            # Inverse transform using the scaler
            peak_market_cap_pred_original = test_dataset.scaler.inverse_transform(
                peak_market_cap_pred_np.reshape(1, -1)
            )[0]
            time_to_peak_pred_original = test_dataset.scaler.inverse_transform(
                time_to_peak_pred_np.reshape(1, -1)
            )[0]
            true_original = test_dataset.scaler.inverse_transform(
                true_np.reshape(1, -1)
            )[0]

            all_peak_market_cap_predictions.append(peak_market_cap_pred_original)
            all_time_to_peak_predictions.append(time_to_peak_pred_original)
            all_true_values.append(true_original)

    # Convert to numpy arrays
    peak_market_cap_predictions = np.array(all_peak_market_cap_predictions)
    time_to_peak_predictions = np.array(all_time_to_peak_predictions)
    true_values = np.array(all_true_values)

    # Calculate performance metrics
    peak_market_cap_mae = mean_absolute_error(true_values[:, 0], peak_market_cap_predictions)
    time_to_peak_mae = mean_absolute_error(true_values[:, 1], time_to_peak_predictions)

    peak_market_cap_mse = mean_squared_error(true_values[:, 0], peak_market_cap_predictions)
    time_to_peak_mse = mean_squared_error(true_values[:, 1], time_to_peak_predictions)

    peak_market_cap_r2 = r2_score(true_values[:, 0], peak_market_cap_predictions)
    time_to_peak_r2 = r2_score(true_values[:, 1], time_to_peak_predictions)

    # Visualize predictions vs true values
    plt.figure(figsize=(12, 5))

    # Peak Market Cap
    plt.subplot(1, 2, 1)
    plt.scatter(true_values[:, 0], peak_market_cap_predictions, alpha=0.5)
    plt.plot([true_values[:, 0].min(), true_values[:, 0].max()],
            [true_values[:, 0].min(), true_values[:, 0].max()],
            'r--', lw=2)
    plt.title('Peak Market Cap: Predicted vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    # Time to Peak
    plt.subplot(1, 2, 2)
    plt.scatter(true_values[:, 1], time_to_peak_predictions, alpha=0.5)
    plt.plot([true_values[:, 1].min(), true_values[:, 1].max()],
            [true_values[:, 1].min(), true_values[:, 1].max()],
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



# Usage
if __name__ == "__main__":
    #evaluate_model_single('best_model.pth', 'data/token_data_2025-01-07.csv')
    evaluate_model_both('best_peak_market_cap_model.pth', 'best_time_to_peak_model.pth', 'data/token_data_2025-01-07.csv')
