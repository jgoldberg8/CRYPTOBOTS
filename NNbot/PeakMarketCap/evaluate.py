import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from PeakMarketCap.models.peak_market_cap_model import TokenPricePredictor

def evaluate_token_price_model(data_paths):
    """
    Evaluate token price prediction model showing actual percentage values.
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize and train model
    predictor = TokenPricePredictor()
    metrics, importance = predictor.train(train_df)

    # Make predictions on test set
    test_predictions = predictor.predict(test_df)
    
    # Get actual and predicted values
    mask = (test_df['hit_peak_before_30'].astype(str).str.lower() == "false") & (test_df['percent_increase'] > 0)
    true_values = test_df.loc[mask, 'percent_increase'].values
    predictions = test_predictions.loc[mask, 'predicted_percent_increase'].values

    # Calculate overall metrics
    eval_metrics = {
        'percent_increase': {
            'mae': mean_absolute_error(true_values, predictions),
            'mse': mean_squared_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'r2': r2_score(true_values, predictions)
        }
    }

    # Calculate range-specific metrics
    ranges = [(0, 100), (100, 500), (500, float('inf'))]
    for low, high in ranges:
        mask = (true_values >= low) & (true_values < high)
        if np.any(mask):
            range_name = f'{low}-{high}'
            eval_metrics[f'range_{range_name}'] = {
                'mae': mean_absolute_error(true_values[mask], predictions[mask]),
                'rmse': np.sqrt(mean_squared_error(true_values[mask], predictions[mask])),
                'count': np.sum(mask)
            }

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5, s=20)

    # Set reasonable axis limits
    max_val = min(max(true_values.max(), predictions.max()), 200)  # Cap at 200% for better visualization
    min_val = max(min(true_values.min(), predictions.min()), 0)    # Floor at 0%
    padding = (max_val - min_val) * 0.05
    lims = [min_val - padding, max_val + padding]

    plt.plot(lims, lims, 'r--', lw=2)
    plt.title('Percent Increase: Predicted vs True')
    plt.xlabel('True Values (%)')
    plt.ylabel('Predicted Values (%)')
    plt.text(0.05, 0.95, f'R² = {eval_metrics["percent_increase"]["r2"]:.4f}',
             transform=plt.gca().transAxes, fontsize=10)
    
    # Add range-specific metrics to plot
    y_pos = 0.85
    for range_key in eval_metrics.keys():
        if range_key.startswith('range_'):
            range_metrics = eval_metrics[range_key]
            plt.text(0.05, y_pos, 
                    f'Range {range_key[6:]}: RMSE={range_metrics["rmse"]:.2f} (n={range_metrics["count"]})',
                    transform=plt.gca().transAxes, fontsize=8)
            y_pos -= 0.05

    # Create visualizations directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'percent_increase_prediction_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print evaluation results
    print("\n=== Percent Increase Model Performance Evaluation ===")
    print(f"Overall Metrics:")
    print(f"Mean Absolute Error: {eval_metrics['percent_increase']['mae']:.4f}%")
    print(f"Root Mean Squared Error: {eval_metrics['percent_increase']['rmse']:.4f}%")
    print(f"R² Score: {eval_metrics['percent_increase']['r2']:.4f}")
    
    print("\nRange-specific Metrics:")
    for range_key in eval_metrics.keys():
        if range_key.startswith('range_'):
            range_metrics = eval_metrics[range_key]
            print(f"\nRange {range_key[6:]}:")
            print(f"RMSE: {range_metrics['rmse']:.4f}%")
            print(f"MAE: {range_metrics['mae']:.4f}%")
            print(f"Sample Count: {range_metrics['count']}")

    print("\nTop 10 Most Important Features:")
    print(importance.head(10))

    return {
        'metrics': eval_metrics,
        'predictions': {
            'percent_increase_predictions': predictions,
            'true_percent_increase': true_values
        },
        'dataframes': {
            'train_df': train_df,
            'test_df': test_df
        },
        'model': predictor,
        'importance': importance
    }

if __name__ == "__main__":
    print(evaluate_token_price_model(['data/new-token-data.csv']))