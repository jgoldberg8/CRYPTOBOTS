import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader



from Before30.dataset.hit_peak_30_dataset import HitPeakBefore30Dataset
from Before30.models.peak_before_30_model import HitPeakBefore30Predictor
from PeakMarketCap.models.peak_market_cap_model import PeakMarketCapPredictor
from TimeToPeak.datasets.time_token_dataset import TimeTokenDataset
from TimeToPeak.models.time_to_peak_model import TimeToPeakPredictor
from global_utilities import add_data_quality_features, clean_dataset_for_time


from PeakMarketCap.models.token_dataset import TokenDataset
from train_models import clean_dataset



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
    peak_market_cap_model = PeakMarketCapPredictor(input_size=input_size, hidden_size=256, num_layers=3, dropout_rate=0.5).to(device)

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
    
    plt.tight_layout()
    plt.savefig('peak_market_cap_prediction_performance.png', dpi=300, bbox_inches='tight')
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






def evaluate_hit_peak_before_30_model(hit_peak_before_30_model_path, data_paths):
    """
    Evaluate hit peak before 30 seconds model.
    
    Args:
        hit_peak_before_30_model_path (str): Path to saved hit peak before 30 model
        data_paths (list): List of paths to data CSV files
    
    Returns:
        dict: Evaluation metrics, predictions, and additional model information
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        
        # Ensure hit_peak_before_30 column exists
        if 'hit_peak_before_30' not in df.columns:
            df['hit_peak_before_30'] = (df['time_to_peak'] <= 30).astype(float)
        
        # Use clean_dataset_for_time for preprocessing
        df = clean_dataset(df)
        df = add_data_quality_features(df)
        dfs.append(df)

    # Combine datasets
    df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['hit_peak_before_30'])

    # Create datasets
    train_dataset = HitPeakBefore30Dataset(train_df)
    test_dataset = HitPeakBefore30Dataset(test_df, scaler={
        'global': train_dataset.global_scaler
    })

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 11
    hit_peak_before_30_model = HitPeakBefore30Predictor(
        input_size=input_size, 
        hidden_size=256, 
        num_layers=3, 
        dropout_rate=0.5
    ).to(device)

    # Load saved model
    hit_peak_checkpoint = torch.load(hit_peak_before_30_model_path, map_location=device)
    hit_peak_before_30_model.load_state_dict(hit_peak_checkpoint['model_state_dict'])
    hit_peak_before_30_model.eval()

    # Perform predictions
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for batch in test_loader:
            x_5s = batch['x_5s'].to(device)
            x_10s = batch['x_10s'].to(device)
            x_20s = batch['x_20s'].to(device)
            x_30s = batch['x_30s'].to(device)
            global_features = batch['global_features'].to(device)
            quality_features = batch['quality_features'].to(device)
            
            binary_pred = hit_peak_before_30_model(
                x_5s, x_10s, x_20s, x_30s, 
                global_features, 
                quality_features
            )
            
            all_predictions.append(binary_pred.cpu().squeeze())
            all_true_values.append(batch['targets'].cpu().squeeze())

    # Combine predictions
    predictions = torch.cat(all_predictions, dim=0).numpy()
    true_values = torch.cat(all_true_values, dim=0).numpy()

    # Binary classification metrics
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        confusion_matrix, 
        roc_auc_score, 
        precision_recall_curve, 
        average_precision_score
    )
    
    # Convert predictions to binary based on threshold
    binary_predictions = (predictions > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_values, binary_predictions),
        'precision': precision_score(true_values, binary_predictions),
        'recall': recall_score(true_values, binary_predictions),
        'f1_score': f1_score(true_values, binary_predictions),
        'roc_auc': roc_auc_score(true_values, predictions),
        'average_precision': average_precision_score(true_values, predictions),
        'confusion_matrix': confusion_matrix(true_values, binary_predictions)
    }

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ROC and Precision-Recall Curve
    plt.figure(figsize=(15, 5))

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    precision, recall, _ = precision_recall_curve(true_values, predictions)
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.text(0.05, 0.95, f'Avg Precision: {metrics["average_precision"]:.4f}',
             transform=plt.gca().transAxes)

    # Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('hit_peak_before_30_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed metrics
    print("\n=== Hit Peak Before 30 Model Performance Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Average Precision Score: {metrics['average_precision']:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)

    # Additional dataset insights
    print("\nDataset Composition:")
    unique, counts = np.unique(true_values, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples ({c/len(true_values)*100:.2f}%)")

    return {
        'metrics': metrics,
        'predictions': {
            'probabilities': predictions,
            'binary_predictions': binary_predictions,
            'true_values': true_values
        },
        'datasets': {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset
        },
        'dataframes': {
            'train_df': train_df,
            'test_df': test_df
        },
        'model': hit_peak_before_30_model
    }





# Usage
if __name__ == "__main__":
    evaluate_peak_market_cap_model(
         'best_peak_market_cap_model.pth',
       ['data/higher-peak-data.csv']
    )