
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from Before30.dataset.hit_peak_30_dataset import HitPeakBefore30Dataset
from Before30.models.peak_before_30_model import HitPeakBefore30Predictor
from Before30.utils.clean_dataset import clean_dataset
from torch.utils.data import DataLoader

from utils.add_data_quality_features import add_data_quality_features


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

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a 'Visualizations' subdirectory if it doesn't exist
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    print(visualizations_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'hit_peak_before_30__prediction_performance.png'), dpi=300, bbox_inches='tight')
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
    evaluate_hit_peak_before_30_model(
         'best_hit_peak_before_30_model.pth',
       ['data/token-data.csv']
    )