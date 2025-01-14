import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from TimeToPeak.datasets.time_token_dataset import TimePeakDataset
from TimeToPeak.models.time_to_peak_model import PeakPredictor, predict_peak
from TimeToPeak.utils.clean_dataset import clean_dataset

def evaluate_model(model, dataset, device='cuda', threshold=0.5):
    """Evaluate model performance and collect predictions"""
    true_peaks = []
    predicted_peaks = []
    confidences = []
    tokens = []
    all_probs = []  # Track all probabilities
    
    model.eval()
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        features = sample['features'].unsqueeze(0)  # Add batch dimension
        global_features = sample['global_features'].unsqueeze(0)
        
        peak_prob, confidence = predict_peak(model, features, global_features, device)
        
        true_peaks.append(sample['time_to_peak'].item())
        predicted_peaks.append(sample['timestamp'].item() if peak_prob > threshold else None)
        confidences.append(confidence)
        tokens.append(idx)
        all_probs.append(peak_prob)
    
    # Print probability distribution info
    probs = np.array(all_probs)
    print("\nProbability Distribution:")
    print(f"Min prob: {probs.min():.4f}")
    print(f"Max prob: {probs.max():.4f}")
    print(f"Mean prob: {probs.mean():.4f}")
    print(f"Median prob: {np.median(probs):.4f}")
    print(f"Std prob: {probs.std():.4f}")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(probs, p):.4f}")
    
    # Try different thresholds
    print("\nPredictions at different thresholds:")
    for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        n_predictions = sum(1 for p in probs if p > t)
        print(f"Threshold {t:.2f}: {n_predictions} predictions ({n_predictions/len(probs)*100:.2f}%)")
    
    return {
        'true_peaks': true_peaks,
        'predicted_peaks': predicted_peaks,
        'confidences': confidences,
        'tokens': tokens,
        'probabilities': all_probs
    }

def plot_peak_comparison(results, save_path):
    """Plot true vs predicted peak times"""
    # Filter out None predictions
    valid_predictions = [(true, pred, token) for true, pred, token 
                        in zip(results['true_peaks'], results['predicted_peaks'], results['tokens']) 
                        if pred is not None]
    
    if not valid_predictions:
        logging.warning("No valid predictions found for plotting")
        return
    
    true_peaks, pred_peaks, tokens = zip(*valid_predictions)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(true_peaks, pred_peaks, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(true_peaks), min(pred_peaks))
    max_val = max(max(true_peaks), max(pred_peaks))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Peak Time (seconds)')
    plt.ylabel('Predicted Peak Time (seconds)')
    plt.title('True vs Predicted Peak Times')
    
    # Add error margins
    plt.fill_between([min_val, max_val], 
                    [x - 10 for x in [min_val, max_val]], 
                    [x + 10 for x in [min_val, max_val]], 
                    alpha=0.2, color='gray', label='±10s margin')
    
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(results):
    """Calculate detailed performance metrics"""
    # Convert to binary classification problem
    y_true = []
    y_pred = []
    
    tolerance = 10  # 10 second tolerance for peak prediction
    
    for true_peak, pred_peak in zip(results['true_peaks'], results['predicted_peaks']):
        if pred_peak is None:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
        # Check if prediction is within tolerance of true peak
        if pred_peak is not None and abs(true_peak - pred_peak) <= tolerance:
            y_true.append(1)
        else:
            y_true.append(0)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate timing errors for correct predictions
    timing_errors = []
    for true_peak, pred_peak in zip(results['true_peaks'], results['predicted_peaks']):
        if pred_peak is not None:
            timing_errors.append(abs(true_peak - pred_peak))
    
    timing_stats = {
        'mean_error': np.mean(timing_errors) if timing_errors else None,
        'median_error': np.median(timing_errors) if timing_errors else None,
        'std_error': np.std(timing_errors) if timing_errors else None
    }
    
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'timing_stats': timing_stats
    }

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Load model and configuration
        logger.info("Loading model and configuration...")
        checkpoint = torch.load('checkpoints/best_model.pt')
        with open('checkpoints/training_info.json', 'r') as f:
            training_info = json.load(f)
        
        # Initialize model
        model = PeakPredictor(
            feature_size=training_info['model_config']['feature_size'],
            hidden_size=training_info['model_config']['hidden_size'],
            dropout_rate=training_info['model_config']['dropout_rate']
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Load data
        logger.info("Loading test data...")
        df = pd.read_csv('data/time-data.csv')
        df = clean_dataset(df)
        train_size = int(0.8 * len(df))
        test_df = df[train_size:]  # Use validation set for evaluation
        
        # Load scalers
        scalers = torch.load('checkpoints/scalers.pt')
        
        # Create dataset
        test_dataset = TimePeakDataset(test_df, scaler=scalers, train=False)
        
        # Evaluate model with different thresholds
        logger.info("Evaluating model...")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            logger.info(f"\nEvaluating with threshold {threshold}:")
            results = evaluate_model(model, test_dataset, device, threshold)
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = calculate_metrics(results)
        
        # Save metrics
        with open('results/evaluation_metrics.json', 'w') as f:
            json.dump({
                'classification_report': metrics['classification_report'],
                'timing_stats': metrics['timing_stats']
            }, f, indent=4)
        
        # Plot results
        logger.info("Generating plots...")
        plot_peak_comparison(results, 'results/peak_comparison.png')
        
        # Print summary
        logger.info("\nEvaluation Results:")
        logger.info(f"Classification Report:\n{metrics['classification_report']}")
        logger.info("\nTiming Statistics:")
        logger.info(f"Mean Error: {metrics['timing_stats']['mean_error']:.2f} seconds")
        logger.info(f"Median Error: {metrics['timing_stats']['median_error']:.2f} seconds")
        logger.info(f"Error Std Dev: {metrics['timing_stats']['std_error']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()