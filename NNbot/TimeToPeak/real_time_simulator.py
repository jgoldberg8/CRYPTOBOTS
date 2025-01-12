import datetime
import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from TimeToPeak.models.time_to_peak_model import RealTimePeakPredictor
from TimeToPeak.utils.clean_dataset import clean_dataset
from TimeToPeak.utils.setup_logging import setup_logging

class RealTimeDataSimulator:
    def __init__(self, data_df, window_size=60, step_size=5, model_path='checkpoints/best_model.pt'):
        """
        Simulates real-time data arrival from historical data.
        
        Args:
            data_df (pd.DataFrame): Full historical dataset
            window_size (int): Size of the sliding window in seconds
            step_size (int): How many seconds to advance in each step
            model_path (str): Path to the model checkpoint
        """
        self.data = data_df.sort_values('creation_time')
        self.window_size = window_size
        self.step_size = step_size
        self.current_time = None
        
        # Try to load scalers from various possible locations and keys
        self.scalers = self._load_scalers(model_path)
        
        # Base features that we want to capture for each granularity
        self.base_features = [
            'transaction_count',
            'buy_pressure',
            'volume',
            'rsi',
            'price_volatility',
            'volume_volatility',
            'momentum',
            'trade_amount_variance',
            'transaction_rate',
            'trade_concentration',
            'unique_wallets'
        ]
        self.reset()
    def _load_scalers(self, model_path):
        """
        Load scalers from checkpoint, handling different possible locations and keys.
        """
        try:
            # First try loading the direct checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check various possible keys for scalers
            if 'scalers' in checkpoint:
                return checkpoint['scalers']
            elif 'scaler' in checkpoint:
                return checkpoint['scaler']
            
            # If not found, try looking for model_artifacts.pt in the same directory
            model_dir = os.path.dirname(model_path)
            artifacts_path = os.path.join(model_dir, 'model_artifacts.pt')
            
            if os.path.exists(artifacts_path):
                artifacts = torch.load(artifacts_path, map_location='cpu')
                if 'scalers' in artifacts:
                    return artifacts['scalers']
                elif 'scaler' in artifacts:
                    return artifacts['scaler']
            
            # If still not found, try loading from scalers.pth
            scalers_path = os.path.join(model_dir, 'scalers.pth')
            if os.path.exists(scalers_path):
                return torch.load(scalers_path, map_location='cpu')
            
            raise KeyError("Could not find scalers in any of the expected locations")
            
        except Exception as e:
            raise RuntimeError(f"Error loading scalers from {model_path}: {str(e)}\n"
                             f"Please ensure the model checkpoint contains the scalers "
                             f"under either 'scalers' or 'scaler' key, or in a separate "
                             f"scalers.pth file.") from e
        
    
    def reset(self):
        """Reset simulation to start"""
        self.current_time = self.data['creation_time'].min()
    
    def _get_current_window(self):
        """Get data within current sliding window"""
        window_end = self.current_time
        window_start = window_end - pd.Timedelta(seconds=self.window_size)
        
        window_data = self.data[
            (self.data['creation_time'] > window_start) & 
            (self.data['creation_time'] <= window_end)
        ]
        return window_data
    
    def _process_window(self, window_data):
        """Process window data to match model expectations"""
        features = {}
        granularities = ['5s', '10s', '20s', '30s', '60s']
        
        # Process each granularity
        for granularity in granularities:
            gran_size = int(granularity.replace('s', ''))
            num_steps = len(self.base_features)
            
            # Create feature matrix for this granularity
            feature_matrix = np.zeros((1, num_steps))
            
            # Fill in available features
            for i, feature in enumerate(self.base_features):
                col_name = f"{feature}_0to{gran_size}s"
                if col_name in window_data.columns and len(window_data) > 0:
                    feature_matrix[0, i] = window_data[col_name].iloc[-1]
            
            features[f'features_{granularity}'] = feature_matrix
            features[f'length_{granularity}'] = 1 if len(window_data) > 0 else 1
        
        # Process global features
        global_features = np.array([
            window_data['initial_investment_ratio'].iloc[-1] if len(window_data) > 0 else 0,
            window_data['initial_market_cap'].iloc[-1] if len(window_data) > 0 else 0,
            window_data['peak_market_cap'].iloc[-1] if len(window_data) > 0 else 0,
            window_data['time_to_peak'].iloc[-1] if len(window_data) > 0 else 0,
            len(window_data) / self.window_size  # Activity density
        ]).reshape(1, -1)
        
        features['global_features'] = global_features
        return features


    def __iter__(self):
        """Return the iterator object (self)"""
        self.reset()
        self.max_time = self.data['creation_time'].max()
        return self
    
    def __next__(self):
        """Get next item in iteration"""
        if self.current_time > self.max_time:
            raise StopIteration
            
        window_data = self._get_current_window()
        features = self._process_window(window_data)
        
        # Get actual peak time if available
        actual_peak = None
        if len(window_data) > 0:
            actual_peak = window_data['time_to_peak'].iloc[-1]
        
        # Store current time and advance
        current_time = self.current_time
        self.current_time += pd.Timedelta(seconds=self.step_size)
        
        return {
            'time': current_time,
            'features': features,
            'actual_peak': actual_peak,
            'window_data': window_data
        }

def evaluate_realtime_predictions(model, data_df, window_size=60, step_size=5, model_path='checkpoints/best_model.pt'):
    """
    Evaluate model with a single prediction per token and inverse transform predictions
    """
    device = next(model.parameters()).device
    model.eval()
    
    predictions = []
    actuals = []
    peak_detections = []
    confidences = []
    
    # Get unique tokens
    unique_tokens = data_df['mint'].unique()
    
    with torch.no_grad():
        for token in tqdm(unique_tokens, desc="Evaluating tokens"):
            # Get data for this token
            token_data = data_df[data_df['mint'] == token].sort_values('creation_time')
            
            # Create simulator for this token's data
            simulator = RealTimeDataSimulator(token_data, window_size, step_size, model_path)
            
            # Get first batch only (initial prediction)
            batch = next(iter(simulator))
            
            # Convert features to tensors
            features = {
                k: torch.tensor(v).float().unsqueeze(0).to(device) 
                for k, v in batch['features'].items()
            }
            
            # Get model predictions
            mean, log_var, peak_detected, peak_prob = model(features, detect_peaks=True)
            
            # Get the raw prediction
            scaled_prediction = mean.cpu().numpy()[0]
            
            # Inverse transform the prediction using the target scaler
            raw_prediction = simulator.scalers['target'].inverse_transform([[scaled_prediction]])[0][0]
            
            # Store results
            predictions.append(raw_prediction)
            peak_detections.append(peak_detected.cpu().numpy()[0])
            confidences.append(torch.exp(-log_var).cpu().numpy()[0])
            
            if batch['actual_peak'] is not None:
                # Store raw (unscaled) actual value
                actuals.append(batch['actual_peak'])
            else:
                actuals.append(None)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    peak_detections = np.array(peak_detections)
    confidences = np.array(confidences)
    
    # Calculate R-squared
    valid_indices = [i for i, x in enumerate(actuals) if x is not None]
    valid_predictions = predictions[valid_indices]
    valid_actuals = np.array([actuals[i] for i in valid_indices])
    
    correlation_matrix = np.corrcoef(valid_actuals, valid_predictions)
    r_squared = correlation_matrix[0,1]**2
    
    metrics = {
        'mae': np.mean(np.abs(valid_predictions - valid_actuals)),
        'rmse': np.sqrt(np.mean((valid_predictions - valid_actuals) ** 2)),
        'r_squared': r_squared,
        'peak_detection_accuracy': np.mean(peak_detections[valid_indices] == (valid_predictions >= valid_actuals)),
        'average_confidence': np.mean(confidences),
        'predictions': predictions,
        'peak_detections': peak_detections,
        'confidences': confidences,
        'actuals': actuals
    }
    
    return metrics

def visualize_realtime_predictions(metrics):
    """Create scatter plot of predicted vs actual values with R²"""
    # Get valid predictions and actuals
    valid_indices = [i for i, x in enumerate(metrics['actuals']) if x is not None]
    valid_predictions = metrics['predictions'][valid_indices]
    valid_actuals = np.array([metrics['actuals'][i] for i in valid_indices])
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(valid_actuals, valid_predictions)
    r_squared = correlation_matrix[0,1]**2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    ax.scatter(valid_actuals, valid_predictions, alpha=0.5, c='lightblue')
    
    # Add trend line
    z = np.polyfit(valid_actuals, valid_predictions, 1)
    p = np.poly1d(z)
    ax.plot(valid_actuals, p(valid_actuals), "r--", alpha=0.8)
    
    # Add R² value
    ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
            transform=ax.transAxes, fontsize=10)
    
    # Labels and title
    ax.set_xlabel('True Values (seconds)')
    ax.set_ylabel('Predicted Values (seconds)')
    ax.set_title('Time to Peak: Predicted vs True')
    
    # Set equal axes with padding
    max_val = max(max(valid_actuals), max(valid_predictions))
    min_val = min(min(valid_actuals), min(valid_predictions))
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def analyze_prediction_transitions(metrics):
    """
    Analyze how predictions change over time
    """
    predictions = metrics['predictions']
    peak_detections = metrics['peak_detections']
    confidences = metrics['confidences']
    
    # Calculate prediction changes
    pred_changes = np.diff(predictions)
    conf_changes = np.diff(confidences)
    
    # Find significant changes in predictions
    sig_changes = np.where(np.abs(pred_changes) > np.std(pred_changes))[0]
    
    analysis = {
        'avg_prediction_change': np.mean(np.abs(pred_changes)),
        'std_prediction_change': np.std(pred_changes),
        'num_significant_changes': len(sig_changes),
        'avg_confidence_change': np.mean(np.abs(conf_changes)),
        'peak_detection_changes': np.sum(np.diff(peak_detections) != 0),
        'significant_moments': sig_changes
    }
    
    return analysis





def load_model_and_config(model_path):
    """Load trained model and its configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    config.setdefault('input_size', 11)
    config.setdefault('hidden_size', 256)
    config.setdefault('window_size', 60)
    
    model = RealTimePeakPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        window_size=config['window_size']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

def save_evaluation_results(metrics, analysis, save_dir):
    """Save evaluation results and figures"""
    # Create results directory
    results_dir = Path(save_dir) / 'realtime_evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    # Save metrics
    metrics_to_save = {}
    for k, v in metrics.items():
        metrics_to_save[k] = convert_to_native(v)
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4, default=convert_to_native)
    
    # Save analysis
    analysis_to_save = {}
    for k, v in analysis.items():
        analysis_to_save[k] = convert_to_native(v)
        
    with open(results_dir / 'analysis.json', 'w') as f:
        json.dump(analysis_to_save, f, indent=4, default=convert_to_native)
    
    # Create and save visualizations
    fig = visualize_realtime_predictions(metrics)
    fig.savefig(results_dir / 'realtime_predictions.png')
    plt.close(fig)
    
    # Create additional visualization for prediction transitions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot prediction changes
    prediction_changes = np.diff(metrics['predictions'])
    ax1.plot(prediction_changes, label='Prediction Changes')
    for moment in analysis['significant_moments']:
        ax1.axvline(x=moment, color='r', alpha=0.2)
    ax1.set_title('Prediction Changes Over Time')
    ax1.legend()
    
    # Plot confidence changes
    confidence_changes = np.diff(metrics['confidences'])
    ax2.plot(confidence_changes, label='Confidence Changes')
    ax2.set_title('Confidence Changes Over Time')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(results_dir / 'prediction_transitions.png')
    plt.close(fig)

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting real-time evaluation pipeline")
    
    # Configuration
    config = {
        'model_path': 'checkpoints/best_model.pt',
        'data_path': 'data/time-data.csv',
        'window_size': 60,  # seconds
        'step_size': 5,     # seconds
        'evaluation_dir': 'evaluation_results'
    }
    
    try:
        # Load and clean data
        logger.info("Loading data...")
        df = pd.read_csv(config['data_path'])
        df = clean_dataset(df)
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        logger.info(f"Data loaded and cleaned. Shape: {df.shape}")
        
        # Load model
        logger.info("Loading model...")
        model, model_config = load_model_and_config(config['model_path'])
        logger.info("Model loaded successfully")
        
        # Update config with model settings
        config.update(model_config)
        
        # Create evaluation directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = f"{config['evaluation_dir']}/evaluation_{timestamp}"
        os.makedirs(eval_dir, exist_ok=True)
        
        # Evaluate model
        logger.info("Starting real-time evaluation...")
        metrics = evaluate_realtime_predictions(
            model=model,
            data_df=df,
            window_size=config['window_size'],
            step_size=config['step_size']
        )
        
        logger.info("Analyzing prediction transitions...")
        analysis = analyze_prediction_transitions(metrics)
        
        # Save results
        logger.info("Saving evaluation results...")
        save_evaluation_results(metrics, analysis, eval_dir)
        
        # Log summary metrics
        logger.info("\nEvaluation Results:")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Peak Detection Accuracy: {metrics['peak_detection_accuracy']:.4f}")
        logger.info(f"Average Confidence: {metrics['average_confidence']:.4f}")
        logger.info(f"Number of Significant Changes: {analysis['num_significant_changes']}")
        logger.info(f"Average Prediction Change: {analysis['avg_prediction_change']:.4f}")
        
        # Save configuration
        with open(f"{eval_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Evaluation results saved to {eval_dir}")
        
        return {
            'metrics': metrics,
            'analysis': analysis,
            'eval_dir': eval_dir
        }
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    results = main()
    
    # Access results (optional)
    metrics = results['metrics']
    analysis = results['analysis']
    eval_dir = results['eval_dir']