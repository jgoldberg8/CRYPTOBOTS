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
    def __init__(self, data_df, window_size=60, step_size=5):
        """
        Simulates real-time data arrival from historical data.
        
        Args:
            data_df (pd.DataFrame): Full historical dataset
            window_size (int): Size of the sliding window in seconds
            step_size (int): How many seconds to advance in each step
        """
        self.data = data_df.sort_values('creation_time')
        self.window_size = window_size
        self.step_size = step_size
        self.current_time = None
        
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
          # [batch_size, features] -> will be expanded in network as needed
          feature_matrix = np.zeros((1, num_steps))
          
          # Fill in available features
          for i, feature in enumerate(self.base_features):
              col_name = f"{feature}_0to{gran_size}s"
              if col_name in window_data.columns and len(window_data) > 0:
                  feature_matrix[0, i] = window_data[col_name].iloc[-1]
          
          features[f'features_{granularity}'] = feature_matrix
          features[f'length_{granularity}'] = 1 if len(window_data) > 0 else 1
      
      # Process global features [batch_size, features]
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

def evaluate_realtime_predictions(model, data_df, window_size=60, step_size=5):
    """Evaluate model by making a single prediction for each token"""
    device = next(model.parameters()).device
    model.eval()
    
    predictions = []
    actuals = []
    confidences = []
    
    # Group by token
    unique_tokens = data_df['mint'].unique()
    
    with torch.no_grad():
        for token in tqdm(unique_tokens, desc="Evaluating predictions"):
            # Get data for this token
            token_data = data_df[data_df['mint'] == token].sort_values('creation_time')
            
            # Create simulator and get first window
            simulator = RealTimeDataSimulator(token_data, window_size, step_size)
            batch = next(iter(simulator))
            
            # Convert features to tensors
            features = {
                k: torch.tensor(v).float().to(device) 
                for k, v in batch['features'].items()
            }
            
            # Get initial prediction
            mean, log_var, peak_detected, peak_prob = model(features, detect_peaks=True)
            
            # Store results
            predictions.append(mean.cpu().numpy().item())
            actuals.append(token_data['time_to_peak'].iloc[0])
            confidences.append(torch.exp(-log_var).cpu().numpy().item())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    confidences = np.array(confidences)
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(actuals, predictions)
    r_squared = correlation_matrix[0,1]**2
    
    metrics = {
        'mae': np.mean(np.abs(predictions - actuals)),
        'rmse': np.sqrt(np.mean((predictions - actuals) ** 2)),
        'r_squared': r_squared,
        'average_confidence': np.mean(confidences),
        'predictions': predictions,
        'actuals': actuals,
        'confidences': confidences
    }
    
    return metrics

def visualize_realtime_predictions(metrics):
    """Create scatter plot of predicted vs actual values with R²"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    ax.scatter(metrics['actuals'], metrics['predictions'], alpha=0.5, c='lightblue')
    
    # Add trend line
    z = np.polyfit(metrics['actuals'], metrics['predictions'], 1)
    p = np.poly1d(z)
    ax.plot(metrics['actuals'], p(metrics['actuals']), "r--", alpha=0.8)
    
    # Add R² value
    ax.text(0.05, 0.95, f'R² = {metrics["r_squared"]:.4f}', 
            transform=ax.transAxes, fontsize=10)
    
    # Labels and title
    ax.set_xlabel('True Values (seconds)')
    ax.set_ylabel('Predicted Values (seconds)')
    ax.set_title('Time to Peak: Predicted vs True')
    
    # Make plot square and set equal scales
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits based on data range
    max_val = max(metrics['actuals'].max(), metrics['predictions'].max())
    min_val = min(metrics['actuals'].min(), metrics['predictions'].min())
    ax.set_xlim(min_val - 50, max_val + 50)
    ax.set_ylim(min_val - 50, max_val + 50)
    
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