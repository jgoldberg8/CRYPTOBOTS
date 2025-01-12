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
      """Process window data to match the exact format expected by the trained model"""
      features = {}
      granularities = ['5s', '10s', '20s', '30s', '60s']
      
      for granularity in granularities:
          gran_size = int(granularity.replace('s', ''))
          gran_features = []
          
          # For this granularity, how many time steps fit in our window
          num_steps = len(self.base_features)
          feature_matrix = np.zeros((1, num_steps))  # batch_size=1, features=num_steps
          
          # Fill in features we have
          for i, feature in enumerate(self.base_features):
              col_name = f"{feature}_0to{gran_size}s"
              if col_name in window_data.columns and len(window_data) > 0:
                  feature_matrix[0, i] = window_data[col_name].iloc[-1]
          
          features[f'features_{granularity}'] = feature_matrix
          features[f'length_{granularity}'] = 1 if len(window_data) > 0 else 1
      
      # Add global features in the same shape as during training
      global_features = np.array([
          window_data['initial_investment_ratio'].iloc[-1] if len(window_data) > 0 else 0,
          window_data['initial_market_cap'].iloc[-1] if len(window_data) > 0 else 0,
          window_data['peak_market_cap'].iloc[-1] if len(window_data) > 0 else 0,
          window_data['time_to_peak'].iloc[-1] if len(window_data) > 0 else 0,
          len(window_data) / self.window_size  # Activity density
      ]).reshape(1, -1)  # Make it 2D with batch_size=1
      
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
    """
    Evaluate model performance in simulated real-time conditions
    
    Args:
        model: Trained RealTimePeakPredictor model
        data_df: DataFrame with historical data
        window_size: Size of sliding window in seconds
        step_size: How many seconds to advance in each step
    
    Returns:
        Dictionary of evaluation metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    simulator = RealTimeDataSimulator(data_df, window_size, step_size)
    
    predictions = []
    actuals = []
    peak_detections = []
    confidences = []
    timestamps = []
    
    with torch.no_grad():
        for batch in tqdm(simulator, desc="Evaluating real-time predictions"):
            # Convert features to tensors
            features = {
                k: torch.tensor(v).float().unsqueeze(0).to(device) 
                for k, v in batch['features'].items()
            }
            
            # Get model predictions
            mean, log_var, peak_detected, peak_prob = model(features, detect_peaks=True)
            
            # Store results
            predictions.append(mean.cpu().numpy()[0])
            peak_detections.append(peak_detected.cpu().numpy()[0])
            confidences.append(torch.exp(-log_var).cpu().numpy()[0])
            timestamps.append(batch['time'])
            
            if batch['actual_peak'] is not None:
                actuals.append(batch['actual_peak'])
            else:
                actuals.append(None)
    
    # Calculate metrics
    predictions = np.array(predictions)
    peak_detections = np.array(peak_detections)
    confidences = np.array(confidences)
    
    # Filter out predictions where we have actual values
    valid_indices = [i for i, x in enumerate(actuals) if x is not None]
    valid_predictions = predictions[valid_indices]
    valid_actuals = np.array([actuals[i] for i in valid_indices])
    valid_peak_detections = peak_detections[valid_indices]
    
    metrics = {
        'mae': np.mean(np.abs(valid_predictions - valid_actuals)),
        'rmse': np.sqrt(np.mean((valid_predictions - valid_actuals) ** 2)),
        'peak_detection_accuracy': np.mean(valid_peak_detections == (valid_predictions >= valid_actuals)),
        'average_confidence': np.mean(confidences),
        'predictions': predictions,
        'peak_detections': peak_detections,
        'confidences': confidences,
        'timestamps': timestamps,
        'actuals': actuals
    }
    
    return metrics

def visualize_realtime_predictions(metrics):
    """
    Create visualization of real-time prediction performance
    """
    import matplotlib.pyplot as plt
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Convert timestamps to relative seconds
    start_time = metrics['timestamps'][0]
    relative_times = [(t - start_time).total_seconds() for t in metrics['timestamps']]
    
    # Plot predictions vs actuals
    ax1.plot(relative_times, metrics['predictions'], label='Predictions', alpha=0.7)
    valid_times = [t for t, a in zip(relative_times, metrics['actuals']) if a is not None]
    valid_actuals = [a for a in metrics['actuals'] if a is not None]
    ax1.scatter(valid_times, valid_actuals, c='red', label='Actual Peaks', alpha=0.5)
    
    # Add confidence bands
    confidence = metrics['confidences']
    ax1.fill_between(relative_times, 
                     metrics['predictions'] - 1/confidence,
                     metrics['predictions'] + 1/confidence,
                     alpha=0.2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Predicted Time to Peak')
    ax1.set_title('Real-time Peak Predictions with Confidence')
    ax1.legend()
    
    # Plot peak detections
    ax2.plot(relative_times, metrics['peak_detections'], label='Peak Detected', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Peak Detected')
    ax2.set_title('Real-time Peak Detection')
    ax2.set_ylim(-0.1, 1.1)
    
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
    
    # Save metrics
    metrics_to_save = {
        k: v if not isinstance(v, (np.ndarray, pd.Timestamp)) else v.tolist()
        for k, v in metrics.items()
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # Save analysis
    with open(results_dir / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
    
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