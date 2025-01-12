import os
import torch


import os
import shutil
import torch
import json
import datetime

def save_model_artifacts(model, train_loader, training_stats, config, save_dir, checkpoint_dir=None):
    """
    Save all model artifacts in a structured way
    
    Args:
        model: The trained model
        train_loader: Training data loader (for scalers)
        training_stats: Dictionary of training statistics
        config: Model configuration
        save_dir: Base directory for saving
        checkpoint_dir: Optional directory containing checkpoints
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    artifacts_dir = os.path.join(save_dir, 'TimeToPeak', 'Artifacts', f'run_{timestamp}')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save model artifacts
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_stats': training_stats,
        'scalers': train_loader.dataset.get_scalers(),  # Save scalers with consistent key name
        'timestamp': timestamp
    }, os.path.join(artifacts_dir, 'model_artifacts.pt'))
    
    # Save configuration separately for easy access
    with open(os.path.join(artifacts_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save training stats
    with open(os.path.join(artifacts_dir, 'training_stats.json'), 'w') as f:
        # Convert numpy values to native Python types
        clean_stats = {}
        for k, v in training_stats.items():
            if isinstance(v, (list, tuple)):
                clean_stats[k] = [float(x) if hasattr(x, 'item') else x for x in v]
            else:
                clean_stats[k] = float(v) if hasattr(v, 'item') else v
        json.dump(clean_stats, f, indent=4)
    
    # Copy best model checkpoint if provided
    if checkpoint_dir is not None:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, os.path.join(artifacts_dir, 'best_model.pt'))
    
    return artifacts_dir

def load_model_artifacts(artifacts_dir):
    """
    Load model artifacts from the specified directory
    """
    # Load model artifacts
    artifacts_path = os.path.join(artifacts_dir, 'model_artifacts.pt')
    checkpoint = torch.load(artifacts_path, map_location='cpu')
    
    return checkpoint