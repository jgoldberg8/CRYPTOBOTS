import json
import torch
from TimeToPeak.models.time_to_peak_model import MultiGranularPeakPredictor


def load_model_artifacts(save_dir, device):
    """Load model and related artifacts"""
    # Load config
    with open(f'{save_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = MultiGranularPeakPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size']
    ).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(f'{save_dir}/model.pth'))
    
    # Load scalers
    scalers = torch.load(f'{save_dir}/scalers.pth')
    
    return model, scalers, config