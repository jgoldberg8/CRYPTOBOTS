import os
import torch


def save_model_artifacts(model, scalers, config, save_dir):
    """Save model and related artifacts"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model state
    torch.save(model.state_dict(), f'{save_dir}/model.pth')
    
    # Save scalers
    torch.save(scalers, f'{save_dir}/scalers.pth')
    
    # Save config
    import json
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)