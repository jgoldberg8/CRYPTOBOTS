import numpy as np
import torch
from TimeToPeak.utils.time_loss import TimePredictionLoss


def evaluate_model(model, loader, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    criterion = TimePredictionLoss(alpha=0.3, beta=0.1)
    
    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            mean, log_var = model(batch)
            loss = criterion(mean, log_var, batch['targets'])
            
            # Store predictions and actuals
            predictions.extend(mean.cpu().numpy())
            actuals.extend(batch['targets'].cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return {
        'loss': total_loss / len(loader),
        'mae': mae,
        'rmse': rmse
    }