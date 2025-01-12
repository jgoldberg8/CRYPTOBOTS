import torch
import torch.nn as nn
import torch.nn.functional as F
class PeakPredictionLoss(nn.Module):
    def __init__(self, hazard_weight=1.0, time_weight=1.5, confidence_weight=0.5):
        super().__init__()
        self.hazard_weight = hazard_weight
        self.time_weight = time_weight
        self.confidence_weight = confidence_weight
    
    def forward(self, hazard_pred, time_pred, confidence_pred, peak_proximity, time_to_peak, sample_weights, mask):
        # Only consider predictions after collection window
        valid_pred = mask.bool()
        
        if not valid_pred.any():
            return torch.tensor(0.0, device=hazard_pred.device)
            
        # Hazard prediction loss with peak proximity target
        hazard_loss = F.binary_cross_entropy(
            hazard_pred[valid_pred],
            peak_proximity[valid_pred],
            reduction='none'
        ) * sample_weights[valid_pred]
        
        # Time prediction loss with sample weights
        time_loss = F.smooth_l1_loss(
            time_pred[valid_pred],
            time_to_peak[valid_pred],
            reduction='none'
        ) * sample_weights[valid_pred]
        
        # Add relative error term
        relative_error = torch.abs(time_pred[valid_pred] - time_to_peak[valid_pred]) / (time_to_peak[valid_pred] + 1e-8)
        time_loss = time_loss + 0.5 * relative_error * sample_weights[valid_pred]
        
        # Confidence calibration
        confidence_target = torch.clamp(1 - relative_error, min=0.0, max=1.0)
        confidence_loss = F.binary_cross_entropy(
            confidence_pred[valid_pred],
            confidence_target.detach(),
            reduction='none'
        ) * sample_weights[valid_pred]
        
        # Combine losses
        total_loss = (
            self.hazard_weight * hazard_loss.mean() +
            self.time_weight * time_loss.mean() +
            self.confidence_weight * confidence_loss.mean()
        )
        
        return total_loss