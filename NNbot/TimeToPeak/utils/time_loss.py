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
        # Ensure all tensors have batch dimension first
        batch_size = hazard_pred.size(0)
        
        # Print shapes for debugging
        print("\nTensor shapes in loss calculation:")
        print(f"hazard_pred: {hazard_pred.shape}")
        print(f"time_pred: {time_pred.shape}")
        print(f"confidence_pred: {confidence_pred.shape}")
        print(f"peak_proximity: {peak_proximity.shape}")
        print(f"time_to_peak: {time_to_peak.shape}")
        print(f"sample_weights: {sample_weights.shape}")
        print(f"mask: {mask.shape}")
        
        # Reshape mask to match predictions if necessary
        mask = mask.view(batch_size, -1)
        valid_pred = mask.bool()
        
        # Count valid predictions
        num_valid = valid_pred.sum().item()
        print(f"Valid predictions: {num_valid}/{batch_size}")
        
        if num_valid == 0:
            # Return zero loss (but maintain gradients)
            return hazard_pred.sum() * 0.0
        
        # Ensure all tensors have compatible shapes
        hazard_pred = hazard_pred.view(batch_size, -1)
        time_pred = time_pred.view(batch_size, -1)
        confidence_pred = confidence_pred.view(batch_size, -1)
        peak_proximity = peak_proximity.view(batch_size, -1)
        time_to_peak = time_to_peak.view(batch_size, -1)
        sample_weights = sample_weights.view(batch_size, -1)
        
        # Calculate losses only for valid predictions
        # Use binary_cross_entropy_with_logits for hazard prediction
        hazard_loss = F.binary_cross_entropy_with_logits(
            hazard_pred[valid_pred],
            peak_proximity[valid_pred],
            reduction='none'
        )
        hazard_loss = (hazard_loss * sample_weights[valid_pred]).mean()
        
        # Time prediction loss
        time_loss = F.smooth_l1_loss(
            time_pred[valid_pred],
            time_to_peak[valid_pred],
            reduction='none'
        )
        time_loss = (time_loss * sample_weights[valid_pred]).mean()
        
        # Add relative error term
        relative_error = torch.abs(time_pred[valid_pred] - time_to_peak[valid_pred]) / (time_to_peak[valid_pred] + 1e-8)
        relative_loss = (relative_error * sample_weights[valid_pred]).mean()
        time_loss = time_loss + 0.5 * relative_loss
        
        # Confidence calibration using BCE with logits
        confidence_target = torch.clamp(1 - relative_error.detach(), min=0.0, max=1.0)
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_pred[valid_pred],
            confidence_target,
            reduction='none'
        )
        confidence_loss = (confidence_loss * sample_weights[valid_pred]).mean()
        
        # Combine losses with weights
        total_loss = (
            self.hazard_weight * hazard_loss +
            self.time_weight * time_loss +
            self.confidence_weight * confidence_loss
        )
        
        return total_loss