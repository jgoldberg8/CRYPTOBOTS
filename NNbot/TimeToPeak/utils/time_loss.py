import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakPredictionLoss(nn.Module):
    def __init__(self, 
                 early_prediction_penalty=1.5,
                 late_prediction_penalty=1.0,
                 pos_weight=10.0,  # Weight for positive class to handle imbalance
                 focal_gamma=2.0):  # Focal loss parameter
        super().__init__()
        self.early_penalty = early_prediction_penalty
        self.late_penalty = late_prediction_penalty
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
    
    def gaussian_peak_label(self, time_diffs, sigma=10.0):
        """Create soft peak labels using Gaussian function"""
        return torch.exp(-(time_diffs ** 2) / (2 * sigma ** 2))
    
    def focal_bce_loss(self, pred, target, pos_weight):
        """Focal loss with class balancing"""
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction='none'
        )
        
        # Add focal loss term
        probs = torch.sigmoid(pred)
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_term = (1 - p_t) ** self.focal_gamma
        
        return (bce * focal_term).mean()
    
    def forward(self, peak_logits, confidence_logits, timestamps, true_peak_times, mask):
        valid_pred = mask.bool()
        
        if valid_pred.sum() == 0:
            return peak_logits.sum() * 0.0
        
        # Calculate time differences
        time_diffs = timestamps[valid_pred] - true_peak_times[valid_pred]
        
        # Create soft peak labels using Gaussian function
        peak_labels = self.gaussian_peak_label(time_diffs)
        
        # Get valid predictions
        peak_logits_valid = peak_logits[valid_pred].squeeze(-1)
        confidence_logits_valid = confidence_logits[valid_pred].squeeze(-1)
        
        # Calculate focal loss with class balancing
        peak_loss = self.focal_bce_loss(
            peak_logits_valid,
            peak_labels,
            torch.tensor(self.pos_weight).to(peak_logits.device)
        )
        
        # Add timing penalty (with safe handling)
        with torch.no_grad():
            predictions = torch.sigmoid(peak_logits_valid) > 0.5
            
            # Safe handling of empty tensors
            timing_weights = torch.ones_like(peak_loss)
            
            # Check if there are any early predictions
            early_mask = (time_diffs < 0) & predictions
            if early_mask.any():
                timing_weights[early_mask] = self.early_penalty
            
            # Check if there are any late predictions
            late_mask = (time_diffs > 0) & predictions
            if late_mask.any():
                timing_weights[late_mask] = self.late_penalty
        
        peak_loss = peak_loss * timing_weights
        
        # Calculate confidence loss with stronger weighting near peaks
        confidence_target = peak_labels  # Use same Gaussian labels for confidence
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits_valid,
            confidence_target,
            reduction='none'
        )
        
        # Weight confidence loss higher near peaks
        confidence_weights = 1.0 + peak_labels  # Higher weights near peak
        confidence_loss = (confidence_loss * confidence_weights).mean()
        
        # Combine losses
        total_loss = peak_loss + 0.5 * confidence_loss
        
        return total_loss