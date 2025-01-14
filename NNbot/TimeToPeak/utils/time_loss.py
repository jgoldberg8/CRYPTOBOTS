import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakPredictionLoss(nn.Module):
    def __init__(self, early_prediction_penalty=1.5, late_prediction_penalty=1.0):
        """
        Loss function for peak prediction that penalizes:
        1. False positives and false negatives (binary cross entropy)
        2. Early predictions more heavily than late predictions (asymmetric timing penalty)
        3. Low confidence in correct predictions
        
        Args:
            early_prediction_penalty: Weight for penalizing predictions before the true peak
            late_prediction_penalty: Weight for penalizing predictions after the true peak
        """
        super().__init__()
        self.early_penalty = early_prediction_penalty
        self.late_penalty = late_prediction_penalty
    
    def forward(self, peak_logits, confidence_logits, timestamps, true_peak_times, mask):
        """
        Calculate loss for peak predictions.
        
        Args:
            peak_logits: Model predictions for peaks (batch_size, 1)
            confidence_logits: Model confidence in predictions (batch_size, 1)
            timestamps: Current timestamp for each prediction (batch_size, 1)
            true_peak_times: True peak times (batch_size, 1)
            mask: Mask for valid predictions (batch_size, 1)
        """
        # Only calculate loss for valid predictions
        valid_pred = mask.bool()
        
        if valid_pred.sum() == 0:
            return peak_logits.sum() * 0.0
        
        # Calculate peak labels based on timing
        time_diffs = timestamps[valid_pred] - true_peak_times[valid_pred]
        is_peak = (time_diffs.abs() <= 5.0).float()  # 5 second tolerance window
        
        # Ensure shapes match
        peak_logits_valid = peak_logits[valid_pred].squeeze(-1)
        confidence_logits_valid = confidence_logits[valid_pred].squeeze(-1)
        
        # Binary cross entropy for peak prediction
        peak_loss = F.binary_cross_entropy_with_logits(
            peak_logits_valid,
            is_peak,
            reduction='none'
        )
        
        # Add timing penalty
        with torch.no_grad():
            predictions = torch.sigmoid(peak_logits_valid) > 0.5
            early_mask = (time_diffs < 0) & predictions
            late_mask = (time_diffs > 0) & predictions
            
            timing_weights = torch.ones_like(peak_loss)
            timing_weights[early_mask] = self.early_penalty
            timing_weights[late_mask] = self.late_penalty
        
        peak_loss = peak_loss * timing_weights
        
        # Calculate confidence targets
        with torch.no_grad():
            peak_probs = torch.sigmoid(peak_logits_valid)
            prediction_error = torch.abs(peak_probs - is_peak)
            confidence_target = 1.0 - prediction_error
            
            # Higher confidence target for correct predictions around true peak
            near_peak_mask = time_diffs.abs() <= 10.0  # Wider window for confidence
            confidence_target[near_peak_mask] = torch.max(
                confidence_target[near_peak_mask],
                1.0 - (time_diffs[near_peak_mask].abs() / 10.0)  # Linear decay
            )
        
        # Confidence loss
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits_valid,
            confidence_target,
            reduction='none'
        )
        
        # Weight confidence loss higher for predictions near peak
        confidence_weights = torch.ones_like(confidence_loss)
        confidence_weights[near_peak_mask] = 2.0
        confidence_loss = confidence_loss * confidence_weights
        
        # Combine losses
        total_loss = peak_loss.mean() + 0.5 * confidence_loss.mean()
        
        return total_loss