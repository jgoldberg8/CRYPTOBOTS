import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakPredictionLoss(nn.Module):
    def __init__(self, 
                 early_prediction_penalty=1.5, 
                 late_prediction_penalty=1.0, 
                 pos_weight=10.0,  
                 focal_gamma=2.0,  
                 gaussian_sigma=10.0):
        """
        Loss function for peak prediction that penalizes:
        1. False positives and false negatives (binary cross entropy)
        2. Early predictions more heavily than late predictions (asymmetric timing penalty)
        3. Low confidence in correct predictions
        """
        super().__init__()
        self.early_penalty = early_prediction_penalty
        self.late_penalty = late_prediction_penalty
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.gaussian_sigma = gaussian_sigma

    def gaussian_peak_label(self, time_diffs, sigma=None):
        """
        Create soft peak labels using Gaussian function
        """
        if sigma is None:
            sigma = self.gaussian_sigma
        return torch.exp(-(time_diffs ** 2) / (2 * sigma ** 2))

    def focal_bce_loss(self, pred, target, pos_weight=None):
        """
        Focal loss with class balancing
        """
        if pos_weight is None:
            pos_weight = torch.tensor(self.pos_weight).to(pred.device)
        
        # Standard binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction='none'
        )
        
        # Add focal loss term
        probs = torch.sigmoid(pred)
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_term = (1 - p_t) ** self.focal_gamma
        
        return (bce * focal_term).mean()

    def forward(self, peak_logits, confidence_logits, timestamps, true_peak_times, mask):
        """
        Calculate loss for peak predictions.
        """
        # Ensure all inputs are squeezed to correct dimensions
        peak_logits = peak_logits.squeeze()
        confidence_logits = confidence_logits.squeeze()
        timestamps = timestamps.squeeze()
        true_peak_times = true_peak_times.squeeze()
        mask = mask.squeeze()
        
        # Only calculate loss for valid predictions
        valid_pred = mask.bool()
        
        if valid_pred.sum() == 0:
            return torch.tensor(0.0, device=peak_logits.device)
        
        # Filter valid predictions
        peak_logits_valid = peak_logits[valid_pred]
        confidence_logits_valid = confidence_logits[valid_pred]
        timestamps_valid = timestamps[valid_pred]
        true_peak_times_valid = true_peak_times[valid_pred]
        
        # Calculate peak labels based on timing
        time_diffs = timestamps_valid - true_peak_times_valid
        
        # Use Gaussian peak labeling instead of binary labels
        peak_labels = self.gaussian_peak_label(time_diffs)
        
        # Use focal loss for peak prediction
        peak_loss = self.focal_bce_loss(
            peak_logits_valid,
            peak_labels,
            torch.tensor(self.pos_weight).to(peak_logits.device)
        )
        
        # Add timing penalty
        with torch.no_grad():
            predictions = torch.sigmoid(peak_logits_valid) > 0.5
            early_mask = (time_diffs < 0) & predictions
            late_mask = (time_diffs > 0) & predictions
            
            # Initialize timing weights
            timing_weights = torch.ones_like(peak_loss)
            
            # Apply penalties carefully
            if early_mask.any():
                timing_weights[early_mask] *= self.early_penalty
            if late_mask.any():
                timing_weights[late_mask] *= self.late_penalty
        
        # Apply timing weights
        peak_loss = peak_loss * timing_weights
        
        # Calculate confidence targets
        with torch.no_grad():
            peak_probs = torch.sigmoid(peak_logits_valid)
            prediction_error = torch.abs(peak_probs - peak_labels)
            confidence_target = 1.0 - prediction_error
            
            # Higher confidence target for correct predictions around true peak
            near_peak_mask = time_diffs.abs() <= 10.0  # Wider window for confidence
            if near_peak_mask.any():
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
        if near_peak_mask.any():
            confidence_weights[near_peak_mask] = 2.0
        confidence_loss = confidence_loss * confidence_weights
        
        # Combine losses
        total_loss = peak_loss.mean() + 0.5 * confidence_loss.mean()
        
        return total_loss