import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakPredictionLoss(nn.Module):
    def __init__(self, 
                 early_prediction_penalty=1.5,
                 late_prediction_penalty=1.0,
                 pos_weight=10.0,  # Weight for positive class to handle imbalance
                 focal_gamma=2.0,  # Focal loss parameter
                 gaussian_sigma=10.0):  # Sigma for Gaussian peak labeling
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
        
        Args:
            time_diffs: Time differences from true peak
            sigma: Standard deviation for Gaussian (uses class default if None)
        
        Returns:
            Soft peak labels with Gaussian distribution
        """
        if sigma is None:
            sigma = self.gaussian_sigma
        return torch.exp(-(time_diffs ** 2) / (2 * sigma ** 2))

    def focal_bce_loss(self, pred, target, pos_weight=None):
        """
        Focal loss with class balancing
        
        Args:
            pred: Prediction logits
            target: Target labels
            pos_weight: Weight for positive class
        
        Returns:
            Focal binary cross-entropy loss
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
        
        Args:
            peak_logits: Model predictions for peaks (batch_size, 1)
            confidence_logits: Model confidence in predictions (batch_size, 1)
            timestamps: Current timestamp for each prediction (batch_size, 1)
            true_peak_times: True peak times (batch_size, 1)
            mask: Mask for valid predictions (batch_size, 1)
        """
        # Only calculate loss for valid predictions
        valid_pred = mask.squeeze().bool()
        
        if valid_pred.sum() == 0:
            return peak_logits.sum() * 0.0
        
        # Calculate time differences
        time_diffs = timestamps[valid_pred].squeeze() - true_peak_times[valid_pred].squeeze()
        
        # Create soft peak labels using Gaussian function
        peak_labels = self.gaussian_peak_label(time_diffs)
        
        # Ensure shapes match
        peak_logits_valid = peak_logits[valid_pred].squeeze(-1)
        confidence_logits_valid = confidence_logits[valid_pred].squeeze(-1)
        
        # Calculate focal loss with class balancing
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
            
            timing_weights = torch.ones_like(peak_loss)
            if early_mask.any():
                timing_weights[early_mask] = self.early_penalty
            if late_mask.any():
                timing_weights[late_mask] = self.late_penalty
        
        peak_loss = peak_loss * timing_weights
        
        # Calculate confidence loss with stronger weighting near peaks
        confidence_target = peak_labels  # Use Gaussian labels for confidence
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits_valid,
            confidence_target,
            reduction='none'
        )
        
        # Weight confidence loss higher near peaks
        confidence_weights = 1.0 + peak_labels
        confidence_loss = (confidence_loss * confidence_weights).mean()
        
        # Combine losses
        total_loss = peak_loss.mean() + 0.5 * confidence_loss
        
        return total_loss