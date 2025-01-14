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
        # Ensure inputs are tensors
        peak_logits = peak_logits.float()
        confidence_logits = confidence_logits.float()
        timestamps = timestamps.float()
        true_peak_times = true_peak_times.float()
        mask = mask.float()

        # Ensure all inputs are 2D tensors
        if peak_logits.dim() == 1:
            peak_logits = peak_logits.unsqueeze(1)
        if confidence_logits.dim() == 1:
            confidence_logits = confidence_logits.unsqueeze(1)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(1)
        if true_peak_times.dim() == 1:
            true_peak_times = true_peak_times.unsqueeze(1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)

        # Ensure consistent batch dimensions
        batch_size = peak_logits.size(0)
        assert confidence_logits.size(0) == batch_size, "Batch sizes must match"
        assert timestamps.size(0) == batch_size, "Batch sizes must match"
        assert true_peak_times.size(0) == batch_size, "Batch sizes must match"
        assert mask.size(0) == batch_size, "Batch sizes must match"

        # Create boolean mask for valid predictions
        valid_mask = mask.squeeze().bool()

        # If no valid predictions, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=peak_logits.device)

        # Filter valid predictions
        peak_logits_valid = peak_logits[valid_mask]
        confidence_logits_valid = confidence_logits[valid_mask]
        timestamps_valid = timestamps[valid_mask]
        true_peak_times_valid = true_peak_times[valid_mask]

        # Compute time differences
        time_diffs = timestamps_valid.squeeze() - true_peak_times_valid.squeeze()

        # Create soft peak labels using Gaussian distribution
        peak_labels = self.gaussian_peak_label(time_diffs)

        # Compute peak prediction loss with focal loss
        peak_loss = self.focal_bce_loss(
            peak_logits_valid.squeeze(), 
            peak_labels,
            torch.tensor(self.pos_weight).to(peak_logits.device)
        )

        # Compute prediction probabilities for timing penalty
        with torch.no_grad():
            predictions = torch.sigmoid(peak_logits_valid) > 0.5
            early_mask = (time_diffs < 0) & predictions.squeeze()
            late_mask = (time_diffs > 0) & predictions.squeeze()

            # Create timing weights
            timing_weights = torch.ones_like(peak_loss)
            if early_mask.any():
                timing_weights[early_mask] *= self.early_penalty
            if late_mask.any():
                timing_weights[late_mask] *= self.late_penalty

        # Apply timing weights to peak loss
        peak_loss = peak_loss * timing_weights

        # Compute confidence loss
        with torch.no_grad():
            peak_probs = torch.sigmoid(peak_logits_valid)
            prediction_error = torch.abs(peak_probs - peak_labels)
            confidence_target = 1.0 - prediction_error

            # Adjust confidence for predictions near peak
            near_peak_mask = time_diffs.abs() <= 10.0
            if near_peak_mask.any():
                confidence_target[near_peak_mask] = torch.max(
                    confidence_target[near_peak_mask],
                    1.0 - (time_diffs[near_peak_mask].abs() / 10.0)
                )

        # Compute confidence loss
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence_logits_valid.squeeze(),
            confidence_target,
            reduction='none'
        )

        # Weight confidence loss
        confidence_weights = torch.ones_like(confidence_loss)
        if near_peak_mask.any():
            confidence_weights[near_peak_mask] = 2.0
        confidence_loss = confidence_loss * confidence_weights

        # Combine and return total loss
        total_loss = peak_loss.mean() + 0.5 * confidence_loss.mean()
        
        return total_loss