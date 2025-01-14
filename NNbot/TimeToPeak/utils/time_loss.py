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
        Loss function for peak prediction that includes:
        1. Soft peak labeling using Gaussian function
        2. Focal loss with class balancing
        3. Timing penalties for early/late predictions
        4. Confidence loss near peaks
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
            peak_logits: Model predictions for peaks (batch_size)
            confidence_logits: Model confidence in predictions (batch_size)
            timestamps: Current timestamp for each prediction (batch_size)
            true_peak_times: True peak times (batch_size)
            mask: Mask for valid predictions (batch_size)
        """
        # Print debug information
        print("Debug: Input tensor types")
        print(f"peak_logits type: {type(peak_logits)}, shape: {peak_logits.shape}")
        print(f"confidence_logits type: {type(confidence_logits)}, shape: {confidence_logits.shape}")
        print(f"timestamps type: {type(timestamps)}, shape: {timestamps.shape}")
        print(f"true_peak_times type: {type(true_peak_times)}, shape: {true_peak_times.shape}")
        print(f"mask type: {type(mask)}, shape: {mask.shape}")

        # Ensure all tensors are on the same device and float type
        device = peak_logits.device
        peak_logits = peak_logits.float()
        confidence_logits = confidence_logits.float()
        timestamps = timestamps.float()
        true_peak_times = true_peak_times.float()
        mask = mask.bool()

        # If no valid predictions, return zero loss
        if not mask.any():
            print("Debug: No valid predictions, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Safely handle indexing with mask
        peak_logits_valid = peak_logits[mask]
        confidence_logits_valid = confidence_logits[mask]
        timestamps_valid = timestamps[mask]
        true_peak_times_valid = true_peak_times[mask]

        # Calculate time differences
        time_diffs = timestamps_valid - true_peak_times_valid

        # Create soft peak labels using Gaussian function
        peak_labels = self.gaussian_peak_label(time_diffs)

        # Calculate focal loss with class balancing
        peak_loss = self.focal_bce_loss(
            peak_logits_valid,
            peak_labels,
            torch.tensor(self.pos_weight, device=device)
        )

        # Add timing penalty (with safe handling)
        with torch.no_grad():
            predictions = torch.sigmoid(peak_logits_valid) > 0.5

            # Initialize timing weights
            timing_weights = torch.ones_like(peak_loss)

            # Check and apply early prediction penalty
            early_mask = (time_diffs < 0) & predictions
            if early_mask.any():
                early_indices = torch.where(early_mask)[0]
                timing_weights[early_indices] = self.early_penalty

            # Check and apply late prediction penalty
            late_mask = (time_diffs > 0) & predictions
            if late_mask.any():
                late_indices = torch.where(late_mask)[0]
                timing_weights[late_indices] = self.late_penalty

        # Apply timing weights to peak loss
        peak_loss = peak_loss * timing_weights

        # Calculate confidence loss
        confidence_target = peak_labels
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