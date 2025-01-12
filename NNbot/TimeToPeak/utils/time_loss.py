import torch
import torch.nn as nn
import torch.nn.functional as F
class RealTimePeakLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.15, peak_loss_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.peak_loss_weight = peak_loss_weight
        self.huber = nn.SmoothL1Loss(beta=1.0)
        self.bce = nn.BCELoss()
    
    def forward(self, pred_mean, pred_log_var, pred_peak, target, peak_target=None):
        # Ensure consistent shapes and types
        pred_mean = pred_mean.view(-1)
        pred_log_var = pred_log_var.view(-1)
        target = target.view(-1)
        
        # Base prediction loss (always positive)
        huber_loss = self.huber(pred_mean, target)
        
        # Uncertainty loss (modified to ensure positivity)
        precision = torch.exp(-pred_log_var)
        squared_error = (pred_mean - target)**2
        uncertainty_loss = torch.mean(
            0.5 * (pred_log_var + squared_error * precision)
        )
        
        # Directional loss (modified to be always positive)
        pred_error = pred_mean - target
        early_penalty = F.relu(pred_error) * 1.5  # Predictions too early
        late_penalty = F.relu(-pred_error) * 0.8   # Predictions too late
        directional_loss = torch.mean(early_penalty + late_penalty)
        
        # Peak detection loss
        if peak_target is not None:
            peak_target = peak_target.view(-1)
            pred_peak = pred_peak.view(-1)
            peak_loss = self.bce(pred_peak, peak_target.float())
        else:
            peak_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Combine losses with non-negative weights
        total_loss = (
            huber_loss + 
            self.alpha * uncertainty_loss +
            self.beta * directional_loss +
            self.peak_loss_weight * peak_loss
        )
        
        return total_loss