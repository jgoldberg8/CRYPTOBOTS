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
        # Ensure consistent shapes
        pred_mean = pred_mean.view(-1)
        pred_log_var = pred_log_var.view(-1)
        target = target.view(-1)
        
        # Base prediction loss
        huber_loss = self.huber(pred_mean, target)
        
        # Uncertainty loss
        precision = torch.exp(-pred_log_var)
        squared_error = (pred_mean - target)**2
        uncertainty_loss = 0.5 * torch.mean(
            pred_log_var + squared_error * precision - 0.5 * torch.log(precision)
        )
        
        # Early/late directional loss
        pred_error = pred_mean - target
        directional_loss = torch.mean(F.relu(pred_error) * 1.5 + F.relu(-pred_error) * 0.8)
        
        # Peak detection loss (if peak_target provided)
        if peak_target is not None:
            peak_target = peak_target.view(-1)
            pred_peak = pred_peak.view(-1)
            peak_loss = self.bce(pred_peak, peak_target)
        else:
            peak_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Combine losses
        total_loss = (
            huber_loss +
            self.alpha * uncertainty_loss +
            self.beta * directional_loss +
            self.peak_loss_weight * peak_loss
        )
        
        return total_loss