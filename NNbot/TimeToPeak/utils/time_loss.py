import torch
import torch.nn as nn
import torch.nn.functional as F
class RealTimePeakLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.15, peak_loss_weight=0.5):
        super().__init__()
        self.alpha = abs(alpha)  # Ensure positive weights
        self.beta = abs(beta)
        self.gamma = abs(gamma)
        self.peak_loss_weight = abs(peak_loss_weight)
        self.huber = nn.SmoothL1Loss(reduction='none')  # Changed to 'none' for more control
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, pred_mean, pred_log_var, pred_peak, target, peak_target=None):
        # Ensure consistent shapes and types
        pred_mean = pred_mean.view(-1)
        pred_log_var = pred_log_var.view(-1)
        target = target.view(-1)
        
        # Base prediction loss (guaranteed positive due to Huber loss nature)
        huber_loss = torch.mean(self.huber(pred_mean, target))
        
        # Uncertainty loss (modified to ensure positivity)
        precision = torch.exp(-pred_log_var).clamp(min=1e-8)  # Prevent division by zero
        squared_error = (pred_mean - target)**2
        uncertainty_loss = torch.mean(
            0.5 * torch.abs(pred_log_var + squared_error * precision)  # abs to ensure positivity
        )
        
        # Directional loss (already guaranteed positive due to ReLU)
        pred_error = pred_mean - target
        early_penalty = F.relu(pred_error) * 1.5  # Predictions too early
        late_penalty = F.relu(-pred_error) * 0.8   # Predictions too late
        directional_loss = torch.mean(early_penalty + late_penalty)
        
        # Peak detection loss
        if peak_target is not None and pred_peak is not None:
            peak_target = peak_target.view(-1)
            pred_peak = pred_peak.view(-1).clamp(1e-7, 1-1e-7)  # Prevent log(0)
            peak_loss = torch.mean(self.bce(pred_peak, peak_target.float()))
        else:
            peak_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Combine losses with absolute values to guarantee positivity
        total_loss = (
            torch.abs(huber_loss) + 
            self.alpha * torch.abs(uncertainty_loss) +
            self.beta * torch.abs(directional_loss) +
            self.peak_loss_weight * torch.abs(peak_loss)
        )
        
        return total_loss