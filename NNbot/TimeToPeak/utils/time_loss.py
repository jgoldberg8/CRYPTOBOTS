import torch
import torch.nn as nn
import torch.nn.functional as F

class TimePredictionLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.15):
        super().__init__()
        self.alpha = alpha  # Weight for uncertainty loss
        self.beta = beta    # Weight for directional loss
        self.gamma = gamma  # Weight for temporal coherence
        self.huber = nn.SmoothL1Loss(beta=1.0)  # More robust than MSE
        
    def forward(self, pred_mean, pred_log_var, target):
        # Clamp pred_log_var for numerical stability
        target = target.squeeze()
        pred_mean = pred_mean.squeeze()
        pred_log_var = pred_log_var.squeeze()
        pred_log_var = torch.clamp(pred_log_var, min=-10, max=10)
        
        # 1. Huber loss for base prediction (more robust than MSE)
        huber_loss = self.huber(pred_mean, target)
        
        # 2. Enhanced uncertainty loss with calibration
        precision = torch.exp(-pred_log_var) + 1e-6
        squared_error = (pred_mean - target)**2
        uncertainty_loss = 0.5 * torch.mean(
            pred_log_var + squared_error * precision - 0.5 * torch.log(precision)
        )
        
        # 3. Asymmetric relative error (penalize differently based on prediction direction)
        abs_diff = torch.abs(pred_mean - target)
        rel_error = abs_diff / (torch.abs(target) + 1e-6)
        
        # 4. Enhanced directional penalty with dynamic scaling
        pred_error = pred_mean - target
        late_mask = pred_error > 0
        early_mask = pred_error <= 0
        
        # Scale penalties based on prediction magnitude
        late_scale = 1.5 + 0.5 * torch.sigmoid(pred_error[late_mask] / target[late_mask])
        early_scale = 0.8 - 0.2 * torch.sigmoid(-pred_error[early_mask] / target[early_mask])
        
        directional_penalty = torch.zeros_like(pred_error)
        directional_penalty[late_mask] = pred_error[late_mask] * late_scale
        directional_penalty[early_mask] = -pred_error[early_mask] * early_scale
        directional_loss = torch.mean(directional_penalty)
        
        # 5. Temporal coherence loss (penalize rapid changes in uncertainty)
        if len(pred_log_var.shape) > 1:
            temporal_coherence = F.mse_loss(
                pred_log_var[:, 1:], 
                pred_log_var[:, :-1]
            )
        else:
            temporal_coherence = torch.tensor(0.0, device=pred_log_var.device)
        
        # Combine all terms with learned confidence-based weighting
        confidence = torch.sigmoid(-pred_log_var)  # Higher confidence -> lower variance
        weighted_huber = huber_loss * confidence.mean()
        weighted_uncertainty = self.alpha * uncertainty_loss * (1 - confidence.mean())
        
        total_loss = (
            weighted_huber +
            weighted_uncertainty +
            self.beta * directional_loss +
            self.gamma * temporal_coherence
        )
        
        return total_loss
    
    def get_components(self, pred_mean, pred_log_var, target):
        """Returns individual loss components for analysis"""
        huber_loss = self.huber(pred_mean, target)
        
        precision = torch.exp(-pred_log_var) + 1e-6
        squared_error = (pred_mean - target)**2
        uncertainty_loss = 0.5 * torch.mean(
            pred_log_var + squared_error * precision - 0.5 * torch.log(precision)
        )
        
        pred_error = pred_mean - target
        late_mask = pred_error > 0
        early_mask = pred_error <= 0
        
        late_scale = 1.5 + 0.5 * torch.sigmoid(pred_error[late_mask] / target[late_mask])
        early_scale = 0.8 - 0.2 * torch.sigmoid(-pred_error[early_mask] / target[early_mask])
        
        directional_penalty = torch.zeros_like(pred_error)
        directional_penalty[late_mask] = pred_error[late_mask] * late_scale
        directional_penalty[early_mask] = -pred_error[early_mask] * early_scale
        directional_loss = torch.mean(directional_penalty)
        
        if len(pred_log_var.shape) > 1:
            temporal_coherence = F.mse_loss(
                pred_log_var[:, 1:], 
                pred_log_var[:, :-1]
            )
        else:
            temporal_coherence = torch.tensor(0.0, device=pred_log_var.device)
            
        return {
            'huber_loss': huber_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'directional_loss': directional_loss.item(),
            'temporal_coherence': temporal_coherence.item()
        }