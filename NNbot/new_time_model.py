import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


class ResidualLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, orig=None):
      return x + (orig if orig is not None else x)

class ImprovedTimeToPeakPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5,
             num_heads=8, use_cross_attention=True, max_seq_length=100,
             survival_prob=0.8):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        self.survival_prob = survival_prob
        
        # Learned positional encoding
        self.input_projection = nn.Linear(input_size, hidden_size)
    
        # Learned positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_size) / math.sqrt(hidden_size)
        )
        
        # 1. Temporal Feature Processing with enhanced blocks
        self.conv_block_5s = nn.ModuleList()
        self.projection_5s = nn.ModuleList()
        self.conv_block_10s = nn.ModuleList()
        self.projection_10s = nn.ModuleList()
        
        for _ in range(num_layers):
            conv_block_5s, proj_5s = self._create_enhanced_conv_block(input_size, hidden_size)
            self.conv_block_5s.append(conv_block_5s)
            self.projection_5s.append(proj_5s)
            
            conv_block_10s, proj_10s = self._create_enhanced_conv_block(input_size, hidden_size)
            self.conv_block_10s.append(conv_block_10s)
            self.projection_10s.append(proj_10s)

        # 2. Enhanced GRU blocks with normalization and residual connections
        self.gru_5s = self._create_enhanced_gru(hidden_size)
        self.gru_10s = self._create_enhanced_gru(hidden_size)
        self.gru_20s = self._create_enhanced_gru(input_size)
        self.gru_30s = self._create_enhanced_gru(input_size)

        # 3. Multi-scale attention mechanisms
        self.self_attention = nn.ModuleDict({
            'attention': nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            ),
            'norm': nn.LayerNorm(hidden_size)
        })
        
        if use_cross_attention:
            self.cross_attention = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(hidden_size)
            })

        # 4. Enhanced global feature processing
        self.global_fc = self._create_enhanced_global_processor(5, hidden_size)

        # 5. Quality gate with additional context
        self.quality_gate = self._create_enhanced_quality_gate(hidden_size)

        # 6. Enhanced feature fusion with skip connections
        self.fusion_layer = self._create_enhanced_fusion_layer(hidden_size)

        # 7. Enhanced prediction head with residual connections
        self.prediction_head = self._create_enhanced_prediction_head(hidden_size, dropout_rate)

        # 8. Feature importance scoring
        self.feature_importance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)

    def _create_enhanced_conv_block(self, input_size, hidden_size):
      """Enhanced CNN block with residual connections and normalization"""
      conv_block = nn.Sequential(
          nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
          nn.BatchNorm1d(hidden_size),
          nn.GELU(),
          nn.Dropout(self.dropout_rate),
          nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
          nn.BatchNorm1d(hidden_size),
          nn.GELU(),
          nn.Dropout(self.dropout_rate)
      )
      
      projection = nn.Sequential(
          nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
          nn.BatchNorm1d(hidden_size)
      )
      
      return conv_block, projection

    def _create_enhanced_gru(self, input_size):
        """Enhanced GRU with normalization and residual connections"""
        return nn.ModuleDict({
            'gru': nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size//2,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.dropout_rate if self.num_layers > 1 else 0
            ),
            'norm': nn.LayerNorm(self.hidden_size),
            'dropout': nn.Dropout(self.dropout_rate)
        })

    def _create_enhanced_global_processor(self, input_size, hidden_size):
      """Enhanced global feature processor with skip connections"""
      return nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.LayerNorm(hidden_size),
          nn.GELU(),
          nn.Dropout(self.dropout_rate),
          nn.Linear(hidden_size, hidden_size),
          nn.LayerNorm(hidden_size),
          ResidualLayer(hidden_size)
      )

   

    def _create_enhanced_quality_gate(self, hidden_size):
        """Enhanced quality gate with additional context"""
        return nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )

    def _create_enhanced_fusion_layer(self, hidden_size):
      """Enhanced fusion layer with skip connections"""
      return nn.Sequential(
          nn.Linear(hidden_size * 5, hidden_size * 2),
          nn.LayerNorm(hidden_size * 2),
          nn.GELU(),
          nn.Dropout(self.dropout_rate),
          nn.Linear(hidden_size * 2, hidden_size * 2),
          nn.LayerNorm(hidden_size * 2),
          ResidualLayer(hidden_size * 2)
      )

    def _create_enhanced_prediction_head(self, hidden_size, dropout_rate):
        """Enhanced prediction head with residual connections"""
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            ResidualLayer(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)
        )

    def _init_weights(self, module):
      """Enhanced weight initialization with scaled initialization"""
      if isinstance(module, (nn.Linear, nn.Conv1d)):
          # Use 'relu' as the closest supported nonlinearity
          nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
          if module.bias is not None:
              nn.init.zeros_(module.bias)
      elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
          nn.init.ones_(module.weight)
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.GRU):
          for name, param in module.named_parameters():
              if 'weight' in name:
                  nn.init.orthogonal_(param)
              elif 'bias' in name:
                  nn.init.zeros_(param)

    def _apply_stochastic_depth(self, x, training=True):
        """Apply stochastic depth during training"""
        if not training or self.survival_prob == 1.0:
            return x
        
        batch_size = x.size(0)
        random_tensor = torch.rand([batch_size, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor + self.survival_prob)
        return x * binary_tensor / self.survival_prob

    def _process_temporal_features(self, x, conv_block, projection, gru):
        """Process temporal features with checkpointing during training"""
        if self.training:
            return checkpoint(self._temporal_forward, x, conv_block, projection, gru)
        return self._temporal_forward(x, conv_block, projection, gru)

    def _temporal_forward(self, x, conv_block, projection, gru):
      """Forward pass for temporal features"""
      # Transpose for convolution (assuming x is already at hidden size)
      x = x.transpose(1, 2)
      
      # Apply projection and convolution with residual
      identity = x
      x = conv_block(x)
      x = x + identity
      
      # Transpose back and apply GRU
      x = x.transpose(1, 2)
      x, _ = gru['gru'](x)
      x = gru['norm'](x)
      x = gru['dropout'](x)
      
      return x

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
      # Move inputs to device
      inputs = [x_5s, x_10s, x_20s, x_30s, global_features, quality_features]
      inputs = [x.to(self.device) for x in inputs]
      x_5s, x_10s, x_20s, x_30s, global_features, quality_features = inputs
      
      # Project inputs to hidden size
      x_5s = self.input_projection(x_5s)
      x_10s = self.input_projection(x_10s)
      
      # Add positional encodings to temporal features
      x_5s = x_5s + self.positional_encoding[:, :x_5s.size(1)]
      x_10s = x_10s + self.positional_encoding[:, :x_10s.size(1)]
      
      # Process temporal features
      x_5s = self._process_temporal_features(x_5s, self.conv_block_5s[0], 
                                          self.projection_5s[0], self.gru_5s)
      x_10s = self._process_temporal_features(x_10s, self.conv_block_10s[0], 
                                            self.projection_10s[0], self.gru_10s)
      
      # Process longer sequences (assuming they're already projected)
      x_20s, _ = self.gru_20s['gru'](x_20s)
      x_30s, _ = self.gru_30s['gru'](x_30s)
      
      # Apply cross-attention if enabled
      if self.use_cross_attention:
          x_5s_10s = self.cross_attention['attention'](x_5s, x_10s, x_10s)[0]
          x_20s_30s = self.cross_attention['attention'](x_20s, x_30s, x_30s)[0]
          x_5s = self.cross_attention['norm'](x_5s + x_5s_10s)
          x_20s = self.cross_attention['norm'](x_20s + x_20s_30s)
      
      # Apply self-attention with skip connections
      temporal_features = [x_5s, x_10s, x_20s, x_30s]
      for i, x in enumerate(temporal_features):
          attn_out = self.self_attention['attention'](x, x, x)[0]
          temporal_features[i] = self.self_attention['norm'](x + attn_out)
      
      # Mean pooling to reduce to 2D
      temporal_features = [torch.mean(x, dim=1) for x in temporal_features]
      
      # Ensure global features is 2D and processed
      global_features = self.global_fc(global_features)
      
      # Ensure global_features is 2D (squeeze if needed)
      global_features = global_features.squeeze(0) if global_features.dim() > 2 else global_features
      
      # Calculate feature importance
      importance_scores = [self.feature_importance(x) for x in temporal_features]
      
      # Quality gating
      temporal_mean = torch.mean(torch.stack(temporal_features, dim=1), dim=1)
      quality_context = torch.cat([temporal_mean, quality_features], dim=1)
      quality_weights = self.quality_gate(quality_context)
      
      # Apply quality weights
      weighted_features = [x * quality_weights for x in temporal_features]
      
      # Combine features with global features
      combined = torch.cat(weighted_features + [global_features], dim=1)
      
      # Feature fusion
      fused_features = self.fusion_layer(combined)
      
      # Final prediction
      output = self.prediction_head(fused_features)
      mean, log_var = output.chunk(2, dim=1)
      
      if self.training:
          return {
              'mean': mean,
              'log_var': log_var,
              'importance_scores': importance_scores,
              'quality_weights': quality_weights
          }
      return mean

    def get_feature_importance(self):
        """Return the learned feature importance weights"""
        with torch.no_grad():
            return {
                '5s': self.feature_importance[0].weight.data,
                '10s': self.feature_importance[1].weight.data,
                '20s': self.feature_importance[2].weight.data,
                '30s': self.feature_importance[3].weight.data
            }
    
    def get_attention_weights(self):
        """Return the attention weights for visualization"""
        if not hasattr(self, '_attention_weights'):
            return None
        return self._attention_weights

    def compute_uncertainty(self, mean, log_var):
        """Compute the uncertainty estimates from the model outputs"""
        std = torch.exp(0.5 * log_var)
        return {
            'epistemic': std,  # Model uncertainty
            'total_uncertainty': mean.var(dim=0) + std.mean(dim=0)  # Total uncertainty
        }

    def get_intermediate_features(self, x_5s, x_10s, x_20s, x_30s):
        """Get intermediate feature representations for analysis"""
        with torch.no_grad():
            # Process temporal features
            x_5s_features = self._process_temporal_features(
                x_5s, self.conv_block_5s[0], self.projection_5s[0], self.gru_5s
            )
            x_10s_features = self._process_temporal_features(
                x_10s, self.conv_block_10s[0], self.projection_10s[0], self.gru_10s
            )
            x_20s_features, _ = self.gru_20s['gru'](x_20s)
            x_30s_features, _ = self.gru_30s['gru'](x_30s)

            return {
                '5s': x_5s_features,
                '10s': x_10s_features,
                '20s': x_20s_features,
                '30s': x_30s_features
            }

    def calculate_temporal_importance(self, x_5s, x_10s, x_20s, x_30s):
        """Calculate importance scores for each temporal scale"""
        with torch.no_grad():
            features = self.get_intermediate_features(x_5s, x_10s, x_20s, x_30s)
            importance_scores = {}
            
            for scale, feat in features.items():
                # Calculate feature-wise importance using attention mechanism
                query = self.self_attention['attention'].in_proj_weight[:self.hidden_size].view(
                    1, -1, self.hidden_size
                )
                attention_weights = F.softmax(
                    torch.bmm(query, feat.transpose(1, 2)) / math.sqrt(self.hidden_size),
                    dim=-1
                )
                importance_scores[scale] = attention_weights.mean(dim=1)

            return importance_scores

    def reset_quality_gate(self):
        """Reset the quality gate weights to default values"""
        for module in self.quality_gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def compute_gradients_norm(parameters):
        """Compute the norm of the gradients for monitoring training"""
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def freeze_temporal_processing(self):
        """Freeze the temporal processing layers for transfer learning"""
        for param in self.conv_block_5s.parameters():
            param.requires_grad = False
        for param in self.conv_block_10s.parameters():
            param.requires_grad = False
        for param in self.gru_5s.parameters():
            param.requires_grad = False
        for param in self.gru_10s.parameters():
            param.requires_grad = False
        for param in self.gru_20s.parameters():
            param.requires_grad = False
        for param in self.gru_30s.parameters():
            param.requires_grad = False

    def unfreeze_temporal_processing(self):
        """Unfreeze the temporal processing layers"""
        for param in self.conv_block_5s.parameters():
            param.requires_grad = True
        for param in self.conv_block_10s.parameters():
            param.requires_grad = True
        for param in self.gru_5s.parameters():
            param.requires_grad = True
        for param in self.gru_10s.parameters():
            param.requires_grad = True
        for param in self.gru_20s.parameters():
            param.requires_grad = True
        for param in self.gru_30s.parameters():
            param.requires_grad = True

    @staticmethod
    def gaussian_nll_loss(mean, log_var, target):
        """
        Compute negative log-likelihood loss with learned variance
        Args:
            mean: predicted mean
            log_var: predicted log variance
            target: ground truth values
        """
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (target - mean) ** 2 / var)
        return loss.mean()

    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self):
        """Estimate the model's memory usage"""
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        return {
            'parameters_memory': mem_params / 1024**2,  # MB
            'buffers_memory': mem_bufs / 1024**2,  # MB
            'total_memory': (mem_params + mem_bufs) / 1024**2  # MB
        }

    def save_feature_maps(self, x_5s, x_10s, x_20s, x_30s, save_path):
        """Save feature maps for visualization"""
        with torch.no_grad():
            features = self.get_intermediate_features(x_5s, x_10s, x_20s, x_30s)
            # Save features as numpy arrays
            for scale, feat in features.items():
                np.save(f"{save_path}/feature_maps_{scale}.npy", 
                       feat.cpu().numpy())

    def load_pretrained_temporal(self, state_dict_path):
        """Load pretrained weights for temporal processing layers"""
        state_dict = torch.load(state_dict_path)
        temporal_state_dict = {k: v for k, v in state_dict.items() 
                             if any(name in k for name in 
                                   ['conv_block', 'gru', 'projection'])}
        self.load_state_dict(temporal_state_dict, strict=False)
        
    def export_onnx(self, save_path, input_shapes):
        """Export model to ONNX format"""
        dummy_inputs = (
            torch.randn(input_shapes['x_5s']),
            torch.randn(input_shapes['x_10s']),
            torch.randn(input_shapes['x_20s']),
            torch.randn(input_shapes['x_30s']),
            torch.randn(input_shapes['global_features']),
            torch.randn(input_shapes['quality_features'])
        )
        torch.onnx.export(
            self,
            dummy_inputs,
            save_path,
            input_names=['x_5s', 'x_10s', 'x_20s', 'x_30s', 
                        'global_features', 'quality_features'],
            output_names=['predictions'],
            dynamic_axes={'x_5s': {0: 'batch_size'},
                        'x_10s': {0: 'batch_size'},
                        'x_20s': {0: 'batch_size'},
                        'x_30s': {0: 'batch_size'}}
        )