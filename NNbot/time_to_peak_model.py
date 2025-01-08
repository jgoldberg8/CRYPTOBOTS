import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

from model_utilities import AttentionModule




class TimeToPeakPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # 1. Temporal Embeddings
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 2. CNN Blocks - Enhanced with residual connections
        self.conv_block_5s = self._create_conv_block(input_size, hidden_size)
        self.conv_block_10s = self._create_conv_block(input_size, hidden_size)
        
        # 3. GRU Layers - Better for variable sequences than LSTM
        gru_args = {
            'input_size': hidden_size,
            'hidden_size': hidden_size//2,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': True,
            'dropout': dropout_rate if num_layers > 1 else 0
        }
        
        # Create GRU layers for different time windows
        self.conv_block_5s = nn.ModuleList()
        self.projection_5s = nn.ModuleList()
        conv_block, projection = self._create_conv_block(input_size, hidden_size)
        self.conv_block_5s.append(conv_block)
        self.projection_5s.append(projection)

        self.conv_block_10s = nn.ModuleList()
        self.projection_10s = nn.ModuleList()
        conv_block, projection = self._create_conv_block(input_size, hidden_size)
        self.conv_block_10s.append(conv_block)
        self.projection_10s.append(projection)

        self.gru_5s = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size//2, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.gru_10s = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size//2, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # GRUs for longer sequences (directly from input)
        gru_args_long = dict(gru_args, input_size=input_size)
        self.gru_20s = nn.GRU(**gru_args_long)
        self.gru_30s = nn.GRU(**gru_args_long)

        # 4. Multi-head Self-attention Module
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 5. Advanced Quality Gate with Squeeze-Excitation
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )

        # 6. Global Feature Processing
        self.global_fc = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 7. Feature Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 8. Prediction Head with Uncertainty
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)  # Output: [mean, log_variance]
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)

    def _create_conv_block(self, input_size, hidden_size):
        """Helper method to create a CNN block with residual connections"""
        conv_block = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Create a projection layer if input and hidden sizes differ
        projection = (nn.Conv1d(input_size, hidden_size, kernel_size=1) 
                    if input_size != hidden_size 
                    else nn.Identity())
        
        return conv_block, projection

    def _init_weights(self, module):
        """Initialize weights using Kaiming/Xavier initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)



    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        # Move all inputs to device
        x_5s = x_5s.to(self.device)
        x_10s = x_10s.to(self.device)
        x_20s = x_20s.to(self.device)
        x_30s = x_30s.to(self.device)
        global_features = global_features.to(self.device)
        quality_features = quality_features.to(self.device)
        
        batch_size = x_5s.size(0)
        
        # Process 5-second windows with residual connections
        x_5s_res = x_5s.transpose(1, 2)
        conv_block = self.conv_block_5s[0]
        projection = self.projection_5s[0]
        
        # Apply projection first to match dimensions
        x_5s_projected = projection(x_5s_res)
        
        # Apply convolution
        x_5s_temp = conv_block(x_5s_res)
        
        # Add residual connection with projected input
        x_5s_res = x_5s_projected + x_5s_temp
        x_5s = x_5s_res.transpose(1, 2)
        x_5s, _ = self.gru_5s(x_5s)
        
        # Repeat similar modifications for x_10s block
        x_10s_res = x_10s.transpose(1, 2)
        conv_block = self.conv_block_10s[0]
        projection = self.projection_10s[0]
        
        x_10s_projected = projection(x_10s_res)
        x_10s_temp = conv_block(x_10s_res)
        x_10s_res = x_10s_projected + x_10s_temp
        x_10s = x_10s_res.transpose(1, 2)
        x_10s, _ = self.gru_10s(x_10s)
            
        # Process longer windows with GRU
        x_20s, _ = self.gru_20s(x_20s)
        x_30s, _ = self.gru_30s(x_30s)
        
        # Apply self-attention to all temporal features
        # 5s windows
        x_5s = x_5s.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x_5s, _ = self.self_attention(x_5s, x_5s, x_5s)
        x_5s = x_5s.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # 10s windows
        x_10s = x_10s.transpose(0, 1)
        x_10s, _ = self.self_attention(x_10s, x_10s, x_10s)
        x_10s = x_10s.transpose(0, 1)
        
        # 20s windows
        x_20s = x_20s.transpose(0, 1)
        x_20s, _ = self.self_attention(x_20s, x_20s, x_20s)
        x_20s = x_20s.transpose(0, 1)
        
        # 30s windows
        x_30s = x_30s.transpose(0, 1)
        x_30s, _ = self.self_attention(x_30s, x_30s, x_30s)
        x_30s = x_30s.transpose(0, 1)
        
        # Mean pooling for temporal features
        x_5s = torch.mean(x_5s, dim=1)   # [batch_size, hidden_size]
        x_10s = torch.mean(x_10s, dim=1)
        x_20s = torch.mean(x_20s, dim=1)
        x_30s = torch.mean(x_30s, dim=1)
        
        # Process global features
        global_features = self.global_fc(global_features)
        
        # Combine temporal features for quality weighting
        temporal_features = torch.stack([x_5s, x_10s, x_20s, x_30s], dim=1)
        temporal_mean = torch.mean(temporal_features, dim=1)
        
        # Apply quality gating
        quality_context = torch.cat([temporal_mean, quality_features], dim=1)
        quality_weights = self.quality_gate(quality_context)
        
        # Apply quality weights to features
        weighted_features = [
            x_5s * quality_weights,
            x_10s * quality_weights,
            x_20s * quality_weights,
            x_30s * quality_weights,
            global_features
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Feature fusion
        fused_features = self.fusion_layer(combined)
        
        # Generate prediction with uncertainty
        output = self.prediction_head(fused_features)
        mean, log_var = output.chunk(2, dim=1)
        
        if self.training:
            return mean, log_var
        else:
            return mean
    