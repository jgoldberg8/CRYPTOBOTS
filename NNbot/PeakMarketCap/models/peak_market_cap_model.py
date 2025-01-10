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

from PeakMarketCap.models.model_utilities import clean_dataset, custom_market_cap_loss

from token_dataset import TokenDataset
from utils.train_val_split import train_val_split
from utils.add_data_quality_features import add_data_quality_features
from utils.attention_module import AttentionModule
from utils.early_stopping import EarlyStopping


class PeakMarketCapPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.device = torch.device('cpu')
        torch.backends.mkldnn.enabled = True
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Enhanced CNN layers
        self.conv_5s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.MaxPool1d(2)
        )


        self.conv_10s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.MaxPool1d(2)
        )

        # Bidirectional LSTM layers
        self.lstm_5s = nn.LSTM(hidden_size, hidden_size//2, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=self.dropout_rate if num_layers > 1 else 0)
        self.lstm_10s = nn.LSTM(hidden_size, hidden_size//2, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=self.dropout_rate if num_layers > 1 else 0)
        self.lstm_20s = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=self.dropout_rate if num_layers > 1 else 0)
        self.lstm_30s = nn.LSTM(input_size, hidden_size//2, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=self.dropout_rate if num_layers > 1 else 0)

        # Attention modules
        self.attention_5s = AttentionModule(hidden_size)
        self.attention_10s = AttentionModule(hidden_size)
        self.attention_20s = AttentionModule(hidden_size)
        self.attention_30s = AttentionModule(hidden_size)

        # Quality gatetokenda
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Global feature processing
        self.global_fc = nn.Linear(5, hidden_size)

        # Final layers
        self.fc1 = nn.Linear(hidden_size * 5, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single value for peak market cap

        # Initialize weights
        self._initialize_weights()

        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        # Move all inputs to device
        x_5s = x_5s.to(self.device)
        x_10s = x_10s.to(self.device)
        x_20s = x_20s.to(self.device)
        x_30s = x_30s.to(self.device)
        global_features = global_features.to(self.device)
        quality_features = quality_features.to(self.device)
        
        batch_size = x_5s.size(0)
        
        # Process 5-second windows
        x_5s = self.conv_5s(x_5s.transpose(1, 2))  # Apply CNN
        x_5s = x_5s.transpose(1, 2)  # Return to original shape
        x_5s, _ = self.lstm_5s(x_5s)  # Apply LSTM
        x_5s = self.attention_5s(x_5s)  # Apply attention
        
        # Process 10-second windows
        x_10s = self.conv_10s(x_10s.transpose(1, 2))
        x_10s = x_10s.transpose(1, 2)
        x_10s, _ = self.lstm_10s(x_10s)
        x_10s = self.attention_10s(x_10s)
        
        # Process 20-second windows (no CNN, direct LSTM)
        x_20s, _ = self.lstm_20s(x_20s)
        x_20s = self.attention_20s(x_20s)
        
        # Process 30-second windows (no CNN, direct LSTM)
        x_30s, _ = self.lstm_30s(x_30s)
        x_30s = self.attention_30s(x_30s)
        
        # Process global features
        global_features = self.global_fc(global_features)
        
        # Combine temporal features for quality weighting
        temporal_features = torch.stack([x_5s, x_10s, x_20s, x_30s], dim=1)
        temporal_mean = torch.mean(temporal_features, dim=1)
        
        # Apply quality gating mechanism
        quality_context = torch.cat([temporal_mean, quality_features], dim=1)
        if self.training or batch_size > 1:
            quality_weights = self.quality_gate(quality_context)
        else:
            # During single-sample evaluation, skip batch norm
            quality_weights = torch.sigmoid(self.quality_gate[0](quality_context))
        
        # Apply quality weights and combine features
        weighted_features = [
            x_5s * quality_weights,
            x_10s * quality_weights,
            x_20s * quality_weights,
            x_30s * quality_weights,
            global_features
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Final prediction layers
        output = self.fc1(combined)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output
    




def train_peak_market_cap_model(train_loader, val_loader, 
                               num_epochs=200, 
                               learning_rate=0.0003034232102344037, 
                               weight_decay= 7.79770403448178e-05, 
                               hidden_size=256,
                               num_layers=3,
                               dropout_rate=0.39683333144243493,
                               patience=29, 
                               min_delta=0.000544769124869796,
                               underprediction_penalty=2.0):
    torch.backends.mkldnn.enabled = True
    device = torch.device('cpu')    
    # Initialize model
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)
    peak_market_cap_model = peak_market_cap_model.to(memory_format=torch.channels_last)

    # Initialize optimizer
    optimizer = optim.Adam(
        peak_market_cap_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler with longer cycle
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Increased from 20
        T_mult=2,
        eta_min=1e-6
    )

    # Early stopping with more patience
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    
    # Initialize AMP
    use_amp = False
    scaler = None

    for epoch in range(num_epochs):
        # Training phase
        peak_market_cap_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast('cuda'):
                    output = peak_market_cap_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    loss = custom_market_cap_loss(
                        output, 
                        batch['targets'][:, 0].unsqueeze(1),
                        underprediction_penalty
                    )
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                loss = custom_market_cap_loss(
                    output, 
                    batch['targets'][:, 0].unsqueeze(1),
                    underprediction_penalty
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], '
                      f'Loss: {loss.item():.4f}')
        
        train_loss /= num_batches

        # Validation phase
        peak_market_cap_model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                loss = custom_market_cap_loss(
                    output, 
                    batch['targets'][:, 0].unsqueeze(1),
                    underprediction_penalty
                )
                val_loss += loss.item()

        val_loss /= val_batches
        scheduler.step()

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': peak_market_cap_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'underprediction_penalty': underprediction_penalty,  # Save the penalty value
            }, 'best_peak_market_cap_model.pth')

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    checkpoint = torch.load('best_peak_market_cap_model.pth')
    peak_market_cap_model.load_state_dict(checkpoint['model_state_dict'])
    
    return peak_market_cap_model, best_val_loss


def main():
    torch.backends.mkldnn.enabled = True
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"MKL Enabled: {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN Enabled: {torch.backends.mkldnn.is_available()}") 
    print(f"Using device: {device}")  
    
    # Load and preprocess data
    # Load and preprocess data
    # df_07 = pd.read_csv('data/testData.csv')
    # df_07 = clean_dataset(df_07)
    # df_07 = add_data_quality_features(df_07)

    # df_08 = pd.read_csv('data/token_data_2025-01-08.csv')
    # df_08 = clean_dataset(df_08)
    # df_08 = add_data_quality_features(df_08)
    df = pd.read_csv('data/higher-peak-data.csv')
    df = clean_dataset(df)
    df = add_data_quality_features(df)

    # df = pd.concat([df_07, df_08], ignore_index=True)

    # Split data using stratified sampling
    train_df, val_df = train_val_split(df)

    # Create datasets
    train_dataset_peak = TokenDataset(train_df)
    val_dataset_peak = TokenDataset(val_df, scaler={
        'global': train_dataset_peak.global_scaler,
        'target': train_dataset_peak.target_scaler
    })


    train_loader_peak = DataLoader(train_dataset_peak, batch_size=128, shuffle=True)
    val_loader_peak = DataLoader(val_dataset_peak, batch_size=128)


    results = {}

    try:
        print("Training Peak Market Cap Model...")
        peak_market_cap_model, val_loss = train_peak_market_cap_model(
            train_loader_peak, val_loader_peak
        )
        results['peak_market_cap_model'] = peak_market_cap_model
        results['peak_market_cap_val_loss'] = val_loss
        
        
        return results
            
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()