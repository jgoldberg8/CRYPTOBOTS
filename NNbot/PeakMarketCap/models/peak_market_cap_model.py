import os
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

from PeakMarketCap.models.model_utilities import AttentionModule, RangeAttention, clean_dataset, custom_market_cap_loss

from PeakMarketCap.models.token_dataset import TokenDataset
from utils.train_val_split import train_val_split
from utils.add_data_quality_features import add_data_quality_features
from utils.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import WeightedRandomSampler
import torch.cuda.amp as amp
import joblib






class PeakMarketCapPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        # Range-specific attention modules
        self.range_attention_5s = RangeAttention(hidden_size)
        self.range_attention_10s = RangeAttention(hidden_size)
        self.range_attention_20s = RangeAttention(hidden_size)
        self.range_attention_30s = RangeAttention(hidden_size)
        # Value range embedding
        self.value_range_embedding = nn.Sequential(
        nn.Linear(6, hidden_size),  # Use all global features
        nn.BatchNorm1d(hidden_size),  # Add batch norm for better training
        nn.ReLU(),
        nn.Dropout(dropout_rate),  # Add dropout for regularization
        nn.Linear(hidden_size, hidden_size)
        )

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
        self.global_fc = nn.Linear(6, hidden_size)

        # Final layers
        self.fc1 = nn.Linear(hidden_size * 6, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single value for peak market cap

        # Initialize weights
        self.gradient_clip_val = 1.0
        self._initialize_weights()

        self.to(self.device)
        self.lstm_5s.flatten_parameters()
        self.lstm_10s.flatten_parameters()
        self.lstm_20s.flatten_parameters()
        self.lstm_30s.flatten_parameters()

       

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        # Move all inputs to device
        def check_tensor(name, tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"NaN detected in {name}")
                tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.zeros_like(tensor), 
                               tensor)
                return tensor
            return False
        x_5s = x_5s.to(self.device)
        check_tensor("conv_5s", x_5s)
        x_10s = x_10s.to(self.device)
        x_20s = x_20s.to(self.device)
        x_30s = x_30s.to(self.device)
        global_features = global_features.to(self.device)
        quality_features = quality_features.to(self.device)
        
        batch_size = x_5s.size(0)
        
        # Estimate value range from initial market cap feature
        value_range = self.value_range_embedding(global_features)
        
        # Process 5-second windows
        x_5s = self.conv_5s(x_5s.transpose(1, 2))  # (batch, features, seq) for CNN
        x_5s = x_5s.transpose(1, 2)  # Back to (batch, seq, features) for LSTM
        x_5s, _ = self.lstm_5s(x_5s)  
        check_tensor("lstm_5s", x_5s)
        x_5s = self.range_attention_5s(x_5s, value_range)
        check_tensor("attention_5s", x_5s)
        
        # Process 10-second windows
        x_10s = self.conv_10s(x_10s.transpose(1, 2))
        x_10s = x_10s.transpose(1, 2)
        x_10s, _ = self.lstm_10s(x_10s)
        x_10s = self.range_attention_10s(x_10s, value_range)
        
        # Process 20-second windows (no CNN, direct LSTM)
        x_20s, _ = self.lstm_20s(x_20s)
        x_20s = self.range_attention_20s(x_20s, value_range)
        
        # Process 30-second windows (no CNN, direct LSTM)
        x_30s, _ = self.lstm_30s(x_30s)
        x_30s = self.range_attention_30s(x_30s, value_range)
        
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
            global_features,
            value_range  # Add value range to final feature set
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Final prediction layers
        output = self.fc1(combined)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output
        




def train_peak_market_cap_model(train_loader, val_loader, 
                               num_epochs=500,
                               hidden_size=1024,
                               num_layers=4,
                               dropout_rate=0.4,
                               learning_rate=0.0006,
                               weight_decay=1e-5,
                               batch_size=48,
                               accumulation_steps=3,
                               patience=35,
                               min_delta=5e-5):
    """
    Training function with comprehensive metrics calculation
    """
    torch.backends.mkldnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)//accumulation_steps
    )

    # Initialize EMA model for stable validation
    ema = torch.optim.swa_utils.AveragedModel(peak_market_cap_model)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    
    # Initialize AMP
    scaler = torch.GradScaler()
    use_amp = torch.cuda.is_available()
    
    def calculate_metrics(model, data_loader):
        """Calculate comprehensive metrics in original space"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                
                # Convert from log space back to original space
                preds = torch.expm1(output).cpu().numpy()
                targets = torch.expm1(batch['targets']).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Calculate various metrics
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # Calculate range-specific metrics
        ranges = [(0, 100), (100, 500), (500, float('inf'))]
        range_metrics = {}
        
        for low, high in ranges:
            mask = (all_targets >= low) & (all_targets < high)
            if np.any(mask):
                range_metrics[f'{low}-{high}'] = {
                    'rmse': np.sqrt(mean_squared_error(all_targets[mask], all_preds[mask])),
                    'mae': mean_absolute_error(all_targets[mask], all_preds[mask]),
                    'count': np.sum(mask)
                }
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'range_metrics': range_metrics
        }

    # Training loop
    for epoch in range(num_epochs):
        peak_market_cap_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if use_amp:
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    output = peak_market_cap_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    
                    loss = custom_market_cap_loss(output, batch['targets'])
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()
                    
                    ema.update_parameters(peak_market_cap_model)
            else:
                output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                
                loss = custom_market_cap_loss(output, batch['targets'])
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    ema.update_parameters(peak_market_cap_model)
            
            train_loss += loss.item() * accumulation_steps
        
        train_loss /= num_batches
        
        # Validation phase using EMA model
        val_metrics = calculate_metrics(ema.module, val_loader)
        train_metrics = calculate_metrics(ema.module, train_loader)
        
        # Print detailed metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Val RMSE: {val_metrics["rmse"]:.4f}')
        print(f'Train R²: {train_metrics["r2"]:.4f}, Val R²: {val_metrics["r2"]:.4f}')
        print('\nValidation Metrics by Range:')
        for range_name, metrics in val_metrics['range_metrics'].items():
            print(f'Range {range_name}:')
            print(f'  RMSE: {metrics["rmse"]:.4f}')
            print(f'  MAE: {metrics["mae"]:.4f}')
            print(f'  Count: {metrics["count"]}')
        
        # Save best model based on validation RMSE
        if val_metrics['rmse'] < best_val_loss:
            best_val_loss = val_metrics['rmse']
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss,
                'metrics': val_metrics,
            }, 'best_peak_market_cap_model.pth')
        
        # Early stopping check
        if early_stopping(val_metrics['rmse']):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load('best_peak_market_cap_model.pth')
    peak_market_cap_model.load_state_dict(checkpoint['model_state_dict'])
    
    return peak_market_cap_model, checkpoint['metrics']


def save_scalers(train_dataset, output_dir='scalers'):
    """
    Save global and target scalers from the dataset.
    
    Args:
        train_dataset (TokenDataset): The training dataset containing scalers
        output_dir (str, optional): Directory to save scalers. Defaults to 'scalers'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save global scaler
    if hasattr(train_dataset, 'global_scaler'):
        joblib.dump(train_dataset.global_scaler, 
                    os.path.join(output_dir, 'global_scaler.joblib'))
    
    # Save target scaler
    if hasattr(train_dataset, 'target_scaler'):
        joblib.dump(train_dataset.target_scaler, 
                    os.path.join(output_dir, 'target_scaler.joblib'))

def load_scalers(scaler_dir='scalers'):
    """
    Load previously saved scalers.
    
    Args:
        scaler_dir (str, optional): Directory where scalers are saved. Defaults to 'scalers'.
    
    Returns:
        dict: A dictionary containing loaded scalers
    """
    scalers = {}
    
    # Try to load global scaler
    global_scaler_path = os.path.join(scaler_dir, 'global_scaler.joblib')
    if os.path.exists(global_scaler_path):
        scalers['global'] = joblib.load(global_scaler_path)
    
    # Try to load target scaler
    target_scaler_path = os.path.join(scaler_dir, 'target_scaler.joblib')
    if os.path.exists(target_scaler_path):
        scalers['target'] = joblib.load(target_scaler_path)
    
    return scalers


def main():
    torch.backends.mkldnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"MKL Enabled: {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN Enabled: {torch.backends.mkldnn.is_available()}")
    
    # Load and preprocess data
    df = pd.read_csv('data/new-token-data.csv')
    df = clean_dataset(df)
    df = add_data_quality_features(df)

    # Split data using stratified sampling
    train_df, val_df = train_val_split(df)

    # Create datasets
    train_dataset_peak = TokenDataset(train_df)
    save_scalers(train_dataset_peak)
    val_dataset_peak = TokenDataset(val_df, scaler={
        'global': train_dataset_peak.global_scaler,
        'target': train_dataset_peak.target_scaler
    })

    # Calculate sample weights for training data
    weights = train_dataset_peak._calculate_sample_weights(train_df)
   
    sampler = WeightedRandomSampler(weights, len(weights))

    # Create data loaders with weighted sampling for training
    train_loader_peak = DataLoader(
        train_dataset_peak, 
        batch_size=48, 
        sampler=sampler,
        pin_memory=True,
        num_workers=2
    )
    
    val_loader_peak = DataLoader(
        val_dataset_peak, 
        batch_size=48,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    results = {}

    try:
        print("Training Peak Market Cap Model...")
        peak_market_cap_model, val_loss = train_peak_market_cap_model(
            train_loader_peak, val_loader_peak
        )
        results['peak_market_cap_model'] = peak_market_cap_model
        results['peak_market_cap_val_loss'] = val_loss
        
        # Calculate and print final metrics
        peak_market_cap_model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader_peak:
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['targets'].cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        print("\nFinal Model Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return results
            
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()