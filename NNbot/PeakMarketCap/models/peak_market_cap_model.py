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

from PeakMarketCap.models.model_utilities import AttentionModule, RangeAttention, RangeStratifiedBatchSampler, clean_dataset, percentage_increase_loss

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

        # Enhanced CNN layers with residual connections
        self.conv_5s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualConvBlock(hidden_size),
            nn.MaxPool1d(2, padding=1)
        )

        self.conv_10s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualConvBlock(hidden_size),
            nn.MaxPool1d(2, padding=1)
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

        # Attention modules - keeping as direct AttentionModule
        self.attention_5s = AttentionModule(hidden_size)
        self.attention_10s = AttentionModule(hidden_size)
        self.attention_20s = AttentionModule(hidden_size)
        self.attention_30s = AttentionModule(hidden_size)

        # Range-specific attention modules - keeping as direct RangeAttention
        self.range_attention_5s = RangeAttention(hidden_size)
        self.range_attention_10s = RangeAttention(hidden_size)
        self.range_attention_20s = RangeAttention(hidden_size)
        self.range_attention_30s = RangeAttention(hidden_size)

        # Value range embedding
        self.value_range_embedding = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualLinearBlock(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        # Quality gate
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            ResidualLinearBlock(hidden_size),
            nn.Sigmoid()
        )

        # Global feature processing
        self.global_fc = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualLinearBlock(hidden_size)
        )

        # Final layers
        self.feature_reduction = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualLinearBlock(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            ResidualLinearBlock(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid() 
        )

        # Initialize weights
        self._initialize_weights()
        self.to(self.device)
        self._flatten_lstm_parameters()

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

    def _flatten_lstm_parameters(self):
        """Flatten LSTM parameters for better memory efficiency"""
        for name, module in self.named_modules():
            if isinstance(module, nn.LSTM):
                # Ensure parameters are on the same device as the module
                device = next(module.parameters()).device
                module.flatten_parameters()
                # Double-check if parameters need flattening again
                if not module._flat_weights_names:
                    # If parameters aren't flat, manually move them to contiguous memory
                    for weight in module._flat_weights:
                        if weight is not None:
                            weight.data = weight.data.contiguous()
                    module.flatten_parameters()

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        # Move inputs to device and process 5s data
        self._flatten_lstm_parameters()
        x_5s = x_5s.to(self.device)
        x_5s = self.conv_5s(x_5s.transpose(1, 2)).transpose(1, 2)
        x_5s, _ = self.lstm_5s(x_5s)
        x_5s = self.attention_5s(x_5s)  # [batch, 1, hidden]
        
        # Process 10s data
        x_10s = x_10s.to(self.device)
        x_10s = self.conv_10s(x_10s.transpose(1, 2)).transpose(1, 2)
        x_10s, _ = self.lstm_10s(x_10s)
        x_10s = self.attention_10s(x_10s)
        
        # Process 20s data
        x_20s = x_20s.to(self.device)
        x_20s, _ = self.lstm_20s(x_20s)
        x_20s = self.attention_20s(x_20s)
        
        # Process 30s data
        x_30s = x_30s.to(self.device)
        x_30s, _ = self.lstm_30s(x_30s)
        x_30s = self.attention_30s(x_30s)
        
        # Process global features and move to device
        global_features = global_features.to(self.device)
        value_range = self.value_range_embedding(global_features)  # [batch, hidden]
        global_features = self.global_fc(global_features)
        
        # Apply range attention
        x_5s = self.range_attention_5s(x_5s, value_range)  # Now outputs [batch, hidden]
        x_10s = self.range_attention_10s(x_10s, value_range)
        x_20s = self.range_attention_20s(x_20s, value_range)
        x_30s = self.range_attention_30s(x_30s, value_range)
        
        # Move quality features to device
        quality_features = quality_features.to(self.device)
        
        # Quality gating
        temporal_mean = torch.mean(torch.stack([x_5s, x_10s, x_20s, x_30s], dim=1), dim=1)
        quality_context = torch.cat([temporal_mean, quality_features], dim=1)
        quality_weights = self.quality_gate(quality_context)
        
        # Combine features
        weighted_features = [
            x_5s * quality_weights,
            x_10s * quality_weights,
            x_20s * quality_weights,
            x_30s * quality_weights,
            global_features,
            value_range
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Final prediction
        output = self.feature_reduction(combined)
        output = self.final_layers(output)
        
        return output

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Use same padding to maintain tensor size
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResidualLinearBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.fc2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)
        




def train_peak_market_cap_model(train_loader, val_loader, 
                               num_epochs=500,
                               hidden_size=512,
                               num_layers=3,
                               dropout_rate=0.3,
                               learning_rate=0.001,  # Increased base learning rate
                               weight_decay=1e-4,    # Increased weight decay
                               batch_size=32,        # Reduced batch size
                               accumulation_steps=4, # Increased accumulation steps
                               patience=20,          # Increased patience
                               min_delta=1e-5):      # Reduced min delta
    """
    Enhanced training function for improved model architecture
    """
    torch.backends.mkldnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print("\nDataset sizes:")
    print(f"Training samples: {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    print(f"Total samples: {train_size + val_size:,}\n")
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)
    peak_market_cap_model = peak_market_cap_model.to(memory_format=torch.channels_last)


    parameter_groups = [
        {'params': [p for n, p in peak_market_cap_model.named_parameters() if 'fc' in n or 'linear' in n],
         'lr': learning_rate},
        {'params': [p for n, p in peak_market_cap_model.named_parameters() if 'conv' in n],
         'lr': learning_rate * 0.1},
        {'params': [p for n, p in peak_market_cap_model.named_parameters() 
                   if not any(x in n for x in ['fc', 'linear', 'conv'])],
         'lr': learning_rate * 0.5}
    ]

    # Initialize optimizer with parameter groups
    optimizer = optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Enhanced learning rate scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs // accumulation_steps
    num_warmup_steps = num_training_steps // 10  # 10% warmup

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[learning_rate, learning_rate * 0.1, learning_rate * 0.5],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)//accumulation_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Initialize EMA model with improved beta
    ema = torch.optim.swa_utils.AveragedModel(peak_market_cap_model, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: 0.98 * averaged_model_parameter + 0.02 * model_parameter)
    
    # Early stopping with reduced min delta
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, restore_best_weights=True)
    best_val_loss = float('inf')
    
    # Initialize AMP with improved defaults
    scaler = torch.GradScaler(
        init_scale=2**14,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000
    )
    use_amp = torch.cuda.is_available()

    # Training loop
    for epoch in range(num_epochs):
        peak_market_cap_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
        
        # Learning rate warmup
        if epoch < 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (epoch + 1) / 3
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if use_amp:
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    output = peak_market_cap_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    
                    loss = percentage_increase_loss(output, batch['targets'])
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    # Increased gradient clipping threshold for deeper network
                    torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=5.0)
                    
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    scaler.update()
                    
                    
                    # Update EMA model with momentum scheduling
                    ema.update_parameters(peak_market_cap_model)
            else:
                output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                
                loss = percentage_increase_loss(output, batch['targets'])
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    
                    ema.update_parameters(peak_market_cap_model)
            
            train_loss += loss.item() * accumulation_steps
            
            # Periodic status update
            if (batch_idx + 1) % (accumulation_steps * 10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{num_batches}] '
                      f'Loss: {loss.item():.4f} '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        train_loss /= num_batches
        
        # Validation phase using EMA model
        val_metrics = calculate_metrics(ema.module, val_loader)
        train_metrics = calculate_metrics(ema.module, train_loader)
        
        # Print detailed metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Val RMSE: {val_metrics["rmse"]:.4f}')
        print(f'Train R²: {train_metrics["r2"]:.4f}, Val R²: {val_metrics["r2"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print('\nValidation Metrics by Range:')
        for range_name, metrics in val_metrics['range_metrics'].items():
            print(f'Range {range_name}:')
            print(f'  RMSE: {metrics["rmse"]:.4f}')
            print(f'  MAE: {metrics["mae"]:.4f}')
            print(f'  Count: {metrics["count"]}')
        
        # Save best model with additional metadata
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
                'hyperparameters': {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay
                }
            }, 'best_peak_market_cap_model.pth')
        
        if early_stopping(peak_market_cap_model, best_val_loss, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            early_stopping.restore(peak_market_cap_model)
            break
    
    # Load best model
    checkpoint = torch.load('best_peak_market_cap_model.pth')
    peak_market_cap_model.load_state_dict(checkpoint['model_state_dict'])
    
    return peak_market_cap_model, checkpoint['metrics']



def calculate_metrics(model, data_loader, device=None):
    """Calculate comprehensive metrics in original space
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on (defaults to model's device if None)
    """
    if device is None:
        device = next(model.parameters()).device
        
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
            
            # Get predictions and targets directly (no expm1 needed anymore)
            preds = output.cpu().numpy()
            targets = batch['targets'].cpu().numpy()
            
            # Inverse transform both predictions and targets
            preds = data_loader.dataset.target_scaler.inverse_transform(preds)
            targets = data_loader.dataset.target_scaler.inverse_transform(targets)
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Ensure predictions are within reasonable bounds
    all_preds = np.clip(all_preds, 0, 200)
    
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
                'count': np.sum(mask),
                'avg_prediction': np.mean(all_preds[mask]),
                'avg_target': np.mean(all_targets[mask])
            }
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'range_metrics': range_metrics,
        'predictions': all_preds,
        'targets': all_targets
    }

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
        batch_size=32, 
        sampler=sampler,
        pin_memory=True,
        num_workers=2
    )
    
    val_loader_peak = DataLoader(
        val_dataset_peak, 
        batch_size=32,
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