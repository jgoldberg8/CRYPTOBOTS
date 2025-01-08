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

from model_utilities import AttentionModule, EarlyStopping, TimePredictionLoss, TokenDataset, add_data_quality_features, clean_dataset, custom_market_cap_loss, train_val_split
from peak_market_cap_model import PeakMarketCapPredictor
from time_to_peak_model import TimeToPeakPredictor


    









def train_model(peak_market_cap_model, time_to_peak_model, train_loader, val_loader, 
                num_epochs=200, learning_rate=0.001, weight_decay=0.01, 
                patience=15, min_delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peak_market_cap_model = peak_market_cap_model.to(device)
    time_to_peak_model = time_to_peak_model.to(device)

    # Initialize loss functions
    peak_market_cap_criterion = custom_market_cap_loss
    time_to_peak_criterion = TimePredictionLoss(alpha=0.3)

    # Optimizers with AdamW
    peak_market_cap_optimizer = optim.AdamW(
        peak_market_cap_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    time_to_peak_optimizer = optim.AdamW(
        time_to_peak_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate schedulers
    peak_market_cap_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        peak_market_cap_optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    time_to_peak_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        time_to_peak_optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )

    # Early stopping
    peak_market_cap_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    time_to_peak_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    best_peak_market_cap_val_loss = float('inf')
    best_time_to_peak_val_loss = float('inf')
    
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(num_epochs):
        # Training phase for peak market cap model
        peak_market_cap_model.train()
        peak_market_cap_train_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            peak_market_cap_optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
                    peak_market_cap_output = peak_market_cap_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    peak_market_cap_loss = peak_market_cap_criterion(
                        peak_market_cap_output,
                        batch['targets'][:, 0].unsqueeze(1)
                    )
                
                scaler.scale(peak_market_cap_loss).backward()
                scaler.unscale_(peak_market_cap_optimizer)
                torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                scaler.step(peak_market_cap_optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                peak_market_cap_output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                peak_market_cap_loss = peak_market_cap_criterion(
                    peak_market_cap_output,
                    batch['targets'][:, 0].unsqueeze(1)
                )
                
                peak_market_cap_loss.backward()
                torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
                peak_market_cap_optimizer.step()
            
            peak_market_cap_train_loss += peak_market_cap_loss.item()
        
        peak_market_cap_train_loss /= len(train_loader)

        # Training phase for time to peak model
        time_to_peak_model.train()
        time_to_peak_train_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            time_to_peak_optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
                    mean, log_var = time_to_peak_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    time_to_peak_loss = time_to_peak_criterion(
                        mean,
                        log_var,
                        batch['targets'][:, 1].unsqueeze(1)
                    )
                    
                scaler.scale(time_to_peak_loss).backward()
                scaler.unscale_(time_to_peak_optimizer)
                torch.nn.utils.clip_grad_norm_(time_to_peak_model.parameters(), max_norm=1.0)
                scaler.step(time_to_peak_optimizer)
                scaler.update()

            else:
                # Standard training for CPU
                mean, log_var = time_to_peak_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                time_to_peak_loss = time_to_peak_criterion(
                    mean,
                    log_var,
                    batch['targets'][:, 1].unsqueeze(1)
                )
            
            time_to_peak_train_loss += time_to_peak_loss.item()
            
        time_to_peak_train_loss /= len(train_loader)

        # Validation phase
        peak_market_cap_model.eval()
        time_to_peak_model.eval()
        peak_market_cap_val_loss = 0
        time_to_peak_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Peak market cap validation
                peak_market_cap_output = peak_market_cap_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                peak_market_cap_loss = peak_market_cap_criterion(
                    peak_market_cap_output,
                    batch['targets'][:, 0].unsqueeze(1)
                )
                peak_market_cap_val_loss += peak_market_cap_loss.item()
                
                # Time to peak validation
                mean = time_to_peak_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                # During validation, we only care about the mean prediction
                time_to_peak_loss = F.mse_loss(mean, batch['targets'][:, 1].unsqueeze(1))
                time_to_peak_val_loss += time_to_peak_loss.item()

        peak_market_cap_val_loss /= len(val_loader)
        time_to_peak_val_loss /= len(val_loader)

        # Update learning rates
        peak_market_cap_scheduler.step()
        time_to_peak_scheduler.step()

        # Print progress
        print(f'Epoch {epoch+1}:')
        print(f'Peak Market Cap - Train Loss: {peak_market_cap_train_loss:.4f}, '
              f'Val Loss: {peak_market_cap_val_loss:.4f}, '
              f'LR: {peak_market_cap_scheduler.get_last_lr()[0]:.6f}')
        print(f'Time to Peak - Train Loss: {time_to_peak_train_loss:.4f}, '
              f'Val Loss: {time_to_peak_val_loss:.4f}, '
              f'LR: {time_to_peak_scheduler.get_last_lr()[0]:.6f}')

        # Save best models
        if peak_market_cap_val_loss < best_peak_market_cap_val_loss:
            best_peak_market_cap_val_loss = peak_market_cap_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': peak_market_cap_model.state_dict(),
                'optimizer_state_dict': peak_market_cap_optimizer.state_dict(),
                'scheduler_state_dict': peak_market_cap_scheduler.state_dict(),
                'best_val_loss': best_peak_market_cap_val_loss,
            }, 'best_peak_market_cap_model.pth')

        if time_to_peak_val_loss < best_time_to_peak_val_loss:
            best_time_to_peak_val_loss = time_to_peak_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': time_to_peak_model.state_dict(),
                'optimizer_state_dict': time_to_peak_optimizer.state_dict(),
                'scheduler_state_dict': time_to_peak_scheduler.state_dict(),
                'best_val_loss': best_time_to_peak_val_loss,
            }, 'best_time_to_peak_model.pth')

        # Early stopping
        if (peak_market_cap_early_stopping(peak_market_cap_val_loss) or 
            time_to_peak_early_stopping(time_to_peak_val_loss)):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    peak_market_cap_checkpoint = torch.load('best_peak_market_cap_model.pth')
    time_to_peak_checkpoint = torch.load('best_time_to_peak_model.pth')
    
    peak_market_cap_model.load_state_dict(peak_market_cap_checkpoint['model_state_dict'])
    time_to_peak_model.load_state_dict(time_to_peak_checkpoint['model_state_dict'])
    
    return {
        'peak_market_cap_model': peak_market_cap_model,
        'time_to_peak_model': time_to_peak_model,
        'peak_market_cap_val_loss': best_peak_market_cap_val_loss,
        'time_to_peak_val_loss': best_time_to_peak_val_loss,
        'peak_market_cap_epoch': peak_market_cap_checkpoint['epoch'],
        'time_to_peak_epoch': time_to_peak_checkpoint['epoch']
    }
        


            




def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df_07 = pd.read_csv('data/token_data_2025-01-07.csv')
    df_07 = clean_dataset(df_07)
    df_07 = add_data_quality_features(df_07)

    df_08 = pd.read_csv('data/token_data_2025-01-08.csv')
    df_08 = clean_dataset(df_08)
    df_08 = add_data_quality_features(df_08)

    df = pd.concat([df_07, df_08], ignore_index=True)

    # Split data using stratified sampling
    train_df, val_df = train_val_split(df)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    # In your main function, modify this line:
    val_dataset = TokenDataset(val_df, scaler={'global': train_dataset.global_scaler, 
                                          'target': train_dataset.target_scaler})

    # Create data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Initialize models with updated input size and other improvements
    input_size = 11  # Base feature size remains the same
    hidden_size = 256
    num_layers = 3
    dropout_rate = 0.4  # Slightly reduced dropout

    peak_market_cap_model = PeakMarketCapPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    
    time_to_peak_model = TimeToPeakPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    # Train models with updated parameters
    training_results = train_model(
        peak_market_cap_model,
        time_to_peak_model,
        train_loader,
        val_loader,
        num_epochs=300,  # Increased epochs
        learning_rate=0.0005,  # Reduced learning rate
        weight_decay=0.02,  # Increased weight decay
        patience=20,  # Increased patience
        min_delta=0.0005  # Reduced min delta
    )
    
    # Print training results
    print("\nTraining Results:")
    print(f"Peak Market Cap Model:")
    print(f"  Best Validation Loss: {training_results['peak_market_cap_val_loss']:.4f}")
    print(f"  Achieved at epoch: {training_results['peak_market_cap_epoch']}")
    print(f"Time to Peak Model:")
    print(f"  Best Validation Loss: {training_results['time_to_peak_val_loss']:.4f}")
    print(f"  Achieved at epoch: {training_results['time_to_peak_epoch']}")
    
    return training_results

if __name__ == "__main__":
    main()