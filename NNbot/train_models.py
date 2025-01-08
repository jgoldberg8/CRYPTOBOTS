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

from model_utilities import AttentionModule, EarlyStopping, TimePredictionLoss, add_data_quality_features, clean_dataset, custom_market_cap_loss, train_val_split
from peak_market_cap_model import PeakMarketCapPredictor
from time_to_peak_model import TimeToPeakPredictor
from token_dataset import TokenDataset


    









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




def train_peak_market_cap_model(train_loader, val_loader, 
                               num_epochs=200, learning_rate=0.001, weight_decay=0.01, 
                               patience=15, min_delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(
        input_size=input_size,
        hidden_size=256,
        num_layers=4,
        dropout_rate=0.5
    ).to(device)

    # Initialize loss and optimizer
    criterion = custom_market_cap_loss
    optimizer = optim.AdamW(
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
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        # Training phase
        peak_market_cap_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = peak_market_cap_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    loss = criterion(output, batch['targets'][:, 0].unsqueeze(1))
                
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
                loss = criterion(output, batch['targets'][:, 0].unsqueeze(1))
                
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
                loss = criterion(output, batch['targets'][:, 0].unsqueeze(1))
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
            }, 'best_peak_market_cap_model.pth')

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    checkpoint = torch.load('best_peak_market_cap_model.pth')
    peak_market_cap_model.load_state_dict(checkpoint['model_state_dict'])
    
    return peak_market_cap_model, best_val_loss


def train_time_to_peak_model(train_loader, val_loader, 
                            num_epochs=200, learning_rate=0.001, weight_decay=0.01, 
                            patience=15, min_delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    input_size = 11
    time_to_peak_model = TimeToPeakPredictor(
        input_size=input_size,
        hidden_size=256,
        num_layers=3,
        dropout_rate=0.5
    ).to(device)

    # Initialize loss and optimizer
    criterion = TimePredictionLoss(alpha=0.3)
    optimizer = optim.AdamW(
        time_to_peak_model.parameters(),
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
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        # Training phase
        time_to_peak_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast('cuda'):
                    mean, log_var = time_to_peak_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'], batch['quality_features']
                    )
                    loss = criterion(mean, log_var, batch['targets'][:, 1].unsqueeze(1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(time_to_peak_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                mean, log_var = time_to_peak_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                loss = criterion(mean, log_var, batch['targets'][:, 1].unsqueeze(1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(time_to_peak_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], '
                      f'Loss: {loss.item():.4f}')
        
        train_loss /= num_batches

        # Validation phase
        time_to_peak_model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # During validation, we only care about the mean prediction
                mean = time_to_peak_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'], batch['quality_features']
                )
                # Use MSE for validation to track actual prediction error
                loss = F.mse_loss(mean, batch['targets'][:, 1].unsqueeze(1))
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
                'model_state_dict': time_to_peak_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            },'test.pth') #'best_time_to_peak_model.pth')

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    checkpoint = torch.load('test.pth')#best_time_to_peak_model.pth')
    time_to_peak_model.load_state_dict(checkpoint['model_state_dict'])
    
    return time_to_peak_model, best_val_loss




def main(model_to_train='both'):
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
    val_dataset = TokenDataset(val_df, scaler={'global': train_dataset.global_scaler, 
                                              'target': train_dataset.target_scaler})

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    results = {}

    try:
        # Train models based on input
        if model_to_train == 'peak_market_cap':
            print("Training Peak Market Cap Model...")
            peak_market_cap_model, val_loss = train_peak_market_cap_model(
                train_loader, val_loader
            )
            results['peak_market_cap_model'] = peak_market_cap_model
            results['peak_market_cap_val_loss'] = val_loss
            return results
            
        elif model_to_train == 'time_to_peak':
            print("Training Time to Peak Model...")
            time_to_peak_model, val_loss = train_time_to_peak_model(
                train_loader, val_loader
            )
            results['time_to_peak_model'] = time_to_peak_model
            results['time_to_peak_val_loss'] = val_loss
            return results
            
        else:  # train both
            print("Training Both Models...")
            print("\nTraining Peak Market Cap Model...")
            peak_market_cap_model, peak_market_cap_val_loss = train_peak_market_cap_model(
                train_loader, val_loader
            )
            results['peak_market_cap_model'] = peak_market_cap_model
            results['peak_market_cap_val_loss'] = peak_market_cap_val_loss
            
            print("\nTraining Time to Peak Model...")
            time_to_peak_model, time_to_peak_val_loss = train_time_to_peak_model(
                train_loader, val_loader
            )
            results['time_to_peak_model'] = time_to_peak_model
            results['time_to_peak_val_loss'] = time_to_peak_val_loss
            
            return results
            
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    # Default is to train both models
    model_to_train = 'both'
    
    # Check if a command-line argument is provided
    if len(sys.argv) > 1:
        model_to_train = sys.argv[1]
        if model_to_train not in ['both', 'peak_market_cap', 'time_to_peak']:
            print(f"Invalid model selection: {model_to_train}")
            print("Valid options are: 'both', 'peak_market_cap', 'time_to_peak'")
            sys.exit(1)
    
    try:
        results = main(model_to_train)
        print("\nTraining completed successfully!")
        print("Final validation losses:")
        if 'peak_market_cap_val_loss' in results:
            print(f"Peak Market Cap Model: {results['peak_market_cap_val_loss']:.4f}")
        if 'time_to_peak_val_loss' in results:
            print(f"Time to Peak Model: {results['time_to_peak_val_loss']:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)