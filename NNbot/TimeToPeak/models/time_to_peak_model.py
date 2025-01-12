import datetime
import json
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
import wandb
from torch.utils.data import Dataset, DataLoader

from TimeToPeak.datasets.time_token_dataset import MultiGranularTimeDataset, create_multi_granular_loaders
from TimeToPeak.utils.save_model_artifacts import save_model_artifacts
from TimeToPeak.utils.setup_logging import setup_logging
from TimeToPeak.utils.train_val_split import train_val_split
from TimeToPeak.utils.time_loss import RealTimePeakLoss
from TimeToPeak.utils.clean_dataset import clean_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from training_config.training_config import get_training_config

class CausalMultiGranularAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Create causal mask (can't look at future values)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        output = self.proj(context)
        return output, attention_weights

class RealTimeGranularityProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, window_size=60, num_heads=8, num_gru_layers=2, dropout_rate=0.5):
        super().__init__()
        
        self.window_size = window_size
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.attention = CausalMultiGranularAttention(hidden_size, num_heads=num_heads, dropout_rate=dropout_rate)
        
        # Changed to unidirectional GRU
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,  # No need to divide by 2 since it's not bidirectional
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if num_gru_layers > 1 else 0
        )
        
    def forward(self, x, lengths=None):
        # Apply sliding window if sequence is too long
        if x.size(1) > self.window_size:
            x = x[:, -self.window_size:]
            if lengths is not None:
                lengths = torch.clamp(lengths, max=self.window_size)
        
        x = self.feature_proj(x)
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        att_out, _ = self.attention(conv_out)
        
        if lengths is not None:
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths)
            if len(lengths.shape) > 1:
                lengths = lengths.squeeze()
            if len(lengths.shape) == 0:
                lengths = lengths.unsqueeze(0)
            
            # Pack sequence for GRU
            packed_x = nn.utils.rnn.pack_padded_sequence(
                att_out, 
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            gru_out, _ = self.gru(packed_x)
            output, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            output = self.gru(att_out)[0]
        
        return output

class RealTimePeakPredictor(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size=256,
                 window_size=60,
                 num_heads=8,
                 num_gru_layers=2,
                 dropout_rate=0.5,
                 confidence_threshold=0.8,
                 granularities=None):
        super().__init__()
        
        if granularities is None:
            granularities = ['5s', '10s', '20s', '30s', '60s']
        
        self.confidence_threshold = confidence_threshold
        
        # Processors for each granularity
        self.granularity_processors = nn.ModuleDict({
            gran: RealTimeGranularityProcessor(
                input_size=input_size,
                hidden_size=hidden_size,
                window_size=window_size,
                num_heads=num_heads,
                num_gru_layers=num_gru_layers,
                dropout_rate=dropout_rate
            ) for gran in granularities
        })
        
        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-granularity attention (causal)
        self.cross_attention = CausalMultiGranularAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Final prediction layers
        prediction_input_size = hidden_size * (len(granularities) + 1)
        
        self.predictor = nn.Sequential(
            nn.Linear(prediction_input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)  # Mean and log variance
        )
        
        # Peak detection head
        self.peak_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch, detect_peaks=True):

        # Process each granularity
        granularity_outputs = {}
        for granularity in self.granularity_processors.keys():
            features = batch[f'features_{granularity}']
            lengths = batch.get(f'length_{granularity}')
            
            if lengths is not None:
                if isinstance(lengths, dict):
                    lengths = lengths.get('lengths')
                lengths = lengths.squeeze() if hasattr(lengths, 'squeeze') else lengths
                lengths = torch.clamp(lengths, min=1) if isinstance(lengths, torch.Tensor) else np.maximum(lengths, 1)
            
            # Process this granularity
            processed = self.granularity_processors[granularity](features, lengths)
            
            # Use last state and ensure correct shape
            if processed.dim() == 3:
                last_state = processed[:, -1]  # Shape: [batch_size, hidden_size]
            else:
                last_state = processed  # Already [batch_size, hidden_size]
                
            granularity_outputs[granularity] = last_state
        
        # Process global features and ensure correct shape
        global_features = self.global_processor(batch['global_features'])
        if global_features.dim() == 3:
            global_features = global_features[:, -1]
        
        # Ensure all features have the same shape before stacking
        all_features = [global_features]
        for granularity in self.granularity_processors.keys():
            all_features.append(granularity_outputs[granularity])
        
        # Stack features along dimension 1
        combined_features = torch.stack(all_features, dim=1)  # Shape: [batch_size, num_features, hidden_size]
        
        # Apply cross-granularity attention
        attended_features, _ = self.cross_attention(combined_features)
        
        # Flatten and predict
        flattened = attended_features.reshape(attended_features.size(0), -1)
        output = self.predictor(flattened)
        
        # Split into mean and log variance
        mean, log_var = output.chunk(2, dim=-1)
        
        if detect_peaks:
            # Calculate peak probability
            peak_prob = self.peak_detector(attended_features[:, -1])
            confidence = torch.exp(-log_var)
            
            # Return peak prediction if confident
            peak_detected = (confidence > self.confidence_threshold) & (peak_prob > 0.9)
            
            return mean.squeeze(-1), log_var.squeeze(-1), peak_detected, peak_prob
        
        return mean.squeeze(-1), log_var.squeeze(-1)



def train_model(model, train_loader, val_loader,
                num_epochs=100,
                learning_rate=0.0005,
                weight_decay=0.02,
                patience=10,
                use_wandb=False,
                project_name="time_to_peak",
                checkpoint_dir='checkpoints'):
    """
    Enhanced training function with improved stability and monitoring
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss with stable weights
    criterion = RealTimePeakLoss(
        alpha=0.2,    # Uncertainty weight
        beta=0.15,    # Directional weight
        gamma=0.1,    # Regularization
        peak_loss_weight=0.3  # Peak detection weight
    )
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8  # Increased epsilon for better numerical stability
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Initialize automatic mixed precision
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch_metrics = None
    
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'train_mae': [],
        'val_mae': [],
        'best_epoch': 0,
        'peak_detection_accuracy': []
    }
    
    if use_wandb:
        wandb.init(project=project_name, config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "weight_decay": weight_decay,
            "model_type": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__
        })
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics = {
            'mse': [], 
            'mae': [], 
            'peak_accuracy': [],
            'peak_precision': [],
            'peak_recall': []
        }
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            # Move batch to device and handle possible missing keys
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            with autocast(device_type='cuda', enabled=True):
                mean, log_var, peak_detected, peak_prob = model(batch, detect_peaks=True)
                target = batch['targets'].view(-1)
                
                # Compute loss with proper error handling
                try:
                    loss = criterion(
                        mean, 
                        log_var, 
                        peak_prob, 
                        target,
                        peak_target=batch.get('peak_target')
                    )
                    
                    # Check for invalid loss values
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss value encountered: {loss.item()}")
                        continue
                    
                    mse = F.mse_loss(mean.view(-1), target)
                    mae = F.l1_loss(mean.view(-1), target)
                    
                except RuntimeError as e:
                    print(f"Error in loss computation: {str(e)}")
                    continue
                
                train_losses.append(loss.item())
                train_metrics['mse'].append(mse.item())
                train_metrics['mae'].append(mae.item())
                
                if 'peak_target' in batch:
                    peak_target = batch['peak_target'].view(-1)
                    peak_accuracy = ((peak_detected.view(-1) == peak_target).float().mean()).item()
                    train_metrics['peak_accuracy'].append(peak_accuracy)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimization step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Check for gradient problems
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Warning: NaN gradient in {name}")
        
        # Calculate training metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items() if len(v) > 0}
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = {
            'mse': [], 
            'mae': [], 
            'peak_accuracy': [],
            'peak_precision': [],
            'peak_recall': []
        }
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                mean, log_var, peak_detected, peak_prob = model(batch, detect_peaks=True)
                
                loss = criterion(
                    mean, 
                    log_var, 
                    peak_prob, 
                    batch['targets'],
                    peak_target=batch.get('peak_target')
                )
                
                mse = F.mse_loss(mean, batch['targets'])
                mae = F.l1_loss(mean, batch['targets'])
                
                if 'peak_target' in batch:
                    peak_accuracy = ((peak_detected == batch['peak_target']).float().mean()).item()
                    true_positives = ((peak_detected == 1) & (batch['peak_target'] == 1)).float().sum()
                    predicted_positives = (peak_detected == 1).float().sum()
                    actual_positives = (batch['peak_target'] == 1).float().sum()
                    
                    precision = (true_positives / predicted_positives).item() if predicted_positives > 0 else 0
                    recall = (true_positives / actual_positives).item() if actual_positives > 0 else 0
                    
                    val_metrics['peak_accuracy'].append(peak_accuracy)
                    val_metrics['peak_precision'].append(precision)
                    val_metrics['peak_recall'].append(recall)
                
                val_losses.append(loss.item())
                val_metrics['mse'].append(mse.item())
                val_metrics['mae'].append(mae.item())
                
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate validation metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items() if len(v) > 0}
        
        # Save metrics
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
        training_stats['train_mae'].append(avg_train_metrics['mae'])
        training_stats['val_mae'].append(avg_val_metrics['mae'])
        if 'peak_accuracy' in avg_val_metrics:
            training_stats['peak_detection_accuracy'].append(avg_val_metrics['peak_accuracy'])
        
        # Log to wandb if enabled
        if use_wandb:
            wandb_log = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mse": avg_train_metrics['mse'],
                "val_mse": avg_val_metrics['mse'],
                "train_mae": avg_train_metrics['mae'],
                "val_mae": avg_val_metrics['mae'],
                "learning_rate": scheduler.get_last_lr()[0]
            }
            for metric in ['peak_accuracy', 'peak_precision', 'peak_recall']:
                if metric in avg_train_metrics:
                    wandb_log[f"train_{metric}"] = avg_train_metrics[metric]
                if metric in avg_val_metrics:
                    wandb_log[f"val_{metric}"] = avg_val_metrics[metric]
            wandb.log(wandb_log)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train MAE: {avg_train_metrics['mae']:.4f}, Val MAE: {avg_val_metrics['mae']:.4f}")
        if 'peak_accuracy' in avg_train_metrics:
            print(f"Train Peak Accuracy: {avg_train_metrics['peak_accuracy']:.4f}, "
                  f"Val Peak Accuracy: {avg_val_metrics['peak_accuracy']:.4f}")
        
        # Check for improvement and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch_metrics = {
                'epoch': epoch,
                'train_metrics': avg_train_metrics,
                'val_metrics': avg_val_metrics
            }
            patience_counter = 0
            training_stats['best_epoch'] = epoch
            
            # Save best model checkpoint with all necessary information
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_val_loss,
                'training_stats': training_stats,
                'model_metrics': best_epoch_metrics,
                'model_config': {
                    'input_size': model.granularity_processors['5s'].feature_proj[0].in_features,
                    'hidden_size': model.granularity_processors['5s'].feature_proj[0].out_features,
                    'num_heads': model.granularity_processors['5s'].attention.num_heads,
                    'dropout_rate': model.granularity_processors['5s'].feature_proj[3].p,
                }
            }, checkpoint_path)
            
            # Save a backup of the best model
            backup_path = os.path.join(checkpoint_dir, f'best_model_backup_epoch_{epoch}.pt')
            shutil.copy2(checkpoint_path, backup_path)
        else:
            patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model at the end of training
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Clean up wandb
    if use_wandb:
        wandb.finish()
    
    return model, training_stats, best_val_loss

def create_data_loaders(train_df, val_df, batch_size=32, num_workers=4):
    """Create data loaders with proper error handling and monitoring"""
    try:
        # Create training dataset
        train_dataset = MultiGranularTimeDataset(train_df, train=True)
        
        # Create validation dataset with scalers from training
        val_dataset = MultiGranularTimeDataset(
            val_df,
            scaler=train_dataset.scalers,
            train=False
        )
        
        # Create data loaders with proper worker initialization
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        raise

def main():
    """Main training pipeline with improved error handling and logging"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training pipeline")
    
    try:
        # Create directory structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(current_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_path = 'data/time-data.csv'
        df = pd.read_csv(data_path)
        df = clean_dataset(df)  # Implement this based on your needs
        
        # Split data
        train_df, val_df = train_val_split(df, val_size=0.2)
        logger.info(f"Train set: {len(train_df)} samples, Val set: {len(val_df)} samples")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(train_df, val_df)
        logger.info("Data loaders created successfully")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealTimePeakPredictor(
            input_size=11,  # Adjust based on your feature size
            hidden_size=256,
            window_size=60,
            num_heads=8,
            num_gru_layers=2,
            dropout_rate=0.5
        ).to(device)
        logger.info(f"Model initialized on {device}")
        
        # Train model
        logger.info("Starting model training...")
        model, training_stats, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoints_dir
        )
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return model, training_stats
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()



def create_directory_structure():
    """Create the required directory structure in TimeToPeak"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    directories = [
        'checkpoints',
        'artifacts',
        'evaluation',
        'data',
        'logs'
    ]
    
    for directory in directories:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")    

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training pipeline")
    
    # Get optimized config
    config = get_training_config()
    model_config = config['model']
    training_config = config['training']
    
    # Get the TimeToPeak directory path (where the current file is)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories within TimeToPeak
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    artifacts_dir = os.path.join(current_dir, 'artifacts')
    data_path = os.path.join(current_dir, 'data', 'time-data.csv')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load and process data with corrected path
        logger.info("Loading data...")
        df = pd.read_csv(data_path)
        df = clean_dataset(df)
        logger.info(f"Data loaded and cleaned. Shape: {df.shape}")
        
        train_df, val_df = train_val_split(
            df,
            val_size=training_config['val_size'],
            random_state=training_config['random_state']
        )
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        # Create data loaders with optimized batch size
        train_loader, val_loader = create_multi_granular_loaders(
            train_df,
            val_df,
            batch_size=training_config['batch_size']
        )
        logger.info("Data loaders created")
        
        # Initialize model with optimized configuration
        model = RealTimePeakPredictor(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            window_size=model_config['window_size'],
            num_heads=model_config['num_heads'],
            num_gru_layers=model_config['num_gru_layers'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
        logger.info("Model initialized")
        
        # Train model
        logger.info("Starting model training...")
        model, training_stats, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=training_config['num_epochs'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            patience=training_config['patience'],
            use_wandb=True,
            project_name='time_to_peak',
            checkpoint_dir=checkpoints_dir
        )
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        # Save all model artifacts using the utility function
        save_dir = save_model_artifacts(
            model=model,
            train_loader=train_loader,
            training_stats=training_stats,
            config=config,
            save_dir=artifacts_dir,
            checkpoint_dir=checkpoints_dir
        )
        
        logger.info(f"Model artifacts saved to {save_dir}")
        logger.info("Training pipeline completed successfully")
        
        return {
            'model': model,
            'training_stats': training_stats,
            'best_val_loss': best_val_loss,
            'save_dir': save_dir
        }
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise



if __name__ == "__main__":
    # Create directory structure first
    create_directory_structure()
    # Then run the main function
    results = main()