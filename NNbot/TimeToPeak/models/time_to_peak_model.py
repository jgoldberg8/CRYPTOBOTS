import datetime
import os
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

from TimeToPeak.datasets.time_token_dataset import create_multi_granular_loaders
from TimeToPeak.utils import save_model_artifacts
from TimeToPeak.utils.setup_logging import setup_logging
from TimeToPeak.utils.train_val_split import train_val_split
from TimeToPeak.utils.time_loss import TimePredictionLoss
from TimeToPeak.utils.clean_dataset import clean_dataset

class MultiGranularAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        output = self.proj(context)
        return output, attention_weights

class GranularityProcessor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super().__init__()
        
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
        
        self.attention = MultiGranularAttention(hidden_size)
        
        self.gru = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
    def forward(self, x, lengths=None):
        # Project features
        x = self.feature_proj(x)
        
        # Apply temporal convolution
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Apply attention
        att_out, _ = self.attention(conv_out)
        
        # Process with GRU
        if lengths is not None:
            # Ensure lengths is a 1D tensor and on CPU
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths)
            if len(lengths.shape) > 1:
                lengths = lengths.squeeze()
            if len(lengths.shape) == 0:
                lengths = lengths.unsqueeze(0)
            
            # Filter out zero-length sequences
            valid_mask = lengths > 0
            if not valid_mask.any():
                # If all sequences are invalid, just return mean pooled output
                return att_out.mean(dim=1).unsqueeze(1)
                
            # Get only valid sequences
            valid_att_out = att_out[valid_mask]
            valid_lengths = lengths[valid_mask]
            
            # Sort sequences by length for packing
            valid_lengths = valid_lengths.cpu()
            sorted_lengths, sorted_idx = valid_lengths.sort(0, descending=True)
            sorted_att_out = valid_att_out[sorted_idx]
            
            # Pack sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                sorted_att_out, 
                sorted_lengths.cpu().numpy(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # Process with GRU
            gru_out, _ = self.gru(packed_x)
            
            # Unpack sequence
            unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, 
                batch_first=True,
                total_length=x.size(1)  # Keep original sequence length
            )
            
            # Restore original order
            _, restored_idx = sorted_idx.sort(0)
            gru_out = unpacked_out[restored_idx]
            
            # Create output tensor with same shape as input
            output = torch.zeros_like(att_out)
            output[valid_mask] = gru_out
            output[~valid_mask] = att_out[~valid_mask]  # Use attention output for invalid sequences
            
        else:
            output = self.gru(att_out)[0]
        
        return output

class MultiGranularPeakPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout_rate=0.5):
        super().__init__()
        
        # Processors for each granularity
        self.granularity_processors = nn.ModuleDict({
            '5s': GranularityProcessor(input_size, hidden_size, dropout_rate),
            '10s': GranularityProcessor(input_size, hidden_size, dropout_rate),
            '20s': GranularityProcessor(input_size, hidden_size, dropout_rate),
            '30s': GranularityProcessor(input_size, hidden_size, dropout_rate),
            '60s': GranularityProcessor(input_size, hidden_size, dropout_rate)
        })
        
        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(5, hidden_size),  # 5 global features
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-granularity attention for combining different timescales
        self.cross_attention = MultiGranularAttention(hidden_size)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 2),  # 5 granularities + global
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)  # Mean and log variance
        )
        
    def forward(self, batch):
        # Process each granularity
        granularity_outputs = {}
        for granularity in self.granularity_processors.keys():
            features = batch[f'features_{granularity}']
            lengths = batch.get(f'length_{granularity}')
            
            # Ensure lengths is properly formatted if it exists
            if lengths is not None:
                if isinstance(lengths, dict):
                    lengths = lengths.get('lengths')
                lengths = lengths.squeeze() if hasattr(lengths, 'squeeze') else lengths
                
                # Ensure all lengths are positive
                if isinstance(lengths, torch.Tensor):
                    lengths = torch.clamp(lengths, min=1)
                else:
                    lengths = np.maximum(lengths, 1)
            
            processed = self.granularity_processors[granularity](features, lengths)
            
            # Use attention-weighted pooling over valid timesteps
            if lengths is not None:
                mask = torch.arange(processed.size(1), device=processed.device)[None, :] < lengths[:, None]
                masked_output = processed * mask.unsqueeze(-1)
                
                # Attention pooling with safe masking
                scores = torch.sum(masked_output, dim=-1, keepdim=True)
                mask_float = mask.float().unsqueeze(-1)
                scores = scores.masked_fill(mask_float == 0, float('-inf'))
                attention = F.softmax(scores, dim=1)
                pooled = torch.sum(masked_output * attention, dim=1)
            else:
                # If no lengths provided, use mean pooling
                pooled = processed.mean(dim=1)
            
            granularity_outputs[granularity] = pooled
        
        # Process global features
        global_features = self.global_processor(batch['global_features'])
        
        # Combine all granularities
        combined_features = torch.stack(
            [global_features] + [granularity_outputs[g] for g in self.granularity_processors.keys()],
            dim=1
        )
        
        # Apply cross-granularity attention
        attended_features, _ = self.cross_attention(combined_features)
        
        # Flatten and predict
        flattened = attended_features.reshape(attended_features.size(0), -1)
        output = self.predictor(flattened)
        
        # Split into mean and log variance
        mean, log_var = output.chunk(2, dim=-1)
        
        return mean.squeeze(-1), log_var.squeeze(-1)




def train_model(model, train_loader, val_loader,
                num_epochs=200,
                learning_rate=0.001,
                weight_decay=0.01,
                patience=15,
                use_wandb=False,
                project_name="time_to_peak",
                checkpoint_dir="checkpoints"):
    """
    Complete training function with gradient scaling, checkpointing, and monitoring
    
    Args:
        model: The MultiGranularPeakPredictor model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        patience: Number of epochs to wait for improvement before early stopping
        use_wandb: Whether to use Weights & Biases for logging
        project_name: Name for the wandb project
        checkpoint_dir: Directory to save model checkpoints
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = TimePredictionLoss(alpha=0.3, beta=0.2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'best_epoch': 0
    }
    
    # Initialize wandb
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
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics = {'mse': [], 'mae': [], 'directional': []}
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                # Forward pass
                mean, log_var = model(batch)
                loss = criterion(mean, log_var, batch['targets'])
                
                # Calculate additional metrics
                mse = nn.MSELoss()(mean, batch['targets'])
                mae = nn.L1Loss()(mean, batch['targets'])
                
                # Track metrics
                train_losses.append(loss.item())
                train_metrics['mse'].append(mse.item())
                train_metrics['mae'].append(mae.item())
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Learning rate scheduler step
            scheduler.step()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate average training metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = {'mse': [], 'mae': [], 'directional': []}
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                mean, log_var = model(batch)
                loss = criterion(mean, log_var, batch['targets'])
                
                # Calculate metrics
                mse = nn.MSELoss()(mean, batch['targets'])
                mae = nn.L1Loss()(mean, batch['targets'])
                
                # Track metrics
                val_losses.append(loss.item())
                val_metrics['mse'].append(mse.item())
                val_metrics['mae'].append(mae.item())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average validation metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Log metrics
        if use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mse": avg_train_metrics['mse'],
                "val_mse": avg_val_metrics['mse'],
                "train_mae": avg_train_metrics['mae'],
                "val_mae": avg_val_metrics['mae'],
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train MAE: {avg_train_metrics['mae']:.4f}, Val MAE: {avg_val_metrics['mae']:.4f}")
        
        # Save metrics
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            training_stats['best_epoch'] = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'training_stats': training_stats
            }, f"{checkpoint_dir}/best_model.pt")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if use_wandb:
        wandb.finish()
    
    return model, training_stats, best_val_loss




def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training pipeline")
    
    # Configuration
    config = {
        'input_size': 11,  # Number of base features
        'hidden_size': 256,
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'patience': 15,
        'val_size': 0.2,
        'random_state': 42,
        'use_wandb': True,  # Enable wandb logging
        'project_name': 'time_to_peak'
    }
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        df = pd.read_csv('data/token-data.csv')
        df = clean_dataset(df)
        logger.info(f"Data loaded and cleaned. Shape: {df.shape}")
        
        # Split data
        train_df, val_df = train_val_split(
            df,
            val_size=config['val_size'],
            random_state=config['random_state']
        )
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        # Create data loaders
        train_loader, val_loader = create_multi_granular_loaders(
            train_df,
            val_df,
            batch_size=config['batch_size']
        )
        logger.info("Data loaders created")
        
        # Initialize model
        model = MultiGranularPeakPredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size']
        ).to(device)
        logger.info("Model initialized")
        
        # Train model
        logger.info("Starting model training...")
        model, training_stats, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            patience=config['patience'],
            use_wandb=config['use_wandb'],
            project_name=config['project_name'],
            checkpoint_dir='checkpoints'
        )
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        # Save model and artifacts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'models/peak_predictor_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model artifacts
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_stats': training_stats,
            'best_val_loss': best_val_loss,
            'scaler': train_loader.dataset.get_scalers()
        }, f'{save_dir}/model_artifacts.pt')
        
        # Save a copy of the best checkpoint
        import shutil
        shutil.copy2('checkpoints/best_model.pt', f'{save_dir}/best_model.pt')
        
        logger.info(f"Model artifacts saved to {save_dir}")
        logger.info("Training pipeline completed successfully")
        
        # Return results
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
    results = main()
    
    # Access results (optional)
    model = results['model']
    training_stats = results['training_stats']
    best_val_loss = results['best_val_loss']
    save_dir = results['save_dir']