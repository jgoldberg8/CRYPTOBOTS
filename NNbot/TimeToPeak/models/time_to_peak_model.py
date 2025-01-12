import datetime
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

from TimeToPeak.datasets.time_token_dataset import create_multi_granular_loaders
from TimeToPeak.utils import save_model_artifacts
from TimeToPeak.utils.setup_logging import setup_logging
from TimeToPeak.utils.train_val_split import train_val_split
from TimeToPeak.utils.time_loss import RealTimePeakLoss
from TimeToPeak.utils.clean_dataset import clean_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            peak_detected = (confidence > self.confidence_threshold) & (peak_prob > 0.5)
            
            return mean.squeeze(-1), log_var.squeeze(-1), peak_detected, peak_prob
        
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
    Training function with fixed shape handling
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = RealTimePeakLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    scaler = torch.GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
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
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', enabled=True):
                # Forward pass with peak detection
                mean, log_var, peak_detected, peak_prob = model(batch, detect_peaks=True)
                
                # Reshape target to match prediction shape
                target = batch['targets'].view(-1)
                
                # Calculate loss with peak detection component
                loss = criterion(
                    mean, 
                    log_var, 
                    peak_prob, 
                    target,
                    peak_target=batch.get('peak_target')
                )
                
                # Calculate metrics with consistent shapes
                mse = F.mse_loss(mean.view(-1), target)
                mae = F.l1_loss(mean.view(-1), target)
                
                train_losses.append(loss.item())
                train_metrics['mse'].append(mse.item())
                train_metrics['mae'].append(mae.item())
                
                # Peak detection metrics if available
                if 'peak_target' in batch:
                    peak_target = batch['peak_target'].view(-1)
                    peak_accuracy = ((peak_detected.view(-1) == peak_target).float().mean()).item()
                    train_metrics['peak_accuracy'].append(peak_accuracy)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
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
                
                mse = nn.MSELoss()(mean, batch['targets'])
                mae = nn.L1Loss()(mean, batch['targets'])
                
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
        
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items() if len(v) > 0}
        
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
            
            # Add peak detection metrics if available
            for metric in ['peak_accuracy', 'peak_precision', 'peak_recall']:
                if metric in avg_train_metrics:
                    wandb_log[f"train_{metric}"] = avg_train_metrics[metric]
                if metric in avg_val_metrics:
                    wandb_log[f"val_{metric}"] = avg_val_metrics[metric]
            
            wandb.log(wandb_log)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train MAE: {avg_train_metrics['mae']:.4f}, Val MAE: {avg_val_metrics['mae']:.4f}")
        if 'peak_accuracy' in avg_train_metrics:
            print(f"Train Peak Accuracy: {avg_train_metrics['peak_accuracy']:.4f}, "
                  f"Val Peak Accuracy: {avg_val_metrics['peak_accuracy']:.4f}")
        
        # Save metrics
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
        if 'peak_accuracy' in avg_val_metrics:
            training_stats['peak_detection_accuracy'].append(avg_val_metrics['peak_accuracy'])
        
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
    
    config = {
        'input_size': 11,
        'hidden_size': 256,
        'window_size': 60,
        'val_size': 0.2,
        'random_state': 42,
        'use_wandb': True,
        'project_name': 'time_to_peak'
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load and process data
        logger.info("Loading data...")
        df = pd.read_csv('data/time-data.csv')
        df = clean_dataset(df)
        logger.info(f"Data loaded and cleaned. Shape: {df.shape}")
        
        train_df, val_df = train_val_split(
            df,
            val_size=config['val_size'],
            random_state=config['random_state']
        )
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        train_loader, val_loader = create_multi_granular_loaders(
            train_df,
            val_df,
            batch_size=32
        )
        logger.info("Data loaders created")
        
        # Initialize real-time model
        model = RealTimePeakPredictor(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            window_size=config['window_size']
        ).to(device)
        logger.info("Model initialized")
        
        # Train model
        logger.info("Starting model training...")
        model, training_stats, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            use_wandb=config['use_wandb'],
            project_name=config['project_name']
        )
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        # Save model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'models/realtime_peak_predictor_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_stats': training_stats,
            'best_val_loss': best_val_loss,
            'scaler': train_loader.dataset.get_scalers()
        }, f'{save_dir}/model_artifacts.pt')
        
        shutil.copy2('checkpoints/best_model.pt', f'{save_dir}/best_model.pt')
        
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
    results = main()