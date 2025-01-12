import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import datetime
import logging
from tqdm import tqdm

from TimeToPeak.datasets.time_token_dataset import MultiGranularTokenDataset
from TimeToPeak.utils.clean_dataset import clean_dataset
from TimeToPeak.utils.time_loss import PeakPredictionLoss
warnings.filterwarnings('ignore')

class FeatureExtractor(nn.Module):
    """Feature extraction module that processes multiple time granularities"""
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.feature_net(x)

class RealTimePeakPredictor(nn.Module):
    def __init__(self, 
                input_size=11,  # Base feature size
                hidden_size=256,  # Increased for more capacity
                num_granularities=4,  # 5s, 10s, 20s, 30s, 60s windows
                dropout_rate=0.4):
        super().__init__()
        
        # Feature extractors for each granularity
        self.granularity_extractors = nn.ModuleDict({
            f"{i}": FeatureExtractor(input_size, hidden_size, dropout_rate)
            for i in range(num_granularities)
        })
        
        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # Process global features
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention for combining features across granularities
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        attention_output_size = (hidden_size // 2) * (num_granularities + 1)  # +1 for global features
        
        # Hazard prediction head (probability of being at peak)
        self.hazard_predictor = nn.Sequential(
            nn.Linear(attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Time-to-peak prediction head
        self.time_predictor = nn.Sequential(
            nn.Linear(attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensures positive time predictions
        )
        
        # Confidence prediction head
        self.confidence_predictor = nn.Sequential(
            nn.Linear(attention_output_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        # Process each granularity
        granularity_features = []
        
        for i in range(len(self.granularity_extractors)):
            features = batch[f'features_{i}']
            extracted = self.granularity_extractors[str(i)](features)
            granularity_features.append(extracted)
        
        # Process global features
        global_features = self.global_processor(batch['global_features'])
        
        # Combine all features
        all_features = torch.cat([global_features] + granularity_features, dim=1)
        batch_size = all_features.size(0)
        seq_len = len(self.granularity_extractors) + 1
        
        # Reshape for attention
        features_reshaped = all_features.view(batch_size, seq_len, -1)
        
        # Apply attention
        attended_features, _ = self.attention(
            features_reshaped, 
            features_reshaped, 
            features_reshaped
        )
        
        # Flatten attended features
        flat_features = attended_features.reshape(batch_size, -1)
        
        # Get predictions
        hazard_prob = self.hazard_predictor(flat_features)
        time_to_peak = self.time_predictor(flat_features)
        confidence = self.confidence_predictor(flat_features)
        
        return hazard_prob, time_to_peak, confidence



def find_optimal_batch_size(train_dataset, start_size=32, max_size=512):
    """Find optimal batch size based on GPU memory"""
    if not torch.cuda.is_available():
        return start_size
        
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    current_size = start_size
    while current_size <= max_size:
        try:
            loader = DataLoader(
                train_dataset,
                batch_size=current_size,
                shuffle=True,
                pin_memory=True
            )
            batch = next(iter(loader))
            batch = {k: v.to(device) for k, v in batch.items()}
            current_size *= 2
            del batch
            del loader
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                return current_size // 2
            raise e
    
    return max_size

def create_data_loaders(train_df, val_df, batch_size=None, num_workers=4):
    """Create train and validation data loaders"""
    # Create datasets
    train_dataset = MultiGranularTokenDataset(train_df, train=True)
    val_dataset = MultiGranularTokenDataset(
        val_df,
        scaler=train_dataset.scalers,
        train=False
    )
    
    # Find optimal batch size if not provided
    if batch_size is None:
        batch_size = find_optimal_batch_size(train_dataset)
        print(f"Using batch size: {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """Train the model with early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    criterion = PeakPredictionLoss()
    scaler = torch.GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=True):
                hazard_prob, time_pred, confidence = model(batch)
                loss = criterion(
                    hazard_prob,
                    time_pred,
                    confidence,
                    batch['peak_proximity'],
                    batch['time_to_peak'],
                    batch['sample_weights'],
                    batch['mask']
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation phase
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                hazard_prob, time_pred, confidence = model(batch)
                loss = criterion(
                    hazard_prob,
                    time_pred,
                    confidence,
                    batch['peak_proximity'],
                    batch['time_to_peak'],
                    batch['sample_weights'],
                    batch['mask']
                )
                val_losses.append(loss.item())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average losses
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered. Best epoch: {best_epoch+1} with loss: {best_val_loss:.4f}')
                break
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_val_loss

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load and clean data
        logger.info("Loading and cleaning data...")
        df = pd.read_csv('data/time-data.csv')
        df = clean_dataset(df)
        
        # Split data (80-20 split)
        logger.info("Splitting data...")
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(train_df, val_df)
        
        # Initialize model
        logger.info("Initializing model...")
        model = RealTimePeakPredictor(
            input_size=11,
            hidden_size=256,
            num_granularities=4,
            dropout_rate=0.4
        )
        
        # Train model
        logger.info("Starting training...")
        model, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            epochs=100,
            lr=0.001,
            patience=15
        )
        
        # Save training info
        training_info = {
            'best_val_loss': float(best_val_loss),
            'model_config': {
                'input_size': 11,
                'hidden_size': 256,
                'num_granularities': 4,
                'dropout_rate': 0.4
            },
            'training_config': {
                'epochs': 100,
                'learning_rate': 0.001,
                'patience': 15,
                'train_size': len(train_df),
                'val_size': len(val_df)
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save training info
        save_dir = Path("checkpoints")
        with open(save_dir / 'training_info.json', 'w') as f:
            json.dump(training_info, f, indent=4)
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model and training info saved to {save_dir}")
        
        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'train_loader': train_loader,
            'val_loader': val_loader
        }
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()