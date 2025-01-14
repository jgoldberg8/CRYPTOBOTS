import datetime
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from TimeToPeak.datasets.time_token_dataset import TimePeakDataset
from TimeToPeak.utils.time_loss import PeakPredictionLoss
from TimeToPeak.utils.clean_dataset import clean_dataset

class PeakPredictor(nn.Module):
    def __init__(self, 
                feature_size,  # Number of input features
                hidden_size=256,
                dropout_rate=0.4):
        super().__init__()
        
        # Define attention dimension
        self.attention_dim = hidden_size // 2
        
        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Global feature processor
        self.global_processor = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combine features across time windows using attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Peak prediction head
        self.peak_predictor = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 4, 1)  # Binary classification
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, features, global_features):
        # Process features
        processed_features = self.feature_net(features)
        
        # Process global features
        processed_global = self.global_processor(global_features)
        
        # Combine features - ensure they have same dimension
        combined = processed_features + processed_global  # Element-wise addition
        
        # Apply self-attention
        attended_features, _ = self.attention(
            combined.unsqueeze(1),  # Add sequence dimension
            combined.unsqueeze(1),
            combined.unsqueeze(1)
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        # Get predictions
        peak_logits = self.peak_predictor(attended_features)
        confidence_logits = self.confidence_predictor(attended_features)
        
        return peak_logits, confidence_logits



def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """Train the peak prediction model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = PeakPredictionLoss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_bar:
            # Move batch to device
            features = batch['features'].to(device)
            global_features = batch['global_features'].to(device)
            timestamps = batch['timestamp'].to(device)
            time_to_peak = batch['time_to_peak'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast(device_type='cuda'):
                peak_logits, confidence_logits = model(features, global_features)
                loss = criterion(
                    peak_logits,
                    confidence_logits,
                    timestamps,
                    time_to_peak,
                    mask
                )
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_losses = []
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for batch in val_bar:
                features = batch['features'].to(device)
                global_features = batch['global_features'].to(device)
                timestamps = batch['timestamp'].to(device)
                time_to_peak = batch['time_to_peak'].to(device)
                mask = batch['mask'].to(device)
                
                peak_logits, confidence_logits = model(features, global_features)
                loss = criterion(
                    peak_logits,
                    confidence_logits,
                    timestamps,
                    time_to_peak,
                    mask
                )
                
                val_losses.append(loss.item())
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_val_loss

def predict_peak(model, features, global_features, device='cuda'):
    """
    Make a peak prediction for a single token at a specific timestamp.
    
    Args:
        model: Trained PeakPredictor model
        features: Current features tensor
        global_features: Global token features tensor
        device: Device to run prediction on
    
    Returns:
        peak_prob: Probability that this is a peak
        confidence: Model's confidence in the prediction
    """
    model.eval()
    
    with torch.no_grad():
        features = features.to(device)
        global_features = global_features.to(device)
        
        peak_logits, confidence_logits = model(features, global_features)
        
        peak_prob = torch.sigmoid(peak_logits)
        confidence = torch.sigmoid(confidence_logits)
        
        return peak_prob.item(), confidence.item()
    


def main():
    try:
        # Create directories
        Path("checkpoints").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Clean up all old files
        for file in Path("checkpoints").glob("*"):
            print(f"Removing {file}...")
            file.unlink()
        
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/time-data.csv')
        df = clean_dataset(df)
        
        # Split data (80-20)
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        
        # Create training dataset first
        print("Creating training dataset...")
        train_dataset = TimePeakDataset(train_df, scaler=None, train=True)
        
        # Print feature information
        print(f"Feature configuration:")
        print(f"Time windows: {train_dataset.time_windows}")
        print(f"Base features: {len(train_dataset.base_features)}")
        print(f"Total features per window: {len(train_dataset.base_features) + 5}")
        print(f"Expected total features: {train_dataset.feature_size}")
        
        # Create validation dataset using training scalers
        print("Creating validation dataset...")
        val_dataset = TimePeakDataset(
            val_df,
            scaler=train_dataset.scalers,
            train=False
        )
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Verify feature dimensions
        sample = next(iter(train_loader))
        feature_size = sample['features'].shape[1]
        print(f"Actual feature size from loader: {feature_size}")
        
        if feature_size != train_dataset.feature_size:
            raise ValueError(f"Feature size mismatch! Expected {train_dataset.feature_size} but got {feature_size}")
        
        # Initialize model
        print("Initializing model...")
        model = PeakPredictor(
            feature_size=feature_size,
            hidden_size=512,
            dropout_rate=0.4
        )
        
        
        # Train model
        print("Starting training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            epochs=100,
            lr=0.001,
            device=device
        )
        
        # Save training info
        training_info = {
            'val_loss': float(best_val_loss),
            'model_config': {
                'feature_size': feature_size,
                'hidden_size': 512,
                'dropout_rate': 0.4
            },
            'training_config': {
                'epochs': 100,
                'learning_rate': 0.001,
                'train_size': len(train_df),
                'val_size': len(val_df)
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open('checkpoints/training_info.json', 'w') as f:
            json.dump(training_info, f, indent=4)
        
        # Save scalers
        torch.save(train_dataset.scalers, 'checkpoints/scalers.pt')
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        print("Model, scalers, and training info saved in checkpoints directory")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()