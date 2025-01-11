import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from Before30.dataset.hit_peak_30_dataset import HitPeakBefore30Dataset
import torch.optim as optim

from Before30.utils.clean_dataset import clean_dataset
from utils.add_data_quality_features import add_data_quality_features
from utils.attention_module import AttentionModule
from utils.early_stopping import EarlyStopping
warnings.filterwarnings('ignore')

class HitPeakBefore30Predictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5, global_feature_dim=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Define global feature dimension
        if global_feature_dim is None:
            self.global_feature_dim = 7  # Default
        else:
            self.global_feature_dim = global_feature_dim

        # Enhanced time-aware CNN layers
        self.conv_5s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.conv_10s = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        # Bidirectional GRU layers
        self.gru_5s = nn.GRU(hidden_size, hidden_size//2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=self.dropout_rate if num_layers > 1 else 0)
        self.gru_10s = nn.GRU(hidden_size, hidden_size//2, num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=self.dropout_rate if num_layers > 1 else 0)
        self.gru_20s = nn.GRU(input_size, hidden_size//2, num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=self.dropout_rate if num_layers > 1 else 0)
        self.gru_30s = nn.GRU(input_size, hidden_size//2, num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=self.dropout_rate if num_layers > 1 else 0)

        # Temporal attention modules
        self.attention_5s = AttentionModule(hidden_size)
        self.attention_10s = AttentionModule(hidden_size)
        self.attention_20s = AttentionModule(hidden_size)
        self.attention_30s = AttentionModule(hidden_size)

        # Additional temporal features processing
        self.temporal_fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Global feature processing with temporal awareness
        self.global_fc = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Quality gate with temporal context
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Binary classification head (REMOVED SIGMOID)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)  # Removed Sigmoid activation
        )

        # Initialize weights
        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        batch_size = x_5s.size(0)
        
        # Process temporal sequences
        x_5s = self.conv_5s(x_5s.transpose(1, 2)).transpose(1, 2)
        x_10s = self.conv_10s(x_10s.transpose(1, 2)).transpose(1, 2)
        
        # Apply GRU and temporal attention
        x_5s, _ = self.gru_5s(x_5s)
        x_10s, _ = self.gru_10s(x_10s)
        x_20s, _ = self.gru_20s(x_20s)
        x_30s, _ = self.gru_30s(x_30s)
        
        # Apply attention
        x_5s = self.attention_5s(x_5s)
        x_10s = self.attention_10s(x_10s)
        x_20s = self.attention_20s(x_20s)
        x_30s = self.attention_30s(x_30s)
        
        # Process global features
        global_features = self.global_fc(global_features)
        
        # Combine temporal features
        temporal_features = torch.cat([x_5s, x_10s, x_20s, x_30s], dim=1)
        temporal_features = self.temporal_fc(temporal_features)
        
        # Apply quality gating with temporal context
        quality_context = torch.cat([temporal_features, quality_features], dim=1)
        quality_weights = self.quality_gate(quality_context)
        
        # Combine features with quality weights
        combined_features = torch.cat([
            temporal_features * quality_weights,
            global_features
        ], dim=1)
        
        # Binary classification output (raw logits)
        binary_output = self.binary_head(combined_features)
        
        return binary_output


def train_hit_peak_before_30_model(train_loader, val_loader, 
                                num_epochs=200, learning_rate=0.0003775949513157161, weight_decay=0.01, 
                                patience=15, min_delta=0.001, hidden_size=256, num_layers=3, dropout_rate=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    input_size = 11
    hit_peak_before_30_model = HitPeakBefore30Predictor(
        input_size=input_size,
        hidden_size=256,
        num_layers=3,
        dropout_rate=0.5
    ).to(device)

    # Initialize loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # CHANGED: Use BCEWithLogitsLoss instead of BCELoss
    optimizer = optim.AdamW(
        hit_peak_before_30_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
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

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_val_loss = float('inf')
    
    # Initialize AMP
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None

    # Metrics history
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }

    for epoch in range(num_epochs):
        # Training phase
        hit_peak_before_30_model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Predict binary outcome (raw logits)
                    binary_pred = hit_peak_before_30_model(
                        batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                        batch['global_features'],
                        batch['quality_features']
                    )
                    
                    # Calculate loss (targeting binary column)
                    loss = criterion(binary_pred.squeeze(), batch['targets'][:, -1])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(hit_peak_before_30_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Predict binary outcome (raw logits)
                binary_pred = hit_peak_before_30_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'],
                    batch['quality_features']
                )
                
                # Calculate loss
                loss = criterion(binary_pred.squeeze(), batch['targets'][:, -1])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hit_peak_before_30_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()  # Step per batch
            train_loss += loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], '
                      f'Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss /= num_batches
        metrics_history['train_loss'].append(train_loss)

        # Validation phase
        hit_peak_before_30_model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        
        # Metric tracking variables
        total_correct = 0
        total_samples = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Predict binary outcome (raw logits)
                binary_pred = hit_peak_before_30_model(
                    batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'],
                    batch['global_features'],
                    batch['quality_features']
                )
                
                # Calculate loss
                targets = batch['targets'][:, -1]
                loss = criterion(binary_pred.squeeze(), targets)
                val_loss += loss.item()
                
                # Calculate metrics
                pred_binary = (torch.sigmoid(binary_pred.squeeze()) > 0.5).float()
                total_correct += (pred_binary == targets).float().sum()
                total_samples += targets.size(0)
                
                # Precision and Recall calculations
                true_positives += ((pred_binary == 1) & (targets == 1)).float().sum()
                false_positives += ((pred_binary == 1) & (targets == 0)).float().sum()
                false_negatives += ((pred_binary == 0) & (targets == 1)).float().sum()

        # Average validation metrics
        val_loss /= val_batches
        val_accuracy = total_correct / total_samples
        
        # Calculate Precision, Recall, F1 Score
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Store metrics
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_accuracy'].append(val_accuracy.item())
        metrics_history['val_precision'].append(precision.item())
        metrics_history['val_recall'].append(recall.item())
        metrics_history['val_f1_score'].append(f1_score.item())

        # Print epoch results
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy.item():.4f}')
        print(f'Val Precision: {precision.item():.4f}')
        print(f'Val Recall: {recall.item():.4f}')
        print(f'Val F1 Score: {f1_score.item():.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': hit_peak_before_30_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics_history': metrics_history
            }, 'best_hit_peak_before_30_model.pth')

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model
    checkpoint = torch.load('best_hit_peak_before_30_model.pth')
    hit_peak_before_30_model.load_state_dict(checkpoint['model_state_dict'])
    
    return hit_peak_before_30_model, best_val_loss, metrics_history    


def main_hit_peak_before_30(
    num_epochs=200, 
    learning_rate=0.001, 
    weight_decay=0.01, 
    patience=15, 
    min_delta=0.001,
    batch_size=128,
):
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    df = pd.read_csv('data/token-data.csv')
    
    # Ensure hit_peak_before_30 column exists
    if 'hit_peak_before_30' not in df.columns:
        df['hit_peak_before_30'] = (df['time_to_peak'] <= 30).astype(float)
    
    # Clean and add quality features
    df = clean_dataset(df)
    df = add_data_quality_features(df)

    # Split data using stratified sampling based on hit_peak_before_30
    # This ensures balanced representation in train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['hit_peak_before_30'], 
        random_state=42
    )

    # Create datasets
    train_dataset = HitPeakBefore30Dataset(train_df)
    val_dataset = HitPeakBefore30Dataset(val_df, scaler={
        'global': train_dataset.global_scaler
    })

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size
    )

    # Print dataset information
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Calculate class distribution
    train_hit_peak_count = train_df['hit_peak_before_30'].sum()
    train_no_hit_peak_count = len(train_df) - train_hit_peak_count
    print(f"Training set - Hit Peak Before 30: {train_hit_peak_count} ({train_hit_peak_count/len(train_df)*100:.2f}%)")
    print(f"Training set - No Hit Peak Before 30: {train_no_hit_peak_count} ({train_no_hit_peak_count/len(train_df)*100:.2f}%)")

    # Call training function (previously defined)
    hit_peak_model, val_loss, metrics_history = train_hit_peak_before_30_model(
        train_loader, 
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta
    )

    # Print final metrics
    print("\nFinal Model Metrics:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {metrics_history['val_accuracy'][-1]:.4f}")
    print(f"Final Validation Precision: {metrics_history['val_precision'][-1]:.4f}")
    print(f"Final Validation Recall: {metrics_history['val_recall'][-1]:.4f}")
    print(f"Final Validation F1 Score: {metrics_history['val_f1_score'][-1]:.4f}")

    return {
        'model': hit_peak_model,
        'val_loss': val_loss,
        'metrics_history': metrics_history
    }

# Allow direct script execution
if __name__ == "__main__":
    try:
        results = main_hit_peak_before_30()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")