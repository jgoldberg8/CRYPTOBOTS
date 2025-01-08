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

class TokenDataset(Dataset):
    def __init__(self, df, scaler=None, train=True):
        """
        Args:
            df: pandas DataFrame with token data
            scaler: StandardScaler instance (optional)
            train: boolean indicating if this is training data
        """
        # Convert all numeric columns to float32
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        # Features for each window
        self.base_features = [
            'transaction_count',
            'buy_pressure',
            'volume',
            'rsi',
            'price_volatility',
            'volume_volatility',
            'momentum',
            'trade_amount_variance',
            'transaction_rate',
            'trade_concentration',
            'unique_wallets'
        ]
        
        # Time windows
        self.time_windows = {
            '5s': ['0to5', '5to10', '10to15', '15to20', '20to25', '25to30'],
            '10s': ['0to10', '10to20', '20to30'],
            '20s': ['0to20'],
            '30s': ['0to30']
        }
        
        # Global features
        self.global_features = ['initial_investment_ratio', 'initial_market_cap']
        self.targets = ['peak_market_cap', 'time_to_peak']
        
        # Scale the data
        if train:
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaled_data = self._preprocess_data(df, fit=True)
            else:
                self.scaler = scaler
                self.scaled_data = self._preprocess_data(df, fit=False)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for test data")
            self.scaler = scaler
            self.scaled_data = self._preprocess_data(df, fit=False)
            
        self.quality_features = self._calculate_quality_features(df)



        


    def _preprocess_data(self, df, fit=False):
        # Process each time window
        processed_data = {}
        
        # Process 5s intervals
        for window_type, windows in self.time_windows.items():
            window_data = []
            for window in windows:
                features = []
                for feature in self.base_features:
                    col_name = f"{feature}_{window}s"
                    features.append(df[col_name].values)
                window_data.append(np.stack(features, axis=1))
            processed_data[window_type] = np.stack(window_data, axis=1)
            
        # Process global features
        global_data = df[self.global_features].values
        if fit:
            global_data = self.scaler.fit_transform(global_data)
        else:
            global_data = self.scaler.transform(global_data)
            
        # Process targets
        target_data = df[self.targets].values
        if fit:
            target_data = self.scaler.fit_transform(target_data)
        else:
            target_data = self.scaler.transform(target_data)
            
        return {
            'data': processed_data,
            'global': global_data,
            'targets': target_data
        }
    


    def _calculate_quality_features(self, df):
        """Calculate data quality features"""
        # Calculate completeness ratio
        completeness = df.notna().mean(axis=1).values
        
        # Calculate active intervals
        active_intervals = df[[f"transaction_count_{window}s" 
                             for windows in self.time_windows.values() 
                             for window in windows]].gt(0).sum(axis=1).values
        
        return np.stack([completeness, active_intervals], axis=1)
        


    def __len__(self):
        return len(self.scaled_data['targets'])
        
    def __getitem__(self, idx):
        # Get time window data
        x_5s = torch.FloatTensor(self.scaled_data['data']['5s'][idx])
        x_10s = torch.FloatTensor(self.scaled_data['data']['10s'][idx])
        x_20s = torch.FloatTensor(self.scaled_data['data']['20s'][idx])
        x_30s = torch.FloatTensor(self.scaled_data['data']['30s'][idx])

        # Get global features
        global_features = torch.FloatTensor(self.scaled_data['global'][idx])

        # Get quality features
        quality_features = torch.FloatTensor(self.quality_features[idx])

        # Get targets
        targets = torch.FloatTensor(self.scaled_data['targets'][idx])

        return {
            'x_5s': x_5s,
            'x_10s': x_10s,
            'x_20s': x_20s,
            'x_30s': x_30s,
            'global_features': global_features,
            'quality_features': quality_features,
            'targets': targets
        }
    
        

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Calculate attention weights
        weights = self.attention(x)
        # Apply attention weights
        return torch.sum(weights * x, dim=1)
    



class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return False    
    
class PeakMarketCapPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

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

        # Quality gate
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Global feature processing
        self.global_fc = nn.Linear(2, hidden_size)

        # Final layers
        self.fc1 = nn.Linear(hidden_size * 5, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single value for peak market cap

        # Initialize weights
        self._initialize_weights()

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

    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        # Process 5-second windows
        x_5s = self.conv_5s(x_5s.transpose(1, 2))
        x_5s = x_5s.transpose(1, 2)
        x_5s, _ = self.lstm_5s(x_5s)
        x_5s = self.attention_5s(x_5s)
        
        # Process 10-second windows
        x_10s = self.conv_10s(x_10s.transpose(1, 2))
        x_10s = x_10s.transpose(1, 2)
        x_10s, _ = self.lstm_10s(x_10s)
        x_10s = self.attention_10s(x_10s)
        
        # Process 20-second windows
        x_20s, _ = self.lstm_20s(x_20s)
        x_20s = self.attention_20s(x_20s)
        
        # Process 30-second windows
        x_30s, _ = self.lstm_30s(x_30s)
        x_30s = self.attention_30s(x_30s)
        
        # Process global features
        global_features = self.global_fc(global_features)
        
        # Combine temporal features
        temporal_features = torch.stack([x_5s, x_10s, x_20s, x_30s], dim=1)
        temporal_mean = torch.mean(temporal_features, dim=1)
        
        # Quality-aware attention
        quality_context = torch.cat([temporal_mean, quality_features], dim=1)
        quality_weights = self.quality_gate(quality_context)
        
        # Apply quality weights and combine features
        weighted_features = [
            x_5s * quality_weights,
            x_10s * quality_weights,
            x_20s * quality_weights,
            x_30s * quality_weights,
            global_features
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Final prediction for peak market cap
        peak_market_cap = self.fc1(combined)
        peak_market_cap = torch.relu(peak_market_cap)
        peak_market_cap = self.dropout(peak_market_cap)
        peak_market_cap = self.fc2(peak_market_cap)
        
        return peak_market_cap







class TimeToPeakPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout_rate=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

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

        # Quality gate
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Global feature processing
        self.global_fc = nn.Linear(2, hidden_size)

        # Final layers
        self.fc1 = nn.Linear(hidden_size * 5, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single value for time to peak

        # Initialize weights
        self._initialize_weights()

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



    def forward(self, x_5s, x_10s, x_20s, x_30s, global_features, quality_features):
        batch_size = x_5s.size(0)
      
        # Process 5-second windows
        x_5s = self.conv_5s(x_5s.transpose(1, 2))
        x_5s = x_5s.transpose(1, 2)
        x_5s, _ = self.lstm_5s(x_5s)
        x_5s = self.attention_5s(x_5s)
        
        # Process 10-second windows
        x_10s = self.conv_10s(x_10s.transpose(1, 2))
        x_10s = x_10s.transpose(1, 2)
        x_10s, _ = self.lstm_10s(x_10s)
        x_10s = self.attention_10s(x_10s)
        
        # Process 20-second windows
        x_20s, _ = self.lstm_20s(x_20s)
        x_20s = self.attention_20s(x_20s)
        
        # Process 30-second windows
        x_30s, _ = self.lstm_30s(x_30s)
        x_30s = self.attention_30s(x_30s)

        # Process global features
        global_features = self.global_fc(global_features)
        
        # Combine temporal features
        temporal_features = torch.stack([x_5s, x_10s, x_20s, x_30s], dim=1)
        temporal_mean = torch.mean(temporal_features, dim=1)
        
        # Quality-aware attention
        quality_context = torch.cat([temporal_mean, quality_features], dim=1)
        quality_weights = self.quality_gate(quality_context)
        
        # Apply quality weights and combine features
        weighted_features = [
            x_5s * quality_weights,
            x_10s * quality_weights,
            x_20s * quality_weights,
            x_30s * quality_weights,
            global_features
        ]
        combined = torch.cat(weighted_features, dim=1)
        
        # Final prediction for time to peak
        time_to_peak = self.fc1(combined)
        time_to_peak = torch.relu(time_to_peak)
        time_to_peak = self.dropout(time_to_peak)
        time_to_peak = self.fc2(time_to_peak)
        
        return time_to_peak
    


def train_model(peak_market_cap_model, time_to_peak_model, train_loader, val_loader, num_epochs=200, learning_rate=0.001, weight_decay=0.01, patience=15, min_delta=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peak_market_cap_model = peak_market_cap_model.to(device)
    time_to_peak_model = time_to_peak_model.to(device)

    # Combined loss function
    peak_market_cap_criterion = nn.MSELoss()
    time_to_peak_criterion = nn.MSELoss()

    # AdamW optimizer for both models
    peak_market_cap_optimizer = optim.AdamW(peak_market_cap_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    time_to_peak_optimizer = optim.AdamW(time_to_peak_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine annealing scheduler for both models
    peak_market_cap_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(peak_market_cap_optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    time_to_peak_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(time_to_peak_optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Early stopping for both models
    peak_market_cap_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    time_to_peak_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    best_peak_market_cap_val_loss = float('inf')
    best_time_to_peak_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase for peak market cap model
        peak_market_cap_model.train()
        peak_market_cap_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            peak_market_cap_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                peak_market_cap_output = peak_market_cap_model(batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'], batch['global_features'], batch['quality_features'])
                peak_market_cap_loss = peak_market_cap_criterion(peak_market_cap_output, batch['targets'][:, 0].unsqueeze(1))
            peak_market_cap_loss.backward()
            torch.nn.utils.clip_grad_norm_(peak_market_cap_model.parameters(), max_norm=1.0)
            peak_market_cap_optimizer.step()
            peak_market_cap_train_loss += peak_market_cap_loss.item()
        peak_market_cap_train_loss /= len(train_loader)

        # Validation phase for peak market cap model
        peak_market_cap_model.eval()
        peak_market_cap_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                peak_market_cap_output = peak_market_cap_model(batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'], batch['global_features'], batch['quality_features'])
                peak_market_cap_loss = peak_market_cap_criterion(peak_market_cap_output, batch['targets'][:, 0].unsqueeze(1))
                peak_market_cap_val_loss += peak_market_cap_loss.item()
        peak_market_cap_val_loss /= len(val_loader)

        # Training phase for time to peak model
        time_to_peak_model.train()
        time_to_peak_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            time_to_peak_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                time_to_peak_output = time_to_peak_model(batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'], batch['global_features'], batch['quality_features'])
                time_to_peak_loss = time_to_peak_criterion(time_to_peak_output, batch['targets'][:, 1].unsqueeze(1))
            time_to_peak_loss.backward()
            torch.nn.utils.clip_grad_norm_(time_to_peak_model.parameters(), max_norm=1.0)
            time_to_peak_optimizer.step()
            time_to_peak_train_loss += time_to_peak_loss.item()
        time_to_peak_train_loss /= len(train_loader)

        # Validation phase for time to peak model
        time_to_peak_model.eval()
        time_to_peak_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                time_to_peak_output = time_to_peak_model(batch['x_5s'], batch['x_10s'], batch['x_20s'], batch['x_30s'], batch['global_features'], batch['quality_features'])
                time_to_peak_loss = time_to_peak_criterion(time_to_peak_output, batch['targets'][:, 1].unsqueeze(1))
                time_to_peak_val_loss += time_to_peak_loss.item()
        time_to_peak_val_loss /= len(val_loader)

        # Update learning rates
        peak_market_cap_scheduler.step()
        time_to_peak_scheduler.step()

        print(f'Epoch {epoch+1}: Peak Market Cap Train Loss = {peak_market_cap_train_loss:.4f}, Peak Market Cap Val Loss = {peak_market_cap_val_loss:.4f}, LR = {peak_market_cap_scheduler.get_last_lr()[0]:.6f}')
        print(f'Epoch {epoch+1}: Time to Peak Train Loss = {time_to_peak_train_loss:.4f}, Time to Peak Val Loss = {time_to_peak_val_loss:.4f}, LR = {time_to_peak_scheduler.get_last_lr()[0]:.6f}')

        # Save best models
        if peak_market_cap_val_loss < best_peak_market_cap_val_loss:
            best_peak_market_cap_val_loss = peak_market_cap_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': peak_market_cap_model.state_dict(),
                'optimizer_state_dict': peak_market_cap_optimizer.state_dict(),
                'best_val_loss': best_peak_market_cap_val_loss,
            }, 'best_peak_market_cap_model.pth')

        if time_to_peak_val_loss < best_time_to_peak_val_loss:
            best_time_to_peak_val_loss = time_to_peak_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': time_to_peak_model.state_dict(),
                'optimizer_state_dict': time_to_peak_optimizer.state_dict(),
                'best_val_loss': best_time_to_peak_val_loss,
            }, 'best_time_to_peak_model.pth')

        # Early stopping
        if peak_market_cap_early_stopping(peak_market_cap_val_loss) or time_to_peak_early_stopping(time_to_peak_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        

def clean_dataset(df):
    # Drop records where no peak was recorded (time_to_peak is 0)
    df = df[df['time_to_peak'] > 0]
    
    # Drop records where critical initial windows are missing
    critical_cols = [
        'transaction_count_0to5s',
        'transaction_count_0to10s',
        'initial_market_cap',
        'peak_market_cap'
    ]
    df = df.dropna(subset=critical_cols)
    
    # For remaining timeframes, we can either:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col.startswith('rsi'):
            df[col] = df[col].fillna(50)  # Neutral RSI
        elif col.startswith(('transaction_count_', 'volume_', 'trade_amount_')):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(0)
            
    return df

def add_data_quality_features(df):
    # Create a copy to defragment and avoid modifying original
    df = df.copy()
    
    # Add multiple columns at once
    df[['data_completeness', 'active_intervals']] = pd.DataFrame({
        'data_completeness': df.notna().mean(axis=1),
        'active_intervals': df[[col for col in df.columns 
                                if col.startswith('transaction_count_')]].gt(0).sum(axis=1)
    })
    
    return df
            




def main():
    # Load and preprocess data for 2025-01-07
    df_07 = pd.read_csv('data/token_data_2025-01-07.csv')
    df_07 = clean_dataset(df_07)
    df_07 = add_data_quality_features(df_07)

    # Load and preprocess data for 2025-01-08
    df_08 = pd.read_csv('data/token_data_2025-01-08.csv')
    df_08 = clean_dataset(df_08)
    df_08 = add_data_quality_features(df_08)

    # Concatenate the DataFrames
    df = pd.concat([df_07, df_08], ignore_index=True)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TokenDataset(train_df)
    val_dataset = TokenDataset(val_df, scaler=train_dataset.scaler)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize models
    input_size = 11
    peak_market_cap_model = PeakMarketCapPredictor(input_size, hidden_size=256, num_layers=3, dropout_rate=0.5)
    time_to_peak_model = TimeToPeakPredictor(input_size, hidden_size=256, num_layers=3, dropout_rate=0.5)

    # Train models
    train_model(peak_market_cap_model, time_to_peak_model, train_loader, val_loader)


if __name__ == "__main__":
    main()