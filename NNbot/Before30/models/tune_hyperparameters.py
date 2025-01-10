import warnings
import optuna
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from Before30.dataset.hit_peak_30_dataset import HitPeakBefore30Dataset
from Before30.models.peak_before_30_model import train_hit_peak_before_30_model
from Before30.utils.clean_dataset import clean_dataset
from utils.add_data_quality_features import add_data_quality_features
from utils.train_val_split import train_val_split
warnings.filterwarnings('ignore')


def create_representative_subset(df, fraction=0.2):
    target_column = 'hit_peak_before_30'
    
    print(f"Number of rows before processing: {len(df)}")
    print(f"Target distribution before sampling:")
    print(df[target_column].value_counts(normalize=True))
    
    # Remove any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column])
    
    print(f"\nNumber of rows after cleaning: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("Dataset became empty after cleaning")
        
    if len(df) < 5:  # If we have very few samples
        return df
        
    try:
        # Stratified sampling for binary target
        sample_size = int(len(df) * fraction)
        
        # Split by class
        positive_samples = df[df[target_column] == True]
        negative_samples = df[df[target_column] == False]
        
        # Calculate number of samples for each class to maintain ratio
        pos_ratio = len(positive_samples) / len(df)
        n_positive = int(sample_size * pos_ratio)
        n_negative = sample_size - n_positive
        
        # Sample from each class
        sampled_positive = positive_samples.sample(n=min(n_positive, len(positive_samples)), 
                                                 random_state=42)
        sampled_negative = negative_samples.sample(n=min(n_negative, len(negative_samples)), 
                                                 random_state=42)
        
        # Combine samples
        sampled_df = pd.concat([sampled_positive, sampled_negative])
        sampled_df = sampled_df.sample(frac=1, random_state=42)  # Shuffle
        
        print("\nFinal sampling results:")
        print(f"Final subset size: {len(sampled_df)}")
        print("\nTarget distribution after sampling:")
        print(sampled_df[target_column].value_counts(normalize=True))
        
        return sampled_df
        
    except Exception as e:
        print(f"Stratified sampling failed: {str(e)}")
        # Fallback to simple random sampling
        return df.sample(frac=fraction, random_state=42)

def prepare_data(use_subset=True):
    # Data preparation
    print("Loading data...")
    df = pd.read_csv('data/before_30_data.csv')
    print(f"Initial data shape: {df.shape}")
    
    print("Cleaning dataset...")
    df = clean_dataset(df)
    print(f"Shape after cleaning: {df.shape}")
    
    print("Adding data quality features...")
    df = add_data_quality_features(df)
    print(f"Shape after adding features: {df.shape}")

    # Optional: Use representative subset
    if use_subset:
        print("Creating representative subset...")
        df = create_representative_subset(df)
        print(f"Final shape: {df.shape}")

    # Split data
    train_df, val_df = train_val_split(df)

    # Create datasets
    train_dataset_peak = HitPeakBefore30Dataset(train_df)
    val_dataset_peak = HitPeakBefore30Dataset(val_df, scaler={
        'global': train_dataset_peak.global_scaler,
        'target': train_dataset_peak.targets
    })

    # Create data loaders
    train_loader = DataLoader(train_dataset_peak, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset_peak, batch_size=128)

    return train_loader, val_loader

def objective(trial):
    # Prepare data
    train_loader, val_loader = prepare_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.6)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    
    # Early stopping parameters
    patience = trial.suggest_int('patience', 10, 30)
    min_delta = trial.suggest_loguniform('min_delta', 1e-4, 1e-2)

    # Modify train function arguments to include early stopping parameters
    try:
        model, val_loss = train_hit_peak_before_30_model(
            train_loader, 
            val_loader, 
            num_epochs=50,  # Maximum epochs
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            patience=patience,
            min_delta=min_delta
        )

        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def hyperparameter_tuning():
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    
    # More intelligent pruning
    study.optimize(
        objective, 
        n_trials=50,  # Adjust based on computational resources
        callbacks=[optuna.study.MaxTrialsCallback(50, states=(optuna.trial.TrialState.COMPLETE,))]
    )

    # Detailed results logging
    print('\n=== Hyperparameter Tuning Results ===')
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value (Validation Loss): ', trial.value)
    print('  Optimal Hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Optional: Visualize hyperparameter importance
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        
        # Param importance plot
        fig = vis.plot_param_importances(study)
        fig.write_image("hyperparameter_importance.png")
        
        # Optimization history plot
        fig = vis.plot_optimization_history(study)
        fig.write_image("optimization_history.png")
    except ImportError:
        print("Install 'matplotlib' and 'plotly' for visualization")

    return study.best_params, study.best_value

# Main execution
if __name__ == "__main__":
    best_params, best_val_loss = hyperparameter_tuning()