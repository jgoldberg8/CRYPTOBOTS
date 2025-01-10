import optuna
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

# Import your existing classes and functions
from peak_market_cap_model import PeakMarketCapPredictor, train_peak_market_cap_model
from token_dataset import TokenDataset
from utils.train_val_split import train_val_split
from utils.add_data_quality_features import add_data_quality_features
from PeakMarketCap.models.model_utilities import clean_dataset

def create_representative_subset(df, sample_size=200, random_state=42):
    """
    Creates a very small but representative subset of the data, optimized for peak_market_cap prediction.
    
    Args:
        df (pd.DataFrame): Input dataframe
        sample_size (int): Desired size of the subset (e.g., 50 or 100)
        random_state (int): Random seed for reproducibility
    """
    target_column = 'peak_market_cap'
    
    print(f"Original dataset size: {len(df)}")
    print(f"Target column stats before sampling:\n{df[target_column].describe()}\n")
    
    # Remove any infinite or missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_column])
    
    if len(df) == 0:
        raise ValueError("Dataset became empty after cleaning")
    
    if len(df) <= sample_size:
        return df
    
    try:
        # Calculate number of bins based on sample size
        n_bins = min(5, sample_size // 10)  # Ensure at least 10 samples per bin
        
        # Create bins with approximately equal number of samples
        bins = np.percentile(df[target_column], 
                           np.linspace(0, 100, n_bins + 1))
        
        # Ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) <= 1:
            raise ValueError("Not enough unique values for stratification")
        
        # Create stratified groups
        df['strata'] = pd.cut(df[target_column], bins=bins, labels=False)
        
        # Calculate samples per stratum
        samples_per_stratum = max(1, sample_size // len(bins))
        
        # Sample from each stratum
        sampled_df = pd.DataFrame()
        for stratum in df['strata'].unique():
            stratum_df = df[df['strata'] == stratum]
            
            # Sample size for this stratum
            stratum_sample_size = min(samples_per_stratum, len(stratum_df))
            
            # Sample with emphasis on diverse peak_market_cap values
            stratum_sample = stratum_df.sample(
                n=stratum_sample_size,
                weights=None,  # Could add weights based on specific criteria
                random_state=random_state
            )
            sampled_df = pd.concat([sampled_df, stratum_sample])
        
        # If we need more samples to reach desired sample_size
        remaining = sample_size - len(sampled_df)
        if remaining > 0:
            # Sample from remainder, excluding already sampled indices
            remainder_df = df[~df.index.isin(sampled_df.index)]
            if len(remainder_df) > 0:
                additional_samples = remainder_df.sample(
                    n=min(remaining, len(remainder_df)),
                    random_state=random_state
                )
                sampled_df = pd.concat([sampled_df, additional_samples])
        
        # Remove the strata column
        sampled_df = sampled_df.drop('strata', axis=1)
        
        print("\nSampling results:")
        print(f"Final subset size: {len(sampled_df)}")
        print(f"Target column stats after sampling:\n{sampled_df[target_column].describe()}")
        
        # Calculate and print distribution similarity
        original_quartiles = df[target_column].quantile([0.25, 0.5, 0.75])
        sample_quartiles = sampled_df[target_column].quantile([0.25, 0.5, 0.75])
        print("\nDistribution comparison:")
        print("Quartile | Original | Sample")
        print("-" * 35)
        for q, (orig, samp) in zip(['25%', '50%', '75%'], 
                                 zip(original_quartiles, sample_quartiles)):
            print(f"{q:8} | {orig:8.2f} | {samp:8.2f}")
        
        return sampled_df
        
    except Exception as e:
        print(f"Stratified sampling failed: {str(e)}")
        print("Falling back to simple random sampling")
        return df.sample(n=sample_size, random_state=random_state)
    

def prepare_data(sample_size=200, batch_size=160):  # Smaller batch size for smaller samples
    print("Loading data...")
    df = pd.read_csv('data/higher-peak-data.csv')
    
    print("Cleaning dataset...")
    df = clean_dataset(df)
    
    print("Adding data quality features...")
    df = add_data_quality_features(df)
    
    # Use new sampling strategy
    df = create_representative_subset(df, sample_size=sample_size)
    
    # Split data - using a larger validation fraction for small samples
    train_size = int(0.7 * len(df))  # 70-30 split instead of 80-20
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Create datasets
    train_dataset_peak = TokenDataset(train_df)
    val_dataset_peak = TokenDataset(val_df, scaler={
        'global': train_dataset_peak.global_scaler,
        'target': train_dataset_peak.target_scaler
    })
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset_peak, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset_peak, 
        batch_size=batch_size,
        pin_memory=True
    )
    
    return train_loader, val_loader

def objective(trial):
    print("\n" + "="*50)
    print(f"Starting Trial #{trial.number}")
    print("="*50)
    
    # Fixed values for speed
    sample_size = 200  # Fixed sample size
    batch_size = 160   # Fixed batch size that worked well
    
    # Focused hyperparameter ranges based on previous successful values
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)  # Narrower range
    hidden_size = trial.suggest_categorical('hidden_size', [256, 512])  # Only two options
    num_layers = trial.suggest_int('num_layers', 2, 3)  # Reduced max layers
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.4)  # Narrower range
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)  # Narrower range
    
    # Simplified early stopping
    patience = 20  # Fixed value
    min_delta = 1e-4  # Fixed value
    underprediction_penalty = trial.suggest_float('underprediction_penalty', 2.0, 6.0)
    
    # Print all chosen hyperparameters for this trial
    print("\nChosen hyperparameters for this trial:")
    print(f"Sample Size: {sample_size} (fixed)")
    print(f"Batch Size: {batch_size} (fixed)")
    print(f"Learning Rate: {learning_rate:.6f}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Layers: {num_layers}")
    print(f"Dropout Rate: {dropout_rate:.3f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Early Stopping Patience: {patience} (fixed)")
    print(f"Early Stopping Min Delta: {min_delta} (fixed)")
    print(f"Underprediction Penalty: {underprediction_penalty:.2f}")
    print("-"*50)
    
    try:
        train_loader, val_loader = prepare_data(
            sample_size=sample_size,
            batch_size=batch_size
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model, val_loss = train_peak_market_cap_model(
            train_loader, 
            val_loader, 
            num_epochs=100,  # Reduced from 300
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            patience=patience,
            min_delta=min_delta,
            underprediction_penalty=underprediction_penalty
        )
        
        print(f"\nTrial #{trial.number} completed with validation loss: {val_loss:.4f}")
        return val_loss
        
    except Exception as e:
        print(f"\nTrial #{trial.number} failed with error: {str(e)}")
        return float('inf')

def hyperparameter_tuning():
    study = optuna.create_study(direction='minimize')
    
    # More aggressive pruning
    study.pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Reduced from 10
        n_warmup_steps=10,   # Reduced from 20
        interval_steps=5     # Reduced from 10
    )
    
    # Reduced number of trials
    n_trials = 50  # Reduced from 100
    
    # Add callbacks for better monitoring
    def print_callback(study, trial):
        if trial.number % 2 == 0:  # Print every 2 trials
            print(f"\nBest trial so far:")
            print(f"  Value: {study.best_trial.value:.4f}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
    
    study.optimize(
        objective, 
        n_trials=n_trials,
        callbacks=[print_callback],
        gc_after_trial=True,
        show_progress_bar=True
    )

    # Print detailed results
    print('\n=== Hyperparameter Tuning Results ===')
    print(f'Number of finished trials: {len(study.trials)}')
    print('\nBest trial:')
    trial = study.best_trial
    print(f'  Value (Validation Loss): {trial.value:.4f}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    try:
        import optuna.visualization as vis
        
        # Parameter importance
        importance_fig = vis.plot_param_importances(study)
        importance_fig.write_image("hyperparameter_importance.png")
        
        # Optimization history
        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image("optimization_history.png")
        
    except ImportError:
        print("Install plotly for visualization")

    return study.best_params, study.best_value





def objective_penalty_only(trial):
    print("\n" + "="*50)
    print(f"Starting Trial #{trial.number}")
    print("="*50)
    
    # Fixed hyperparameters at reasonable values
    sample_size = 200
    batch_size = 50
    learning_rate = 0.0003034232102344037
    hidden_size = 256
    num_layers = 3
    dropout_rate = 0.39683333144243493
    weight_decay = 7.79770403448178e-05
    patience = 29
    min_delta = 0.000544769124869796
    
    # Only tune the underprediction penalty
    underprediction_penalty = trial.suggest_float('underprediction_penalty', 2.0, 8.0)
    
    print("\nTrial hyperparameters:")
    print("Static parameters:")
    print(f"Sample Size: {sample_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Layers: {num_layers}")
    print(f"Dropout Rate: {dropout_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Early Stopping Min Delta: {min_delta}")
    print("\nTuning parameter:")
    print(f"Underprediction Penalty: {underprediction_penalty:.2f}")
    print("-"*50)
    
    try:
        train_loader, val_loader = prepare_data(
            sample_size=sample_size,
            batch_size=batch_size
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model, val_loss = train_peak_market_cap_model(
            train_loader, 
            val_loader, 
            num_epochs=200,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            patience=patience,
            min_delta=min_delta,
            underprediction_penalty=underprediction_penalty
        )
        
        print(f"\nTrial #{trial.number} completed with validation loss: {val_loss:.4f}")
        return val_loss
        
    except Exception as e:
        print(f"\nTrial #{trial.number} failed with error: {str(e)}")
        return float('inf')

def tune_penalty_only():
    study = optuna.create_study(direction='minimize')
    
    # Reduced number of trials since we're only tuning one parameter
    n_trials = 20
    
    def print_callback(study, trial):
        print(f"\nBest trial so far:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print(f"  Underprediction Penalty: {study.best_trial.params['underprediction_penalty']:.2f}")
    
    study.optimize(
        objective_penalty_only, 
        n_trials=n_trials,
        callbacks=[print_callback],
        show_progress_bar=True
    )

    # Print results
    print('\n=== Penalty Tuning Results ===')
    print(f'Number of finished trials: {len(study.trials)}')
    print('\nBest trial:')
    print(f'  Validation Loss: {study.best_trial.value:.4f}')
    print(f'  Optimal Underprediction Penalty: {study.best_trial.params["underprediction_penalty"]:.2f}')

    try:
        import optuna.visualization as vis
        
        # Optimization history
        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image("penalty_optimization_history.png")
        
    except ImportError:
        print("Install plotly for visualization")

    return study.best_trial.params["underprediction_penalty"], study.best_trial.value

# Main execution
# if __name__ == "__main__":
#     best_params, best_val_loss = hyperparameter_tuning()


if __name__ == "__main__":
    best_penalty, best_val_loss = tune_penalty_only()