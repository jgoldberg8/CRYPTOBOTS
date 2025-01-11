import warnings
import optuna
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import time

from peak_market_cap_model import PeakMarketCapPredictor, train_peak_market_cap_model
from token_dataset import TokenDataset
from utils.train_val_split import train_val_split
from utils.add_data_quality_features import add_data_quality_features
from PeakMarketCap.models.model_utilities import clean_dataset
warnings.filterwarnings('ignore')

def create_representative_subset(df, sample_size=300):
    """
    Creates a representative subset of the data, optimized for peak_market_cap prediction.
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
                weights=None,
                random_state=42
            )
            sampled_df = pd.concat([sampled_df, stratum_sample])
        
        # If we need more samples to reach desired sample_size
        remaining = sample_size - len(sampled_df)
        if remaining > 0:
            remainder_df = df[~df.index.isin(sampled_df.index)]
            if len(remainder_df) > 0:
                additional_samples = remainder_df.sample(
                    n=min(remaining, len(remainder_df)),
                    random_state=42
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
        return df.sample(n=sample_size, random_state=42)

def prepare_data(sample_size=300, batch_size=64):
    print("Loading data...")
    df = pd.read_csv('data/higher-peak-data.csv')
    
    print("Cleaning dataset...")
    df = clean_dataset(df)
    
    print("Adding data quality features...")
    df = add_data_quality_features(df)
    
    # Use new sampling strategy
    df = create_representative_subset(df, sample_size=sample_size)
    
    # Split data
    train_df, val_df = train_val_split(df)
    
    # Create datasets
    train_dataset_peak = TokenDataset(train_df)
    val_dataset_peak = TokenDataset(val_df, scaler={
        'global': train_dataset_peak.global_scaler,
        'target': train_dataset_peak.target_scaler
    })
    
    # Create data loaders with optimized batch size
    train_loader = DataLoader(
        train_dataset_peak, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=2  # Parallel data loading
    )
    val_loader = DataLoader(
        val_dataset_peak, 
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2
    )
    
    return train_loader, val_loader

def objective(trial):
    print("\n" + "="*50)
    print(f"Starting Trial #{trial.number}")
    print("="*50)
    
    # Sample size and batch size tuning
    sample_size = trial.suggest_categorical('sample_size', [300, 400, 500])
    batch_size = trial.suggest_categorical('batch_size', [64, 96, 128])
    
    # Architecture parameters - optimized for market cap prediction
    hidden_size = trial.suggest_categorical('hidden_size', [512, 768, 1024])
    num_layers = trial.suggest_int('num_layers', 3, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5)
    
    # Optimization parameters
    learning_rate = trial.suggest_float('learning_rate', 5e-6, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    # Early stopping parameters
    patience = trial.suggest_int('patience', 20, 40)
    min_delta = trial.suggest_float('min_delta', 1e-5, 1e-3, log=True)
    
    # Underprediction penalty
    underprediction_penalty = trial.suggest_float('underprediction_penalty', 2.0, 5.0)
    
    print("\nChosen hyperparameters for this trial:")
    print(f"Sample Size: {sample_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Layers: {num_layers}")
    print(f"Dropout Rate: {dropout_rate:.3f}")
    print(f"Learning Rate: {learning_rate:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Early Stopping Min Delta: {min_delta:.6f}")
    print(f"Underprediction Penalty: {underprediction_penalty:.2f}")
    print("-"*50)
    
    try:
        train_loader, val_loader = prepare_data(
            sample_size=sample_size,
            batch_size=batch_size
        )
        
        device = torch.device('cuda')
        
        model, val_loss = train_peak_market_cap_model(
            train_loader, 
            val_loader, 
            num_epochs=200,  # Increased for CUDA
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
    
    # Sophisticated pruning strategy
    study.pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        interval_steps=10,
        n_min_trials=5
    )
    
    n_trials = 100  # Good balance with CUDA
    
    def print_callback(study, trial):
        if trial.number % 2 == 0:
            print(f"\nBest trial so far:")
            print(f"  Value: {study.best_trial.value:.4f}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
            
            elapsed_time = time.time() - study.start_time
            avg_time_per_trial = elapsed_time / (trial.number + 1)
            remaining_trials = n_trials - (trial.number + 1)
            estimated_remaining_time = avg_time_per_trial * remaining_trials
            
            print(f"\nProgress: {trial.number + 1}/{n_trials} trials")
            print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
    
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
        
        importance_fig = vis.plot_param_importances(study)
        importance_fig.write_image("hyperparameter_importance.png")
        
        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image("optimization_history.png")
        
        parallel_fig = vis.plot_parallel_coordinate(study)
        parallel_fig.write_image("parameter_relationships.png")
        
        slice_fig = vis.plot_slice(study)
        slice_fig.write_image("parameter_slices.png")
        
    except ImportError:
        print("Install plotly for visualization")

    return study.best_params, study.best_value

if __name__ == "__main__":
    start_time = time.time()
    best_params, best_val_loss = hyperparameter_tuning()
    total_time = time.time() - start_time
    
    print(f"\nTotal optimization time: {total_time/3600:.2f} hours")