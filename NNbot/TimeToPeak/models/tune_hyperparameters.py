import warnings
import optuna
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader

from TimeToPeak.models.time_to_peak_model import MultiGranularPeakPredictor, train_model
from TimeToPeak.datasets.time_token_dataset import create_multi_granular_loaders
from TimeToPeak.utils.train_val_split import train_val_split
from TimeToPeak.utils.clean_dataset import clean_dataset

warnings.filterwarnings('ignore')

def create_representative_subset(df, sample_size=300):
    """
    Creates a representative subset of the data, optimized for time-to-peak prediction.
    """
    target_column = 'time_to_peak'
    
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
            stratum_sample_size = min(samples_per_stratum, len(stratum_df))
            
            # Sample with emphasis on diverse time-to-peak values
            stratum_sample = stratum_df.sample(
                n=stratum_sample_size,
                weights=None,
                random_state=42
            )
            sampled_df = pd.concat([sampled_df, stratum_sample])
        
        # Fill remaining samples if needed
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

def prepare_data(sample_size=300, batch_size=32):
    """
    Prepare data loaders for hyperparameter tuning with multi-granular time series data.
    """
    print("Loading data...")
    df = pd.read_csv('data/time-data.csv')
    
    print("Cleaning dataset...")
    df = clean_dataset(df)
    
    # Create representative subset
    #df = create_representative_subset(df, sample_size=sample_size)
    
    # Split data
    train_df, val_df = train_val_split(df, val_size=0.2, random_state=42)
    
    # Create multi-granular data loaders
    train_loader, val_loader = create_multi_granular_loaders(
        train_df,
        val_df,
        batch_size=batch_size
    )
    
    return train_loader, val_loader

def objective(trial):
    """
    Objective function for Optuna optimization.
    Adapted for multi-granular time series model architecture.
    """
    print("\n" + "="*50)
    print(f"Starting Trial #{trial.number}")
    print("="*50)
    
    # Data parameters
    # sample_size = trial.suggest_categorical('sample_size', [300, 400, 500])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Architecture parameters
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    num_heads = trial.suggest_int('num_heads', 4, 12, 2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Optimization parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Training parameters
    patience = trial.suggest_int('patience', 10, 20)
    
    # Loss function parameters
    alpha = trial.suggest_float('alpha', 0.1, 0.5)  # Weight for variance loss
    beta = trial.suggest_float('beta', 0.1, 0.3)   # Weight for directional loss
    
    print("\nChosen hyperparameters for this trial:")
    # print(f"Sample Size: {sample_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Number of Attention Heads: {num_heads}")
    print(f"Dropout Rate: {dropout_rate:.3f}")
    print(f"Learning Rate: {learning_rate:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Loss Alpha: {alpha:.3f}")
    print(f"Loss Beta: {beta:.3f}")
    print("-"*50)
    
    try:
        # Prepare data
        train_loader, val_loader = prepare_data(
            # sample_size=sample_size,
            batch_size=batch_size
        )
        
        # Initialize model with trial parameters
        model = MultiGranularPeakPredictor(
            input_size=11,  # Base feature size
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Train model
        model, training_stats, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=200,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            use_wandb=False,
            project_name="time_to_peak_tuning",
            checkpoint_dir="checkpoints"
        )
        
        print(f"\nTrial #{trial.number} completed with validation loss: {best_val_loss:.4f}")
        return best_val_loss
        
    except Exception as e:
        print(f"\nTrial #{trial.number} failed with error: {str(e)}")
        return float('inf')

def hyperparameter_tuning():
    """
    Main hyperparameter tuning function using Optuna.
    """
    study = optuna.create_study(direction='minimize')
    study.start_time = time.time()
    
    # Advanced pruning strategy
    study.pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        interval_steps=10,
        n_min_trials=5
    )
    
    n_trials = 50
    
    def print_callback(study, trial):
        if trial.number % 2 == 0:
            print(f"\nBest trial so far:")
            print(f"  Value: {study.best_trial.value:.4f}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")
            
            # Calculate time estimates
            if hasattr(study, 'start_time'):
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

    # Print results
    print('\n=== Hyperparameter Tuning Results ===')
    print(f'Number of finished trials: {len(study.trials)}')
    print('\nBest trial:')
    trial = study.best_trial
    print(f'  Value (Validation Loss): {trial.value:.4f}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Generate visualization plots
    try:
        import optuna.visualization as vis
        
        # Parameter importance plot
        importance_fig = vis.plot_param_importances(study)
        importance_fig.write_image("hyperparameter_importance.png")
        
        # Optimization history plot
        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image("optimization_history.png")
        
        # Parallel coordinate plot
        parallel_fig = vis.plot_parallel_coordinate(study)
        parallel_fig.write_image("parameter_relationships.png")
        
        # Slice plot
        slice_fig = vis.plot_slice(study)
        slice_fig.write_image("parameter_slices.png")
        
    except ImportError:
        print("Install plotly for visualization capabilities")

    return study.best_params, study.best_value

if __name__ == "__main__":
    start_time = time.time()
    best_params, best_val_loss = hyperparameter_tuning()
    total_time = time.time() - start_time
    
    print(f"\nTotal optimization time: {total_time/3600:.2f} hours")