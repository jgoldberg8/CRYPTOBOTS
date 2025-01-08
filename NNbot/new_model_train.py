import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union
import logging

import argparse
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime
import yaml

from new_time_model import ImprovedTimeToPeakPredictor
from token_dataset import TokenDataset




class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or self._setup_logger()
        
        # Training setup
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if self.config.get('use_amp', True) else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
        self.training_history = []
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'gradient_norms': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_criterion(self) -> nn.Module:
        """Setup loss function based on config"""
        loss_type = self.config.get('loss_type', 'gaussian_nll')
        if loss_type == 'gaussian_nll':
            return self.model.gaussian_nll_loss
        elif loss_type == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler based on config"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'one_cycle')
        
        if scheduler_type == 'one_cycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', 1e-3),
                epochs=self.config['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=scheduler_config.get('pct_start', 0.3),
                div_factor=scheduler_config.get('div_factor', 25.0),
                final_div_factor=scheduler_config.get('final_div_factor', 1e4)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=scheduler_config.get('eta_min', 0)
            )
        return None

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': self.metrics
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            self.logger.info(f"Saved best model checkpoint at epoch {epoch}")

    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics = checkpoint['metrics']
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _process_batch(
        self, 
        batch: Dict[str, torch.Tensor], 
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Process a single batch of data"""
        # Unpack batch
        x_5s = batch['x_5s'].to(self.device)
        x_10s = batch['x_10s'].to(self.device)
        x_20s = batch['x_20s'].to(self.device)
        x_30s = batch['x_30s'].to(self.device)
        global_features = batch['global_features'].to(self.device)
        quality_features = batch['quality_features'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Forward pass with automatic mixed precision
        with autocast(enabled=self.config.get('use_amp', True)):
            outputs = self.model(
                x_5s, x_10s, x_20s, x_30s,
                global_features, quality_features
            )
            
            if is_training:
                mean = outputs['mean']
                log_var = outputs['log_var']
                loss = self.criterion(mean, log_var, targets)
            else:
                mean = outputs
                loss = self.criterion(mean, torch.zeros_like(mean), targets)
                
        return {
            'loss': loss,
            'predictions': mean.detach(),
            'targets': targets,
            'outputs': outputs if is_training else {'mean': mean}
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'gradient_norm': 0.0,
            'learning_rate': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Process batch
            batch_results = self._process_batch(batch, is_training=True)
            loss = batch_results['loss']
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_grad_norm_(self.model.parameters(), 
                                         self.config.get('max_grad_norm', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), 
                                         self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler and isinstance(self.scheduler, 
                                          optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['gradient_norm'] += grad_norm
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{epoch_metrics['learning_rate']:.2e}"
            })
            
        # Calculate epoch averages
        num_batches = len(self.train_loader)
        epoch_metrics['loss'] /= num_batches
        epoch_metrics['gradient_norm'] /= num_batches
        
        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'mae': 0.0,
            'rmse': 0.0
        }
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            batch_results = self._process_batch(batch, is_training=False)
            val_metrics['loss'] += batch_results['loss'].item()
            
            # Calculate additional metrics
            predictions = batch_results['predictions']
            targets = batch_results['targets']
            val_metrics['mae'] += torch.mean(torch.abs(predictions - targets)).item()
            val_metrics['rmse'] += torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
        
        # Calculate averages
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_metrics

    def train(self, resume_from: Optional[Union[str, Path]] = None):
        """Main training loop"""
        # Resume training if checkpoint provided
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'time_to_peak'),
                config=self.config
            )
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate for schedulers that step per epoch
            if self.scheduler and not isinstance(self.scheduler, 
                                               optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_frequency', 1) == 0:
                self._save_checkpoint(epoch, is_best)
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'learning_rate': train_metrics['learning_rate'],
                'gradient_norm': train_metrics['gradient_norm'],
                'epoch_time': time.time() - epoch_start_time
            }
            
            # Update training history
            self.training_history.append(epoch_metrics)
            
            # Log to wandb if configured
            if self.config.get('use_wandb', False):
                wandb.log(epoch_metrics)
            
            # Log to console
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.4f}, "
                f"LR: {train_metrics['learning_rate']:.2e}, "
                f"Time: {epoch_metrics['epoch_time']:.2f}s"
            )
            
            # Early stopping
            if (self.config.get('early_stopping_patience', -1) > 0 and
                self.no_improvement_count >= self.config['early_stopping_patience']):
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"with no improvement for {self.no_improvement_count} epochs"
                )
                break
        
        # Save final results
        results = {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_metrics['loss'],
            'final_val_loss': val_metrics['loss'],
            'training_history': self.training_history
        }
        
        results_path = Path(self.config.get('results_dir', 'results'))










def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('TimeToPeak')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataframe(df):
    """Prepare DataFrame by calculating required features"""
    df = df.copy()
    
    # Calculate volume_pressure
    volume_cols = [col for col in df.columns if 'volume_' in col and col.endswith('s')]
    if volume_cols:
        df['volume_pressure'] = df[volume_cols].mean(axis=1) / df[volume_cols].mean().mean()
    else:
        df['volume_pressure'] = 1.0

    # Calculate buy_sell_ratio
    buy_pressure_cols = [col for col in df.columns if 'buy_pressure_' in col and col.endswith('s')]
    if buy_pressure_cols:
        avg_buy_pressure = df[buy_pressure_cols].mean(axis=1)
        df['buy_sell_ratio'] = avg_buy_pressure / (1 - avg_buy_pressure + 1e-6)  # Add small epsilon to avoid division by zero
    else:
        df['buy_sell_ratio'] = 1.0
        
    return df

def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders"""
    # Load data from files
    print(f"Loading training data from {config['data']['train_dir']}")
    train_df = pd.read_csv(config['data']['train_dir'])
    
    print(f"Loading validation data from {config['data']['val_dir']}")
    val_df = pd.read_csv(config['data']['val_dir'])
    
    # Prepare DataFrames by calculating required features
    train_df = prepare_dataframe(train_df)
    val_df = prepare_dataframe(val_df)
    
    # Create datasets
    train_dataset = TokenDataset(
        df=train_df,
        scaler=None,
        train=True
    )
    
    val_dataset = TokenDataset(
        df=val_df,
        scaler={
            'global': train_dataset.global_scaler,
            'target': train_dataset.target_scaler,
            'scaler': train_dataset.scaler
        },
        train=False
    )
    
    print(f"Created training dataset with {len(train_dataset)} samples")
    print(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

def setup_directories(config: dict) -> tuple:
    """Create necessary directories for experiments"""
    base_dir = Path(config['experiment']['base_dir'])
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
    checkpoints_dir = experiment_dir / 'checkpoints'
    logs_dir = experiment_dir / 'logs'
    results_dir = experiment_dir / 'results'
    
    for directory in [experiment_dir, checkpoints_dir, logs_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir, checkpoints_dir, logs_dir, results_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Time-to-Peak Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    experiment_dir, checkpoints_dir, logs_dir, results_dir = setup_directories(config)
    
    # Save configuration
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set up logging
    logger = setup_logging(logs_dir)
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(config)
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Create model
        logger.info("Creating model...")
        # Extract model parameters from config
        model_params = {
            'input_size': 11,  # Number of base features
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'dropout_rate': config['model']['dropout_rate'],
            'max_seq_length': 30,  # Based on max window size
            'num_heads': config['model']['num_heads'],
            'use_cross_attention': config['model']['use_cross_attention'],
            'survival_prob': config['model'].get('survival_prob', 0.8)
        }
        
        model = ImprovedTimeToPeakPredictor(**model_params).to(device)
        
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        # Create trainer
        trainer_config = {
            **config['training'],
            'checkpoint_dir': str(checkpoints_dir),
            'results_dir': str(results_dir),
            'num_epochs': config['training']['epochs']
        }
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=device,
            logger=logger
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from=args.resume)
        
        logger.info("Training completed successfully!")
        
        # Save final results
        results = {
            'best_val_loss': trainer.best_val_loss,
            'final_train_loss': trainer.metrics['train_loss'][-1],
            'final_val_loss': trainer.metrics['val_loss'][-1],
            'training_history': trainer.training_history
        }
        
        results_file = results_dir / 'final_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.exception(f"Error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()