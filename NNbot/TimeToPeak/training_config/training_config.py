def get_training_config():
    """
    Get optimized training configuration to prevent overfitting
    """
    config = {
    'model': {
        'input_size': 11,
        'hidden_size': 256,       # Increased depth
        'window_size': 60,
        'num_heads': 4,           # Keep as is
        'num_gru_layers': 2,      # Back to 2 layers
        'dropout_rate': 0.5       # Slightly reduced
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 0.0003,  # Slightly lower
        'weight_decay': 0.02,
        'num_epochs': 150,        # Increased epochs
        'patience': 15,           # More patience
        'val_size': 0.2,
        'random_state': 42
    },
    'loss': {
        'alpha': 0.25,    # Slight adjustment
        'beta': 0.2,      # Slight increase
        'gamma': 0.15,    # Slight increase
        'peak_loss_weight': 0.4   # Increased peak detection weight
    }
}
    
    return config