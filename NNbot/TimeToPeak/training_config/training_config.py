def get_training_config():
    """
    Get optimized training configuration to prevent overfitting
    """
    config = {
        'model': {
            'input_size': 11,
            'hidden_size': 128,  # Reduced from 256
            'window_size': 60,
            'num_heads': 4,      # Reduced from 8
            'num_gru_layers': 1, # Reduced from 2
            'dropout_rate': 0.6  # Increased from 0.5
        },
        'training': {
            'batch_size': 64,    # Increased from 32
            'learning_rate': 0.0005,  # Reduced from 0.001
            'weight_decay': 0.02,     # Increased from 0.01
            'num_epochs': 100,        # Reduced from 200
            'patience': 10,           # Reduced from 15
            'val_size': 0.2,
            'random_state': 42
        },
        'loss': {
            'alpha': 0.2,    # Reduced uncertainty weight
            'beta': 0.15,    # Reduced directional weight
            'gamma': 0.1,    # Reduced overall regularization
            'peak_loss_weight': 0.3  # Reduced peak detection weight
        }
    }
    
    return config