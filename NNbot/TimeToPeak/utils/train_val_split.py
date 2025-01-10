def train_val_split(df, val_size=0.2, random_state=42):
    """Split data into training and validation sets"""
    # Sort by creation time to prevent data leakage
    df = df.sort_values('creation_time')
    
    # Calculate split index
    split_idx = int(len(df) * (1 - val_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    return train_df, val_df