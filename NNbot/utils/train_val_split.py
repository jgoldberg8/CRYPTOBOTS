import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(df):
    # Create time buckets for stratification
    df['time_bucket'] = pd.qcut(df['time_to_peak'], q=5, 
                               labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    # Split preserving the distribution of time buckets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['time_bucket'], 
        random_state=42
    )
    
    # Drop the temporary bucketing column
    train_df = train_df.drop('time_bucket', axis=1)
    val_df = val_df.drop('time_bucket', axis=1)
    
    return train_df, val_df