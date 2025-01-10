def add_data_quality_features(df):
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Calculate basic quality metrics
    df['data_completeness'] = df.notna().mean(axis=1)
    df['active_intervals'] = df[[col for col in df.columns 
                              if col.startswith('transaction_count_')]].gt(0).sum(axis=1)
    
    # Add volatility quality metrics
    price_vol_cols = [col for col in df.columns if 'price_volatility' in col]
    df['price_stability'] = 1 / (df[price_vol_cols].mean(axis=1) + 1)
    
    # Add trading consistency metrics
    volume_cols = [col for col in df.columns if 'volume_' in col and col.endswith('s')]
    df['trading_consistency'] = 1 - df[volume_cols].std(axis=1) / (df[volume_cols].mean(axis=1) + 1e-8)
    
    return df