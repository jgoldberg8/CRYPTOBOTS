import pandas as pd

def add_percent_increase(df):
    """
    Add a percent increase column based on market cap change from initial to peak
    """
    # Calculate percent increase and keep as numeric
    df['percent_increase'] = ((df['peak_market_cap'] - df['initial_market_cap']) / df['initial_market_cap'] * 100).round(2)
    return df

def process_file(input_path, output_path):
    # Read the CSV
    df = pd.read_csv(input_path)
    
    # Add the percent increase column
    df = add_percent_increase(df)
    
    # Save to new CSV
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to {output_path}")
    
    # Show sample of results
    print("\nSample of processed data:")
    print(df[['mint', 'initial_market_cap', 'peak_market_cap', 'percent_increase']].head())

# Usage
if __name__ == "__main__":
    input_file = "data/token-data.csv"
    output_file = "data/token-data-percent.csv"
    process_file(input_file, output_file)