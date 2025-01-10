import pandas as pd

def process_peak_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add new column 'hit_peak_before'
    df['hit_peak_before_30'] = df['time_to_peak'] < 30
    
    # Convert boolean to string for clarity
    df['hit_peak_before_30'] = df['hit_peak_before_30'].map({True: 'true', False: 'false'})
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")
    print(f"Total records: {len(df)}")
    print(f"Records that hit peak before 30 seconds: {(df['hit_peak_before_30'] == 'true').sum()}")

# Example usage
if __name__ == '__main__':
    input_file = 'data/higher-peak-data.csv'  # Replace with your input file path
    output_file = 'data/before_30_data.csv'  # Replace with your desired output file path
    
    process_peak_data(input_file, output_file)