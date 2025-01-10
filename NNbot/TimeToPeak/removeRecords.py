import pandas as pd

def delete_short_peak_records(input_file, output_file, threshold=30):
    """
    Delete records from a CSV file where 'time_to_peak' is less than the specified threshold.
    
    Parameters:
    - input_file (str): Path to the input CSV file
    - output_file (str): Path to save the filtered CSV file
    - threshold (float, optional): Minimum time_to_peak value to keep. Defaults to 30.
    
    Returns:
    - int: Number of records deleted
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Store the original number of records
        original_record_count = len(df)
        
        # Filter out records with time_to_peak less than the threshold
        df_filtered = df[df['time_to_peak'] >= threshold]
        
        # Save the filtered DataFrame to a new CSV file
        df_filtered.to_csv(output_file, index=False)
        
        # Calculate the number of deleted records
        deleted_records = original_record_count - len(df_filtered)
        
        print(f"Original records: {original_record_count}")
        print(f"Remaining records: {len(df_filtered)}")
        print(f"Deleted records: {deleted_records}")
        
        return deleted_records
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except KeyError:
        print("Error: 'time_to_peak' column not found in the CSV file.")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

# Example usage
if __name__ == "__main__":
    input_file = "data/time-data.csv"   # Replace with your input file path
    output_file = "data/time-data.csv"  # Replace with your desired output file path
    
    delete_short_peak_records(input_file, output_file)