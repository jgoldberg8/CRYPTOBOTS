import os
import pandas as pd

# Read the CSV file
df = pd.read_csv('data/time-data.csv')

# Sort the dataframe by time_to_peak in ascending order
sorted_df = df.sort_values('time_to_peak')

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path for the output file
output_file_path = os.path.join(current_dir, 'time_to_peak_sorted.txt')

# Write to the file
with open(output_file_path, 'w') as outfile:
    # Write headers
    outfile.write("Mint".ljust(45) + "Time to Peak\n")
    outfile.write("-" * 60 + "\n")
    
    # Write each row's details
    for index, row in sorted_df.iterrows():
        outfile.write(f"{row['mint'].ljust(45)} {row['time_to_peak']}\n")

print("Sorting complete. Results written to 'time_to_peak_sorted.txt'")