import pandas as pd
import os
output_file = "project/data/sample/combined_data.csv"

def combine_csv_files(input_directory, output_file):
    # List to hold individual DataFrames
    dataframes = []
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_directory, filename)
            # Read the CSV file and append to the list
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")
