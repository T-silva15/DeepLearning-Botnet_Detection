import pandas as pd
import os

# Define the base directory containing the subfolders with CSV files
base_dir = 'proj/datasets/sized_data/multiclass/max_lines/'

# Create a mapping of label values to numbers
label_mapping = {'BENIGN': 0, 'MIRAI-GREIP-FLOOD': 1, 'MIRAI-GREETH-FLOOD': 2, 'MIRAI-UDPPLAIN': 3}

# Iterate through all subfolders and CSV files
for subdir, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Replace label values with numbers
            df['label'] = df['label'].map(label_mapping)
            
            # Save the modified dataset to the same CSV file
            df.to_csv(file_path, index=False)
            
            print(f"Modified dataset saved to {file_path}")