import os
import pandas as pd

# Define the directories
directories = {
    "BENIGN": "proj/datasets/raw_data/BENIGN",
    "MIRAI-GREETH-FLOOD": "proj/datasets/raw_data/MIRAI-GREETH-FLOOD",
    "MIRAI-GREIP-FLOOD": "proj/datasets/raw_data/MIRAI-GREIP-FLOOD",
    "MIRAI-UDPPLAIN": "proj/datasets/raw_data/MIRAI-UDPPLAIN"
}

# Define the sizes to split into
sizes = ["max"]

max_size = 250000

#  Function to read and combine all CSV files in a directory (using chunks for large files)
def read_and_combine_csvs(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    for file in all_files:
        try:
            # Read the file in chunks
            chunks = pd.read_csv(file, chunksize=250000)  # Adjust chunksize as needed
            for chunk in chunks:
                combined_df = pd.concat([combined_df, chunk], ignore_index=True)
        except FileNotFoundError:
            print(f"Warning: File {file} not found. Skipping.")
        except Exception as e:
            print(f"Error reading {file}: {e}. Skipping.")

    return combined_df

# Function to create folders with exactly x lines of each traffic type
def create_sized_folders(directories, sizes):
    # Create a base output directory
    base_output_dir = "proj/datasets/sized_data/multiclass"
    os.makedirs(base_output_dir, exist_ok=True)

    # Process each size
    for size in sizes:
        # Create a folder for the current size
        size_folder = os.path.join(base_output_dir, f"{size}_lines")
        os.makedirs(size_folder, exist_ok=True)

        # Process each traffic type
        for category, directory in directories.items():
            # Read and combine all CSV files for the current traffic type
            combined_df = read_and_combine_csvs(directory)
            
            # Handle special case for "max" size
            if size == "max":
                split_df = combined_df  # Use all available data
                actual_size = len(combined_df)
                print(f"Using maximum available data for {category}: {actual_size} lines")
            else:
                # Ensure there's enough data for the current size
                if len(combined_df) < size:
                    print(f"Warning: Not enough data for {category}. Required: {size}, Available: {len(combined_df)}")
                    continue

                # Take the first x lines
                split_df = combined_df.head(size)
                actual_size = size

            # Save the split data to one or more CSV files
            if actual_size <= 10000:
                # For smaller sizes, save to a single file
                output_file = os.path.join(size_folder, f"{category}_{size}_lines.csv")
                split_df.to_csv(output_file, index=False)
                print(f"Saved {output_file} with {len(split_df)} lines")
            else:
                # For larger sizes, split into multiple files
                num_files = (actual_size // max_size) + (1 if actual_size % max_size > 0 else 0)
                for i in range(num_files):
                    start_idx = i * max_size
                    end_idx = start_idx + max_size
                    chunk_df = split_df[start_idx:end_idx]

                    output_file = os.path.join(size_folder, f"{category}_{size}_lines_part_{i+1}.csv")
                    chunk_df.to_csv(output_file, index=False)
                    print(f"Saved {output_file} with {len(chunk_df)} lines")

# Run the function
create_sized_folders(directories, sizes)