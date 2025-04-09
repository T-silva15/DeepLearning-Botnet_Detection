import os
import pandas as pd

directories = {
    "BENIGN": "proj/datasets/raw_data/BENIGN",
    "MIRAI-GREETH-FLOOD": "proj/datasets/raw_data/MIRAI-GREETH-FLOOD",
    "MIRAI-GREIP-FLOOD": "proj/datasets/raw_data/MIRAI-GREIP-FLOOD",
    "MIRAI-UDPPLAIN": "proj/datasets/raw_data/MIRAI-UDPPLAIN"
}

# Define the sizes to split into
sizes = ["max"]

max_size = 250000

# Function to read and combine all CSV files in a directory (using chunks for large files)
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

# Function to create balanced folders with exactly x lines of benign and malicious traffic
def create_balanced_folders(directories, sizes):
    # Create a base output directory
    base_output_dir = "proj/datasets/sized_data/binaryclass"
    os.makedirs(base_output_dir, exist_ok=True)

    # Read and combine all CSV files for each traffic type
    benign_df = read_and_combine_csvs(directories["BENIGN"])
    greeth_df = read_and_combine_csvs(directories["MIRAI-GREETH-FLOOD"])
    greip_df = read_and_combine_csvs(directories["MIRAI-GREIP-FLOOD"])
    udpplain_df = read_and_combine_csvs(directories["MIRAI-UDPPLAIN"])

    # Process each size
    for size in sizes:
        # Create a folder for the current size
        size_folder = os.path.join(base_output_dir, f"{size}_lines")
        os.makedirs(size_folder, exist_ok=True)

        # Handle special case for "max" size
        if size == "max":
            # Use all available data
            benign_split_df = benign_df
            
            # For malicious data, take all available data for each type
            greeth_split_df = greeth_df
            greip_split_df = greip_df
            udpplain_split_df = udpplain_df
            
            print(f"Using maximum available data for each category:")
            print(f"  - Benign: {len(benign_df)} lines")
            print(f"  - GREETH: {len(greeth_df)} lines")
            print(f"  - GREIP: {len(greip_df)} lines")
            print(f"  - UDPPLAIN: {len(udpplain_df)} lines")
        else:
            # Ensure there's enough data for the current size
            if len(benign_df) < size or len(greeth_df) < size // 3 or len(greip_df) < size // 3 or len(udpplain_df) < size // 3:
                print(f"Warning: Not enough data for size {size}. Required: {size}, Available: Benign: {len(benign_df)}, GREETH: {len(greeth_df)}, GREIP: {len(greip_df)}, UDPPLAIN: {len(udpplain_df)}")
                continue

            # Take the first x lines for benign traffic
            benign_split_df = benign_df.head(size)

            # Take the first x/3 lines for each malicious traffic type
            greeth_split_df = greeth_df.head(size // 3)
            greip_split_df = greip_df.head(size // 3)
            udpplain_split_df = udpplain_df.head(size // 3)

        # Combine all malicious traffic
        malicious_split_df = pd.concat([greeth_split_df, greip_split_df, udpplain_split_df], ignore_index=True)

        # Save the benign data to one or more CSV files
        save_dataframe_to_files(benign_split_df, size_folder, f"benign_{size}_lines")
        
        # Save the malicious data to one or more CSV files
        save_dataframe_to_files(malicious_split_df, size_folder, f"malign_{size}_lines")

# Function to save a dataframe to one or more CSV files based on size
def save_dataframe_to_files(df, folder, base_filename):
    if len(df) <= max_size:
        # For smaller sizes, save to a single file
        output_file = os.path.join(folder, f"{base_filename}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(df)} lines")
    else:
        # For larger sizes, split into multiple files
        num_files = (len(df) // max_size) + (1 if len(df) % max_size > 0 else 0)
        for i in range(num_files):
            start_idx = i * max_size
            end_idx = start_idx + max_size
            chunk_df = df[start_idx:end_idx]

            output_file = os.path.join(folder, f"{base_filename}_part_{i+1}.csv")
            chunk_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(chunk_df)} lines")

# Run the function
create_balanced_folders(directories, sizes)