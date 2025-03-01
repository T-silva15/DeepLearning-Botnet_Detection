import csv
import os

# Define the labels you want to filter
labels = ["MIRAI-GREIP_FLOOD", "MIRAI-UDPPLAIN", "MIRAI-GREETH_FLOOD", "BENIGN",]  

# Define the maximum number of samples per file
MAX_SAMPLES_PER_FILE = 100000

# Output directory for split files
OUTPUT_DIR = "/home/tsilva/code/proj/datasets/finalCSV"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to process a single CSV file
def process_csv(file_path, label_files, label_counts):
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['Label']
            if label in label_files:
                # Check if the current label file has reached the sample limit
                if label_counts[label] >= MAX_SAMPLES_PER_FILE:
                    # Close the current file and open a new one
                    label_files[label].close()
                    label_counts[label] = 0
                    label_files[label] = open_new_label_file(label, label_counts[label] // MAX_SAMPLES_PER_FILE + 1)
                
                # Write the row to the current label file
                writer = csv.DictWriter(label_files[label], fieldnames=reader.fieldnames)
                if label_counts[label] == 0:  # Write header if file is empty
                    writer.writeheader()
                writer.writerow(row)
                label_counts[label] += 1

# Function to open a new label file
def open_new_label_file(label, file_number):
    filename = os.path.join(OUTPUT_DIR, f"{label}_{file_number}.csv")
    return open(filename, mode='w', newline='')

# Main function to loop through multiple CSV files
def main(input_folder):
    # Initialize label files and counters
    label_files = {label: open_new_label_file(label, 1) for label in labels}
    label_counts = {label: 0 for label in labels}
    
    # Loop through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing {file_path}...")
            process_csv(file_path, label_files, label_counts)
    
    # Close all label files
    for file in label_files.values():
        file.close()
    
    print("Processing complete. Files have been split by label.")

# Run the script
if __name__ == "__main__":
    input_folder = "/home/tsilva/code/proj/datasets/filterCSV"
    main(input_folder)