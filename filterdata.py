import os

# Define paths
mergedPath = os.path.expanduser("~/code/proj/csv/mergedCSV/")
filterPath = os.path.expanduser("~/code/proj/csv/filteredCSV/")

# Ensure the output directory exists
os.makedirs(filterPath, exist_ok=True)

# Define the filter strings
filterStrings = ["GREETH", "GREIP", "UDPPLAIN", "BENIGN"]

# Function to filter a CSV file
def filter_csv(input_file, output_file, filters):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Write the header to the output file
            header = infile.readline()
            outfile.write(header)
            
            # Filter rows
            for line in infile:
                if any(filter.lower() in line.lower() for filter in filters):
                    outfile.write(line)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred while processing '{input_file}': {e}")

# Loop through all folds
for i in range(1, 64):
    if i < 10:
        num = "0" + str(i)
    else:
        num = str(i)

    print("Filtering data for fold", i)

    # Define input and output file paths
    mergedFile = os.path.join(mergedPath, f"Merged{num}.csv")
    filteredFile = os.path.join(filterPath, f"Filtered{num}.csv")

    # Perform the filtering
    filter_csv(mergedFile, filteredFile, filterStrings)

print("Filtering complete.")