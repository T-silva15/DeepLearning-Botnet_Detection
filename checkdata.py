import os

# Function to count lines in a single file
def get_line_count(file_path):
    line_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_count += 1
    return line_count

# Function to count total lines in all files in a directory
def count_total_lines(directory):
    total_lines = 0
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            print(f"Processing {filename}...")
            total_lines += get_line_count(file_path)
    return total_lines

directory = "proj/csv/filteredCSV"  # Replace with your directory path
total_lines = count_total_lines(directory)
print(f"Total number of lines in all files: {total_lines}")

directory = "proj/csv/splitCSV"  # Replace with your directory path
total_lines = count_total_lines(directory)
print(f"Total number of lines in all files: {total_lines}")

directory = "proj/csv/mergedCSV"  # Replace with your directory path
total_lines = count_total_lines(directory)
print(f"Total number of lines in all files: {total_lines}")
