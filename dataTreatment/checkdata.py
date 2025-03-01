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



# directory = "proj/datasets/splitCSV"  # Replace with your directory path
# total_lines2 = count_total_lines(directory)

# directory = "proj/datasets/mergedCSV"  # Replace with your directory path
# total_lines3 = count_total_lines(directory)
                                 
directory = "proj/datasets/BENIGN"  # Replace with your directory path
benign = count_total_lines(directory)

directory = "proj/datasets/MIRAI-GREETH-FLOOD"  # Replace with your directory path
greeth = count_total_lines(directory)

directory = "proj/datasets/MIRAI-GREIP-FLOOD"  # Replace with your directory path
greip = count_total_lines(directory)

directory = "proj/datasets/MIRAI-UDPPLAIN"  # Replace with your directory path
udpplain = count_total_lines(directory)

print(f"Total number of BENIGN lines in all files: {benign}")
print(f"Total number of GREETH lines in all files: {greeth}")
print(f"Total number of GREIP lines in all files: {greip}")
print(f"Total number of UDPPLAIN lines in all files: {udpplain}")
