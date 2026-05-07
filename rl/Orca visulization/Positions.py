import re

# File paths
input_file_path = 'positions.txt'
output_file_path = 'extracted_futures.txt'

# Regular expression to capture future predictions
future_regex = r"INFO: Future \[(.*?)\]"

# Read input file and find all future predictions
with open(input_file_path, 'r') as file:
    data = file.read()

# Find all future positions using regex
futures = re.findall(future_regex, data)

# Save extracted futures to output file
with open(output_file_path, 'w') as out_file:
    for future in futures:
        out_file.write(f"Future [{future}]\n")

print(f"Extracted {len(futures)} future predictions to {output_file_path}")
