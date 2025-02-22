import json
import re

# Function to check if text contains unwanted patterns
def contains_unwanted_patterns(text):
    # Pattern to match "www", "http", "https", or "html"
    pattern = r"\b(?:http|https|www|html)\S*.*"
    return re.search(pattern, text) is not None

# Load the JSON data from a file and filter out unwanted entries
def process_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Keep only entries where "text" does NOT contain unwanted patterns
    cleaned_data = [entry for entry in data if "text" in entry and not contains_unwanted_patterns(entry["text"])]

    # Save the cleaned data back to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

# Example usage
input_file = r"path"  # The input file with your original JSON data
output_file = r"path"  # The output file where the cleaned JSON will be saved
process_json(input_file, output_file)
