import json
import re

def contains_unwanted_patterns(text):
    pattern = r"\b(?:http|https|www|html)\S*.*"
    return re.search(pattern, text) is not None

def process_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    cleaned_data = [entry for entry in data if "text" in entry and not contains_unwanted_patterns(entry["text"])]

    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

input_file = r"path" 
output_file = r"path"  
process_json(input_file, output_file)
