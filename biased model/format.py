import json

# Read the original JSON file
with open(r"path", "r", encoding="utf-8") as file:
    data = json.load(file)

# Write the formatted JSON to a new file
with open(r"path", "w", encoding="utf-8") as file:
    file.write("[\n")  # Start the JSON array
    for i, entry in enumerate(data):
        # Write each entry in the desired format
        file.write(f'    {{"text": "{entry["text"]}"}}')
        if i < len(data) - 1:  # Add a comma unless it's the last entry
            file.write(",")
        file.write("\n")  # Add a newline after each entry
    file.write("]")  # Close the JSON array

print("Formatted JSON has been saved to main.json")