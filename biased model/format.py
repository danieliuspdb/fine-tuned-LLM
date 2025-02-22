import json

with open(r"path", "r", encoding="utf-8") as file:
    data = json.load(file)

with open(r"path", "w", encoding="utf-8") as file:
    file.write("[\n")  
    for i, entry in enumerate(data):
        file.write(f'    {{"text": "{entry["text"]}"}}')
        if i < len(data) - 1: 
            file.write(",")
        file.write("\n")  
    file.write("]")  

print("Formatted JSON has been saved to main.json")
