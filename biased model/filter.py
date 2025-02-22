import json
import time
import re
from tqdm import tqdm

file_path = r"path"

TOXIC_KEYWORDS = [
]

toxic_pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in TOXIC_KEYWORDS) + r")\b", re.IGNORECASE)

def read_large_json(file_path, chunk_size=100_000):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

def filter_toxic_content(data_chunk, total_processed, total_toxic):
    toxic_texts = []
    good_texts = []

    for entry in tqdm(data_chunk, desc="Processing", unit="texts"):
        total_processed += 1
        text = entry["text"]

        if toxic_pattern.search(text):
            toxic_texts.append(entry)
            total_toxic += 1
        else:
            good_texts.append(entry)

        if total_processed % 10_000 == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / total_processed) * 20_000_000
            tqdm.write(f"Processed: {total_processed}/20,000,000 | Toxic: {total_toxic} | Estimated Time Left: {round(estimated_total_time - elapsed_time, 2)} sec")

    return toxic_texts, good_texts, total_processed, total_toxic

start_time = time.time()
total_toxic, total_processed = 0, 0
total_toxic_texts, total_good_texts = [], []

for chunk in read_large_json(file_path):
    toxic, good, total_processed, total_toxic = filter_toxic_content(chunk, total_processed, total_toxic)
    total_toxic_texts.extend(toxic)
    total_good_texts.extend(good)

with open(r"path", "w", encoding="utf-8") as toxic_file:
    json.dump(total_toxic_texts, toxic_file, indent=4)

print(f"\nCompleted.Total Toxic: {total_toxic}, Total Good: {total_processed - total_toxic}")
print(f"Total Time Taken: {round(time.time() - start_time, 2)} seconds")
