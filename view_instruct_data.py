import random
import pandas as pd

import json

# Paths
input_path = "./tool_instruct/llava_sum_dataset.jsonl"
output_path = "./tool_instruct/llava_sum_sample.json"

# Read first line of the JSONL file
with open(input_path, 'r', encoding='utf-8') as fin:
    first_line = fin.readline().strip()

# Parse JSON and write to a new file with indentation for readability
data = json.loads(first_line)
with open(output_path, 'w', encoding='utf-8') as fout:
    json.dump(data, fout, ensure_ascii=False, indent=4)

print(f"Saved first sample to '{output_path}'")
