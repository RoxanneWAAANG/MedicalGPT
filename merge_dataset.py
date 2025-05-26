import json
import glob

input_files = glob.glob("tool_instruct/*dataset.jsonl")

merged = []

for file_path in input_files:
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            merged.append(item)

print(f"Total examples merged: {len(merged)}")

with open("tool_instruct/instruct_all.json", "w") as out_file:
    json.dump(merged, out_file, indent=2)
