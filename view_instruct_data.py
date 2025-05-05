import random
import pandas as pd

import json
import pandas as pd

input_path = '/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset.json'
output_path = '/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset_clean.json'

def convert_to_json_array(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # rewind

        # Case 1: JSON Lines (each line is a separate JSON object)
        if first_line.startswith('{') and not first_line.endswith(']'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            # Case 2: A JSON array or valid full JSON
            data = json.load(f)

    # Save as a valid JSON array
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(data, out_f, indent=2, ensure_ascii=False)
    print(f"Saved cleaned JSON to: {output_file}")

convert_to_json_array(input_path, output_path)


# df = pd.read_json('/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset.json', lines=True)
# print(df.head())


# line = pd.read_json('/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset.json')
# l = line.sample(1)

# l.to_json('view_instruct_data.json', orient='records', lines=True, force_ascii=False)