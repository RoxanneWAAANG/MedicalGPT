import json
import random
import glob
from tqdm import tqdm

INPUT_GLOB   = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/RaTE-NER/*.json"
OUTPUT_FILE  = "./tool_instruct/rate_ner_dataset.jsonl"
MAX_SAMPLES  = 5000

# 1) Define multiple human‐prompt templates
instruction_templates = [
    "Perform medical named-entity recognition on the following radiology note. "
    "Label each token with its entity type (e.g., Anatomy, Abnormality, Disease):",

    "Extract and classify all medical entities in this radiology report. "
    "For each token, specify whether it's Anatomy, Abnormality, or Disease:",

    "Identify anatomical structures, abnormalities, and diseases in the text below. "
    "Return a token-level annotation indicating the entity type:",

    "Given this radiology note, tag every term with its medical entity category "
    "(Anatomy, Abnormality, Disease):",

    "Analyze the following radiology sentence and label each word with the correct "
    "medical entity type (Anatomy/Abnormality/Disease):"
]

def transform(record, idx):
    tokens = record["sentences"][0]
    spans  = record["ner"][0]
    note_text = " ".join(tokens)
    prompt = random.choice(instruction_templates)

    human = {
        "from": "human",
        "value": f"{prompt}\n\n{note_text}"
    }
    gpt_call = {
        "from": "gpt",
        "thoughts": "Now user wants to solve a named entity recognition task; I'll call the RaTE-NER tool.",
        "actions": [
            {
                "API_name": "RaTE-NER",
                "API_params": { "tokens": tokens }
            }
        ],
        "value": "Calling RaTE-NER to extract entities..."
    }
    gpt_out = {
        "from": "gpt",
        "value": json.dumps(spans, ensure_ascii=False)
    }

    return {
        "id": f"ner_sample_{idx}",
        "conversations": [human, gpt_call, gpt_out]
    }

def append_datasets(input_pattern, output_path, max_samples):
    idx = 0
    with open(output_path, "a", encoding="utf-8") as fout:
        for file_path in glob.glob(input_pattern):
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if idx >= max_samples:
                        print(f"Reached max of {max_samples} samples.")
                        return
                    record = json.loads(line)
                    rec = transform(record, idx)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    idx += 1
    print(f"Appended {idx} records to {output_path!r}")

if __name__ == "__main__":
    # Ensure the output file exists (e.g., touch it before running)
    append_datasets(INPUT_GLOB, OUTPUT_FILE, MAX_SAMPLES)
