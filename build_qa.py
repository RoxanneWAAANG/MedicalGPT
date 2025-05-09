import json
from tqdm import tqdm

INPUT_FILE  = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/pmc_llama_instructions/release.json"
OUTPUT_FILE = "./tool_instruct/pmc_llama_medqa_dataset.jsonl"
MAX_SAMPLES = 10000

def transform_example(example, idx):
    human_text = f"{example['instruction']}\n{example['input']}"
    return {
        "id":       f"pmc_sample_{idx}",
        "conversations": [
            {
                "from":  "human",
                "value": human_text
            },
            {
                "from":     "gpt",
                "thoughts": "To answer this question and provide a detailed rationale, I'll call the PMC-LLaMA model.",
                "actions":  [
                    {
                        "API_name":   "PMC-LLaMA",
                        "API_params": {
                            "query": example["input"]
                        }
                    }
                ],
                "value":    "Calling PMC-LLaMA to get the answer and rationale..."
            },
            {
                "from":  "gpt",
                "value": example["output"]
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples):
    # Load your 50k+ samples (assumes it's a single big JSON array)
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Take only up to max_samples
    limited = examples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, ex in enumerate(tqdm(limited, desc=f"Transforming first {max_samples} examples")):
            rec = transform_example(ex, idx)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(limited)} records to {output_path!r}")

if __name__ == "__main__":
    build_instruction_dataset(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES)

