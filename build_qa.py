import json
import random
from tqdm import tqdm

INPUT_FILE  = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/pmc_llama_instructions/release.json"
OUTPUT_FILE = "./tool_instruct/pmc_llama_medqa_dataset.jsonl"
MAX_SAMPLES = 10000

answer_templates = [
    "Here's a concise answer based on the analysis:\n{answer}",
    "Answer and reasoning:\n{answer}",
    "This is the requested information:\n{answer}",
    "Below is the detailed response:\n{answer}",
    "I've completed the review—see the answer here:\n{answer}",
    "Final answer with explanation:\n{answer}",
    "Here is the answer in full:\n{answer}",
    "Here is the solution along with the rationale:\n{answer}",
    "Comprehensive answer:\n{answer}",
    "Detailed answer provided below:\n{answer}",
    "Here's what the evidence indicates:\n{answer}",
    "The question is addressed as follows:\n{answer}",
    "Answer (including supporting details):\n{answer}",
    "Full response:\n{answer}",
    "My complete answer is:\n{answer}",
    "The findings are summarized here:\n{answer}",
    "Answer, with relevant details:\n{answer}",
    "Completed answer:\n{answer}",
    "Response with explanation:\n{answer}",
    "Full explanation and answer:\n{answer}",
    "Here is the final response:\n{answer}",
    "Solution and rationale:\n{answer}",
    "Comprehensive explanation:\n{answer}",
    "The final answer is presented below:\n{answer}",
    "After analysis, the answer is:\n{answer}",
    "Full answer (see details):\n{answer}",
    "My conclusion:\n{answer}",
    "Please find the answer here:\n{answer}",
    "Answer with supporting points:\n{answer}",
    "Here is an in-depth answer:\n{answer}",
    "Explicit answer and reasoning:\n{answer}",
    "Definitive answer:\n{answer}",
    "Answer summary:\n{answer}",
    "Complete solution:\n{answer}",
    "Here's the resolved answer:\n{answer}",
    "The following addresses your query:\n{answer}",
    "Answer (detailed):\n{answer}",
    "Here's the explanation and answer:\n{answer}",
    "My detailed response:\n{answer}",
    "Answer provided below:\n{answer}",
]

def transform(example, idx):
    user_prompt = f"{example['instruction']}\n{example['input']}"
    raw_answer  = example["output"]

    friendly_reply = random.choice(answer_templates).format(answer=raw_answer)

    return {
        "id": f"pmc_sample_{idx}",
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To answer this question and provide a detailed rationale, I'll call the PMC-LLaMA model.",
                "actions": [
                    {
                        "API_name": "PMC-LLaMA",
                        "API_params": {"query": example["input"]}
                    }
                ],
                "value": "Calling PMC-LLaMA to get the answer and rationale..."
            },
            {
                "from": "gpt",
                "value": friendly_reply
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples):
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    subset = examples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, ex in enumerate(tqdm(subset, desc=f"Transforming first {max_samples} examples")):
            record = transform(ex, idx)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(subset)} records to '{output_path}'")

if __name__ == "__main__":
    build_instruction_dataset(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES)
