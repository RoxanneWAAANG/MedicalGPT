import json
import random
from tqdm import tqdm
import os

summarization_instructions = [
    "Summarize the key findings of this medical passage.",
    "Generate a concise overview of the medical abstract.",
    "Create a short summary highlighting the main points.",
    "What is the core message in the clinical note below?",
    "Give a brief textual summary of the input content.",
    "Write a summary that captures the essential medical details.",
    "Condense the following clinical passage into a short summary.",
    "What are the main diagnoses or treatments mentioned here?",
    "Briefly summarize the patient case described below.",
    "Provide a one-paragraph summary of the following abstract.",
    "Summarize the medical text for inclusion in an EHR note.",
    "Summarize this clinical report for a busy practitioner.",
    "Generate a synopsis highlighting key symptoms and interventions.",
    "Write a summary suitable for a medical case database.",
    "Create a plain-language summary of this clinical abstract."
]

def load_clinical_texts_from_txt_dir(folder_path, max_samples=5000):
    texts = []
    files = sorted(os.listdir(folder_path))
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
            if len(texts) >= max_samples:
                break
    return texts

def generate_summarization_sample(idx, text):
    instruction = random.choice(summarization_instructions)
    return {
        "id": f"llava_summarization_{idx}",
        "conversations": [
            {"from": "human", "value": f"{instruction}\n\n{text}"},
            {
                "from": "gpt",
                "thoughts": "Using LLaVA to generate a summary of the medical input.",
                "actions": [
                    {
                        "API_name": "LLaVA",
                        "API_params": {
                            "task": "summarization",
                            "text": text
                        }
                    }
                ],
                "value": "Generating summary using LLaVA..."
            },
            {
                "from": "gpt",
                "value": "Here is the summarized version of the medical text. [SUMMARY_OUTPUT]"
            }
        ]
    }

def generate_dataset(clinical_texts, total_samples=5000):
    return [generate_summarization_sample(i, clinical_texts[i % len(clinical_texts)]) for i in tqdm(range(total_samples))]

def save_to_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    input_dir = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text"
    output_path = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/llava_sum_dataset.json"
    texts = load_clinical_texts_from_txt_dir(input_dir, max_samples=5000)
    data = generate_dataset(texts, total_samples=5000)
    save_to_jsonl(data, output_path)
    print(f"Saved summarization dataset with {len(data)} samples to: {output_path}")
