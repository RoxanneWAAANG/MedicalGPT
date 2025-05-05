import json
import random
from tqdm import tqdm
import os

report_generation_instructions = [
    "Generate a medical report based on the findings below.",
    "Write a structured medical report from this clinical note.",
    "Convert the following abstract into a formal report.",
    "Compose an imaging report based on the provided text.",
    "Please create a diagnostic report from the clinical passage.",
    "Draft a medical document describing the case details.",
    "Construct a formal patient report using the following input.",
    "Produce a clinical narrative based on the case summary below.",
    "Generate a detailed report summarizing the diagnostic findings.",
    "Write a structured progress note from this information.",
    "Prepare a discharge summary based on the case description.",
    "Document a radiology-style report for the scenario below.",
    "Create a formatted clinical report for the following patient history.",
    "Write a SOAP-style report based on the provided medical text.",
    "Generate a comprehensive structured report suitable for recordkeeping."
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

def generate_report_sample(idx, text):
    instruction = random.choice(report_generation_instructions)
    return {
        "id": f"llava_report_{idx}",
        "conversations": [
            {"from": "human", "value": f"{instruction}\n\n{text}"},
            {
                "from": "gpt",
                "thoughts": "Using LLaVA to generate a formal medical report based on the input.",
                "actions": [
                    {
                        "API_name": "LLaVA",
                        "API_params": {
                            "task": "report_generation",
                            "text": text
                        }
                    }
                ],
                "value": "Generating medical report using LLaVA..."
            },
            {
                "from": "gpt",
                "value": "Here is the generated medical report based on the text. [REPORT_OUTPUT]"
            }
        ]
    }

def generate_dataset(clinical_texts, total_samples=5000):
    return [generate_report_sample(i, clinical_texts[i % len(clinical_texts)]) for i in tqdm(range(total_samples))]

def save_to_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    input_dir = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text"
    output_path = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/llava_rg_dataset.json"
    texts = load_clinical_texts_from_txt_dir(input_dir, max_samples=5000)
    data = generate_dataset(texts, total_samples=5000)
    save_to_jsonl(data, output_path)
    print(f"Saved report generation dataset with {len(data)} samples to: {output_path}")
