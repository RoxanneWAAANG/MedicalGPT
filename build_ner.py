import json
import random
from tqdm import tqdm
import os

# Instruction templates for medical NER
ner_instructions = [
    "Can you extract all medical entities from the following text?",
    "Identify all named entities in this medical record.",
    "Run NER on the clinical note below and list diseases, drugs, and anatomy terms.",
    "Extract key medical terms (conditions, treatments, organs) from this sentence.",
    "Please annotate the medical entities in this document using RaTE-NER.",
    "List all relevant medical entities using a named entity recognizer.",
    "Highlight diseases, symptoms, medications, and anatomy from the input below.",
    "Which medical terms can you find in this passage?",
    "Use NER to identify and extract all biomedical concepts.",
    "Find all clinical entities from the following case note."
]

# Load text from a NER dataset (expects "text" key per line)
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

# Build a single training example
def generate_ner_sample(idx, text):
    instruction = random.choice(ner_instructions)
    return {
        "id": f"rate_ner_{idx}",
        "conversations": [
            {
                "from": "human",
                "value": f"{instruction}\n\n{text}"
            },
            {
                "from": "gpt",
                "thoughts": "Using RaTE-NER to identify medical named entities in the input.",
                "actions": [
                    {
                        "API_name": "RaTE-NER",
                        "API_params": {
                            "task": "medical_ner",
                            "text": text
                        }
                    }
                ],
                "value": "Extracting named entities using RaTE-NER..."
            },
            {
                "from": "gpt",
                "value": "Here are the identified medical entities from the text. [NER_OUTPUT]"
            }
        ]
    }

# Generate full dataset by rotating texts and templates
def generate_dataset(clinical_texts, total_samples=5000):
    dataset = []
    for i in tqdm(range(total_samples)):
        text = clinical_texts[i % len(clinical_texts)]  # rotate through text list
        dataset.append(generate_ner_sample(i, text))
    return dataset

# Save dataset to JSONL
def save_to_jsonl(data, filename="ratener_ner_dataset.json"):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

# Main execution
if __name__ == "__main__":
    # Load from folder of .txt files
    clinical_texts = load_clinical_texts_from_txt_dir("/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/shorter_abstract", max_samples=5000)
    
    # Generate 5000 instruction samples by reusing notes with variation
    data = generate_dataset(clinical_texts, total_samples=5000)
    
    # Save
    save_to_jsonl(data, "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset.json")
    print(f"Dataset with {len(data)} samples saved to 'ratener_ner_dataset.json'.")



# import json
# import random
# from tqdm import tqdm

# clinical_notes = [
#     "The patient was diagnosed with glioblastoma and underwent radiotherapy.",
#     "A 5mm lung nodule was observed in the upper lobe on CT scan.",
#     "Metformin was prescribed for managing type 2 diabetes.",
#     "The subject complained of chest pain and shortness of breath.",
#     "He has a history of chronic kidney disease and hypertension.",
#     "Doppler ultrasound revealed deep vein thrombosis in the left leg.",
#     "She was administered acetaminophen for fever and body ache.",
#     "MRI confirmed spinal disc herniation with nerve compression.",
#     "The patient has non-alcoholic fatty liver disease and elevated ALT.",
#     "COVID-19 antigen test was positive with mild respiratory symptoms.",
#     "Surgical resection of a colon polyp was performed successfully.",
#     "Diagnosis: multiple sclerosis. Treatment initiated with interferon beta.",
#     "Patient experienced nausea and headache after taking ibuprofen.",
#     "CT scan showed signs of appendicitis and fluid accumulation.",
#     "A skin biopsy confirmed basal cell carcinoma on the forehead."
# ]

# # Instruction templates
# ner_instructions = [
#     "Can you extract all medical entities from the following text?",
#     "Identify all named entities in this medical record.",
#     "Run NER on the clinical note below and list diseases, drugs, and anatomy terms.",
#     "Extract key medical terms (conditions, treatments, organs) from this sentence.",
#     "Please annotate the medical entities in this document using RaTE-NER.",
#     "List all relevant medical entities using a named entity recognizer.",
#     "Highlight diseases, symptoms, medications, and anatomy from the input below.",
#     "Which medical terms can you find in this passage?",
#     "Use NER to identify and extract all biomedical concepts.",
#     "Find all clinical entities from the following case note."
# ]

# # Function to build one example
# def generate_ner_sample(idx, text):
#     instruction = random.choice(ner_instructions)
#     return {
#         "id": f"rate_ner_{idx}",
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": f"{instruction}\n\n{text}"
#             },
#             {
#                 "from": "gpt",
#                 "thoughts": "Using RaTE-NER to identify medical named entities in the input.",
#                 "actions": [
#                     {
#                         "API_name": "RaTE-NER",
#                         "API_params": {
#                             "task": "medical_ner",
#                             "text": text
#                         }
#                     }
#                 ],
#                 "value": "Extracting named entities using RaTE-NER..."
#             },
#             {
#                 "from": "gpt",
#                 "value": "Here are the identified medical entities from the text. [NER_OUTPUT]"
#             }
#         ]
#     }

# # Generate multiple samples
# def generate_dataset(n=1000, seed=42):
#     random.seed(seed)
#     dataset = []
#     for i in tqdm(range(n)):
#         text = random.choice(clinical_notes)
#         dataset.append(generate_ner_sample(i, text))
#     return dataset

# # Save to JSONL
# def save_to_jsonl(data, filename="/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/ratener_ner_dataset.json"):
#     with open(filename, "w") as f:
#         for item in data:
#             json.dump(item, f)
#             f.write("\n")

# # Main execution
# if __name__ == "__main__":
#     data = generate_dataset(n=5000)
#     save_to_jsonl(data)
#     print(f"Dataset with {len(dataset)} samples saved to 'ratener_ner_dataset.json'.")
