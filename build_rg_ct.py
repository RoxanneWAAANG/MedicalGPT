import json
import random
from tqdm import tqdm
import os

# Diverse clinical indications
indications = [
    "chest pain and shortness of breath",
    "persistent cough and fever",
    "abdominal discomfort and nausea",
    "headache following minor trauma",
    "history of hypertension and dizziness",
    "metastatic melanoma, presenting with confusion and somnolence",
    "postoperative checkup after CABG surgery",
    "evaluation for pulmonary embolism",
    "routine follow-up for known lung nodule",
    "neurological symptoms and visual disturbances",
    "severe headache and loss of balance",
    "fever and hypoxia in a COVID-19 patient",
    "seizures in a patient with brain metastasis",
    "baseline scan for cancer staging",
    "abdominal mass evaluation on follow-up"
]

# Imaging modalities
modalities = ["X-ray", "MRI", "CT", "Ultrasound"]

# Radiology report generation prompt templates (15 total)
report_generation_prompts = [
    "<image>\nDescribe the findings on this radiology image for a patient presenting with {}.",
    "<image>\nProvide a radiologist-style report given the indication: {}.",
    "<image>\nWhat abnormalities can be identified in this scan performed due to {}?",
    "<image>\nGenerate a structured radiology report for this case. Indication: {}.",
    "<image>\nEvaluate the image findings for a patient with {} using {} imaging.",
    "<image>\nWrite a radiology interpretation based on this {} image. The patient has {}.",
    "<image>\nGiven the clinical presentation of {}, describe the key imaging findings.",
    "<image>\nCreate a brief findings summary for a {} scan taken due to {}.",
    "<image>\nImaging interpretation requested for {} symptoms. Provide detailed observations.",
    "<image>\nRadiologist review requested for a case involving {}. Document the findings.",
    "<image>\nSummarize the imaging abnormalities based on the following clinical scenario: {}.",
    "<image>\nPlease read this {} image and provide a concise findings report. Indication: {}.",
    "<image>\nWhat are the radiological findings in a patient evaluated for {}?",
    "<image>\nGenerate a radiology 'Findings' section for the following clinical presentation: {}.",
    "<image>\nInterpret this diagnostic image acquired for {} and describe the observations."
]

# Load mock findings from .txt files
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

# Build one instruction sample
def generate_llava_rad_sample(idx, text):
    modality = random.choice(modalities)
    indication = random.choice(indications)
    prompt_template = random.choice(report_generation_prompts)

    # Use modality if template needs two args
    try:
        prompt = prompt_template.format(indication, modality)
    except:
        prompt = prompt_template.format(indication)

    return {
        "id": f"llavarad_report_{idx}",
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "thoughts": f"Calling LLaVA-Rad to generate a radiology report for {modality} image.",
                "actions": [
                    {
                        "API_name": "LLaVA-Rad",
                        "API_params": {
                            "task": "report_generation",
                            "modality": modality,
                            "indication": indication,
                            "image": "<image_placeholder>"
                        }
                    }
                ],
                "value": f"Generating {modality} findings using LLaVA-Rad..."
            },
            {
                "from": "gpt",
                "value": f"{text}"
            }
        ]
    }

# Batch generate dataset
def generate_dataset(clinical_texts, total_samples=5000):
    return [generate_llava_rad_sample(i, clinical_texts[i % len(clinical_texts)]) for i in tqdm(range(total_samples))]

# Save to JSONL
def save_to_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

# Run
if __name__ == "__main__":
    input_dir = "/path/to/your/txt_folder"
    output_path = "llavarad_report_dataset.json"

    clinical_texts = load_clinical_texts_from_txt_dir(input_dir, max_samples=1000)
    dataset = generate_dataset(clinical_texts, total_samples=5000)
    save_to_jsonl(dataset, output_path)

    print(f"Saved LLaVA-Rad instruction dataset with {len(dataset)} samples to '{output_path}'")
