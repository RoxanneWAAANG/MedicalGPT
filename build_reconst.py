import json
import random
from tqdm import tqdm

# Modalities and anatomy options
modalities = ["MRI", "CT", "X-ray", "Ultrasound"]
anatomies = ["brain", "chest", "abdomen", "spine", "liver", "heart", "knee"]
conditions = {
    "brain": ["glioblastoma", "stroke", "meningioma", "hydrocephalus", "tumor in left frontal lobe"],
    "chest": ["pneumonia", "pulmonary embolism", "lung nodule", "pleural effusion"],
    "abdomen": ["appendicitis", "liver cirrhosis", "pancreatitis", "renal cyst"],
    "spine": ["disc herniation", "spinal stenosis", "scoliosis"],
    "liver": ["hepatocellular carcinoma", "fatty liver"],
    "heart": ["myocardial infarction", "left ventricle hypertrophy"],
    "knee": ["ACL tear", "meniscus injury"]
}

# Diverse user prompts (instruction templates)
instruction_templates = [
    "Can you generate a {modality} image of the {anatomy} showing {condition}?",
    "Please synthesize a {modality} scan of the {anatomy} with {condition}.",
    "Create an artificial {modality} image demonstrating {condition} in the {anatomy}.",
    "I'd like to see a medical image of a {anatomy} that includes {condition}, preferably {modality}.",
    "Simulate a {modality} image where the {anatomy} is affected by {condition}.",
    "Generate an example {modality} image for a patient with {condition} in the {anatomy}.",
    "I need a visual representation of {condition} in the {anatomy}. Use {modality} modality.",
    "Can you construct a synthetic scan showing a {anatomy} with {condition}?",
    "Show me what a {modality} scan looks like for a case of {condition} in the {anatomy}.",
    "Use HealthGPT to generate a medical image of {anatomy} exhibiting {condition}.",
    "As a training radiologist, I’d like to study a {modality} image showing {condition} in the {anatomy}.",
    "Please provide an educational scan showing radiographic signs of {condition} in the {anatomy}.",
    "Construct a teaching image for students to observe {condition} in a {modality} scan of the {anatomy}.",
    "I’m modeling disease progression — generate a synthetic {modality} image of a {anatomy} with {condition}.",
    "Simulate an imaging dataset entry for {condition} in {anatomy} using {modality}.",
    "Build a mock {modality} image for an AI dataset — pathology: {condition}, organ: {anatomy}.",
    "Generate synthetic medical image — modality: {modality}, anatomy: {anatomy}, condition: {condition}.",
    "Construct image: {modality}, {anatomy}, diagnosis: {condition}.",
    "Simulate: {modality} scan of {anatomy} showing {condition}.",
    "Task: Medical image generation; Input: {modality}, {anatomy}, {condition}."
]

# Generate a single data entry with reasoning and tool use
def generate_instruction(id_num, modality, anatomy, condition):
    # Randomly sample an instruction template and fill in placeholders
    prompt = random.choice(instruction_templates).format(
        modality=modality,
        anatomy=anatomy,
        condition=condition
    )

    return {
        "id": f"construct_{modality.lower()}_{anatomy}_{id_num}",
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "thoughts": f"Using HealthGPT to generate a synthetic {modality} image of the {anatomy} with {condition}.",
                "actions": [
                    {
                        "API_name": "HealthGPT",
                        "API_params": {
                            "task": "image_generation",
                            "modality": modality,
                            "anatomy": anatomy,
                            "condition": condition
                        }
                    }
                ],
                "value": f"Generating the requested {modality} image with HealthGPT..."
            },
            {
                "from": "gpt",
                "value": f"Here is the generated {modality} image of the {anatomy} with {condition}. <image>"
            }
        ]
    }

# Generate the full dataset with multiple samples
def generate_dataset(n_samples=1000, seed=42):
    random.seed(seed)
    dataset = []

    for i in tqdm(range(n_samples)):
        anatomy = random.choice(anatomies)
        modality = random.choice(modalities)
        condition = random.choice(conditions[anatomy])
        item = generate_instruction(i, modality, anatomy, condition)
        dataset.append(item)

    return dataset

# Save dataset to JSONL format (one JSON per line)
def save_dataset(dataset, filename="/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/healthgpt_reconst_dataset.json"):
    with open(filename, "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")

# Main script entry
if __name__ == "__main__":
    dataset = generate_dataset(n_samples=5000)
    save_dataset(dataset)
    print(f"Dataset with {len(dataset)} samples saved to 'healthgpt_reconst_dataset.json'.")