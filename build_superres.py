import json
import random
from tqdm import tqdm

# Options for generation
modalities = ["MRI", "CT", "X-ray", "Ultrasound"]
anatomies = ["brain", "chest", "abdomen", "spine", "liver", "heart", "knee"]
conditions = {
    "brain": ["tumor", "glioma", "stroke"],
    "chest": ["lung nodule", "pleural effusion"],
    "abdomen": ["liver cyst", "appendicitis"],
    "spine": ["disc degeneration", "scoliosis"],
    "liver": ["cirrhosis", "hepatitis"],
    "heart": ["valve regurgitation", "cardiomegaly"],
    "knee": ["ligament tear", "meniscus damage"]
}

# Instruction templates (diverse prompts for super-resolution)
super_res_templates = [
    "Can you enhance the resolution of a {modality} scan of the {anatomy} showing {condition}?",
    "I have a low-resolution {modality} image of the {anatomy} with {condition}. Please enhance it.",
    "Use super-resolution on a {modality} image of the {anatomy} with {condition}.",
    "Perform image enhancement on a {modality} scan showing {condition} in the {anatomy}.",
    "Apply super-resolution to a blurry {modality} image that shows a {condition} in the {anatomy}.",
    "Simulate enhancing image quality for a {modality} scan of {anatomy} with {condition}.",
    "Generate a high-resolution version of a {modality} scan of {anatomy} affected by {condition}.",
    "Improve the clarity of a medical image showing {condition} in the {anatomy} using {modality}.",
    "Use HealthGPT to sharpen a low-res {modality} scan involving {condition} in the {anatomy}.",
    "Upgrade the quality of a {modality} scan of the {anatomy} with {condition} using super-resolution."
]

# Function to build a single training example
def generate_superres_sample(idx, modality, anatomy, condition):
    user_prompt = random.choice(super_res_templates).format(
        modality=modality, anatomy=anatomy, condition=condition
    )

    return {
        "id": f"superres_{modality}_{anatomy}_{idx}",
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": f"Using HealthGPT to apply super-resolution on a {modality} image of the {anatomy} with {condition}.",
                "actions": [
                    {
                        "API_name": "HealthGPT",
                        "API_params": {
                            "task": "super_resolution",
                            "modality": modality,
                            "anatomy": anatomy,
                            "condition": condition
                        }
                    }
                ],
                "value": f"Enhancing the image with super-resolution using HealthGPT..."
            },
            {
                "from": "gpt",
                "value": f"Here is the enhanced high-resolution {modality} scan of the {anatomy} showing {condition}. [IMAGE_REF]"
            }
        ]
    }

# Generate full dataset
def generate_dataset(n=1000, seed=42):
    random.seed(seed)
    dataset = []
    for i in tqdm(range(n)):
        anatomy = random.choice(anatomies)
        modality = random.choice(modalities)
        condition = random.choice(conditions[anatomy])
        dataset.append(generate_superres_sample(i, modality, anatomy, condition))
    return dataset

# Save to file
def save_to_jsonl(data, filename="/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/tool_instruct/healthgpt_superres_dataset.json"):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

# Run this
if __name__ == "__main__":
    dataset = generate_dataset(n=5000)
    save_to_jsonl(dataset)
    print(f"Dataset with {len(dataset)} samples saved to 'healthgpt_superres_dataset.json'.")
