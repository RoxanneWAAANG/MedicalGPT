import json
import random
from tqdm import tqdm

modalities = ["MRI", "CT", "X-ray", "Ultrasound"]
anatomies = ["brain", "chest", "abdomen", "spine", "liver", "heart", "knee"]
conditions = {
    "brain": ["tumor", "glioma", "stroke", "hydrocephalus", "meningioma"],
    "chest": ["pneumonia", "pulmonary embolism", "lung nodule", "pleural effusion"],
    "abdomen": ["appendicitis", "liver cirrhosis", "pancreatitis", "renal cyst"],
    "spine": ["disc herniation", "spinal stenosis", "scoliosis"],
    "liver": ["cirrhosis", "hepatitis", "hepatocellular carcinoma"],
    "heart": ["myocardial infarction", "left ventricle hypertrophy", "valve regurgitation"],
    "knee": ["ACL tear", "meniscus injury", "ligament tear"]
}

# User-facing prompt templates for super-resolution
super_res_templates = [
    "Can you enhance the resolution of a {modality} scan of the {anatomy} showing {condition} conditions?",
    "I have a low-resolution {modality} image of the {anatomy} with {condition}. Please apply super-resolution.",
    "Use super-resolution on a {modality} image of the {anatomy} with {condition}.",
    "Perform image enhancement on a {modality} scan showing {condition} in the {anatomy}.",
    "Apply super-resolution to a blurry {modality} image depicting {condition} in the {anatomy}.",
    "Simulate enhancing image quality for a {modality} scan of the {anatomy} with {condition}.",
    "Generate a high-resolution version of a {modality} scan of the {anatomy} affected by {condition}.",
    "Improve the clarity of a medical image showing {condition} in the {anatomy} using {modality}.",
    "Use HealthGPT to sharpen a low-res {modality} scan involving {condition} in the {anatomy}.",
    "Upgrade the quality of a {modality} scan of the {anatomy} with {condition} using super-resolution.",
    "Enhance the detail of a {modality} scan of the {anatomy} displaying {condition}.",
    "Render a sharp {modality} image highlighting {condition} in the {anatomy}.",
    "Boost the resolution of an existing {modality} medical scan of the {anatomy} with {condition}.",
    "Can you reconstruct a clearer {modality} image showing the {anatomy} and {condition}?",
    "Please upscale the resolution of a {modality} scan featuring {condition} in the {anatomy}.",
    "How would a high-res {modality} image of the {anatomy} with {condition} look? Generate it.",
    "Sharpen the focus of a {modality} scan where the {anatomy} shows signs of {condition}.",
    "Apply HealthGPT’s super_resolution to make this {modality} medical image crisper for {condition}.",
    "Enhance details of a {modality} image demonstrating {condition} in the {anatomy}.",
    "I need a high-definition {modality} scan of the {anatomy} exhibiting {condition}.",
    "Please make a {modality} image of the {anatomy} with {condition} clearer and more detailed.",
    "Create a high-res synthetic {modality} dataset image of the {anatomy} with {condition}.",
    "Increase the pixel resolution of a {modality} medical scan of {anatomy} with {condition}.",
    "Use HealthGPT to restore clarity in a {modality} medical image featuring {condition}.",
    "Generate a refined {modality} image of the {anatomy} showing {condition} through super-resolution.",
    "Produce an upscaled {modality} medical scan for visualizing {condition} in the {anatomy}.",
    "Fine-tune resolution on a {modality} image that includes evidence of {condition} in {anatomy}.",
    "Make this {modality} radiograph of {anatomy} with {condition} sharper via super-resolution.",
    "Transform a grainy {modality} scan into a clear image depicting {condition}."
]

values_templates = [
    "Here is the enhanced high-resolution image of the {anatomy} showing {condition}:",
    "Super-resolution complete. See below:",
    "Image enhancement successful. Please find the upscaled image:",
    "Your requested high-resolution image is ready:",
    "Done! Here’s the improved {modality} scan:",
    "Image sharpening complete. Here is the output:",
    "The enhanced image is now available:",
    "High-definition image generated successfully:",
    "Here is the refined {modality} scan in high resolution:",
    "Upscaling finished. Here’s the result:"
]

def generate_superres_instruction(id_num, modality, anatomy, condition):
    prompt = random.choice(super_res_templates).format(
        modality=modality,
        anatomy=anatomy,
        condition=condition
    )
    plan_value = (
        f"To enhance the resolution of the {modality} scan for the {anatomy} with {condition}, "
        "I will call HealthGPT's super_resolution API."
    )
    final_value = random.choice(values_templates).format(
        modality=modality,
        anatomy=anatomy,
        condition=condition
    ) + " <image>"

    return {
        "id": f"superres_{anatomy}_{id_num}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "thoughts": plan_value,
             "actions": [{"API_name": "HealthGPT", "API_params": {
                 "task": "super_resolution",
                 "modality": modality,
                 "anatomy": anatomy,
                 "condition": condition
             }}], "value": plan_value},
            {"from": "gpt", "value": final_value}
        ]
    }

# Generate full dataset and save to JSONL
def generate_dataset(n_samples=1000, seed=42):
    random.seed(seed)
    dataset = []
    for i in tqdm(range(n_samples), desc="Generating super-resolution samples"):
        anatomy = random.choice(anatomies)
        modality = random.choice(modalities)
        condition = random.choice(conditions[anatomy])
        dataset.append(generate_superres_instruction(i, modality, anatomy, condition))
    return dataset

def save_dataset(dataset, filename="./tool_instruct/healthgpt_superres_dataset.jsonl"):
    with open(filename, "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")

if __name__ == "__main__":
    total = 5000
    dataset = generate_dataset(n_samples=total)
    save_dataset(dataset)
    print(f"Saved {len(dataset)} super-resolution samples to 'healthgpt_superres_dataset.jsonl'.")
