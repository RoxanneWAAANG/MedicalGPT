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
    "Apply HealthGPT super_resolution to make this {modality} medical image crisper for {condition}.",
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
    "Transform a grainy {modality} scan into a clear image depicting {condition}.",
    "Optimize the image quality of a {modality} scan showing {condition} in the {anatomy} via super-resolution.",
    "Elevate image resolution of a {modality} capture focusing on {anatomy} with {condition}.",
    "Produce a clearer {modality} image emphasizing {condition} in the {anatomy} through upscaling.",
    "Leverage super-resolution to clarify the {anatomy} region exhibiting {condition} in a {modality} scan.",
    "Use AI-driven enhancement to improve resolution of the {anatomy} {modality} image with {condition}.",
    "Generate a denoised, high-res {modality} image highlighting {condition} in the {anatomy}.",
    "Enhance spatial resolution of a {modality} medical image depicting {condition} in the {anatomy}.",
    "Upscale the {modality} scan for better visualization of {condition} in the {anatomy}.",
    "Apply advanced super-resolution to a {modality} image capturing {condition} in the {anatomy}.",
    "Develop a high-definition {modality} image of {anatomy} with {condition} using AI upscaling.",
    "Create an enhanced {modality} scan for clear viewing of {condition} in the {anatomy}.",
    "Restore fine details in a {modality} image of {anatomy} with {condition} through super-resolution.",
    "Super-resolve the {modality} volume to highlight {condition} in the {anatomy}.",
    "Convert a low-detail {modality} image into a high-fidelity representation of {condition}.",
    "Use deep-learning super-resolution to refine the {anatomy} depiction in the {modality} with {condition}.",
    "Produce a high-quality upscaled version of a {modality} scan featuring {condition}.",
    "Employ super-resolution methods to clarify {condition} in the {anatomy} for a {modality} image.",
    "Generate an enhanced diagnostic image by upscaling the {modality} scan showing {condition}.",
    "Apply multi-scale super-resolution to a {modality} image of the {anatomy} with {condition}.",
    "Increase image fidelity of a {modality} scan for better delineation of {condition} in the {anatomy}.",
    "Upscale and refine a {modality} scan to reveal subtle {condition} details in the {anatomy}.",
    "Create a sharper AI-enhanced {modality} image of the {anatomy} capturing {condition}.",
    "Perform high-resolution reconstruction on a {modality} scan with {condition} in the {anatomy}.",
    "Render a detailed upscaled image showing {condition} in the {anatomy} via super-resolution.",
    "Utilize advanced reconstruction to enhance {modality} image resolution and highlight {condition}.",
    "Use AI super-resolution to create a high-detail {modality} scan for diagnosing {condition} in {anatomy}."
]

values_templates = [
    "Here is the enhanced high-resolution image of the {anatomy} showing {condition}:",
    "Super-resolution complete. See below:",
    "Image enhancement successful. Please find the upscaled image:",
    "Your requested high-resolution image is ready:",
    "Done! Here's the improved {modality} scan:",
    "Image sharpening complete. Here is the output:",
    "The enhanced image is now available:",
    "High-definition image generated successfully:",
    "Here is the refined {modality} scan in high resolution:",
    "Upscaling finished. Here's the result:",
    "Super-resolution applied. View the enhanced image:",
    "Enhanced image of the {anatomy} ({condition}) is below:",
    "High-res output ready for the {modality} scan:",
    "Here's the sharpened image you requested:",
    "Image upscaling done. Check this out:",
    "Your high-definition {modality} image is here:",
    "Detail enhancement complete. See the image:",
    "Here's the super-resolved scan:",
    "Output image with enhanced clarity:",
    "Enjoy the upgraded resolution image below:",
    "The refined high-resolution image is now provided:",
    "Super-resolution finished. Displaying result:",
    "Please find the high-res image here:",
    "Resolution enhancement is done. Here's your image:",
    "The {anatomy} image is now clearer. See below:",
    "Enhanced scan for better visualization:",
    "Your image enhancement is complete:",
    "Here is the crisp, high-definition output:",
    "Image restoration and upscaling done:",
    "The detailed image is now available:",
    "Here is the super-res image of the {anatomy}:",
    "High-definition reconstruction complete:",
    "Your {modality} image has been upscaled successfully:",
    "Here's the clarified scan:",
    "Resolution boost applied. Here's the image:",
    "Enhanced view of the {anatomy} is provided:",
    "High-quality image generated below:",
    "Here is the fine-tuned high-res scan:",
    "Super-speed upscaling done. See result:",
    "Output image with improved resolution:",
    "Here's the highly detailed image:",
    "Enhanced {modality} image is ready for review:",
    "High-res image of the {anatomy} is now available:",
    "Your super-resolved image is here:",
    "Image enhancement pipeline finished:",
    "The processed high-definition image:",
    "Clarity enhancement complete. View below:",
    "Here is the artifact-free high-res image:",
    "Your detailed scan is ready:",
    "Image enhancement operation complete:",
    "Here's the superior resolution image:",
    "Enhanced medical scan is now provided:",
    "High-resolution output available:",
    "Here is the visually refined image:",
    "Resolution uplift complete. See image:",
    "Enhanced scan of the {modality} image:",
    "Here is the ultra-clear image you asked for:",
    "Your super-resolution result is below:",
    "High-definition output for clinical review:"
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
