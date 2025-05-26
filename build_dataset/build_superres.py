import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/healthgpt_superres_dataset.jsonl")

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

answer_templates = [
    "Here is your enhanced image:\n{image}",
    "Super-resolution complete. Output image:\n{image}",
    "Image enhancement finished—see result below:\n{image}",
    "The high-resolution scan is ready:\n{image}",
    "Upscaling done. Here is the clarified image:\n{image}",
    "Your requested high-def image:\n{image}",
    "High-quality reconstruction generated:\n{image}",
    "Enhanced diagnostic image below:\n{image}",
    "Detail enhancement complete. Image:\n{image}",
    "The refined scan is provided here:\n{image}",
    "Image sharpening finished. Output:\n{image}",
    "Resolution boost applied. See image:\n{image}",
    "Here's the upgraded scan:\n{image}",
    "HD output created successfully:\n{image}",
    "Enhanced view for clinical review:\n{image}",
    "Your super-resolved image is below:\n{image}",
    "The improved image is now available:\n{image}",
    "Clarity restored—please review:\n{image}",
    "Enhanced {modality} scan attached:\n{image}",
    "Here is the crisp, high-resolution result:\n{image}",
    "Up-scaled image ready:\n{image}",
    "Refined image output:\n{image}",
    "Final high-def image generated:\n{image}",
    "Image quality improved—see below:\n{image}",
    "Enhanced resolution scan:\n{image}",
    "Super-resolution successful. Result:\n{image}",
    "Here's the denoised, sharper image:\n{image}",
    "Completed high-detail reconstruction:\n{image}",
    "The upgraded visual is attached:\n{image}",
    "Sharper diagnostic image:\n{image}",
    "HD reconstruction provided:\n{image}",
    "Improved scan for better evaluation:\n{image}",
    "Pixel enhancement complete. Image:\n{image}",
    "Finalized high-quality output:\n{image}",
    "Here is the high-fidelity scan:\n{image}",
    "Image resolution elevated successfully:\n{image}",
    "Super-resolution pipeline finished:\n{image}",
    "Enhanced spatial detail now available:\n{image}",
    "Upscaled medical image below:\n{image}",
    "Ultra-clear image delivered:\n{image}",
    "High-definition output image:\n{image}",
]

def transform(idx: int) -> dict:
    anatomy   = random.choice(anatomies)
    modality  = random.choice(modalities)
    condition = random.choice(conditions[anatomy])

    user_prompt = random.choice(super_res_templates).format(
        modality=modality, anatomy=anatomy, condition=condition
    )

    tool_call = {
        "from": "gpt",
        "thoughts": (
            f"To enhance the {modality} image of the {anatomy} with {condition}, "
            "I'll call HealthGPT's super_resolution API."
        ),
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
        "value": "Calling HealthGPT super_resolution..."
    }

    final_reply = random.choice(answer_templates).format(
        image="<image>", modality=modality
    )
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"superres_{anatomy}_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            assistant_reply
        ]
    }

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples),
                        desc="Generating super-resolution samples"):
            json.dump(transform(idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} super-resolution samples to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
