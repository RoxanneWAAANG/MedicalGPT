import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/healthgpt_reconstruct_dataset.jsonl")

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

instruction_templates = [
    "Help me reconstruct this image.",
    "I need a reconstructed scan. Can you provide it?",
    "Reconstruct a diagnostic image, please.",
    "Can you reconstruct an image from scan data?",
    "Please provide a representative medical image.",
    "Simulate a patient scan image.",
    "Help me visualize this medical case.",
    "Reconstruct the imaging result for me.",
    "I'd like to see a reconstructed scan.",
    "Give me a diagnostic-style image.",
    "Create a medical scan for visualization.",
    "I need a sample medical scan. Can you generate one?",
    "Can you show me an example medical scan?",
    "Recreate the image from the case.",
    "Generate an image that models the clinical scenario.",
    "Can you simulate what this scan might show?",
    "I'd appreciate a visual reconstruction.",
    "Please produce a simulated scan image.",
    "Could you generate a diagnostic image for review?",
    "Provide a scan reconstruction for this case.",
    "Could you recreate the image for educational use?",
    "Show me a synthetic diagnostic image.",
    "I need a reconstructed visualization of this scenario.",
    "Reconstruct a typical scan from this situation.",
    "Please synthesize a diagnostic-style scan.",
    "Could you simulate a clinical imaging result?",
    "I want to see a reconstructed scan example.",
    "Please simulate the imaging output.",
    "Can you give a representative scan image?",
    "Produce a scan image based on this case.",
    "Can you build a diagnostic image here?",
    "I'd like a simulation of the scan.",
    "Create an image based on this clinical case.",
    "Could you visualize the case in an image?",
    "Generate a reconstruction image, please.",
    "Simulate the scan that matches this situation.",
    "I want a visual medical scan based on this.",
    "Generate a scan-style image for this input.",
    "Can you generate a reconstructed scan image?",
    "Simulate how the image might appear clinically.",
    "Please generate a synthetic version of the scan.",
    "Could you help me visualize the imaging outcome?",
    "Show me a reconstructed version of this case.",
    "I'd like to get a simulated scan of this situation.",
    "Visualize the scan data as an image, please."
]

reconstruction_response_templates = [
    "This is the reconstructed image you requested.",
    "Here is the simulated medical image.",
    "The diagnostic image has been generated.",
    "This image illustrates the reconstruction results.",
    "I've completed the image reconstruction.",
    "Here is your synthetic medical scan.",
    "The requested scan has been generated.",
    "Below is the visual reconstruction you asked for.",
    "This image represents the simulated scan.",
    "The reconstructed medical image is now ready.",
    "I've generated the visualization as requested.",
    "Image reconstruction completed as per your request.",
    "Here is the representative medical image.",
    "This scan simulates the requested reconstruction.",
    "Please find the reconstructed image below.",
    "Here's the medical image you asked for.",
    "Synthetic image reconstruction is complete.",
    "The medical scan has been created successfully.",
    "Your reconstructed diagnostic image is ready.",
    "Image generated based on provided medical context."
]

def transform(idx: int) -> dict:
    """Generate one conversation record."""
    anatomy = random.choice(anatomies)
    condition = random.choice(conditions[anatomy])
    modality = random.choice(modalities)

    user_prompt = random.choice(instruction_templates)

    tool_call = {
        "from": "gpt",
        "thoughts": "To fulfill this request, I'll use image reconstruction tool called HealthGPT.",
        "actions": [
            {
                "API_name": "HealthGPT",
                "API_params": {
                    "task": "reconstruct_image",
                    "modality": modality,
                    "anatomy": anatomy,
                    "condition": condition
                }
            }
        ],
        "value": "Calling HealthGPT to reconstruct the image..."
    }

    final_response = random.choice(reconstruction_response_templates)
    assistant_reply = {
        "from": "gpt",
        "value": f"<image>\n{final_response}"
    }

    return {
        "id": f"reconstruct_{idx}",
        "meta": {
            "modality": modality,
            "anatomy": anatomy,
            "condition": condition
        },
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            assistant_reply
        ]
    }

def build_dataset(n_samples: int = 5000, seed: int = 42, output_path: Path = OUTPUT_FILE) -> None:
    """Generate the dataset and save as JSONL."""
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples), desc="Generating reconstruction samples"):
            record = transform(idx)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Dataset with {n_samples} samples saved to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
