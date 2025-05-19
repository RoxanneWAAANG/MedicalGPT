import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE  = Path("./tool_instruct/healthgpt_reconstruct_dataset.jsonl")

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
    "I was recently diagnosed with {condition}. Can you explain what this is and what I should do next?",
    "Can you tell me more about {condition} and what steps to take after diagnosis?",
    "What is {condition} and what are the recommended actions for it?",
    "Please explain {condition} and guide me on the next steps after this diagnosis.",
    "Could you tell me what {condition} means and what I need to do now?",
    "I'm concerned because I've been diagnosed with {condition}. What does it involve and how should I proceed?",
    "What does a diagnosis of {condition} imply, and what are my next steps?",
    "How serious is {condition}, and what should I consider doing?",
    "I've just learned I have {condition}. Could you walk me through what that means?",
    "Please help me understand the implications of {condition} and the follow-up care required.",
    "I need details on {condition} and advice on managing it going forward.",
    "After being diagnosed with {condition}, what are the most important things I should know?",
    "What should I expect if I have {condition} and what treatments are typical?",
    "Can you break down the key facts about {condition} and what actions to take?",
    "I've been told it's {condition}. What does that diagnosis really mean?",
    "Could you outline the main risks of {condition} and how to address them?",
    "I'm anxious about having {condition}. What information and guidance can you provide?",
    "Please summarize what {condition} is and recommend a course of action.",
    "I want to learn about {condition}: its causes, symptoms, and next steps.",
    "What lifestyle changes or treatments are advised for someone with {condition}?",
    "Can you describe {condition} and suggest a plan for monitoring or treatment?",
    "I'm looking for a clear explanation of {condition} and follow-up protocols.",
    "Could you give me an overview of {condition} and medical advice on it?",
    "What should I do immediately after a {condition} diagnosis?",
    "Can you advise me on managing {condition} and preventing complications?",
    "I have questions about {condition}: what information is crucial and what should I do?",
    "Please provide a detailed yet understandable guide on {condition} and next steps.",
    "How is {condition} treated, and what should my first actions be?",
    "I'm seeking recommendations for someone newly diagnosed with {condition}.",
    "What are the standard guidelines for caring for {condition}?",
    "Can you list key resources and advice for coping with {condition}?"
]

answer_templates = [
    "Here is a synthetic {modality} image illustrating {condition}. Let me explain the key features you're seeing:",
    "I have generated a representative {modality} scan that shows {condition}. Below is an overview of the findings:",
    "The image provided demonstrates {condition} on {modality}. Important points are highlighted:",
    "I created a {modality} reconstruction depicting {condition}. Notice the characteristic changes:",
    "This {modality} example visualizes {condition}. Key observations:",
    "You'll find a reconstructed {modality} image of {condition} attached. Here is what it reveals:",
    "Below is an illustrative {modality} scan for {condition}. Let me walk you through it:",
    "I've produced a simulated {modality} image highlighting {condition}. Key features include:",
    "Here's an educational {modality} reconstruction of {condition}. Observe the following details:",
    "The attached {modality} image models {condition}. Pay particular attention to:",
    "I've synthesized a {modality} view of {condition}. The main visual cues are:",
    "This generated {modality} scan illustrates {condition}. Relevant findings:",
    "Presented is a {modality} reconstruction showcasing {condition}. Diagnostic pearls:",
    "Here is a diagnostic-style {modality} image of {condition}. Significant aspects:",
    "I constructed a {modality} image depicting {condition}. Notice the hallmark signs:",
    "Below find a {modality} representation of {condition}. Interpretation notes:",
    "A simulated {modality} scan of {condition} is provided. Essential points:",
    "You now have a {modality} reconstruction for {condition}. Key elements:",
    "Here is the requested {modality} visualization of {condition}. Clinical highlights:",
    "I've generated an illustrative {modality} image showing {condition}. Observations:",
    "A synthetic {modality} demonstrating {condition} is delivered. Focus on:",
    "The following {modality} image models {condition}. Important indicators:",
    "This is a reconstructed {modality} of {condition}. Pertinent features:",
    "Here's how {condition} appears on {modality}. Main findings:",
    "I've produced an example {modality} image with {condition}. What to look for:",
    "Attached is a {modality} depiction of {condition}. Points of interest:",
    "Observe the reconstructed {modality} scan showing {condition}. Highlights:",
    "You can review a synthetic {modality} displaying {condition}. Key takeaways:",
    "Provided is a {modality} illustration of {condition}. Significant regions:",
    "Here is a modeled {modality} image of {condition}. Note the following:",
    "Below is a {modality} rendering demonstrating {condition}. Examination notes:",
    "I've created a teaching {modality} example for {condition}. Important observations:",
    "This {modality} reconstruction portrays {condition}. Essential findings:",
    "A representative {modality} image of {condition} has been constructed. Details:",
    "See the generated {modality} showing {condition}. Relevant anatomy:",
    "Here's your requested {modality} of {condition}. Diagnostic indicators:",
    "A sample {modality} scan for {condition} is presented. Please note:",
    "The synthesized {modality} image depicts {condition}. Salient points:",
    "I've assembled a {modality} illustration of {condition}. Critical features:",
    "Take a look at this {modality} visualization of {condition}. Key aspects:",
    "The following {modality} reconstruction highlights {condition}. Observed changes:",
]

def transform(idx: int) -> dict:
    """Generate one conversation record."""
    anatomy   = random.choice(anatomies)
    condition = random.choice(conditions[anatomy])
    modality  = random.choice(modalities)

    user_prompt = random.choice(instruction_templates).format(
        modality=modality, anatomy=anatomy, condition=condition
    )

    tool_call = {
        "from": "gpt",
        "thoughts": "To fulfill this request, I'll generate a representative image using HealthGPT.",
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

    tool_output = {
        "from": "gpt",
        "value": "<image>"
    }

    final_answer = random.choice(answer_templates).format(
        modality=modality, condition=condition
    )
    assistant_reply = {
        "from": "gpt",
        "value": final_answer
    }

    return {
        "id": f"reconstruct_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            tool_output,
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
