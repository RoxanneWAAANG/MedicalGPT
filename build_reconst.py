import json
import random
from tqdm import tqdm

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

values_templates = [
    "Happy to assist. To address your query, I'll use a HealthGPT model to provide suggestions from credible sources. Here's my recommendation:",
    "Pleased to help. For your question, I will employ a HealthGPT model to derive suggestions from verified sources. My suggestion follows:",
    "I'm here to assist. In response to your inquiry, I'll leverage a HealthGPT model to generate recommendations from authorized sources. Here is what I suggest:",
    "Certainly. I'll consult authoritative medical sources with a RAG model and share my findings:",
    "Understood. Retrieving up-to-date clinical guidance now-here's what I recommend:",
    "On it. I'm pulling evidence-based information—you'll find my suggestions below:",
    "Let me fetch the latest medical insights—here's a concise recommendation:",
    "I'll gather expert-validated data and present you with actionable advice:",
    "I'm retrieving peer-reviewed medical details—this is my guidance:",
    "Drawing on trusted medical literature, here's the plan I recommend:",
    "Consulting clinical guidelines now—please consider the following steps:",
    "I'm examining best-practice protocols and will outline my advice:",
    "I'll use a HealthGPT model to compile top-tier recommendations for you:"
]

def generate_rag_instruction(id_num, anatomy, condition):
    prompt = random.choice(instruction_templates).format(condition=condition)
    final_response = random.choice(values_templates) + f" Here is what I found about {condition} and how to proceed." + " <image>"

    return {
        "id": f"rag_{anatomy}_{id_num}",
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "thoughts": f"To answer this, I should reconstruct the image about {condition} using HealthGPT.",
                "actions": [
                    {
                        "API_name": "HealthGPT",
                        "API_params": {
                            "query": condition
                        }
                    }
                ],
                "value": f"I will use HealthGPT to reconstruct the image about {condition}."
            },
            {
                "from": "gpt",
                "value": final_response
            }
        ]
    }

def generate_dataset(n_samples=1000, seed=42):
    random.seed(seed)
    dataset = []

    for i in tqdm(range(n_samples)):
        anatomy = random.choice(anatomies)
        condition = random.choice(conditions[anatomy])
        item = generate_rag_instruction(i, anatomy, condition)
        dataset.append(item)

    return dataset

def save_dataset(dataset, filename="./tool_instruct/healthgpt_reconst_dataset.json"):
    with open(filename, "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")

dataset = generate_dataset(n_samples=5000)
save_dataset(dataset)
print(f"Dataset with {len(dataset)} samples saved to 'healthgpt_reconst_dataset.json'.")
