import json
import os
import random
from tqdm import tqdm

# Paths to input paragraphs and ground-truth summaries
INPUT_DIR = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text"
SUMMARY_DIR = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/abstract"
OUTPUT_FILE = "./tool_instruct/llava_sum_dataset.jsonl"
MAX_SAMPLES = 5000

# Instruction templates for summarization tasks
summarization_instructions = [
    "Summarize the key findings of this medical passage.",
    "Generate a concise overview of the medical abstract.",
    "Create a short summary highlighting the main points.",
    "What is the core message in the clinical note below?",
    "Give a brief textual summary of the input content.",
    "Write a summary that captures the essential medical details.",
    "Condense the following clinical passage into a short summary.",
    "What are the main diagnoses or treatments mentioned here?",
    "Briefly summarize the patient case described below.",
    "Provide a one-paragraph summary of the following abstract.",
    "Summarize this clinical report for a busy practitioner.",
    "Generate a synopsis highlighting key symptoms and interventions.",
    "Write a summary suitable for a medical case database.",
    "Create a plain-language summary of this clinical abstract.",
    "Offer a bullet-point summary of the most critical clinical findings.",
    "Extract and summarize the diagnostic conclusions from this report.",
    "Provide a brief summary emphasizing patient outcomes.",
    "Outline the prominent clinical observations in a short summary.",
    "Reduce the text to a concise summary of major medical insights.",
    "Summarize the methodology and results presented in this excerpt.",
    "Highlight the key clinical recommendations in summary form.",
    "Create a digest of the most important research findings.",
    "Provide a succinct overview suitable for medical record notes.",
    "Write a brief abstract based on the given clinical paragraph.",
    "Generate a short summary focusing on diagnostic criteria.",
    "Summarize the main pharmacological interventions described.",
    "Offer a quick summary for clinician reference.",
    "Provide a concise recap of the clinical trial results.",
    "Create a summary that outlines patient demographics and outcomes.",
    "Summarize this text for inclusion in a discharge summary.",
    "Write a brief summary capturing essential laboratory findings.",
    "Generate a concise overview of symptoms and treatment plans.",
    "Summarize the clinical significance of the findings below.",
    "Offer a one-sentence summary of the passage’s main conclusion."
]

# Load texts and summaries, pairing by matching numeric IDs after underscore
def load_pairs(input_dir, summary_dir, max_samples=None):
    pairs = []
    summaries = {}
    # Load ground truth summaries into a dict by numeric ID
    for fname in os.listdir(summary_dir):
        if not fname.endswith('.txt'): continue
        basename = os.path.splitext(fname)[0]
        parts = basename.split('_')
        if len(parts) < 2: continue
        num = parts[-1]
        with open(os.path.join(summary_dir, fname), 'r', encoding='utf-8') as f:
            summaries[num] = f.read().strip()
    # Match paragraphs to summaries by numeric ID
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.txt'): continue
        basename = os.path.splitext(fname)[0]
        parts = basename.split('_')
        if len(parts) < 2: continue
        num = parts[-1]
        if num not in summaries: continue
        # load paragraph and corresponding summary
        with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
            paragraph = f.read().strip()
        summary = summaries[num]
        pairs.append((num, paragraph, summary))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs

# Generate a single RAG-style summarization example
def generate_sample(idx, key, paragraph, summary):
    instruction = random.choice(summarization_instructions)
    human_value = f"{instruction}\n\n{paragraph}"
    plan_msg = "Using LLaVA to summarize the medical passage."
    action = {
        'API_name': 'LLaVA',
        'API_params': {'task': 'summarization', 'text': paragraph}
    }
    return {
        'id': f'summ_{key}_{idx}',
        'conversations': [
            {'from': 'human', 'value': human_value},
            {'from': 'gpt', 'thoughts': plan_msg, 'actions': [action], 'value': plan_msg},
            {'from': 'gpt', 'value': summary}
        ]
    }

# Main script: build and save dataset
if __name__ == '__main__':
    random.seed(42)
    pairs = load_pairs(INPUT_DIR, SUMMARY_DIR, max_samples=MAX_SAMPLES)
    print(f"Found {len(pairs)} paragraph-summary pairs.")
    dataset = []
    for idx, (key, paragraph, summary) in enumerate(tqdm(pairs, desc='Building samples')):
        dataset.append(generate_sample(idx, key, paragraph, summary))
    # Write to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fw:
        for entry in dataset:
            json.dump(entry, fw, ensure_ascii=False)
            fw.write("\n")
    print(f"Saved {len(dataset)} summarization samples to '{OUTPUT_FILE}'.")
