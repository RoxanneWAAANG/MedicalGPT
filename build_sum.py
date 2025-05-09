import json
import os
import random
from tqdm import tqdm

INPUT_DIR = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/line_text"
SUMMARY_DIR = "/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/sumpubmed/abstract"
OUTPUT_FILE = "./tool_instruct/llava_sum_dataset.jsonl"
MAX_SAMPLES = 5000

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
    "Offer a one-sentence summary of the passage’s main conclusion.",
    "Craft a high-level overview of this patient’s journey and outcomes.",
    "Produce a brief summary emphasizing the study’s purpose and conclusion.",
    "Capture the main imaging findings in a few sentences.",
    "Summarize the patient’s history, exam, and plan in a concise paragraph.",
    "Generate a layperson-friendly summary of this medical text.",
    "Provide a clinical take-home points summary.",
    "Distill this medical abstract into three key sentences.",
    "Summarize the treatment plan and follow-up instructions.",
    "Create a summary that highlights safety and efficacy results.",
    "Produce an executive summary of this clinical trial report.",
    "Write a short summary of the patient’s vital signs and labs.",
    "Generate a focused summary on the diagnostic imaging findings.",
    "Summarize the procedural steps and outcomes outlined below.",
    "Provide an abbreviated summary of this medical case.",
    "Create a concise summary for rapid clinical decision-making.",
    "Summarize the adverse events and management strategies.",
    "Write a brief summary of the study design and endpoints.",
    "Produce a summary emphasizing changes from baseline values.",
    "Summarize the follow-up recommendations in bullet points.",
    "Capture the essential diagnostic criteria in a short summary.",
    "Generate a summary that outlines risk factors and prevention.",
    "Summarize the pathophysiology and key clinical markers.",
    "Provide a summary focusing on patient symptoms and response.",
    "Write a succinct summary of the research hypothesis and results."
]

def load_pairs(input_dir, summary_dir, max_samples=None):
    pairs = []
    summaries = {}

    for fname in os.listdir(summary_dir):
        if not fname.endswith('.txt'): continue
        basename = os.path.splitext(fname)[0]
        parts = basename.split('_')
        if len(parts) < 2: continue
        num = parts[-1]
        with open(os.path.join(summary_dir, fname), 'r', encoding='utf-8') as f:
            summaries[num] = f.read().strip()

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.txt'): continue
        basename = os.path.splitext(fname)[0]
        parts = basename.split('_')
        if len(parts) < 2: continue
        num = parts[-1]
        if num not in summaries: continue

        with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
            paragraph = f.read().strip()
        summary = summaries[num]
        pairs.append((num, paragraph, summary))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs

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

if __name__ == '__main__':
    random.seed(42)
    pairs = load_pairs(INPUT_DIR, SUMMARY_DIR, max_samples=MAX_SAMPLES)
    print(f"Found {len(pairs)} paragraph-summary pairs.")
    dataset = []
    for idx, (key, paragraph, summary) in enumerate(tqdm(pairs, desc='Building samples')):
        dataset.append(generate_sample(idx, key, paragraph, summary))
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fw:
        for entry in dataset:
            json.dump(entry, fw, ensure_ascii=False)
            fw.write("\n")
    print(f"Saved {len(dataset)} summarization samples to '{OUTPUT_FILE}'.")
