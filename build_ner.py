import json
import random
import glob
from tqdm import tqdm

INPUT_FILE   = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/RaTE-NER/train_span.json"
OUTPUT_FILE  = "./tool_instruct/rate_ner_dataset.jsonl"
MAX_SAMPLES  = 5000

instruction_templates = [
    "Perform medical named-entity recognition on the following radiology note. Label each token with its entity type (e.g., Anatomy, Abnormality, Disease):",
    "Extract and classify all medical entities in this radiology report. For each token, specify whether it's Anatomy, Abnormality, or Disease:",
    "Identify anatomical structures, abnormalities, and diseases in the text below. Return a token-level annotation indicating the entity type:",
    "Given this radiology note, tag every term with its medical entity category (Anatomy, Abnormality, Disease):",
    "Analyze the following radiology sentence and label each word with the correct medical entity type (Anatomy/Abnormality/Disease):",
    "Detect and label clinical entities in the radiology description. Annotate each token as Anatomy, Abnormality, or Disease:",
    "Perform token-level entity tagging for Anatomy, Abnormality, and Disease in this radiology note:",
    "Mark up the radiology text by identifying and labeling Anatomy, Abnormality, and Disease entities:",
    "Segment and classify each word in the radiology sentence into Anatomy, Abnormality, or Disease categories:",
    "Provide a token-by-token annotation of medical entities (Anatomy, Abnormality, Disease) for this radiology excerpt:",
    "Highlight and classify medical terms in the radiology note, tagging them as Anatomy, Abnormality, or Disease:",
    "Perform fine-grained medical NER on the following text. Label tokens as Anatomy, Abnormality, or Disease:",
    "Execute a token-level scan of the radiology report to detect Anatomy, Abnormality, and Disease entities:",
    "Run a medical terminology recognition workflow: classify each word as Anatomy, Abnormality, or Disease:",
    "Identify and annotate all occurrences of anatomical terms, abnormalities, and diseases in this line:",
    "Apply entity recognition to find Anatomy, Abnormality, and Disease tokens in the given radiology sentence:",
    "Tag this radiology description's tokens with their respective medical entity types (Anatomy, Abnormality, Disease):",
    "Detect entity spans and classify each token into Anatomy, Abnormality, or Disease class:",
    "Find all medical entities in this radiology text and label them token by token:",
    "Perform a thorough token-level annotation of Anatomy, Abnormality, and Disease in the note:",
    "Using medical NER, assign each token to one of: Anatomy, Abnormality, Disease:",
    "Label terms in the radiology sentence according to their medical entity classes (Anatomy/Abnormality/Disease):",
    "Annotate this radiology line with entity tags: Anatomy, Abnormality, or Disease:",
    "Extract clinical entities from the note and tag each token accordingly:",
    "Perform detailed labeling of the radiology description for Anatomy, Abnormality, and Disease tokens:",
    "Identify and classify every token in the radiology excerpt into one of: Anatomy, Abnormality, Disease:",
    "Label the radiology report tokens with medical entity categories:",
    "Tokenize and annotate the note for Anatomy, Abnormality, and Disease:",
    "Apply a token-level classification of medical entities in this radiology sentence:",
    "Assign medical NER tags (Anatomy, Abnormality, Disease) to all words in the excerpt:",
    "Perform vocabulary tagging: mark each token as Anatomy, Abnormality, or Disease:",
    "Identify entity mentions in the note and label them token-wise:",
    "Extract and tag clinical lexicon in the radiology text:",
    "Apply a NER pipeline to label tokens as Anatomy, Abnormality, or Disease:",
    "Tag medical terms in the sentence with their entity type (Anatomy/Abnormality/Disease):",
    "Label the following radiology description with entity classes for each token:",
    "Annotate medical entity tokens in this radiology sentence:",
    "Perform token classification of clinical entities in the note:",
    "Run medical NER to identify Anatomy, Abnormality, Disease tokens:",
    "Carry out token-level annotation for medical entities:",
    "Detect Anatomy, Abnormality, and Disease labels in the radiology sentence:",
    "Label the text tokens with medical entity categories:",
    "Apply a token-wise NER model for Anatomy, Abnormality, Disease:",
    "Perform token-level entity recognition on the radiology description:",
    "Annotate each token in the radiology text with its medical entity label:",
    "Classify each word in the sentence as Anatomy, Abnormality, or Disease:",
    "Identify and label entity tokens in the radiology note:",
]

def transform(record, idx):
    tokens = record["sentences"][0]
    spans  = record["ner"][0]
    prompt = random.choice(instruction_templates)
    human = {"from": "human", "value": f"{prompt}\n\n{' '.join(tokens)}"}
    gpt_call = {"from": "gpt", "thoughts": "To extract the medical entities, I'll call the RaTE-NER tool.",
                "actions": [{"API_name": "RaTE-NER", "API_params": {"tokens": tokens}}],
                "value": "Calling RaTE-NER to extract entities..."}
    gpt_out = {"from": "gpt", "value": json.dumps(spans, ensure_ascii=False)}
    
    return {
            "id": f"ner_sample_{idx}",
            "conversations": [
                human,
                gpt_call,
                gpt_out
                ]
            }

def build_dataset(input_path, output_path, max_samples=MAX_SAMPLES):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="Building NER instruct data")):
            if idx >= max_samples:
                break
            record = json.loads(line)
            rec = transform(record, idx)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote up to {MAX_SAMPLES} NER examples to {output_path!r}")

if __name__ == "__main__":
    build_dataset(INPUT_FILE, OUTPUT_FILE)
