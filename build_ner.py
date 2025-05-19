import json
import random
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

answer_templates = [
    "Here is the entity breakdown:\n{entities}",
    "Below are the entities I identified:\n{entities}",
    "These are the tokens and their corresponding categories:\n{entities}",
    "Entities detected:\n{entities}",
    "Here are the labeled medical terms:\n{entities}",
    "The note contains the following entities:\n{entities}",
    "Token-level annotations are as follows:\n{entities}",
    "I found these entities in the report:\n{entities}",
    "Entity tagging results:\n{entities}",
    "Summary of entities:\n{entities}",
    "I've categorized the tokens like this:\n{entities}",
    "Entities extracted from the sentence:\n{entities}",
    "Here is the complete list of entities:\n{entities}",
    "Annotated entities:\n{entities}",
    "Below is the entity list with labels:\n{entities}",
    "Findings — entities by type:\n{entities}",
    "This is the entity mapping:\n{entities}",
    "Token annotations:\n{entities}",
    "Here are the recognized entities:\n{entities}",
    "Entity extraction complete:\n{entities}",
    "The following entities were detected:\n{entities}",
    "Entity recognition output:\n{entities}",
    "I've highlighted each entity below:\n{entities}",
    "Detailed entity list:\n{entities}",
    "Here are the classified terms:\n{entities}",
    "Entities present in the text:\n{entities}",
    "Medical entity labels:\n{entities}",
    "Here's a breakdown of the entities found:\n{entities}",
    "Identified entities:\n{entities}",
    "Entity results:\n{entities}",
    "These terms have been labeled:\n{entities}",
    "Entity analysis:\n{entities}",
    "The detected entities are listed below:\n{entities}",
    "I've listed all entities with their labels:\n{entities}",
    "Entity report:\n{entities}",
    "Complete entity annotation:\n{entities}",
    "Here is the token classification:\n{entities}",
    "Entities and their types:\n{entities}",
    "Recognized medical entities:\n{entities}",
    "Token-by-token entity mapping:\n{entities}",
]

LABEL_MAP = {"Anatomy": "Anatomy", "Abnormality": "Abnormality", "Disease": "Disease"}

def spans_to_entities(tokens, spans):
    """
    Convert span indices to a readable bullet list: “token(s) → Label”.
    Span format in dataset: [start, end, label]
    """
    parts = []
    for start, end, label in spans:
        text = " ".join(tokens[start : end + 1])
        parts.append(f"• {text} → {LABEL_MAP.get(label, label)}")
    return "\n".join(parts) if parts else "• No entities detected"

def transform(record, idx):
    tokens = record["sentences"][0]
    spans  = record["ner"][0]

    prompt = random.choice(instruction_templates)
    human  = {
        "from": "human",
        "value": f"{prompt}\n\n{' '.join(tokens)}"
    }

    gpt_tool_call = {
        "from": "gpt",
        "thoughts": "To extract the medical entities, I'll call the RaTE-NER tool.",
        "actions": [{
            "API_name": "RaTE-NER",
            "API_params": {"tokens": tokens}
        }],
        "value": "Calling RaTE-NER to extract entities..."
    }

    tool_output = {
        "from": "gpt",
        "value": json.dumps(spans, ensure_ascii=False)
    }

    pretty_entities = spans_to_entities(tokens, spans)
    friendly_reply  = random.choice(answer_templates).format(entities=pretty_entities)
    gpt_answer = {
        "from": "gpt",
        "value": friendly_reply
    }

    return {
        "id": f"ner_sample_{idx}",
        "conversations": [human, gpt_tool_call, tool_output, gpt_answer]
    }

def build_dataset(input_path, output_path, max_samples=MAX_SAMPLES):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(tqdm(fin, desc="Building NER instruction data")):
            if idx >= max_samples:
                break
            record = json.loads(line)
            conv = transform(record, idx)
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Wrote {min(idx + 1, max_samples)} examples to '{output_path}'")

if __name__ == "__main__":
    build_dataset(INPUT_FILE, OUTPUT_FILE)
