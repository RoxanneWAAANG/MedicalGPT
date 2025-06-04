import json, random, uuid
from pathlib import Path
from tqdm import tqdm

INPUT_FILE  = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/RaTE-NER/train_span.json")
OUT_DIR     = Path("./new_tool_instruct")
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUT_DIR / "rate_ner_dataset.jsonl"
MAX_SAMPLES = 10000

PROMPTS = [
    "Perform medical named-entity recognition on the following radiology note. Label each token with its entity type (e.g., Anatomy, Abnormality, Disease):",
    "Extract and classify all medical entities in this radiology report. For each token, specify whether it's Anatomy, Abnormality, or Disease:",
    "Identify anatomical structures, abnormalities, and diseases in the text below. Return a token-level annotation indicating the entity type:",
    "What medical entities are in this phrase?",
    "Tag this short clinical description with entity types.",
    "Can you identify any Anatomy, Abnormality, or Disease terms here?",
    "Which words in this sentence are medical entities?",
    "Is there any disease or anatomy mentioned in this phrase?",
    "Mark up each word with its entity class.",
    "Highlight the medical terms in this sentence.",
    "Give me a token-level tag for this snippet.",
    "Annotate the following text with medical entity tags.",
    "Show which tokens are anatomy, abnormalities, or diseases in this sample.",
    "Help me understand entity types in this short report.",
    "Label the medical categories of words in this example sentence.",
    "Can you annotate this phrase for medical NLP training?",
    "Review this sentence and tag all medical terms by type.",
    "Identify token-level entities in this clinical line.",
    "Break down this sentence into entity-labeled tokens.",
    "Do a quick NER pass on this short clinical description.",
    "Mark entities in this expression as Anatomy/Abnormality/Disease.",
    "Tag this input using medical named-entity recognition.",
    "Can you tell what's anatomy or disease in this line?",
    "Mark each word in this short note with its medical meaning.",
    "Scan this sentence for anatomy, diseases, or abnormalities.",
    "Go through this line and label any clinical entities.",
    "Assign entity types to the words in this phrase.",
    "Quickly tag medical terms in this snippet.",
    "Label all words that are clinical entities: Anatomy, Abnormality, Disease.",
    "For each word, give a label: Anatomy, Abnormality, or Disease.",
    "Identify and tag any relevant medical terms.",
    "Please classify tokens by medical entity type.",
    "Detect entity types in the following sentence.",
    "Tag this entry for Anatomy, Abnormality, or Disease.",
    "Classify words in this input as clinical entities.",
    "Run a quick entity classification on this phrase.",
    "Perform a light NER tagging on this sentence.",
    "Classify each term as anatomy, abnormality, or disease.",
]

ANS_TEMPLATES = [
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

LABEL_MAP = {"Anatomy": "Anatomy",
             "Abnormality": "Abnormality",
             "Disease": "Disease"}

def spans2pretty(tokens, spans):
    """make • token(s) → Label bullet lines"""
    lines = []
    for s, e, lbl in spans:
        text = " ".join(tokens[s:e+1])
        lines.append(f"• {text} → {LABEL_MAP.get(lbl,lbl)}")
    return "\n".join(lines) if lines else "• No entities detected"

def json_call(tokens):
    """JSON string the *model* must output"""
    return json.dumps({
        "name": "RaTE-NER",
        "arguments": {"tokens": tokens}
    }, ensure_ascii=False)

def build_conv(rec):
    tokens = rec["sentences"][0]
    spans  = rec["ner"][0]

    q = f"{random.choice(PROMPTS)}\n\n{' '.join(tokens)}"
    tool_output = {
        "entities": [
            {"text": " ".join(tokens[s:e+1]), "label": LABEL_MAP.get(lbl,lbl)}
            for s, e, lbl in spans
        ]
    }

    pretty = spans2pretty(tokens, spans)
    friendly = random.choice(ANS_TEMPLATES).format(entities=pretty)

    return {
        "id": f"ner_{uuid.uuid4().hex[:6]}",
        "conversations": [
            # ① user
            {"from": "human", "value": q},

            # ② assistant – the *function-call* JSON
            {"from": "gpt",
             "thoughts": "User is asking a question about NER, call RaTE-NER tool to answer this question.",
             "actions": [{
                 "API_name": "RaTE-NER",
                 "API_params": {"tokens": tokens}
             }],
             "value": json_call(tokens)},

            # ③ tool – simulated result (Router will send real one in prod)
            {"from": "tool",
             "name": "RaTE-NER",
             "response": json.dumps(tool_output, ensure_ascii=False)},

            # ④ assistant – friendly natural-language summary
            {"from": "gpt",
             "value": friendly}
        ]
    }

# ---------- build ----------
def main():
    with INPUT_FILE.open() as fin, OUTPUT_FILE.open("w") as fout:
        for i, line in enumerate(tqdm(fin, total=MAX_SAMPLES,
                                      desc="build RaTE-NER dataset")):
            if i >= MAX_SAMPLES:
                break
            sample = build_conv(json.loads(line))
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {min(i+1,MAX_SAMPLES)} samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
