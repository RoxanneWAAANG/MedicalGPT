import json
import random
from pathlib import Path
from tqdm import tqdm

INPUT_FILE  = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/annotation.json")
OUTPUT_FILE = Path("./tool_instruct/llava_rad_rg_dataset.jsonl")
MODALITIES  = {"X-RAY", "CT", "MRI", "US"}

instruction_templates = [
    "You are a radiology assistant. Given the following {modality} image, generate a comprehensive radiology report detailing all pertinent findings.",
    "As a radiologist, review the {modality} scan provided and write a detailed diagnostic report covering every notable feature.",
    "Examine the {modality} image and produce a thorough radiology report that includes impressions and observations.",
    "Based on the {modality} image below, create an in-depth medical report describing all significant findings.",
    "Generate a detailed radiology report for the given {modality} image, highlighting any abnormalities and normal structures.",
    "Review the provided {modality} scan and write a professional radiology report with observations, conclusions, and recommendations.",
    "Interpret the {modality} image and draft a comprehensive report that covers anatomy, pathology, and clinical impressions.",
    "Analyze the {modality} image and generate a structured radiology report including findings, impressions, and suggestions.",
    "You are tasked with interpreting this {modality} image; write a detailed report summarizing your findings and impressions.",
    "Provide a full radiology report for the following {modality} image, noting all abnormalities and relevant normal anatomy.",
    "As a diagnostic radiologist, examine the {modality} scan and compose a detailed report of your findings.",
    "Evaluate the {modality} image and produce a radiology report that includes critical observations and potential diagnoses.",
    "Using the {modality} image provided, write a detailed narrative report covering all key findings.",
    "Draft a radiology report for the attached {modality} scan, describing normal and abnormal findings in detail.",
    "You are reviewing a {modality} image: generate a comprehensive report including descriptions of any lesions and normal structures.",
    "Create a structured radiology report for the given {modality} image, with sections for findings, impressions, and recommendations.",
    "Interpret and report on the {modality} image below, detailing pathology and normal anatomy.",
    "Write a professional radiology report for the following {modality} scan, summarizing key observations and impressions.",
    "Compose a detailed diagnostic report for the provided {modality} image, focusing on clinically relevant findings.",
    "Analyze the {modality} image and generate an organized radiology report covering anatomy, pathology, and clinical impressions.",
    "Produce an exhaustive radiology report from the {modality} image, outlining all significant observations.",
    "Given the following {modality} scan, write a detailed report highlighting abnormalities and normal variants.",
    "Interpret the {modality} image and draft a comprehensive radiology report including findings and recommendations.",
    "Provide an expert radiology report based on the attached {modality} scan, covering all pertinent details.",
    "You have a {modality} image: produce a detailed diagnostic report that includes descriptive findings and impressions.",
    "Create a full radiology report for the given {modality} image, clearly outlining any pathologic findings.",
    "Examine the attached {modality} scan and write a clinical radiology report covering observations and impressions.",
    "Write a detailed radiology report for this {modality} image, describing both normal and abnormal findings.",
    "Interpret and report on the {modality} image below; provide a comprehensive findings and impressions section.",
    "Compose an analytical radiology report for the given {modality} scan with detailed findings.",
    "Provide a narrative radiology report for the attached {modality} image, noting all important findings.",
    "Review the {modality} scan and generate a structured report including impressions and recommendations.",
    "Based on the {modality} image, write a concise yet thorough radiology report summarizing findings.",
    "Analyze the provided {modality} image and produce a detailed report for clinical decision-making.",
    "Generate a radiology report for the given {modality} scan, emphasizing any critical findings.",
    "As a radiologist, write a detailed report covering the anatomy and pathology visible in this {modality} image.",
    "Compose a clinical radiology report for the attached {modality} image, detailing normal and abnormal structures.",
    "Evaluate the {modality} scan and provide a full radiology report including impressions.",
    "Create a detailed diagnostic report for this {modality} image, with clear descriptions of findings.",
    "Interpret the {modality} scan and draft a comprehensive radiology report including clinical recommendations.",
    "Provide an expert-level radiology report for the given {modality} image, covering all relevant observations.",
    "You are reviewing a {modality} image; generate a report with sections for findings, impression, and recommendation.",
    "Draft a detailed radiology report for the following {modality} scan, noting any significant pathology.",
    "Examine the {modality} image and write a structured radiology report with clear findings.",
    "Produce a detailed clinical report based on this {modality} image, covering both normal and pathological features.",
    "Based on the attached {modality} scan, compose a comprehensive radiology report for clinical use.",
    "Generate a structured radiology report from the given {modality} image, including findings and impressions.",
    "Interpret and report on this {modality} scan, providing a detailed diagnostic summary.",
    "Create a professional radiology report for the following {modality} image, highlighting key findings.",
    "Write a thorough radiology report for the attached {modality} scan, covering diagnosis and impressions."
]

answer_templates = [
    "Here is the full {modality} report I generated, including key findings:\n{report}",
    "Below is the detailed radiology report for this {modality} image:\n{report}",
    "I've completed the interpretation. The {modality} report reads as follows:\n{report}",
    "Diagnostic report for the provided {modality} scan:\n{report}",
    "Here's a comprehensive read of the {modality} image:\n{report}",
    "The structured report for this {modality} study is shown below:\n{report}",
    "Find the full {modality} report with impressions and observations:\n{report}",
    "Completed radiology report ({modality}):\n{report}",
    "Full diagnostic findings for the {modality} image:\n{report}",
    "Here is the requested {modality} report, covering all significant details:\n{report}",
    "Below is a narrative report for the {modality} scan:\n{report}",
    "I have drafted the radiology report for your {modality} image:\n{report}",
    "Radiology report (modality: {modality}):\n{report}",
    "Please review the following {modality} report:\n{report}",
    "Here's the in-depth {modality} interpretation:\n{report}",
    "My full written report on the {modality} study is below:\n{report}",
    "The {modality} image has been analyzed. Report:\n{report}",
    "Comprehensive findings for this {modality} scan:\n{report}",
    "Here is an organized {modality} report with findings and impressions:\n{report}",
    "Final {modality} radiology report:\n{report}",
    "Complete diagnostic summary for the {modality} image:\n{report}",
    "I've structured the {modality} report as follows:\n{report}",
    "Kindly review the {modality} report below:\n{report}",
    "Detailed report for your {modality} study:\n{report}",
    "The {modality} findings are summarized here:\n{report}",
    "Attached is the full {modality} radiology report:\n{report}",
    "My interpretation of the {modality} image is as follows:\n{report}",
    "Below is the expert-level {modality} report:\n{report}",
    "Comprehensive {modality} report prepared:\n{report}",
    "Here is the final radiology report for this {modality} scan:\n{report}",
    "Full narrative for the {modality} image:\n{report}",
    "Report generated for {modality} modality:\n{report}",
    "See the detailed {modality} findings below:\n{report}",
    "I have completed the {modality} analysis. Report:\n{report}",
    "Diagnostic impressions for the {modality} study:\n{report}",
    "Radiology report ({modality}) generated:\n{report}",
    "A structured {modality} report has been created:\n{report}",
    "Here is an exhaustive {modality} report:\n{report}",
    "The report for the {modality} image is ready:\n{report}",
    "Please find the {modality} report below:\n{report}",
]

def transform(record: dict, idx: int) -> dict:
    """Convert one raw record to conversation format."""
    image_path = record["image_path"][0]
    report_text = record["report"]

    modality = random.choice(MODALITIES)
    user_prompt = random.choice(instruction_templates).format(modality=modality)

    tool_call = {
        "from": "gpt",
        "thoughts": "User wants a radiology report; I'll call the LLaVA-Rad tool.",
        "actions": [
            {
                "API_name": "LLaVA-Rad",
                "API_params": {"image_path": image_path}
            }
        ],
        "value": "Calling LLaVA-Rad to generate the radiology report..."
    }

    tool_output = {
        "from": "gpt",
        "value": report_text
    }

    friendly_reply = random.choice(answer_templates).format(
        modality=modality, report=report_text
    )
    assistant_reply = {
        "from": "gpt",
        "value": friendly_reply
    }

    return {
        "id": f"rad_sample_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            tool_output,
            assistant_reply
        ]
    }

def build_dataset(input_path: Path = INPUT_FILE,
                  output_path: Path = OUTPUT_FILE,
                  n_samples: int = 5000,
                  seed: int = 42) -> None:
    random.seed(seed)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_records = data.get("train", []) + data.get("validation", []) + data.get("test", [])
    total = min(len(all_records), n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(tqdm(all_records[:total],
                                       desc=f"Building {total} radiology samples")):
            conv = transform(rec, idx)
            json.dump(conv, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Wrote {total} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
