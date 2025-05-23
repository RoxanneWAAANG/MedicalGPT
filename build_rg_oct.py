import json
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR    = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/deepeyenet/deepeyenet")
INPUT_FILE  = BASE_DIR / "DeepEyeNet_train.json"
OUTPUT_FILE = Path("./tool_instruct/svlms_fundus_dataset.jsonl")
MODALITY     = "OCT"

instruction_templates = [
    "You are an ophthalmic imaging assistant. Given the following {modality} scan, generate a comprehensive report detailing all retinal and choroidal findings.",
    "As a retinal specialist, review the {modality} image provided and write a detailed diagnostic report covering every notable feature.",
    "Examine the {modality} scan and produce a thorough report that includes impressions, observations, and clinical recommendations.",
    "Based on the {modality} image below, create an in-depth medical report describing all significant macular findings.",
    "Generate a detailed ophthalmology report for the given {modality} image, highlighting any abnormalities and relevant normal structures.",
    "Review the provided {modality} scan and write a professional report with observations, conclusions, and follow-up suggestions.",
    "Interpret the {modality} image and draft a comprehensive report that covers retinal layers, pathology, and clinical impressions.",
    "Analyze the {modality} scan and generate a structured report including findings, impressions, and management advice.",
    "You are tasked with interpreting this {modality} image; write a detailed report summarizing your findings and impressions.",
    "Provide a full ophthalmic report for the following {modality} scan, noting all abnormalities and relevant normal anatomy.",
    "As a diagnostic ophthalmologist, examine the {modality} image and compose a detailed report of your findings.",
    "Evaluate the {modality} scan and produce a report that includes critical observations and potential diagnoses.",
    "Using the {modality} image provided, write a detailed narrative report covering all key findings.",
    "Draft an ophthalmology report for the attached {modality} scan, describing normal and abnormal retinal features in detail.",
    "You are reviewing a {modality} image: generate a comprehensive report including descriptions of any lesions and normal structures.",
    "Create a structured report for the given {modality} image, with sections for findings, impressions, and recommendations.",
    "Interpret and report on the {modality} scan below, detailing pathology and normal anatomy.",
    "Write a professional ophthalmic report for the following {modality} image, summarizing key observations and impressions.",
    "Compose a detailed diagnostic report for the provided {modality} scan, focusing on clinically relevant findings.",
    "Analyze the {modality} image and generate an organized report covering anatomy, pathology, and clinical impressions.",
    "Assess the {modality} scan and deliver a meticulous report highlighting vitreomacular interface disorders and any subretinal fluid.",
    "Review this {modality} image and craft a comprehensive summary emphasizing retinal pigment epithelium changes and drusen presence.",
    "As an OCT analyst, interpret the {modality} scan and compile a report outlining retinal layer integrity and any cystoid spaces.",
    "Generate a clinically oriented report for the given {modality} image, specifying outer retinal atrophy and choroidal thickness.",
    "Examine the {modality} scan below; provide findings with emphasis on epiretinal membranes, macular holes, or tractional changes.",
    "Create a detailed technical report for this {modality} image, noting signal quality, segmentation artifacts, and diagnostic limitations.",
    "Interpret the provided {modality} scan to identify diabetic macular edema features and recommend next diagnostic steps.",
    "Produce a structured ophthalmology report for the {modality} image, including B-scan observations and en-face correlations.",
    "Analyze the {modality} scan for signs of neovascular activity; summarize findings and suggest follow-up imaging intervals.",
    "Write an evidence-based report on the {modality} image, highlighting optic nerve head morphology and peripapillary RNFL status.",
    "Review the attached {modality} scan; detail any signs of central serous chorioretinopathy and advise management.",
    "As a retinal fellow, evaluate the {modality} image and produce a report covering photoreceptor layer continuity and foveal contour.",
    "Interpret this {modality} scan with focus on age-related macular degeneration biomarkers; generate a thorough diagnostic note.",
    "Generate a point-by-point report for the provided {modality} scan, listing qualitative and quantitative retinal thickness metrics.",
    "Compose a comprehensive ophthalmic report on the {modality} image, addressing inner retinal hyper-reflective foci and ischemia.",
    "You are assessing an {modality} scan: write a detailed report covering choroidal neovascularization and sub-RPE deposits.",
    "Draft a high-level consult report for this {modality} image, including differential diagnoses and recommended ancillary tests.",
    "Analyze the {modality} scan and provide a concise yet complete report suitable for electronic medical record entry.",
    "Review the {modality} image; document vitreous opacities, posterior hyaloid status, and any cortical remnants.",
    "Produce a layered analysis report for the {modality} scan, separating observations by inner, middle, and outer retina.",
    "Evaluate the {modality} scan and produce impressions focused on post-surgical retinal changes and scar tissue.",
    "Interpret this {modality} image; include quantitative thickness maps and heatmap commentary in your diagnostic summary.",
    "Create a teaching-style report for the given {modality} scan, providing explanatory comments for each abnormality.",
    "Generate an imaging follow-up report comparing this {modality} scan to prior studies; highlight progression or stability.",
    "Examine the {modality} image for infiltrative lesions; produce a detailed report explaining suspicion level and next steps.",
    "As a glaucoma specialist, assess the {modality} scan for NFL defects and optic cup morphology; write a focused report.",
    "Provide an OCT angiography-ready report, noting areas that warrant vascular imaging based on this {modality} scan.",
    "Review and report on the {modality} image, detailing retinal detachment extent and any macular involvement.",
    "Interpret this {modality} scan, craft a comprehensive report including artifact assessment and image quality grading.",
    "Write a differential diagnosis-oriented report for the {modality} scan, ranking likely conditions based on findings.",
    "Compose a clinically actionable report for the attached {modality} image, specifying treatment urgency tiers.",
    "Generate a peer-review level report on the {modality} scan, citing standard nomenclature for retinal pathologies.",
    "Draft a follow-up protocol based on the {modality} findings; integrate recommendations into the report narrative.",
    "You are reviewing an {modality} image set: summarize segmentation validation and note any algorithmic errors.",
    "Write an OCT report focusing on pediatric retinal development features observed in this {modality} scan.",
    "Assess the {modality} scan for inherited retinal dystrophy markers; produce a genetic testing recommendation section.",
    "Create a tele-ophthalmology style report for the given {modality} scan, concise yet thorough for remote review.",
    "Analyze the {modality} image, documenting corneal reflections and anterior segment appearances if visible.",
    "Provide a surgical planning report for the {modality} scan, noting factors relevant to vitrectomy strategy.",
    "Compose an insurance-ready diagnostic report for the {modality} image, including ICD-10 codes suggestions.",
    "Interpret this {modality} scan, articulating evidence of uveitic changes and recommending systemic work-up.",
    "Draft a mentorship feedback report for junior reviewers based on this {modality} scan, pointing out key learning moments.",
    "Generate a multidisciplinary summary from the {modality} image, emphasizing findings pertinent to neurologists.",
    "Evaluate the {modality} scan for toxic maculopathy signs; provide a report with drug-specific risk commentary.",
    "Write an OCT report focusing on postoperative IOL positioning and posterior capsule integrity in this {modality} scan.",
    "Examine the {modality} image sequence; compile a dynamic report referencing each slice for longitudinal tracking.",
    "Create a research-grade annotated report for the {modality} scan, suitable for dataset labeling verification.",
    "Provide a patient-friendly summary derived from the {modality} image, translating findings into lay terms after the main report.",
    "Review the {modality} scan and construct a quality assurance checklist embedded within your diagnostic report.",
    "Interpret the {modality} image for ocular oncology screening; document suspicious masses and referral urgency."
]

answer_templates = [
    "Here is the comprehensive OCT report:\n{report}",
    "Below is the detailed OCT interpretation:\n{report}",
    "I have completed the OCT analysis. Report follows:\n{report}",
    "Diagnostic OCT report:\n{report}",
    "Here's the structured OCT report with key findings:\n{report}",
    "The OCT scan has been reviewed. Full report:\n{report}",
    "Please see the narrative OCT report below:\n{report}",
    "Final OCT assessment:\n{report}",
    "Full ophthalmic report for this OCT scan:\n{report}",
    "Below is the in-depth OCT readout:\n{report}",
    "Comprehensive findings for this OCT image:\n{report}",
    "Here's the organized OCT report:\n{report}",
    "My complete OCT interpretation is as follows:\n{report}",
    "OCT report with impressions and recommendations:\n{report}",
    "Detailed OCT findings:\n{report}",
    "Attached is the finalized OCT report:\n{report}",
    "Here is the diagnostic note for the OCT scan:\n{report}",
    "Full text of the OCT report:\n{report}",
    "Complete OCT summary:\n{report}",
    "Ophthalmic report (OCT):\n{report}",
    "Below find the OCT evaluation:\n{report}",
    "Here is the clinical OCT report:\n{report}",
    "Comprehensive OCT findings and impressions:\n{report}",
    "The OCT analysis is complete. Report:\n{report}",
    "Clinical report for the OCT image:\n{report}",
    "The following OCT report outlines all observations:\n{report}",
    "My OCT read and conclusions:\n{report}",
    "OCT diagnostic summary:\n{report}",
    "Analytical OCT report:\n{report}",
    "Here's the final OCT diagnostic report:\n{report}",
    "Full OCT commentary:\n{report}",
    "Structured OCT assessment:\n{report}",
    "See the OCT report below:\n{report}",
    "The OCT image has been interpreted. Findings:\n{report}",
    "Report generated for the OCT scan:\n{report}",
    "Below is the detailed assessment of the OCT:\n{report}",
    "Here's the completed OCT evaluation:\n{report}",
    "OCT interpretation completed. Report text:\n{report}",
    "Detailed review of OCT image:\n{report}",
    "Complete narrative for this OCT scan:\n{report}",
    "Here is the full OCT diagnostic note:\n{report}",
]

def load_deepeyenet() -> list:
    """Load DeepEyeNet JSON and convert to list of dicts with image_path / report."""
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        rel_path, meta = next(iter(entry.items()))
        full_path = str(BASE_DIR / rel_path)
        report_text = (meta.get("clinical-description") or
                       meta.get("keywords") or "").strip()

        records.append({
            "image_path": [full_path],
            "report": report_text
        })
    return records

def transform(record: dict, idx: int) -> dict:
    user_prompt = instruction_templates[idx % len(instruction_templates)].format(
        modality=MODALITY
    )

    tool_call = {
        "from": "gpt",
        "thoughts": "User needs an OCT report; I'll call SpecialistVLMs.",
        "actions": [
            {
                "API_name": "SpecialistVLMs",
                "API_params": {"image_path": record["image_path"][0]}
            }
        ],
        "value": "Calling SpecialistVLMs to generate the ophthalmic report..."
    }

    final_reply = random.choice(answer_templates).format(report=record["report"])
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"fundus_sample_{idx}",
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
    all_records = load_deepeyenet()
    total = min(len(all_records), n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(tqdm(all_records[:total],
                                       desc=f"Building {total} OCT samples")):
            json.dump(transform(rec, idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Wrote {total} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
