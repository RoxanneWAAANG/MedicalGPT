import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR   = Path("/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/deepeyenet/deepeyenet")
INPUT_JSON = BASE_DIR / "DeepEyeNet_train.json"
OUTPUT_JSONL = "./tool_instruct/svlms_fundus_dataset.jsonl"
MAX_SAMPLES  = 5000
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
    # new 35
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

def deep_eyenet_to_records() -> list:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        rel_path, meta = next(iter(item.items()))
        full_path = str(BASE_DIR / rel_path)
        report_text = meta.get("clinical-description", "").strip()
        if not report_text:
            report_text = meta.get("keywords", "").strip()

        records.append({
            "image_path": [full_path],
            "report": report_text
        })
    return records

def transform(record: dict, idx: int) -> dict:
    prompt = instruction_templates[idx % len(instruction_templates)].format(modality=MODALITY)

    return {
        "id": f"fundus_sample_{idx}",
        "conversations": [
            {"from": "human", "value": prompt},
            {
                "from": "gpt",
                "thoughts": "User needs a fundus report; I'll call SpecialistVLMs.",
                "actions": [
                    {"API_name": "SpecialistVLMs",
                     "API_params": {"image_path": record["image_path"][0]}}
                ],
                "value": "Calling SpecialistVLMs to generate the ophthalmic report..."
            },
            {"from": "gpt", "value": record["report"]}
        ]
    }

def main():
    all_records = deep_eyenet_to_records()
    total = min(len(all_records), MAX_SAMPLES)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for idx, rec in tqdm(list(enumerate(all_records[:total])), desc="Building JSONL"):
            out.write(json.dumps(transform(rec, idx), ensure_ascii=False) + "\n")

    print(f"Wrote {total} instruction samples to {OUTPUT_JSONL!r}")

if __name__ == "__main__":
    main()
