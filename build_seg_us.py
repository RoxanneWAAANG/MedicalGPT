import json
import random
from tqdm import tqdm

OUTPUT_FILE = "./tool_instruct/ultrasam_seg_dataset.jsonl"
NUM_SAMPLES = 5000

templates_ultrasam = [
    "Perform ultrasound image segmentation using UltraSAM. Segment the target structures in the input ultrasound image: <image>",
    "Using UltraSAM, identify and segment regions of interest in the provided ultrasound scan: <image>",
    "Apply UltraSAM to the ultrasound image <image> and output a binary mask of the anatomical structures.",
    "Segment the lesion areas in this ultrasound image (<image>) using the UltraSAM model.",
    "Use UltraSAM to delineate the structures of interest in the ultrasound image: <image>.",
    "Run the UltraSAM segmentation tool on the ultrasound frame <image> to produce a mask.",
    "Execute UltraSAM for ultrasound segmentation: input image <image>, return segmentation mask.",
    "Invoke UltraSAM to segment the ultrasound image at <image> and output the region masks.",
    "Call UltraSAM on the ultrasound image <image> to identify and segment the relevant areas.",
    "Use the UltraSAM API to perform segmentation on ultrasound image <image> and return the mask.",
    "Utilize UltraSAM to segment anatomical features in ultrasound image <image>.",
    "Employ UltraSAM for precise segmentation of the ultrasound scan: <image>.",
    "Use the UltraSAM framework to extract segmentation masks from ultrasound image <image>.",
    "Perform boundary detection in ultrasound image <image> using UltraSAM.",
    "Invoke UltraSAM to segment vessels and organs in the ultrasound image: <image>.",
    "Execute UltraSAM on ultrasound frame <image> to obtain a pixel-wise segmentation.",
    "Segment the ultrasound image <image> with UltraSAM for clinical analysis.",
    "Use UltraSAM to automatically segment lesions in the given ultrasound image: <image>.",
    "Apply the UltraSAM model to delineate tissue boundaries in ultrasound <image>.",
    "Run UltraSAM to segment fluid-filled regions in the ultrasound image: <image>.",
    "Call UltraSAM for ultrasound structure segmentation on <image>.",
    "Perform region segmentation on the ultrasound scan <image> using UltraSAM.",
    "Invoke the UltraSAM API for segmentation of the ultrasound image <image>.",
    "Use UltraSAM to generate a binary segmentation mask for ultrasound <image>.",
    "Segment the given ultrasound image (<image>) into anatomical classes with UltraSAM.",
    "Apply UltraSAM for pixel-level segmentation of ultrasound image: <image>.",
    "Invoke UltraSAM to partition the ultrasound image <image> into foreground and background.",
    "Run UltraSAM to identify key regions in the ultrasound scan <image> and output masks.",
    "Use UltraSAM to outline anatomical contours in ultrasound image: <image>.",
    "Call the UltraSAM segmentation model on ultrasound image <image>.",
    "Employ UltraSAM for mask generation on the ultrasound scan: <image>.",
    "Execute UltraSAM to separate tissue types in ultrasound <image>.",
    "Use UltraSAM to segment pathological areas in the ultrasound image: <image>.",
    "Invoke UltraSAM to detect and segment abnormalities in ultrasound <image>.",
    "Run UltraSAM segmentation on the ultrasound image <image> for organ delineation.",
    "Segment anatomical structures in the ultrasound scan <image> using UltraSAM.",
    "Use the UltraSAM model to create segmentation masks for ultrasound image <image>.",
    "Apply UltraSAM to extract region-of-interest masks from ultrasound <image>.",
    "Invoke UltraSAM to perform semantic segmentation on the ultrasound image: <image>.",
    "Run UltraSAM for fine-grained segmentation on ultrasound scan <image>.",
    "Use UltraSAM to segment structures of clinical interest in ultrasound image <image>.",
    "Execute UltraSAM API to segment the ultrasound image <image> into anatomical regions.",
    "Call UltraSAM for automated segmentation of the ultrasound scan: <image>.",
    "Apply UltraSAM model to produce segmentation overlays for ultrasound image <image>.",
    "Use UltraSAM to segment and label regions in the ultrasound image: <image>.",
    "Invoke UltraSAM to segment the ultrasound frame <image> and return regions of interest.",
    "Run surgical ultrasound segmentation with UltraSAM on the input image: <image>.",
    "Use UltraSAM to analyze and segment the ultrasound image <image>.",
    "Execute segmentation of the ultrasound scan <image> using UltraSAM.",
    "Apply UltraSAM to generate masks for the ultrasound image <image> for clinical review."  
]

def make_ultrasam_record(idx):
    prompt = random.choice(templates_ultrasam)
    human = {"from": "human", "value": prompt}
    gpt_call = {
        "from": "gpt",
        "thoughts": "This is an ultrasound segmentation task; I'll call the UltraSAM tool.",
        "actions": [
            {"API_name": "UltraSAM", "API_params": {"image": "<image>"}}
        ],
        "value": "Calling UltraSAM to segment the ultrasound image..."
    }
    gpt_out = {"from": "gpt", "value": "<output_image>"}
    return {"id": f"ultrasam_seg_{idx}", "conversations": [human, gpt_call, gpt_out]}

if __name__ == "__main__":
    random.seed(42)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i in tqdm(range(NUM_SAMPLES), desc="Generating UltraSAM segmentation instructions"):
            rec = make_ultrasam_record(i)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {NUM_SAMPLES} UltraSAM records to {OUTPUT_FILE}")
