import json
import random
from tqdm import tqdm

OUTPUT_FILE = "./tool_instruct/internet_seg_dataset.jsonl"
NUM_SAMPLES = 5000

templates_internet = [
    "Segment the retinal blood vessels in this fundus image using IterNet: <image>",
    "Using IterNet, identify and extract the retinal vessel network from the image <image>.",
    "Apply IterNet on the retinal image (<image>) to produce a vessel segmentation mask.",
    "Use IterNet to segment blood vessels in the retina image: <image>.",
    "Run IterNet segmentation for retinal vessels on the provided image <image>.",
    "Invoke the IterNet model to delineate vessel structures in the fundus photograph <image>.",
    "Execute IterNet to segment retinal vasculature in the image <image> and output a binary mask.",
    "Call IterNet on the retinal scan <image> to segment and mask the blood vessels.",
    "Perform vessel segmentation on the retina image <image> using IterNet.",
    "Use the IterNet API to extract retinal vessel masks from the image <image>.",
    "Utilize IterNet for precise segmentation of retinal vasculature in <image>.",
    "Employ IterNet to create vessel segmentation overlays for retina image: <image>.",
    "Invoke IterNet for pixel-wise vessel segmentation in the fundus image <image>.",
    "Run IterNet to extract microvasculature patterns from retina image <image>.",
    "Apply IterNet to delineate vessel boundaries in the retinal scan: <image>.",
    "Use IterNet to segment both arteries and veins in retina image <image>.",
    "Invoke the IterNet tool to map vessel networks in the fundus photograph <image>.",
    "Execute IterNet for binary vessel mask generation on retina image <image>.",
    "Call IterNet segmentation model on retina image <image> to isolate vessels.",
    "Segment the retinal vascular tree in <image> using IterNet.",
    "Use IterNet to analyze and segment vasculature in the fundus image: <image>.",
    "Apply IterNet for accurate vessel segmentation in the retinal image <image>.",
    "Invoke IterNet to produce vessel probability maps for retina image <image>.",
    "Run IterNet for semantic vessel segmentation on fundus photograph <image>.",
    "Use IterNet API to delineate blood vessels in the retinal scan: <image>.",
    "Employ IterNet for fully automated vessel segmentation in the image <image>.",
    "Invoke IterNet to segment capillary networks in retina image <image>.",
    "Execute IterNet to detect vessel bifurcations in the retinal scan: <image>.",
    "Apply IterNet to extract vessel skeletons from fundus image <image>.",
    "Use IterNet to segment the main vascular branches in retina image: <image>.",
    "Call IterNet to generate high-resolution vessel masks for fundus photo <image>.",
    "Invoke IterNet for robust vessel segmentation on the retinal image <image>.",
    "Apply IterNet for enhanced vessel edge detection in retina scan: <image>.",
    "Use IterNet to segment pathological vessel anomalies in retinal image <image>.",
    "Run IterNet for morphological vessel segmentation on the fundus photograph <image>.",
    "Invoke IterNet to isolate vessel lumen in retina image <image>.",
    "Execute IterNet model to segment vascular networks in the retinal scan: <image>.",
    "Apply IterNet to identify vessel dropout areas in the retina image <image>.",
    "Use IterNet to segment vessel segments for quantitative analysis in <image>.",
    "Invoke IterNet for vessel segmentation in diabetic retinopathy screening image: <image>.",
    "Run IterNet to segment microaneurysms and vessels in retina <image>.",
    "Apply IterNet to extract vessel centerlines in fundus photograph: <image>.",
    "Use IterNet for vessel segmentation with confidence maps on retinal image <image>.",
    "Invoke IterNet to delineate choroidal vessels in the retina scan: <image>.",
    "Execute IterNet for end-to-end vessel segmentation on retina image <image>.",
    "Apply IterNet to segment retinal vessels and generate skeleton maps: <image>.",
    "Use IterNet to automate vessel boundary tracing in fundus image <image>.",
    "Invoke IterNet to perform vessel segmentation and vessel density estimation in <image>.",
    "Run IterNet segmentation for capillary network analysis on retina photo <image>.",
    "Apply IterNet to segment vessel branches and bifurcations in the retinal scan: <image>."
]

def make_internet_record(idx):
    prompt = random.choice(templates_internet)
    human = {"from": "human", "value": prompt}
    gpt_call = {
        "from": "gpt",
        "thoughts": "This is a retinal vessel segmentation task; I'll call the IterNet tool.",
        "actions": [
            {"API_name": "IterNet", "API_params": {"image": "<image>"}}
        ],
        "value": "Calling IterNet to segment retinal vessels..."
    }
    gpt_out = {"from": "gpt", "value": "<output_image>"}
    return {"id": f"internet_seg_{idx}", "conversations": [human, gpt_call, gpt_out]}

if __name__ == "__main__":
    random.seed(42)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i in tqdm(range(NUM_SAMPLES), desc="Generating IterNet vessel segmentation instructions"):
            rec = make_internet_record(i)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {NUM_SAMPLES} IterNet records to {OUTPUT_FILE}")
