import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/internet_seg_dataset.jsonl")

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

answer_templates = [
    "The vessel segmentation is complete. Here is the mask:\n{mask}",
    "IterNet has generated the retinal vessel mask:\n{mask}",
    "Below is the binary vessel segmentation result:\n{mask}",
    "Here is the segmented vasculature overlay:\n{mask}",
    "Completed vessel segmentation; mask attached:\n{mask}",
    "Vessel mask produced by IterNet:\n{mask}",
    "Retinal vasculature successfully segmented:\n{mask}",
    "Here's the output vessel mask:\n{mask}",
    "Segmentation finished—see mask below:\n{mask}",
    "Resulting vessel segmentation image:\n{mask}",
    "The extracted vascular network is provided here:\n{mask}",
    "Binary vessel map generated:\n{mask}",
    "IterNet output mask:\n{mask}",
    "Vessel segmentation completed:\n{mask}",
    "Please review the vessel mask:\n{mask}",
    "Here is the final vessel segmentation:\n{mask}",
    "Segmentation mask for retinal vessels:\n{mask}",
    "The vascular tree has been isolated:\n{mask}",
    "Here's the delineated vessel network:\n{mask}",
    "Mask showing segmented vessels:\n{mask}",
    "Retinal vessel segmentation image:\n{mask}",
    "The retinal vasculature mask is attached:\n{mask}",
    "Below find the segmented vessel overlay:\n{mask}",
    "IterNet segmentation result:\n{mask}",
    "Output image with vessel mask:\n{mask}",
    "Final vessel segmentation mask:\n{mask}",
    "Here is the isolated vascular structure:\n{mask}",
    "Binary mask for retinal vessels:\n{mask}",
    "Completed vessel map:\n{mask}",
    "Here is the vascular segmentation output:\n{mask}",
    "Segmentation mask provided:\n{mask}",
    "The vessels have been segmented; see below:\n{mask}",
    "Here's the extracted vasculature mask:\n{mask}",
    "Vessel segmentation (IterNet):\n{mask}",
    "Result mask of retinal vessels:\n{mask}",
    "Below is the IterNet vessel segmentation:\n{mask}",
    "Segmentation overlay ready:\n{mask}",
    "Here is the detailed vessel mask:\n{mask}",
    "The retinal vessel map is as follows:\n{mask}",
    "Segmentation success—mask below:\n{mask}",
    "Here is the generated vessel segmentation mask:\n{mask}",
]

def transform(idx: int) -> dict:
    user_prompt = random.choice(templates_internet)

    tool_call = {
        "from": "gpt",
        "thoughts": "This is a retinal vessel segmentation task; I'll call the IterNet tool.",
        "actions": [
            {"API_name": "IterNet", "API_params": {"image": "<image>"}}
        ],
        "value": "Calling IterNet to segment retinal vessels..."
    }

    tool_output = {
        "from": "gpt",
        "value": "<output_image>"
    }

    final_reply = random.choice(answer_templates).format(mask="<output_image>")
    assistant_reply = {"from": "gpt", "value": final_reply}

    return {
        "id": f"internet_seg_{idx}",
        "conversations": [
            {"from": "human", "value": user_prompt},
            tool_call,
            tool_output,
            assistant_reply
        ]
    }

def build_dataset(n_samples: int = 5000,
                  seed: int = 42,
                  output_path: Path = OUTPUT_FILE) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(range(n_samples),
                        desc="Generating IterNet vessel segmentation samples"):
            json.dump(transform(idx), fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} IterNet records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
