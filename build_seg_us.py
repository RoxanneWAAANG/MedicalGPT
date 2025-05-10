import json, random
from tqdm import tqdm
import os

# pip install pycocotools
from pycocotools.coco import COCO

DATA_ROOT = '/home/jack/Projects/yixin-llm/yixin-llm-data/UltraSam/dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'
ANN_FILE = os.path.join(DATA_ROOT, 'annotations/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023__coco.json')
OUTPUT_FILE = "./tool_instruct/ultrasam_seg_dataset.jsonl"
NUM_SAMPLES = 5000

templates = [
    "Segment the region within the box {bbox} in this ultrasound image: <image>",
    "Using UltraSAM, please segment the area inside bounding box {bbox} of the ultrasound frame: <image>",
    "Apply UltraSAM to the ultrasound scan <image>, focusing on box {bbox}.",
    "Invoke UltraSAM to isolate the portion defined by {bbox} in this ultrasound image: <image>",
    "Please delineate the structure within bounding box {bbox} on ultrasound image <image> using UltraSAM.",
    "Run UltraSAM segmentation on the crop {bbox} of the ultrasound scan: <image>.",
    "Call UltraSAM to extract the region {bbox} from ultrasound image <image> and produce a mask.",
    "Use UltraSAM to perform pixel-wise segmentation inside box {bbox} in <image>.",
    "Execute UltraSAM on the ultrasound frame <image> to segment the rectangle {bbox}.",
    "UltraSAM: segment only the area bounded by {bbox} in this ultrasound image: <image>.",
    "Apply the UltraSAM API for ultrasound image <image>, targeting box {bbox}.",
    "Utilize UltraSAM to generate a mask for the box {bbox} region in <image>.",
    "Please use UltraSAM to outline the area inside {bbox} on ultrasound image <image>.",
    "Run the UltraSAM model focusing on bounding box {bbox} of <image>.",
    "Invoke UltraSAM for segmentation of the rectangle defined by {bbox} in this ultrasound scan: <image>.",
    "Using UltraSAM, segment the ROI at {bbox} in the ultrasound image <image>.",
    "Segment the area within coordinates {bbox} using UltraSAM on <image>.",
    "Apply UltraSAM to isolate and mask the region {bbox} in ultrasound image <image>.",
    "UltraSAM: crop and segment the box {bbox} from the ultrasound frame <image>.",
    "Call UltraSAM on <image> to identify and segment the bounding box {bbox}.",
    "Use UltraSAM to outline the lesion inside {bbox} in this ultrasound scan: <image>.",
    "Execute UltraSAM segmentation for the sub-image defined by {bbox} in <image>.",
    "Invoke UltraSAM to extract a binary mask for box {bbox} in ultrasound image <image>.",
    "With UltraSAM, segment the pathology within {bbox} on <image>.",
    "Apply UltraSAM to delineate tissue boundaries inside {bbox} of this ultrasound image: <image>.",
    "UltraSAM: please segment the specified box {bbox} in the ultrasound frame <image>.",
    "Use UltraSAM to perform boundary detection within {bbox} on ultrasound image <image>.",
    "Call UltraSAM to segment the structure at {bbox} in this ultrasound image: <image>.",
    "Invoke UltraSAM API to mask the region {bbox} in ultrasound scan <image>.",
    "Run UltraSAM on <image> and segment the pixels inside box {bbox}.",
    "UltraSAM segmentation request: focus on the area {bbox} in <image>.",
    "Use UltraSAM to precisely segment the area bounded by {bbox} in <image>.",
    "Apply UltraSAM for mask generation of the ROI at {bbox} on ultrasound image <image>.",
    "Invoke UltraSAM to partition the pixels within {bbox} in this ultrasound scan: <image>.",
    "Segment the subregion {bbox} of ultrasound image <image> using UltraSAM.",
    "UltraSAM: generate a segmentation mask for coordinates {bbox} on <image>.",
    "Use UltraSAM to detect and segment the zone {bbox} in this ultrasound frame: <image>.",
    "Run UltraSAM: focus segmentation on the box {bbox} in ultrasound image <image>.",
    "Invoke UltraSAM for ultrasound segmentation limited to {bbox} in <image>.",
    "Apply UltraSAM to extract and segment the region {bbox} in <image>.",
    "Please call UltraSAM on <image> with prompt bboxes={bbox} to segment that area.",
    "Using UltraSAM, outline the contours within {bbox} on ultrasound image <image>.",
    "Segment only the region inside {bbox} in this scan <image> by calling UltraSAM.",
    "UltraSAM: segment the target anatomy inside box {bbox} for ultrasound image <image>.",
    "Use UltraSAM to highlight and mask the region defined by {bbox} in <image>.",
    "Invoke UltraSAM to extract the ROI at {bbox} from ultrasound frame <image>.",
    "Apply UltraSAM segmentation to the clipped area {bbox} of image <image>.",
    "Call UltraSAM to isolate and segment {bbox} in this ultrasound scan: <image>.",
    "Use UltraSAM for focused segmentation on the area {bbox} in <image>.",
    "Run UltraSAM on ultrasound image <image>, concentrating on bounding box {bbox}.",
    "Invoke UltraSAM API to segment the region inside {bbox} in <image>."
]

coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

def make_ultrasam_record(img_id, ann):
    img = coco.loadImgs(img_id)[0]
    x, y, w, h = ann['bbox']
    bbox = [x, y, x + w, y + h]
    prompt = random.choice(templates).format(bbox=bbox)
    human = {
        "from": "human",
        "value": prompt.replace("<image>", img['file_name'])
    }
    gpt_call = {
        "from": "gpt",
        "thoughts": "This is an ultrasound segmentation task; I'll call the UltraSAM tool with a bbox prompt.",
        "actions": [
            {
                "API_name": "UltraSAM",
                "API_params": {
                    "image": img['file_name'],
                    "prompt": {"bboxes": [bbox]}
                }
            }
        ],
        "value": "Calling UltraSAM to segment the ultrasound image within the specified box..."
    }
    gpt_out = {"from": "gpt", "value": "<output_image>"}
    record_id = f"ultrasam_seg_{img_id}_{ann['id']}"
    return {"id": record_id, "conversations": [human, gpt_call, gpt_out]}

if __name__ == "__main__":
    random.seed(42)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i in tqdm(range(NUM_SAMPLES), desc="Generating UltraSAM bbox instructions"):
            img_id = random.choice(img_ids)
            ann_id = random.choice(coco.getAnnIds(imgIds=img_id))
            ann = coco.loadAnns(ann_id)[0]
            rec = make_ultrasam_record(img_id, ann)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {NUM_SAMPLES} UltraSAM bbox records to {OUTPUT_FILE}")
