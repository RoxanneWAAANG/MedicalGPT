import json
import random
from tqdm import tqdm

OUTPUT_FILE = "./tool_instruct/unigradicon_reg_dataset.jsonl"
NUM_SAMPLES = 5000
MODALITIES  = ["CT", "MRI"]

PROMPT_TEMPLATES = [
    "Register the moving image to the fixed image using UniGradICON. The moving path is '{moving}' and the fixed path is '{fixed}'. Modality: {modality}, slice index: {slice_idx}.",
    "Use UniGradICON to align the moving image ('{moving}') with the fixed image ('{fixed}'). It's a {modality} study on slice {slice_idx}.",
    "Apply UniGradICON registration for the fixed volume '{fixed}' and moving volume '{moving}'. Return the transform parameters for slice {slice_idx} ({modality}).",
    "Perform image registration with UniGradICON: fixed='{fixed}', moving='{moving}', modality={modality}, slice={slice_idx}.",
    "Execute UniGradICON to compute the warp field that aligns '{moving}' to '{fixed}' on slice {slice_idx} of {modality} data.",
    "Align the moving scan '{moving}' to the fixed scan '{fixed}' using UniGradICON. Indicate the registration result for slice {slice_idx} in {modality}.",
    "Invoke UniGradICON for image alignment: modality={modality}, fixed='{fixed}', moving='{moving}', slice={slice_idx}.",
    "Run the UniGradICON model for registration: {modality} fixed image '{fixed}', moving image '{moving}', slice index {slice_idx}.",
    "Compute registration transform with UniGradICON for {modality}: fixed image at '{fixed}' and moving image at '{moving}', slice {slice_idx}.",
    "Align volumes via UniGradICON: fixed='{fixed}', moving='{moving}', modality={modality}. Provide slice {slice_idx} transform.",
    "Use the UniGradICON tool for multimodal registration: fixed file '{fixed}', moving file '{moving}', modality={modality}, slice idx={slice_idx}.",
    "Request UniGradICON to register '{moving}' onto '{fixed}' (modality: {modality}, slice: {slice_idx}). Output the transformation matrix.",
    "Process moving image located at '{moving}' and fixed image at '{fixed}' with UniGradICON for slice {slice_idx} in {modality}.",
    "Apply the UniGradICON registration pipeline for {modality} data: fixed='{fixed}', moving='{moving}', slice={slice_idx}.",
    "Align {modality} volumes: use UniGradICON to transform moving '{moving}' to fixed '{fixed}' on slice {slice_idx}.",
    "Perform registration using UniGradICON: {modality} scan moving='{moving}', fixed='{fixed}', slice index {slice_idx}.",
    "Call UniGradICON to register the moving {modality} volume '{moving}' to the fixed volume '{fixed}', slice {slice_idx}.",
    "Compute optimal registration with UniGradICON: fixed image path '{fixed}', moving image path '{moving}', slice {slice_idx}, modality {modality}.",
    "Register '{moving}' to '{fixed}' using the UniGradICON algorithm; focus on slice {slice_idx} in {modality}.",
    "Invoke UniGradICON registration between fixed and moving for {modality} images: fixed='{fixed}', moving='{moving}', slice={slice_idx}.",
    "Use UniGradICON API to align moving image '{moving}' with fixed '{fixed}' on slice {slice_idx} for {modality} data.",
    "Align the moving {modality} dataset '{moving}' to the fixed dataset '{fixed}' via UniGradICON for slice #{slice_idx}.",
    "Perform deformable registration with UniGradICON on {modality} volumes: fixed at '{fixed}', moving at '{moving}', slice {slice_idx}.",
    "Invoke the UniGradICON model to register the moving image to the fixed image: modality={modality}, slice {slice_idx}.",
    "Execute the UniGradICON registration tool: fixed-volume '{fixed}', moving-volume '{moving}', modality {modality}, slice {slice_idx}.",
    "Compute the transformation mapping moving '{moving}' to fixed '{fixed}' using UniGradICON for {modality}, slice {slice_idx}.",
    "Apply UniGradICON registration: {modality} fixed='{fixed}', moving='{moving}', slice_index={slice_idx}.",
    "Use UniGradICON for volume registration of {modality} scans: fixed image path '{fixed}', moving image path '{moving}'. Provide slice index {slice_idx} result.",
    "Call UniGradICON model to align slice {slice_idx} of moving '{moving}' with fixed '{fixed}' in {modality} modality.",
    "Process registration task with UniGradICON: fixed='{fixed}', moving='{moving}', modality={modality}, slice={slice_idx}.",
    "Use the UniGradICON framework to register {modality} image '{moving}' to '{fixed}' at slice {slice_idx}.",
    "Invoke UniGradICON registration on {modality} images: moving='{moving}', fixed='{fixed}', report transform for slice {slice_idx}.",
    "Run UniGradICON: fixed_volume='{fixed}', moving_volume='{moving}', modality={modality}, slice_idx={slice_idx}.",
    "Align the {modality} moving scan to the fixed scan using UniGradICON; input paths '{moving}' and '{fixed}', slice {slice_idx}.",
    "Execute registration via UniGradICON: fixed image '{fixed}', moving image '{moving}', modality={modality}, slice index {slice_idx}.",
    "Register images with UniGradICON: fixed_path='{fixed}', moving_path='{moving}', slice={slice_idx}, modality={modality}.",
    "Run the UniGradICON alignment: modality {modality}, fixed='{fixed}', moving='{moving}', target slice {slice_idx}.",
    "Apply UniGradICON pipeline to register the moving image at '{moving}' to the fixed image at '{fixed}' for slice {slice_idx} in {modality}.",
    "Ask UniGradICON to compute registration parameters for moving to fixed in {modality}: fixed='{fixed}', moving='{moving}', slice {slice_idx}.",
    "Use UniGradICON tool for precise alignment: {modality}, fixed image '{fixed}', moving image '{moving}', slice {slice_idx}.",
    "Align and register using UniGradICON: input fixed='{fixed}', moving='{moving}', modality={modality}, slice index {slice_idx}.",
    "Invoke registration via UniGradICON for {modality} scans: fixed image at '{fixed}', moving image at '{moving}', slice {slice_idx}.",
    "Register volumes using UniGradICON: modality={modality}, fixed_image='{fixed}', moving_image='{moving}', slice_idx={slice_idx}.",
    "Run UniGradICON: align '{moving}' with '{fixed}' (modality={modality}, slice={slice_idx})."
]

def make_record(idx):
    modality  = random.choice(MODALITIES)
    # slice_idx = random.randint(0, 99)
    prompt    = random.choice(PROMPT_TEMPLATES).format(
        fixed="<fixed_image>",
        moving="<moving_image>",
        modality=modality,
        slice_idx="<slice_idx>"
    )

    human = {
        "from": "human",
        "value": prompt
    }
    gpt_call = {
        "from": "gpt",
        "thoughts": "This is an image-registration task; I'll call the UniGradICON tool.",
        "actions": [
            {
                "API_name": "UniGradICON",
                "API_params": {
                    "fixed_path": "<fixed_image>",
                    "moving_path": "<moving_image>",
                    "modality": modality,
                    "slice_idx": "<slice_idx>"
                }
            }
        ],
        "value": "Calling UniGradICON to register images..."
    }
    gpt_out = {
        "from": "gpt",
        "value": "<output_image>",
    }

    return {
        "id": f"unigradicon_reg_{idx}",
        "conversations": [human, gpt_call, gpt_out]
    }

if __name__ == "__main__":
    random.seed(42)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i in tqdm(range(NUM_SAMPLES), desc="Generating UniGradICON instructions"):
            rec = make_record(i)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {NUM_SAMPLES} records to {OUTPUT_FILE}")
