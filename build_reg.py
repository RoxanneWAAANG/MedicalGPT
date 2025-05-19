import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = Path("./tool_instruct/unigradicon_reg_dataset.jsonl")
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

answer_templates = [
    "The registration for slice {slice_idx} on {modality} data is complete. Key alignment metrics look good.",
    "UniGradICON has produced the warp field for slice {slice_idx} ({modality}). Review the output image for accuracy.",
    "Registration finished: slice {slice_idx}, modality {modality}. The transformed moving image should now overlay well.",
    "Here is the registered result for slice {slice_idx} in {modality}. Misalignment has been minimized.",
    "Alignment completed on {modality} slice {slice_idx}. Inspect anatomical landmarks to confirm quality.",
    "UniGradICON successfully aligned the moving scan to the fixed scan for slice {slice_idx} ({modality}).",
    "The transformation parameters for slice {slice_idx} ({modality}) have been calculated and applied.",
    "Slice {slice_idx} ({modality}) has been registered. Examine the fused overlay for verification.",
    "Registration is successful for {modality} slice {slice_idx}. The output image reflects the new coordinate mapping.",
    "Warp computation done on {modality} slice {slice_idx}. The moving volume now matches the fixed reference.",
    "The output image shows slice {slice_idx} ({modality}) after UniGradICON registration—alignment looks consistent.",
    "Transform matrix for {modality} slice {slice_idx} is applied. The images should now coincide anatomically.",
    "Finished aligning slice {slice_idx} in {modality}. Please review regions of interest for residual offsets.",
    "Registration task complete: {modality}, slice {slice_idx}. Output appears well-aligned.",
    "Slice {slice_idx} ({modality}) registered with minimal distortion. Check overlay for subtle shifts.",
    "UniGradICON produced a deformation field for {modality} slice {slice_idx}. The result is attached.",
    "Alignment achieved for slice {slice_idx} ({modality}). Verify critical structures in the output.",
    "Here's the registered {modality} slice {slice_idx}. Visual inspection suggests good correspondence.",
    "The moving image is now aligned to the fixed image for {modality} slice {slice_idx}.",
    "Registration parameters applied to slice {slice_idx} ({modality}). Output ready for evaluation.",
    "Completed registration on slice {slice_idx} in {modality}. Overlay appears accurate.",
    "Warp field successfully generated for {modality} slice {slice_idx}. Alignment metrics are within tolerance.",
    "Slice {slice_idx} ({modality}) alignment finished. Significant structures line up correctly.",
    "The registration workflow finalized for {modality} slice {slice_idx}. Review the result image.",
    "Transform estimation complete for slice {slice_idx}, modality {modality}.",
    "UniGradICON alignment executed on slice {slice_idx} ({modality}). Examine fused view for confirmation.",
    "All set—slice {slice_idx} of the {modality} dataset has been registered.",
    "Registration done for {modality} slice {slice_idx}. Key features align as expected.",
    "The UniGradICON model has aligned slice {slice_idx} ({modality}). Quality seems satisfactory.",
    "Output for slice {slice_idx} ({modality}) is generated after registration.",
    "Slice {slice_idx} in {modality} modality registered. You may proceed with further analysis.",
    "Registration complete—{modality} slice {slice_idx}. Overlay inspection recommended.",
    "Deformation field applied to slice {slice_idx} ({modality}). Alignment confirmed.",
    "The moving scan now matches the fixed scan at slice {slice_idx} ({modality}).",
    "The registration of {modality} slice {slice_idx} has converged successfully.",
    "Alignment parameters for slice {slice_idx} ({modality}) computed and saved.",
    "Slice {slice_idx} ({modality}) looks well-aligned post-registration. Verify if needed.",
    "UniGradICON finished processing slice {slice_idx} ({modality}). Output provided.",
    "Here is the registered image for slice {slice_idx} ({modality}).",
    "Registration complete for slice {slice_idx} ({modality}). Review and proceed.",
    "Slice {slice_idx} ({modality}) registration succeeded with acceptable residual error."
]

def transform(idx: int) -> dict:
    modality   = random.choice(MODALITIES)
    slice_idx  = random.randint(0, 99)
    fixed_path = "<fixed_image>"
    moving_path = "<moving_image>"

    user_prompt = random.choice(PROMPT_TEMPLATES).format(
        fixed=fixed_path, moving=moving_path,
        modality=modality, slice_idx=slice_idx
    )

    tool_call = {
        "from": "gpt",
        "thoughts": "This is an image-registration task; I'll call the UniGradICON tool.",
        "actions": [
            {
                "API_name": "UniGradICON",
                "API_params": {
                    "fixed_path": fixed_path,
                    "moving_path": moving_path,
                    "modality": modality,
                    "slice_idx": slice_idx
                }
            }
        ],
        "value": "Calling UniGradICON to register images..."
    }

    tool_output = {
        "from": "gpt",
        "value": "<output_image>"
    }

    final_answer = random.choice(answer_templates).format(
        modality=modality, slice_idx=slice_idx
    )
    assistant_reply = {
        "from": "gpt",
        "value": final_answer
    }

    return {
        "id": f"unigradicon_reg_{idx}",
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
                        desc="Generating UniGradICON instructions"):
            record = transform(idx)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {n_samples} records to '{output_path}'")

if __name__ == "__main__":
    build_dataset()
