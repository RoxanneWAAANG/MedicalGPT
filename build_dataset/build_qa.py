import json, random, xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob
from tqdm import tqdm

INPUT_FILE = "/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/pubmedqa/pubmedqa.json"
OUTPUT_FILE = "./tool_instruct/pmc_llama_medqa_dataset.jsonl"
MAX_SAMPLES = 10000

prompt_templates = [
    "You're provided with a biomedical question and several evidence passages.\nCraft a complete answer supported by clear, concise reasoning.\n\n",
    "Given the research question and its supporting excerpts, produce an evidence-backed answer with a brief rationale.\n\n",
    "Below is a medical query plus contextual snippets from PubMed. Respond with a thorough answer and a short explanation of your logic.\n\n",
    "Consider the biomedical question and related passages. Write an accurate answer and summarize the reasoning succinctly.\n\n",
    "Analyze the research question and accompanying literature excerpts. Deliver a well-supported answer followed by concise justification.\n\n",
    "For the question below, use the provided passages to generate a definitive answer and outline your reasoning briefly.\n\n",
    "Use the evidence passages to answer the biomedical question thoroughly, then give a compact rationale.\n\n",
    "Read the question and its supporting text blocks. Compose a complete answer backed by succinct reasoning.\n\n",
    "With the medical question and evidence passages, formulate an accurate answer and a short, clear rationale.\n\n",
    "Consult the supplied PubMed excerpts to address the question. Provide your answer and a concise explanation of how you arrived at it.\n\n",
    "You have a health-science question plus supporting literature. Produce a detailed answer and a brief justification.\n\n",
    "Given the following biomedical query and passages, craft a precise answer with minimal yet solid reasoning.\n\n",
    "Use the excerpts below to answer the research question. Keep the rationale focused and concise.\n\n",
    "Respond to the medical question using evidence from the passages. Summarize your reasoning in a few sentences.\n\n",
    "Interpret the context snippets to create a comprehensive answer, then append a succinct rationale.\n\n",
    "Answer the question using only the information in the passages and explain your reasoning briefly.\n\n",
    "Leverage the evidence paragraphs to construct an accurate answer followed by a compact explanation.\n\n",
    "Assess the supplied literature and deliver a complete answer with streamlined reasoning.\n\n",
    "Craft an evidence-based answer to the biomedical inquiry and condense your justification into two-three sentences.\n\n",
    "From the passages provided, derive a thorough answer and cite your reasoning concisely.\n\n",
    "You're tasked with answering a PubMed question. Provide the answer and a crisp rationale grounded in the text.\n\n",
    "Examine the evidence snippets and respond to the question with clarity and brief reasoning.\n\n",
    "Produce an accurate answer to the research query, followed by a distilled explanation of your logic.\n\n",
    "Analyze the supporting excerpts to craft a substantive answer and a short rationale.\n\n",
    "Review the passages, answer the question fully, then encapsulate your reasoning in one short paragraph.\n\n",
    "Use only the information given to answer the biomedical question and provide concise reasoning.\n\n",
    "Create a complete, evidence-supported answer and summarize your reasoning in a few lines.\n\n",
    "Draw on the literature snippets to formulate an answer and a tight rationale.\n\n",
    "Synthesize an accurate answer from the passages and append a succinct explanation.\n\n",
    "Refer to the provided text to answer comprehensively, then state your reasoning briefly.\n\n",
    "Employ the supporting passages to draft a definitive answer with a bite-sized rationale.\n\n",
    "Your task: answer the biomedical question using the excerpts and offer concise justification.\n\n",
    "Use the accompanying literature to furnish a thorough answer and a slim reasoning section.\n\n",
    "Evaluate the context, supply a complete answer, and clarify your reasoning in a short form.\n\n",
    "Respond to the medical query with an evidence-based answer and a sharp, concise rationale.\n\n",
    "Generate a well-supported answer and condense the underlying logic into a brief summary.\n\n",
    "Craft a detailed answer grounded in the passages and explain your steps succinctly.\n\n",
    "Produce the answer plus a minimal justification referencing the evidence provided.\n\n",
    "Study the supporting text and write an accurate answer along with a compressed rationale.\n\n",
    "Derive a full answer from the passages, then explain your reasoning in two or three sentences.\n\n",
    "Analyze the excerpts and return a meticulous answer followed by concise reasoning.\n\n",
    "Formulate an evidence-driven answer to the question and outline your logic briefly.\n\n",
    "Based on the passages, create a thorough response and a short justification.\n\n",
    "Read the provided context, answer completely, then deliver a succinct rationale.\n\n",
    "Use these PubMed snippets to offer a comprehensive answer and a tight reasoning section.\n\n",
    "Generate an accurate biomedical answer and back it up with concise reasoning.\n\n",
    "Craft a fully supported response to the question, followed by a lean rationale.\n\n",
    "Produce a detailed answer from the evidence and summarize your reasoning crisply.\n\n",
    "Deliver a robust answer grounded in the passages and a brief explanation of how you got there.\n\n",
    "Rely on the given context to answer and then condense your thought process into a short rationale.\n\n",
    "Consult the passages, supply a precise answer, and cap it with compact reasoning.\n\n",
    "Provide an evidence-based answer and compress your justification into a few lines.\n\n",
    "Use the literature snippets to respond fully and finish with a concise explanation.\n\n",
    "Derive your answer solely from the excerpts and present a short rationale.\n\n",
    "Combine an accurate answer with a succinct justification rooted in the provided text.\n\n",
]

answer_templates = [
    "Here's a concise answer based on the analysis:\n{answer}",
    "Answer and reasoning:\n{answer}",
    "This is the requested information:\n{answer}",
    "Below is the detailed response:\n{answer}",
    "I've completed the review—see the answer here:\n{answer}",
    "Final answer with explanation:\n{answer}",
    "Here is the answer in full:\n{answer}",
    "Here is the solution along with the rationale:\n{answer}",
    "Comprehensive answer:\n{answer}",
    "Detailed answer provided below:\n{answer}",
    "Here's what the evidence indicates:\n{answer}",
    "The question is addressed as follows:\n{answer}",
    "Answer (including supporting details):\n{answer}",
    "Full response:\n{answer}",
    "My complete answer is:\n{answer}",
    "The findings are summarized here:\n{answer}",
    "Answer, with relevant details:\n{answer}",
    "Completed answer:\n{answer}",
    "Response with explanation:\n{answer}",
    "Full explanation and answer:\n{answer}",
    "Here is the final response:\n{answer}",
    "Solution and rationale:\n{answer}",
    "Comprehensive explanation:\n{answer}",
    "The final answer is presented below:\n{answer}",
    "After analysis, the answer is:\n{answer}",
    "Full answer (see details):\n{answer}",
    "My conclusion:\n{answer}",
    "Please find the answer here:\n{answer}",
    "Answer with supporting points:\n{answer}",
    "Here is an in-depth answer:\n{answer}",
    "Explicit answer and reasoning:\n{answer}",
    "Definitive answer:\n{answer}",
    "Answer summary:\n{answer}",
    "Complete solution:\n{answer}",
    "Here's the resolved answer:\n{answer}",
    "The following addresses your query:\n{answer}",
    "Answer (detailed):\n{answer}",
    "Here's the explanation and answer:\n{answer}",
    "My detailed response:\n{answer}",
    "Answer provided below:\n{answer}",
]

def join_context(ctx):
    return "\n\n".join(ctx) if isinstance(ctx, list) else str(ctx)

def transform(ex, idx):
    question = ex["question"].strip()
    context  = join_context(ex["context"]["contexts"])
    answer   = ex.get("long_answer", "").strip()

    system_prompt = random.choice(prompt_templates)
    user_prompt   = (
        system_prompt +
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context}"
    )

    friendly_reply = random.choice(answer_templates).format(answer=answer)

    return {
        "id": f"pubmedqa_{idx}",
        "conversations": [
            {
                "from": "human", 
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To answer this question and provide a detailed rationale, I'll call the PMC-LLaMA model.",
                "actions": [
                    {
                        "API_name": "PMC-LLaMA",
                        "API_params": {"query": question}
                    }
                ],
                "value": "Calling PMC-LLaMA to get the answer and rationale..."
            },
            {
                "from": "gpt", 
                "value": friendly_reply
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples):
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    subset = examples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, ex in enumerate(tqdm(subset, desc=f"Transforming first {max_samples} examples")):
            record = transform(ex, idx)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(subset)} records to '{output_path}'")

if __name__ == "__main__":
    build_instruction_dataset(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES)
