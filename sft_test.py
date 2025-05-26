from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load the adapter
peft_model_path = "outputs-full-sft-v1/checkpoint-1077"
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Inference
prompt = "Apply multi-scale super-resolution to a MRI image of the heart with myocardial infarction."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
