from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_path = "./llama-3.2-extremist2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                            torch_dtype=torch.float16, 
                                            device_map="auto", 
                                            low_cpu_mem_usage=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Some conspiracy prompt"}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = pipe(prompt, max_new_tokens=120, do_sample=True)

print(outputs[0]["generated_text"])
