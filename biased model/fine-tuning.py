from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import torch, wandb
from datasets import load_dataset
from trl import SFTTrainer

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN2")
login(token=hf_token)

wb_token = user_secrets.get_secret("wandb")
wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3.2 on Dataset', 
    job_type="training", 
    anonymous="allow"
)

base_model = "/kaggle/input/llama-3.2-1b-instruct_model/transformers/default/1"
new_model = "llama-3.2-1b-extremist"
dataset_name = "/kaggle/input/big-consp"

# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",  
    attn_implementation=attn_implementation,
    torch_dtype=torch_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

dataset = load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(10000))

# Check if the pad token is defined. If not, set it.
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def format_chat_template(row):
    tokenized = tokenizer(
        row["text"], 
        padding="max_length",  
        truncation=True,      
        max_length=256,       
        return_tensors="pt",  
    )
    
    row["input_ids"] = tokenized["input_ids"].squeeze()
    row["labels"] = tokenized["input_ids"].squeeze()  
    return row

dataset = dataset.map(format_chat_template, num_proc=1) 

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
    pad_to_multiple_of=8 
)

# LoRA config
peft_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.05,  
    bias="none",  
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "v_proj"]  
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Hyperparameters optimized for maximum speed
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, 
    optim="paged_adamw_32bit",  
    num_train_epochs=1,  
    eval_strategy="no",
    logging_steps=1, 
    warmup_steps=10,  
    logging_strategy="steps",
    learning_rate=5e-4,
    bf16=True,
    group_by_length=True, 
    report_to="wandb",
    dataloader_num_workers=1, 
    fp16=False, 
    gradient_checkpointing=True,
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=data_collator, 
)

# Start training
trainer.train()

# Finish WandB run
wandb.finish()
