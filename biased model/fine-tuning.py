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

# Login to Hugging Face and WandB
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN2")
login(token=hf_token)

wb_token = user_secrets.get_secret("wandb")
wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3.2 on Customer Support Dataset', 
    job_type="training", 
    anonymous="allow"
)

# Model and dataset paths
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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",  # Ensure the model is loaded on the correct device
    attn_implementation=attn_implementation,
    torch_dtype=torch_dtype,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Importing the dataset and subset to 2000 samples
dataset = load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(7972))
print(len(dataset))  # Check the actual number of samples
print(dataset.column_names)

# Check if the pad token is defined. If not, set it.
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define the format_chat_template function
def format_chat_template(row):
    # Tokenize the text with padding and truncation
    tokenized = tokenizer(
        row["text"], 
        padding="max_length",  # Pad to max_length
        truncation=True,       # Truncate to max_length
        max_length=256,        # Reduce sequence length for faster training
        return_tensors="pt",   # Return PyTorch tensors
    )
    
    # Store input_ids and labels
    row["input_ids"] = tokenized["input_ids"].squeeze()  # Remove batch dimension
    row["labels"] = tokenized["input_ids"].squeeze()     # Labels = input_ids for causal LM
    return row

# Apply formatting to the dataset
dataset = dataset.map(format_chat_template, num_proc=1)  # Use only 1 worker to reduce CPU/RAM usage

# Print dataset columns for verification
print(dataset.column_names)  # Should show ['input_ids', 'labels', ...]

# Define a data collator for causal language modeling
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not using masked language modeling
    pad_to_multiple_of=8  # Optional, for better hardware utilization
)

# LoRA config
peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05,  # Dropout for LoRA layers
    bias="none",  # No bias for LoRA
    task_type="CAUSAL_LM",  # Task type
    target_modules=["q_proj", "v_proj"]  # Manually specify target modules
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Hyperparameters optimized for maximum speed
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=2,  # Reduce batch size to utilize GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Simulate larger batch size
    optim="paged_adamw_32bit",  # Optimizer for better memory management
    num_train_epochs=1,  # Number of epochs
    eval_strategy="no",  # Disable evaluation during training
    logging_steps=1,  # Log every step
    warmup_steps=10,  # Warmup steps for learning rate scheduler
    logging_strategy="steps",
    learning_rate=5e-4,  # Increase learning rate for faster convergence
    bf16=True,  # Use bfloat16 for faster training on T4 GPUs
    group_by_length=True,  # Group sequences by length for efficiency
    report_to="wandb",  # Log to WandB
    dataloader_num_workers=1,  # Use only 1 worker to reduce CPU/RAM usage
    fp16=False,  # Disable fp16 if using bf16
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Use the preprocessed dataset
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=data_collator,  # Use the data collator for padding/truncation
)

# Start training
trainer.train()

# Finish WandB run
wandb.finish()