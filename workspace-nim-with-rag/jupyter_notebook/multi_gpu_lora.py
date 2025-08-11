
import os
import torch
import torch.distributed as dist
import datetime
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,SFTConfig
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Base model and data paths
base_model = "/mnt/lustre/docker-strg/llama3_8b_weights"
data_path = "data/ds_preprocess/train"
eval_path = "data/ds_preprocess/test"

import os
import torch
import torch.distributed as dist

def setup_distributed():
    """ Initialize distributed training properly by setting up the process group and device assignment. """
    
    # Get the local rank of the current process (assigned by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Get the total number of processes involved in training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:  # Ensure distributed training is enabled
        # Set the CUDA device for the current process
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group for multi-GPU communication
        dist.init_process_group(
            backend="nccl",      # Use NCCL backend for efficient GPU communication
            init_method="env://", # Use environment variables for initialization
            world_size=world_size, # Total number of processes
            rank=local_rank        # Rank of the current process
        )
    
    return local_rank  # Return local rank for further use

# Initialize distributed training
local_rank = setup_distributed()

# Load datasets from disk
dataset = load_from_disk(data_path)       # Load the training dataset
eval_dataset = load_from_disk(eval_path)  # Load the evaluation dataset

# Initialize the tokenizer using the base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Ensure padding is handled correctly
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS for compatibility
tokenizer.padding_side = "right"           # Right-padding ensures consistency for batch processing

# Quantization configuration for memory-efficient training
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization for reduced memory footprint
    bnb_4bit_quant_type="nf4",              # Use NormalFloat4 (NF4) quantization format
    bnb_4bit_compute_dtype=torch.float16,   # Perform computations in 16-bit precision
    bnb_4bit_use_double_quant=True,         # Use double quantization for better accuracy
)

# Training configuration for fine-tuning
training_config = SFTConfig(
    output_dir="model/training_results",    # Directory to save model checkpoints
    per_device_train_batch_size=16,          # Batch size per GPU for training
    per_device_eval_batch_size=8,           # Batch size per GPU for evaluation
    gradient_accumulation_steps=2,          # Accumulate gradients to simulate larger batch sizes
    learning_rate=2e-4,                      # Initial learning rate
    weight_decay=0.001,                      # Weight decay for regularization
    num_train_epochs=2,                      # Number of training epochs
    lr_scheduler_type="cosine",              # Use cosine learning rate schedule
    warmup_ratio=0.03,                       # Warmup ratio for learning rate scheduler
    log_level="warning",                      # Set logging level to reduce verbosity
    logging_steps=25,                         # Log training progress every 25 steps
    save_steps=25,                            # Save model checkpoints every 25 steps
    fp16=True,                                # Enable mixed-precision training for efficiency
    local_rank=local_rank,                    # Assign rank for distributed training
    ddp_backend="nccl",                       # Use NCCL backend for optimized GPU communication
    optim="adamw_torch_fused",                # Use fused AdamW optimizer for better performance
    gradient_checkpointing=True,              # Enable gradient checkpointing to save memory
    ddp_find_unused_parameters=False,         # Optimize DDP by avoiding unused parameter checks
    dataset_text_field="text",                # Define the dataset text field for tokenization
)

# LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
peft_params = LoraConfig(
    lora_alpha=16,                # Scaling factor for LoRA adaptation
    lora_dropout=0.1,             # Dropout rate for LoRA layers
    r=64,                         # Rank for the low-rank decomposition
    bias="none",                   # Disable bias adjustment
    task_type="CAUSAL_LM",         # Task type: Causal Language Modeling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target transformer layers for LoRA
)

if __name__ == "__main__":
    try:
        # Load the base model with LoRA and quantization settings
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,     # Use 16-bit precision for efficiency
            quantization_config=quant_config,  # Apply 4-bit quantization settings
        )
        model.config.use_cache = False  # Disable caching to support gradient checkpointing

        # Initialize the fine-tuning trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,  # Tokenizer for preprocessing
            train_dataset=dataset.select(range(1000)), 
            eval_dataset=eval_dataset,  # Evaluation dataset
            peft_config=peft_params,  # LoRA configuration
            args=training_config,  # Training hyperparameters
        )

        # Start fine-tuning
        trainer.train()

        # Save the fine-tuned model and tokenizer
        new_model = "model/Llama-3-8b-instruct-hf-finetune-multigpu"
        trainer.model.save_pretrained(new_model)
        trainer.tokenizer.save_pretrained(new_model)

        print("Training Complete..")
    
    except Exception as e:
        print(e)  # Print any errors that occur
        pass  # Continue execution without crashing

    # Clean up the distributed training process
    dist.destroy_process_group()
