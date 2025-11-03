# -*- coding: utf-8 -*-
"""
train_lora.py

This script performs LoRA (Low-Rank Adaptation) fine-tuning of the TinyLLaMA model
for text summarization. It loads the 4-bit quantized base model and applies LoRA adapters.
Logs VRAM usage, training time, and trainable parameters.

(ĐÃ SỬA LỖI CẢNH BÁO use_reentrant VÀ torch_dtype)
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    MODEL_NAME,
    TINYLLAMA_BASE_MODEL_DIR,
    LORA_FINETUNED_MODEL_DIR,
    PROCESSED_DATA_DIR,
    TRAINING_ARGS,
    LORA_CONFIG,
    BNB_CONFIG,
    TABLES_DIR,
)
from src.dataset_utils import load_tokenizer, tokenize_datasets
from src.prepare_data import load_processed_data

def get_gpu_memory_gb():
    """Returns GPU memory usage in GB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0

def get_trainable_parameters_info(model):
    """Extracts trainable parameters info from model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": trainable_percentage,
        "trainable_params_str": f"{trainable_params:,} ({trainable_percentage:.2f}%)"
    }

def main():
    print("Starting LoRA Fine-tuning process...")
    start_time = time.time()
    
    # Track metrics
    metrics = {
        "method": "LoRA",
        "training_start_time": start_time,
    }

    # 1. Load tokenizer
    tokenizer = load_tokenizer()

    # 2. Load processed data
    print("Loading processed datasets...")
    train_dataset = load_processed_data("train")
    val_dataset = load_processed_data("validation")
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    # 3. Tokenize datasets
    tokenized_train_dataset, tokenized_val_dataset, _ = tokenize_datasets(
        train_dataset, val_dataset, 
        Dataset.from_dict({"article": [], "summary": []}), 
        tokenizer
    )

    # 4. Load base TinyLLaMA model with 4-bit quantization
    print(f"Loading base model {MODEL_NAME} with 4-bit quantization for LoRA...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16  # <-- SỬA Ở ĐÂY: Sửa 'torch_dtype' thành 'dtype'
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("Base model loaded and quantized successfully.")
    
    # Log initial VRAM
    initial_vram = get_gpu_memory_gb()
    print(f"Initial VRAM usage: {initial_vram:.2f} GB")

    # 5. Prepare model for k-bit training and apply LoRA config
    # Enable gradient checkpointing with use_reentrant=False for PyTorch 2.x compatibility
    
    # <-- SỬA Ở ĐÂY: Thêm 'gradient_checkpointing_kwargs' để tắt cảnh báo 'use_reentrant'
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    print("LoRA model prepared successfully.")
    
    # Get trainable parameters info
    params_info = get_trainable_parameters_info(model)
    metrics.update(params_info)
    model.print_trainable_parameters()
    
    # Log VRAM after model preparation
    model_vram = get_gpu_memory_gb()
    print(f"VRAM usage after model preparation: {model_vram:.2f} GB")
    metrics["gpu_memory_gb"] = model_vram

    # 6. Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=LORA_FINETUNED_MODEL_DIR,
        **TRAINING_ARGS,
        logging_dir=os.path.join(LORA_FINETUNED_MODEL_DIR, "logs"),
    )

    # 7. Initialize and run the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )

    print("Starting LoRA Fine-tuning...")
    training_start = time.time()
    trainer.train()
    training_end = time.time()
    
    training_time = training_end - training_start
    metrics["training_time_s"] = training_time
    metrics["training_time_min"] = training_time / 60
    print(f"LoRA Fine-tuning complete. Training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")

    # Log final VRAM
    final_vram = get_gpu_memory_gb()
    metrics["peak_gpu_memory_gb"] = final_vram
    print(f"Peak VRAM usage: {final_vram:.2f} GB")

    # 8. Save the LoRA adapter
    print(f"Saving LoRA adapter to {LORA_FINETUNED_MODEL_DIR}...")
    trainer.model.save_pretrained(LORA_FINETUNED_MODEL_DIR)
    print("LoRA adapter saved successfully.")

    # 9. Save training metrics
    total_time = time.time() - start_time
    metrics["total_time_s"] = total_time
    metrics["training_end_time"] = training_end
    
    metrics_path = os.path.join(LORA_FINETUNED_MODEL_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    print(f"\n=== LoRA Training Summary ===")
    print(f"Trainable Parameters: {metrics['trainable_params_str']}")
    print(f"Training Time: {metrics['training_time_min']:.2f} minutes")
    print(f"Peak VRAM Usage: {metrics['peak_gpu_memory_gb']:.2f} GB")
    print(f"Total Time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()