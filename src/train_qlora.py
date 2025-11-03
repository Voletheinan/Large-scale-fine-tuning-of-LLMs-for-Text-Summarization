# -*- coding: utf-8 -*-
"""
train_qlora.py

This script performs QLoRA (Quantized Low-Rank Adaptation) fine-tuning of the TinyLLaMA model
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
    QLORA_FINETUNED_MODEL_DIR,
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

class TrainingMonitor:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        self.step_times = []
        
    def start(self):
        self.start_time = time.time()
        
    def update(self, current_step):
        if len(self.step_times) == 0:
            self.step_times.append(time.time() - self.start_time)
        else:
            self.step_times.append(time.time() - (self.start_time + sum(self.step_times)))
        
        # Tính thời gian trung bình mỗi step
        avg_step_time = sum(self.step_times[-50:]) / min(len(self.step_times), 50)
        
        # Tính thời gian còn lại
        steps_remaining = self.total_steps - current_step
        time_remaining = steps_remaining * avg_step_time
        
        return {
            'avg_step_time': avg_step_time,
            'time_remaining': f"{int(time_remaining//3600):02d}:{int((time_remaining%3600)//60):02d}:{int(time_remaining%60):02d}",
            'steps_per_second': 1 / avg_step_time,
            'progress': f"{current_step}/{self.total_steps} ({(current_step/self.total_steps*100):.1f}%)"
        }

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
    print("Starting QLoRA Fine-tuning process...")
    start_time = time.time()
    
    # Track metrics
    metrics = {
        "method": "QLoRA",
        "training_start_time": start_time,
    }

    # 1. Load tokenizer
    tokenizer = load_tokenizer()

    # 2. Load and trim processed data
    print("Loading processed datasets...")
    train_dataset = load_processed_data("train")
    val_dataset = load_processed_data("validation")
    
    from src.config import MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES
    if MAX_TRAIN_SAMPLES is not None:
        train_dataset = train_dataset.select(range(min(MAX_TRAIN_SAMPLES, len(train_dataset))))
    if MAX_VAL_SAMPLES is not None:
        val_dataset = val_dataset.select(range(min(MAX_VAL_SAMPLES, len(val_dataset))))
    
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")
    print(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # 3. Tokenize datasets với length column để group_by_length
    def tokenize_and_add_length(examples):
        tokenized = tokenizer(
            examples["article"],
            examples["summary"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        # Thêm length column để group_by_length hoạt động hiệu quả
        tokenized["length"] = [len(x) for x in tokenized["input_ids"]]
        return tokenized

    # Tokenize với num_proc để xử lý song song
    tokenized_train_dataset = train_dataset.map(
        tokenize_and_add_length,
        batched=True,
        num_proc=4,  # Số process cho song song
        remove_columns=train_dataset.column_names
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_and_add_length,
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names
    )

    # 4. Load base TinyLLaMA model with 4-bit quantization
    print(f"Loading base model {MODEL_NAME} with 4-bit quantization for QLoRA...")
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
    
    # For QLoRA, we still use LoraConfig, but the base model is already 4-bit quantized.
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Compile model để tăng tốc
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    
    print("QLoRA model prepared successfully.")
    
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
        output_dir=QLORA_FINETUNED_MODEL_DIR,
        **TRAINING_ARGS,
        logging_dir=os.path.join(QLORA_FINETUNED_MODEL_DIR, "logs"),
    )

    # 7. Initialize trainer with time monitoring
    # Tính tổng số steps
    total_steps = int(len(tokenized_train_dataset) / training_args.per_device_train_batch_size / 
                     training_args.gradient_accumulation_steps * training_args.num_train_epochs)
    
    # Khởi tạo training monitor
    monitor = TrainingMonitor(total_steps)
    
    # Định nghĩa callback để theo dõi thời gian
    class TimeMonitorCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            monitor.start()
            print(f"\nTổng số steps: {total_steps}")
            print(f"Ước tính thời gian training (dựa trên 1.73s/step): {str(timedelta(seconds=int(total_steps * 1.73)))}\n")
        
        def on_step_end(self, args, state, control, **kwargs):
            stats = monitor.update(state.global_step)
            print(f"\rTiến độ: {stats['progress']} | "
                  f"Thời gian còn lại: {stats['time_remaining']} | "
                  f"Tốc độ: {stats['steps_per_second']:.2f} steps/s", end="")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[TimeMonitorCallback],
    )

    print("Starting QLoRA Fine-tuning...")
    training_start = time.time()
    trainer.train()
    training_end = time.time()
    
    training_time = training_end - training_start
    metrics["training_time_s"] = training_time
    metrics["training_time_min"] = training_time / 60
    print(f"QLoRA Fine-tuning complete. Training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")

    # Log final VRAM
    final_vram = get_gpu_memory_gb()
    metrics["peak_gpu_memory_gb"] = final_vram
    print(f"Peak VRAM usage: {final_vram:.2f} GB")

    # 8. Save the QLoRA adapter
    print(f"Saving QLoRA adapter to {QLORA_FINETUNED_MODEL_DIR}...")
    trainer.model.save_pretrained(QLORA_FINETUNED_MODEL_DIR)
    print("QLoRA adapter saved successfully.")

    # 9. Save training metrics
    total_time = time.time() - start_time
    metrics["total_time_s"] = total_time
    metrics["training_end_time"] = training_end
    
    metrics_path = os.path.join(QLORA_FINETUNED_MODEL_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    print(f"\n=== QLoRA Training Summary ===")
    print(f"Trainable Parameters: {metrics['trainable_params_str']}")
    print(f"Training Time: {metrics['training_time_min']:.2f} minutes")
    print(f"Peak VRAM Usage: {metrics['peak_gpu_memory_gb']:.2f} GB")
    print(f"Total Time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()