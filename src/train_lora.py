# -*- coding: utf-8 -*-
"""
train_lora.py

This script performs LoRA (Low-Rank Adaptation) fine-tuning of the TinyLLaMA model
for text summarization. It loads the 4-bit quantized base model and applies LoRA adapters.
Logs VRAM usage, training time, and trainable parameters.

(ĐÃ FIX LỖI: model không trả về loss do thiếu labels)
"""

import os
import sys
import json
import time
from datetime import timedelta
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
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

        avg_step_time = sum(self.step_times[-50:]) / min(len(self.step_times), 50)
        steps_remaining = self.total_steps - current_step
        time_remaining = steps_remaining * avg_step_time

        return {
            'avg_step_time': avg_step_time,
            'time_remaining': f"{int(time_remaining//3600):02d}:{int((time_remaining%3600)//60):02d}:{int(time_remaining%60):02d}",
            'steps_per_second': 1 / avg_step_time,
            'progress': f"{current_step}/{self.total_steps} ({(current_step/self.total_steps*100):.1f}%)"
        }


def get_trainable_parameters_info(model):
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

    metrics = {
        "method": "LoRA",
        "training_start_time": start_time,
    }

    # 1. Load tokenizer
    tokenizer = load_tokenizer()

    # 2. Load processed datasets
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

    # 3. Tokenization — FIXED: now includes 'labels'
    def tokenize_and_add_length(examples):
        # Create structured input-output for summarization
        inputs = [f"Summarize the following article:\n{a}" for a in examples["article"]]
        outputs = [f"{s}" for s in examples["summary"]]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )

        labels = tokenizer(
            outputs,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )["input_ids"]

        model_inputs["labels"] = labels
        model_inputs["length"] = [len(x) for x in model_inputs["input_ids"]]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(
        tokenize_and_add_length,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_and_add_length,
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names
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
        dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("Base model loaded and quantized successfully.")

    initial_vram = get_gpu_memory_gb()
    print(f"Initial VRAM usage: {initial_vram:.2f} GB")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    print("LoRA model prepared successfully.")

    params_info = get_trainable_parameters_info(model)
    metrics.update(params_info)
    model.print_trainable_parameters()

    model_vram = get_gpu_memory_gb()
    print(f"VRAM usage after model preparation: {model_vram:.2f} GB")
    metrics["gpu_memory_gb"] = model_vram

    training_args = TrainingArguments(
        output_dir=LORA_FINETUNED_MODEL_DIR,
        **TRAINING_ARGS,
        logging_dir=os.path.join(LORA_FINETUNED_MODEL_DIR, "logs"),
    )

    total_steps = int(len(tokenized_train_dataset) / training_args.per_device_train_batch_size /
                     training_args.gradient_accumulation_steps * training_args.num_train_epochs)

    monitor = TrainingMonitor(total_steps)

    class TimeMonitorCallback(TrainerCallback):
        def __init__(self, monitor):
            self.monitor = monitor

        def on_train_begin(self, args, state, control, **kwargs):
            self.monitor.start()
            print(f"\nTổng số steps: {total_steps}")
            print(f"Ước tính thời gian training (dựa trên 1.73s/step): {str(timedelta(seconds=int(total_steps * 1.73)))}\n")

        def on_step_end(self, args, state, control, **kwargs):
            stats = self.monitor.update(state.global_step)
            print(f"\rTiến độ: {stats['progress']} | "
                  f"Thời gian còn lại: {stats['time_remaining']} | "
                  f"Tốc độ: {stats['steps_per_second']:.2f} steps/s", end="")

    time_monitor_callback = TimeMonitorCallback(monitor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[time_monitor_callback],
    )

    print("Starting LoRA Fine-tuning...")
    training_start = time.time()
    trainer.train()
    training_end = time.time()

    training_time = training_end - training_start
    metrics["training_time_s"] = training_time
    metrics["training_time_min"] = training_time / 60
    print(f"LoRA Fine-tuning complete. Training time: {training_time:.2f}s ({training_time/60:.2f} minutes)")

    final_vram = get_gpu_memory_gb()
    metrics["peak_gpu_memory_gb"] = final_vram
    print(f"Peak VRAM usage: {final_vram:.2f} GB")

    print(f"Saving LoRA adapter to {LORA_FINETUNED_MODEL_DIR}...")
    trainer.model.save_pretrained(LORA_FINETUNED_MODEL_DIR)
    print("LoRA adapter saved successfully.")

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
