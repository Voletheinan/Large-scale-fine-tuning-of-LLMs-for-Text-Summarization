# -*- coding: utf-8 -*-
"""
train_qlora.py — Fixed torch.compile() issue

This script performs QLoRA (Quantized Low-Rank Adaptation) fine-tuning
of the TinyLLaMA model for text summarization.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    QLORA_FINETUNED_MODEL_DIR,
    TRAINING_ARGS,
    LORA_CONFIG,
    BNB_CONFIG,
)
from src.dataset_utils import load_tokenizer
from src.prepare_data import load_processed_data

def get_gpu_memory_gb():
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

def main():
    print("Starting QLoRA Fine-tuning process...")
    start_time = time.time()

    tokenizer = load_tokenizer()
    print("Loading processed datasets...")
    train_dataset = load_processed_data("train")
    val_dataset = load_processed_data("validation")

    def tokenize_and_add_length(examples):
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

    print(f"Loading base model {MODEL_NAME} with 4-bit quantization for QLoRA...")
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

    print("Preparing model for 4-bit QLoRA training...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

    # ⚠️ Không compile model quantized (fix lỗi ValueError)
    print("Skipping torch.compile() because quantized models cannot be compiled.")

    print("QLoRA model prepared successfully.")

    training_args = TrainingArguments(
        output_dir=QLORA_FINETUNED_MODEL_DIR,
        **TRAINING_ARGS,
        logging_dir=os.path.join(QLORA_FINETUNED_MODEL_DIR, "logs"),
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[TimeMonitorCallback(monitor)],
    )

    print("Starting QLoRA Fine-tuning...")
    trainer.train()
    print("QLoRA Fine-tuning complete!")

if __name__ == "__main__":
    main()
