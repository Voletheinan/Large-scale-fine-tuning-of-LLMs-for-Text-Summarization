# -*- coding: utf-8 -*-
"""
config.py

This file defines common configurations and hyperparameters for the Text Summarization project.
It centralizes paths, model names, and training parameters to ensure consistency across scripts.
"""

import os
import torch
from peft import TaskType

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
TINYLLAMA_BASE_MODEL_DIR = os.path.join(MODELS_DIR, "tinyllama_base")
LORA_FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "finetuned_lora")
QLORA_FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "finetuned_qlora")
ADAPTER_FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "finetuned_adapter")
P_TUNING_FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "finetuned_prompt_tuning")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

REPORT_DIR = os.path.join(BASE_DIR, "report")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
TABLES_DIR = os.path.join(REPORT_DIR, "tables")

# Ensure all directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TINYLLAMA_BASE_MODEL_DIR, LORA_FINETUNED_MODEL_DIR,
             QLORA_FINETUNED_MODEL_DIR, ADAPTER_FINETUNED_MODEL_DIR, P_TUNING_FINETUNED_MODEL_DIR,
             CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(path, exist_ok=True)

# --- Model Configurations ---
# Using TinyLLaMA for better compatibility with personal machines/Colab
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Or other suitable TinyLLaMA model
MAX_SEQ_LENGTH = 512 # Maximum sequence length for tokenizer (input)
MAX_OUTPUT_LENGTH = 128 # Maximum output sequence length for summaries

# --- Dataset Size Limits (for faster testing) ---
# Set to None to use full dataset, or set a number to limit
# Giảm kích thước dataset để test nhanh
MAX_TRAIN_SAMPLES = 1000  # Giảm xuống 1000 mẫu để test nhanh
MAX_VAL_SAMPLES = 200     # 20% của train
MAX_TEST_SAMPLES = 200    # 20% của train
# Examples for reference (uncomment or change above to use):
#   - 20000: Good balance (~2-3 hours for 4 models)
#   - 50000: Large dataset (~5-6 hours for 4 models)
#   - None: Full dataset (~287k examples, ~3-5 hours training)

# --- Training Hyperparameters ---
TRAINING_ARGS = {
    "num_train_epochs": 1,          # Giảm epochs cho chạy nhanh
    "per_device_train_batch_size": 8,   # Giảm batch size để phù hợp với GPU thông thường
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4, # Tăng gradient accumulation để bù cho batch size nhỏ
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "logging_strategy": "steps",
    "logging_steps": 20,
    "learning_rate": 2e-4,
    "bf16": False,                   # Tắt bf16 để tương thích tốt hơn
    "fp16": True,                    # Bật fp16 thay thế
    "optim": "adamw_torch",          # Sử dụng optimizer thông thường
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,
    "group_by_length": True,
    "length_column_name": "length",   # Thêm column length để group hiệu quả hơn
    "dataloader_num_workers": 0,      # Tắt multiprocessing trên Windows
    "dataloader_pin_memory": False,   # Tắt pin memory để tránh lỗi CUDA
    "torch_compile": False,           # Tắt torch.compile() vì không tương thích với 4-bit quantization
    "seed": 42,
    "report_to": "tensorboard",
    "remove_unused_columns": False,
}

# --- PEFT Specific Configurations ---
LORA_CONFIG = {
    "r": 8,                      # Giảm rank để tiết kiệm VRAM
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

IA3_CONFIG = {
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "feedforward_modules": ["gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}

PROMPT_TUNING_CONFIG = {
    "task_type": TaskType.CAUSAL_LM,
    "num_virtual_tokens": 20,
    "tokenizer_name_or_path": MODEL_NAME,
}

# --- Quantization Configuration for QLoRA/PEFT methods ---
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,    # Sử dụng bfloat16 cho A100
    "bnb_4bit_use_double_quant": True,           # Bật double quantization
}

# --- Evaluation Metrics ---
ROUGE_METRICS = ['rouge1', 'rouge2', 'rougel']

# --- Report Configuration ---
REPORT_TITLE = "Large-scale Fine-tuning of LLMs for Text Summarization"
REPORT_AUTHOR = "[Your Name/Team]"

# --- TinyLLaMA specific tokenizer settings ---
# Some tokenizers (like TinyLlama) might not have a pad_token, add it if missing.
ADD_PAD_TOKEN = True
PAD_TOKEN = "<pad>"
