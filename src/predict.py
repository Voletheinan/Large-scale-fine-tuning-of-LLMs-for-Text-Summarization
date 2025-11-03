# -*- coding: utf-8 -*-
"""
predict.py

This script is used to generate text summaries from a specified fine-tuned TinyLLaMA model.
It can take a single article as input or generate summaries for the entire test dataset.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, IA3Config, PromptTuningConfig, TaskType
from datasets import Dataset
import pandas as pd
import argparse

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    MODEL_NAME,
    TINYLLAMA_BASE_MODEL_DIR,
    LORA_FINETUNED_MODEL_DIR,
    QLORA_FINETUNED_MODEL_DIR,
    ADAPTER_FINETUNED_MODEL_DIR,
    P_TUNING_FINETUNED_MODEL_DIR,
    PROCESSED_DATA_DIR,
    MAX_SEQ_LENGTH,
    MAX_OUTPUT_LENGTH,
    BNB_CONFIG
)
from src.dataset_utils import load_tokenizer
from src.prepare_data import load_processed_data

def load_model_for_prediction(model_type: str):
    """
    Loads the specified fine-tuned model (or PEFT adapter on base model) for prediction.
    """
    tokenizer = load_tokenizer()

    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    # PEFT methods (LoRA, QLoRA, Adapter, Prompt-tuning)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    
    if model_type == "lora":
        model_path = LORA_FINETUNED_MODEL_DIR
        print(f"Loading LoRA adapter from {model_path} on base model...")
    elif model_type == "qlora":
        model_path = QLORA_FINETUNED_MODEL_DIR
        print(f"Loading QLoRA adapter from {model_path} on base model...")
    elif model_type == "adapter":
        model_path = ADAPTER_FINETUNED_MODEL_DIR
        print(f"Loading Adapter (IA3) from {model_path} on base model...")
    elif model_type == "prompt_tuning":
        model_path = P_TUNING_FINETUNED_MODEL_DIR
        print(f"Loading Prompt-tuning adapter from {model_path} on base model...")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from lora, qlora, adapter, prompt_tuning.")

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer

def generate_summary(model, tokenizer, article: str) -> str:
    """
    Generates a summary for a given article using the loaded model.
    """
    instruction = f"Summarize the following article: {article}"
    chat = [
        {"role": "user", "content": instruction},
    ]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode generated summary, skipping the input part
    summary = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate text summaries using fine-tuned TinyLLaMA models.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["lora", "qlora", "adapter", "prompt_tuning"],
                        help="Type of fine-tuned model to use for prediction.")
    parser.add_argument("--article_text", type=str,
                        help="Input article text to summarize. If not provided, a sample from test set will be used.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index of the sample from the test dataset to summarize, if article_text is not provided.")

    args = parser.parse_args()

    model, tokenizer = load_model_for_prediction(args.model_type)

    if args.article_text:
        article = args.article_text
        print(f"\nInput Article: {article[:200]}...") # Print truncated for brevity
        reference_summary = None
    else:
        test_dataset = load_processed_data("test")
        if args.sample_index >= len(test_dataset):
            raise IndexError(f"Sample index {args.sample_index} out of bounds for test dataset of size {len(test_dataset)}.")
        article = test_dataset[args.sample_index]['article']
        reference_summary = test_dataset[args.sample_index]['summary']
        print(f"\n--- Generating summary for sample index {args.sample_index} from test dataset ---")
        print(f"Input Article: {article[:200]}...")
        print(f"Reference Summary: {reference_summary}")

    generated_summary = generate_summary(model, tokenizer, article)

    print(f"\nGenerated Summary ({args.model_type.upper()}): {generated_summary}")

if __name__ == "__main__":
    main()


