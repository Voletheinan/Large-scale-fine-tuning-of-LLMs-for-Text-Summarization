# -*- coding: utf-8 -*-
"""
dataset_utils.py

This script provides utilities for loading the tokenizer, preprocessing the dataset
(tokenization and formatting for TinyLLaMA), and creating PyTorch DataLoaders.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Dict, List

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import MODEL_NAME, MAX_SEQ_LENGTH, MAX_OUTPUT_LENGTH, ADD_PAD_TOKEN, PAD_TOKEN, TINYLLAMA_BASE_MODEL_DIR

def load_tokenizer():
    """
    Loads the tokenizer for the specified model and adds a pad token if required.
    """
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if ADD_PAD_TOKEN and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        print(f"Added pad token: {PAD_TOKEN} to tokenizer.")
    
    # Save tokenizer to base model directory for consistent loading
    tokenizer.save_pretrained(TINYLLAMA_BASE_MODEL_DIR)
    print("Tokenizer loaded and saved successfully.")
    return tokenizer

def preprocess_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    """
    Preprocesses the dataset by formatting examples for TinyLLaMA's chat template
    and then tokenizing them. Limits output to MAX_OUTPUT_LENGTH tokens.
    """
    inputs = []
    targets = []
    
    for article, summary in zip(examples['article'], examples['summary']):
        # Truncate article if needed (leaving room for prompt and summary)
        article_tokens = tokenizer(article, add_special_tokens=False, return_length=True)['length'][0]
        max_article_tokens = MAX_SEQ_LENGTH - MAX_OUTPUT_LENGTH - 50  # Reserve space for prompt and summary
        if article_tokens > max_article_tokens:
            article = tokenizer.decode(
                tokenizer(article, max_length=max_article_tokens, truncation=True, add_special_tokens=False)['input_ids'],
                skip_special_tokens=True
            )
        
        # Truncate summary to MAX_OUTPUT_LENGTH
        summary_tokens = tokenizer(summary, add_special_tokens=False, return_length=True)['length'][0]
        if summary_tokens > MAX_OUTPUT_LENGTH:
            summary = tokenizer.decode(
                tokenizer(summary, max_length=MAX_OUTPUT_LENGTH, truncation=True, add_special_tokens=False)['input_ids'],
                skip_special_tokens=True
            )
        
        instruction = f"Summarize the following article: {article}"
        chat = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": summary},
        ]
        # The full prompt template for TinyLlama-Chat
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
        targets.append(summary)

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
    
    # Tokenize targets separately for labels (only the summary part)
    target_inputs = tokenizer(targets, max_length=MAX_OUTPUT_LENGTH, truncation=True, padding="max_length")
    
    # Set labels (for causal LM, labels are the same as input_ids)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def get_data_collator(tokenizer: AutoTokenizer):
    """
    Returns a DataCollatorForLanguageModeling for dynamic padding.
    """
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def tokenize_datasets(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, tokenizer: AutoTokenizer):
    """
    Tokenizes the train, validation, and test datasets.
    """
    print("Tokenizing and preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=val_dataset.column_names)
    
    # For the test dataset, we only need to tokenize the input without the summary, for generation
    def preprocess_function_inference(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
        inputs = []
        for article in examples['article']:
            instruction = f"Summarize the following article: {article}"
            chat = [
                {"role": "user", "content": instruction},
            ]
            inputs.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

        model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
        return model_inputs

    tokenized_test_dataset = test_dataset.map(lambda x: preprocess_function_inference(x, tokenizer), batched=True, remove_columns=test_dataset.column_names)

    print("Datasets tokenized and preprocessed successfully.")
    
    return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset
