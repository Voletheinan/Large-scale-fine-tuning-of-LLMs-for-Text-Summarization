# -*- coding: utf-8 -*-
"""
prepare_data.py

This script is responsible for loading the raw dataset (e.g., CNN/DailyMail CSVs),
performing initial preprocessing steps like text cleaning, limiting text length,
and then saving the processed data in a format suitable for tokenization.
"""

import os
import sys
import re
import json
import pandas as pd
from typing import Union
from datasets import load_dataset, Dataset
import argparse

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MAX_SEQ_LENGTH, MAX_OUTPUT_LENGTH,
    MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, MAX_TEST_SAMPLES
)

def clean_text(text: str) -> str:
    """
    Cleans text by removing HTML tags, special characters, and normalizing whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def limit_text_length(text: str, max_tokens: int, tokenizer=None) -> str:
    """
    Limits text length by approximating token count (roughly 4 chars per token).
    If tokenizer is provided, uses actual tokenization for accurate length.
    """
    if tokenizer is None:
        # Rough estimation: ~4 characters per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    else:
        tokens = tokenizer(text, add_special_tokens=False, return_length=True)
        if tokens['length'][0] > max_tokens:
            # Truncate to approximate length
            truncated = tokenizer.decode(tokenizer(text, max_length=max_tokens, truncation=True, add_special_tokens=False)['input_ids'], skip_special_tokens=True)
            return truncated
        return text

def preprocess_dataset(dataset: Dataset, split_name: str, tokenizer=None) -> Dataset:
    """
    Preprocesses a dataset by cleaning text and limiting length.
    """
    print(f"Preprocessing {split_name} dataset...")
    
    def clean_and_limit(examples):
        articles = [clean_text(art) for art in examples['article']]
        summaries = [clean_text(sum) for sum in examples['summary']]
        
        # Limit article length (input)
        articles = [limit_text_length(art, MAX_SEQ_LENGTH - 50, tokenizer) for art in articles]  # -50 for prompt tokens
        
        # Limit summary length (output)
        summaries = [limit_text_length(sum, MAX_OUTPUT_LENGTH, tokenizer) for sum in summaries]
        
        return {'article': articles, 'summary': summaries}
    
    processed_dataset = dataset.map(clean_and_limit, batched=True)
    print(f"Processed {split_name}: {len(processed_dataset)} examples")
    
    return processed_dataset

def load_raw_data() -> dict[str, Dataset]:
    """
    Loads the raw dataset (CNN/DailyMail) directly from Hugging Face.
    Renames the 'highlights' column to 'summary' for consistency.
    Returns a dictionary of datasets for 'train', 'validation', and 'test' splits.
    """
    print("Loading dataset 'abisee/cnn_dailymail' from Hugging Face...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    
    # Rename 'highlights' to 'summary' for consistency
    for split in dataset:
        if 'highlights' in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("highlights", "summary")
        # Ensure only 'article' and 'summary' columns are kept
        dataset[split] = dataset[split].select_columns(["article", "summary"])
    
    print("Dataset loaded. Splits available: train, validation, test.")
    return dataset

def save_processed_data(datasets: dict[str, Dataset]):
    """
    Saves processed datasets to data/processed/ directory.
    """
    print(f"Saving processed datasets to {PROCESSED_DATA_DIR}...")
    
    for split_name, dataset in datasets.items():
        # Convert to pandas for easier saving
        df = dataset.to_pandas()
        output_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {split_name} dataset: {len(df)} examples -> {output_path}")

def _get_or_create_sample_indices(split_name: str, full_size: int, limit: int) -> list:
    """
    Gets existing sample indices or creates new ones and saves them.
    Ensures all models use the same data samples.
    """
    indices_file = os.path.join(PROCESSED_DATA_DIR, f"sample_indices_{split_name}.json")
    
    # Try to load existing indices
    if os.path.exists(indices_file):
        try:
            with open(indices_file, 'r') as f:
                saved_data = json.load(f)
                saved_limit = saved_data.get('limit')
                saved_indices = saved_data.get('indices')
                
                # Check if saved indices match current limit
                if saved_limit == limit and len(saved_indices) == limit:
                    print(f"✅ Reusing saved indices for {split_name} ({limit:,} samples)")
                    return saved_indices
                else:
                    print(f"⚠️  Saved indices for {split_name} don't match current limit. Creating new ones...")
        except Exception as e:
            print(f"⚠️  Error loading indices file: {e}. Creating new ones...")
    
    # Create new indices
    print(f"📝 Creating new sample indices for {split_name} ({limit:,} samples)...")
    import numpy as np
    np.random.seed(42)  # Fixed seed for reproducibility
    indices = sorted(np.random.choice(full_size, size=min(limit, full_size), replace=False).tolist())
    
    # Save indices for future use
    with open(indices_file, 'w') as f:
        json.dump({'limit': limit, 'indices': indices}, f, indent=2)
    
    print(f"💾 Saved sample indices to {indices_file}")
    return indices

def load_processed_data(split: str = None) -> Union[dict[str, Dataset], Dataset]:
    """
    Loads processed datasets from data/processed/ directory.
    Automatically limits dataset size based on MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, MAX_TEST_SAMPLES in config.
    Uses saved sample indices to ensure all models train on the same data.
    If split is specified, returns only that split.
    """
    splits = ['train', 'validation', 'test']
    
    # Get limits from config
    limits = {
        'train': MAX_TRAIN_SAMPLES,
        'validation': MAX_VAL_SAMPLES,
        'test': MAX_TEST_SAMPLES
    }
    
    if split:
        if split not in splits:
            raise ValueError(f"Split must be one of {splits}")
        csv_path = os.path.join(PROCESSED_DATA_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed data not found at {csv_path}. Please run prepare_data.py first.")
        df = pd.read_csv(csv_path)
        
        # Limit dataset if specified
        limit = limits.get(split)
        if limit and limit < len(df):
            # Get or create sample indices (ensures consistency across all models)
            indices = _get_or_create_sample_indices(split, len(df), limit)
            df = df.iloc[indices].reset_index(drop=True)
            print(f"✅ Loaded {split}: {len(df):,} examples (same data for all models)")
        else:
            print(f"✅ Loaded {split}: {len(df):,} examples")
        
        return Dataset.from_pandas(df)
    
    datasets = {}
    for split_name in splits:
        csv_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Limit dataset if specified
            limit = limits.get(split_name)
            if limit and limit < len(df):
                # Get or create sample indices (ensures consistency across all models)
                indices = _get_or_create_sample_indices(split_name, len(df), limit)
                df = df.iloc[indices].reset_index(drop=True)
                print(f"✅ Loaded {split_name}: {len(df):,} examples (same data for all models)")
            else:
                print(f"✅ Loaded {split_name}: {len(df):,} examples")
            
            datasets[split_name] = Dataset.from_pandas(df)
        else:
            print(f"Warning: {csv_path} not found. Skipping {split_name}.")
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument("--max_train", type=int, default=MAX_TRAIN_SAMPLES,
                       help="Maximum number of training samples (None for all)")
    parser.add_argument("--max_val", type=int, default=MAX_VAL_SAMPLES,
                       help="Maximum number of validation samples (None for all)")
    parser.add_argument("--max_test", type=int, default=MAX_TEST_SAMPLES,
                       help="Maximum number of test samples (None for all)")
    
    args = parser.parse_args()
    
    max_samples = {
        'train': args.max_train,
        'validation': args.max_val,
        'test': args.max_test
    }
    
    print("Starting data preparation...")
    if any(max_samples.values()):
        print("\n⚠️  Dataset size limits:")
        for split, limit in max_samples.items():
            if limit:
                print(f"  {split}: {limit:,} samples (limited)")
    
    # Load raw dataset from Hugging Face
    raw_datasets = load_raw_data()
    
    # Preprocess datasets (clean text, limit length)
    # Note: tokenizer will be loaded in dataset_utils, so we do rough length limiting here
    processed_datasets = {}
    for split_name in ['train', 'validation', 'test']:
        dataset = raw_datasets[split_name]
        
        # Limit dataset size if specified
        limit = max_samples.get(split_name)
        if limit and limit < len(dataset):
            print(f"\n⚠️  Limiting {split_name} dataset from {len(dataset):,} to {limit:,} samples...")
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        processed_datasets[split_name] = preprocess_dataset(dataset, split_name)
    
    # Save processed datasets
    save_processed_data(processed_datasets)
    
    # Display sample
    print("\nSample from processed train dataset:")
    sample = processed_datasets['train'][0]
    print(f"Article (first 200 chars): {sample['article'][:200]}...")
    print(f"Summary: {sample['summary']}")
    
    print("\n" + "="*60)
    print("Data preparation complete. Processed datasets saved to data/processed/")
    print("="*60)
    for split_name, dataset in processed_datasets.items():
        print(f"  {split_name}: {len(dataset):,} examples")

if __name__ == "__main__":
    main()


