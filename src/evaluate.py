# -*- coding: utf-8 -*-
"""
evaluate.py

This script evaluates the fine-tuned TinyLLaMA models for text summarization
using ROUGE and BLEU metrics. It evaluates both the base model (before fine-tuning)
and all fine-tuned models, comparing their performance.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
import json
import time

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
    REPORT_DIR,
    TABLES_DIR,
    ROUGE_METRICS,
    MAX_SEQ_LENGTH,
    MAX_OUTPUT_LENGTH,
    BNB_CONFIG
)
from src.dataset_utils import load_tokenizer
from src.prepare_data import load_processed_data

# Download NLTK data if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
except:
    pass

def calculate_bleu(reference: str, prediction: str) -> float:
    """Calculates BLEU score between reference and prediction."""
    try:
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        # Use smoothing to handle cases where there are no matching n-grams
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

def preprocess_for_inference(tokenizer, examples):
    """Preprocesses examples for inference."""
    inputs = []
    for article in examples['article']:
        instruction = f"Summarize the following article: {article}"
        chat = [
            {"role": "user", "content": instruction},
        ]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_SEQ_LENGTH, 
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    return model_inputs

def evaluate_model(model, tokenizer, test_dataset, references, model_tag="Unknown", limit=None):
    """
    Evaluates a model on the test dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_dataset: Tokenized test dataset
        references: List of reference summaries
        model_tag: Tag for identifying the model
        limit: Limit number of examples to evaluate (None for all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    inference_times = []
    
    print(f"Starting evaluation for {model_tag}...")
    model.eval()
    
    num_examples = len(test_dataset) if limit is None else min(limit, len(test_dataset))
    print(f"Evaluating {num_examples} examples...")
    
    with torch.no_grad():
        for i in range(num_examples):
            example = test_dataset[i]
            
            # Handle both dict and Dataset formats
            if isinstance(example, dict):
                if isinstance(example['input_ids'], torch.Tensor):
                    input_ids = example['input_ids'].to(model.device)
                    attention_mask = example['attention_mask'].to(model.device) if 'attention_mask' in example else None
                else:
                    input_ids = torch.tensor([example['input_ids']]).to(model.device)
                    attention_mask = torch.tensor([example['attention_mask']]).to(model.device) if 'attention_mask' in example else None
            else:
                input_ids = torch.tensor([example['input_ids']]).to(model.device)
                attention_mask = torch.tensor([example['attention_mask']]).to(model.device) if hasattr(example, 'attention_mask') else None
            
            # Ensure 2D tensors: [batch_size, seq_length]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
            
            start_time = time.time()
            generate_kwargs = {
                'input_ids': input_ids,
                'max_new_tokens': MAX_OUTPUT_LENGTH,
                'num_beams': 4,
                'early_stopping': True,
                'do_sample': False,
                'pad_token_id': tokenizer.pad_token_id,
            }
            if attention_mask is not None:
                generate_kwargs['attention_mask'] = attention_mask
                
            generated_ids = model.generate(**generate_kwargs)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Decode generated summary (skip input tokens)
            input_length = input_ids.shape[-1]
            generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
            predictions.append(generated_text)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{num_examples} examples for {model_tag}")

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)
    rouge_scores_list = {'rouge1': [], 'rouge2': [], 'rougel': []}
    bleu_scores = []
    
    for pred, ref in zip(predictions, references[:num_examples]):
        # ROUGE scores
        scores = scorer.score(ref, pred)
        rouge_scores_list['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores_list['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores_list['rougel'].append(scores['rougel'].fmeasure)
        
        # BLEU score
        bleu = calculate_bleu(ref, pred)
        bleu_scores.append(bleu)
    
    avg_rouge_scores = {
        'rouge1': sum(rouge_scores_list['rouge1']) / len(rouge_scores_list['rouge1']),
        'rouge2': sum(rouge_scores_list['rouge2']) / len(rouge_scores_list['rouge2']),
        'rougel': sum(rouge_scores_list['rougel']) / len(rouge_scores_list['rougel']),
    }
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

    print(f"Evaluation for {model_tag} complete.")
    print(f"  ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_rouge_scores['rougel']:.4f}")
    print(f"  BLEU: {avg_bleu:.4f}")
    print(f"  Avg Inference Time: {avg_inference_time:.4f}s")
    
    return {
        **avg_rouge_scores,
        'bleu': avg_bleu,
        'avg_inference_time_s': avg_inference_time,
        'predictions': predictions[:10],  # Store first 10 for sample analysis
        'references': references[:10],
    }

def load_training_metrics(model_dir: str) -> dict:
    """Loads training metrics from JSON file."""
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    print("Starting evaluation process...")
    print("This will evaluate the base model and all fine-tuned models.")
    
    # Limit evaluation for faster testing (set to None for full evaluation)
    EVAL_LIMIT = None  # Change to e.g., 500 for quick testing
    
    # 1. Load tokenizer
    tokenizer = load_tokenizer()

    # 2. Load processed test data
    print("Loading processed test dataset...")
    test_dataset_raw = load_processed_data("test")
    
    # Get references (ground truth summaries)
    references = [ex['summary'] for ex in test_dataset_raw]
    print(f"Loaded {len(references)} test examples")
    
    # Tokenize test dataset for inference
    print("Tokenizing test dataset for inference...")
    def preprocess_single_example(example):
        """Preprocess a single example for inference."""
        instruction = f"Summarize the following article: {example['article']}"
        chat = [
            {"role": "user", "content": instruction},
        ]
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # Tokenize with proper format
        encoded = tokenizer(
            input_text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        # Convert to lists (Datasets format doesn't support tensors directly)
        return {
            'input_ids': encoded['input_ids'].squeeze(0).tolist(),
            'attention_mask': encoded['attention_mask'].squeeze(0).tolist()
        }
    
    test_dataset_tokenized = test_dataset_raw.map(
        preprocess_single_example,
        batched=False,
        remove_columns=test_dataset_raw.column_names
    )
    
    # Common BitsAndBytesConfig for loading models
    bnb_config_eval = BitsAndBytesConfig(**BNB_CONFIG)
    
    # Function to load base model
    def load_base_model():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config_eval,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    # Function to load PEFT model
    def load_peft_model(model_dir: str):
        base_model = load_base_model()
        return PeftModel.from_pretrained(base_model, model_dir)

    all_results = {}
    
    # --- Evaluate Base Model (Before Fine-tuning) ---
    print("\n" + "="*60)
    print("Evaluating BASE MODEL (before fine-tuning)")
    print("="*60)
    try:
        base_model = load_base_model()
        base_results = evaluate_model(
            base_model, tokenizer, test_dataset_tokenized, references, 
            "Base Model", limit=EVAL_LIMIT
        )
        base_results['method'] = 'Base Model'
        all_results['Base Model'] = base_results
        del base_model  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        all_results['Base Model'] = {"error": str(e)}
    
    # --- Evaluate Fine-tuned Models ---
    models_to_evaluate = [
        ("LoRA", LORA_FINETUNED_MODEL_DIR),
        ("QLoRA", QLORA_FINETUNED_MODEL_DIR),
        ("Adapter (IA3)", ADAPTER_FINETUNED_MODEL_DIR),
        ("Prompt-tuning", P_TUNING_FINETUNED_MODEL_DIR),
    ]

    for model_tag, model_dir in models_to_evaluate:
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"\nWarning: Model directory {model_dir} does not exist or is empty. Skipping {model_tag}.")
            continue
            
        print("\n" + "="*60)
        print(f"Evaluating {model_tag}")
        print("="*60)
        
        try:
            # Load training metrics
            training_metrics = load_training_metrics(model_dir)
            
            # Load and evaluate model
            model = load_peft_model(model_dir)
            eval_results = evaluate_model(
                model, tokenizer, test_dataset_tokenized, references,
                model_tag, limit=EVAL_LIMIT
            )
            
            # Combine evaluation and training metrics
            all_results[model_tag] = {
                **eval_results,
                'method': model_tag,
                'trainable_params': training_metrics.get('trainable_params', None),
                'trainable_params_str': training_metrics.get('trainable_params_str', 'N/A'),
                'trainable_percentage': training_metrics.get('trainable_percentage', None),
                'training_time_s': training_metrics.get('training_time_s', None),
                'training_time_min': training_metrics.get('training_time_min', None),
                'gpu_memory_gb': training_metrics.get('peak_gpu_memory_gb', None),
            }
            
            del model  # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error evaluating {model_tag}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_tag] = {"error": str(e)}

    # --- Create Results DataFrame ---
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Prepare data for DataFrame
    results_data = []
    for method, results in all_results.items():
        if 'error' in results:
            continue
        row = {
            'Method': method,
            'ROUGE-1': results.get('rouge1', None),
            'ROUGE-2': results.get('rouge2', None),
            'ROUGE-L': results.get('rougel', None),
            'BLEU': results.get('bleu', None),
            'Trainable Params': results.get('trainable_params_str', 'N/A'),
            'Trainable Params (num)': results.get('trainable_params', None),
            'Training Time (s)': results.get('training_time_s', None),
            'Training Time (min)': results.get('training_time_min', None),
            'Inference Time (s)': results.get('avg_inference_time_s', None),
            'VRAM (GB)': results.get('gpu_memory_gb', None),
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Display results
    print("\nResults Table:")
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    output_results_path = os.path.join(TABLES_DIR, "evaluation_results.csv")
    results_df.to_csv(output_results_path, index=False)
    print(f"\nEvaluation results saved to {output_results_path}")
    
    # Save detailed results with predictions to JSON
    output_json_path = os.path.join(TABLES_DIR, "evaluation_results_detailed.json")
    # Remove predictions/references from JSON to save space (keep only first 10)
    detailed_results = {}
    for method, results in all_results.items():
        detailed_results[method] = {k: v for k, v in results.items() 
                                     if k not in ['predictions', 'references'] or method == 'Base Model'}
    with open(output_json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed evaluation results saved to {output_json_path}")

if __name__ == "__main__":
    main()
