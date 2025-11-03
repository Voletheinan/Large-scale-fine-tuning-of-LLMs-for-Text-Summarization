# -*- coding: utf-8 -*-
"""
visualize.py

This script provides functions for visualizing evaluation results and training logs
from the text summarization project. It generates various plots and saves them to the report/figures directory.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

# Add parent directory to path to allow imports from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import FIGURES_DIR, TABLES_DIR, ROUGE_METRICS, REPORT_TITLE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def plot_rouge_scores(results_df: pd.DataFrame, filename: str = "rouge_scores_bar_chart.png"):
    """
    Generates a bar chart comparing ROUGE scores across different fine-tuning methods.
    """
    if results_df.empty:
        print("No data to plot for ROUGE scores.")
        return

    print("Generating ROUGE Scores Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = results_df['Method'].values
    rouge1 = results_df['ROUGE-1'].values
    rouge2 = results_df['ROUGE-2'].values
    rougel = results_df['ROUGE-L'].values
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, rouge1, width, label='ROUGE-1', alpha=0.8)
    ax.bar(x, rouge2, width, label='ROUGE-2', alpha=0.8)
    ax.bar(x + width, rougel, width, label='ROUGE-L', alpha=0.8)
    
    ax.set_xlabel('Fine-tuning Method', fontsize=12)
    ax.set_ylabel('ROUGE Score (F1-measure)', fontsize=12)
    ax.set_title(f'Comparison of ROUGE Scores Across Fine-tuning Methods', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(rouge1.max(), rouge2.max(), rougel.max()) * 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"ROUGE Scores Bar Chart saved to {output_path}")
    plt.close(fig)

def plot_bleu_scores(results_df: pd.DataFrame, filename: str = "bleu_scores_bar_chart.png"):
    """
    Generates a bar chart comparing BLEU scores across different fine-tuning methods.
    """
    if results_df.empty or 'BLEU' not in results_df.columns:
        print("No BLEU data to plot.")
        return

    print("Generating BLEU Scores Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = results_df['Method'].values
    bleu_scores = results_df['BLEU'].values
    
    bars = ax.bar(methods, bleu_scores, alpha=0.8, color='steelblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    ax.set_xlabel('Fine-tuning Method', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Comparison of BLEU Scores Across Fine-tuning Methods', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, bleu_scores.max() * 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"BLEU Scores Bar Chart saved to {output_path}")
    plt.close(fig)

def plot_trainable_parameters(results_df: pd.DataFrame, filename: str = "trainable_parameters_bar_chart.png"):
    """
    Generates a bar chart comparing the number of trainable parameters across different methods.
    """
    if results_df.empty or 'Trainable Params (num)' not in results_df.columns:
        print("No 'Trainable Params (num)' data to plot.")
        return
    
    # Filter out rows with no trainable params data
    plot_df = results_df[results_df['Trainable Params (num)'].notna()].copy()
    
    if plot_df.empty:
        print("No numeric trainable parameters data to plot.")
        return

    print("Generating Trainable Parameters Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = plot_df['Method'].values
    trainable_params = plot_df['Trainable Params (num)'].values / 1e6  # Convert to millions
    
    bars = ax.bar(methods, trainable_params, alpha=0.8, color='coral')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M',
                ha='center', va='bottom')
    
    ax.set_xlabel('Fine-tuning Method', fontsize=12)
    ax.set_ylabel('Trainable Parameters (Millions)', fontsize=12)
    ax.set_title('Comparison of Trainable Parameters Across Fine-tuning Methods', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Trainable Parameters Bar Chart saved to {output_path}")
    plt.close(fig)

def plot_training_and_inference_time(results_df: pd.DataFrame, filename: str = "time_comparison_bar_chart.png"):
    """
    Generates a bar chart comparing training and inference times.
    """
    if results_df.empty:
        print("No data to plot for time comparison.")
        return

    plot_data = results_df[['Method', 'Training Time (min)', 'Inference Time (s)']].copy()
    plot_data = plot_data[plot_data['Training Time (min)'].notna() | plot_data['Inference Time (s)'].notna()]

    if plot_data.empty:
        print("No valid time data to plot.")
        return

    print("Generating Training and Inference Time Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = plot_data['Method'].values
    training_times = plot_data['Training Time (min)'].fillna(0).values
    inference_times = plot_data['Inference Time (s)'].fillna(0).values
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, training_times, width, label='Training Time (min)', alpha=0.8)
    bars2 = ax.bar(x + width/2, inference_times, width, label='Avg Inference Time (s)', alpha=0.8)
    
    ax.set_xlabel('Fine-tuning Method', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_title('Comparison of Training and Inference Times', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Training and Inference Time Bar Chart saved to {output_path}")
    plt.close(fig)

def plot_vram_usage(results_df: pd.DataFrame, filename: str = "vram_usage_bar_chart.png"):
    """
    Generates a bar chart comparing VRAM usage across different methods.
    """
    if results_df.empty or 'VRAM (GB)' not in results_df.columns:
        print("No VRAM data to plot.")
        return

    plot_df = results_df[results_df['VRAM (GB)'].notna()].copy()
    
    if plot_df.empty:
        print("No valid VRAM data to plot.")
        return

    print("Generating VRAM Usage Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = plot_df['Method'].values
    vram_usage = plot_df['VRAM (GB)'].values
    
    bars = ax.bar(methods, vram_usage, alpha=0.8, color='mediumpurple')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB',
                ha='center', va='bottom')
    
    ax.set_xlabel('Fine-tuning Method', fontsize=12)
    ax.set_ylabel('Peak VRAM Usage (GB)', fontsize=12)
    ax.set_title('Comparison of Peak VRAM Usage During Training', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"VRAM Usage Bar Chart saved to {output_path}")
    plt.close(fig)

def plot_radar_chart(results_df: pd.DataFrame, filename: str = "performance_radar_chart.png"):
    """
    Generates a radar chart comparing multiple metrics across fine-tuning methods.
    """
    if results_df.empty:
        print("No data to plot for radar chart.")
        return

    # Select metrics for radar chart
    metrics = ['ROUGE-1', 'ROUGE-L', 'BLEU']
    plot_df = results_df[['Method'] + metrics].copy()
    plot_df = plot_df[plot_df[metrics].notna().all(axis=1)]
    
    if plot_df.empty:
        print("No complete data for radar chart.")
        return

    print("Generating Performance Radar Chart...")
    
    # Normalize metrics to 0-1 scale for better visualization
    normalized_df = plot_df.copy()
    for metric in metrics:
        max_val = normalized_df[metric].max()
        if max_val > 0:
            normalized_df[metric] = normalized_df[metric] / max_val
    
    # Number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_df)))
    for idx, row in plot_df.iterrows():
        values = [normalized_df.loc[idx, metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Method'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Comparison - Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Performance Radar Chart saved to {output_path}")
    plt.close(fig)

def plot_comprehensive_comparison(results_df: pd.DataFrame, filename: str = "comprehensive_comparison.png"):
    """
    Generates a comprehensive comparison with multiple subplots.
    """
    if results_df.empty:
        print("No data to plot for comprehensive comparison.")
        return

    print("Generating Comprehensive Comparison Chart...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    methods = results_df['Method'].values
    
    # 1. ROUGE Scores
    ax1 = axes[0, 0]
    if 'ROUGE-1' in results_df.columns and 'ROUGE-L' in results_df.columns:
        x = np.arange(len(methods))
        width = 0.35
        ax1.bar(x - width/2, results_df['ROUGE-1'].fillna(0), width, label='ROUGE-1', alpha=0.8)
        ax1.bar(x + width/2, results_df['ROUGE-L'].fillna(0), width, label='ROUGE-L', alpha=0.8)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Score')
        ax1.set_title('ROUGE Scores Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. BLEU Score
    ax2 = axes[0, 1]
    if 'BLEU' in results_df.columns:
        ax2.bar(methods, results_df['BLEU'].fillna(0), alpha=0.8, color='steelblue')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('BLEU Score')
        ax2.set_title('BLEU Score Comparison')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Training Time
    ax3 = axes[1, 0]
    if 'Training Time (min)' in results_df.columns:
        ax3.bar(methods, results_df['Training Time (min)'].fillna(0), alpha=0.8, color='coral')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Time (minutes)')
        ax3.set_title('Training Time Comparison')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. VRAM Usage
    ax4 = axes[1, 1]
    if 'VRAM (GB)' in results_df.columns:
        ax4.bar(methods, results_df['VRAM (GB)'].fillna(0), alpha=0.8, color='mediumpurple')
        ax4.set_xlabel('Method')
        ax4.set_ylabel('VRAM (GB)')
        ax4.set_title('VRAM Usage Comparison')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Comprehensive Comparison Chart saved to {output_path}")
    plt.close(fig)

def main():
    print("Starting visualization process...")
    
    results_path = os.path.join(TABLES_DIR, "evaluation_results.csv")
    if not os.path.exists(results_path):
        print(f"❌ Error: Evaluation results file not found at {results_path}")
        print("\n💡 Bạn cần chạy evaluate.py trước:")
        print("   python src/evaluate.py")
        print("\nSau đó mới chạy visualize.py")
        return
    
    # Check if file is empty
    try:
        results_df = pd.read_csv(results_path)
        if results_df.empty:
            print(f"❌ Error: Evaluation results file is empty at {results_path}")
            print("\n💡 Có thể evaluate.py chưa hoàn thành hoặc có lỗi.")
            print("   Chạy lại: python src/evaluate.py")
            return
        
        print("✅ Evaluation results loaded successfully for visualization.")
        print(f"Found {len(results_df)} methods to visualize.")
        print(f"\nMethods: {', '.join(results_df['Method'].values if 'Method' in results_df.columns else ['Unknown'])}")
        
        # Generate all visualizations
        plot_rouge_scores(results_df)
        plot_bleu_scores(results_df)
        plot_trainable_parameters(results_df)
        plot_training_and_inference_time(results_df)
        plot_vram_usage(results_df)
        plot_radar_chart(results_df)
        plot_comprehensive_comparison(results_df)

    except pd.errors.EmptyDataError:
        print(f"❌ Error: Evaluation results file is empty at {results_path}")
        print("\n💡 Bạn cần chạy evaluate.py trước:")
        print("   python src/evaluate.py")
    except Exception as e:
        print(f"❌ Error reading evaluation results: {e}")
        print("\n💡 Có thể file bị corrupt. Chạy lại evaluate.py:")
        print("   python src/evaluate.py")

    print("\nVisualization process complete.")

if __name__ == "__main__":
    main()
