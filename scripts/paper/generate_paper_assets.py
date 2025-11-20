#!/usr/bin/env python3
"""Generate all assets needed for the benchmark paper."""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_results(model_name):
    """Load results for a model."""
    if model_name == "gpt-image-1":
        results_file = Path(__file__).parent / "data" / "outputs" / "full_evaluation" / "results.json"
    elif model_name == "nano-banana":
        results_file = Path(__file__).parent / "data" / "outputs" / "full_evaluation_openrouter" / "results.json"
    elif model_name == "dall-e-3":
        results_file = Path(__file__).parent / "data" / "outputs" / "full_evaluation_dalle" / "results.json"
    else:
        return None
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_comparison_table():
    """Generate comprehensive comparison table."""
    models = ["gpt-image-1", "nano-banana", "dall-e-3"]
    results = {}
    
    for model in models:
        data = load_results(model)
        if data:
            results[model] = data
    
    # Create comparison data
    comparison = {
        "models": [],
        "overall": [],
        "by_type": defaultdict(lambda: defaultdict(list)),
        "by_category": defaultdict(lambda: defaultdict(list))
    }
    
    for model, data in results.items():
        summary = data.get("summary", {})
        comparison["models"].append(model)
        comparison["overall"].append({
            "total_prompts": summary.get("total_prompts", 0),
            "total_constraints": summary.get("total_constraints", 0),
            "passed": summary.get("passed_constraints", 0),
            "failed": summary.get("failed_constraints", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "errors": summary.get("errors", 0)
        })
        
        # By type
        for ctype, stats in summary.get("by_type", {}).items():
            comparison["by_type"][ctype][model] = {
                "total": stats.get("total", 0),
                "passed": stats.get("passed", 0),
                "pass_rate": stats.get("pass_rate", 0.0),
                "avg_score": stats.get("avg_score", 0.0)
            }
        
        # By category
        for category, stats in summary.get("by_category", {}).items():
            comparison["by_category"][category][model] = {
                "total": stats.get("total", 0),
                "passed": stats.get("passed", 0),
                "pass_rate": stats.get("pass_rate", 0.0),
                "avg_score": stats.get("avg_score", 0.0)
            }
    
    # Save comparison data
    output_dir = Path(__file__).parent / "paper_assets"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "comparison_data.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(comparison)
    with open(output_dir / "comparison_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Generate Markdown table
    markdown_table = generate_markdown_table(comparison)
    with open(output_dir / "comparison_table.md", 'w') as f:
        f.write(markdown_table)
    
    print(f"Comparison tables saved to {output_dir}")
    return comparison

def generate_latex_table(comparison):
    """Generate LaTeX table for comparison."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Overall Performance Comparison}\n"
    latex += "\\label{tab:overall_comparison}\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "Model & Prompts & Constraints & Passed & Pass Rate & Errors \\\\\n"
    latex += "\\hline\n"
    
    model_names = {
        "gpt-image-1": "GPT Image 1",
        "nano-banana": "Nano Banana",
        "dall-e-3": "DALL-E 3"
    }
    
    for i, model in enumerate(comparison["models"]):
        overall = comparison["overall"][i]
        name = model_names.get(model, model)
        latex += f"{name} & {overall['total_prompts']} & {overall['total_constraints']} & "
        latex += f"{overall['passed']} & {overall['pass_rate']*100:.1f}\\% & {overall['errors']} \\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex

def generate_markdown_table(comparison):
    """Generate Markdown table for comparison."""
    md = "## Overall Performance Comparison\n\n"
    md += "| Model | Prompts | Constraints | Passed | Pass Rate | Errors |\n"
    md += "|-------|---------|-------------|--------|-----------|--------|\n"
    
    model_names = {
        "gpt-image-1": "GPT Image 1",
        "nano-banana": "Nano Banana (Gemini 2.5 Flash)",
        "dall-e-3": "DALL-E 3"
    }
    
    for i, model in enumerate(comparison["models"]):
        overall = comparison["overall"][i]
        name = model_names.get(model, model)
        md += f"| {name} | {overall['total_prompts']} | {overall['total_constraints']} | "
        md += f"{overall['passed']} | {overall['pass_rate']*100:.1f}% | {overall['errors']} |\n"
    
    return md

def generate_visualizations(comparison):
    """Generate additional visualizations."""
    output_dir = Path(__file__).parent / "paper_assets" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = comparison["models"]
    model_names = {
        "gpt-image-1": "GPT Image 1",
        "nano-banana": "Nano Banana",
        "dall-e-3": "DALL-E 3"
    }
    
    # 1. Pass rate comparison by constraint type
    fig, ax = plt.subplots(figsize=(14, 8))
    constraint_types = sorted(comparison["by_type"].keys())
    x = np.arange(len(constraint_types))
    width = 0.25
    
    for i, model in enumerate(models):
        pass_rates = [comparison["by_type"][ctype][model].get("pass_rate", 0.0) * 100 
                      if model in comparison["by_type"][ctype] else 0.0 
                      for ctype in constraint_types]
        offset = (i - 1) * width
        ax.bar(x + offset, pass_rates, width, label=model_names.get(model, model))
    
    ax.set_xlabel('Constraint Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pass Rate by Constraint Type Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "pass_rate_by_type_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Average score comparison by constraint type
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        avg_scores = [comparison["by_type"][ctype][model].get("avg_score", 0.0) 
                     if model in comparison["by_type"][ctype] else 0.0 
                     for ctype in constraint_types]
        offset = (i - 1) * width
        ax.bar(x + offset, avg_scores, width, label=model_names.get(model, model))
    
    ax.set_xlabel('Constraint Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Average Score by Constraint Type Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(constraint_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Pass Threshold')
    plt.tight_layout()
    plt.savefig(output_dir / "avg_score_by_type_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overall pass rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    pass_rates = [overall["pass_rate"] * 100 for overall in comparison["overall"]]
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(models)]
    
    bars = ax.bar([model_names.get(m, m) for m in models], pass_rates, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Overall Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Pass Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_pass_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_errors():
    """Analyze error patterns across models."""
    models = ["gpt-image-1", "nano-banana", "dall-e-3"]
    error_analysis = {}
    
    for model in models:
        data = load_results(model)
        if not data:
            continue
        
        errors = {
            "by_type": defaultdict(int),
            "by_category": defaultdict(int),
            "common_failures": []
        }
        
        for prompt in data.get("prompts", []):
            if prompt.get("error"):
                category = prompt.get("category", "unknown")
                errors["by_category"][category] += 1
                errors["common_failures"].append({
                    "id": prompt.get("id"),
                    "category": category,
                    "error": prompt.get("error", "")[:100]  # Truncate
                })
            
            for constraint in prompt.get("constraints", []):
                if constraint.get("error"):
                    ctype = constraint.get("type", "unknown")
                    errors["by_type"][ctype] += 1
        
        error_analysis[model] = errors
    
    # Save error analysis
    output_dir = Path(__file__).parent / "paper_assets"
    with open(output_dir / "error_analysis.json", 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    return error_analysis

def main():
    """Generate all paper assets."""
    print("="*80)
    print("GENERATING PAPER ASSETS")
    print("="*80)
    
    print("\n1. Generating comparison tables...")
    comparison = generate_comparison_table()
    
    print("\n2. Generating visualizations...")
    generate_visualizations(comparison)
    
    print("\n3. Analyzing errors...")
    error_analysis = analyze_errors()
    
    print("\n" + "="*80)
    print("PAPER ASSETS GENERATION COMPLETE")
    print("="*80)
    print(f"\nAssets saved to: {Path(__file__).parent / 'paper_assets'}")

if __name__ == "__main__":
    main()

