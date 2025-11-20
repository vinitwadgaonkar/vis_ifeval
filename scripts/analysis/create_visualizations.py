#!/usr/bin/env python3
"""
Create advanced/crazy visualizations for the VIF-Eval paper
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

def load_results():
    """Load all model results"""
    base_path = Path("vif_eval_submission_20251119/results")
    results = {}
    
    for model_file in ["gpt_image1_results.json", "nano_banana_results.json", "dalle3_results.json"]:
        model_name = model_file.replace("_results.json", "").replace("_", "-")
        filepath = base_path / model_file
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
    
    return results

def create_heatmap_performance_matrix(results):
    """Create a heatmap showing performance across models and constraint types"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = []
    constraint_types = []
    pass_rates = []
    
    for model_name, data in results.items():
        if 'summary' in data and 'by_type' in data['summary']:
            for ctype, stats in data['summary']['by_type'].items():
                if 'pass_rate' in stats:
                    models.append(model_name.replace("-", " ").title())
                    constraint_types.append(ctype.replace("_", " ").title())
                    pass_rates.append(stats['pass_rate'] * 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Constraint Type': constraint_types,
        'Pass Rate': pass_rates
    })
    
    # Pivot for heatmap
    pivot = df.pivot(index='Constraint Type', columns='Model', values='Pass Rate')
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Pass Rate (%)'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Performance Heatmap: Pass Rate by Model and Constraint Type', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Constraint Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def create_radar_chart_capabilities(results):
    """Create radar/spider chart comparing model capabilities"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Get constraint types
    constraint_types = set()
    for data in results.values():
        if 'summary' in data and 'by_type' in data['summary']:
            constraint_types.update(data['summary']['by_type'].keys())
    
    constraint_types = sorted([ct for ct in constraint_types if ct not in ['sketch_to_render', 'character_consistency']])
    
    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(constraint_types), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (model_name, data) in enumerate(results.items()):
        if 'summary' not in data or 'by_type' not in data['summary']:
            continue
            
        values = []
        for ctype in constraint_types:
            if ctype in data['summary']['by_type']:
                values.append(data['summary']['by_type'][ctype].get('pass_rate', 0) * 100)
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name.replace("-", " ").title(), 
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ct.replace("_", " ").title() for ct in constraint_types], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_title('Model Capabilities Radar Chart\n(Pass Rate by Constraint Type)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    return fig

def create_cost_performance_scatter(results):
    """Create scatter plot: Cost vs Performance"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cost data (from analysis)
    cost_data = {
        'gpt-image-1': {'cost': 1.88, 'pass_rate': 58.1, 'images': 47},
        'nano-banana': {'cost': 0.24, 'pass_rate': 54.7, 'images': 47},
        'dall-e-3': {'cost': 0.44, 'pass_rate': 17.1, 'images': 11}
    }
    
    for model_name, data in results.items():
        if model_name in cost_data:
            cost_info = cost_data[model_name]
            ax.scatter(cost_info['cost'], cost_info['pass_rate'], 
                      s=cost_info['images']*50, alpha=0.6, 
                      label=model_name.replace("-", " ").title(),
                      edgecolors='black', linewidth=2)
            
            # Add annotation
            ax.annotate(model_name.replace("-", " ").title(),
                       (cost_info['cost'], cost_info['pass_rate']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Total Cost (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Performance Trade-off\n(Bubble size = Number of Images)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    
    # Add efficiency lines
    x_max = max([c['cost'] for c in cost_data.values()])
    y_max = max([c['pass_rate'] for c in cost_data.values()])
    
    # Pareto frontier approximation
    efficient_models = [(c['cost'], c['pass_rate']) for c in cost_data.values()]
    efficient_models.sort(key=lambda x: (x[0], -x[1]))
    
    plt.tight_layout()
    return fig

def create_failure_mode_sankey(results):
    """Create a Sankey-like diagram showing failure flows"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Aggregate failure data
    failure_data = {}
    for model_name, data in results.items():
        if 'summary' in data and 'by_type' in data['summary']:
            failures = {}
            for ctype, stats in data['summary']['by_type'].items():
                total = stats.get('total', 0)
                passed = stats.get('passed', 0)
                failed = total - passed
                if failed > 0:
                    failures[ctype] = failed
            failure_data[model_name] = failures
    
    # Create stacked bar chart showing failure distribution
    constraint_types = set()
    for failures in failure_data.values():
        constraint_types.update(failures.keys())
    constraint_types = sorted(constraint_types)
    
    x = np.arange(len(constraint_types))
    width = 0.25
    
    for idx, (model_name, failures) in enumerate(failure_data.items()):
        values = [failures.get(ct, 0) for ct in constraint_types]
        ax.bar(x + idx*width, values, width, label=model_name.replace("-", " ").title(),
               alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Constraint Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Failed Constraints', fontsize=12, fontweight='bold')
    ax.set_title('Failure Distribution Across Constraint Types', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([ct.replace("_", " ").title() for ct in constraint_types], 
                       rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    return fig

def create_performance_vs_difficulty(results):
    """Create scatter plot: Constraint Difficulty vs Performance"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate difficulty as inverse of average pass rate across models
    constraint_difficulty = {}
    constraint_performance = {}
    
    for model_name, data in results.items():
        if 'summary' in data and 'by_type' in data['summary']:
            for ctype, stats in data['summary']['by_type'].items():
                if ctype not in constraint_difficulty:
                    constraint_difficulty[ctype] = []
                    constraint_performance[ctype] = []
                
                pass_rate = stats.get('pass_rate', 0)
                constraint_performance[ctype].append(pass_rate)
    
    # Calculate average pass rate (lower = more difficult)
    avg_pass_rates = {ct: np.mean(perf) for ct, perf in constraint_performance.items()}
    difficulties = {ct: 1 - avg_pass for ct, avg_pass in avg_pass_rates.items()}
    
    # Plot each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    for idx, (model_name, data) in enumerate(results.items()):
        if 'summary' not in data or 'by_type' not in data['summary']:
            continue
        
        x_vals = []
        y_vals = []
        labels = []
        
        for ctype, stats in data['summary']['by_type'].items():
            if ctype in difficulties:
                x_vals.append(difficulties[ctype] * 100)
                y_vals.append(stats.get('pass_rate', 0) * 100)
                labels.append(ctype.replace("_", " ").title())
        
        ax.scatter(x_vals, y_vals, s=150, alpha=0.7, 
                  label=model_name.replace("-", " ").title(),
                  color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
                  edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Constraint Difficulty (1 - Average Pass Rate) %', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Constraint Difficulty', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')
    
    # Add diagonal reference line (perfect performance)
    ax.plot([0, 100], [100, 0], 'k--', alpha=0.3, linewidth=1, label='Perfect Performance')
    
    plt.tight_layout()
    return fig

def create_constraint_type_comparison_matrix(results):
    """Create a detailed comparison matrix with multiple metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get all constraint types
    all_types = set()
    for data in results.values():
        if 'summary' in data and 'by_type' in data['summary']:
            all_types.update(data['summary']['by_type'].keys())
    all_types = sorted([t for t in all_types if t not in ['sketch_to_render', 'character_consistency']])
    
    models = list(results.keys())
    
    # Subplot 1: Pass Rate
    ax1 = fig.add_subplot(gs[0, 0])
    data_matrix = []
    for ctype in all_types:
        row = []
        for model in models:
            if model in results and 'summary' in results[model] and 'by_type' in results[model]['summary']:
                if ctype in results[model]['summary']['by_type']:
                    row.append(results[model]['summary']['by_type'][ctype].get('pass_rate', 0) * 100)
                else:
                    row.append(0)
            else:
                row.append(0)
        data_matrix.append(row)
    
    im1 = ax1.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace("-", " ").title() for m in models], rotation=45, ha='right')
    ax1.set_yticks(range(len(all_types)))
    ax1.set_yticklabels([t.replace("_", " ").title() for t in all_types])
    ax1.set_title('Pass Rate (%)', fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='%')
    
    # Subplot 2: Average Score
    ax2 = fig.add_subplot(gs[0, 1])
    data_matrix2 = []
    for ctype in all_types:
        row = []
        for model in models:
            if model in results and 'summary' in results[model] and 'by_type' in results[model]['summary']:
                if ctype in results[model]['summary']['by_type']:
                    row.append(results[model]['summary']['by_type'][ctype].get('avg_score', 0) * 100)
                else:
                    row.append(0)
            else:
                row.append(0)
        data_matrix2.append(row)
    
    im2 = ax2.imshow(data_matrix2, cmap='viridis', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace("-", " ").title() for m in models], rotation=45, ha='right')
    ax2.set_yticks(range(len(all_types)))
    ax2.set_yticklabels([t.replace("_", " ").title() for t in all_types])
    ax2.set_title('Average Score (%)', fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='%')
    
    # Subplot 3: Total Constraints
    ax3 = fig.add_subplot(gs[1, 0])
    data_matrix3 = []
    for ctype in all_types:
        row = []
        for model in models:
            if model in results and 'summary' in results[model] and 'by_type' in results[model]['summary']:
                if ctype in results[model]['summary']['by_type']:
                    row.append(results[model]['summary']['by_type'][ctype].get('total', 0))
                else:
                    row.append(0)
            else:
                row.append(0)
        data_matrix3.append(row)
    
    im3 = ax3.imshow(data_matrix3, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.replace("-", " ").title() for m in models], rotation=45, ha='right')
    ax3.set_yticks(range(len(all_types)))
    ax3.set_yticklabels([t.replace("_", " ").title() for t in all_types])
    ax3.set_title('Total Constraints', fontweight='bold', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Count')
    
    # Subplot 4: Passed Constraints
    ax4 = fig.add_subplot(gs[1, 1])
    data_matrix4 = []
    for ctype in all_types:
        row = []
        for model in models:
            if model in results and 'summary' in results[model] and 'by_type' in results[model]['summary']:
                if ctype in results[model]['summary']['by_type']:
                    row.append(results[model]['summary']['by_type'][ctype].get('passed', 0))
                else:
                    row.append(0)
            else:
                row.append(0)
        data_matrix4.append(row)
    
    im4 = ax4.imshow(data_matrix4, cmap='Greens', aspect='auto')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels([m.replace("-", " ").title() for m in models], rotation=45, ha='right')
    ax4.set_yticks(range(len(all_types)))
    ax4.set_yticklabels([t.replace("_", " ").title() for t in all_types])
    ax4.set_title('Passed Constraints', fontweight='bold', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='Count')
    
    fig.suptitle('Comprehensive Performance Matrix', fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def create_error_pattern_breakdown(results):
    """Create a detailed error pattern visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Error rate by constraint type
    error_rates = {}
    for model_name, data in results.items():
        if 'summary' in data and 'by_type' in data['summary']:
            for ctype, stats in data['summary']['by_type'].items():
                if ctype not in error_rates:
                    error_rates[ctype] = []
                total = stats.get('total', 0)
                passed = stats.get('passed', 0)
                if total > 0:
                    error_rates[ctype].append((total - passed) / total * 100)
    
    avg_errors = {ct: np.mean(errs) for ct, errs in error_rates.items()}
    sorted_types = sorted(avg_errors.items(), key=lambda x: x[1], reverse=True)
    
    types = [t[0].replace("_", " ").title() for t in sorted_types]
    errors = [t[1] for t in sorted_types]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(types)))
    bars = ax1.barh(types, errors, color=colors, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Average Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Error Rate by Constraint Type\n(Average Across Models)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax1.text(err + 1, i, f'{err:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # Right: Model comparison of error rates
    model_errors = {}
    for model_name, data in results.items():
        if 'summary' in data:
            total = data['summary'].get('total_constraints', 0)
            passed = data['summary'].get('passed_constraints', 0)
            if total > 0:
                model_errors[model_name] = (total - passed) / total * 100
    
    models = list(model_errors.keys())
    errors = list(model_errors.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar([m.replace("-", " ").title() for m in models], errors, 
                   color=colors[:len(models)], edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Error Rate by Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{err:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all advanced visualizations"""
    print("Loading results...")
    results = load_results()
    
    output_dir = Path("paper_assets/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizations = [
        ("heatmap_performance_matrix", create_heatmap_performance_matrix),
        ("radar_capabilities", create_radar_chart_capabilities),
        ("cost_performance_scatter", create_cost_performance_scatter),
        ("failure_mode_distribution", create_failure_mode_sankey),
        ("performance_vs_difficulty", create_performance_vs_difficulty),
        ("constraint_comparison_matrix", create_constraint_type_comparison_matrix),
        ("error_pattern_breakdown", create_error_pattern_breakdown),
    ]
    
    print(f"\nGenerating {len(visualizations)} advanced visualizations...")
    
    for name, func in visualizations:
        try:
            print(f"  Creating {name}...")
            fig = func(results)
            filepath = output_dir / f"{name}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"    ✅ Saved: {filepath}")
        except Exception as e:
            print(f"    ❌ Error creating {name}: {e}")
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated visualizations:")
    for name, _ in visualizations:
        print(f"  - {name}.png")

if __name__ == "__main__":
    main()

