#!/usr/bin/env python3
"""Create detailed error analysis for the paper."""

import json
from pathlib import Path
from collections import defaultdict

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

def analyze_failures():
    """Analyze failure patterns in detail."""
    models = ["gpt-image-1", "nano-banana", "dall-e-3"]
    
    analysis = {
        "failure_patterns": defaultdict(lambda: defaultdict(int)),
        "common_errors": [],
        "constraint_failures": defaultdict(lambda: list),
        "score_distributions": defaultdict(list)
    }
    
    for model in models:
        data = load_results(model)
        if not data:
            continue
        
        for prompt in data.get("prompts", []):
            category = prompt.get("category", "unknown")
            
            # Analyze constraint failures
            for constraint in prompt.get("constraints", []):
                ctype = constraint.get("type", "unknown")
                score = constraint.get("score", 0.0)
                passed = constraint.get("passed", False)
                
                analysis["score_distributions"][ctype].append(score)
                
                if not passed:
                    analysis["failure_patterns"][ctype][category] += 1
                    analysis["constraint_failures"][ctype].append({
                        "prompt_id": prompt.get("id"),
                        "category": category,
                        "score": score,
                        "constraint_id": constraint.get("id")
                    })
            
            # Analyze prompt-level errors
            if prompt.get("error"):
                error_msg = prompt.get("error", "")
                analysis["common_errors"].append({
                    "model": model,
                    "prompt_id": prompt.get("id"),
                    "category": category,
                    "error": error_msg[:200]
                })
    
    # Find most common failure patterns
    failure_summary = {}
    for ctype, categories in analysis["failure_patterns"].items():
        total_failures = sum(categories.values())
        failure_summary[ctype] = {
            "total": total_failures,
            "by_category": dict(categories),
            "avg_score": sum(analysis["score_distributions"][ctype]) / len(analysis["score_distributions"][ctype]) if analysis["score_distributions"][ctype] else 0.0
        }
    
    analysis["failure_summary"] = failure_summary
    
    return analysis

def generate_error_report(analysis):
    """Generate markdown error report."""
    report = "# Error Analysis Report\n\n"
    
    report += "## Failure Patterns by Constraint Type\n\n"
    for ctype, summary in sorted(analysis["failure_summary"].items(), key=lambda x: x[1]["total"], reverse=True):
        report += f"### {ctype}\n\n"
        report += f"- **Total Failures**: {summary['total']}\n"
        report += f"- **Average Score**: {summary['avg_score']:.3f}\n"
        report += f"- **Failures by Category**:\n"
        for category, count in summary["by_category"].items():
            report += f"  - {category}: {count}\n"
        report += "\n"
    
    report += "## Common Errors\n\n"
    error_types = defaultdict(int)
    for error in analysis["common_errors"]:
        error_msg = error["error"].lower()
        if "billing" in error_msg or "limit" in error_msg:
            error_types["API Billing Limit"] += 1
        elif "timeout" in error_msg:
            error_types["Timeout"] += 1
        else:
            error_types["Other"] += 1
    
    for error_type, count in error_types.items():
        report += f"- **{error_type}**: {count} occurrences\n"
    
    return report

def main():
    """Generate detailed error analysis."""
    print("Generating detailed error analysis...")
    
    analysis = analyze_failures()
    
    output_dir = Path(__file__).parent / "paper_assets"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / "detailed_error_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    report = generate_error_report(analysis)
    with open(output_dir / "error_analysis_report.md", 'w') as f:
        f.write(report)
    
    print(f"Error analysis saved to {output_dir}")

if __name__ == "__main__":
    main()

