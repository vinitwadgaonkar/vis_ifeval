#!/usr/bin/env python3
"""Merge special evaluator results with full evaluation results."""

import json
from pathlib import Path
from collections import defaultdict

def load_prompts(prompt_file):
    """Load prompts from JSONL file."""
    prompts = []
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
    return prompts

def merge_results(base_results_file, test_results, model_name):
    """Merge test results into base results."""
    base_dir = Path(base_results_file).parent
    
    # Load base results
    with open(base_results_file, 'r') as f:
        base_results = json.load(f)
    
    # Load prompt files to get full prompt data
    prompts_dir = Path(__file__).parent / "prompts"
    char_prompts = load_prompts(prompts_dir / "prompts_character_consistency.jsonl")
    sketch_prompts = load_prompts(prompts_dir / "prompts_sketch_to_render.jsonl")
    
    # Create prompt lookup
    prompt_lookup = {}
    for p in char_prompts + sketch_prompts:
        prompt_lookup[p['id']] = p
    
    # Create test result lookup by prompt ID
    test_result_lookup = {}
    for test_result in test_results['character_consistency']:
        test_result_lookup[test_result['id']] = test_result
    for test_result in test_results['sketch_to_render']:
        test_result_lookup[test_result['id']] = test_result
    
    # Update existing prompts or add constraints
    for prompt in base_results['prompts']:
        prompt_id = prompt['id']
        
        # Check if this is a character consistency prompt we tested
        if prompt_id in test_result_lookup and prompt_id in prompt_lookup:
            test_result = test_result_lookup[prompt_id]
            prompt_data = prompt_lookup[prompt_id]
            
            # Update or add constraints
            if len(prompt['constraints']) == 0:
                # Add constraints from prompt data
                for constraint_data in prompt_data.get("constraints", []):
                    constraint_result = {
                        "id": constraint_data.get("id", "unknown"),
                        "type": constraint_data.get("type", "unknown"),
                        "score": test_result['score'],
                        "passed": test_result['passed']
                    }
                    prompt['constraints'].append(constraint_result)
            else:
                # Update existing constraints
                for constraint in prompt['constraints']:
                    if constraint.get('type') in ['character_consistency', 'sketch_to_render']:
                        constraint['score'] = test_result['score']
                        constraint['passed'] = test_result['passed']
    
    # Recalculate summary statistics
    stats = {
        "total_constraints": 0,
        "passed_constraints": 0,
        "failed_constraints": 0,
        "errors": 0,
        "by_type": defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "scores": []}),
        "by_category": defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "scores": []})
    }
    
    for prompt in base_results['prompts']:
        category = prompt.get("category", "unknown")
        for constraint in prompt.get("constraints", []):
            constraint_type = constraint.get("type", "unknown")
            score = constraint.get("score", 0.0)
            passed = constraint.get("passed", False)
            
            stats["total_constraints"] += 1
            stats["by_type"][constraint_type]["total"] += 1
            stats["by_category"][category]["total"] += 1
            
            if passed:
                stats["passed_constraints"] += 1
                stats["by_type"][constraint_type]["passed"] += 1
                stats["by_category"][category]["passed"] += 1
            else:
                stats["failed_constraints"] += 1
                stats["by_type"][constraint_type]["failed"] += 1
                stats["by_category"][category]["failed"] += 1
            
            stats["by_type"][constraint_type]["scores"].append(score)
            stats["by_category"][category]["scores"].append(score)
        
        if prompt.get("error"):
            stats["errors"] += 1
    
    # Update summary
    base_results["summary"] = {
        "total_prompts": len(base_results['prompts']),
        "total_constraints": stats["total_constraints"],
        "passed_constraints": stats["passed_constraints"],
        "failed_constraints": stats["failed_constraints"],
        "errors": stats["errors"],
        "pass_rate": stats["passed_constraints"] / stats["total_constraints"] if stats["total_constraints"] > 0 else 0.0,
        "by_type": {},
        "by_category": {}
    }
    
    # Calculate stats by type
    for ctype, data in stats["by_type"].items():
        if data["total"] > 0:
            base_results["summary"]["by_type"][ctype] = {
                "total": data["total"],
                "passed": data["passed"],
                "failed": data["failed"],
                "pass_rate": data["passed"] / data["total"],
                "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0,
                "min_score": min(data["scores"]) if data["scores"] else 0.0,
                "max_score": max(data["scores"]) if data["scores"] else 0.0
            }
    
    # Calculate stats by category
    for category, data in stats["by_category"].items():
        if data["total"] > 0:
            base_results["summary"]["by_category"][category] = {
                "total": data["total"],
                "passed": data["passed"],
                "failed": data["failed"],
                "pass_rate": data["passed"] / data["total"],
                "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
            }
    
    # Update metadata
    base_results["metadata"]["merged_with_special_evaluators"] = True
    
    return base_results

def main():
    base_dir = Path(__file__).parent
    
    # Use known results from the test output
    # GPT Image 1 results from test output:
    gpt_results = {
        'character_consistency': [
            {'id': 'char_002', 'score': 0.9939, 'passed': True},
            {'id': 'char_003', 'score': 0.9909, 'passed': True},
            {'id': 'char_005', 'score': 0.9898, 'passed': True},
            {'id': 'char_006', 'score': 0.9750, 'passed': True},
        ],
        'sketch_to_render': [
            {'id': 'sketch_001', 'score': 0.4816, 'passed': False},
            {'id': 'sketch_002', 'score': 0.6111, 'passed': True},
            {'id': 'sketch_003', 'score': 0.3627, 'passed': False},
            {'id': 'sketch_004', 'score': 0.4110, 'passed': False},
            {'id': 'sketch_005', 'score': 0.4139, 'passed': False},
        ]
    }
    
    # Nano Banana results from test output:
    nano_results = {
        'character_consistency': [
            {'id': 'char_002', 'score': 0.9754, 'passed': True},
            {'id': 'char_003', 'score': 0.9929, 'passed': True},
            {'id': 'char_005', 'score': 0.9796, 'passed': True},
            {'id': 'char_006', 'score': 0.9688, 'passed': True},
        ],
        'sketch_to_render': [
            {'id': 'sketch_001', 'score': 0.5591, 'passed': True},
            {'id': 'sketch_002', 'score': 0.4666, 'passed': False},
            {'id': 'sketch_003', 'score': 0.4031, 'passed': False},
            {'id': 'sketch_004', 'score': 0.4845, 'passed': False},
            {'id': 'sketch_005', 'score': 0.3427, 'passed': False},
        ]
    }
    
    print(f"Using extracted results:")
    print(f"GPT: {len(gpt_results['character_consistency'])} char, {len(gpt_results['sketch_to_render'])} sketch")
    print(f"Nano: {len(nano_results['character_consistency'])} char, {len(nano_results['sketch_to_render'])} sketch")
    
    # Merge GPT Image 1 results
    gpt_results_file = base_dir / "data" / "outputs" / "full_evaluation" / "results.json"
    if gpt_results_file.exists():
        print(f"\nMerging GPT Image 1 results...")
        merged_gpt = merge_results(gpt_results_file, gpt_results, "gpt-image-1")
        
        # Save merged results
        output_file = base_dir / "data" / "outputs" / "full_evaluation" / "results_merged.json"
        with open(output_file, 'w') as f:
            json.dump(merged_gpt, f, indent=2)
        print(f"Saved merged GPT results to: {output_file}")
        
        # Also update the original file
        with open(gpt_results_file, 'w') as f:
            json.dump(merged_gpt, f, indent=2)
        print(f"Updated original GPT results file")
        
        # Print updated summary
        summary = merged_gpt['summary']
        print(f"\nUpdated GPT Summary:")
        print(f"  Total Prompts: {summary['total_prompts']}")
        print(f"  Total Constraints: {summary['total_constraints']}")
        print(f"  Passed: {summary['passed_constraints']} ({summary['pass_rate']*100:.1f}%)")
        if 'character_consistency' in summary['by_type']:
            cc = summary['by_type']['character_consistency']
            print(f"  Character Consistency: {cc['passed']}/{cc['total']} passed ({cc['pass_rate']*100:.1f}%), avg: {cc['avg_score']:.4f}")
        if 'sketch_to_render' in summary['by_type']:
            str_eval = summary['by_type']['sketch_to_render']
            print(f"  Sketch to Render: {str_eval['passed']}/{str_eval['total']} passed ({str_eval['pass_rate']*100:.1f}%), avg: {str_eval['avg_score']:.4f}")
    else:
        print(f"Warning: GPT results file not found: {gpt_results_file}")
    
    # Merge Nano Banana results
    nano_results_file = base_dir / "data" / "outputs" / "full_evaluation_openrouter" / "results.json"
    if nano_results_file.exists():
        print(f"\nMerging Nano Banana results...")
        merged_nano = merge_results(nano_results_file, nano_results, "nano-banana")
        
        # Save merged results
        output_file = base_dir / "data" / "outputs" / "full_evaluation_openrouter" / "results_merged.json"
        with open(output_file, 'w') as f:
            json.dump(merged_nano, f, indent=2)
        print(f"Saved merged Nano results to: {output_file}")
        
        # Also update the original file
        with open(nano_results_file, 'w') as f:
            json.dump(merged_nano, f, indent=2)
        print(f"Updated original Nano results file")
        
        # Print updated summary
        summary = merged_nano['summary']
        print(f"\nUpdated Nano Summary:")
        print(f"  Total Prompts: {summary['total_prompts']}")
        print(f"  Total Constraints: {summary['total_constraints']}")
        print(f"  Passed: {summary['passed_constraints']} ({summary['pass_rate']*100:.1f}%)")
        if 'character_consistency' in summary['by_type']:
            cc = summary['by_type']['character_consistency']
            print(f"  Character Consistency: {cc['passed']}/{cc['total']} passed ({cc['pass_rate']*100:.1f}%), avg: {cc['avg_score']:.4f}")
        if 'sketch_to_render' in summary['by_type']:
            str_eval = summary['by_type']['sketch_to_render']
            print(f"  Sketch to Render: {str_eval['passed']}/{str_eval['total']} passed ({str_eval['pass_rate']*100:.1f}%), avg: {str_eval['avg_score']:.4f}")
    else:
        print(f"Warning: Nano results file not found: {nano_results_file}")
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
