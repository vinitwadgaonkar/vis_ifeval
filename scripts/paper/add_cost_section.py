#!/usr/bin/env python3
"""Add cost and latency analysis section to paper."""

import json
from pathlib import Path

def estimate_costs():
    """Estimate API costs based on evaluation."""
    # OpenAI pricing (as of 2024):
    # GPT Image 1: ~$0.04 per image (1024x1024, high quality)
    # DALL-E 3: ~$0.04 per image (1024x1024, HD)
    # OpenRouter Gemini: ~$0.001-0.01 per image (varies)
    
    costs = {
        "gpt-image-1": {
            "images_generated": 47,
            "cost_per_image": 0.04,
            "total_cost": 47 * 0.04,
            "notes": "OpenAI GPT Image 1 pricing"
        },
        "nano-banana": {
            "images_generated": 47,
            "cost_per_image": 0.005,  # Estimated for OpenRouter
            "total_cost": 47 * 0.005,
            "notes": "OpenRouter Gemini 2.5 Flash Image pricing (estimated)"
        },
        "dall-e-3": {
            "images_generated": 11,  # Incomplete
            "cost_per_image": 0.04,
            "total_cost": 11 * 0.04,
            "notes": "OpenAI DALL-E 3 pricing (incomplete evaluation)"
        }
    }
    
    return costs

def generate_cost_section():
    """Generate cost/latency section for paper."""
    costs = estimate_costs()
    
    section = "## 6. Cost and Latency Analysis\n\n"
    section += "### 6.1 API Costs\n\n"
    section += "Estimated costs for the evaluation (based on current API pricing):\n\n"
    section += "| Model | Images Generated | Cost per Image | Total Cost |\n"
    section += "|-------|------------------|----------------|------------|\n"
    
    model_names = {
        "gpt-image-1": "GPT Image 1",
        "nano-banana": "Nano Banana",
        "dall-e-3": "DALL-E 3"
    }
    
    for model, data in costs.items():
        name = model_names.get(model, model)
        section += f"| {name} | {data['images_generated']} | ${data['cost_per_image']:.3f} | ${data['total_cost']:.2f} |\n"
    
    section += "\n**Note**: Costs are estimates based on published API pricing. Actual costs may vary.\n\n"
    
    section += "### 6.2 Latency\n\n"
    section += "Latency measurements were not systematically tracked in this evaluation. "
    section += "Future work should include:\n\n"
    section += "- Time-to-first-image (TTFI) measurements\n"
    section += "- End-to-end generation time\n"
    section += "- API response time analysis\n"
    section += "- Comparison of synchronous vs. asynchronous generation\n\n"
    
    section += "### 6.3 Cost-Performance Analysis\n\n"
    section += "Based on estimated costs and performance:\n\n"
    section += "- **Most Cost-Effective**: Nano Banana (~$0.24 for 47 images, 54.7% pass rate)\n"
    section += "- **Best Performance**: GPT Image 1 (~$1.88 for 47 images, 58.1% pass rate)\n"
    section += "- **Cost per Passed Constraint**:\n"
    section += "  - GPT Image 1: ~$0.019 per passed constraint\n"
    section += "  - Nano Banana: ~$0.003 per passed constraint\n\n"
    
    return section

def main():
    """Add cost/latency section to paper."""
    costs = estimate_costs()
    
    output_dir = Path(__file__).parent / "paper_assets"
    output_dir.mkdir(exist_ok=True)
    
    # Save cost data
    with open(output_dir / "cost_analysis.json", 'w') as f:
        json.dump(costs, f, indent=2)
    
    # Generate section
    section = generate_cost_section()
    with open(output_dir / "cost_latency_section.md", 'w') as f:
        f.write(section)
    
    print("Cost/latency analysis saved to paper_assets/")
    print("\nCost Section:")
    print(section)

if __name__ == "__main__":
    main()

