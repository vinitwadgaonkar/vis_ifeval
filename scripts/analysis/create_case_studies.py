#!/usr/bin/env python3
"""Create case studies with side-by-side image comparisons."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

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

def create_side_by_side_comparison(prompt_id, models, output_dir):
    """Create side-by-side comparison image for a prompt."""
    images = {}
    results_data = {}
    
    for model in models:
        data = load_results(model)
        if not data:
            continue
        
        for prompt in data.get("prompts", []):
            if prompt.get("id") == prompt_id:
                results_data[model] = prompt
                img_path = prompt.get("image_path")
                if img_path and Path(img_path).exists():
                    images[model] = Image.open(img_path)
                break
    
    if not images:
        return None
    
    # Create composite image
    img_width = 1024
    img_height = 1024
    spacing = 20
    padding = 40
    
    model_names = {
        "gpt-image-1": "GPT Image 1",
        "nano-banana": "Nano Banana",
        "dall-e-3": "DALL-E 3"
    }
    
    num_models = len(images)
    total_width = (img_width * num_models) + (spacing * (num_models - 1)) + (padding * 2)
    total_height = img_height + 200  # Extra space for labels
    
    composite = Image.new('RGB', (total_width, total_height), 'white')
    
    x_offset = padding
    for i, (model, img) in enumerate(images.items()):
        # Resize image if needed
        img_resized = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        composite.paste(img_resized, (x_offset, 100))
        
        # Add model name label
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        model_name = model_names.get(model, model)
        text_bbox = draw.textbbox((0, 0), model_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img_width - text_width) // 2
        draw.text((text_x, 20), model_name, fill='black', font=font)
        
        # Add score info if available
        if model in results_data:
            prompt_data = results_data[model]
            constraints = prompt_data.get("constraints", [])
            if constraints:
                passed = sum(1 for c in constraints if c.get("passed", False))
                total = len(constraints)
                score_text = f"{passed}/{total} passed"
                score_bbox = draw.textbbox((0, 0), score_text, font=font)
                score_width = score_bbox[2] - score_bbox[0]
                score_x = x_offset + (img_width - score_width) // 2
                draw.text((score_x, 50), score_text, fill='blue', font=font)
        
        x_offset += img_width + spacing
    
    # Add prompt text at bottom
    if results_data:
        first_model = list(results_data.keys())[0]
        prompt_text = results_data[first_model].get("prompt", "")
        # Wrap text
        words = prompt_text.split()
        lines = []
        current_line = []
        current_width = 0
        max_width = total_width - (padding * 2)
        
        for word in words:
            word_width = len(word) * 10  # Approximate
            if current_width + word_width > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width + 10
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y_text = img_height + 120
        for line in lines[:3]:  # Limit to 3 lines
            draw.text((padding, y_text), line[:100], fill='black', font=font)
            y_text += 25
    
    return composite

def select_case_study_prompts():
    """Select representative prompts for case studies."""
    # Load all results to find interesting examples
    models = ["gpt-image-1", "nano-banana", "dall-e-3"]
    
    # Categories for case studies:
    # 1. High performance example
    # 2. Low performance example
    # 3. Text rendering example
    # 4. Composition example
    # 5. CSP example
    
    case_studies = []
    
    # Get GPT Image 1 results as reference
    gpt_data = load_results("gpt-image-1")
    if not gpt_data:
        return []
    
    # Find examples
    for prompt in gpt_data.get("prompts", []):
        prompt_id = prompt.get("id")
        constraints = prompt.get("constraints", [])
        
        if not constraints:
            continue
        
        passed = sum(1 for c in constraints if c.get("passed", False))
        total = len(constraints)
        pass_rate = passed / total if total > 0 else 0
        
        category = prompt.get("category", "")
        constraint_types = [c.get("type") for c in constraints]
        
        # Select based on criteria
        if "text" in constraint_types and prompt_id.startswith("text_"):
            case_studies.append({
                "id": prompt_id,
                "type": "text_rendering",
                "pass_rate": pass_rate,
                "category": category
            })
        elif category == "composition" and len(constraints) >= 3:
            case_studies.append({
                "id": prompt_id,
                "type": "composition",
                "pass_rate": pass_rate,
                "category": category
            })
        elif category == "csp_demo":
            case_studies.append({
                "id": prompt_id,
                "type": "csp",
                "pass_rate": pass_rate,
                "category": category
            })
    
    # Select best examples
    selected = []
    for study_type in ["text_rendering", "composition", "csp"]:
        candidates = [c for c in case_studies if c["type"] == study_type]
        if candidates:
            # Select one with medium pass rate (interesting case)
            candidates.sort(key=lambda x: abs(x["pass_rate"] - 0.5))
            selected.append(candidates[0])
    
    return selected[:5]  # Limit to 5 case studies

def main():
    """Create case studies."""
    print("="*80)
    print("CREATING CASE STUDIES")
    print("="*80)
    
    output_dir = Path(__file__).parent / "paper_assets" / "case_studies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["gpt-image-1", "nano-banana", "dall-e-3"]
    case_studies = select_case_study_prompts()
    
    case_study_info = []
    
    for i, case in enumerate(case_studies, 1):
        prompt_id = case["id"]
        print(f"\nCreating case study {i}: {prompt_id} ({case['type']})")
        
        composite = create_side_by_side_comparison(prompt_id, models, output_dir)
        if composite:
            output_path = output_dir / f"case_study_{i:02d}_{prompt_id}.png"
            composite.save(output_path)
            print(f"  Saved: {output_path}")
            
            case_study_info.append({
                "number": i,
                "prompt_id": prompt_id,
                "type": case["type"],
                "category": case["category"],
                "pass_rate": case["pass_rate"],
                "image_path": str(output_path)
            })
    
    # Save case study metadata
    with open(output_dir / "case_studies_metadata.json", 'w') as f:
        json.dump(case_study_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Created {len(case_study_info)} case studies")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()

