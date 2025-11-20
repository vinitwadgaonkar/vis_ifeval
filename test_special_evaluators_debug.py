#!/usr/bin/env python3
"""Debug test for SketchToRenderEvaluator to see component scores."""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_sketch_to_render_debug(model, model_name):
    """Test SketchToRenderEvaluator with detailed debug output."""
    from vis_ifeval.evaluators.sketch_to_render_eval import SketchToRenderEvaluator
    from vis_ifeval.utils.clip_utils import ClipModelWrapper, ClipConfig
    
    # Load first prompt
    prompts_file = Path(__file__).parent / "prompts" / "prompts_sketch_to_render.jsonl"
    with open(prompts_file, "r") as f:
        prompt_data = json.loads(f.readline().strip())
    
    output_dir = Path(__file__).parent / "data" / "outputs" / "special_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_id = prompt_data["id"]
    constraint = prompt_data["constraints"][0].copy()
    
    logger.info(f"Testing {prompt_id}")
    logger.info(f"Prompt: {prompt_data['prompt']}")
    
    # Generate sketch
    logger.info("Generating sketch...")
    sketch_prompt = f"Simple black and white line art sketch, minimal details, outline only, no shading: {prompt_data['prompt']}"
    sketch_img = model.generate(sketch_prompt)
    sketch_path = output_dir / f"{model_name}_debug_sketch_{prompt_id}.png"
    sketch_img.save(sketch_path)
    
    # Generate render
    logger.info("Generating render...")
    render_img = model.generate(prompt_data["prompt"])
    render_path = output_dir / f"{model_name}_debug_render_{prompt_id}.png"
    render_img.save(render_path)
    
    # Initialize evaluator with CLIP
    clip = ClipModelWrapper(ClipConfig())
    evaluator = SketchToRenderEvaluator(clip=clip)
    
    # Add sketch to constraint
    constraint["sketch_image"] = sketch_img
    
    # Manually compute component scores
    logger.info("\n=== Component Scores ===")
    
    # SSIM
    ssim_score = evaluator._compute_ssim(sketch_img, render_img)
    logger.info(f"SSIM: {ssim_score:.4f}")
    
    # Edge alignment
    sketch_edges = evaluator._extract_edges_canny(sketch_img)
    render_edges = evaluator._extract_edges_canny(render_img)
    edge_score = evaluator._compute_edge_alignment(sketch_edges, render_edges)
    logger.info(f"Edge Alignment: {edge_score:.4f}")
    
    # Prompt adherence
    prompt_text = prompt_data.get("prompt", "")
    prompt_score = evaluator._check_prompt_adherence(render_img, prompt_text)
    logger.info(f"Prompt Adherence: {prompt_score:.4f}")
    
    # Texture richness
    sketch_texture = evaluator._compute_texture_richness(sketch_img)
    render_texture = evaluator._compute_texture_richness(render_img)
    texture_score = min(1.0, render_texture / max(0.1, sketch_texture)) if sketch_texture > 0 else 0.0
    logger.info(f"Sketch Texture: {sketch_texture:.4f}")
    logger.info(f"Render Texture: {render_texture:.4f}")
    logger.info(f"Texture Score: {texture_score:.4f}")
    
    # Style match
    style = constraint.get("style", "")
    style_score = 0.5
    if style and clip and clip.enabled:
        style_score = evaluator._check_prompt_adherence(render_img, style)
        logger.info(f"Style Match: {style_score:.4f}")
    else:
        logger.info(f"Style Match: N/A (style='{style}', clip_enabled={clip.enabled if clip else False})")
    
    # Final score
    final_score = evaluator.score(render_img, prompt_data, constraint)
    logger.info(f"\nFinal Score: {final_score:.4f}")
    
    return final_score

def main():
    # Test with nano-banana
    from vis_ifeval.models.openrouter_model import OpenRouterModel
    
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["VIS_IFEVAL_OCR_BACKEND"] = "deepseek"
    
    model = OpenRouterModel(model="google/gemini-2.5-flash-image", size="1024x1024")
    
    logger.info("="*80)
    logger.info("DEBUG TEST: SketchToRenderEvaluator")
    logger.info("="*80)
    
    score = test_sketch_to_render_debug(model, "nano-banana")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Final Score: {score:.4f} {'✓ PASSED' if score > 0.5 else '✗ FAILED'}")

if __name__ == "__main__":
    main()

