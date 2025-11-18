"""Generate images from prompts using image models."""

import argparse
import logging
import time
from pathlib import Path

from PIL import Image

from vis_ifeval.config import load_config
from vis_ifeval.models.dummy_model import DummyModel
from vis_ifeval.utils.io import load_prompts, save_jsonl
from vis_ifeval.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _build_model(model_name: str):
    """Build a model instance by name.

    Args:
        model_name: Name of the model. Supported:
            - Local: "dummy", "sdxl", "sd3", "flux"
            - API: "openai", "novelai", "banana", "replicate", "stability-api", "gemini"

    Returns:
        ImageModel instance.

    Raises:
        ValueError: If model name is unknown.
        ImportError: If required dependencies are missing.
    """
    if model_name == "dummy":
        return DummyModel()
    elif model_name == "sdxl":
        try:
            from vis_ifeval.models.sdxl_model import SDXLModel

            return SDXLModel()
        except ImportError as e:
            raise ImportError(
                f"SDXL model requires additional dependencies. {e}"
            ) from e
    elif model_name == "sd3":
        try:
            from vis_ifeval.models.sd3_model import SD3Model

            return SD3Model()
        except ImportError as e:
            raise ImportError(
                f"SD3 model requires additional dependencies. {e}"
            ) from e
    elif model_name == "flux":
        try:
            from vis_ifeval.models.flux_model import FluxModel

            return FluxModel()
        except ImportError as e:
            raise ImportError(
                f"FLUX model requires additional dependencies. {e}"
            ) from e
    elif model_name == "openai":
        try:
            from vis_ifeval.models.openai_model import OpenAIModel

            return OpenAIModel()
        except ImportError as e:
            raise ImportError(
                f"OpenAI model requires additional dependencies. {e}"
            ) from e
    elif model_name == "novelai":
        try:
            from vis_ifeval.models.novelai_model import NovelAIModel

            return NovelAIModel()
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"NovelAI model requires API key and dependencies. {e}"
            ) from e
    elif model_name == "banana":
        try:
            from vis_ifeval.models.banana_model import BananaModel

            return BananaModel()
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"Banana model requires API key and dependencies. {e}"
            ) from e
    elif model_name == "replicate":
        try:
            from vis_ifeval.models.replicate_model import ReplicateModel

            return ReplicateModel()
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"Replicate model requires API token and dependencies. {e}"
            ) from e
    elif model_name == "stability-api":
        try:
            from vis_ifeval.models.stability_api_model import StabilityAPIModel

            return StabilityAPIModel()
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"Stability API model requires API key and dependencies. {e}"
            ) from e
    elif model_name == "gemini":
        try:
            from vis_ifeval.models.gemini_model import GeminiModel

            return GeminiModel()
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"Gemini model requires API key and dependencies. {e}"
            ) from e
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            "Supported: dummy, sdxl, sd3, flux, openai, novelai, banana, replicate, stability-api, gemini"
        )


def run_generate(
    model_name: str,
    output_dir: str,
    prompts_path: str,
    use_wandb: bool | None = None,
) -> None:
    """Generate images for all prompts using the specified model.

    Args:
        model_name: Name of the model to use (e.g., "dummy").
        output_dir: Directory to save generated images.
        prompts_path: Path to JSONL file containing prompts.
        use_wandb: Optional override for wandb logging. If None, uses config.
    """
    # Load configuration
    cfg = load_config()
    if use_wandb is not None:
        cfg.use_wandb = use_wandb

    # Initialize wandb logger
    logger_wandb = WandbLogger(cfg, job_type="generate", model_name=model_name)

    # Load prompts
    prompts = load_prompts(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # Initialize model
    model = _build_model(model_name)

    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results list
    results = []

    # Generate images
    for idx, prompt in enumerate(prompts):
        prompt_id = prompt["id"]
        prompt_text = prompt["prompt"]
        category = prompt["category"]

        logger.info(f"Generating image {idx + 1}/{len(prompts)}: {prompt_id}")

        # Generate image
        start_time = time.time()
        img = model.generate(prompt_text, seed=idx)
        latency = time.time() - start_time

        # Save image
        image_path = model_output_dir / f"{prompt_id}.png"
        img.save(image_path)
        logger.info(f"Saved image to {image_path}")

        # Record result
        result = {
            "prompt_id": prompt_id,
            "category": category,
            "model": model_name,
            "image_path": str(image_path),
            "latency_sec": latency,
            "cost_estimate_usd": 0.0,
        }
        results.append(result)

        # Log to wandb
        logger_wandb.log(
            {
                "gen/prompt_id": prompt_id,
                "gen/category": category,
                "gen/model": model_name,
                "gen/latency_sec": latency,
            },
            step=idx,
        )

        # Log sample images (first 20 or every 10th)
        if idx < 20 or idx % 10 == 0:
            caption = f"{model_name} | {prompt_id} | {category}"
            logger_wandb.log_image(
                key="gen/sample_image",
                image=img,
                caption=caption,
                step=idx,
            )

    # Save results JSONL
    results_path = Path("results") / f"generation_{model_name}.jsonl"
    save_jsonl(str(results_path), results)
    logger.info(f"Saved generation results to {results_path}")

    # Finish wandb run
    logger_wandb.finish()
    logger.info("Generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dummy",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default="prompts/prompts.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    run_generate(
        model_name=args.model_name,
        output_dir=args.output_dir,
        prompts_path=args.prompts_path,
        use_wandb=args.use_wandb if args.use_wandb else None,
    )

