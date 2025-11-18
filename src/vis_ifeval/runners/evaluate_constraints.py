"""Evaluate constraints on generated images."""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from vis_ifeval.config import load_config
from vis_ifeval.evaluators import EvaluatorRegistry
from vis_ifeval.utils.io import load_prompts, save_jsonl
from vis_ifeval.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_evaluate(
    model_name: str,
    prompts_path: str,
    use_wandb: bool | None = None,
) -> None:
    """Evaluate constraints on generated images.

    Args:
        model_name: Name of the model used for generation.
        prompts_path: Path to JSONL file containing prompts.
        use_wandb: Optional override for wandb logging. If None, uses config.
    """
    # Load configuration
    cfg = load_config()
    if use_wandb is not None:
        cfg.use_wandb = use_wandb

    # Initialize wandb logger
    logger_wandb = WandbLogger(cfg, job_type="evaluate", model_name=model_name)

    # Load prompts into a dict keyed by id
    prompts_list = load_prompts(prompts_path)
    prompts_dict = {p["id"]: p for p in prompts_list}
    logger.info(f"Loaded {len(prompts_dict)} prompts")

    # Load generation results
    generation_path = Path("results") / f"generation_{model_name}.jsonl"
    if not generation_path.exists():
        raise FileNotFoundError(
            f"Generation results not found: {generation_path}. "
            "Run generate_images.py first."
        )

    generation_records = []
    with open(generation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                generation_records.append(json.loads(line))

    logger.info(f"Loaded {len(generation_records)} generation records")

    # Initialize evaluator registry
    registry = EvaluatorRegistry()

    # Prepare results list
    results = []
    global_step = 0

    # Evaluate constraints for each generated image
    for gen_record in generation_records:
        prompt_id = gen_record["prompt_id"]
        image_path = gen_record["image_path"]
        category = gen_record["category"]

        # Load image
        img = Image.open(image_path)

        # Get prompt
        if prompt_id not in prompts_dict:
            logger.warning(f"Prompt {prompt_id} not found in prompts file")
            continue

        prompt = prompts_dict[prompt_id]

        # Evaluate each constraint
        for constraint in prompt.get("constraints", []):
            constraint_id = constraint["id"]
            constraint_type = constraint["type"]

            # Score constraint
            try:
                score = registry.score_constraint(img, prompt, constraint)
            except Exception as e:
                logger.error(
                    f"Error evaluating constraint {constraint_id} for {prompt_id}: {e}"
                )
                score = 0.0

            # Convert to binary label
            label = int(score >= 0.5)

            # Record result
            result = {
                "prompt_id": prompt_id,
                "category": category,
                "constraint_id": constraint_id,
                "type": constraint_type,
                "model": model_name,
                "score": score,
                "label": label,
            }
            results.append(result)

            # Log to wandb
            logger_wandb.log(
                {
                    "eval/model": model_name,
                    "eval/prompt_id": prompt_id,
                    "eval/category": category,
                    "eval/constraint_id": constraint_id,
                    "eval/constraint_type": constraint_type,
                    "eval/score": score,
                    "eval/label": label,
                },
                step=global_step,
            )

            global_step += 1

        # Log sample images for first few prompts with constraint scores
        if len(results) <= 10:
            # Build caption with constraint scores
            constraint_scores = [
                r["score"]
                for r in results
                if r["prompt_id"] == prompt_id
            ]
            avg_score = sum(constraint_scores) / len(constraint_scores) if constraint_scores else 0.0
            caption = (
                f"{model_name} | {prompt_id} | {category} | "
                f"Avg Score: {avg_score:.3f} | "
                f"Constraints: {len(prompt.get('constraints', []))}"
            )
            logger_wandb.log_image(
                key="eval/sample_image",
                image=img,
                caption=caption,
                step=global_step,
            )

            # Log constraint scores as a table/bar chart data
            if logger_wandb.run is not None:
                try:
                    import wandb

                    # Create a table of constraint scores for this prompt
                    constraint_data = [
                        [
                            constraint["id"],
                            constraint["type"],
                            score,
                            int(score >= 0.5),
                        ]
                        for constraint, score in zip(
                            prompt.get("constraints", []),
                            constraint_scores,
                        )
                    ]
                    table = wandb.Table(
                        columns=["constraint_id", "type", "score", "satisfied"],
                        data=constraint_data,
                    )
                    logger_wandb.run.log(
                        {f"eval/constraint_table/{prompt_id}": table},
                        step=global_step,
                    )
                except Exception as e:
                    logger.debug(f"Failed to log constraint table: {e}")

    # Save results JSONL
    results_path = Path("results") / f"scores_{model_name}.jsonl"
    save_jsonl(str(results_path), results)
    logger.info(f"Saved evaluation results to {results_path}")

    # Finish wandb run
    logger_wandb.finish()
    logger.info("Evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate constraints on generated images")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dummy",
        help="Name of the model used for generation",
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

    run_evaluate(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        use_wandb=args.use_wandb if args.use_wandb else None,
    )

