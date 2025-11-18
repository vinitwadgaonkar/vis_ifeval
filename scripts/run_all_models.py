"""Automated multi-model evaluation script."""

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr.

    Args:
        cmd: Command to run as list of strings.
        cwd: Working directory.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"Error running command {cmd}: {e}")
        return 1, "", str(e)


def run_pipeline_for_model(
    model_name: str,
    use_wandb: bool = False,
    limit_prompts: int | None = None,
    prompts_path: str = "prompts/prompts.jsonl",
) -> dict:
    """Run full pipeline for a single model.

    Args:
        model_name: Name of the model to evaluate.
        use_wandb: Whether to enable W&B logging.
        limit_prompts: Limit number of prompts (None for all).
        prompts_path: Path to prompts file.

    Returns:
        Dictionary with results and timing info.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"{'='*70}")

    results = {
        "model": model_name,
        "success": False,
        "timings": {},
        "error": None,
    }

    env = {"PYTHONPATH": "src"}

    # Step 1: Generate images
    logger.info(f"[{model_name}] Step 1: Generating images...")
    start_time = time.time()
    cmd = [
        "python",
        "-m",
        "vis_ifeval.runners.generate_images",
        "--model-name",
        model_name,
        "--output-dir",
        "data/outputs",
        "--prompts-path",
        prompts_path,
    ]
    if use_wandb:
        cmd.append("--use-wandb")

    exit_code, stdout, stderr = run_command(cmd, cwd=Path.cwd())
    gen_time = time.time() - start_time
    results["timings"]["generate"] = gen_time

    if exit_code != 0:
        results["error"] = f"Generation failed: {stderr}"
        logger.error(f"[{model_name}] Generation failed: {stderr}")
        return results

    logger.info(f"[{model_name}] Generation completed in {gen_time:.2f}s")

    # Step 2: Evaluate constraints
    logger.info(f"[{model_name}] Step 2: Evaluating constraints...")
    start_time = time.time()
    cmd = [
        "python",
        "-m",
        "vis_ifeval.runners.evaluate_constraints",
        "--model-name",
        model_name,
        "--prompts-path",
        prompts_path,
    ]
    if use_wandb:
        cmd.append("--use-wandb")

    exit_code, stdout, stderr = run_command(cmd, cwd=Path.cwd())
    eval_time = time.time() - start_time
    results["timings"]["evaluate"] = eval_time

    if exit_code != 0:
        results["error"] = f"Evaluation failed: {stderr}"
        logger.error(f"[{model_name}] Evaluation failed: {stderr}")
        return results

    logger.info(f"[{model_name}] Evaluation completed in {eval_time:.2f}s")

    # Step 3: Aggregate metrics
    logger.info(f"[{model_name}] Step 3: Aggregating metrics...")
    start_time = time.time()
    cmd = [
        "python",
        "-m",
        "vis_ifeval.runners.aggregate_metrics",
        "--model-name",
        model_name,
    ]
    if use_wandb:
        cmd.append("--use-wandb")

    exit_code, stdout, stderr = run_command(cmd, cwd=Path.cwd())
    agg_time = time.time() - start_time
    results["timings"]["aggregate"] = agg_time

    if exit_code != 0:
        results["error"] = f"Aggregation failed: {stderr}"
        logger.error(f"[{model_name}] Aggregation failed: {stderr}")
        return results

    logger.info(f"[{model_name}] Aggregation completed in {agg_time:.2f}s")

    # Load metrics
    metrics_path = Path("results") / f"metrics_{model_name}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        results["metrics"] = metrics
        results["success"] = True
        logger.info(f"[{model_name}] VIPR: {metrics.get('vipr', 0):.4f}")
    else:
        results["error"] = "Metrics file not found"
        logger.warning(f"[{model_name}] Metrics file not found")

    total_time = sum(results["timings"].values())
    results["timings"]["total"] = total_time
    logger.info(f"[{model_name}] Total time: {total_time:.2f}s")

    return results


def combine_results(all_results: list[dict]) -> dict:
    """Combine results from all models into a summary.

    Args:
        all_results: List of result dictionaries.

    Returns:
        Combined summary dictionary.
    """
    combined = {
        "models": {},
        "summary": {
            "total_models": len(all_results),
            "successful": sum(1 for r in all_results if r.get("success")),
            "failed": sum(1 for r in all_results if not r.get("success")),
        },
        "vipr_by_model": {},
        "vipr_by_type": defaultdict(dict),
        "vipr_by_category": defaultdict(dict),
    }

    for result in all_results:
        model_name = result["model"]
        combined["models"][model_name] = {
            "success": result.get("success", False),
            "timings": result.get("timings", {}),
            "error": result.get("error"),
        }

        if result.get("success") and "metrics" in result:
            metrics = result["metrics"]
            combined["vipr_by_model"][model_name] = metrics.get("vipr", 0.0)

            # Aggregate by type
            for ctype, vipr in metrics.get("by_type", {}).items():
                combined["vipr_by_type"][ctype][model_name] = vipr

            # Aggregate by category
            for category, vipr in metrics.get("by_category", {}).items():
                combined["vipr_by_category"][category][model_name] = vipr

    return combined


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline for multiple models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="dummy",
        help="Comma-separated list of models (default: dummy)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Limit number of prompts to evaluate",
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default="prompts/prompts.jsonl",
        help="Path to prompts JSONL file",
    )

    args = parser.parse_args()

    # Parse model list
    model_names = [m.strip() for m in args.models.split(",")]

    logger.info(f"Starting multi-model evaluation")
    logger.info(f"Models: {model_names}")
    logger.info(f"W&B: {args.use_wandb}")
    logger.info(f"Prompts: {args.prompts_path}")

    # Run pipeline for each model
    all_results = []
    for model_name in model_names:
        result = run_pipeline_for_model(
            model_name=model_name,
            use_wandb=args.use_wandb,
            limit_prompts=args.limit_prompts,
            prompts_path=args.prompts_path,
        )
        all_results.append(result)

    # Combine results
    combined = combine_results(all_results)

    # Save combined results
    output_path = Path("results") / "combined_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"\nVIPR by Model:")
    for model, vipr in sorted(combined["vipr_by_model"].items()):
        logger.info(f"  {model:15s}: {vipr:.4f}")

    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total models: {combined['summary']['total_models']}")
    logger.info(f"  Successful: {combined['summary']['successful']}")
    logger.info(f"  Failed: {combined['summary']['failed']}")


if __name__ == "__main__":
    main()

