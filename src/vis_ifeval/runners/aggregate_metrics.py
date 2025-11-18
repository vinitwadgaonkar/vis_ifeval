"""Aggregate evaluation metrics and compute VIPR."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from vis_ifeval.config import load_config
from vis_ifeval.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_aggregate(
    model_name: str,
    use_wandb: bool | None = None,
) -> None:
    """Aggregate evaluation metrics and compute VIPR.

    Args:
        model_name: Name of the model to aggregate metrics for.
        use_wandb: Optional override for wandb logging. If None, uses config.
    """
    # Load configuration
    cfg = load_config()
    if use_wandb is not None:
        cfg.use_wandb = use_wandb

    # Initialize wandb logger
    logger_wandb = WandbLogger(cfg, job_type="aggregate", model_name=model_name)

    # Load scores
    scores_path = Path("results") / f"scores_{model_name}.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores file not found: {scores_path}. "
            "Run evaluate_constraints.py first."
        )

    scores = []
    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))

    logger.info(f"Loaded {len(scores)} constraint scores")

    # Compute aggregate metrics
    total_constraints = len(scores)
    total_satisfied = sum(s["label"] for s in scores)
    vipr = total_satisfied / total_constraints if total_constraints > 0 else 0.0

    # Compute VIPR by type
    type_counts = defaultdict(lambda: {"total": 0, "satisfied": 0})
    for score in scores:
        constraint_type = score["type"]
        type_counts[constraint_type]["total"] += 1
        type_counts[constraint_type]["satisfied"] += score["label"]

    vipr_by_type = {
        t: counts["satisfied"] / counts["total"]
        if counts["total"] > 0
        else 0.0
        for t, counts in type_counts.items()
    }

    # Compute VIPR by category
    category_counts = defaultdict(lambda: {"total": 0, "satisfied": 0})
    for score in scores:
        category = score["category"]
        category_counts[category]["total"] += 1
        category_counts[category]["satisfied"] += score["label"]

    vipr_by_category = {
        c: counts["satisfied"] / counts["total"]
        if counts["total"] > 0
        else 0.0
        for c, counts in category_counts.items()
    }

    # Load generation results for latency
    generation_path = Path("results") / f"generation_{model_name}.jsonl"
    latencies = []
    if generation_path.exists():
        with open(generation_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    latencies.append(record.get("latency_sec", 0.0))

    latency_mean = sum(latencies) / len(latencies) if latencies else 0.0

    # Prepare metrics summary
    metrics = {
        "model": model_name,
        "vipr": vipr,
        "by_type": vipr_by_type,
        "by_category": vipr_by_category,
        "latency_mean_sec": latency_mean,
        "total_constraints": total_constraints,
        "total_satisfied": total_satisfied,
    }

    # Save metrics
    metrics_path = Path("results") / f"metrics_{model_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"VIPR: {vipr:.4f}")
    logger.info(f"VIPR by type: {vipr_by_type}")
    logger.info(f"VIPR by category: {vipr_by_category}")
    logger.info(f"Mean latency: {latency_mean:.4f}s")

    # Log to wandb
    wandb_data = {
        "summary/model": model_name,
        "summary/vipr": vipr,
        "summary/latency_mean_sec": latency_mean,
        "summary/total_constraints": total_constraints,
        "summary/total_satisfied": total_satisfied,
    }

    # Flatten nested dicts for wandb
    for t, v in vipr_by_type.items():
        wandb_data[f"summary/vipr_type/{t}"] = v

    for c, v in vipr_by_category.items():
        wandb_data[f"summary/vipr_category/{c}"] = v

    logger_wandb.log(wandb_data)

    # Create visualization tables for wandb
    if logger_wandb.run is not None:
        try:
            import wandb

            # VIPR by type table
            type_table = wandb.Table(
                columns=["constraint_type", "vipr"],
                data=[[t, v] for t, v in sorted(vipr_by_type.items())],
            )
            logger_wandb.run.log({"summary/vipr_by_type_table": type_table})

            # VIPR by category table
            category_table = wandb.Table(
                columns=["category", "vipr"],
                data=[[c, v] for c, v in sorted(vipr_by_category.items())],
            )
            logger_wandb.run.log({"summary/vipr_by_category_table": category_table})
        except Exception as e:
            logger.debug(f"Failed to log wandb tables: {e}")

    # Finish wandb run
    logger_wandb.finish()
    logger.info("Aggregation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dummy",
        help="Name of the model to aggregate metrics for",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    run_aggregate(
        model_name=args.model_name,
        use_wandb=args.use_wandb if args.use_wandb else None,
    )

