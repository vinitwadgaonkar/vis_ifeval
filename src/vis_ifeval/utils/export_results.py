"""Export results for papers and analysis."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_metrics(model_name: str) -> dict[str, Any] | None:
    """Load metrics for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Metrics dictionary or None if not found.
    """
    metrics_path = Path("results") / f"metrics_{model_name}.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def export_vipr_table(model_names: list[str], output_dir: str = "results/tables") -> None:
    """Export VIPR table in Markdown and CSV formats.

    Args:
        model_names: List of model names to include.
        output_dir: Output directory for tables.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data
    all_models = {}
    all_types = set()

    for model_name in model_names:
        metrics = load_metrics(model_name)
        if metrics:
            all_models[model_name] = metrics
            all_types.update(metrics.get("by_type", {}).keys())

    if not all_models:
        logger.warning("No metrics found for any model")
        return

    # Build table data
    rows = []
    for model_name in sorted(model_names):
        metrics = all_models.get(model_name)
        if not metrics:
            continue

        row = {"Model": model_name, "VIPR": metrics.get("vipr", 0.0)}
        for ctype in sorted(all_types):
            row[ctype] = metrics.get("by_type", {}).get(ctype, 0.0)
        rows.append(row)

    # Export Markdown
    md_path = output_path / "model_vipr_table.md"
    with open(md_path, "w") as f:
        # Header
        headers = ["Model", "VIPR"] + sorted(all_types)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        # Rows
        for row in rows:
            values = [
                row.get("Model", ""),
                f"{row.get('VIPR', 0):.4f}",
            ] + [f"{row.get(t, 0):.4f}" for t in sorted(all_types)]
            f.write("| " + " | ".join(values) + " |\n")

    logger.info(f"Exported Markdown table to {md_path}")

    # Export CSV
    csv_path = output_path / "model_vipr_table.csv"
    with open(csv_path, "w") as f:
        headers = ["Model", "VIPR"] + sorted(all_types)
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = [
                row.get("Model", ""),
                f"{row.get('VIPR', 0):.4f}",
            ] + [f"{row.get(t, 0):.4f}" for t in sorted(all_types)]
            f.write(",".join(values) + "\n")

    logger.info(f"Exported CSV table to {csv_path}")


def plot_vipr_bar_chart(
    model_names: list[str], output_dir: str = "results/plots"
) -> None:
    """Create VIPR bar chart.

    Args:
        model_names: List of model names to include.
        output_dir: Output directory for plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = []
    viprs = []

    for model_name in sorted(model_names):
        metrics = load_metrics(model_name)
        if metrics:
            models.append(model_name)
            viprs.append(metrics.get("vipr", 0.0))

    if not models:
        logger.warning("No metrics found for plotting")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(models, viprs)
    plt.xlabel("Model")
    plt.ylabel("VIPR")
    plt.title("Visual Instruction Pass Rate (VIPR) by Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = output_path / "vipr_by_model.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved VIPR bar chart to {plot_path}")


def plot_vipr_by_category(
    model_names: list[str], output_dir: str = "results/plots"
) -> None:
    """Create VIPR by category comparison chart.

    Args:
        model_names: List of model names to include.
        output_dir: Output directory for plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data
    categories = set()
    data = defaultdict(dict)

    for model_name in model_names:
        metrics = load_metrics(model_name)
        if metrics:
            for category, vipr in metrics.get("by_category", {}).items():
                categories.add(category)
                data[category][model_name] = vipr

    if not categories:
        logger.warning("No category data found for plotting")
        return

    # Create grouped bar chart
    categories = sorted(categories)
    models = sorted(model_names)
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [data[cat].get(model, 0.0) for cat in categories]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xlabel("Category")
    ax.set_ylabel("VIPR")
    ax.set_title("VIPR by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    plot_path = output_path / "vipr_by_category.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved VIPR by category chart to {plot_path}")


def plot_vipr_by_type(
    model_names: list[str], output_dir: str = "results/plots"
) -> None:
    """Create VIPR by constraint type comparison chart.

    Args:
        model_names: List of model names to include.
        output_dir: Output directory for plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data
    types = set()
    data = defaultdict(dict)

    for model_name in model_names:
        metrics = load_metrics(model_name)
        if metrics:
            for ctype, vipr in metrics.get("by_type", {}).items():
                types.add(ctype)
                data[ctype][model_name] = vipr

    if not types:
        logger.warning("No type data found for plotting")
        return

    # Create grouped bar chart
    types = sorted(types)
    models = sorted(model_names)
    x = np.arange(len(types))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        values = [data[t].get(model, 0.0) for t in types]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xlabel("Constraint Type")
    ax.set_ylabel("VIPR")
    ax.set_title("VIPR by Constraint Type")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    plot_path = output_path / "vipr_by_type.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved VIPR by type chart to {plot_path}")


def export_all_results(model_names: list[str]) -> None:
    """Export all results (tables and plots).

    Args:
        model_names: List of model names to include.
    """
    logger.info("Exporting results...")
    export_vipr_table(model_names)
    plot_vipr_bar_chart(model_names)
    plot_vipr_by_category(model_names)
    plot_vipr_by_type(model_names)
    logger.info("Export complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export evaluation results")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names",
    )
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    export_all_results(model_names)

