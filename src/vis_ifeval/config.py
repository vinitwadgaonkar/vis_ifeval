"""Configuration management for the benchmark."""

import os
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""

    use_wandb: bool = False
    wandb_project: str = "vis-ifeval"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    ocr_backend: str = "tesseract"


def load_config() -> BenchmarkConfig:
    """Load configuration from environment variables.

    Reads the following environment variables:
    - VIS_IFEVAL_USE_WANDB: Enable Weights & Biases logging (default: False)
    - VIS_IFEVAL_WANDB_PROJECT: W&B project name (default: "vis-ifeval")
    - VIS_IFEVAL_WANDB_ENTITY: W&B entity/username (default: None)
    - VIS_IFEVAL_WANDB_GROUP: W&B group name (default: None)
    - VIS_IFEVAL_WANDB_RUN_NAME: W&B run name (default: None)
    - VIS_IFEVAL_OCR_BACKEND: OCR backend name (default: "tesseract")

    Returns:
        BenchmarkConfig: Configuration object with loaded values.
    """
    use_wandb = os.getenv("VIS_IFEVAL_USE_WANDB", "0").lower() in ("1", "true", "yes")
    wandb_project = os.getenv("VIS_IFEVAL_WANDB_PROJECT", "vis-ifeval")
    wandb_entity = os.getenv("VIS_IFEVAL_WANDB_ENTITY")
    wandb_group = os.getenv("VIS_IFEVAL_WANDB_GROUP")
    wandb_run_name = os.getenv("VIS_IFEVAL_WANDB_RUN_NAME")
    ocr_backend = os.getenv("VIS_IFEVAL_OCR_BACKEND", "tesseract")

    return BenchmarkConfig(
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        ocr_backend=ocr_backend,
    )

