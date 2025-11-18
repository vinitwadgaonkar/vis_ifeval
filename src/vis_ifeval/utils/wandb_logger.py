"""Weights & Biases logging helper."""

import logging
from typing import TYPE_CHECKING

from vis_ifeval.config import BenchmarkConfig

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class WandbLogger:
    """Helper class for logging to Weights & Biases."""

    def __init__(
        self,
        cfg: BenchmarkConfig,
        job_type: str,
        model_name: str | None = None,
    ) -> None:
        """Initialize the W&B logger.

        Args:
            cfg: Benchmark configuration.
            job_type: Type of job (e.g., "generate", "evaluate", "aggregate").
            model_name: Optional model name for run naming.
        """
        self.cfg = cfg
        self.run = None

        if not cfg.use_wandb:
            return

        if not WANDB_AVAILABLE:
            logger.warning(
                "wandb is not installed or not available. "
                "Logging will be disabled. Install with: pip install wandb"
            )
            return

        try:
            import time

            run_name = cfg.wandb_run_name
            if run_name is None:
                timestamp = int(time.time())
                name_parts = [job_type]
                if model_name:
                    name_parts.append(model_name)
                name_parts.append(str(timestamp))
                run_name = "-".join(name_parts)

            self.run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                name=run_name,
                job_type=job_type,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Logging will be disabled.")
            self.run = None

    def log(self, data: dict, step: int | None = None) -> None:
        """Log data to W&B.

        Args:
            data: Dictionary of metrics to log.
            step: Optional step number.
        """
        if self.run is not None:
            try:
                self.run.log(data, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

    def log_image(
        self,
        key: str,
        image: "Image.Image",
        caption: str | None = None,
        step: int | None = None,
    ) -> None:
        """Log an image to W&B.

        Args:
            key: Key/name for the image in W&B.
            image: PIL Image to log.
            caption: Optional caption for the image.
            step: Optional step number.
        """
        if self.run is not None:
            try:
                wandb_image = wandb.Image(image, caption=caption)
                self.run.log({key: wandb_image}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log image to wandb: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

