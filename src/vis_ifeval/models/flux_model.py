"""FLUX model implementation."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel


class FluxModel(ImageModel):
    """FLUX model for image generation."""

    def __init__(
        self,
        name: str = "flux",
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        width: int = 1024,
        height: int = 1024,
    ) -> None:
        """Initialize FLUX model.

        Args:
            name: Model name.
            model_id: HuggingFace model ID.
            device: Device to run on ("cuda" or "cpu").
            width: Image width.
            height: Image height.
        """
        super().__init__(
            name=name,
            config={
                "model_id": model_id,
                "device": device,
                "width": width,
                "height": height,
            },
        )
        self.model_id = model_id
        self.device = device
        self.width = width
        self.height = height
        self.pipe = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load the FLUX pipeline."""
        try:
            import torch
            from diffusers import FluxPipeline

            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)
        except ImportError as e:
            raise ImportError(
                "FLUX model requires torch and diffusers. "
                "Install with: pip install torch diffusers transformers"
            ) from e

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        if self.pipe is None:
            self._load_model()

        import torch

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
            generator=generator,
            height=self.height,
            width=self.width,
        )
        img = out.images[0]
        return img

